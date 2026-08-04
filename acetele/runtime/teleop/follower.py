"""Transport-independent follower synchronization and command admission."""

from __future__ import annotations

import time
from dataclasses import replace
from functools import wraps
from pathlib import Path
from threading import Event, RLock, Thread
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from acetele.control import (
    CartesianTeleopController,
    CartesianTeleopDiagnostics,
    CartesianTeleopTuning,
    StreamingPositionTuning,
)
from acetele.core import EndEffectorPose, JointCommand, RobotCommand, RobotState
from acetele.model import ArmKinematics
from acetele.runtime.follower_runtime import FollowerRuntime
from acetele.runtime.safety import RuntimeSafetyState
from acetele.runtime.teleop.synchronization import (
    FollowerSyncController,
    FollowerSyncStatus,
    LeaderSyncMode,
    TeleopMode,
)


def _serialized_session(method):
    """Serialize session state while leaving the transport and bus workers independent."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)

    return wrapped


class FollowerTeleopSession:
    """Transport-independent synchronization around a follower runtime."""

    def __init__(
        self,
        runtime: FollowerRuntime,
        *,
        session_timeout_ns: int = 500_000_000,
        command_deadline_ns: int = 50_000_000,
        teleop_mode: TeleopMode = TeleopMode.JOINT,
        translation_scale: float = 2.0,
        rotation_scale: float = 1.0,
        cartesian_tuning: CartesianTeleopTuning = CartesianTeleopTuning(),
        motion_tuning: StreamingPositionTuning = StreamingPositionTuning(),
        motion_cycle_hz: Optional[float] = 100.0,
        cartesian_controller: Optional[CartesianTeleopController] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(runtime, FollowerRuntime):
            raise ValueError("follower session requires a FollowerRuntime")
        for name, value in (
            ("session_timeout_ns", session_timeout_ns),
            ("command_deadline_ns", command_deadline_ns),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.runtime = runtime
        if not callable(clock_ns):
            raise ValueError("follower session clock_ns must be callable")
        self._clock_ns = clock_ns
        try:
            self.teleop_mode = TeleopMode(teleop_mode)
        except ValueError as exc:
            raise ValueError("teleop_mode must be 'joint' or 'ee_pose'") from exc
        self._sync = FollowerSyncController(session_timeout_ns)
        # Freeze routing metadata at construction. Runtime commands can then be
        # validated and partitioned without consulting mutable ROS messages.
        self._arm_groups = tuple(arm.name for arm in runtime.spec.arms)
        if not isinstance(motion_tuning, StreamingPositionTuning):
            raise ValueError("motion_tuning must be StreamingPositionTuning")
        self._motion_tuning = motion_tuning
        if motion_cycle_hz is not None and (
            not np.isfinite(motion_cycle_hz) or motion_cycle_hz <= 0.0
        ):
            raise ValueError("motion_cycle_hz must be finite and positive or None")
        self._motion_period_ns = (
            None if motion_cycle_hz is None else round(1e9 / motion_cycle_hz)
        )
        self._state_lock = RLock()
        self._motion_stop = Event()
        self._motion_worker: Optional[Thread] = None
        self._motion_error: Optional[BaseException] = None
        self._arm_names = MappingProxyType(
            {
                arm.name: tuple(joint.name for joint in arm.joints)
                for arm in runtime.spec.arms
            }
        )
        joint_groups = runtime.joint_groups
        self._end_effector_names = MappingProxyType(
            {
                f"{arm.name}.end_effector": joint_groups[
                    f"{arm.name}.end_effector"
                ].names
                for arm in runtime.spec.arms
                if arm.end_effector is not None
            }
        )
        self._command_lifetimes_ns = MappingProxyType(
            {
                group_name: runtime.command_lifetime_ns(
                    group_name,
                    minimum_ns=command_deadline_ns,
                )
                for group_name in runtime.joint_groups
            }
        )
        self._cartesian_controller = self._resolve_cartesian_controller(
            cartesian_controller,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            tuning=cartesian_tuning,
        )
        self._latest_state: Optional[RobotState] = None
        # Network callbacks only replace these immutable targets. The local state loop
        # republishes them at the actuator cadence, so packet jitter cannot trip the
        # bus watchdog or insert a mode transition into an otherwise smooth motion.
        self._motion_targets: dict[str, JointCommand] = {}
        self._soft_stop_started_ns: Optional[int] = None
        self._soft_stop_settled_since_ns: Optional[int] = None

    @property
    def mode(self) -> LeaderSyncMode:
        """Return the currently accepted leader mode."""

        with self._state_lock:
            return self._sync.mode

    @property
    def status(self) -> FollowerSyncStatus:
        """Return synchronization status, with runtime faults taking precedence."""

        if self.runtime.diagnostics().safety.state == RuntimeSafetyState.FAULT:
            return FollowerSyncStatus.FAULT
        with self._state_lock:
            return self._sync.status

    @property
    def arm_names(self) -> tuple[str, ...]:
        """Return flattened arm names in configured assembly order."""

        return tuple(
            name
            for group_name in self._arm_groups
            for name in self._arm_names[group_name]
        )

    @property
    def end_effector_names(self) -> Mapping[str, tuple[str, ...]]:
        """Return immutable end-effector group names and joint order."""

        return self._end_effector_names

    def connect(self) -> None:
        """Connect the owned follower runtime."""

        self.runtime.connect()
        try:
            self._start_motion_loop()
        except BaseException as start_error:
            try:
                self.runtime.disconnect()
            except BaseException as cleanup_error:
                raise start_error from cleanup_error
            raise

    @_serialized_session
    def hold_position(self) -> None:
        """Enable actuator holding at the latest measured position without tracking."""

        with self._state_lock:
            safety_state = self.runtime.diagnostics().safety.state
            if safety_state == RuntimeSafetyState.SAFE_DISABLED:
                # Runtime enable seeds every actuator goal from the latest complete state
                # sample before torque is enabled, so standalone holding cannot reuse a
                # stale power-on target.
                self.runtime.set_enabled(True)
                safety_state = RuntimeSafetyState.READY
            if safety_state in (RuntimeSafetyState.READY, RuntimeSafetyState.ACTIVE):
                self.runtime.hold()
            elif safety_state != RuntimeSafetyState.HOLD:
                raise RuntimeError(
                    f"cannot hold follower position from {safety_state.value}"
                )

    @_serialized_session
    def reset_peer(self) -> None:
        """Hold motion and invalidate all state owned by a previous leader session."""

        with self._state_lock:
            safety_state = self.runtime.diagnostics().safety.state
            if safety_state in (
                RuntimeSafetyState.READY,
                RuntimeSafetyState.ACTIVE,
                RuntimeSafetyState.HOLD,
            ):
                # Runtime hold clears every bus mailbox and advances its command
                # generation, so no frame from the retired peer can execute later.
                self.runtime.hold()
            self._sync.reset_peer()
            self._latest_state = None
            self._motion_targets.clear()
            self._clear_soft_stop()
            self._reset_cartesian_cycle()

    def read(self, *, now_ns: int) -> RobotState:
        """Return the latest state without coupling transport work to motion timing."""

        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("follower read time must be a non-negative integer")
        self._raise_motion_error()
        if self._motion_period_ns is None:
            state = self.runtime.read()
            with self._state_lock:
                self._latest_state = state
            self.step_motion(now_ns=now_ns)
            return state
        with self._state_lock:
            cached_state = self._latest_state
        if cached_state is not None:
            return cached_state
        # The first actor sample can become ready between connect() and the worker's first
        # cycle. Reading it here is side-effect free and does not clock motion.
        state = self.runtime.read()
        with self._state_lock:
            self._latest_state = state
        return state

    @_serialized_session
    def set_mode(self, mode: LeaderSyncMode) -> None:
        """Apply a leader mode and its corresponding runtime safety action."""

        if not isinstance(mode, LeaderSyncMode):
            raise ValueError("follower mode must be a LeaderSyncMode")
        with self._state_lock:
            if mode == self._sync.mode and mode != LeaderSyncMode.STOP:
                return
            self._reset_cartesian_cycle()
            self._motion_targets.clear()
            self._clear_soft_stop()
            safety_state = self.runtime.diagnostics().safety.state
            # Translate network-level synchronization into the stricter runtime lifecycle.
            # The sync controller is updated only after the hardware action succeeds.
            if mode == LeaderSyncMode.IDLE:
                # IDLE revokes remote command ownership. It must never energize hardware:
                # standalone startup holding is an explicit local policy implemented by
                # ``hold_position()``, not a side effect of an unauthenticated network mode.
                if safety_state in (RuntimeSafetyState.READY, RuntimeSafetyState.ACTIVE):
                    self.runtime.hold()
            elif mode in (LeaderSyncMode.SYNC_REQUEST, LeaderSyncMode.READY):
                # Synchronization holds the follower while the leader moves to its pose;
                # torque remains enabled, but streaming commands are invalidated.
                if safety_state == RuntimeSafetyState.FAULT:
                    raise RuntimeError("cannot synchronize a faulted follower")
                if safety_state == RuntimeSafetyState.SAFE_DISABLED:
                    self.runtime.set_enabled(True)
                    safety_state = RuntimeSafetyState.READY
                if safety_state in (RuntimeSafetyState.READY, RuntimeSafetyState.ACTIVE):
                    self.runtime.hold()
            elif mode == LeaderSyncMode.TRACKING:
                # READY admits the first arm heartbeat; RobotRuntime promotes that accepted
                # command to ACTIVE atomically with motion publication.
                if safety_state == RuntimeSafetyState.FAULT:
                    raise RuntimeError("cannot track with a faulted follower")
                if safety_state in (
                    RuntimeSafetyState.SAFE_DISABLED,
                    RuntimeSafetyState.HOLD,
                ):
                    self.runtime.set_enabled(True)
            elif mode == LeaderSyncMode.HOLD:
                if (
                    safety_state != RuntimeSafetyState.FAULT
                    and safety_state != RuntimeSafetyState.HOLD
                ):
                    self.runtime.hold()
            elif mode == LeaderSyncMode.STOP:
                self.runtime.emergency_stop()
            self._sync.set_mode(mode)

    def write_arm(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        *,
        now_ns: int,
    ) -> bool:
        """Submit the arm heartbeat command for the current synchronization cycle."""

        if self.teleop_mode != TeleopMode.JOINT:
            return False
        return self.write_tracking_frame(
            arm_names=names,
            arm_positions=positions,
            now_ns=now_ns,
        )

    def write_arm_pose(
        self,
        pose: EndEffectorPose,
        *,
        now_ns: int,
    ) -> bool:
        """Map one source TCP pose into a bounded arm command and refresh heartbeat."""

        if self.teleop_mode != TeleopMode.EE_POSE:
            return False
        return self.write_tracking_frame(
            arm_pose=pose,
            now_ns=now_ns,
        )

    @_serialized_session
    def write_tracking_frame(
        self,
        *,
        arm_names: Optional[Sequence[str]] = None,
        arm_positions: Optional[Sequence[float]] = None,
        arm_pose: Optional[EndEffectorPose] = None,
        end_effectors: Mapping[
            str,
            tuple[Sequence[str], Sequence[float]],
        ] = MappingProxyType({}),
        now_ns: int,
    ) -> bool:
        """Validate and commit every command from one leader frame atomically."""

        if not self._sync.command_allowed:
            return False
        commands: dict[str, JointCommand]
        if self.teleop_mode == TeleopMode.JOINT:
            if arm_names is None or arm_positions is None or arm_pose is not None:
                raise ValueError("joint teleoperation requires one arm joint command")
            commands = self._commands_for_groups(
                self._arm_names,
                arm_names,
                arm_positions,
                now_ns=now_ns,
            )
        else:
            if arm_pose is None or arm_names is not None or arm_positions is not None:
                raise ValueError("ee_pose teleoperation requires one pose command")
            if self._cartesian_controller is None:
                raise RuntimeError("Cartesian controller is unavailable")
            if self._latest_state is None:
                return False
            group_name = self._arm_groups[0]
            current = self._latest_state.joints[group_name].positions
            result = self._cartesian_controller.solve(
                arm_pose,
                current,
                timestamp_ns=now_ns,
            )
            commands = self._commands_for_groups(
                {group_name: self._arm_names[group_name]},
                self._arm_names[group_name],
                result.positions,
                now_ns=now_ns,
            )
        for group_name, (names, positions) in end_effectors.items():
            try:
                expected = self._end_effector_names[group_name]
            except KeyError as exc:
                raise ValueError(f"unknown end-effector group '{group_name}'") from exc
            commands.update(
                self._commands_for_groups(
                    {group_name: expected},
                    names,
                    positions,
                    now_ns=now_ns,
                )
            )
        # Network arrival is deliberately not an actuator clock. Commit only the newest
        # validated targets here; ``step_motion()`` republishes them from the follower's
        # stable local cycle and owns the hardware lease.
        accepted_ns = self._local_time()
        self._motion_targets.update(commands)
        # Only accepted network arm frames refresh the peer lease. Local target replay
        # keeps the actuator loop healthy but cannot conceal a disconnected leader.
        self._sync.accept_command(accepted_ns)
        return True

    @_serialized_session
    def step_motion(self, *, now_ns: int) -> bool:
        """Submit the latest target from the follower's local periodic loop.

        A short network gap therefore converges to the last commanded position instead
        of issuing HOLD and re-enable transactions. A missed *local* loop still trips the
        independent bus watchdog, which requires a fresh synchronization to recover.
        """

        self.update(now_ns=now_ns)
        soft_stopping = self._soft_stop_started_ns is not None
        if self._sync.status != FollowerSyncStatus.TRACKING and not soft_stopping:
            return False
        if any(group_name not in self._motion_targets for group_name in self._arm_groups):
            return False
        submitted_at_ns = self._local_time()
        generation = self.runtime.generation
        commands = {
            group_name: replace(
                command,
                submitted_at_ns=submitted_at_ns,
                deadline_ns=(
                    submitted_at_ns + self._command_lifetimes_ns[group_name]
                ),
                generation=generation,
            )
            for group_name, command in self._motion_targets.items()
        }
        try:
            self.runtime.write(RobotCommand(commands))
        except (RuntimeError, ValueError) as motion_error:
            # A local actuator-loop failure is materially different from packet jitter.
            # Close admission and force the normal synchronization path to re-establish
            # generation and position references; never auto-resume from a late packet.
            self._motion_targets.clear()
            self._clear_soft_stop()
            self._sync.set_mode(LeaderSyncMode.HOLD)
            self._reset_cartesian_cycle()
            hold_error: Optional[BaseException] = None
            try:
                if self.runtime.diagnostics().safety.state in (
                    RuntimeSafetyState.READY,
                    RuntimeSafetyState.ACTIVE,
                ):
                    self.runtime.hold()
            except BaseException as exc:
                hold_error = exc
            if hold_error is not None:
                raise motion_error from hold_error
            raise
        if soft_stopping and self._soft_stop_complete(now_ns):
            self.runtime.hold()
            self._motion_targets.clear()
            self._clear_soft_stop()
        return True

    def end_effector_pose(
        self,
        state: RobotState,
        *,
        timestamp_ns: int,
    ) -> EndEffectorPose:
        """Return the measured follower TCP pose for feedback and diagnostics."""

        if self.teleop_mode != TeleopMode.EE_POSE or self._cartesian_controller is None:
            raise RuntimeError("follower end-effector pose is available only in ee_pose mode")
        if not isinstance(state, RobotState):
            raise ValueError("follower pose requires a RobotState")
        group_name = self._arm_groups[0]
        return self._cartesian_controller.kinematics.forward(
            state.joints[group_name].positions,
            timestamp_ns=timestamp_ns,
        )

    def cartesian_diagnostics(self) -> Optional[CartesianTeleopDiagnostics]:
        """Return the immutable most recent Cartesian solve diagnostics."""

        if self._cartesian_controller is None:
            return None
        return self._cartesian_controller.diagnostics()

    @_serialized_session
    def write_end_effector(
        self,
        group_name: str,
        names: Sequence[str],
        positions: Sequence[float],
        *,
        now_ns: int,
    ) -> bool:
        """Submit end-effector motion only after this cycle has an arm heartbeat."""

        last_arm_command_ns = self._sync.last_command_ns
        if (
            self.update(now_ns=now_ns) != FollowerSyncStatus.TRACKING
            or last_arm_command_ns is None
        ):
            # Auxiliary traffic never establishes or refreshes the remote arm lease.
            return False
        try:
            expected = self._end_effector_names[group_name]
        except KeyError as exc:
            raise ValueError(f"unknown end-effector group '{group_name}'") from exc
        commands = self._commands_for_groups(
            {group_name: expected},
            names,
            positions,
            now_ns=now_ns,
        )
        self._motion_targets.update(commands)
        return True

    @_serialized_session
    def update(self, *, now_ns: int) -> FollowerSyncStatus:
        """Start a bounded local stop when the remote heartbeat expires."""

        previous = self._sync.status
        current = self._sync.update(now_ns)
        if previous == FollowerSyncStatus.TRACKING and current == FollowerSyncStatus.LOST:
            self._reset_cartesian_cycle()
            if not self._begin_soft_stop(now_ns):
                self._motion_targets.clear()
                self.runtime.hold()
        return current

    def close(self, *, preserve_hold: bool = False) -> None:
        """Disconnect, optionally preserving the last holding goal after a fault."""

        loop_error = self._stop_motion_loop()
        try:
            self.runtime.disconnect(preserve_hold=preserve_hold)
        except BaseException as disconnect_error:
            if loop_error is not None:
                raise disconnect_error from loop_error
            raise
        if loop_error is not None:
            raise loop_error

    def _begin_soft_stop(self, now_ns: int) -> bool:
        """Replace remote targets with braking goals derived from measured motion."""

        state = self._latest_state
        if state is None:
            return False
        commands: dict[str, JointCommand] = {}
        for group_name, group in self.runtime.joint_groups.items():
            measured = state.joints.get(group_name)
            if measured is None:
                return False
            target = measured.positions.copy()
            if group_name in self._arm_names:
                metadata = self.runtime.preflight.arms[group_name]
                velocity_limits = np.minimum(
                    np.asarray(metadata.velocity_limits, dtype=float),
                    self._motion_tuning.velocity_limit_rad_s,
                )
                velocity = np.clip(
                    measured.velocities,
                    -velocity_limits,
                    velocity_limits,
                )
                target += (
                    np.sign(velocity)
                    * np.square(velocity)
                    / (2.0 * self._motion_tuning.acceleration_limit_rad_s2)
                )
                target = np.clip(
                    target,
                    np.asarray(metadata.lower_limits, dtype=float),
                    np.asarray(metadata.upper_limits, dtype=float),
                )
            commands.update(
                self._commands_for_groups(
                    {group_name: group.names},
                    group.names,
                    target,
                    now_ns=now_ns,
                )
            )
        self._motion_targets = commands
        self._soft_stop_started_ns = now_ns
        self._soft_stop_settled_since_ns = None
        return True

    def _soft_stop_complete(self, now_ns: int) -> bool:
        """Return true after measured arm velocity settles or the stop budget expires."""

        if self._soft_stop_started_ns is None:
            return False
        elapsed_ns = now_ns - self._soft_stop_started_ns
        if elapsed_ns >= round(self._motion_tuning.stop_timeout_s * 1e9):
            return True
        state = self._latest_state
        if state is None:
            return False
        stopped = all(
            np.all(
                np.abs(state.joints[group_name].velocities)
                <= self._motion_tuning.stop_velocity_threshold_rad_s
            )
            for group_name in self._arm_groups
        )
        if not stopped:
            self._soft_stop_settled_since_ns = None
            return False
        if self._soft_stop_settled_since_ns is None:
            self._soft_stop_settled_since_ns = now_ns
            return False
        return now_ns - self._soft_stop_settled_since_ns >= round(
            self._motion_tuning.stop_settle_time_s * 1e9
        )

    def _clear_soft_stop(self) -> None:
        """Forget transient braking state after a safety or synchronization transition."""

        self._soft_stop_started_ns = None
        self._soft_stop_settled_since_ns = None

    def _start_motion_loop(self) -> None:
        """Start the transport-independent monotonic state and motion scheduler."""

        if self._motion_period_ns is None:
            return
        if self._motion_worker is not None and self._motion_worker.is_alive():
            raise RuntimeError("follower motion loop is already running")
        self._motion_stop.clear()
        self._motion_error = None
        worker = Thread(
            target=self._run_motion_loop,
            name="acetele-follower-motion",
            daemon=True,
        )
        self._motion_worker = worker
        worker.start()

    def _run_motion_loop(self) -> None:
        """Read measured state and refresh motion independently of adapter publishing."""

        if self._motion_period_ns is None:
            return
        next_tick_ns = self._local_time()
        while not self._motion_stop.is_set():
            now_ns = self._local_time()
            wait_s = (next_tick_ns - now_ns) / 1e9
            if wait_s > 0.0 and self._motion_stop.wait(wait_s):
                return
            now_ns = self._local_time()
            try:
                state = self.runtime.read()
                with self._state_lock:
                    self._latest_state = state
                self.step_motion(now_ns=now_ns)
            except BaseException as exc:
                if self._motion_stop.is_set():
                    return
                with self._state_lock:
                    initial_state_pending = self._latest_state is None
                safety_state = self.runtime.diagnostics().safety.state
                if initial_state_pending and safety_state != RuntimeSafetyState.FAULT:
                    next_tick_ns = max(
                        next_tick_ns + self._motion_period_ns,
                        self._local_time(),
                    )
                    continue
                with self._state_lock:
                    self._motion_error = exc
                    self._motion_targets.clear()
                    self._clear_soft_stop()
                if safety_state in (
                    RuntimeSafetyState.READY,
                    RuntimeSafetyState.ACTIVE,
                ):
                    try:
                        self.runtime.hold()
                    except BaseException:
                        pass
                return
            next_tick_ns = max(
                next_tick_ns + self._motion_period_ns,
                self._local_time(),
            )

    def _stop_motion_loop(self) -> Optional[BaseException]:
        """Stop the local scheduler without allowing an unbounded shutdown wait."""

        worker = self._motion_worker
        if worker is None:
            return None
        self._motion_stop.set()
        timeout_s = max(1.0, 5.0 * (self._motion_period_ns or 0) / 1e9)
        worker.join(timeout_s)
        if worker.is_alive():
            return RuntimeError("follower motion loop did not stop within its deadline")
        self._motion_worker = None
        return None

    def _raise_motion_error(self) -> None:
        """Surface a failed local scheduler to ROS/ZMQ instead of serving stale state."""

        with self._state_lock:
            error = self._motion_error
        if error is not None:
            raise RuntimeError("follower local motion loop failed") from error

    def _commands_for_groups(
        self,
        groups: Mapping[str, tuple[str, ...]],
        names: Sequence[str],
        positions: Sequence[float],
        *,
        now_ns: int,
    ) -> dict[str, JointCommand]:
        """Split one ordered wire vector into generation-bound runtime commands."""

        names = tuple(names)
        positions = tuple(positions)
        expected_names = tuple(
            name for group_names in groups.values() for name in group_names
        )
        if names != expected_names:
            raise ValueError(
                f"follower command expects joint order {expected_names}, got {names}"
            )
        if len(positions) != len(expected_names):
            raise ValueError(
                f"follower command expects {len(expected_names)} positions, got {len(positions)}"
            )
        generation = self.runtime.generation
        result: dict[str, JointCommand] = {}
        offset = 0
        # Group insertion order is the public wire order established from RobotSpec.
        for group_name, group_names in groups.items():
            count = len(group_names)
            group = self.runtime.joint_groups[group_name]
            velocity_limits = None
            acceleration_limits = None
            if group_name in self._arm_names:
                metadata = self.runtime.preflight.arms[group_name]
                velocity_limits = np.minimum(
                    np.asarray(metadata.velocity_limits, dtype=float),
                    self._motion_tuning.velocity_limit_rad_s,
                )
                # This is first a software trajectory constraint. Adapters that expose a
                # matching hardware profile may additionally encode it in the servo.
                acceleration_limits = np.full(
                    count,
                    self._motion_tuning.acceleration_limit_rad_s2,
                )
            result[group_name] = JointCommand(
                group_names,
                positions[offset : offset + count],
                now_ns,
                now_ns + self._command_lifetimes_ns[group_name],
                generation,
                group.unit,
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
            )
            offset += count
        return result

    def _resolve_cartesian_controller(
        self,
        supplied: Optional[CartesianTeleopController],
        *,
        translation_scale: float,
        rotation_scale: float,
        tuning: CartesianTeleopTuning,
    ) -> Optional[CartesianTeleopController]:
        """Create the one-arm Cartesian controller before any hardware is connected."""

        if self.teleop_mode == TeleopMode.JOINT:
            if supplied is not None:
                raise ValueError("joint teleop mode does not accept a Cartesian controller")
            return None
        if len(self.runtime.spec.arms) != 1:
            raise ValueError("ee_pose teleoperation currently requires exactly one arm")
        arm = self.runtime.spec.arms[0]
        if arm.tool_frame is None:
            raise ValueError(
                f"arm '{arm.name}' requires tool_frame for ee_pose teleoperation"
            )
        controller = supplied
        if controller is None:
            if self.runtime.spec.urdf_path is None:
                raise ValueError("ee_pose teleoperation requires RobotSpec.urdf_path")
            kinematics = ArmKinematics(
                Path(self.runtime.spec.urdf_path),
                self._arm_names[arm.name],
                arm.tool_frame,
            )
            controller = CartesianTeleopController(
                kinematics,
                translation_scale=translation_scale,
                rotation_scale=rotation_scale,
                tuning=tuning,
            )
        if controller.kinematics.joint_names != self._arm_names[arm.name]:
            raise ValueError("follower kinematics joint order does not match RobotSpec")
        return controller

    def _reset_cartesian_cycle(self) -> None:
        """Invalidate relative-pose anchors whenever synchronization changes."""

        if self._cartesian_controller is not None:
            self._cartesian_controller.reset()

    def _local_time(self) -> int:
        """Read and validate the local monotonic clock used for safety leases."""

        value = self._clock_ns()
        if type(value) is not int or value < 0:
            raise RuntimeError("follower session clock returned an invalid timestamp")
        return value


__all__ = ["FollowerTeleopSession"]
