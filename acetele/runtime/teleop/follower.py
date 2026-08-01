"""Transport-independent follower synchronization and command admission."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

from acetele.control import (
    CartesianTeleopController,
    CartesianTeleopDiagnostics,
    CartesianTeleopTuning,
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


class FollowerTeleopSession:
    """Transport-independent synchronization around a follower runtime."""

    def __init__(
        self,
        runtime: FollowerRuntime,
        *,
        heartbeat_timeout_ns: int = 100_000_000,
        command_deadline_ns: int = 50_000_000,
        teleop_mode: TeleopMode = TeleopMode.JOINT,
        translation_scale: float = 2.0,
        rotation_scale: float = 1.0,
        cartesian_tuning: CartesianTeleopTuning = CartesianTeleopTuning(),
        cartesian_controller: Optional[CartesianTeleopController] = None,
    ) -> None:
        if not isinstance(runtime, FollowerRuntime):
            raise ValueError("follower session requires a FollowerRuntime")
        for name, value in (
            ("heartbeat_timeout_ns", heartbeat_timeout_ns),
            ("command_deadline_ns", command_deadline_ns),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if heartbeat_timeout_ns != runtime.command_timeout_ns:
            raise ValueError(
                "follower heartbeat timeout must match the runtime command timeout"
            )
        self.runtime = runtime
        try:
            self.teleop_mode = TeleopMode(teleop_mode)
        except ValueError as exc:
            raise ValueError("teleop_mode must be 'joint' or 'ee_pose'") from exc
        self._sync = FollowerSyncController(heartbeat_timeout_ns)
        # Freeze routing metadata at construction. Runtime commands can then be
        # validated and partitioned without consulting mutable ROS messages.
        self._arm_groups = tuple(arm.name for arm in runtime.spec.arms)
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

    @property
    def mode(self) -> LeaderSyncMode:
        """Return the currently accepted leader mode."""

        return self._sync.mode

    @property
    def status(self) -> FollowerSyncStatus:
        """Return synchronization status, with runtime faults taking precedence."""

        if self.runtime.diagnostics().safety.state == RuntimeSafetyState.FAULT:
            return FollowerSyncStatus.FAULT
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

    def hold_position(self) -> None:
        """Enable actuator holding at the latest measured position without tracking."""

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

    def reset_peer(self) -> None:
        """Hold motion and invalidate all state owned by a previous leader session."""

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
        self._reset_cartesian_cycle()

    def read(self, *, now_ns: int) -> RobotState:
        """Read state and enforce synchronization heartbeat timeout."""

        state = self.runtime.read()
        self._latest_state = state
        self.update(now_ns=now_ns)
        return state

    def set_mode(self, mode: LeaderSyncMode) -> None:
        """Apply a leader mode and its corresponding runtime safety action."""

        if not isinstance(mode, LeaderSyncMode):
            raise ValueError("follower mode must be a LeaderSyncMode")
        if mode == self._sync.mode:
            return
        self._reset_cartesian_cycle()
        safety_state = self.runtime.diagnostics().safety.state
        # Translate network-level synchronization into the stricter runtime lifecycle.
        # The sync controller is updated only after the hardware action succeeds.
        if mode == LeaderSyncMode.IDLE:
            if safety_state not in (
                RuntimeSafetyState.SAFE_DISABLED,
                RuntimeSafetyState.FAULT,
            ):
                self.runtime.set_enabled(False)
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
        elif mode == LeaderSyncMode.STOP:
            if safety_state != RuntimeSafetyState.FAULT:
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
        self.runtime.write(RobotCommand(commands))
        # Refresh synchronization only after the complete frame is staged successfully.
        self._sync.accept_command(now_ns)
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

    def write_end_effector(
        self,
        group_name: str,
        names: Sequence[str],
        positions: Sequence[float],
        *,
        now_ns: int,
    ) -> bool:
        """Submit end-effector motion only after this cycle has an arm heartbeat."""

        if (
            self._sync.status != FollowerSyncStatus.TRACKING
            or self._sync.last_command_ns is None
        ):
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
        self.runtime.write(RobotCommand(commands))
        return True

    def update(self, *, now_ns: int) -> FollowerSyncStatus:
        """Enforce heartbeat timeout and enter HOLD when tracking is lost."""

        previous = self._sync.status
        current = self._sync.update(now_ns)
        if previous == FollowerSyncStatus.TRACKING and current == FollowerSyncStatus.LOST:
            self._reset_cartesian_cycle()
            if self.runtime.diagnostics().safety.state != RuntimeSafetyState.HOLD:
                self.runtime.hold()
        return current

    def close(self) -> None:
        """Disconnect the owned follower runtime."""

        self.runtime.disconnect()

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
            result[group_name] = JointCommand(
                group_names,
                positions[offset : offset + count],
                now_ns,
                now_ns + self._command_lifetimes_ns[group_name],
                generation,
                group.unit,
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


__all__ = ["FollowerTeleopSession"]
