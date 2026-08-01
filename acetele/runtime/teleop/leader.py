"""Transport-independent leader synchronization and teleoperation session."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from acetele.core import EndEffectorPose, JointCommand, RobotCommand, RobotState
from acetele.model import ArmKinematics, wrap_to_pi
from acetele.runtime.robot import RobotRuntime
from acetele.runtime.safety import RuntimeSafetyState
from acetele.runtime.teleop.synchronization import (
    FollowerSyncStatus,
    LeaderSyncMode,
    TeleopMode,
)


class LeaderTeleopSession:
    """Pure Python leader alignment and tracking state machine."""

    def __init__(
        self,
        runtime: RobotRuntime,
        *,
        follower_timeout_ns: int = 500_000_000,
        sync_tolerance_rad: float = 0.03,
        sync_stable_ns: int = 200_000_000,
        sync_velocity_limit_rad_s: float = 2.0,
        sync_acceleration_limit_rad_s2: float = 3.0,
        command_deadline_ns: int = 50_000_000,
        teleop_mode: TeleopMode = TeleopMode.JOINT,
        kinematics: Optional[ArmKinematics] = None,
    ) -> None:
        if not isinstance(runtime, RobotRuntime):
            raise ValueError("leader session requires a RobotRuntime")
        for name, value in (
            ("follower_timeout_ns", follower_timeout_ns),
            ("sync_stable_ns", sync_stable_ns),
            ("command_deadline_ns", command_deadline_ns),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(sync_tolerance_rad) or sync_tolerance_rad <= 0.0:
            raise ValueError("sync_tolerance_rad must be finite and positive")
        for name, limit in (
            ("sync_velocity_limit_rad_s", sync_velocity_limit_rad_s),
            ("sync_acceleration_limit_rad_s2", sync_acceleration_limit_rad_s2),
        ):
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.runtime = runtime
        try:
            self.teleop_mode = TeleopMode(teleop_mode)
        except ValueError as exc:
            raise ValueError("teleop_mode must be 'joint' or 'ee_pose'") from exc
        self.mode = LeaderSyncMode.IDLE
        self._follower_timeout_ns = follower_timeout_ns
        self._sync_tolerance_rad = sync_tolerance_rad
        self._sync_stable_ns = sync_stable_ns
        self._sync_velocity_limit_rad_s = sync_velocity_limit_rad_s
        self._sync_acceleration_limit_rad_s2 = sync_acceleration_limit_rad_s2
        self._arm_groups = tuple(arm.name for arm in runtime.spec.arms)
        self._arm_names = {
            arm.name: tuple(joint.name for joint in arm.joints)
            for arm in runtime.spec.arms
        }
        # Alignment uses acceleration only when preflight proved that the selected
        # adapter can encode it; teleop remains independent of protocol families.
        self._acceleration_groups = frozenset(
            arm.name
            for arm in runtime.spec.arms
            if runtime.preflight.buses[arm.bus].supports_acceleration_limits
        )
        self._flat_arm_names = tuple(
            name
            for group_name in self._arm_groups
            for name in self._arm_names[group_name]
        )
        self._kinematics = self._resolve_kinematics(kinematics)
        # A slow bus may need a command lifetime longer than the network default;
        # derive it once from the runtime's validated bus schedule.
        self._command_lifetimes_ns = {
            group_name: runtime.command_lifetime_ns(
                group_name,
                minimum_ns=command_deadline_ns,
            )
            for group_name in self._arm_groups
        }
        self._follower_positions: Optional[np.ndarray] = None
        self._last_follower_state_ns: Optional[int] = None
        self._follower_status = FollowerSyncStatus.IDLE
        self._last_follower_status_ns: Optional[int] = None
        self._sync_target: Optional[np.ndarray] = None
        self._stable_since_ns: Optional[int] = None

    @property
    def arm_names(self) -> tuple[str, ...]:
        """Return flattened arm names in configured assembly order."""

        return self._flat_arm_names

    @property
    def follower_status(self) -> FollowerSyncStatus:
        """Return the last status received from the follower."""

        return self._follower_status

    def end_effector_pose(
        self,
        state: RobotState,
        *,
        timestamp_ns: int,
    ) -> EndEffectorPose:
        """Return the current leader TCP pose while Cartesian teleoperation is selected."""

        if self.teleop_mode != TeleopMode.EE_POSE or self._kinematics is None:
            raise RuntimeError("leader end-effector pose is available only in ee_pose mode")
        if not isinstance(state, RobotState):
            raise ValueError("leader pose requires a RobotState")
        group_name = self._arm_groups[0]
        return self._kinematics.forward(
            state.joints[group_name].positions,
            timestamp_ns=self._time(timestamp_ns),
        )

    def connect(self) -> None:
        """Connect the owned leader runtime."""

        self.runtime.connect()

    def reset_peer(self) -> None:
        """Stop alignment and discard observations from a restarted follower."""

        if self.mode == LeaderSyncMode.STOP:
            return
        safety_state = self.runtime.diagnostics().safety.state
        if safety_state in (
            RuntimeSafetyState.READY,
            RuntimeSafetyState.ACTIVE,
            RuntimeSafetyState.HOLD,
        ):
            # A peer reset must release the complete leader, including a gripper used
            # as the next synchronization cycle's physical start trigger.
            self.runtime.set_enabled(False)
        self.mode = LeaderSyncMode.IDLE
        self._follower_positions = None
        self._last_follower_state_ns = None
        self._follower_status = FollowerSyncStatus.IDLE
        self._last_follower_status_ns = None
        self._sync_target = None
        self._stable_since_ns = None

    def observe_follower_state(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        *,
        now_ns: int,
    ) -> None:
        """Record one correctly ordered follower sample for alignment."""

        names = tuple(names)
        values = np.asarray(positions, dtype=float)
        if names != self._flat_arm_names:
            raise ValueError(
                f"follower state expects joint order {self._flat_arm_names}, got {names}"
            )
        if values.shape != (len(names),) or not np.all(np.isfinite(values)):
            raise ValueError("follower state positions must be a finite arm vector")
        self._follower_positions = values.copy()
        self._last_follower_state_ns = self._time(now_ns)

    def observe_follower_status(
        self,
        status: FollowerSyncStatus,
        *,
        now_ns: int,
    ) -> None:
        """Record one timestamped follower synchronization status."""

        if not isinstance(status, FollowerSyncStatus):
            raise ValueError("follower status must be a FollowerSyncStatus")
        self._follower_status = status
        self._last_follower_status_ns = self._time(now_ns)

    def step(self, *, now_ns: int) -> RobotState:
        """Read leader state and advance alignment or tracking behavior."""

        now_ns = self._time(now_ns)
        state = self.runtime.read()
        if self.mode == LeaderSyncMode.STOP:
            return state
        if self.mode == LeaderSyncMode.TRACKING and (
            self._follower_status in (FollowerSyncStatus.LOST, FollowerSyncStatus.FAULT)
            or not self._follower_recent(now_ns)
            or not self._status_recent(now_ns)
        ):
            # Any missing follower heartbeat invalidates tracking. Re-enter alignment
            # rather than resuming from a potentially stale remote pose.
            self.request_sync()
        if self.mode == LeaderSyncMode.IDLE and self._follower_recent(now_ns):
            self.request_sync()
        if self.mode in (LeaderSyncMode.SYNC_REQUEST, LeaderSyncMode.READY):
            self._align(state, now_ns)
        return state

    def request_sync(self) -> None:
        """Begin alignment from the latest valid follower pose."""

        if self.mode == LeaderSyncMode.STOP:
            raise RuntimeError("cannot synchronize a stopped leader")
        self.mode = LeaderSyncMode.SYNC_REQUEST
        self._sync_target = None
        self._stable_since_ns = None

    def start_tracking(self) -> None:
        """Enter tracking after synchronization criteria have held."""

        if self.mode != LeaderSyncMode.READY:
            raise RuntimeError("leader must be READY before tracking starts")
        safety_state = self.runtime.diagnostics().safety.state
        # The leader becomes passive once aligned: tracking publishes encoder state and
        # must not leave its own servos actively holding the synchronization target.
        if safety_state != RuntimeSafetyState.SAFE_DISABLED:
            self.runtime.set_enabled(False)
        self.mode = LeaderSyncMode.TRACKING

    def stop(self) -> None:
        """Latch STOP and emergency-stop the leader runtime."""

        if self.mode != LeaderSyncMode.STOP:
            self.mode = LeaderSyncMode.STOP
            self.runtime.emergency_stop()

    def close(self) -> None:
        """Disconnect the owned leader runtime."""

        self.runtime.disconnect()

    def _align(self, state: RobotState, now_ns: int) -> None:
        """Move the leader to one captured follower pose and require stable convergence."""

        if not self._follower_recent(now_ns) or self._follower_positions is None:
            return
        if self._sync_target is None:
            # Freeze one target for the entire alignment attempt. Following a moving
            # follower here would make the READY criterion unstable and unpredictable.
            self._sync_target = self._follower_positions.copy()
        safety_state = self.runtime.diagnostics().safety.state
        if safety_state in (RuntimeSafetyState.SAFE_DISABLED, RuntimeSafetyState.HOLD):
            # Only arm joints hold the synchronization target. End effectors stay
            # passive so the operator can move the gripper trigger into TRACKING.
            self.runtime.enable_arm_groups(self._arm_groups)
        elif safety_state == RuntimeSafetyState.FAULT:
            raise RuntimeError("cannot align a faulted leader")
        self.runtime.write(self._target_command(now_ns))

        current = np.concatenate(
            [state.joints[group].positions for group in self._arm_groups]
        )
        # Compare shortest angular distance so joints near the wrap boundary do not
        # appear one full revolution apart.
        error = wrap_to_pi(self._sync_target - current)
        if np.all(np.abs(error) <= self._sync_tolerance_rad):
            if self._stable_since_ns is None:
                self._stable_since_ns = now_ns
            elif now_ns - self._stable_since_ns >= self._sync_stable_ns:
                self.mode = LeaderSyncMode.READY
        else:
            self._stable_since_ns = None

    def _target_command(self, now_ns: int) -> RobotCommand:
        """Split the frozen alignment vector into bounded per-arm commands."""

        if self._sync_target is None:
            raise RuntimeError("leader synchronization target is unavailable")
        generation = self.runtime.generation
        commands: dict[str, JointCommand] = {}
        offset = 0
        groups = self.runtime.joint_groups
        # Only FEETECH position profiles expose an acceleration field. Other buses get
        # the common velocity limit without pretending to support that capability.
        for group_name in self._arm_groups:
            names = self._arm_names[group_name]
            count = len(names)
            commands[group_name] = JointCommand(
                names,
                self._sync_target[offset : offset + count],
                now_ns,
                now_ns + self._command_lifetimes_ns[group_name],
                generation,
                groups[group_name].unit,
                velocity_limits=np.full(count, self._sync_velocity_limit_rad_s),
                acceleration_limits=(
                    np.full(count, self._sync_acceleration_limit_rad_s2)
                    if group_name in self._acceleration_groups
                    else None
                ),
            )
            offset += count
        return RobotCommand(commands)

    def _follower_recent(self, now_ns: int) -> bool:
        """Return whether the last follower joint sample is still usable."""

        return (
            self._last_follower_state_ns is not None
            and now_ns - self._last_follower_state_ns <= self._follower_timeout_ns
        )

    def _status_recent(self, now_ns: int) -> bool:
        """Return whether the last follower synchronization status is still usable."""

        return (
            self._last_follower_status_ns is not None
            and now_ns - self._last_follower_status_ns <= self._follower_timeout_ns
        )

    def _resolve_kinematics(
        self,
        supplied: Optional[ArmKinematics],
    ) -> Optional[ArmKinematics]:
        """Build and validate the single-arm model required only by ee_pose mode."""

        if self.teleop_mode == TeleopMode.JOINT:
            if supplied is not None:
                raise ValueError("joint teleop mode does not accept Cartesian kinematics")
            return None
        if len(self.runtime.spec.arms) != 1:
            raise ValueError("ee_pose teleoperation currently requires exactly one arm")
        arm = self.runtime.spec.arms[0]
        if arm.tool_frame is None:
            raise ValueError(
                f"arm '{arm.name}' requires tool_frame for ee_pose teleoperation"
            )
        model = supplied or ArmKinematics(
            self.runtime.preflight.urdf_path,
            self._arm_names[arm.name],
            arm.tool_frame,
        )
        if model.joint_names != self._arm_names[arm.name]:
            raise ValueError("leader kinematics joint order does not match RobotSpec")
        return model

    @staticmethod
    def _time(value: int) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("session time must be a non-negative integer")
        return value


__all__ = ["LeaderTeleopSession"]
