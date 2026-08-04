"""Thread-free arm position conditioning with transactional state updates.

The pipeline operates in SI units. It can prepare controller state with an output and
commit that state only after all target buses accept the motion batch, preventing failed
multi-bus submissions from advancing command history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from acetele.core import JointCommand, JointState, JointUnit
from acetele.model import ArmModelMetadata, unwrap_near, wrap_to_pi


@dataclass(frozen=True)
class StreamingPositionTuning:
    """Conservative follower limits for discontinuous latest-value targets.

    Reissuing the latest network target removes timing jitter from the actuator clock;
    these limits additionally keep a target replacement from requesting an instantaneous
    velocity change after a longer packet gap. Every arm applies both limits in software;
    capable adapters also forward the same profile to the servo as a second boundary.
    """

    velocity_limit_rad_s: float = 4.0
    acceleration_limit_rad_s2: float = 12.0
    stop_velocity_threshold_rad_s: float = 0.05
    stop_settle_time_s: float = 0.10
    stop_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"streaming position {name} must be finite and positive")


@dataclass(frozen=True)
class PositionControlDiagnostics:
    """Read-only target and constraint flags for one arm."""

    target_rad: np.ndarray
    position_error_rad: np.ndarray
    command_limited: np.ndarray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = np.asarray(getattr(self, name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class _PositionControlState:
    last_update_ns: Optional[int]
    last_output: Optional[np.ndarray]
    last_velocity: Optional[np.ndarray]
    diagnostic_target: np.ndarray
    limited: np.ndarray


@dataclass(frozen=True)
class _PreparedPositionCommand:
    pipeline: "PositionControlPipeline"
    command: JointCommand
    state: _PositionControlState


class PositionControlPipeline:
    """Thread-free position command conditioning for one arm."""

    def __init__(
        self,
        metadata: ArmModelMetadata,
    ) -> None:
        self._metadata = metadata
        self._count = len(metadata.joint_names)
        self._lower = np.asarray(metadata.lower_limits, dtype=float)
        self._upper = np.asarray(metadata.upper_limits, dtype=float)
        self._velocity_limits = np.asarray(metadata.velocity_limits, dtype=float)
        self._feedback: Optional[JointState] = None
        self._last_update_ns: Optional[int] = None
        self._last_output: Optional[np.ndarray] = None
        self._last_velocity: Optional[np.ndarray] = None
        self._diagnostic_target = np.zeros(self._count)
        self._limited = np.zeros(self._count, dtype=bool)

    def update_feedback(self, state: JointState) -> None:
        """Set the measured arm state used by model-based feed-forward."""

        if state.names != self._metadata.joint_names or state.unit != JointUnit.RADIAN:
            raise ValueError("position pipeline feedback does not match its arm")
        self._feedback = state
        if self._last_output is None:
            self._last_output = state.positions.copy()
            self._last_velocity = np.clip(
                state.velocities,
                -self._velocity_limits,
                self._velocity_limits,
            )
            self._diagnostic_target = state.positions.copy()

    def rebase_to_feedback(self) -> None:
        """Restart controller history from the latest measured arm position."""
        self._last_update_ns = None
        self._last_output = (
            None
            if self._feedback is None
            else np.clip(self._feedback.positions, self._lower, self._upper)
        )
        self._last_velocity = (
            None
            if self._feedback is None
            else np.clip(
                self._feedback.velocities,
                -self._velocity_limits,
                self._velocity_limits,
            )
        )
        self._diagnostic_target = (
            np.zeros(self._count)
            if self._feedback is None
            else self._feedback.positions.copy()
        )
        self._limited.fill(False)

    def apply(self, command: JointCommand, *, now_ns: int) -> JointCommand:
        """Condition and immediately commit a command for single-stage callers."""

        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("position pipeline time must be a non-negative integer")
        if command.names != self._metadata.joint_names or command.unit != JointUnit.RADIAN:
            raise ValueError("position pipeline command does not match its arm")
        # Establish a legal target, then enforce physical position and slew limits on
        # the command that reaches the hardware. Model effort belongs to the separate
        # Torque-mode controller and must never be translated into a position offset.
        target = unwrap_near(command.positions, 0.5 * (self._lower + self._upper))
        target = np.clip(target, self._lower, self._upper)
        self._diagnostic_target = target.copy()
        dt = self._dt(now_ns)
        bounded = target.copy()
        limited = ~np.isclose(command.positions, bounded)

        if self._last_output is not None:
            velocity_limits = self._velocity_limits.copy()
            if command.velocity_limits is not None:
                velocity_limits = np.minimum(velocity_limits, command.velocity_limits)
            if command.acceleration_limits is None:
                maximum_step = velocity_limits * dt
                rate_limited = np.clip(
                    bounded,
                    self._last_output - maximum_step,
                    self._last_output + maximum_step,
                )
                next_velocity = (rate_limited - self._last_output) / dt
            else:
                acceleration_limits = command.acceleration_limits
                error = bounded - self._last_output
                # The square-root bound is the greatest velocity that can still brake
                # inside the remaining distance. Acceleration limiting then makes target
                # replacement and direction reversal continuous in velocity.
                braking_velocity = np.maximum(
                    0.0,
                    np.sqrt(2.0 * acceleration_limits * np.abs(error))
                    - acceleration_limits * dt,
                )
                desired_velocity = np.sign(error) * np.minimum(
                    velocity_limits,
                    braking_velocity,
                )
                previous_velocity = (
                    np.zeros(self._count)
                    if self._last_velocity is None
                    else np.clip(
                        self._last_velocity,
                        -velocity_limits,
                        velocity_limits,
                    )
                )
                velocity_step = acceleration_limits * dt
                next_velocity = np.clip(
                    desired_velocity,
                    previous_velocity - velocity_step,
                    previous_velocity + velocity_step,
                )
                rate_limited = self._last_output + 0.5 * (
                    previous_velocity + next_velocity
                ) * dt
                rate_limited = np.clip(rate_limited, self._lower, self._upper)
                # Snap only when the previous velocity can legally reach zero in this
                # interval. This removes sub-count dithering without violating the
                # acceleration bound at the final sample.
                settled = (
                    np.abs(bounded - rate_limited)
                    <= 0.5 * acceleration_limits * dt * dt
                ) & (np.abs(previous_velocity) <= acceleration_limits * dt)
                if np.any(settled):
                    rate_limited[settled] = bounded[settled]
                    next_velocity[settled] = 0.0
            limited |= ~np.isclose(rate_limited, bounded)
            bounded = rate_limited
            self._last_velocity = next_velocity.copy()
        else:
            self._last_velocity = np.zeros(self._count)
        self._last_output = bounded.copy()
        self._last_update_ns = now_ns
        self._limited = limited
        return JointCommand(
            command.names,
            bounded,
            command.submitted_at_ns,
            command.deadline_ns,
            command.generation,
            command.unit,
            command.velocity_limits,
            command.acceleration_limits,
            command.effort_limits,
        )

    def prepare(self, command: JointCommand, *, now_ns: int) -> _PreparedPositionCommand:
        """Compute a command without committing controller history."""
        original = self._snapshot_state()
        try:
            prepared_command = self.apply(command, now_ns=now_ns)
            prepared_state = self._snapshot_state()
        finally:
            self._restore_state(original)
        return _PreparedPositionCommand(self, prepared_command, prepared_state)

    def commit(self, prepared: _PreparedPositionCommand) -> None:
        """Commit controller memory after its motion batch becomes visible."""

        if not isinstance(prepared, _PreparedPositionCommand) or prepared.pipeline is not self:
            raise ValueError("prepared position command belongs to another pipeline")
        self._restore_state(prepared.state)

    def diagnostics(self) -> PositionControlDiagnostics:
        """Return a snapshot that cannot mutate controller-owned arrays."""

        position_error = (
            np.full(self._count, np.nan)
            if self._feedback is None
            else np.asarray(
                wrap_to_pi(self._diagnostic_target - self._feedback.positions),
                dtype=float,
            )
        )
        return PositionControlDiagnostics(
            self._diagnostic_target,
            position_error,
            self._limited,
        )

    def _snapshot_state(self) -> _PositionControlState:
        """Copy every mutable controller field for a speculative update."""

        return _PositionControlState(
            self._last_update_ns,
            None if self._last_output is None else self._last_output.copy(),
            None if self._last_velocity is None else self._last_velocity.copy(),
            self._diagnostic_target.copy(),
            self._limited.copy(),
        )

    def _restore_state(self, state: _PositionControlState) -> None:
        """Restore a snapshot without sharing its writable NumPy arrays."""

        self._last_update_ns = state.last_update_ns
        self._last_output = (
            None if state.last_output is None else state.last_output.copy()
        )
        self._last_velocity = (
            None if state.last_velocity is None else state.last_velocity.copy()
        )
        self._diagnostic_target = state.diagnostic_target.copy()
        self._limited = state.limited.copy()

    def _dt(self, now_ns: int) -> float:
        """Return a bounded command interval after stalls or clock jitter."""

        if self._last_update_ns is None:
            return 0.001
        return float(np.clip((now_ns - self._last_update_ns) / 1e9, 0.001, 0.05))


__all__ = [
    "PositionControlDiagnostics",
    "PositionControlPipeline",
    "StreamingPositionTuning",
]
