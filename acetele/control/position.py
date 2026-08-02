"""Thread-free arm position conditioning with transactional state updates.

The pipeline operates in SI units. It can prepare controller state with an output and
commit that state only after all target buses accept the motion batch, preventing failed
multi-bus submissions from advancing adaptive memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from acetele.core import JointCommand, JointState, JointUnit
from acetele.model import ArmModelMetadata, unwrap_near, wrap_to_pi
from acetele.specification import ControlSpec


@dataclass(frozen=True)
class PositionControlDiagnostics:
    """Read-only offsets and constraint flags for one arm."""

    target_rad: np.ndarray
    position_error_rad: np.ndarray
    gravity_offset_rad: np.ndarray
    adaptive_estimate_rad: np.ndarray
    adaptive_offset_rad: np.ndarray
    adaptive_active: np.ndarray
    adaptive_saturated: np.ndarray
    command_limited: np.ndarray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = np.asarray(getattr(self, name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class _PositionControlState:
    estimate: np.ndarray
    offset: np.ndarray
    target: np.ndarray
    stable_target: np.ndarray
    learning_target: np.ndarray
    stable_since_ns: np.ndarray
    motion_direction: np.ndarray
    last_update_ns: Optional[int]
    last_output: Optional[np.ndarray]
    diagnostic_target: np.ndarray
    gravity_offset: np.ndarray
    active: np.ndarray
    saturated: np.ndarray
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
        control: ControlSpec,
        *,
        pin_model: Any = None,
    ) -> None:
        self._metadata = metadata
        self._control = control
        self._tuning = control.position_tuning
        self._count = len(metadata.joint_names)
        self._lower = np.asarray(metadata.lower_limits, dtype=float)
        self._upper = np.asarray(metadata.upper_limits, dtype=float)
        self._velocity_limits = np.asarray(metadata.velocity_limits, dtype=float)
        self._feedback: Optional[JointState] = None
        self._pin_model = pin_model
        self._pin_data = None if pin_model is None else pin_model.createData()
        if control.gravity_position and pin_model is None:
            raise ValueError("gravity position compensation requires a Pinocchio model")
        self._compliance = (
            np.zeros(self._count)
            if control.gravity_compliance_rad_per_nm is None
            else np.asarray(control.gravity_compliance_rad_per_nm, dtype=float)
        )
        self._estimate = np.zeros(self._count)
        self._offset = np.zeros(self._count)
        self._target = np.full(self._count, np.nan)
        self._stable_target = np.full(self._count, np.nan)
        self._learning_target = np.full(self._count, np.nan)
        self._stable_since_ns = np.full(self._count, -1, dtype=np.int64)
        self._motion_direction = np.zeros(self._count)
        self._last_update_ns: Optional[int] = None
        self._last_output: Optional[np.ndarray] = None
        self._diagnostic_target = np.zeros(self._count)
        self._gravity_offset = np.zeros(self._count)
        self._active = np.zeros(self._count, dtype=bool)
        self._saturated = np.zeros(self._count, dtype=bool)
        self._limited = np.zeros(self._count, dtype=bool)

    def update_feedback(self, state: JointState) -> None:
        """Set the measured arm state used by model and adaptive terms."""

        if state.names != self._metadata.joint_names or state.unit != JointUnit.RADIAN:
            raise ValueError("position pipeline feedback does not match its arm")
        self._feedback = state
        if self._last_output is None:
            self._last_output = state.positions.copy()
            self._diagnostic_target = state.positions.copy()

    def rebase_to_feedback(self) -> None:
        """Restart controller history from the latest measured arm position."""
        self._estimate.fill(0.0)
        self._offset.fill(0.0)
        self._target.fill(np.nan)
        self._stable_target.fill(np.nan)
        self._learning_target.fill(np.nan)
        self._stable_since_ns.fill(-1)
        self._motion_direction.fill(0.0)
        self._last_update_ns = None
        self._last_output = (
            None
            if self._feedback is None
            else np.clip(self._feedback.positions, self._lower, self._upper)
        )
        self._diagnostic_target = (
            np.zeros(self._count)
            if self._feedback is None
            else self._feedback.positions.copy()
        )
        self._gravity_offset.fill(0.0)
        self._active.fill(False)
        self._saturated.fill(False)
        self._limited.fill(False)

    def apply(self, command: JointCommand, *, now_ns: int) -> JointCommand:
        """Condition and immediately commit a command for single-stage callers."""

        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("position pipeline time must be a non-negative integer")
        if command.names != self._metadata.joint_names or command.unit != JointUnit.RADIAN:
            raise ValueError("position pipeline command does not match its arm")
        # Establish a legal physical target first, then add compensation, and finally
        # enforce position and slew limits. Reordering these steps defeats anti-windup.
        target = unwrap_near(command.positions, 0.5 * (self._lower + self._upper))
        target = np.clip(target, self._lower, self._upper)
        self._diagnostic_target = target.copy()
        self._saturated.fill(False)
        dt = self._dt(now_ns)
        gravity_offset = self._gravity_compensation()
        adaptive_offset = self._adaptive_compensation(
            target,
            gravity_offset,
            now_ns,
            dt,
        )
        compensated = target + gravity_offset + adaptive_offset
        bounded = np.clip(compensated, self._lower, self._upper)
        limited = ~np.isclose(compensated, bounded)

        if self._last_output is not None:
            velocity_limits = self._velocity_limits.copy()
            if command.velocity_limits is not None:
                velocity_limits = np.minimum(velocity_limits, command.velocity_limits)
            maximum_step = velocity_limits * dt
            rate_limited = np.clip(
                bounded,
                self._last_output - maximum_step,
                self._last_output + maximum_step,
            )
            limited |= ~np.isclose(rate_limited, bounded)
            bounded = rate_limited
        self._last_output = bounded.copy()
        self._last_update_ns = now_ns
        self._gravity_offset = gravity_offset
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
            self._gravity_offset,
            self._estimate,
            self._offset,
            self._active,
            self._saturated,
            self._limited,
        )

    def _snapshot_state(self) -> _PositionControlState:
        """Copy every mutable controller field for a speculative update."""

        return _PositionControlState(
            self._estimate.copy(),
            self._offset.copy(),
            self._target.copy(),
            self._stable_target.copy(),
            self._learning_target.copy(),
            self._stable_since_ns.copy(),
            self._motion_direction.copy(),
            self._last_update_ns,
            None if self._last_output is None else self._last_output.copy(),
            self._diagnostic_target.copy(),
            self._gravity_offset.copy(),
            self._active.copy(),
            self._saturated.copy(),
            self._limited.copy(),
        )

    def _restore_state(self, state: _PositionControlState) -> None:
        """Restore a snapshot without sharing its writable NumPy arrays."""

        self._estimate = state.estimate.copy()
        self._offset = state.offset.copy()
        self._target = state.target.copy()
        self._stable_target = state.stable_target.copy()
        self._learning_target = state.learning_target.copy()
        self._stable_since_ns = state.stable_since_ns.copy()
        self._motion_direction = state.motion_direction.copy()
        self._last_update_ns = state.last_update_ns
        self._last_output = (
            None if state.last_output is None else state.last_output.copy()
        )
        self._diagnostic_target = state.diagnostic_target.copy()
        self._gravity_offset = state.gravity_offset.copy()
        self._active = state.active.copy()
        self._saturated = state.saturated.copy()
        self._limited = state.limited.copy()

    def _dt(self, now_ns: int) -> float:
        """Return a bounded integration interval after stalls or clock jitter."""

        if self._last_update_ns is None:
            return self._tuning.minimum_dt_s
        return float(
            np.clip(
                (now_ns - self._last_update_ns) / 1e9,
                self._tuning.minimum_dt_s,
                self._tuning.maximum_dt_s,
            )
        )

    def _gravity_compensation(self) -> np.ndarray:
        """Map model gravity torque to calibrated position feed-forward."""

        if not self._control.gravity_position or self._feedback is None:
            return np.zeros(self._count)
        import pinocchio as pin

        torque = pin.rnea(
            self._pin_model,
            self._pin_data,
            self._feedback.positions,
            self._feedback.velocities,
            np.zeros(self._count),
        )
        return np.asarray(torque, dtype=float) * self._compliance

    def _adaptive_compensation(
        self,
        target: np.ndarray,
        gravity_offset: np.ndarray,
        now_ns: int,
        dt: float,
    ) -> np.ndarray:
        """Learn a bounded residual offset only around a stable target.

        Target motion, direction reversals, and joint motion suspend or reset learning
        so ordinary tracking lag is not mistaken for a persistent load disturbance.
        """

        if not self._control.adaptive_position or self._feedback is None:
            return np.zeros(self._count)
        # Learn only quasi-static residuals. Deliberate tracking motion is not evidence
        # of backlash or gravity disturbance and must not be integrated.
        measured = self._feedback.positions
        velocity = self._feedback.velocities
        for index in range(self._count):
            previous_target = self._target[index]
            reset = False
            if np.isnan(previous_target):
                self._stable_target[index] = target[index]
                self._learning_target[index] = target[index]
            else:
                target_delta = float(wrap_to_pi(target[index] - previous_target))
                if abs(target_delta) > self._tuning.target_direction_threshold_rad:
                    direction = float(np.sign(target_delta))
                    if self._motion_direction[index] not in (0.0, direction):
                        reset = True
                    self._motion_direction[index] = direction
                if (
                    abs(
                        wrap_to_pi(target[index] - self._learning_target[index])
                    )
                    > self._tuning.target_reset_threshold_rad
                ):
                    reset = True
            if reset:
                # A learned offset belongs to one local load posture and direction.
                # Carrying it through a reversal can initially reinforce backlash.
                self._estimate[index] = 0.0
                self._learning_target[index] = target[index]
                self._stable_target[index] = target[index]
                self._stable_since_ns[index] = -1

            target_stable = (
                abs(float(wrap_to_pi(target[index] - self._stable_target[index])))
                <= self._tuning.target_stable_threshold_rad
            )
            velocity_stable = (
                abs(float(velocity[index]))
                <= self._tuning.velocity_threshold_rad_s
            )
            if not target_stable:
                self._stable_target[index] = target[index]
                self._stable_since_ns[index] = -1
            if velocity_stable:
                if self._stable_since_ns[index] < 0:
                    self._stable_since_ns[index] = now_ns
            else:
                self._stable_since_ns[index] = -1
            active = (
                not reset
                and velocity_stable
                and self._stable_since_ns[index] >= 0
                and (now_ns - self._stable_since_ns[index]) / 1e9
                >= self._tuning.stable_time_s
            )
            self._active[index] = active
            error = float(wrap_to_pi(target[index] - measured[index]))
            estimate_candidate = self._estimate[index]
            if active and abs(error) > self._tuning.adaptive_deadband_rad:
                estimate_candidate += (
                    self._tuning.adaptation_rate_per_s * dt * error
                )
            available_lower = max(
                -self._tuning.maximum_adaptive_offset_rad,
                self._lower[index] - target[index] - gravity_offset[index],
            )
            available_upper = min(
                self._tuning.maximum_adaptive_offset_rad,
                self._upper[index] - target[index] - gravity_offset[index],
            )
            bounded_estimate = float(
                np.clip(estimate_candidate, available_lower, available_upper)
            )
            self._saturated[index] = bool(
                active
                and abs(error) > self._tuning.adaptive_deadband_rad
                and not math.isclose(estimate_candidate, bounded_estimate)
            )
            self._estimate[index] = bounded_estimate
            # Filter the estimate before exposing it to the servo's own position loop;
            # the estimate may reset immediately, but the commanded offset must not.
            alpha = 1.0 - math.exp(
                -self._tuning.offset_filter_bandwidth_per_s * dt
            )
            self._offset[index] = float(
                np.clip(
                    self._offset[index] + alpha * (self._estimate[index] - self._offset[index]),
                    available_lower,
                    available_upper,
                )
            )
            self._target[index] = target[index]
        return self._offset.copy()


__all__ = ["PositionControlDiagnostics", "PositionControlPipeline"]
