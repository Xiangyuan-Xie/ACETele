"""Low-latency robust position/velocity estimation for quantized servo feedback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class StateEstimatorTuning:
    """Noise, innovation-gate, acceleration, and recovery parameters."""

    acceleration_std_rad_s2: float
    position_std_rad: float
    velocity_std_rad_s: float
    max_acceleration_rad_s2: float = 40.0
    nis_threshold: float = 16.0
    position_gate_rad: Optional[float] = None
    reanchor_gate_rad: Optional[float] = None
    velocity_consistency_rad_s: Optional[float] = None
    minimum_dt_s: float = 0.001
    maximum_dt_s: float = 0.05
    reset_dt_s: float = 0.25
    reanchor_samples: int = 3

    def __post_init__(self) -> None:
        positive = (
            "acceleration_std_rad_s2",
            "position_std_rad",
            "velocity_std_rad_s",
            "max_acceleration_rad_s2",
            "nis_threshold",
            "minimum_dt_s",
            "maximum_dt_s",
            "reset_dt_s",
        )
        if any(
            not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0
            for name in positive
        ):
            raise ValueError("state estimator tuning values must be finite and positive")
        if self.minimum_dt_s > self.maximum_dt_s or self.maximum_dt_s >= self.reset_dt_s:
            raise ValueError("state estimator dt limits are inconsistent")
        for name in (
            "position_gate_rad",
            "reanchor_gate_rad",
            "velocity_consistency_rad_s",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when provided")
        if type(self.reanchor_samples) is not int or self.reanchor_samples < 2:
            raise ValueError("reanchor_samples must be an integer of at least two")


@dataclass(frozen=True)
class StateEstimate:
    """Immutable filtered state and per-channel observation acceptance flags."""

    positions: np.ndarray
    velocities: np.ndarray
    position_accepted: np.ndarray
    velocity_accepted: np.ndarray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            array = np.asarray(getattr(self, name)).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)


class RobustJointStateEstimator:
    """Constant-velocity Kalman filter with Joseph updates and innovation gates.

    The equations follow standard Kalman filtering practice also used by FilterPy and
    Stone Soup; the explicit innovation rejection follows the same design principle as
    robot_localization. This implementation is independent and keeps no third-party
    filter runtime dependency.

    References: filterpy.readthedocs.io, stonesoup.readthedocs.io, and
    github.com/cra-ros-pkg/robot_localization.
    """

    def __init__(self, joint_count: int, tuning: StateEstimatorTuning) -> None:
        if type(joint_count) is not int or joint_count <= 0:
            raise ValueError("joint_count must be a positive integer")
        if not isinstance(tuning, StateEstimatorTuning):
            raise ValueError("tuning must be StateEstimatorTuning")
        self._count = joint_count
        self._tuning = tuning
        self._state = np.zeros((joint_count, 2))
        self._covariance = np.zeros((joint_count, 2, 2))
        self._last_timestamp_s: Optional[float] = None
        self._last_sample_id: Optional[int] = None
        self._last_estimate: Optional[StateEstimate] = None
        self._last_measured_position = np.full(joint_count, np.nan)
        self._last_measured_timestamp = np.full(joint_count, np.nan)
        self._reanchor_candidate = np.full(joint_count, np.nan)
        self._rejection_streak = np.zeros(joint_count, dtype=np.int64)
        self._position_rejections = np.zeros(joint_count, dtype=np.int64)
        self._velocity_rejections = np.zeros(joint_count, dtype=np.int64)
        self._position_innovation = np.full(joint_count, np.nan)
        self._velocity_innovation = np.full(joint_count, np.nan)
        self._position_nis = np.full(joint_count, np.nan)
        self._velocity_nis = np.full(joint_count, np.nan)
        self._position_gate_actual = np.full(joint_count, np.nan)
        self._velocity_gate_actual = np.full(joint_count, np.nan)
        self._velocity_limited = np.zeros(joint_count, dtype=bool)

    def update(
        self,
        positions: Sequence[float] | np.ndarray,
        velocities: Sequence[float] | np.ndarray,
        *,
        timestamp_s: float,
        sample_id: Optional[int] = None,
    ) -> StateEstimate:
        """Fuse one timestamped position/velocity sample into a coherent estimate."""

        measured_position = self._vector(positions, "positions")
        measured_velocity = self._vector(velocities, "velocities")
        if not math.isfinite(timestamp_s):
            raise ValueError("state estimator timestamp must be finite")
        if sample_id is not None and type(sample_id) is not int:
            raise ValueError("state estimator sample_id must be an integer or None")
        if (
            sample_id is not None
            and sample_id == self._last_sample_id
            and self._last_estimate is not None
        ):
            # Repeated transport snapshots are not new measurements and therefore
            # must not shrink covariance or create an artificial zero-velocity sample.
            return self._last_estimate
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            if self._last_estimate is None:
                raise RuntimeError("state estimator has no prior estimate")
            self._last_sample_id = sample_id
            return self._last_estimate
        elapsed = (
            None
            if self._last_timestamp_s is None
            else timestamp_s - self._last_timestamp_s
        )
        # A long observation gap invalidates the constant-velocity prediction. Rebase
        # from position with zero velocity instead of integrating stale dynamics.
        if elapsed is None or elapsed > self._tuning.reset_dt_s:
            estimate = self._initialize(measured_position, timestamp_s)
        else:
            dt = float(
                np.clip(
                    elapsed,
                    self._tuning.minimum_dt_s,
                    self._tuning.maximum_dt_s,
                )
            )
            estimate = self._filter(
                measured_position,
                measured_velocity,
                timestamp_s,
                dt,
            )
        self._last_timestamp_s = timestamp_s
        self._last_sample_id = sample_id
        self._last_estimate = estimate
        return estimate

    def diagnostics(self) -> dict[str, np.ndarray]:
        """Return independent innovation, gate, rejection, and covariance arrays."""

        accepted_position = (
            np.zeros(self._count, dtype=bool)
            if self._last_estimate is None
            else self._last_estimate.position_accepted
        )
        accepted_velocity = (
            np.zeros(self._count, dtype=bool)
            if self._last_estimate is None
            else self._last_estimate.velocity_accepted
        )
        return {
            "estimated_positions": self._state[:, 0].copy(),
            "estimated_velocities": self._state[:, 1].copy(),
            "position_accepted": accepted_position.copy(),
            "velocity_accepted": accepted_velocity.copy(),
            "position_rejection_count": self._position_rejections.copy(),
            "velocity_rejection_count": self._velocity_rejections.copy(),
            "position_innovation": self._position_innovation.copy(),
            "velocity_innovation": self._velocity_innovation.copy(),
            "position_nis": self._position_nis.copy(),
            "velocity_nis": self._velocity_nis.copy(),
            "position_gate": self._position_gate_actual.copy(),
            "velocity_consistency_gate": self._velocity_gate_actual.copy(),
            "velocity_limited": self._velocity_limited.copy(),
            "consecutive_position_rejections": self._rejection_streak.copy(),
            "covariance_diagonal": np.diagonal(
                self._covariance,
                axis1=1,
                axis2=2,
            ).copy(),
        }

    def _initialize(self, measured: np.ndarray, timestamp_s: float) -> StateEstimate:
        """Anchor finite positions and reset velocity/covariance after a long gap."""

        accepted = np.isfinite(measured)
        self._position_rejections += ~accepted
        self._state[:, 0] = np.where(accepted, measured, self._state[:, 0])
        self._state[:, 1] = 0.0
        self._covariance[:] = np.diag(
            [
                self._tuning.position_std_rad**2,
                self._tuning.velocity_std_rad_s**2,
            ]
        )
        self._last_measured_position = np.where(accepted, measured, np.nan)
        self._last_measured_timestamp = np.where(accepted, timestamp_s, np.nan)
        self._reanchor_candidate.fill(np.nan)
        self._rejection_streak.fill(0)
        self._position_innovation = np.where(accepted, 0.0, np.nan)
        self._position_nis = np.where(accepted, 0.0, np.inf)
        self._position_gate_actual.fill(self._position_gate())
        self._velocity_gate_actual.fill(np.nan)
        self._velocity_innovation.fill(np.nan)
        self._velocity_nis.fill(np.nan)
        self._velocity_limited.fill(False)
        return StateEstimate(
            self._state[:, 0],
            self._state[:, 1],
            accepted,
            np.zeros(self._count, dtype=bool),
        )

    def _filter(
        self,
        measured_position: np.ndarray,
        measured_velocity: np.ndarray,
        timestamp_s: float,
        dt: float,
    ) -> StateEstimate:
        """Run predict, gated scalar updates, and physical velocity limiting."""

        # A white-acceleration model yields the smallest state needed for low-latency
        # position/velocity fusion while retaining a physically meaningful process Q.
        transition = np.array(((1.0, dt), (0.0, 1.0)))
        acceleration_gain = np.array((0.5 * dt * dt, dt))
        process_noise = self._tuning.acceleration_std_rad_s2**2 * np.outer(
            acceleration_gain,
            acceleration_gain,
        )
        position_accepted = np.zeros(self._count, dtype=bool)
        velocity_accepted = np.zeros(self._count, dtype=bool)
        self._velocity_limited.fill(False)
        for index in range(self._count):
            # Predict each joint independently; bus joints have no measured covariance
            # coupling, so a block-diagonal filter avoids unnecessary matrix work.
            state = transition @ self._state[index]
            covariance = (
                transition @ self._covariance[index] @ transition.T + process_noise
            )
            predicted_velocity = float(state[1])
            previous_position = self._last_measured_position[index]
            previous_timestamp = self._last_measured_timestamp[index]
            # Compare in a continuous angle neighborhood. Wrapping only at the public
            # boundary avoids a false 2*pi innovation near the branch cut.
            position = self._unwrap(measured_position[index], state[0])
            innovation = position - state[0]
            variance = covariance[0, 0] + self._tuning.position_std_rad**2
            nis = innovation * innovation / variance if math.isfinite(innovation) else math.inf
            physical_gate = (
                self._position_gate()
                + 0.5 * self._tuning.max_acceleration_rad_s2 * dt * dt
            )
            self._position_gate_actual[index] = physical_gate
            # NIS rejects statistically inconsistent samples; the physical gate also
            # bounds motion possible under the configured acceleration assumption.
            accepted = (
                math.isfinite(position)
                and nis <= self._tuning.nis_threshold
                and abs(innovation) <= physical_gate
            )
            self._position_innovation[index] = innovation
            self._position_nis[index] = nis
            if accepted:
                state, covariance = self._joseph_update(
                    state,
                    covariance,
                    position,
                    np.array((1.0, 0.0)),
                    self._tuning.position_std_rad**2,
                )
                position_accepted[index] = True
                self._last_measured_position[index] = position
                self._last_measured_timestamp[index] = timestamp_s
                self._rejection_streak[index] = 0
                self._reanchor_candidate[index] = np.nan
            else:
                self._position_rejections[index] += 1
                # Coherent consecutive rejections indicate a real state transition,
                # while an isolated rejection remains classified as a sensor spike.
                if self._consider_reanchor(index, position):
                    state[0] = position
                    state[1] = 0.0
                    covariance = np.diag(
                        [
                            self._tuning.position_std_rad**2,
                            self._tuning.velocity_std_rad_s**2,
                        ]
                    )
                    position_accepted[index] = True
                    self._last_measured_position[index] = position
                    self._last_measured_timestamp[index] = timestamp_s

            # FEETECH velocity is quantized and less trustworthy than position. Accept
            # it only when both NIS and finite-difference consistency agree.
            velocity = measured_velocity[index]
            velocity_innovation = velocity - state[1]
            velocity_variance = covariance[1, 1] + self._tuning.velocity_std_rad_s**2
            velocity_nis = (
                velocity_innovation * velocity_innovation / velocity_variance
                if math.isfinite(velocity_innovation)
                else math.inf
            )
            consistent, consistency_gate = self._velocity_consistent(
                velocity,
                position,
                timestamp_s,
                previous_position,
                previous_timestamp,
            )
            self._velocity_gate_actual[index] = consistency_gate
            if (
                math.isfinite(velocity)
                and velocity_nis <= self._tuning.nis_threshold
                and consistent
            ):
                state, covariance = self._joseph_update(
                    state,
                    covariance,
                    velocity,
                    np.array((0.0, 1.0)),
                    self._tuning.velocity_std_rad_s**2,
                )
                velocity_accepted[index] = True
            else:
                self._velocity_rejections[index] += 1
            self._velocity_innovation[index] = velocity_innovation
            self._velocity_nis[index] = velocity_nis

            # Position correction can inject velocity through cross-covariance. Clamp
            # that state change even when both scalar observations passed their gates.
            maximum_change = self._tuning.max_acceleration_rad_s2 * dt
            limited_velocity = float(
                np.clip(
                    state[1],
                    predicted_velocity - maximum_change,
                    predicted_velocity + maximum_change,
                )
            )
            self._velocity_limited[index] = not math.isclose(limited_velocity, state[1])
            state[1] = limited_velocity
            self._state[index] = state
            self._covariance[index] = self._sanitize_covariance(covariance)
        return StateEstimate(
            self._state[:, 0],
            self._state[:, 1],
            position_accepted,
            velocity_accepted,
        )

    def _consider_reanchor(self, index: int, measurement: float) -> bool:
        """Distinguish a persistent position step from an isolated encoder spike."""

        if not math.isfinite(measurement):
            self._rejection_streak[index] = 0
            self._reanchor_candidate[index] = np.nan
            return False
        candidate = self._reanchor_candidate[index]
        if math.isfinite(candidate) and abs(measurement - candidate) <= self._reanchor_gate():
            self._rejection_streak[index] += 1
        else:
            self._reanchor_candidate[index] = measurement
            self._rejection_streak[index] = 1
        if self._rejection_streak[index] < self._tuning.reanchor_samples:
            return False
        self._rejection_streak[index] = 0
        self._reanchor_candidate[index] = np.nan
        return True

    def _velocity_consistent(
        self,
        velocity: float,
        position: float,
        timestamp_s: float,
        previous_position: float,
        previous_timestamp: float,
    ) -> tuple[bool, float]:
        """Compare reported velocity with displacement over the measurement interval."""

        if not all(math.isfinite(value) for value in (velocity, position, previous_position, previous_timestamp)):
            return False, math.nan
        elapsed = timestamp_s - previous_timestamp
        if elapsed <= 0.0:
            return False, math.nan
        derived = (position - previous_position) / elapsed
        tolerance = max(
            self._velocity_gate(),
            4.0 * self._tuning.position_std_rad / elapsed,
        )
        if abs(velocity - derived) > tolerance:
            return False, tolerance
        accepted = not (
            abs(velocity) > tolerance
            and abs(derived) > tolerance
            and math.copysign(1.0, velocity) != math.copysign(1.0, derived)
        )
        return accepted, tolerance

    @staticmethod
    def _joseph_update(
        state: np.ndarray,
        covariance: np.ndarray,
        measurement: float,
        observation: np.ndarray,
        measurement_variance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply one scalar Kalman observation using Joseph covariance form."""

        innovation = measurement - float(observation @ state)
        variance = float(observation @ covariance @ observation + measurement_variance)
        gain = covariance @ observation / variance
        updated_state = state + gain * innovation
        # Joseph form is intentionally retained because it preserves covariance
        # symmetry and positive semidefiniteness under floating-point roundoff.
        residual = np.eye(2) - np.outer(gain, observation)
        updated_covariance = (
            residual @ covariance @ residual.T
            + np.outer(gain, gain) * measurement_variance
        )
        return updated_state, updated_covariance

    @staticmethod
    def _sanitize_covariance(covariance: np.ndarray) -> np.ndarray:
        """Restore numerical symmetry and clamp roundoff-negative eigenvalues."""

        covariance = 0.5 * (covariance + covariance.T)
        if not np.all(np.isfinite(covariance)):
            raise FloatingPointError("state estimator covariance became non-finite")
        values, vectors = np.linalg.eigh(covariance)
        return vectors @ np.diag(np.maximum(values, 1e-12)) @ vectors.T

    def _position_gate(self) -> float:
        return self._tuning.position_gate_rad or 8.0 * self._tuning.position_std_rad

    def _reanchor_gate(self) -> float:
        return self._tuning.reanchor_gate_rad or 4.0 * self._tuning.position_std_rad

    def _velocity_gate(self) -> float:
        return (
            self._tuning.velocity_consistency_rad_s
            or 6.0 * self._tuning.velocity_std_rad_s
        )

    def _vector(self, values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.shape != (self._count,):
            raise ValueError(f"state estimator {name} must have length {self._count}")
        return result.copy()

    @staticmethod
    def _unwrap(measurement: float, reference: float) -> float:
        if not math.isfinite(measurement):
            return measurement
        return measurement + 2.0 * math.pi * round(
            (reference - measurement) / (2.0 * math.pi)
        )


__all__ = [
    "RobustJointStateEstimator",
    "StateEstimate",
    "StateEstimatorTuning",
]
