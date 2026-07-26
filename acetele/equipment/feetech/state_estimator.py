"""Low-latency robust state estimation for FEETECH joints.

The implementation follows the standard linear Kalman predict/update equations,
uses the Joseph covariance update documented by FilterPy and Stone Soup, and
applies normalized-innovation gating in the style used by robot_localization.
It is intentionally implemented locally to keep the hardware path lightweight.

References:
https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/kalman_filter.html
https://stonesoup.readthedocs.io/en/v1.9/stonesoup.updater.html
https://github.com/cra-ros-pkg/robot_localization/blob/ros2/params/ekf.yaml
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from acetele.equipment.feetech.servo_specs import (
    PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
)
from acetele.utils.angle import unwrap_near

ENCODER_POSITION_RESOLUTION_RAD = np.pi / 2048.0
STATE_ESTIMATOR_ACCELERATION_STD = 6.0
STATE_ESTIMATOR_POSITION_STD = 4.0 * ENCODER_POSITION_RESOLUTION_RAD
STATE_ESTIMATOR_VELOCITY_STD = 12.0 * PROFILE_VELOCITY_UNIT_RAD_PER_SEC
STATE_ESTIMATOR_NIS_THRESHOLD = 16.0
STATE_ESTIMATOR_MAX_ACCELERATION_RAD_PER_SEC2 = 40.0
STATE_ESTIMATOR_POSITION_GATE_COUNTS = 8.0
STATE_ESTIMATOR_POSITION_MOTION_GATE_COUNTS = 4.0
STATE_ESTIMATOR_POSITION_REANCHOR_COUNTS = 4.0
STATE_ESTIMATOR_POSITION_REANCHOR_SAMPLES = 3
STATE_ESTIMATOR_VELOCITY_CONSISTENCY_COUNTS = 6.0
STATE_ESTIMATOR_POSITION_VELOCITY_COUNTS = 4.0
STATE_ESTIMATOR_MIN_DT = 0.001
STATE_ESTIMATOR_MAX_DT = 0.05
STATE_ESTIMATOR_RESET_DT = 0.25
STATE_ESTIMATOR_COVARIANCE_EPSILON = 1e-12


@dataclass(frozen=True)
class FeeTechStateEstimate:
    positions: np.ndarray
    velocities: np.ndarray
    position_accepted: np.ndarray
    velocity_accepted: np.ndarray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = np.asarray(getattr(self, name)).copy()
            object.__setattr__(self, name, value)


class FeeTechStateEstimator:
    """Constant-velocity Kalman estimator with robust observation gates."""

    def __init__(self, joint_count: int):
        if not isinstance(joint_count, int) or isinstance(joint_count, bool) or joint_count <= 0:
            raise ValueError("joint_count must be a positive integer")
        self._joint_count = joint_count
        self._state = np.zeros((joint_count, 2), dtype=float)
        self._covariance = np.zeros((joint_count, 2, 2), dtype=float)
        self._last_input_positions = np.full(joint_count, np.nan)
        self._last_input_velocities = np.full(joint_count, np.nan)
        self._last_measured_positions = np.full(joint_count, np.nan)
        self._last_measured_timestamps = np.full(joint_count, np.nan)
        self._reanchor_positions = np.full(joint_count, np.nan)
        self._consecutive_position_rejections = np.zeros(joint_count, dtype=np.int64)
        self._position_rejection_count = np.zeros(joint_count, dtype=np.int64)
        self._velocity_rejection_count = np.zeros(joint_count, dtype=np.int64)
        self._last_position_innovation = np.full(joint_count, np.nan)
        self._last_velocity_innovation = np.full(joint_count, np.nan)
        self._last_position_nis = np.full(joint_count, np.nan)
        self._last_velocity_nis = np.full(joint_count, np.nan)
        self._last_position_gate = np.zeros(joint_count, dtype=float)
        self._last_velocity_gate = np.zeros(joint_count, dtype=float)
        self._last_velocity_consistency_gate = np.zeros(joint_count, dtype=float)
        self._last_velocity_limited = np.zeros(joint_count, dtype=bool)
        self._last_timestamp: Optional[float] = None
        self._last_sample_id: Optional[int] = None
        self._last_estimate: Optional[FeeTechStateEstimate] = None

    def reset(self) -> None:
        self._state.fill(0.0)
        self._covariance.fill(0.0)
        self._last_input_positions.fill(np.nan)
        self._last_input_velocities.fill(np.nan)
        self._last_measured_positions.fill(np.nan)
        self._last_measured_timestamps.fill(np.nan)
        self._reanchor_positions.fill(np.nan)
        self._consecutive_position_rejections.fill(0)
        self._position_rejection_count.fill(0)
        self._velocity_rejection_count.fill(0)
        self._last_position_innovation.fill(np.nan)
        self._last_velocity_innovation.fill(np.nan)
        self._last_position_nis.fill(np.nan)
        self._last_velocity_nis.fill(np.nan)
        self._last_position_gate.fill(0.0)
        self._last_velocity_gate.fill(0.0)
        self._last_velocity_consistency_gate.fill(0.0)
        self._last_velocity_limited.fill(False)
        self._last_timestamp = None
        self._last_sample_id = None
        self._last_estimate = None

    def update(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        *,
        timestamp: float,
        sample_id: Optional[int] = None,
    ) -> FeeTechStateEstimate:
        measured_positions = self._measurement_array("positions", positions)
        measured_velocities = self._measurement_array("velocities", velocities)
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("state estimator timestamp must be finite")
        if sample_id is not None and (
            not isinstance(sample_id, (int, np.integer)) or isinstance(sample_id, (bool, np.bool_))
        ):
            raise ValueError("state estimator sample_id must be an integer or None")
        normalized_sample_id = None if sample_id is None else int(sample_id)
        if (
            normalized_sample_id is not None
            and normalized_sample_id == self._last_sample_id
            and self._last_estimate is not None
        ):
            return self._copy_estimate(self._last_estimate)

        self._last_input_positions = measured_positions.copy()
        self._last_input_velocities = measured_velocities.copy()

        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            estimate = self._freeze_out_of_order_sample(normalized_sample_id)
            return self._copy_estimate(estimate)

        elapsed = None if self._last_timestamp is None else timestamp - self._last_timestamp
        if elapsed is None or elapsed > STATE_ESTIMATOR_RESET_DT:
            estimate = self._initialize(measured_positions, measured_velocities, timestamp)
        else:
            estimate = self._update_filter(
                measured_positions,
                measured_velocities,
                dt=float(np.clip(elapsed, STATE_ESTIMATOR_MIN_DT, STATE_ESTIMATOR_MAX_DT)),
                timestamp=timestamp,
            )

        self._last_timestamp = timestamp
        self._last_sample_id = normalized_sample_id
        self._last_estimate = estimate
        return self._copy_estimate(estimate)

    def get_diagnostics(self) -> dict[str, np.ndarray]:
        if self._last_estimate is None:
            position_accepted = np.zeros(self._joint_count, dtype=bool)
            velocity_accepted = np.zeros(self._joint_count, dtype=bool)
        else:
            position_accepted = self._last_estimate.position_accepted.copy()
            velocity_accepted = self._last_estimate.velocity_accepted.copy()
        return {
            "measured_positions": self._last_input_positions.copy(),
            "measured_velocities": self._last_input_velocities.copy(),
            "estimated_positions": self._state[:, 0].copy(),
            "estimated_velocities": self._state[:, 1].copy(),
            "position_accepted": position_accepted,
            "velocity_accepted": velocity_accepted,
            "position_rejection_count": self._position_rejection_count.copy(),
            "velocity_rejection_count": self._velocity_rejection_count.copy(),
            "position_innovation": self._last_position_innovation.copy(),
            "velocity_innovation": self._last_velocity_innovation.copy(),
            "position_nis": self._last_position_nis.copy(),
            "velocity_nis": self._last_velocity_nis.copy(),
            "position_gate": self._last_position_gate.copy(),
            "velocity_gate": self._last_velocity_gate.copy(),
            "velocity_consistency_gate": self._last_velocity_consistency_gate.copy(),
            "velocity_limited": self._last_velocity_limited.copy(),
            "consecutive_position_rejections": self._consecutive_position_rejections.copy(),
            "covariance_diagonal": np.diagonal(self._covariance, axis1=1, axis2=2).copy(),
        }

    def _initialize(
        self,
        measured_positions: np.ndarray,
        measured_velocities: np.ndarray,
        timestamp: float,
    ) -> FeeTechStateEstimate:
        position_accepted = np.isfinite(measured_positions)
        velocity_accepted = np.zeros(self._joint_count, dtype=bool)
        fallback_positions = (
            self._state[:, 0].copy()
            if self._last_timestamp is not None
            else np.zeros(self._joint_count, dtype=float)
        )
        self._position_rejection_count += ~position_accepted
        self._velocity_rejection_count += ~np.isfinite(measured_velocities)
        self._state[:, 0] = np.where(
            position_accepted,
            measured_positions,
            fallback_positions,
        )
        self._state[:, 1] = 0.0
        self._reset_covariance()
        self._last_measured_positions = np.where(position_accepted, measured_positions, np.nan)
        self._last_measured_timestamps = np.where(position_accepted, timestamp, np.nan)
        self._reanchor_positions.fill(np.nan)
        self._consecutive_position_rejections.fill(0)
        self._last_position_innovation = np.where(position_accepted, 0.0, np.nan)
        self._last_velocity_innovation = measured_velocities.copy()
        self._last_position_nis = np.where(position_accepted, 0.0, np.inf)
        self._last_velocity_nis.fill(np.nan)
        self._last_position_gate.fill(self._position_gate(STATE_ESTIMATOR_MIN_DT))
        self._last_velocity_gate.fill(0.0)
        self._last_velocity_consistency_gate.fill(0.0)
        self._last_velocity_limited.fill(False)
        return FeeTechStateEstimate(
            positions=self._state[:, 0],
            velocities=self._state[:, 1],
            position_accepted=position_accepted,
            velocity_accepted=velocity_accepted,
        )

    def _update_filter(
        self,
        measured_positions: np.ndarray,
        measured_velocities: np.ndarray,
        dt: float,
        timestamp: float,
    ) -> FeeTechStateEstimate:
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        acceleration_gain = np.array([0.5 * dt * dt, dt])
        process_noise = STATE_ESTIMATOR_ACCELERATION_STD**2 * np.outer(
            acceleration_gain,
            acceleration_gain,
        )
        position_accepted = np.zeros(self._joint_count, dtype=bool)
        velocity_accepted = np.zeros(self._joint_count, dtype=bool)
        velocity_limited = np.zeros(self._joint_count, dtype=bool)

        for index in range(self._joint_count):
            state = transition @ self._state[index]
            covariance = transition @ self._covariance[index] @ transition.T + process_noise
            predicted_velocity = float(state[1])
            previous_position = self._last_measured_positions[index]
            previous_position_timestamp = self._last_measured_timestamps[index]

            position_measurement = self._unwrap_measurement(measured_positions[index], state[0])
            position_innovation = position_measurement - state[0]
            position_variance = STATE_ESTIMATOR_POSITION_STD**2
            maneuver_position_std = (
                0.5 * STATE_ESTIMATOR_MAX_ACCELERATION_RAD_PER_SEC2 * dt * dt
            )
            position_innovation_variance = (
                covariance[0, 0]
                + position_variance
                + maneuver_position_std * maneuver_position_std
            )
            position_nis = self._normalized_innovation_squared(
                position_innovation,
                position_innovation_variance,
            )
            position_gate = self._position_gate(dt)
            measured_velocity_for_gate = float(measured_velocities[index])
            if np.isfinite(measured_velocity_for_gate):
                position_gate += min(
                    STATE_ESTIMATOR_POSITION_MOTION_GATE_COUNTS
                    * ENCODER_POSITION_RESOLUTION_RAD,
                    min(
                        STATE_ESTIMATOR_MAX_ACCELERATION_RAD_PER_SEC2 * dt,
                        abs(measured_velocity_for_gate - predicted_velocity),
                    )
                    * dt,
                )
            position_valid = (
                np.isfinite(position_measurement)
                and abs(position_innovation) <= position_gate
                and position_nis <= STATE_ESTIMATOR_NIS_THRESHOLD
            )
            reanchored = False
            if position_valid:
                state, covariance = self._correct(
                    state,
                    covariance,
                    measurement_index=0,
                    innovation=position_innovation,
                    measurement_variance=position_variance,
                )
                position_accepted[index] = True
                self._clear_reanchor_candidate(index)
            else:
                self._position_rejection_count[index] += 1
                reanchored = self._update_reanchor_candidate(index, position_measurement)
                if reanchored:
                    state[:] = (self._reanchor_positions[index], 0.0)
                    covariance = self._initial_covariance()
                    position_accepted[index] = True
                    self._clear_reanchor_candidate(index)

            velocity_measurement = float(measured_velocities[index])
            velocity_innovation = velocity_measurement - state[1]
            velocity_variance = STATE_ESTIMATOR_VELOCITY_STD**2
            maneuver_velocity_std = STATE_ESTIMATOR_MAX_ACCELERATION_RAD_PER_SEC2 * dt
            velocity_innovation_variance = (
                covariance[1, 1]
                + velocity_variance
                + maneuver_velocity_std * maneuver_velocity_std
            )
            velocity_nis = self._normalized_innovation_squared(
                velocity_innovation,
                velocity_innovation_variance,
            )
            velocity_gate = float(
                np.sqrt(STATE_ESTIMATOR_NIS_THRESHOLD * velocity_innovation_variance)
            )
            velocity_consistency_gate = 0.0
            velocity_consistent = False
            measurement_dt = timestamp - previous_position_timestamp
            if (
                not reanchored
                and np.isfinite(position_measurement)
                and np.isfinite(previous_position)
                and measurement_dt > 0.0
            ):
                position_velocity = (position_measurement - previous_position) / measurement_dt
                velocity_consistency_gate = max(
                    STATE_ESTIMATOR_VELOCITY_CONSISTENCY_COUNTS
                    * PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
                    STATE_ESTIMATOR_POSITION_VELOCITY_COUNTS
                    * ENCODER_POSITION_RESOLUTION_RAD
                    / measurement_dt,
                )
                velocity_consistent = (
                    abs(velocity_measurement - position_velocity)
                    <= velocity_consistency_gate
                )
            velocity_valid = (
                np.isfinite(velocity_measurement)
                and not reanchored
                and velocity_nis <= STATE_ESTIMATOR_NIS_THRESHOLD
                and velocity_consistent
            )
            if velocity_valid:
                state, covariance = self._correct(
                    state,
                    covariance,
                    measurement_index=1,
                    innovation=velocity_measurement - state[1],
                    measurement_variance=velocity_variance,
                )
                velocity_accepted[index] = True
            else:
                self._velocity_rejection_count[index] += 1

            velocity_delta = STATE_ESTIMATOR_MAX_ACCELERATION_RAD_PER_SEC2 * dt
            limited_velocity = float(
                np.clip(
                    state[1],
                    predicted_velocity - velocity_delta,
                    predicted_velocity + velocity_delta,
                )
            )
            velocity_limited[index] = not np.isclose(limited_velocity, state[1])
            state[1] = limited_velocity
            covariance = self._stabilize_covariance(covariance)

            self._state[index] = state
            self._covariance[index] = covariance
            if np.isfinite(position_measurement):
                self._last_measured_positions[index] = position_measurement
                self._last_measured_timestamps[index] = timestamp
            self._last_position_innovation[index] = position_innovation
            self._last_velocity_innovation[index] = velocity_innovation
            self._last_position_nis[index] = position_nis
            self._last_velocity_nis[index] = velocity_nis
            self._last_position_gate[index] = position_gate
            self._last_velocity_gate[index] = velocity_gate
            self._last_velocity_consistency_gate[index] = velocity_consistency_gate

        self._last_velocity_limited = velocity_limited
        return FeeTechStateEstimate(
            positions=self._state[:, 0],
            velocities=self._state[:, 1],
            position_accepted=position_accepted,
            velocity_accepted=velocity_accepted,
        )

    def _freeze_out_of_order_sample(self, sample_id: Optional[int]) -> FeeTechStateEstimate:
        self._position_rejection_count += 1
        self._velocity_rejection_count += 1
        self._last_position_innovation.fill(np.nan)
        self._last_velocity_innovation.fill(np.nan)
        self._last_position_nis.fill(np.inf)
        self._last_velocity_nis.fill(np.inf)
        self._last_velocity_limited.fill(False)
        estimate = FeeTechStateEstimate(
            positions=self._state[:, 0],
            velocities=self._state[:, 1],
            position_accepted=np.zeros(self._joint_count, dtype=bool),
            velocity_accepted=np.zeros(self._joint_count, dtype=bool),
        )
        self._last_sample_id = sample_id
        self._last_estimate = estimate
        return estimate

    def _update_reanchor_candidate(self, index: int, measurement: float) -> bool:
        if not np.isfinite(measurement):
            self._clear_reanchor_candidate(index)
            return False
        candidate = self._reanchor_positions[index]
        tolerance = (
            STATE_ESTIMATOR_POSITION_REANCHOR_COUNTS * ENCODER_POSITION_RESOLUTION_RAD
        )
        if not np.isfinite(candidate) or abs(measurement - candidate) > tolerance:
            self._reanchor_positions[index] = measurement
            self._consecutive_position_rejections[index] = 1
            return False
        count = int(self._consecutive_position_rejections[index]) + 1
        self._reanchor_positions[index] = candidate + (measurement - candidate) / count
        self._consecutive_position_rejections[index] = count
        return count >= STATE_ESTIMATOR_POSITION_REANCHOR_SAMPLES

    def _clear_reanchor_candidate(self, index: int) -> None:
        self._reanchor_positions[index] = np.nan
        self._consecutive_position_rejections[index] = 0

    @staticmethod
    def _position_gate(dt: float) -> float:
        return (
            STATE_ESTIMATOR_POSITION_GATE_COUNTS * ENCODER_POSITION_RESOLUTION_RAD
            + 0.5 * STATE_ESTIMATOR_MAX_ACCELERATION_RAD_PER_SEC2 * dt * dt
        )

    @staticmethod
    def _unwrap_measurement(measurement: float, reference: float) -> float:
        if not np.isfinite(measurement):
            return float("nan")
        return float(
            unwrap_near(
                np.asarray([measurement]),
                np.asarray([reference]),
            )[0]
        )

    @staticmethod
    def _normalized_innovation_squared(
        innovation: float,
        innovation_variance: float,
    ) -> float:
        if not np.isfinite(innovation) or not np.isfinite(innovation_variance):
            return float("inf")
        if innovation_variance <= 0.0:
            return float("inf")
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(innovation * innovation / innovation_variance)
        return result if np.isfinite(result) else float("inf")

    @staticmethod
    def _correct(
        state: np.ndarray,
        covariance: np.ndarray,
        *,
        measurement_index: int,
        innovation: float,
        measurement_variance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        observation = np.zeros(2, dtype=float)
        observation[measurement_index] = 1.0
        innovation_variance = float(
            observation @ covariance @ observation + measurement_variance
        )
        gain = covariance @ observation / innovation_variance
        corrected_state = state + gain * innovation
        residual = np.eye(2) - np.outer(gain, observation)
        corrected_covariance = (
            residual @ covariance @ residual.T
            + np.outer(gain, gain) * measurement_variance
        )
        return corrected_state, corrected_covariance

    @staticmethod
    def _initial_covariance() -> np.ndarray:
        return np.diag(
            [
                STATE_ESTIMATOR_POSITION_STD**2,
                STATE_ESTIMATOR_VELOCITY_STD**2,
            ]
        )

    def _reset_covariance(self) -> None:
        self._covariance[:] = self._initial_covariance()

    @staticmethod
    def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (covariance + covariance.T)
        if not np.all(np.isfinite(symmetric)):
            return FeeTechStateEstimator._initial_covariance()
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric)))
        if minimum_eigenvalue < STATE_ESTIMATOR_COVARIANCE_EPSILON:
            symmetric += np.eye(2) * (
                STATE_ESTIMATOR_COVARIANCE_EPSILON - minimum_eigenvalue
            )
        return symmetric

    def _measurement_array(self, name: str, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape != (self._joint_count,):
            raise ValueError(f"state estimator {name} must contain one value per joint")
        return array.copy()

    @staticmethod
    def _copy_estimate(estimate: FeeTechStateEstimate) -> FeeTechStateEstimate:
        return FeeTechStateEstimate(
            positions=estimate.positions,
            velocities=estimate.velocities,
            position_accepted=estimate.position_accepted,
            velocity_accepted=estimate.velocity_accepted,
        )


def read_feetech_state_sample(driver: Any, fallback_sequence: int):
    sample_reader = getattr(driver, "get_state_sample", None)
    if callable(sample_reader):
        return sample_reader(), fallback_sequence

    from acetele.equipment.feetech.feetech_driver import FeeTechStateSample

    position, velocity, current = driver.get_state()
    next_sequence = fallback_sequence + 1
    return (
        FeeTechStateSample(
            position=position,
            velocity=velocity,
            current=current,
            timestamp=time.monotonic(),
            sequence=next_sequence,
        ),
        next_sequence,
    )


__all__ = [
    "FeeTechStateEstimate",
    "FeeTechStateEstimator",
    "read_feetech_state_sample",
]
