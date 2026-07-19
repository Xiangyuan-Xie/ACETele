from __future__ import annotations

import time
from threading import Event, Thread
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

from acetele.config.robot_config import ArmConfig
from acetele.equipment.feetech.feetech_driver import (
    FEETECH_SIGNED_15_BIT_MAX,
    FeeTechDriver,
    TorqueEnable,
)
from acetele.equipment.feetech.servo_specs import (
    HLS_PROFILE_DEFAULTS_BY_SERVO,
    KT_MAPPING,
    NO_LOAD_CURRENT,
    PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2,
    PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
    validate_feetech_servo_models,
)
from acetele.equipment.joint_device import JointDeviceState, _normalize_profile_value
from acetele.utils.angle import unwrap_near, wrap_to_pi
from acetele.utils.joint import normalize_joint_ids

ADAPTIVE_COMPENSATION_DEADBAND = 0.02
ADAPTIVE_COMPENSATION_ADAPTATION_RATE = 8.0
ADAPTIVE_COMPENSATION_FILTER_BANDWIDTH = 4.0
ADAPTIVE_COMPENSATION_MAX_OFFSET = 0.10
ADAPTIVE_COMPENSATION_STABLE_TIME = 0.20
ADAPTIVE_COMPENSATION_TARGET_STABLE_THRESHOLD = 0.008
ADAPTIVE_COMPENSATION_TARGET_RESET_THRESHOLD = 0.05
ADAPTIVE_COMPENSATION_TARGET_DIRECTION_THRESHOLD = 0.002
ADAPTIVE_COMPENSATION_VELOCITY_THRESHOLD = 0.05
ADAPTIVE_COMPENSATION_MIN_DT = 0.001
ADAPTIVE_COMPENSATION_MAX_DT = 0.05
FEETECH_PROFILE_VELOCITY_MAX = 32767
FEETECH_PROFILE_ACCELERATION_MAX = 255
FEETECH_PROFILE_CURRENT_MAX = 32767


class FeeTechArm:
    def __init__(
        self,
        config: ArmConfig,
        driver: Optional[FeeTechDriver] = None,
        pin_model: pin.Model = None,
        position_limits: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    ):
        self._ids = np.asarray(config.joint_ids, dtype=int)
        self._joint_names = config.joint_names
        self._dof = len(self._ids)

        self._signs = np.asarray(config.joint_signs, dtype=float)
        self._enable_gravity_compensation = config.enable_gravity_compensation
        self._control_period = config.control_period
        if not config.servo_models:
            raise ValueError("servo_models must be specified")
        self._servo_models = np.asarray(config.servo_models)
        validate_feetech_servo_models(config.servo_models, context="arm")
        self._profile_acceleration_defaults = np.array(
            [HLS_PROFILE_DEFAULTS_BY_SERVO[servo_model]["acceleration"] for servo_model in self._servo_models]
        )
        self._profile_current_defaults = np.array(
            [HLS_PROFILE_DEFAULTS_BY_SERVO[servo_model]["current"] for servo_model in self._servo_models]
        )
        self._profile_velocity_defaults = np.array(
            [HLS_PROFILE_DEFAULTS_BY_SERVO[servo_model]["velocity"] for servo_model in self._servo_models]
        )

        self._home_poses = np.asarray(config.home_poses, dtype=float)

        self._torque_current_mapping = np.array([KT_MAPPING[servo] * 1000.0 for servo in self._servo_models])
        self._no_load_current = np.array([NO_LOAD_CURRENT[servo] for servo in self._servo_models])
        self._init_position_limits(position_limits)
        self._init_adaptive_compensation(config)

        if pin_model is not None:
            self._pin_model = pin_model
            self._pin_data = self._pin_model.createData()
        else:
            self._pin_model = None
            self._pin_data = None

        if self._enable_gravity_compensation:
            if self._pin_model is None:
                raise ValueError("Pinocchio model is None, please provide a valid model.")

            self._null_space_joint_target = self._home_poses
            self._null_space_kp = 0.1
            self._null_space_kd = 0.01

            self._gravity_comp_modifier = 1.2

            self._stiction_dither_flag = np.ones(self._dof, dtype=bool)
            self._stiction_comp_enable_velocity = 0.9
            self._stiction_comp_gain = 0.6

            self._stop_flag = Event()
            self._control_thread = None

        if driver is None:
            port = config.port
            if port is None:
                raise ValueError("physical arm requires a serial port")
            self._driver = FeeTechDriver(self._ids, port)
        else:
            self._driver = driver
        self._close_driver = driver is None

    @property
    def ids(self) -> np.ndarray:
        return self._ids.copy()

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    def _resolve_indices(
        self,
        ids: Optional[Sequence[int]],
        value_count: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ids_array = (
            self._ids.copy()
            if ids is None
            else np.asarray(normalize_joint_ids(ids, field_name="arm joint ids"), dtype=int)
        )
        if value_count is not None and len(ids_array) != value_count:
            raise ValueError("arm joint ids must match the command length")
        if len(set(ids_array.tolist())) != len(ids_array):
            raise ValueError("arm joint ids must be unique")
        unknown = ids_array[~np.isin(ids_array, self._ids)]
        if len(unknown):
            raise ValueError(f"unknown arm joint id {int(unknown[0])}")
        index_by_id = {int(joint_id): index for index, joint_id in enumerate(self._ids)}
        indices = np.asarray([index_by_id[int(joint_id)] for joint_id in ids_array], dtype=int)
        return ids_array, indices

    def _init_position_limits(
        self,
        position_limits: Optional[Tuple[Sequence[float], Sequence[float]]],
    ) -> None:
        self._position_limits_available = position_limits is not None
        if position_limits is None:
            self._position_lower_limits = np.full(self._dof, -np.inf)
            self._position_upper_limits = np.full(self._dof, np.inf)
            return

        if len(position_limits) != 2:
            raise ValueError("position_limits must contain lower and upper limits.")
        lower_limits = np.asarray(position_limits[0], dtype=float)
        upper_limits = np.asarray(position_limits[1], dtype=float)
        if lower_limits.shape != (self._dof,) or upper_limits.shape != (self._dof,):
            raise ValueError("position_limits must have the same length as joint_ids.")
        if not np.all(np.isfinite(lower_limits)) or not np.all(np.isfinite(upper_limits)):
            raise ValueError("position_limits must be finite.")
        if np.any(lower_limits >= upper_limits):
            raise ValueError("position_limits lower values must be less than upper values.")
        self._position_lower_limits = lower_limits.copy()
        self._position_upper_limits = upper_limits.copy()

    def _init_adaptive_compensation(self, config: ArmConfig) -> None:
        self._adaptive_compensation_enabled = config.enable_adaptive_compensation
        if self._adaptive_compensation_enabled and not self._position_limits_available:
            raise ValueError("position_limits are required when adaptive compensation is enabled.")

        self._adaptive_compensation_estimate = np.zeros(self._dof)
        self._adaptive_compensation_offset = np.zeros(self._dof)
        self._adaptive_compensation_target = np.full(self._dof, np.nan)
        self._adaptive_compensation_stable_target = np.full(self._dof, np.nan)
        self._adaptive_compensation_learning_target = np.full(self._dof, np.nan)
        self._adaptive_compensation_compensated_target = np.full(self._dof, np.nan)
        self._adaptive_compensation_stable_since = np.full(self._dof, np.nan)
        self._adaptive_compensation_motion_direction = np.zeros(self._dof)
        self._adaptive_compensation_last_update = np.full(self._dof, np.nan)
        self._adaptive_compensation_last_error = np.zeros(self._dof)
        self._adaptive_compensation_last_dt = np.zeros(self._dof)
        self._adaptive_compensation_active = np.zeros(self._dof, dtype=bool)
        self._adaptive_compensation_last_limited = np.zeros(self._dof, dtype=bool)
        self._adaptive_compensation_command_limited = np.zeros(self._dof, dtype=bool)
        self._adaptive_compensation_last_reset = np.zeros(self._dof, dtype=bool)

    def get_adaptive_compensation_state(self) -> Dict[str, Any]:
        return {
            "enabled": self._adaptive_compensation_enabled,
            "active": self._adaptive_compensation_active.copy(),
            "estimate_rad": self._adaptive_compensation_estimate.copy(),
            "offset_rad": self._adaptive_compensation_offset.copy(),
            "target_rad": self._adaptive_compensation_target.copy(),
            "stable_target_rad": self._adaptive_compensation_stable_target.copy(),
            "compensated_target_rad": self._adaptive_compensation_compensated_target.copy(),
            "last_error_rad": self._adaptive_compensation_last_error.copy(),
            "last_dt": self._adaptive_compensation_last_dt.copy(),
            "last_limited": self._adaptive_compensation_last_limited.copy(),
            "command_limited": self._adaptive_compensation_command_limited.copy(),
            "last_reset": self._adaptive_compensation_last_reset.copy(),
        }

    def act(self) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        return self.get_state().act()

    def get_state(self) -> JointDeviceState:
        encoded_pos, encoded_vel, encoded_current = self._driver.get_state()

        raw_positions = np.array([encoded_pos[int(ft_id)] for ft_id in self._ids]) * self._signs * np.pi / 2048.0
        public_positions = wrap_to_pi(raw_positions)

        velocities = np.array([encoded_vel[int(ft_id)] for ft_id in self._ids]) * self._signs * 0.732 * np.pi / 30

        raw_currents = np.array([encoded_current[int(ft_id)] for ft_id in self._ids], dtype=float)
        currents = raw_currents * 6.5
        torques_kgcmf_mag = np.maximum(np.abs(currents) - self._no_load_current, 0.0) / self._torque_current_mapping
        torques_Nm_mag = torques_kgcmf_mag * 0.0981
        torque_sign = np.sign(-raw_currents * self._signs)
        torques_Nm_signed = torques_Nm_mag * torque_sign

        return JointDeviceState(
            public_positions=public_positions,
            raw_positions=raw_positions,
            velocities=velocities,
            motor_torque_magnitude=torques_Nm_mag,
            motor_torque_signed=torques_Nm_signed,
        )

    def set_torque(self, torques: Sequence[float], ids: Optional[Sequence[int]] = None):
        torques_Nm = np.asarray(torques, dtype=float)
        if torques_Nm.ndim != 1 or not np.all(np.isfinite(torques_Nm)):
            raise ValueError("torques must be a finite one-dimensional sequence")
        ids_array, indices = self._resolve_indices(ids, len(torques_Nm))
        torques_kgcmf = torques_Nm / 0.0981
        torque_current_mapping = self._torque_current_mapping[indices]
        no_load_current = self._no_load_current[indices]
        signs = self._signs[indices]
        currents = ((torque_current_mapping * np.abs(torques_kgcmf) + no_load_current) * np.sign(torques_kgcmf)) * signs
        encoded_current_values = np.rint(currents / -6.5)
        if np.any(np.abs(encoded_current_values) > FEETECH_SIGNED_15_BIT_MAX):
            raise ValueError(
                "encoded torque current must be between "
                f"{-FEETECH_SIGNED_15_BIT_MAX} and "
                f"{FEETECH_SIGNED_15_BIT_MAX}"
            )
        encoded_currents = encoded_current_values.astype(int)
        self._driver.set_current(ids_array, encoded_currents)

    def set_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ) -> None:
        (
            positions_array,
            ids_array,
            indices,
            currents_raw,
            velocities_raw,
            accelerations_raw,
        ) = self._prepare_position_command(
            positions,
            ids=ids,
            velocities=velocities,
            accelerations=accelerations,
            torque=torque,
        )

        bounded_positions, target_limited_flags = self._limit_position_targets(
            positions_array,
            indices,
        )
        positions_array = self._apply_adaptive_compensation(
            requested_positions=positions_array,
            bounded_positions=bounded_positions,
            target_limited_flags=target_limited_flags,
            indices=indices,
        )
        encoded_positions = self._encode_positions(positions_array, indices)
        self._driver.set_position(
            ids_array,
            encoded_positions,
            currents_raw=currents_raw,
            velocities_raw=velocities_raw,
            accelerations_raw=accelerations_raw,
        )

    def validate_position_command(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ) -> None:
        positions_array, _, indices, _, _, _ = self._prepare_position_command(
            positions,
            ids=ids,
            velocities=velocities,
            accelerations=accelerations,
            torque=torque,
        )
        bounded_positions, _ = self._limit_position_targets(
            positions_array,
            indices,
        )
        self._encode_positions(bounded_positions, indices)

    def _prepare_position_command(
        self,
        positions: Sequence[float],
        *,
        ids: Optional[Sequence[int]],
        velocities: Optional[Sequence[float] | float],
        accelerations: Optional[Sequence[float] | float],
        torque: Optional[Sequence[float] | float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positions_array = np.asarray(positions, dtype=float)
        if positions_array.ndim != 1 or not np.all(np.isfinite(positions_array)):
            raise ValueError("positions must be a finite one-dimensional sequence")
        positions_array = positions_array.copy()
        ids_array, indices = self._resolve_indices(ids, len(positions_array))
        count = len(ids_array)

        torque_array = self._profile_array("torque", torque, count)
        if torque_array is None:
            current_values = self._profile_current_defaults[indices]
        else:
            torques_kgcmf = torque_array / 0.0981
            current_values = (
                self._torque_current_mapping[indices] * np.abs(torques_kgcmf)
                + self._no_load_current[indices]
            ) / 6.5
        currents_raw = self._encode_profile_values(
            "current",
            current_values,
            FEETECH_PROFILE_CURRENT_MAX,
        )

        velocity_array = self._profile_array("velocities", velocities, count)
        velocity_values = (
            self._profile_velocity_defaults[indices]
            if velocity_array is None
            else velocity_array / PROFILE_VELOCITY_UNIT_RAD_PER_SEC
        )
        velocities_raw = self._encode_profile_values(
            "velocity",
            velocity_values,
            FEETECH_PROFILE_VELOCITY_MAX,
        )

        acceleration_array = self._profile_array("accelerations", accelerations, count)
        acceleration_values = (
            self._profile_acceleration_defaults[indices]
            if acceleration_array is None
            else acceleration_array / PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2
        )
        accelerations_raw = self._encode_profile_values(
            "acceleration",
            acceleration_values,
            FEETECH_PROFILE_ACCELERATION_MAX,
        )
        return (
            positions_array,
            ids_array,
            indices,
            currents_raw,
            velocities_raw,
            accelerations_raw,
        )

    def _encode_positions(
        self,
        positions: Sequence[float],
        indices: np.ndarray,
    ) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            encoded = np.rint(
                np.asarray(positions, dtype=float)
                * self._signs[indices]
                * 2048.0
                / np.pi
            )
        if not np.all(np.isfinite(encoded)) or np.any(
            np.abs(encoded) > FEETECH_SIGNED_15_BIT_MAX
        ):
            raise ValueError(
                "encoded positions must be between "
                f"{-FEETECH_SIGNED_15_BIT_MAX} and "
                f"{FEETECH_SIGNED_15_BIT_MAX}"
            )
        return encoded.astype(int)

    @staticmethod
    def _profile_array(
        name: str,
        value: Optional[Sequence[float] | float],
        count: int,
    ) -> Optional[np.ndarray]:
        normalized = _normalize_profile_value(name, value, count)
        if normalized is None:
            return None
        array = np.asarray(normalized, dtype=float)
        if array.ndim == 0:
            return np.full(count, float(array))
        return array.copy()

    @staticmethod
    def _encode_profile_values(
        name: str,
        values: Sequence[float],
        maximum: int,
    ) -> np.ndarray:
        rounded = np.rint(np.asarray(values, dtype=float))
        if not np.all(np.isfinite(rounded)) or np.any(rounded < 0) or np.any(rounded > maximum):
            raise ValueError(f"encoded {name} must be between 0 and {maximum}")
        return rounded.astype(int)

    def _apply_adaptive_compensation(
        self,
        requested_positions: np.ndarray,
        bounded_positions: np.ndarray,
        target_limited_flags: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        if not getattr(self, "_adaptive_compensation_enabled", False):
            return bounded_positions.copy()

        now = time.monotonic()
        state = self.get_state()
        compensated = bounded_positions.copy()

        for local_index, global_index in enumerate(indices):
            requested_target = float(requested_positions[local_index])
            raw_measured = float(state.raw_positions[global_index])
            public_measured = float(state.public_positions[global_index])
            measured_velocity = float(state.velocities[global_index])
            previous_target = self._adaptive_compensation_target[global_index]
            previous_update = self._adaptive_compensation_last_update[global_index]
            self._adaptive_compensation_last_reset[global_index] = False
            self._adaptive_compensation_last_limited[global_index] = False
            self._adaptive_compensation_command_limited[global_index] = False
            self._adaptive_compensation_active[global_index] = False

            if np.isnan(previous_update):
                dt = ADAPTIVE_COMPENSATION_MIN_DT
            else:
                dt = float(
                    np.clip(
                        now - previous_update,
                        ADAPTIVE_COMPENSATION_MIN_DT,
                        ADAPTIVE_COMPENSATION_MAX_DT,
                    )
                )
            self._adaptive_compensation_last_dt[global_index] = dt

            feedback_finite = np.all(np.isfinite([raw_measured, public_measured, measured_velocity]))
            lower_limit = self._position_lower_limits[global_index]
            upper_limit = self._position_upper_limits[global_index]
            bounded_target = float(bounded_positions[local_index])
            target_limited = bool(target_limited_flags[local_index])

            reset = False
            if np.isnan(previous_target):
                self._adaptive_compensation_stable_target[global_index] = requested_target
                self._adaptive_compensation_learning_target[global_index] = requested_target
            else:
                target_delta = float(wrap_to_pi(requested_target - previous_target))
                if abs(target_delta) > ADAPTIVE_COMPENSATION_TARGET_DIRECTION_THRESHOLD:
                    direction = float(np.sign(target_delta))
                    previous_direction = self._adaptive_compensation_motion_direction[global_index]
                    if previous_direction != 0.0 and direction != previous_direction:
                        reset = True
                    self._adaptive_compensation_motion_direction[global_index] = direction

                learning_target = self._adaptive_compensation_learning_target[global_index]
                if np.isnan(learning_target):
                    self._adaptive_compensation_learning_target[global_index] = requested_target
                else:
                    target_from_learning_pose = float(wrap_to_pi(requested_target - learning_target))
                    if abs(target_from_learning_pose) > ADAPTIVE_COMPENSATION_TARGET_RESET_THRESHOLD:
                        reset = True

            if reset:
                self._adaptive_compensation_estimate[global_index] = 0.0
                self._adaptive_compensation_learning_target[global_index] = requested_target
                self._adaptive_compensation_stable_target[global_index] = requested_target
                self._adaptive_compensation_stable_since[global_index] = np.nan
                self._adaptive_compensation_last_reset[global_index] = True

            stable_target = self._adaptive_compensation_stable_target[global_index]
            target_is_stable = (
                np.isfinite(stable_target)
                and abs(float(wrap_to_pi(requested_target - stable_target)))
                <= ADAPTIVE_COMPENSATION_TARGET_STABLE_THRESHOLD
            )
            velocity_is_stable = feedback_finite and abs(measured_velocity) <= ADAPTIVE_COMPENSATION_VELOCITY_THRESHOLD
            if not target_is_stable:
                self._adaptive_compensation_stable_target[global_index] = requested_target
                self._adaptive_compensation_stable_since[global_index] = np.nan
            if velocity_is_stable:
                if np.isnan(self._adaptive_compensation_stable_since[global_index]):
                    self._adaptive_compensation_stable_since[global_index] = now
            else:
                self._adaptive_compensation_stable_since[global_index] = np.nan

            stable_since = self._adaptive_compensation_stable_since[global_index]
            active = (
                not reset
                and velocity_is_stable
                and np.isfinite(stable_since)
                and now - stable_since >= ADAPTIVE_COMPENSATION_STABLE_TIME
            )
            self._adaptive_compensation_active[global_index] = active

            self._adaptive_compensation_target[global_index] = requested_target
            self._adaptive_compensation_last_update[global_index] = now
            if not feedback_finite:
                self._adaptive_compensation_last_error[global_index] = np.nan
                self._adaptive_compensation_compensated_target[global_index] = bounded_target
                self._adaptive_compensation_command_limited[global_index] = target_limited
                compensated[local_index] = bounded_target
                continue

            bounded_public_target = float(wrap_to_pi(bounded_target))
            error = float(wrap_to_pi(bounded_public_target - public_measured))
            self._adaptive_compensation_last_error[global_index] = error

            estimate_candidate = self._adaptive_compensation_estimate[global_index]
            if active and abs(error) > ADAPTIVE_COMPENSATION_DEADBAND:
                estimate_candidate += ADAPTIVE_COMPENSATION_ADAPTATION_RATE * dt * error

            available_offset_lower = max(
                -ADAPTIVE_COMPENSATION_MAX_OFFSET,
                lower_limit - bounded_target,
            )
            available_offset_upper = min(
                ADAPTIVE_COMPENSATION_MAX_OFFSET,
                upper_limit - bounded_target,
            )
            estimate = float(np.clip(estimate_candidate, available_offset_lower, available_offset_upper))
            estimate_limited = not np.isclose(estimate_candidate, estimate)

            offset_alpha = 1.0 - np.exp(-ADAPTIVE_COMPENSATION_FILTER_BANDWIDTH * dt)
            offset_candidate = self._adaptive_compensation_offset[global_index] + offset_alpha * (
                estimate - self._adaptive_compensation_offset[global_index]
            )
            offset = float(np.clip(offset_candidate, available_offset_lower, available_offset_upper))
            offset_limited = not np.isclose(offset_candidate, offset)

            unclipped_compensated_target = bounded_target + offset
            compensated_target = float(np.clip(unclipped_compensated_target, lower_limit, upper_limit))
            final_limited = not np.isclose(unclipped_compensated_target, compensated_target)

            self._adaptive_compensation_last_limited[global_index] = estimate_limited or offset_limited
            self._adaptive_compensation_estimate[global_index] = estimate
            self._adaptive_compensation_offset[global_index] = offset
            self._adaptive_compensation_compensated_target[global_index] = compensated_target
            self._adaptive_compensation_command_limited[global_index] = (
                target_limited or offset_limited or final_limited
            )
            compensated[local_index] = compensated_target

        return compensated

    def _limit_position_targets(
        self,
        positions: Sequence[float],
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions_array = np.asarray(positions, dtype=float).copy()
        limited = np.zeros(len(positions_array), dtype=bool)
        if not getattr(self, "_position_limits_available", False):
            return positions_array, limited

        lower_limits = self._position_lower_limits[indices]
        upper_limits = self._position_upper_limits[indices]
        limit_midpoints = 0.5 * lower_limits + 0.5 * upper_limits
        unwrapped = unwrap_near(positions_array, limit_midpoints)
        bounded = np.clip(unwrapped, lower_limits, upper_limits)
        limited = ~np.isclose(unwrapped, bounded)
        return bounded, limited

    def get_frequency(self) -> float:
        return self._driver.get_frequency()

    def set_torque_enable(self, enable: TorqueEnable, ids: Optional[Sequence[int]] = None):
        ids_array, _ = self._resolve_indices(ids)
        self._driver.set_torque_enable(ids_array, [enable] * len(ids_array))

    def close(self):
        first_error: Optional[BaseException] = None
        cleanup_error: Optional[BaseException] = None
        try:
            if self._enable_gravity_compensation:
                self.stop_control_loop()
        except BaseException as exc:
            first_error = exc
        try:
            self._driver.set_torque_enable(
                self._ids,
                [TorqueEnable.Disable] * len(self._ids),
                force=True,
                wait=True,
            )
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            elif cleanup_error is None:
                cleanup_error = exc
        if self._close_driver:
            try:
                self._driver.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                elif cleanup_error is None:
                    cleanup_error = exc
        if first_error is not None:
            if cleanup_error is not None:
                raise first_error from cleanup_error
            raise first_error

    def start_control_loop(self):
        if not self._enable_gravity_compensation:
            raise RuntimeError("Gravity compensation must be enabled to start the control loop.")

        if not self._control_thread or not self._control_thread.is_alive():
            self._control_thread = Thread(target=self._control_loop, daemon=True)
            self._stop_flag.clear()
            self._control_thread.start()

    def stop_control_loop(self):
        if not self._enable_gravity_compensation:
            raise RuntimeError("Gravity compensation must be enabled to stop the control loop.")

        if self._control_thread and self._control_thread.is_alive():
            self._stop_flag.set()
            self._control_thread.join()
            self._control_thread = None

    def _control_loop(self):
        while not self._stop_flag.is_set():
            loop_start = time.perf_counter()
            joint_pos, joint_vel, _ = self.act()

            tau_n = self._null_space_regulation(joint_pos, joint_vel)  # 零空间投影
            tau_g = self._gravity_compensation(joint_pos, joint_vel)  # 重力补偿
            tau_ss = self._friction_compensation(tau_g, joint_vel)  # 摩擦力补偿
            tau = tau_n + tau_g + tau_ss
            self.set_torque(tau)
            sleep_time = self._control_period - (time.perf_counter() - loop_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    def _null_space_regulation(self, joint_pos, joint_vel):
        J = pin.computeJointJacobian(self._pin_model, self._pin_data, joint_pos, self._dof)
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self._dof) - J_dagger @ J
        q_error = joint_pos - self._null_space_joint_target
        tau_n = null_space_projector @ (-self._null_space_kp * q_error - self._null_space_kd * joint_vel)
        return tau_n

    def _gravity_compensation(self, joint_pos, joint_vel):
        tau_g = pin.rnea(self._pin_model, self._pin_data, joint_pos, joint_vel, np.zeros_like(joint_vel))
        tau_g *= self._gravity_comp_modifier
        return tau_g

    def _friction_compensation(self, tau_g, joint_vel):
        tau_ss = np.zeros(self._dof)
        for i in range(self._dof):
            if abs(joint_vel[i]) < self._stiction_comp_enable_velocity:
                if self._stiction_dither_flag[i]:
                    tau_ss[i] += self._stiction_comp_gain * abs(tau_g[i])
                else:
                    tau_ss[i] -= self._stiction_comp_gain * abs(tau_g[i])
                self._stiction_dither_flag[i] = ~self._stiction_dither_flag[i]
        return tau_ss


__all__ = ["FeeTechArm"]
