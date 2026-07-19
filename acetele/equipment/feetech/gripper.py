from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from acetele.config.robot_config import FeeTechGripperConfig
from acetele.equipment.feetech.feetech_driver import FeeTechDriver, TorqueEnable
from acetele.equipment.feetech.servo_specs import (
    HLS_PROFILE_DEFAULTS_BY_SERVO,
    KT_MAPPING,
    NO_LOAD_CURRENT,
    PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2,
    PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
    validate_feetech_servo_models,
)
from acetele.equipment.joint_device import JointDeviceState, _normalize_profile_value
from acetele.utils.angle import wrap_to_pi
from acetele.utils.joint import normalize_joint_ids

FEETECH_PROFILE_VELOCITY_MAX = 32767
FEETECH_PROFILE_ACCELERATION_MAX = 255
FEETECH_PROFILE_CURRENT_MAX = 32767


class FeeTechGripper:
    """Single-axis FEETECH gripper with a normalized public position."""

    def __init__(
        self,
        config: FeeTechGripperConfig,
        driver: Optional[FeeTechDriver] = None,
    ):
        self.config = config
        self._id = config.joint_id
        self._ids = np.asarray(config.joint_ids, dtype=int)
        self._joint_names = config.joint_names
        self._sign = float(config.joint_sign)
        self._travel_range_counts = config.travel_range_counts
        self._servo_model = config.servo_model
        validate_feetech_servo_models((self._servo_model,), context="gripper")
        defaults = HLS_PROFILE_DEFAULTS_BY_SERVO[self._servo_model]
        self._profile_acceleration_default = defaults["acceleration"]
        self._profile_current_default = defaults["current"]
        self._profile_velocity_default = defaults["velocity"]
        self._torque_current_mapping = KT_MAPPING[self._servo_model] * 1000.0
        self._no_load_current = NO_LOAD_CURRENT[self._servo_model]
        if driver is None and config.port is None:
            raise ValueError("physical gripper requires a serial port")
        self._driver = driver if driver is not None else FeeTechDriver(self._ids, str(config.port))
        self._close_driver = driver is None

    @property
    def ids(self) -> np.ndarray:
        return self._ids.copy()

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    def act(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.get_state().act()

    def get_state(self) -> JointDeviceState:
        encoded_pos, encoded_vel, encoded_current = self._driver.get_state()
        signed_position_counts = float(encoded_pos[self._id]) * self._sign
        raw_position = signed_position_counts * np.pi / 2048.0
        wrapped_position = float(wrap_to_pi(raw_position))
        public_position = float(
            np.clip(
                wrapped_position
                / (self._travel_range_counts * np.pi / 2048.0),
                0.0,
                1.0,
            )
        )
        velocity = float(encoded_vel[self._id]) * self._sign * PROFILE_VELOCITY_UNIT_RAD_PER_SEC
        raw_current = float(encoded_current[self._id])
        torque_kgcmf = max(abs(raw_current * 6.5) - self._no_load_current, 0.0) / self._torque_current_mapping
        torque_magnitude = torque_kgcmf * 0.0981
        return JointDeviceState(
            public_positions=np.asarray([public_position]),
            raw_positions=np.asarray([raw_position]),
            velocities=np.asarray([velocity]),
            motor_torque_magnitude=np.asarray([torque_magnitude]),
            motor_torque_signed=np.asarray(
                [torque_magnitude * float(np.sign(-raw_current * self._sign))]
            ),
        )

    def set_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ) -> None:
        (
            position,
            current_raw,
            velocity_raw,
            acceleration_raw,
        ) = self._prepare_position_command(
            positions,
            ids=ids,
            velocities=velocities,
            accelerations=accelerations,
            torque=torque,
        )
        encoded_position = int(np.rint(position * self._travel_range_counts)) * int(
            self._sign
        )
        self._driver.set_position(
            [self._id],
            [encoded_position],
            currents_raw=[current_raw],
            velocities_raw=[velocity_raw],
            accelerations_raw=[acceleration_raw],
        )

    def validate_position_command(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ) -> None:
        self._prepare_position_command(
            positions,
            ids=ids,
            velocities=velocities,
            accelerations=accelerations,
            torque=torque,
        )

    def _prepare_position_command(
        self,
        positions: Sequence[float],
        *,
        ids: Optional[Sequence[int]],
        velocities: Optional[Sequence[float] | float],
        accelerations: Optional[Sequence[float] | float],
        torque: Optional[Sequence[float] | float],
    ) -> tuple[float, int, int, int]:
        positions_array = np.asarray(positions, dtype=float)
        if positions_array.shape != (1,) or not np.all(np.isfinite(positions_array)):
            raise ValueError("gripper command must contain one finite position")
        if ids is not None and normalize_joint_ids(ids, field_name="gripper joint ids") != (self._id,):
            raise ValueError("gripper command contains an unknown joint id")

        torque_value = self._single_profile_value("torque", torque)
        current_value = (
            self._profile_current_default
            if torque_value is None
            else (
                self._torque_current_mapping * abs(torque_value / 0.0981)
                + self._no_load_current
            )
            / 6.5
        )
        current_raw = self._encode_profile_value(
            "current",
            current_value,
            FEETECH_PROFILE_CURRENT_MAX,
        )

        velocity_value = self._single_profile_value("velocities", velocities)
        velocity_raw = self._encode_profile_value(
            "velocity",
            self._profile_velocity_default
            if velocity_value is None
            else velocity_value / PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
            FEETECH_PROFILE_VELOCITY_MAX,
        )
        acceleration_value = self._single_profile_value("accelerations", accelerations)
        acceleration_raw = self._encode_profile_value(
            "acceleration",
            self._profile_acceleration_default
            if acceleration_value is None
            else acceleration_value / PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2,
            FEETECH_PROFILE_ACCELERATION_MAX,
        )
        return (
            float(np.clip(positions_array[0], 0.0, 1.0)),
            current_raw,
            velocity_raw,
            acceleration_raw,
        )

    @staticmethod
    def _single_profile_value(
        name: str,
        value: Optional[Sequence[float] | float],
    ) -> Optional[float]:
        normalized = _normalize_profile_value(name, value, 1)
        if normalized is None:
            return None
        array = np.asarray(normalized, dtype=float)
        return float(array if array.ndim == 0 else array[0])

    @staticmethod
    def _encode_profile_value(name: str, value: float, maximum: int) -> int:
        rounded = float(np.rint(value))
        if not np.isfinite(rounded) or rounded < 0 or rounded > maximum:
            raise ValueError(f"encoded {name} must be between 0 and {maximum}")
        return int(rounded)

    def set_torque_enable(
        self,
        enable: TorqueEnable,
        ids: Optional[Sequence[int]] = None,
    ) -> None:
        if ids is not None and normalize_joint_ids(ids, field_name="gripper joint ids") != (self._id,):
            raise ValueError("gripper torque command contains an unknown joint id")
        self._driver.set_torque_enable([self._id], [enable])

    def close(self) -> None:
        first_error: Optional[BaseException] = None
        cleanup_error: Optional[BaseException] = None
        try:
            self._driver.set_torque_enable(
                [self._id],
                [TorqueEnable.Disable],
                force=True,
                wait=True,
            )
        except BaseException as exc:
            first_error = exc
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


__all__ = ["FeeTechGripper"]
