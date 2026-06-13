from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from acetele.equipment.base_equipment import BaseEquipment
from acetele.equipment.feetech.feetech_driver import FeeTechDriver, TorqueEnable
from acetele.equipment.feetech.servo_specs import (
    HLS_PROFILE_DEFAULTS_BY_SERVO,
    KT_MAPPING,
    NO_LOAD_CURRENT,
    PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2,
    PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
)
from acetele.utils.gripper import GRIPPER_DECODING_SCALE, GRIPPER_ENCODING_SCALE


@dataclass
class GripperState:
    public_position: float
    raw_position: float
    velocity: float
    motor_torque_magnitude: float
    motor_torque_signed: float


class Gripper(BaseEquipment):
    def __init__(self, config: Dict[str, Any], driver: FeeTechDriver):
        super().__init__()
        if "joint_id" not in config:
            raise ValueError("gripper.single.joint_id must be specified.")
        self._id = int(config["joint_id"])
        self._ids = np.array([self._id])
        self._sign = float(config["joint_sign"])
        self._home_pose = float(config.get("home_pose", 0.0))
        self._gripper_type = config["gripper_type"]
        self._gripper_encoding_scale = GRIPPER_ENCODING_SCALE[self._gripper_type]
        self._gripper_decoding_scale = GRIPPER_DECODING_SCALE[self._gripper_type]
        self._servo_type = config["servo_type"]
        if self._servo_type not in HLS_PROFILE_DEFAULTS_BY_SERVO:
            raise ValueError(f"unsupported servo_type: {self._servo_type}")
        self._profile_acceleration_default = HLS_PROFILE_DEFAULTS_BY_SERVO[self._servo_type]["acceleration"]
        self._profile_current_default = HLS_PROFILE_DEFAULTS_BY_SERVO[self._servo_type]["current"]
        self._profile_velocity_default = HLS_PROFILE_DEFAULTS_BY_SERVO[self._servo_type]["velocity"]
        self._torque_current_mapping = KT_MAPPING[self._servo_type] * 1000.0
        self._no_load_current = NO_LOAD_CURRENT[self._servo_type]
        self._driver = driver

    @property
    def id(self) -> int:
        return self._id

    @property
    def ids(self):
        return self._ids

    def act(self) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        state = self.get_state()
        return (
            np.array([state.public_position]),
            np.array([state.velocity]),
            np.array([state.motor_torque_magnitude]),
        )

    def get_state(self) -> GripperState:
        encoded_pos, encoded_vel, encoded_current = self._driver.get_state()
        raw_position = float(encoded_pos[self._id]) * self._sign * np.pi / 2048.0
        public_position = raw_position % (2 * np.pi)
        if public_position > np.pi:
            public_position -= 2 * np.pi
        elif public_position <= -np.pi:
            public_position += 2 * np.pi
        public_position = float(np.clip(public_position * self._gripper_encoding_scale, 0.0, 1.0))

        velocity = float(encoded_vel[self._id]) * self._sign * PROFILE_VELOCITY_UNIT_RAD_PER_SEC
        raw_current = float(encoded_current[self._id])
        torque_kgcmf = max(abs(raw_current * 6.5) - self._no_load_current, 0.0) / self._torque_current_mapping
        torque_magnitude = torque_kgcmf * 0.0981
        return GripperState(
            public_position=public_position,
            raw_position=raw_position,
            velocity=velocity,
            motor_torque_magnitude=torque_magnitude,
            motor_torque_signed=torque_magnitude * float(np.sign(-raw_current * self._sign)),
        )

    def set_position(
        self,
        position: float,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        torque: Optional[float] = None,
    ):
        encoded_position = int(
            np.around(float(np.clip(position, 0.0, 1.0)) * self._gripper_decoding_scale * self._sign * 2048.0 / np.pi)
        )
        if torque is None:
            current_raw = self._profile_current_default
        else:
            current_raw = int(
                np.around(
                    (self._torque_current_mapping * abs(float(torque) / 0.0981) + self._no_load_current) / 6.5
                )
            )
        velocity_raw = (
            self._profile_velocity_default
            if velocity is None
            else int(np.around(float(velocity) / PROFILE_VELOCITY_UNIT_RAD_PER_SEC))
        )
        acceleration_raw = (
            self._profile_acceleration_default
            if acceleration is None
            else int(np.around(float(acceleration) / PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2))
        )
        self._driver.set_position(
            [self._id],
            [encoded_position],
            currents_raw=[current_raw],
            velocities_raw=[velocity_raw],
            accelerations_raw=[acceleration_raw],
        )

    def set_torque_enable(self, enable: TorqueEnable):
        self._driver.set_torque_enable([self._id], [enable])

    def close(self):
        self.set_torque_enable(TorqueEnable.Disable)
