from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

from acetele.config.config_loader import ConfigLoader
from acetele.equipment.base_equipment import BaseEquipment
from acetele.equipment.feetech.feetech_driver import FeeTechDriver, TorqueEnable
from acetele.equipment.feetech.servo_specs import (
    HLS_PROFILE_DEFAULTS_BY_SERVO,
    KT_MAPPING,
    NO_LOAD_CURRENT,
    PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2,
    PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
)
from acetele.utils.external_torque import ExternalTorqueObserver


@dataclass
class LinkerState:
    public_positions: np.ndarray
    raw_positions: np.ndarray
    velocities: np.ndarray
    motor_torque_magnitude: np.ndarray
    motor_torque_signed: np.ndarray


class Linker(BaseEquipment):
    def __init__(
        self,
        config: Dict[str, Any],
        driver: Optional[FeeTechDriver] = None,
        pin_model: pin.Model = None,
    ):
        super().__init__()
        self._ids = np.array(config["joint_ids"])
        self._dof = len(self._ids)

        self._signs = np.array(config["joint_signs"])
        self._enable_gravity_compensation = config["enable_gravity_compensation"]
        self._enable_estimate_external_torque = config["enable_estimate_external_torque"]
        self._control_period = float(config.get("control_period", 0.004))
        if "servo_types" not in config:
            raise ValueError("servo_types must be specified.")
        self._servo_types = np.array(config["servo_types"])
        if len(self._servo_types) != len(self._ids):
            raise ValueError("servo_types must have the same length as joint_ids.")
        unknown_servo_types = sorted(set(self._servo_types) - set(HLS_PROFILE_DEFAULTS_BY_SERVO))
        if unknown_servo_types:
            raise ValueError(f"unsupported servo_types: {unknown_servo_types}")
        self._profile_acceleration_defaults = np.array(
            [HLS_PROFILE_DEFAULTS_BY_SERVO[servo_type]["acceleration"] for servo_type in self._servo_types]
        )
        self._profile_current_defaults = np.array(
            [HLS_PROFILE_DEFAULTS_BY_SERVO[servo_type]["current"] for servo_type in self._servo_types]
        )
        self._profile_velocity_defaults = np.array(
            [HLS_PROFILE_DEFAULTS_BY_SERVO[servo_type]["velocity"] for servo_type in self._servo_types]
        )
        self._driver = driver if driver is not None else FeeTechDriver(self._ids, config["port"])
        self._close_driver = driver is None

        self._home_poses = np.asarray(config["home_poses"], dtype=float)

        self._torque_current_mapping = np.array([KT_MAPPING[servo] * 1000.0 for servo in self._servo_types])
        self._no_load_current = np.array([NO_LOAD_CURRENT[servo] for servo in self._servo_types])

        if pin_model:
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

            self._feedback_external_torque = np.zeros(self._dof)
            self._torque_feedback_scalar = 0.05
            self._torque_feedback_damping = 0.0

            self._lock = Lock()
            self._stop_flag = Event()
            self._control_thread = None

        if self._enable_estimate_external_torque:
            self._external_torque_observer = ExternalTorqueObserver(self._pin_model, self._dof)

    @property
    def ids(self):
        return self._ids

    @property
    def external_torque_estimation_enabled(self) -> bool:
        return bool(self._enable_estimate_external_torque)

    @property
    def external_wrench_frame_name(self) -> str:
        if not self.external_torque_estimation_enabled:
            return "link_5"
        return self._external_torque_observer.frame_name

    def act(self) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        state = self.get_linker_state()
        return state.public_positions, state.velocities, state.motor_torque_magnitude

    def get_linker_state(self) -> LinkerState:
        encoded_pos, encoded_vel, encoded_current = self._driver.get_state()

        raw_positions = np.array([encoded_pos[int(ft_id)] for ft_id in self._ids]) * self._signs * np.pi / 2048.0
        public_positions = raw_positions.copy()

        velocities = np.array([encoded_vel[int(ft_id)] for ft_id in self._ids]) * self._signs * 0.732 * np.pi / 30

        raw_currents = np.array([encoded_current[int(ft_id)] for ft_id in self._ids], dtype=float)
        currents = raw_currents * 6.5
        torques_kgcmf_mag = np.maximum(np.abs(currents) - self._no_load_current, 0.0) / self._torque_current_mapping
        torques_Nm_mag = torques_kgcmf_mag * 0.0981
        torque_sign = np.sign(-raw_currents * self._signs)
        torques_Nm_signed = torques_Nm_mag * torque_sign

        return LinkerState(
            public_positions=public_positions,
            raw_positions=raw_positions,
            velocities=velocities,
            motor_torque_magnitude=torques_Nm_mag,
            motor_torque_signed=torques_Nm_signed,
        )

    def set_torque(self, torques: Sequence[float], ids: Optional[Sequence[int]] = None):
        if ids is None:
            ids = self._ids
        ids = np.asarray(ids)

        assert len(ids) == len(torques), "ids and torques must have the same length."
        assert np.all(np.isin(ids, self._ids)), "ids is illegal."
        indices = np.array([int(np.where(self._ids == int(ft_id))[0][0]) for ft_id in ids])

        torques_Nm = np.asarray(torques)
        torques_kgcmf = torques_Nm / 0.0981
        torque_current_mapping = self._torque_current_mapping[indices]
        no_load_current = self._no_load_current[indices]
        signs = self._signs[indices]
        currents = ((torque_current_mapping * np.abs(torques_kgcmf) + no_load_current) * np.sign(torques_kgcmf)) * signs
        encoded_currents = np.around(currents / -6.5).astype(int)
        self._driver.set_current(ids, encoded_currents)

    def set_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ):
        if ids is None:
            ids = self._ids
        ids = np.asarray(ids)

        assert len(ids) == len(positions), "ids and positions must have the same length."
        assert np.all(np.isin(ids, self._ids)), "ids is illegal."
        indices = np.array([int(np.where(self._ids == int(ft_id))[0][0]) for ft_id in ids])
        if torque is not None:
            torque_array = np.asarray(torque, dtype=float)
            if torque_array.ndim == 0:
                torque_array = np.full(len(ids), float(torque_array))
            assert len(torque_array) == len(ids), "ids and torques must have the same length."
        else:
            torque_array = None

        positions_array = np.asarray(positions, dtype=float).copy()
        signs = self._signs[indices]
        encoded_positions = np.around(positions_array * signs * 2048.0 / np.pi).astype(int)

        if torque_array is None:
            currents_raw = self._profile_current_defaults[indices].copy()
        else:
            torques_Nm = np.asarray(torque_array, dtype=float)
            torques_kgcmf = torques_Nm / 0.0981
            torque_current_mapping = self._torque_current_mapping[indices]
            no_load_current = self._no_load_current[indices]
            currents = torque_current_mapping * np.abs(torques_kgcmf) + no_load_current
            currents_raw = np.around(currents / 6.5).astype(int)
        if velocities is None:
            velocities_raw = self._profile_velocity_defaults[indices].copy()
        else:
            velocities_array = np.asarray(velocities, dtype=float)
            if velocities_array.ndim == 0:
                velocities_array = np.full(len(ids), float(velocities_array))
            assert len(velocities_array) == len(ids), "ids and velocities must have the same length."
            velocities_raw = np.around(velocities_array / PROFILE_VELOCITY_UNIT_RAD_PER_SEC).astype(int)
        if accelerations is None:
            accelerations_raw = self._profile_acceleration_defaults[indices].copy()
        else:
            accelerations_array = np.asarray(accelerations, dtype=float)
            if accelerations_array.ndim == 0:
                accelerations_array = np.full(len(ids), float(accelerations_array))
            assert len(accelerations_array) == len(ids), "ids and accelerations must have the same length."
            accelerations_raw = np.around(accelerations_array / PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2).astype(int)
        self._driver.set_position(
            ids,
            encoded_positions,
            currents_raw=currents_raw,
            velocities_raw=velocities_raw,
            accelerations_raw=accelerations_raw,
        )

    def get_frequency(self) -> float:
        return self._driver.get_frequency()

    def set_torque_enable(self, enable: TorqueEnable, ids: Optional[Sequence[int]] = None):
        if ids is None:
            ids = self._ids
        ids = np.asarray(ids)
        assert np.all(np.isin(ids, self._ids)), "ids is illegal."
        self._driver.set_torque_enable(ids, [enable] * len(ids))

    def close(self):
        if self._enable_gravity_compensation:
            self.stop_control_loop()
        self._driver.set_torque_enable(self._ids, [TorqueEnable.Disable] * len(self._ids))
        time.sleep(0.1)
        if self._close_driver:
            self._driver.close()

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
            tau_fb = self._torque_feedback(joint_vel)  # 力反馈
            tau = tau_n + tau_g + tau_ss + tau_fb
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

    def _torque_feedback(self, joint_vel):
        tau_fb = (
            self._torque_feedback_scalar * self._feedback_external_torque - self._torque_feedback_damping * joint_vel
        )
        return tau_fb

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        with self._lock:
            self._feedback_external_torque = external_torque

    def reset_external_torque_estimator(self, joint_pos, joint_vel):
        self._external_torque_observer.reset(joint_pos, joint_vel)

    def update_momentum_observer(self, joint_pos, joint_vel, joint_effort, dt):
        return self._external_torque_observer.update(joint_pos, joint_vel, joint_effort, dt)

    def estimate_joint_external_torque(self, joint_pos, joint_vel, joint_effort, dt):
        return self.update_momentum_observer(joint_pos, joint_vel, joint_effort, dt)

    def external_wrench_from_joint_torque(self, joint_pos, joint_torque):
        return self._external_torque_observer.wrench_from_joint_torque(joint_pos, joint_torque)


if __name__ == "__main__":
    config_loader = ConfigLoader()
    config = config_loader.get_linker_config()[0]
    linker = Linker(config)
    try:
        with np.printoptions(suppress=True):
            while True:
                print(np.around(linker.act(), 4))
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        linker.close()
