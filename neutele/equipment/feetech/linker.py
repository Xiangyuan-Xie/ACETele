import os
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

from neutele.config.config_loader import ConfigLoader
from neutele.equipment.base_equipment import BaseEquipment
from neutele.equipment.feetech.feetech_driver import FeeTechDriver, TorqueEnable

KT_MAPPING = {
    "HL3950": 1.0 / 20.8,
    "HL3930": 1.0 / 12.5,
    "HL3915": 1.0 / 9.3,
}

NO_LOAD_CURRENT = {
    "HL3915": 260,
}


class Linker(BaseEquipment):
    def __init__(self, station_type: str, config: Dict[str, Any], driver: Optional[FeeTechDriver] = None):
        super().__init__()
        self._ids = np.array(config["joint_ids"])
        self._dof = len(self._ids)
        self._gripper_ids = np.array(config["gripper_ids"])
        self._signs = np.array(config["joint_signs"])
        self._home_poses = np.array(config["home_poses"])
        self._dynamic_enable = config["dynamic_enable"]
        self._driver = driver if driver is not None else FeeTechDriver(self._ids, config["port"])

        if self._dynamic_enable:
            self._servo_types = np.array(config["servo_types"])
            self._torque_current_mapping = np.array([KT_MAPPING[servo] * 1000.0 for servo in self._servo_types])
            self._no_load_current = np.array([NO_LOAD_CURRENT[servo] for servo in self._servo_types])

            self._external_torque = np.zeros_like(self._dof)

            urdf_model_path = str(
                (
                    Path(__file__).resolve().parent.parent
                    / ".."
                    / "station"
                    / station_type
                    / "description"
                    / f"{station_type}.urdf"
                )
            )
            urdf_model_dir = os.path.dirname(urdf_model_path)
            self.pin_model, _, _ = pin.buildModelsFromUrdf(filename=urdf_model_path, package_dirs=urdf_model_dir)
            self.pin_data = self.pin_model.createData()

            self.null_space_joint_target = self._home_poses
            self.null_space_kp = 0.1
            self.null_space_kd = 0.01

            self.gravity_comp_modifier = 1.0

            self.stiction_dither_flag = np.ones(self._dof, dtype=bool)
            self.stiction_comp_enable_speed = 0.9
            self.stiction_comp_gain = 0.6

            self.torque_feedback_scalar = 0.05
            self.torque_feedback_damping = 0.0

            self._lock = Lock()
            self._stop_flag = Event()
            self._control_thread = Thread(target=self._control_loop, daemon=True)
            self._control_thread.start()

    def act(self, encode_gripper: bool = True) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        encoded_pos, encoded_vel = self._driver.get_pos_and_vel()

        positions = np.array(list(encoded_pos.values())) * self._signs * np.pi / 2048.0
        positions[positions > np.pi] -= 2 * np.pi
        positions[positions < -np.pi] += 2 * np.pi
        if encode_gripper:
            gripper_mask = np.isin(self._ids, self._gripper_ids)
            positions[gripper_mask] = 1.0 - np.clip(positions[gripper_mask] / (np.pi / 4.0), 0.0, 1.0)

        velocities = np.array(list(encoded_vel.values())) * self._signs * 0.732 * np.pi / 30

        return positions, velocities

    def set_torque(self, torques: Sequence[float], ids: Optional[Sequence[int]] = None):
        if ids is None:
            ids = self._ids

        assert len(ids) == len(torques), "ids and torques must have the same length."
        assert np.all(np.isin(ids, self._ids)), "ids is illegal."

        torques_Nm = np.asarray(torques)
        torques_kgcmf = torques_Nm / 0.0981
        currents = (
            (self._torque_current_mapping * np.abs(torques_kgcmf) + self._no_load_current) * np.sign(torques_kgcmf)
        ) * self._signs
        encoded_currents = np.around(currents / -6.5).astype(int)
        self._driver.set_current(ids[:-2], encoded_currents[:-2])

    def set_position(
        self, positions: Sequence[float], ids: Optional[Sequence[int]] = None, encode_gripper: bool = True
    ):
        if ids is None:
            ids = self._ids

        assert len(ids) == len(positions), "ids and positions must have the same length."
        assert np.all(np.isin(ids, self._ids)), "ids is illegal."

        positions_array = np.asarray(positions)
        if encode_gripper:
            gripper_mask = np.isin(ids, self._gripper_ids)
            assert 0.0 <= positions_array[gripper_mask] <= 1.0, "gripper id out of range."
            positions_array[gripper_mask] *= np.pi / 4.0 * self._signs[gripper_mask]
        encoded_positions = np.around(
            positions_array * self._signs[np.searchsorted(ids, self._ids)] * 2048.0 / np.pi
        ).astype(int)
        self._driver.set_position(ids, encoded_positions)

    def set_position_and_torque(
        self,
        positions: Sequence[float],
        torques: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        encode_gripper: bool = True,
    ):
        if ids is None:
            ids = self._ids

        assert len(ids) == len(positions) == len(torques), "ids, positions and torques must have the same length."
        assert np.all(np.isin(ids, self._ids)), "ids is illegal."

        positions_array = np.asarray(positions)
        if encode_gripper:
            gripper_mask = np.isin(ids, self._gripper_ids)
            assert 0.0 <= positions_array[gripper_mask] <= 1.0, "gripper id out of range."
            positions_array[gripper_mask] *= np.pi / 4.0 * self._signs[gripper_mask]
        encoded_positions = np.around(
            positions_array * self._signs[np.searchsorted(ids, self._ids)] * 2048.0 / np.pi
        ).astype(int)

        torques_Nm = np.asarray(torques)
        torques_kgcmf = torques_Nm / 0.0981
        currents = self._torque_current_mapping * np.abs(torques_kgcmf) + self._no_load_current
        encoded_currents = np.around(currents / 6.5).astype(int)

        self._driver.set_position_and_current(ids, encoded_positions, encoded_currents)

    def move_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        step_size: float = 0.02,
        min_steps: int = 2,
        max_steps: int = 100,
    ) -> float:
        if ids is None:
            ids = self._ids

        target_pos = np.asarray(positions)
        current_pos, _ = self.act()
        errors = np.abs(target_pos - current_pos)

        max_error = np.max(errors)
        if max_error < 0.001:
            self.set_position(ids=ids, positions=positions)
            return 1

        num_steps = int(np.ceil(max_error / step_size))
        num_steps = max(min_steps, min(num_steps, max_steps))

        for i in range(num_steps + 1):
            t = i / num_steps
            interp_pos = current_pos + (target_pos - current_pos) * t
            self.set_position(ids=ids, positions=interp_pos)

        return num_steps

    def get_frequency(self) -> float:
        return self._driver.get_frequency()

    def close(self):
        if self._dynamic_enable:
            self._stop_flag.set()
            self._control_thread.join()
            # self._driver.set_torque_enable(self._ids, [TorqueEnable.Disable] * len(self._ids))
        self._driver.set_torque_enable(self._ids, [TorqueEnable.Disable] * len(self._ids))
        time.sleep(0.1)
        self._driver.close()

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        with self._lock:
            self._external_torque = external_torque

    def _control_loop(self):
        while not self._stop_flag.is_set():
            pos, vel = self.act(encode_gripper=False)

            tau_n = self._null_space_regulation(pos, vel)  # 零空间投影
            tau_g = self._gravity_compensation(pos, vel)  # 重力补偿
            tau_ss = self._friction_compensation(tau_g, vel)  # 摩擦力补偿
            tau_fb = self._torque_feedback(vel)  # 力反馈
            tau = tau_n + tau_g + tau_ss + tau_fb
            self.set_torque(tau)
            # self.set_position_and_torque(positions=pos, torques=tau, encode_gripper=False)

            time.sleep(0.01)

    def _null_space_regulation(self, arm_joint_pos, arm_joint_vel):
        J = pin.computeJointJacobian(self.pin_model, self.pin_data, arm_joint_pos, self._dof)
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self._dof) - J_dagger @ J
        q_error = arm_joint_pos - self.null_space_joint_target
        tau_n = null_space_projector @ (-self.null_space_kp * q_error - self.null_space_kd * arm_joint_vel)
        return tau_n

    def _gravity_compensation(self, arm_joint_pos, arm_joint_vel):
        tau_g = pin.rnea(self.pin_model, self.pin_data, arm_joint_pos, arm_joint_vel, np.zeros_like(arm_joint_vel))
        tau_g *= self.gravity_comp_modifier
        return tau_g

    def _friction_compensation(self, tau_g, arm_joint_vel):
        tau_ss = np.zeros(self._dof)
        for i in range(self._dof):
            if abs(arm_joint_vel[i]) < self.stiction_comp_enable_speed:
                if self.stiction_dither_flag[i]:
                    tau_ss[i] += self.stiction_comp_gain * abs(tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain * abs(tau_g[i])
                self.stiction_dither_flag[i] = ~self.stiction_dither_flag[i]
        return tau_ss

    def _torque_feedback(self, arm_joint_vel):
        tau_fb = self.torque_feedback_scalar * self._external_torque
        tau_fb -= self.torque_feedback_damping * arm_joint_vel
        return tau_fb


if __name__ == "__main__":
    config_loader = ConfigLoader()
    linker = Linker(config_loader.config["basic"]["station_type"], config_loader.config["linker"]["single"])
    try:
        with np.printoptions(suppress=True):
            while True:
                # linker.act(encode_gripper=False)
                print(np.around(linker.act(encode_gripper=False), 4))
                time.sleep(0.05)
    except Exception as e:
        raise e
    finally:
        linker.close()
