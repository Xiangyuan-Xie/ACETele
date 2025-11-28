import os
import time
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

from neutele.config.config_loader import ConfigLoader
from neutele.equipment.base_equipment import BaseEquipment
from neutele.equipment.feetech.feetech_driver import FeeTechDriver, TorqueEnable

KT_MAPPING = {
    "HL3950": 0.3 * 20.8,
    "HL3930": 0.65 * 12.5,
    "HL3915": 9.3,
    "HL3612": 9.2,
}


class Linker(BaseEquipment):
    def __init__(self, config: Dict[str, Any], driver: Optional[FeeTechDriver] = None):
        super().__init__()
        self._ids = np.array(config["joint_ids"])
        self._signs = np.array(config["joint_signs"])
        self._home_poses = np.array(config["home_poses"])
        self._dynamic_enable = config["dynamic_enable"]
        self._driver = driver if driver is not None else FeeTechDriver(self._ids, config["port"])

        self._dof = 5

        if self._dynamic_enable:
            self._servo_types = np.array(config["servo_types"])
            self._torque_current_mapping = np.array(
                [-1.0 / 0.0981 / KT_MAPPING[servo] * 1000.0 / 6.5 for servo in self._servo_types]
            )

            self._external_torque = np.zeros_like(self._dof)

            urdf_model_path = os.path.abspath(
                "G:\\NEU_Tele\\neutele\\station\\flying_hand\\urdf\\leader\\flying_hand_leader_tmp.urdf"
            )
            urdf_model_dir = os.path.dirname(urdf_model_path)
            self.pin_model, _, _ = pin.buildModelsFromUrdf(filename=urdf_model_path, package_dirs=urdf_model_dir)
            self.pin_data = self.pin_model.createData()

            self.null_space_joint_target = [0.0, 0.0, 1.5708, 0.0, 0.0]
            self.null_space_kp = 0.1
            self.null_space_kd = 0.01

            self.gravity_comp_modifier = 1.0

            self.stiction_dither_flag = np.ones(self._dof, dtype=bool)
            self.stiction_comp_enable_speed = 0.9
            # self.stiction_comp_gain = 0.6
            self.stiction_comp_gain = np.array([1.0, 0.6, 1.0, 0.6, 0.6])

            self.torque_feedback_scalar = 0.05
            self.torque_feedback_damping = 0.0

            self._lock = Lock()
            self._stop_flag = Event()
            self._control_thread = Thread(target=self._control_worker, daemon=True)
            self._control_thread.start()

    def act(self) -> Tuple[Sequence[float], Sequence[float]]:
        pos, vel = self._driver.get_pos_and_vel()
        pos = np.array([v for k, v in pos.items() if k in self._ids]) * self._signs * np.pi / 2048.0
        vel = np.array([v for k, v in vel.items() if k in self._ids]) * self._signs * 0.732 * np.pi / 30
        return pos, vel

    def set_torque(self, ids: Sequence[int], torques: Sequence[float]):
        assert len(ids) == len(torques), "ids and torques must have the same length."
        currents = np.around(torques * self._torque_current_mapping * self._signs[:4]).astype(int)
        ids = ids[:3]
        currents = currents[:3]
        self._driver.set_current(ids, currents)

    def get_frequency(self) -> float:
        return self._driver.get_frequency()

    def close(self):
        if self._dynamic_enable:
            self._stop_flag.set()
            self._control_thread.join()
            self._driver.set_torque_enable(self._ids, [TorqueEnable.Disable] * len(self._ids))
        time.sleep(0.1)
        self._driver.close()

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        with self._lock:
            self._external_torque = external_torque

    def _control_worker(self):
        while not self._stop_flag.is_set():
            pos, vel = self.act()
            pos = np.append(pos, 0.0)
            vel = np.append(vel, 0.0)

            tau_n = self._null_space_regulation(pos, vel)  # 零空间投影
            tau_g = self._gravity_compensation(pos, vel)  # 重力补偿
            tau_ss = self._friction_compensation(tau_g, vel)  # 摩擦力补偿
            tau_fb = self._torque_feedback(vel)
            torque_arm = tau_n + tau_g + tau_ss + tau_fb
            self.set_torque(self._ids, torque_arm[:4])

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
                    tau_ss[i] += self.stiction_comp_gain[i] * abs(tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain[i] * abs(tau_g[i])
                self.stiction_dither_flag[i] = ~self.stiction_dither_flag[i]
        return tau_ss

    def _torque_feedback(self, arm_joint_vel):
        tau_fb = self.torque_feedback_scalar * self._external_torque
        tau_fb -= self.torque_feedback_damping * arm_joint_vel
        return tau_fb


if __name__ == "__main__":
    config_loader = ConfigLoader()
    linker = Linker(config_loader.config["linker"]["single"])
    while True:
        print(linker.act())
        time.sleep(0.05)
