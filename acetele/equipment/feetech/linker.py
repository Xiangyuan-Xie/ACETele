import os
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

from acetele.config.config_loader import ConfigLoader
from acetele.equipment.base_equipment import BaseEquipment
from acetele.equipment.feetech.feetech_driver import FeeTechDriver, TorqueEnable

KT_MAPPING = {
    "HL3950": 1.0 / 20.8,
    "HL3930": 1.0 / 12.5,
    "HL3915": 1.0 / 9.3,
}

NO_LOAD_CURRENT = {
    "HL3950": 330,
    "HL3930": 150,
    "HL3915": 260,
}

GRIPPER_ENCODING_SCALE = {
    "leader": np.pi / 4.0,
}
GRIPPER_DECODING_SCALE = {k: 1.0 / v for k, v in GRIPPER_ENCODING_SCALE.items()}


class Linker(BaseEquipment):
    def __init__(self, station_type: str, config: Dict[str, Any], driver: Optional[FeeTechDriver] = None):
        super().__init__()
        self._ids = np.array(config["joint_ids"])
        self._dof = len(self._ids)

        self._signs = np.array(config["joint_signs"])
        self._home_poses = np.array(config["home_poses"])
        self._enable_dynamic = config["enable_dynamic"]
        self._use_thread_backend = config["use_thread_backend"]
        self._driver = driver if driver is not None else FeeTechDriver(self._ids, config["port"])

        self._gripper_id = config["gripper_id"]
        if self._gripper_id >= 0:
            self._gripper_type = config["gripper_type"]
            self._gripper_encoding_scale = GRIPPER_ENCODING_SCALE[self._gripper_type]
            self._gripper_decoding_scale = GRIPPER_DECODING_SCALE[self._gripper_type]

        self._servo_types = np.array(config["servo_types"])
        self._torque_current_mapping = np.array([KT_MAPPING[servo] * 1000.0 for servo in self._servo_types])
        self._no_load_current = np.array([NO_LOAD_CURRENT[servo] for servo in self._servo_types])

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

        if self._enable_dynamic:
            self.null_space_joint_target = self._home_poses
            self.null_space_kp = 0.1
            self.null_space_kd = 0.01

            self.gravity_comp_modifier = 1.0

            self.stiction_dither_flag = np.ones(self._dof, dtype=bool)
            self.stiction_comp_enable_speed = 0.9
            self.stiction_comp_gain = 0.6

            self._feedback_external_torque = np.zeros_like(self._dof)
            self.torque_feedback_scalar = 0.05
            self.torque_feedback_damping = 0.0

            if self._use_thread_backend:
                self._lock = Lock()
                self._stop_flag = Event()
                self._control_thread = Thread(target=self._control_loop, daemon=True)
                self._control_thread.start()

            self.K = np.eye(self._dof) * 15.0  # Observer gain
            self.p_hat = np.zeros(self._dof)  # Estimated momentum
            self.ee_frame_id = self.pin_model.getFrameId("link_5")  # End-effector frame ID
            self.viscous_friction_gain = 0.02

    def act(
        self, encode_gripper: bool = True, cal_torque_sign: bool = False
    ) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        encoded_pos, encoded_vel, encoded_current = self._driver.get_state()

        positions = np.array(list(encoded_pos.values())) * self._signs * np.pi / 2048.0
        if encode_gripper:
            positions = self._encode_gripper(positions)

        velocities = np.array(list(encoded_vel.values())) * self._signs * 0.732 * np.pi / 30

        currents = np.array(list(encoded_current.values())) * 6.5
        torques_kgcmf_mag = np.maximum(np.abs(currents) - self._no_load_current, 0.0) / self._torque_current_mapping
        torques_Nm_mag = torques_kgcmf_mag * 0.0981

        if cal_torque_sign:
            if not self._enable_dynamic:
                pin.rnea(self.pin_model, self.pin_data, positions, velocities, np.zeros_like(velocities))
            torques_Nm = torques_Nm_mag * np.sign(self.pin_data.tau) * -self._signs
            return positions, velocities, torques_Nm
        else:
            return positions, velocities, torques_Nm_mag

    def _encode_gripper(self, positions: Sequence[float]):
        if self._gripper_id >= 0:
            positions_array = np.asarray(positions)
            gripper = positions_array[-1] % (2 * np.pi)
            if gripper > np.pi:
                gripper -= 2 * np.pi
            elif gripper <= -np.pi:
                gripper += 2 * np.pi
            positions_array[-1] = 1.0 - np.clip(gripper * self._gripper_encoding_scale, 0.0, 1.0)
        return positions_array

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
        signs = self._signs[np.searchsorted(self._ids, ids)]
        if encode_gripper and self._gripper_id >= 0 and self._gripper_id in ids:
            positions_array[-1] *= self._gripper_decoding_scale * signs[-1]
        encoded_positions = np.around(positions_array * signs * 2048.0 / np.pi).astype(int)
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
        signs = self._signs[np.searchsorted(self._ids, ids)]
        if encode_gripper and self._gripper_id >= 0 and self._gripper_id in ids:
            positions_array[-1] *= self._gripper_decoding_scale * signs[-1]
        encoded_positions = np.around(positions_array * signs * 2048.0 / np.pi).astype(int)

        current_positions_dict, _, _ = self._driver.get_state()
        current_positions_array = np.array([current_positions_dict[ft_id] for ft_id in ids])
        encoded_positions += np.round((current_positions_array - encoded_positions) / 4096.0).astype(int) * 4096

        torques_Nm = np.asarray(torques)
        torques_kgcmf = torques_Nm / 0.0981
        currents = self._torque_current_mapping * np.abs(torques_kgcmf) + self._no_load_current
        encoded_currents = np.around(currents / 6.5).astype(int)

        self._driver.set_position_and_current(ids, encoded_positions, encoded_currents)

    def move_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        step_size: float = 0.01,
        min_steps: int = 2,
        max_steps: int = 100,
    ) -> float:
        if ids is None:
            ids = self._ids

        ids = np.asarray(ids)
        positions_array = np.asarray(positions)
        assert len(ids) == len(positions_array), "ids and positions must have the same length."
        assert np.all(np.isin(ids, self._ids)), "ids is illegal."

        current_pos, _, _ = self.act()
        indices = np.searchsorted(self._ids, ids)
        current_pos = current_pos[indices]

        target_pos = positions_array
        errors = (target_pos - current_pos + np.pi / 2) % (2 * np.pi) - np.pi / 2

        max_error = np.max(np.abs(errors))
        if max_error < 0.001:
            self.set_position(ids=ids, positions=positions_array)
            return 1

        num_steps = int(np.ceil(max_error / step_size))
        num_steps = max(min_steps, min(num_steps, max_steps))

        for i in range(num_steps + 1):
            t = i / num_steps
            interp_pos = current_pos + errors * t
            self.set_position(ids=ids, positions=interp_pos)

        return num_steps

    def get_frequency(self) -> float:
        return self._driver.get_frequency()

    def close(self):
        if self._enable_dynamic and self._use_thread_backend:
            self._stop_flag.set()
            self._control_thread.join()
            # self._driver.set_torque_enable(self._ids, [TorqueEnable.Disable] * len(self._ids))
        self._driver.set_torque_enable(self._ids, [TorqueEnable.Disable] * len(self._ids))
        time.sleep(0.1)
        self._driver.close()

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        with self._lock:
            self._feedback_external_torque = external_torque

    def _control_loop(self):
        while not self._stop_flag.is_set():
            joint_pos, joint_vel, _ = self.act(encode_gripper=False)

            tau_n = self._null_space_regulation(joint_pos, joint_vel)  # 零空间投影
            tau_g = self._gravity_compensation(joint_pos, joint_vel)  # 重力补偿
            tau_ss = self._friction_compensation(tau_g, joint_vel)  # 摩擦力补偿
            tau_fb = self._torque_feedback(joint_vel)  # 力反馈
            tau = tau_n + tau_g + tau_ss + tau_fb
            self.set_torque(tau)
            # self.set_position_and_torque(positions=pos, torques=tau, encode_gripper=False)

            time.sleep(0.01)

    def _null_space_regulation(self, joint_pos, joint_vel):
        J = pin.computeJointJacobian(self.pin_model, self.pin_data, joint_pos, self._dof)
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self._dof) - J_dagger @ J
        q_error = joint_pos - self.null_space_joint_target
        tau_n = null_space_projector @ (-self.null_space_kp * q_error - self.null_space_kd * joint_vel)
        return tau_n

    def _gravity_compensation(self, joint_pos, joint_vel):
        tau_g = pin.rnea(self.pin_model, self.pin_data, joint_pos, joint_vel, np.zeros_like(joint_vel))
        tau_g *= self.gravity_comp_modifier
        return tau_g

    def _friction_compensation(self, tau_g, joint_vel):
        tau_ss = np.zeros(self._dof)
        for i in range(self._dof):
            if abs(joint_vel[i]) < self.stiction_comp_enable_speed:
                if self.stiction_dither_flag[i]:
                    tau_ss[i] += self.stiction_comp_gain * abs(tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain * abs(tau_g[i])
                self.stiction_dither_flag[i] = ~self.stiction_dither_flag[i]
        return tau_ss

    def _torque_feedback(self, joint_vel):
        tau_fb = self.torque_feedback_scalar * self._feedback_external_torque - self.torque_feedback_damping * joint_vel
        return tau_fb

    def _momentum_observer_step(self, joint_pos, joint_vel, joint_effort, dt):
        pin.computeAllTerms(self.pin_model, self.pin_data, joint_pos, joint_vel)

        M = self.pin_data.M
        g = self.pin_data.g

        # C^T * q_dot term
        C = self.pin_data.C
        beta = C.T @ joint_vel

        p = M @ joint_vel

        # Friction compensation
        tau_f = self.stiction_comp_gain * np.sign(joint_vel) + self.viscous_friction_gain * joint_vel

        # Momentum observer update
        # p_hat_dot = tau - tau_f - g + C^T*v + K*(p - p_hat)
        p_hat_dot = joint_effort - tau_f - g + beta + self.K @ (p - self.p_hat)
        self.p_hat += p_hat_dot * dt

        tau_ext_hat = self.K @ (p - self.p_hat)

        return tau_ext_hat

    def estimate_joint_external_torque(self, joint_pos, joint_vel, joint_effort, dt):
        tau_ext = self._momentum_observer_step(joint_pos, joint_vel, joint_effort, dt)
        return tau_ext

    def estimate_ee_external_wrench(self, joint_pos, joint_vel, joint_effort, dt):
        tau_ext = self._momentum_observer_step(joint_pos, joint_vel, joint_effort, dt)

        pin.computeJointJacobians(self.pin_model, self.pin_data, joint_pos)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        J = pin.getFrameJacobian(
            self.pin_model, self.pin_data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # Use pseudo-inverse to map joint torques to end-effector wrench
        # J.T * f_ext = tau_ext  => f_ext = (J.T)^+ * tau_ext
        f_ext = np.linalg.pinv(J.T) @ tau_ext
        return f_ext


if __name__ == "__main__":
    config_loader = ConfigLoader()
    linker = Linker(config_loader.config["basic"]["station_type"], config_loader.config["linker"]["single"])
    try:
        with np.printoptions(suppress=True):
            while True:
                # print(np.around(linker.act(encode_gripper=False), 4))
                print(np.around(linker.act(), 4))
                time.sleep(0.05)
    except Exception as e:
        raise e
    finally:
        linker.close()
