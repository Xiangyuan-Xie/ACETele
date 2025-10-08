import os

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin


class MujocoBase:
    def __init__(self, model_path: str):
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, self._data)

        self.dof = 5

        urdf_model_path = os.path.abspath(
            "G:\\NEU_Tele\\neutele\\station\\flying_hand\\urdf\\flying_hand_leader_tmp.urdf"
        )
        urdf_model_dir = os.path.dirname(urdf_model_path)
        self.pin_model, _, _ = pin.buildModelsFromUrdf(filename=urdf_model_path, package_dirs=urdf_model_dir)
        self.pin_data = self.pin_model.createData()

        self.null_space_joint_target = [0.0, 0.0, 1.5708, 0.0, 0.0]
        self.null_space_kp = 0.1
        self.null_space_kd = 0.01

        self.tau_g = np.zeros(self.dof)
        self.gravity_comp_modifier = 1.0

        self.stiction_dither_flag = np.ones(self.dof, dtype=bool)
        self.stiction_comp_enable_speed = 0.9
        self.stiction_comp_gain = 0.6

        self._data.qpos = self.null_space_joint_target
        self._data.qvel = np.zeros_like(self._data.qpos)

    def control(self):
        pos = self._data.qpos
        vel = self._data.qvel

        torque_arm = np.zeros(self.dof)
        torque_arm += self._null_space_regulation(pos, vel)  # 零空间投影
        torque_arm += self._gravity_compensation(pos, vel)  # 重力补偿
        # torque_arm += self._friction_compensation(vel)  # 摩擦力补偿

        self._data.ctrl[:] = torque_arm

    def run(self):
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            while viewer.is_running():
                self.control()
                mujoco.mj_step(self._model, self._data)
                viewer.sync()

    def _null_space_regulation(self, arm_joint_pos, arm_joint_vel):
        J = pin.computeJointJacobian(self.pin_model, self.pin_data, arm_joint_pos, self.dof)
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self.dof) - J_dagger @ J
        q_error = arm_joint_pos - self.null_space_joint_target
        tau_n = null_space_projector @ (-self.null_space_kp * q_error - self.null_space_kd * arm_joint_vel)
        return tau_n

    def _gravity_compensation(self, arm_joint_pos, arm_joint_vel):
        self.tau_g = pin.rnea(self.pin_model, self.pin_data, arm_joint_pos, arm_joint_vel, np.zeros_like(arm_joint_vel))
        self.tau_g *= self.gravity_comp_modifier
        return self.tau_g

    def _friction_compensation(self, arm_joint_vel):
        tau_ss = np.zeros(self.dof)
        for i in range(self.dof):
            if abs(arm_joint_vel[i]) < self.stiction_comp_enable_speed:
                if self.stiction_dither_flag[i]:
                    tau_ss[i] += self.stiction_comp_gain * abs(self.tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain * abs(self.tau_g[i])
                self.stiction_dither_flag[i] = ~self.stiction_dither_flag[i]
        return tau_ss


if __name__ == "__main__":
    base = MujocoBase("G:\\NEU_Tele\\neutele\\station\\flying_hand\\urdf\\flying_hand_leader.xml")
    base.run()
