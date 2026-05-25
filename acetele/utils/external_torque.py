from __future__ import annotations

import numpy as np
import pinocchio as pin

_OBSERVER_GAIN = 15.0
_LPF_ALPHA = 0.2
_MAX_OBSERVER_DT = 0.05
_STATIC_FRICTION = 0.0
_VISCOUS_FRICTION_GAIN = 0.02
_EXTERNAL_WRENCH_FRAME_NAME = "link_5"


class ExternalTorqueObserver:
    def __init__(self, pin_model: pin.Model, dof: int):
        if pin_model is None:
            raise ValueError("Pinocchio model is None, please provide a valid model.")
        if pin_model.nv != dof:
            raise ValueError(f"Pinocchio model nv ({pin_model.nv}) must match joint_ids length ({dof}).")

        self._pin_model = pin_model
        self._pin_data = self._pin_model.createData()
        self._dof = dof
        self._K = np.eye(self._dof) * _OBSERVER_GAIN
        self._p_hat = np.zeros(self._dof)
        self._external_wrench_frame_name = _EXTERNAL_WRENCH_FRAME_NAME
        self._ee_frame_id = self._pin_model.getFrameId(self._external_wrench_frame_name)
        if self._ee_frame_id == len(self._pin_model.frames):
            raise ValueError(f"external_wrench_frame '{self._external_wrench_frame_name}' not found.")
        self._observer_static_friction = _STATIC_FRICTION
        self._viscous_friction_gain = _VISCOUS_FRICTION_GAIN
        self._max_observer_dt = _MAX_OBSERVER_DT
        self._observer_initialized = False
        self._estimated_external_torque = np.zeros(self._dof)
        self._lpf_alpha = _LPF_ALPHA

    @property
    def frame_name(self) -> str:
        return self._external_wrench_frame_name

    def reset(self, joint_pos, joint_vel):
        pin.computeAllTerms(self._pin_model, self._pin_data, joint_pos, joint_vel)
        self._p_hat = self._pin_data.M @ joint_vel
        self._estimated_external_torque = np.zeros(self._dof)
        self._observer_initialized = True

    def update(self, joint_pos, joint_vel, joint_effort, dt):
        pin.computeAllTerms(self._pin_model, self._pin_data, joint_pos, joint_vel)
        M = self._pin_data.M
        g = self._pin_data.g
        C = self._pin_data.C
        beta = C.T @ joint_vel
        p = M @ joint_vel
        if not self._observer_initialized:
            self._p_hat = p.copy()
            self._estimated_external_torque = np.zeros(self._dof)
            self._observer_initialized = True
            return self._estimated_external_torque
        if dt <= 0.0 or dt > self._max_observer_dt:
            self._p_hat = p.copy()
            self._estimated_external_torque = np.zeros(self._dof)
            return self._estimated_external_torque
        tau_f = self._observer_static_friction * np.sign(joint_vel) + self._viscous_friction_gain * joint_vel
        p_hat_dot = joint_effort - tau_f - g + beta + self._K @ (p - self._p_hat)
        self._p_hat += p_hat_dot * dt
        tau_ext_raw = self._K @ (p - self._p_hat)
        self._estimated_external_torque = self._lpf_alpha * tau_ext_raw + (
            1.0 - self._lpf_alpha
        ) * self._estimated_external_torque
        return self._estimated_external_torque

    def wrench_from_joint_torque(self, joint_pos, joint_torque):
        pin.computeJointJacobians(self._pin_model, self._pin_data, joint_pos)
        pin.updateFramePlacements(self._pin_model, self._pin_data)
        J = pin.getFrameJacobian(
            self._pin_model, self._pin_data, self._ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.linalg.pinv(J.T) @ joint_torque
