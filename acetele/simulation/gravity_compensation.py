import os
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin


class GravityCompensationEnv:
    """
    A gravity compensation environment for robotic manipulator control in MuJoCo.

    This class implements a comprehensive control system featuring:
    - Gravity compensation using Pinocchio's RNEA algorithm
    - Null-space regulation for redundancy resolution
    - Friction compensation with dithering strategy
    - Stiction compensation for static friction effects
    """

    def __init__(self, model_path: str):
        """
        Initialize the gravity compensation environment.

        Args:
            model_path (str): Path to the MuJoCo XML model file.

        Raises:
            FileNotFoundError: If model files are not found.
            ValueError: If model parameters are invalid.
        """
        # Initialize MuJoCo model and simulation data
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Degrees of freedom for the robotic arm
        self.dof = 5

        # Initialize Pinocchio model for dynamics computations
        urdf_model_path = model_path.replace(".xml", ".urdf")
        urdf_model_dir = os.path.dirname(urdf_model_path)
        self.pin_model, _, _ = pin.buildModelsFromUrdf(filename=urdf_model_path, package_dirs=urdf_model_dir)
        self.pin_data = self.pin_model.createData()

        # Null-space regulation parameters
        self.null_space_joint_target = [0.0, 1.5708, 0.0, 0.0, 0.0]  # Desired joint configuration [rad]
        self.null_space_kp = 0.1  # Proportional gain for null-space regulation
        self.null_space_kd = 0.01  # Derivative gain for null-space regulation

        # Gravity compensation parameters
        self.tau_g = np.zeros(self.dof)  # Gravity torque vector
        self.gravity_comp_modifier = 1.0  # Gravity compensation scaling factor

        # Friction compensation parameters
        self.stiction_dither_flag = np.ones(self.dof, dtype=bool)  # Dithering state flags
        self.stiction_comp_enable_speed = 0.9  # Velocity threshold for stiction compensation [rad/s]
        self.stiction_comp_gain = 0.6  # Stiction compensation gain

        # Initialize simulation to null-space target configuration
        self.mj_data.qpos = self.null_space_joint_target
        self.mj_data.qvel = np.zeros_like(self.mj_data.qpos)
        self.mj_data.qacc = np.zeros_like(self.mj_data.qpos)

    def control(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Main control callback for MuJoCo simulation.

        This method computes and applies the composite control torque consisting of:
        1. Null-space regulation for redundancy management
        2. Gravity compensation for static load balancing
        3. Friction compensation for motion smoothness

        Args:
            model (mujoco.MjModel): MuJoCo model (unused but required for callback).
            data (mujoco.MjData): MuJoCo simulation data containing current state.
        """
        # Extract current joint states
        pos = self.mj_data.qpos
        vel = self.mj_data.qvel

        # Initialize control torque vector
        torque_arm = np.zeros(self.dof)

        # Apply null-space regulation for redundancy resolution
        torque_arm += self.null_space_regulation(pos, vel)

        # Apply gravity compensation for static load balancing
        torque_arm += self.gravity_compensation(pos, vel)

        # Apply friction compensation for motion smoothness
        # torque_arm += self.friction_compensation(vel)

        # Set computed control torque
        self.mj_data.ctrl = torque_arm

    def run(self):
        """
        Start the gravity compensation simulation.

        Sets up the control callback and launches the MuJoCo viewer
        for interactive visualization and control.
        """
        mujoco.set_mjcb_control(self.control)
        mujoco.viewer.launch(self.mj_model, self.mj_data)

    def null_space_regulation(self, arm_joint_pos: np.ndarray, arm_joint_vel: np.ndarray) -> np.ndarray:
        """
        Compute null-space regulation torque for redundancy resolution.

        This method projects joint-space errors onto the null-space of the
        Jacobian to regulate redundant degrees of freedom without affecting
        end-effector position.

        Args:
            arm_joint_pos (np.ndarray): Current joint positions [rad].
            arm_joint_vel (np.ndarray): Current joint velocities [rad/s].

        Returns:
            np.ndarray: Null-space regulation torque [N·m].
        """
        # Compute Jacobian for the end-effector
        J = pin.computeJointJacobian(self.pin_model, self.pin_data, arm_joint_pos, self.dof)

        # Compute Moore-Penrose pseudoinverse of Jacobian
        J_dagger = np.linalg.pinv(J)

        # Compute null-space projector: I - J⁺J
        null_space_projector = np.eye(self.dof) - J_dagger @ J

        # Compute joint position error
        q_error = arm_joint_pos - self.null_space_joint_target

        # Compute null-space torque with PD control
        tau_n = null_space_projector @ (-self.null_space_kp * q_error - self.null_space_kd * arm_joint_vel)

        return tau_n

    def gravity_compensation(self, arm_joint_pos: np.ndarray, arm_joint_vel: np.ndarray) -> np.ndarray:
        """
        Compute gravity compensation torque using recursive Newton-Euler algorithm.

        This method calculates the gravitational loads on each joint for
        static balancing of the manipulator.

        Args:
            arm_joint_pos (np.ndarray): Current joint positions [rad].
            arm_joint_vel (np.ndarray): Current joint velocities [rad/s].

        Returns:
            np.ndarray: Gravity compensation torque [N·m].
        """
        # Compute gravity torque using RNEA (zero acceleration condition)
        self.tau_g = pin.rnea(self.pin_model, self.pin_data, arm_joint_pos, arm_joint_vel, np.zeros_like(arm_joint_vel))

        # Apply scaling factor (can be used for partial compensation)
        self.tau_g *= self.gravity_comp_modifier

        return self.tau_g

    def friction_compensation(self, arm_joint_vel: np.ndarray) -> np.ndarray:
        """
        Compute friction compensation torque with dithering strategy.

        This method implements a stiction compensation technique that applies
        alternating positive/negative torques to break static friction when
        joint velocities are below a threshold.

        Args:
            arm_joint_vel (np.ndarray): Current joint velocities [rad/s].

        Returns:
            np.ndarray: Friction compensation torque [N·m].
        """
        # Initialize stiction compensation torque
        tau_ss = np.zeros(self.dof)

        # Apply compensation to each joint individually
        for i in range(self.dof):
            # Check if velocity is below threshold for stiction compensation
            if abs(arm_joint_vel[i]) < self.stiction_comp_enable_speed:
                # Apply alternating torque based on dither flag
                if self.stiction_dither_flag[i]:
                    tau_ss[i] += self.stiction_comp_gain * abs(self.tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain * abs(self.tau_g[i])

                # Toggle dither flag for next iteration
                self.stiction_dither_flag[i] = ~self.stiction_dither_flag[i]

        return tau_ss


if __name__ == "__main__":
    """
    Main execution entry point for the gravity compensation environment.

    Initializes the environment with a specific robot model and runs
    the gravity compensation simulation.
    """
    # Initialize environment with leader robot model
    agent = GravityCompensationEnv(
        str(
            (
                Path(__file__).resolve().parent
                / ".."
                / "station"
                / "flying_hand"
                / "description"
                / "leader"
                / "leader.xml"
            ).resolve()
        )
    )

    # Run the simulation
    agent.run()
