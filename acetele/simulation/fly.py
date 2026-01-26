import enum
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from acetele.core.integrate import TeleCore
from acetele.simulation.policy_interface import Policy
from acetele.simulation.utils import (
    FirstOrderFilter,
    body_to_world_velocity,
    calculate_slider_position,
    heading_to_world_velocity,
    root_rotation_matrix,
)
from acetele.utils.datacollector import DataCollector


class Status(enum.Enum):
    STANDBY = 0
    TAKING_OFF = 1
    FLYING = 2
    LANDING = 3


class MujocoBase:
    """
    Base class for MuJoCo simulation environment with teleoperation capabilities.

    Handles simulation setup, camera rendering, policy execution, and
    communication with external clients.
    """

    def __init__(self, model_path: str):
        """Initialize the MuJoCo simulation environment.

        Args:
            model_path: Path to the MuJoCo XML model file
        """
        # Load MuJoCo model and data
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Simulation parameters
        self.decimation = 5  # Number of simulation steps per control step
        self.policy = Policy(str(Path(__file__).resolve().parent / "weight" / "test.onnx"))  # Policy for control
        self.rotor_direction = np.array([-1.0, 1.0, -1.0, 1.0])  # Direction mapping for rotors

        # Physics controllers
        self.rotor_filter = FirstOrderFilter(
            time_constant_up=0.0125,
            time_constant_down=0.025,
            initial_state=np.zeros(4),
        )

        # Observation data structure
        self.obs_data = {
            "command": np.zeros(8),  # Base command
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),  # Orientation history (quaternions)
            "lin_vel_b": np.zeros(3),  # Linear velocity in body frame
            "ang_vel_b": np.zeros(3),  # Angular velocity in body frame
            "servo_position": np.zeros(5),  # Servo joint positions
            "servo_velocity": np.zeros((3, 5)),  # Servo joint velocities (history)
            "last_action": np.zeros(4),  # Previous control action
        }

        self.tele_core = TeleCore()
        self.reference_pose = np.array([0.0, 0.0, 1.0, 0.0])
        self.desired_rotor_velocity = np.zeros(4)
        self.base_command = np.zeros(4)
        self.arm_command = np.zeros(5)
        self.status = Status.FLYING
        self.last_lin_vel_h = np.zeros(3)

        self.counter = 0

        pos = self.tele_core.act()[:5]
        self.mj_data.qpos[11:16] = pos
        self.mj_data.qvel[10:15] = np.zeros(5)
        self.mj_data.qacc[10:15] = np.zeros(5)
        # self.tele_core._station.move_position(pos)

        self.data_collector = DataCollector(
            experiment_name="flying_hand", robot_type="x500_arm", sampling_rate=100, save_dir="../../data"
        )

    def update_command(self):
        """Update the command data"""
        command_raw = self.tele_core.act()

        base_command_vel_h = np.empty(4)
        base_command_vel_h[:2] = 0.5 * command_raw[5:7]
        base_command_vel_h[2] = 0.5 * command_raw[9]
        base_command_vel_h[3] = 0.5 * (command_raw[7] - command_raw[8])

        current_pos = self.mj_data.sensordata[:3]
        current_quat = self.mj_data.sensordata[3:7]
        root_rot = Rotation.from_quat(current_quat, scalar_first=True)
        _, _, root_yaw = root_rot.as_euler("XYZ", degrees=False)
        lin_vel_w = body_to_world_velocity(self.obs_data["lin_vel_b"], self.mj_data.sensordata[3:7])
        ang_vel_w = body_to_world_velocity(self.obs_data["ang_vel_b"], self.mj_data.sensordata[3:7])

        base_command_vel_w = np.empty(4)
        base_command_vel_w[:3] = heading_to_world_velocity(base_command_vel_h[:3], current_quat)
        base_command_vel_w[3] = base_command_vel_h[3]

        lin_vel_is_zero = np.abs(base_command_vel_w[:3]) < 1e-6
        ang_vel_is_zero = np.abs(base_command_vel_w[3]) < 1e-6

        if not (
            np.any(np.abs(base_command_vel_h[:3]) > 1e-6)  # 本次命令非0
            and np.any(np.abs(self.last_lin_vel_h) > 1e-6)  # 上次命令非0
            and np.any(np.abs(np.cross(base_command_vel_h[:3], self.last_lin_vel_h)) > 1e-6)  # 本次命令与上次命令共线
        ):
            self.reference_pose[:3] = np.where(lin_vel_is_zero, self.reference_pose[:3], current_pos)
        self.reference_pose[3] = np.where(ang_vel_is_zero, self.reference_pose[3], root_yaw)

        base_command_vel_err_w = np.empty(4)
        base_command_vel_err_w[:3] = base_command_vel_w[:3] - lin_vel_w
        base_command_vel_err_w[3] = base_command_vel_w[3] - ang_vel_w[2]

        pos_err = self.reference_pose[:3] - current_pos
        yaw_err = self.reference_pose[3] - root_yaw
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
        base_command_pos_err_w = np.empty(4)
        base_command_pos_err_w[:3] = np.where(lin_vel_is_zero, pos_err, np.zeros_like(pos_err))
        base_command_pos_err_w[3] = np.where(ang_vel_is_zero, yaw_err, np.zeros_like(yaw_err))

        self.base_command = np.concatenate([base_command_vel_err_w, base_command_pos_err_w])

        leader_pos = command_raw[:5]
        arm_command = np.empty(7)
        arm_command[:4] = leader_pos[:4]
        arm_command[4] = leader_pos[4] * -1.723
        arm_command[5] = -(0.04225 - calculate_slider_position(abs(arm_command[4])))  # Gripper left
        arm_command[6] = -arm_command[5]  # Gripper right
        # arm_command = np.empty(5)
        # arm_command[:4] = leader_pos[:4]
        # arm_command[4] = leader_pos[4] * -1.723
        self.arm_command = arm_command

    def update_observation(self):
        """Update the observation data."""
        # self.obs_data["orientation"] = self.mj_data.sensordata[3:7]
        self.obs_data["orientation"] = root_rotation_matrix(self.mj_data.sensordata[3:7])
        self.obs_data["lin_vel_b"] = self.mj_data.sensordata[7:10]
        self.obs_data["ang_vel_b"] = self.mj_data.sensordata[10:13]
        self.obs_data["servo_position"] = np.append(self.mj_data.qpos[11:15], 0.0)
        self.obs_data["servo_velocity"] = np.roll(self.obs_data["servo_velocity"], shift=-1, axis=0)
        self.obs_data["servo_velocity"][-1] = np.append(self.mj_data.qvel[10:14], 0.0)

        self.update_command()
        self.obs_data["command"] = self.base_command

    def control(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Execute control loop: process observations and compute control actions."""
        if self.status == Status.STANDBY:
            pass
        elif self.status == Status.TAKING_OFF:
            pass
        elif self.status == Status.FLYING:
            if self.counter % self.decimation == 0:
                if self.counter > 0:
                    self.counter = 0

                self.update_observation()
                obs = np.concatenate([v.ravel() for v in self.obs_data.values()])
                rotor_action = self.policy.action(obs)[0]
                self.obs_data["last_action"] = rotor_action
                self.desired_rotor_velocity = np.clip(rotor_action * 1000, 0, 1000)

                self.data_collector.collect(np.concatenate([self.base_command, self.arm_command]), np.zeros(9))

                # current_quat = self.mj_data.sensordata[3:7]
                # root_rot = Rotation.from_quat(current_quat, scalar_first=True)
                # _, root_pitch, _ = root_rot.as_euler("XYZ", degrees=False)
                # self.arm_command[0] += root_pitch

            # Convert policy output to actuator commands
            rotor_thrust_action = (self.mj_data.qvel[6:10] * 10) ** 2 * 8.54858e-06

            self.mj_data.qvel[6:10] = (
                self.rotor_filter.update(self.desired_rotor_velocity, model.opt.timestep) * self.rotor_direction / 10
            )
            mujoco.mj_inverse(self.mj_model, self.mj_data)
            rotor_velocity_action = (
                np.clip(self.mj_data.qfrc_inverse[6:10], -0.018, 0.018)
                - rotor_thrust_action * 0.016 * self.rotor_direction
            )

            self.mj_data.ctrl = np.concatenate([rotor_velocity_action, rotor_thrust_action, self.arm_command])

            self.counter += 1
        elif self.status == Status.LANDING:
            pass
        else:
            raise NotImplementedError

    def run(self):
        """Run the simulation and launch MuJoCo viewer."""
        mujoco.set_mjcb_control(self.control)
        mujoco.viewer.launch(self.mj_model, self.mj_data)

    def close(self):
        """Cleanup resources and close connections."""
        self.tele_core.close()
        # self.data_collector.save()


if __name__ == "__main__":
    # Initialize and run simulation
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH = (
        BASE_DIR / ".." / "station" / "flying_hand" / "description" / "x500_arm_lite" / "x500_arm_lite.xml"
    ).resolve()
    agent = MujocoBase(str(MODEL_PATH))
    try:
        agent.run()
    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        agent.close()
