import enum
import os
from typing import Dict

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from acetele.simulation.policy_interface import Policy
from acetele.simulation.utils import FirstOrderFilter
from acetele.utils.dataloader import DataLoader

os.environ["MUJOCO_GL"] = "glfw"


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

        # Camera and rendering setup
        self.resolution = (1280, 720)
        self.cameras: Dict[str, dict] = {
            "external_camera": {},
            "front_camera": {},
            "wrist_camera": {},
        }
        # self.init_glfw()
        # self.init_camera()

        # Simulation parameters
        self.decimation = 5  # Number of simulation steps per control step
        self.policy = Policy("weight/policy.onnx")  # Policy for control
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

        self.reference_pose = np.array([0.0, 0.0, 1.0, 0.0])
        self.desired_rotor_velocity = np.zeros(4)
        self.base_command = np.zeros(4)
        self.arm_command = np.zeros(5)
        self.status = Status.FLYING
        self.last_lin_vel_h = np.zeros(3)

        self.counter = 0

        self.mj_data.qpos[11:16] = np.array([-1.5708, 3.1416, 0.0, 0.0, 0.0])
        self.mj_data.qvel[10:15] = np.zeros(5)
        self.mj_data.qacc[10:15] = np.zeros(5)

        # self.data_loader = DataLoader(load_path="G://NEU_Tele/data/flying_hand_1767103088.h5")
        self.data_loader = DataLoader(load_path="G://NEU_Tele/data/flying_hand_1767159664.h5")

    def init_glfw(self):
        """Initialize GLFW for offscreen rendering."""
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")

        # Create offscreen window
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(*self.resolution, "Offscreen", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(self.window)

    def init_camera(self):
        """Initialize camera configurations for rendering."""
        for camera_name in self.cameras.keys():
            # Get camera ID from model
            camera_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id == -1:
                print(f"Warning: Camera '{camera_name}' not found, skipping")
                continue

            # Create camera and associated rendering objects
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = camera_id

            self.cameras[camera_name] = {
                "id": camera_id,
                "camera": camera,
                "scene": mujoco.MjvScene(self.mj_model, maxgeom=10000),
                "option": mujoco.MjvOption(),
                "perturb": mujoco.MjvPerturb(),
                "context": mujoco.MjrContext(self.mj_model, mujoco.mjtFontScale.mjFONTSCALE_100.value),
                "viewport": mujoco.MjrRect(0, 0, *self.resolution),
            }

    def get_image(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Capture RGB and depth images from all cameras.

        Returns:
            Dictionary containing RGB and depth images for each camera
        """
        images = {}

        for name, cfg in self.cameras.items():
            cam = cfg["camera"]
            scene = cfg["scene"]
            option = cfg["option"]
            perturb = cfg["perturb"]
            context = cfg["context"]
            viewport = cfg["viewport"]

            # Initialize buffers for image data
            rgb = np.empty((*self.resolution[::-1], 3), dtype=np.uint8)
            depth = np.empty(self.resolution[::-1], dtype=np.float32)

            # Render scene and read pixels
            mujoco.mjv_updateScene(self.mj_model, self.mj_data, option, perturb, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            mujoco.mjr_readPixels(rgb, depth, viewport, context)

            images[f"{name}_rgb"] = rgb
            images[f"{name}_depth"] = depth

        return images

    def update_observation(self):
        """Update the observation data."""
        self.obs_data["orientation"] = self.mj_data.sensordata[3:7]
        self.obs_data["lin_vel_b"] = self.mj_data.sensordata[7:10]
        self.obs_data["ang_vel_b"] = self.mj_data.sensordata[10:13]
        self.obs_data["servo_position"] = np.append(self.mj_data.qpos[11:15], 0.0)
        self.obs_data["servo_velocity"] = np.roll(self.obs_data["servo_velocity"], shift=-1, axis=0)
        self.obs_data["servo_velocity"][-1] = np.append(self.mj_data.qvel[10:14], 0.0)

        replay_action = self.data_loader.act()["leader_position"]
        self.base_command = replay_action[:8]
        self.arm_command = replay_action[8:]

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

                current_quat = self.mj_data.sensordata[3:7]
                root_rot = Rotation.from_quat(current_quat, scalar_first=True)
                _, root_pitch, _ = root_rot.as_euler("XYZ", degrees=False)
                self.arm_command[0] += root_pitch

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
        # for name in self.cameras.keys():
        #     os.makedirs(f"image/{name}", exist_ok=True)
        # for i in tqdm(range(len(self.data_loader) * int(round(0.01 / self.mj_model.opt.timestep)))):
        #     mujoco.mj_step(self.mj_model, self.mj_data)
        #     if self.counter % self.decimation == 0:
        #         image = self.get_image()
        #         for name in self.cameras.keys():
        #             img = image[f"{name}_rgb"]
        #             bgr_img = np.flipud(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #             cv2.imwrite(f"image/{name}/rgb_{i:06d}.png", bgr_img)

    def close(self):
        """Cleanup resources and close connections."""


if __name__ == "__main__":
    # Initialize and run simulation
    agent = MujocoBase("description/x500_arm_v2/x500_arm_v2.xml")
    # agent = MujocoBase("G:\\NEU_Tele\\acetele\\simulation\\description\\x500_arm_v1\\x500_arm_v1.xml")
    # agent = MujocoBase("G:\\NEU_Tele\\acetele\\simulation\\description\\x650_arm_v2\\x650_arm_v2.xml")
    try:
        agent.run()
    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        agent.close()
