import os

os.environ["MUJOCO_GL"] = "glfw"

import time
from typing import Dict

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from OpenGL import GL

from neutele.core.integrate import TeleCore
from neutele.simulation.policy_interface import Policy


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
        self.decimation = 2  # Number of simulation steps per control step
        self.track_policy = Policy("weight/track2.onnx")
        # self.hover_policy = Policy("weight/hover.onnx")
        self.rotor_direction = np.array([-1.0, 1.0, -1.0, 1.0])  # Direction mapping for rotors

        # Observation data structure
        self.obs_data = {
            "relative_position": np.zeros((2, 3)),  # Position history
            "orientation": np.zeros((2, 4)),  # Orientation history (quaternions)
            "lin_vel_b": np.zeros(3),  # Linear velocity in body frame
            "ang_vel_b": np.zeros(3),  # Angular velocity in body frame
            "servo_position": np.zeros(5),  # Servo joint positions
            "servo_velocity": np.zeros((3, 5)),  # Servo joint velocities
            "last_action": np.zeros(4),  # Previous control action
            "command": np.zeros(4),  # Teleoperation command
        }
        self.command = np.zeros(9)  # Current teleoperation command

        # Initialize teleoperation core
        self.tele_core = TeleCore()

        # Setup ZMQ communication
        # self.zmq_server = PublisherServer()
        # self.zmq_client = subprocess.Popen([sys.executable, "teleoperation_client.py"])

        # Camera and rendering setup
        self.resolution = (1280, 720)
        self.cameras: Dict[str, dict] = {
            "front_camera": {},
            "wrist_camera": {},
        }
        # self.init_glfw()
        # self.init_camera()

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

        # Print OpenGL information for debugging
        print("Vendor:", GL.glGetString(GL.GL_VENDOR).decode())
        print("Renderer:", GL.glGetString(GL.GL_RENDERER).decode())
        print("OpenGL Version:", GL.glGetString(GL.GL_VERSION).decode())

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

    def get_data(self) -> Dict[str, np.ndarray]:
        """Extract and format sensor data from simulation.

        Returns:
            Dictionary containing all relevant sensor data
        """
        return {
            "relative_position": self.mj_data.sensordata[:3],
            "orientation": self.mj_data.sensordata[3:7],
            "lin_vel_b": self.mj_data.sensordata[7:10],
            "ang_vel_b": self.mj_data.sensordata[10:13],
            "servo_position": self.mj_data.qpos[11:16],  # base(7) + rotors(4) + [servos(5)]
            "servo_velocity": self.mj_data.qvel[10:15],  # base(6) + rotors(4) + [servos(5)]
            "rotor_velocity": np.clip(self.obs_data["last_action"] * 1000, 0, 1000) * self.rotor_direction,
            "servo_torque": self.mj_data.qfrc_actuator[4:9],  # rotor(4) + [servos(5)]
            "command": self.command,
        }

    def get_command(self) -> np.ndarray:
        """Get teleoperation command from teleoperation core.

        Returns:
            Formatted command array for control
        """
        command = self.tele_core.act()
        self.command[:7] = command[:7]  # joint positions and XY position commands
        self.command[7] = 0.75 * command[9]  # Z position command (scaled)
        self.command[8] = 0.5 * (command[7] - command[8])  # Yaw command from differential input
        return self.command

    def control(self):
        """Execute control loop: process observations and compute control actions."""
        # Update observation data
        data = self.get_data()
        self.obs_data["relative_position"] = np.roll(self.obs_data["relative_position"], shift=-1, axis=0)
        self.obs_data["relative_position"][-1] = data["relative_position"]
        self.obs_data["orientation"] = np.roll(self.obs_data["orientation"], shift=-1, axis=0)
        self.obs_data["orientation"][-1] = data["orientation"]
        self.obs_data["lin_vel_b"] = data["lin_vel_b"]
        self.obs_data["ang_vel_b"] = data["ang_vel_b"]
        self.obs_data["servo_position"] = data["servo_position"]
        self.obs_data["servo_velocity"] = np.roll(self.obs_data["servo_velocity"], shift=-1, axis=0)
        self.obs_data["servo_velocity"][-1] = data["servo_velocity"]

        # Get policy action
        command = self.get_command()
        arm_command = command[:5]
        joystick_command = command[5:9]
        # is_hover_command = np.all(np.abs(joystick_command) <= 0.005)
        is_hover_command = False
        if is_hover_command:
            self.obs_data["command"] = np.concatenate(
                [self.obs_data["relative_position"][-1], self.obs_data["orientation"][-1]]
            )
            obs = np.concatenate([v.ravel() for v in self.obs_data.values()])
            rotor_action = self.hover_policy.action(obs)[0]
        else:
            self.obs_data["command"] = joystick_command
            obs = np.concatenate([v.ravel() for v in self.obs_data.values()])
            rotor_action = self.track_policy.action(obs)[0]
        self.obs_data["last_action"] = rotor_action

        # Convert policy output to actuator commands
        rotor_velocity_action = np.clip(rotor_action * 1000, 0, 1000) * self.rotor_direction
        rotor_thrust_action = rotor_velocity_action**2 * 8.54858e-06  # Thrust coefficient
        # servo_action = np.array([0.7854, 0.7854, 0.0, 0.0, 0.0])  # Fixed servo positions
        servo_action = arm_command

        # Combine rotor and servo actions
        action = np.concatenate([rotor_velocity_action / 50, rotor_thrust_action, servo_action])
        # action = np.concatenate([np.zeros(4), rotor_thrust_action, servo_action])
        self.mj_data.ctrl = action

    def run(self):
        frame_count = 0
        last_time = time.perf_counter()
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running():
                self.control()
                mujoco.mj_step(self.mj_model, self.mj_data)
                viewer.sync()
                frame_count += 1
                now = time.perf_counter()
                if now - last_time >= 5.0:
                    fps = frame_count / (now - last_time)
                    print(f"[Debug] 当前帧率: {fps:.1f} Hz")
                    frame_count = 0
                    last_time = now

    # def run(self):
    #     """Main simulation execution loop with performance monitoring."""
    #     frame_count = 0
    #     last_time = time.time()
    #
    #     # Performance counters
    #     control_time = 0
    #     simulation_time = 0
    #     rendering_time = 0
    #     comm_time = 0
    #
    #     # Main simulation loop
    #     while self.zmq_client.poll() is None:
    #         # 1. Control step timing
    #         control_start = time.time()
    #         self.control()
    #         control_time += time.time() - control_start
    #
    #         # 2. Simulation step timing
    #         sim_start = time.time()
    #         for _ in range(self.decimation):
    #             mujoco.mj_step(self.mj_model, self.mj_data)
    #         simulation_time += time.time() - sim_start
    #
    #         # 3. Rendering timing
    #         render_start = time.time()
    #         message = {
    #             "Image": self.get_image(),  # Use simplest image capture
    #             "Data": self.get_data(),
    #         }
    #         rendering_time += time.time() - render_start
    #
    #         # 4. Communication timing
    #         comm_start = time.time()
    #         self.zmq_server.send(message)
    #         comm_time += time.time() - comm_start
    #
    #         # Performance reporting
    #         frame_count += 1
    #         current_time = time.time()
    #         if frame_count >= 50:
    #             total_time = current_time - last_time
    #             fps = frame_count / total_time
    #             # print(f"[Simulator] Loop FPS: {fps:.2f}")
    #             # print(f"  Control: {control_time / frame_count * 1000:.1f}ms/frame")
    #             # print(f"  Simulation: {simulation_time / frame_count * 1000:.1f}ms/frame")
    #             # print(f"  Rendering: {rendering_time / frame_count * 1000:.1f}ms/frame")
    #             # print(f"  Communication: {comm_time / frame_count * 1000:.1f}ms/frame")
    #
    #             # Reset counters
    #             frame_count = 0
    #             last_time = current_time
    #             control_time = simulation_time = rendering_time = comm_time = 0

    def close(self):
        """Cleanup resources and close connections."""
        self.tele_core.close()
        self.zmq_server.close()


if __name__ == "__main__":
    # Initialize and run simulation
    agent = MujocoBase("G:\\NEU_Tele\\neutele\\station\\flying_hand\\urdf\\x500_arm\\x500_arm.xml")
    try:
        agent.run()
    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        agent.close()
