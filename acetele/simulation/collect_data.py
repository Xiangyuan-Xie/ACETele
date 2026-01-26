import os
import subprocess
import sys
import time
from typing import Dict

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from OpenGL import GL

from acetele.simulation.fly_no_rotor import MujocoBase
from acetele.simulation.network import PublisherServer

os.environ["MUJOCO_GL"] = "glfw"


class DataCollector(MujocoBase):
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
        super().__init__(model_path)

        # Setup ZMQ communication
        self.zmq_server = PublisherServer()
        self.zmq_client = subprocess.Popen([sys.executable, "teleoperation_client.py"])

        # Camera and rendering setup
        self.resolution = (1280, 720)
        self.cameras: Dict[str, dict] = {
            "front_camera": {},
            "wrist_camera": {},
        }
        self.init_glfw()
        self.init_camera()

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

    def run(self):
        """Main simulation execution loop with performance monitoring."""
        frame_count = 0
        last_time = time.time()

        # Performance counters
        control_time = 0
        simulation_time = 0
        rendering_time = 0
        comm_time = 0

        # Main simulation loop
        while self.zmq_client.poll() is None:
            # 1. Control step timing
            control_start = time.time()
            self.control(self.mj_model, self.mj_data)
            control_time += time.time() - control_start

            # 2. Simulation step timing
            sim_start = time.time()
            # for _ in range(self.decimation):
            mujoco.mj_step(self.mj_model, self.mj_data)
            simulation_time += time.time() - sim_start

            # 3. Rendering timing
            render_start = time.time()
            message = {
                "Image": self.get_image(),  # Use simplest image capture
                "Data": self.get_data(),
            }
            rendering_time += time.time() - render_start

            # 4. Communication timing
            comm_start = time.time()
            self.zmq_server.send(message)
            comm_time += time.time() - comm_start

            # Performance reporting
            frame_count += 1
            current_time = time.time()
            if frame_count >= 50:
                total_time = current_time - last_time
                fps = frame_count / total_time
                print(f"[Simulator] Loop FPS: {fps:.2f}")
                print(f"  Control: {control_time / frame_count * 1000:.1f}ms/frame")
                print(f"  Simulation: {simulation_time / frame_count * 1000:.1f}ms/frame")
                print(f"  Rendering: {rendering_time / frame_count * 1000:.1f}ms/frame")
                print(f"  Communication: {comm_time / frame_count * 1000:.1f}ms/frame")

                # Reset counters
                frame_count = 0
                last_time = current_time
                control_time = simulation_time = rendering_time = comm_time = 0


if __name__ == "__main__":
    # Initialize and run simulation
    agent = DataCollector("G:\\NEU_Tele\\acetele\\simulation\\description\\x500_arm_v2\\x500_arm_v2.xml")
    try:
        agent.run()
    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        agent.close()
