import mujoco
import mujoco.viewer
import numpy as np

from acetele.core.integrate import TeleCore
from acetele.utils.datacollector import DataCollector


class GraspEnv:
    """
    A teleoperation environment for MuJoCo simulation that integrates with
    the TeleCore system for haptic device control.

    This class manages the MuJoCo simulation environment, handles the
    integration with teleoperation hardware, and provides the control
    interface for bilateral teleoperation.
    """

    def __init__(self, model_path: str):
        """
        Initialize the teleoperation environment.

        Args:
            model_path (str): Path to the MuJoCo XML model file defining
                              the robot/leader system configuration.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
            mujoco.FatalError: If the XML model file is invalid or corrupted.
        """
        # Initialize teleoperation core for hardware communication
        self.tele_core = TeleCore()
        # Load MuJoCo model from XML file
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        # Create MuJoCo simulation data structure
        self.mj_data = mujoco.MjData(self.mj_model)
        # Reset simulation data to initial state
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Initialize data collector for teleoperation experiments
        self.data_collector = DataCollector(
            experiment_name="test",
            robot_type="arm",
            sampling_rate=500,
            save_dir="../../data",
        )

    def control(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Control callback function for MuJoCo simulation.

        This method is called at each simulation step to update robot
        control based on teleoperation input. It implements inverse
        dynamics control using position commands from the haptic device.

        Args:
            model (mujoco.MjModel): MuJoCo model object (unused in this
                                     implementation but required by callback).
            data (mujoco.MjData): MuJoCo data object containing current
                                  simulation state.
        """
        # Get leader device position commands
        leader_pos = self.tele_core.act()

        # Prepare arm control commands
        arm_command = np.empty(6)
        # Copy first 4 position commands directly
        arm_command[:4] = leader_pos[:4]
        # Convert gripper command to joint positions
        arm_command[4] = leader_pos[4] * -0.04225
        arm_command[5] = leader_pos[4] * 0.04225

        # Apply control commands to simulation
        data.ctrl = arm_command

        # Extract follower position for data collection
        follower_pos = data.qpos[:5].copy()
        # Normalize gripper position for data collection
        follower_pos[4] = np.clip(follower_pos[4] / -0.04225, 0.0, 1.0)

        # Collect teleoperation data
        self.data_collector.collect(leader_pos[:5], follower_pos)

    def run(self):
        """
        Start the teleoperation simulation.

        This method sets up the control callback and launches the MuJoCo
        viewer for interactive simulation. The simulation runs until the
        viewer is closed by the user.

        Note:
            The method blocks until the viewer window is closed.
        """
        # Register control callback with MuJoCo
        mujoco.set_mjcb_control(self.control)
        # Launch interactive MuJoCo viewer
        mujoco.viewer.launch(self.mj_model, self.mj_data)

    def close(self):
        """
        Clean up resources and shutdown teleoperation system.

        This method should be called to properly close hardware connections
        and release resources when the simulation is finished.
        """
        # Close teleoperation hardware connections
        self.tele_core.close()
        # Save collected teleoperation data
        self.data_collector.save()


if __name__ == "__main__":
    """
    Main execution entry point for the teleoperation environment.

    This script initializes the teleoperation system with a specific
    robot model and runs the simulation with error handling for
    graceful shutdown.
    """
    # Initialize teleoperation environment with robot model
    agent = GraspEnv("../station/flying_hand/description/follower/follower.xml")

    try:
        # Start the teleoperation simulation
        agent.run()
    except Exception as e:
        # Handle and report runtime errors
        print(f"Simulation error: {e}")
    finally:
        # Ensure proper cleanup on exit
        agent.close()
