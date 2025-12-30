import mujoco
import mujoco.viewer
import numpy as np

from neutele.utils.dataloader import DataLoader


class GraspReplayEnv:
    """
    A teleoperation replay environment for MuJoCo simulation.

    This class manages the MuJoCo simulation environment for replaying
    recorded teleoperation data, allowing visualization and analysis of
    previously captured teleoperation sessions.
    """

    def __init__(self, model_path: str):
        """
        Initialize the teleoperation replay environment.

        Args:
            model_path (str): Path to the MuJoCo XML model file defining
                              the robot/follower system configuration.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
            mujoco.FatalError: If the XML model file is invalid or corrupted.
        """
        # Load MuJoCo model from XML file
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        # Create MuJoCo simulation data structure
        self.mj_data = mujoco.MjData(self.mj_model)
        # Reset simulation data to initial state
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Initialize data loader for replaying recorded teleoperation data
        self.data_loader = DataLoader("G:/NEU_Tele/data/test_1766756169.h5")

    def control(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Control callback function for MuJoCo simulation replay.

        This method is called at each simulation step to replay recorded
        teleoperation data. It reads leader position commands from the
        data loader and applies them to the robot simulation.

        Args:
            model (mujoco.MjModel): MuJoCo model object (unused in this
                                     implementation but required by callback).
            data (mujoco.MjData): MuJoCo data object containing current
                                  simulation state.
        """
        # Get next frame of recorded teleoperation data
        replay_data = self.data_loader.act()

        if replay_data is not None:
            # Extract leader position from recorded data
            leader_pos = replay_data["leader_position"]

            # Prepare arm control commands based on recorded leader position
            arm_command = np.empty(6)
            # Copy first 4 position commands directly
            arm_command[:4] = leader_pos[:4]
            # Convert gripper command to joint positions
            arm_command[4] = leader_pos[4] * -0.04225
            arm_command[5] = leader_pos[4] * 0.04225

            # Apply control commands to simulation
            data.ctrl = arm_command

    def run(self):
        """
        Start the teleoperation replay simulation.

        This method sets up the control callback and launches the MuJoCo
        viewer for visualization of the recorded teleoperation session.
        The simulation runs until the viewer is closed or all data is replayed.

        Note:
            The method blocks until the viewer window is closed.
        """
        # Register control callback with MuJoCo
        mujoco.set_mjcb_control(self.control)
        # Launch interactive MuJoCo viewer
        mujoco.viewer.launch(self.mj_model, self.mj_data)

    def close(self):
        """
        Clean up resources of the replay environment.

        This method can be extended to perform any necessary cleanup
        when the replay simulation is finished.
        """


if __name__ == "__main__":
    """
    Main execution entry point for the teleoperation replay environment.

    This script initializes the teleoperation replay system with a specific
    robot model and runs the simulation with error handling for graceful shutdown.
    """
    # Initialize teleoperation replay environment with follower robot model
    agent = GraspReplayEnv("../station/flying_hand/description/follower/follower.xml")

    try:
        # Start the teleoperation replay simulation
        agent.run()
    except Exception as e:
        # Handle and report runtime errors
        print(f"Simulation error: {e}")
    finally:
        # Ensure proper cleanup on exit
        agent.close()
