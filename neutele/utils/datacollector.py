import os
from datetime import datetime
from typing import Sequence

import h5py
import numpy as np


class DataCollector:
    """
    Data collector for teleoperation sessions.

    This class collects and stores teleoperation data in real-time,
    organizing it in a structured HDF5 format for later analysis and replay.
    """

    def __init__(self, experiment_name: str, robot_type: str, sampling_rate: float, save_dir: str):
        """
        Initialize the data collector with experiment configuration.

        Parameters
        ----------
        experiment_name : str
            Human-readable experiment name for identification.
        robot_type : str
            Type or model of the robot being used (e.g., UR5e, Panda).
        sampling_rate : float
            Control or data logging frequency in Hertz (Hz).
        save_dir : str
            Directory path where the HDF5 file will be saved.

        Notes
        -----
        The collector creates a timestamped experiment ID to ensure
        unique filenames for each data collection session.
        """
        # Ensure the save directory exists
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)

        # Generate unique experiment ID with timestamp
        self._experiment_id = f"{experiment_name}_{int(datetime.now().timestamp())}"

        # Initialize internal data buffers
        self._data = {
            "experiment_id": self._experiment_id,
            "robot_type": robot_type,
            "sampling_rate": sampling_rate,
            "teleop": {
                "leader_position": [],  # Buffer for leader positions
                "follower_position": [],  # Buffer for follower positions
            },
        }

    def collect(self, leader_position: Sequence[float], follower_position: Sequence[float]):
        """
        Record one timestep of teleoperation data.

        This method appends the current leader and follower positions
        to internal buffers for later storage.

        Parameters
        ----------
        leader_position : Sequence[float]
            Leader device position (joint or Cartesian) for the current timestep.
        follower_position : Sequence[float]
            Follower robot position (joint or Cartesian) for the current timestep.

        Notes
        -----
        Data is stored in memory as Python lists and converted to numpy arrays
        only during the save operation to minimize performance overhead.
        """
        # Append data to internal buffers with float64 precision
        teleop_data = self._data["teleop"]
        # Type assertion to help the type checker
        assert isinstance(teleop_data, dict)

        teleop_data["leader_position"].append(np.asarray(leader_position, dtype=np.float64))
        teleop_data["follower_position"].append(np.asarray(follower_position, dtype=np.float64))

    def save(self):
        """
        Save all collected data to an HDF5 file.

        This method converts internal buffers to numpy arrays and writes
        them to an HDF5 file with appropriate metadata and structure.

        Notes
        -----
        The HDF5 file is created with the following structure:
        - Root attributes: Experiment metadata
        - /teleop/leader_position: Leader position dataset
        - /teleop/follower_position: Follower position dataset
        """
        # Construct output filename
        filename = os.path.join(self._save_dir, f"{self._experiment_id}.h5")

        # Convert lists to stacked numpy arrays
        leader_pos = np.stack(self._data["teleop"]["leader_position"], axis=0)
        follower_pos = np.stack(self._data["teleop"]["follower_position"], axis=0)

        with h5py.File(filename, "w") as f:
            # ==========================================================
            # Root attributes (experiment-level metadata)
            # ==========================================================
            f.attrs["experiment_id"] = self._data["experiment_id"]
            f.attrs["robot_type"] = self._data["robot_type"]
            f.attrs["sampling_rate"] = self._data["sampling_rate"]

            # ==========================================================
            # /teleop group
            # ==========================================================
            teleop_grp = f.create_group("teleop")
            # Create datasets for leader and follower positions
            teleop_grp.create_dataset("leader_position", data=leader_pos, dtype=np.float64)
            teleop_grp.create_dataset("follower_position", data=follower_pos, dtype=np.float64)

        print(f"HDF5 data saved to: {filename}")
