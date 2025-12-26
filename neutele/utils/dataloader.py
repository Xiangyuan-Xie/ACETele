import os
from typing import Dict, Optional

import h5py
import numpy as np


class DataLoader:
    """
    DataLoader for teleoperation data stored in HDF5 format.

    This class provides an interface to load and iterate through recorded
    teleoperation sessions, including leader and follower position data.
    """

    def __init__(self, load_path: str):
        """
        Initialize the data loader and open the HDF5 file.

        Parameters
        ----------
        load_path : str
            Path to the HDF5 file containing teleoperation data.

        Raises
        ------
        FileNotFoundError
            If the specified HDF5 file does not exist.
        KeyError
            If the HDF5 file does not contain the required '/teleop' group.
        ValueError
            If leader and follower data have mismatched lengths.
        """
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"HDF5 file not found: {load_path}")

        # Initialize cursor for sequential data access
        self._cursor = 0

        # Open HDF5 file in read-only mode
        self._file = h5py.File(load_path, "r")

        # Extract metadata from file attributes
        self.metadata = {
            "experiment_id": self._file.attrs.get("experiment_id"),
            "robot_type": self._file.attrs.get("robot_type"),
            "sampling_rate": self._file.attrs.get("sampling_rate"),
            "created_at": self._file.attrs.get("created_at"),
        }

        # Verify required 'teleop' group exists
        if "teleop" not in self._file:
            raise KeyError("Group '/teleop' not found in HDF5 file.")

        # Access the teleoperation data group
        teleop_grp = self._file["teleop"]

        # Load leader and follower position datasets
        self._leader_pos_ds = teleop_grp["leader_position"]
        self._follower_pos_ds = teleop_grp["follower_position"]

        # Validate data consistency
        if self._leader_pos_ds.shape[0] != self._follower_pos_ds.shape[0]:
            raise ValueError("Leader and follower data length mismatch.")

        # Store total number of timesteps
        self._length = self._leader_pos_ds.shape[0]

    def __len__(self) -> int:
        """
        Return the total number of timesteps in the dataset.

        Returns
        -------
        int
            Number of timesteps (data points) available.
        """
        return self._length

    def reset(self) -> None:
        """Reset the internal read cursor to the beginning of the dataset."""
        self._cursor = 0

    def act(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieve the next timestep of teleoperation data.

        This method advances the internal cursor and returns a dictionary
        containing the leader and follower positions for the current timestep.

        Returns
        -------
        Optional[Dict[str, np.ndarray]]
            A dictionary containing:
            - 'leader_position': numpy array of leader device positions
            - 'follower_position': numpy array of follower robot positions
            - 'index': current timestep index
            Returns None when all data has been read.
        """
        # Check if cursor has reached the end of the dataset
        if self._cursor >= self._length:
            return None

        # Construct data sample for current timestep
        sample = {
            "leader_position": self._leader_pos_ds[self._cursor],
            "follower_position": self._follower_pos_ds[self._cursor],
            "index": self._cursor,
        }

        # Advance to next timestep
        self._cursor += 1
        return sample

    def close(self) -> None:
        """
        Close the underlying HDF5 file.

        This method should be called to properly release file resources
        when the DataLoader is no longer needed.
        """
        self._file.close()
