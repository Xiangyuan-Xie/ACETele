from typing import Sequence

import numpy as np

GRIPPER_ENCODING_SCALE = {
    "ace_leader": 4.0 / np.pi,
    "ace_follower": 2048.0 / (896.0 * np.pi),
}
GRIPPER_DECODING_SCALE = {k: 1.0 / v for k, v in GRIPPER_ENCODING_SCALE.items()}


def decode_normalized_gripper_home_pose(
    home_poses: Sequence[float],
    joint_ids: Sequence[int],
    gripper_id: int,
    gripper_type: str,
) -> np.ndarray:
    home_poses_array = np.asarray(home_poses, dtype=float).copy()
    if gripper_id < 0:
        return home_poses_array

    joint_ids_array = np.asarray(joint_ids)
    gripper_indices = np.where(joint_ids_array == gripper_id)[0]
    if len(gripper_indices) != 1:
        raise ValueError(f"gripper_id {gripper_id} must appear exactly once in joint_ids.")

    gripper_index = int(gripper_indices[0])
    gripper_home_pose = home_poses_array[gripper_index]
    if not 0.0 <= gripper_home_pose <= 1.0:
        raise ValueError("Gripper home pose must be between 0.0 and 1.0.")

    home_poses_array[gripper_index] = gripper_home_pose * GRIPPER_DECODING_SCALE[gripper_type]
    return home_poses_array
