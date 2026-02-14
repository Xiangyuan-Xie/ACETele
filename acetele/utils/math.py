import math
from typing import List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def quat_mul(q1: Sequence[float], q2: Sequence[float]) -> List[float]:
    """Multiply two quaternions (scalar-first order) and return the result."""
    R1 = Rotation.from_quat(np.asarray(q1, dtype=np.float64).reshape(1, 4), scalar_first=True)
    R2 = Rotation.from_quat(np.asarray(q2, dtype=np.float64).reshape(1, 4), scalar_first=True)
    R = R1 * R2
    return list(R.as_quat(scalar_first=True)[0])


def quat_conjugate(q: Sequence[float]) -> List[float]:
    """Return the conjugate (inverse) of a quaternion (scalar-first)."""
    R = Rotation.from_quat(np.asarray(q, dtype=np.float64).reshape(1, 4), scalar_first=True)
    return list(R.inv().as_quat(scalar_first=True)[0])


def quat_rotate(q: Sequence[float], v: Sequence[float]) -> List[float]:
    """Rotate a 3D vector v by quaternion q (scalar-first)."""
    R = Rotation.from_quat(np.asarray(q, dtype=np.float64).reshape(1, 4), scalar_first=True)
    vec = np.asarray(v, dtype=np.float64).reshape(1, 3)
    res = R.apply(vec)[0]
    return [float(res[0]), float(res[1]), float(res[2])]


def body_to_world_velocity(v_body, quaternion, quat_format: str = "wxyz"):
    """Transform velocities from body frame to world frame using quaternion."""
    v_body = np.asarray(v_body, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    is_single_vector = v_body.ndim == 1
    if is_single_vector:
        v_body = v_body.reshape(1, 3)
    is_single_quat = quaternion.ndim == 1
    if is_single_quat:
        quaternion = quaternion.reshape(1, 4)
    if quat_format.lower() == "wxyz":
        R = Rotation.from_quat(quaternion, scalar_first=True)
    elif quat_format.lower() == "xyzw":
        R = Rotation.from_quat(quaternion)
    else:
        raise ValueError(f"Unsupported quat_format: {quat_format}. Please use 'wxyz' or 'xyzw'.")
    v_world = R.apply(v_body)
    if is_single_vector:
        return v_world[0]
    return v_world


def world_to_body_velocity(v_world, quaternion, quat_format: str = "wxyz"):
    """Transform velocities from world frame to body frame using quaternion."""
    v_world = np.asarray(v_world, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    is_single_vector = v_world.ndim == 1
    if is_single_vector:
        v_world = v_world.reshape(1, 3)
    is_single_quat = quaternion.ndim == 1
    if is_single_quat:
        quaternion = quaternion.reshape(1, 4)
    if quat_format.lower() == "wxyz":
        R = Rotation.from_quat(quaternion, scalar_first=True)
    elif quat_format.lower() == "xyzw":
        R = Rotation.from_quat(quaternion)
    else:
        raise ValueError(f"Unsupported quat_format: {quat_format}. Please use 'wxyz' or 'xyzw'.")
    v_body = R.inv().apply(v_world)
    if is_single_vector:
        return v_body[0]
    return v_body


def heading_to_world_velocity(v_heading, quaternion, quat_format: str = "wxyz"):
    """Transform heading-frame (yaw-only) velocities to world frame using quaternion yaw."""
    v_heading = np.asarray(v_heading, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    is_single_vector = v_heading.ndim == 1
    if is_single_vector:
        v_heading = v_heading.reshape(1, 3)
    is_single_quat = quaternion.ndim == 1
    if is_single_quat:
        quaternion = quaternion.reshape(1, 4)
    n_vectors = v_heading.shape[0]
    n_quats = quaternion.shape[0]
    if n_quats not in (1, n_vectors):
        raise ValueError(f"Number of quaternions ({n_quats}) must be 1 or equal to number of vectors ({n_vectors})")
    if quat_format.lower() == "wxyz":
        R = Rotation.from_quat(quaternion, scalar_first=True)
    elif quat_format.lower() == "xyzw":
        R = Rotation.from_quat(quaternion)
    else:
        raise ValueError(f"Unsupported quat_format: {quat_format}. Please use 'wxyz' or 'xyzw'.")
    euler_angles = R.as_euler("ZYX", degrees=False)
    yaw = euler_angles[0, 0] if n_quats == 1 else euler_angles[:, 0]
    R_heading = Rotation.from_euler("z", yaw, degrees=False)
    v_world = R_heading.apply(v_heading)
    if is_single_vector and n_quats == 1:
        return v_world[0]
    return v_world


def world_to_heading_velocity(v_world, quaternion, quat_format: str = "wxyz"):
    """Transform world-frame velocities to heading frame (yaw-only) using quaternion yaw."""
    v_world = np.asarray(v_world, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    is_single_vector = v_world.ndim == 1
    if is_single_vector:
        v_world = v_world.reshape(1, 3)
    is_single_quat = quaternion.ndim == 1
    if is_single_quat:
        quaternion = quaternion.reshape(1, 4)
    n_vectors = v_world.shape[0]
    n_quats = quaternion.shape[0]
    if n_quats not in (1, n_vectors):
        raise ValueError(f"Number of quaternions ({n_quats}) must be 1 or equal to number of vectors ({n_vectors})")
    if quat_format.lower() == "wxyz":
        R = Rotation.from_quat(quaternion, scalar_first=True)
    elif quat_format.lower() == "xyzw":
        R = Rotation.from_quat(quaternion)
    else:
        raise ValueError(f"Unsupported quat_format: {quat_format}. Please use 'wxyz' or 'xyzw'.")
    euler_angles = R.as_euler("ZYX", degrees=False)
    yaw = euler_angles[0, 0] if n_quats == 1 else euler_angles[:, 0]
    R_heading = Rotation.from_euler("z", yaw, degrees=False)
    v_heading = R_heading.inv().apply(v_world)
    if is_single_vector and n_quats == 1:
        return v_heading[0]
    return v_heading


def calculate_slider_position(
    theta_rad: float, r: float = 0.02821, L: float = 0.0343, calibration_offset: float = 0.665
):
    """Compute gripper slider position from angle with calibration offset."""
    adjusted_theta = theta_rad + calibration_offset
    x = r * math.cos(adjusted_theta) + math.sqrt(L * L - (r * math.sin(adjusted_theta)) ** 2)
    return np.clip(x - 0.00778, 0, 0.04225)


def root_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return flattened 3x3 rotation matrices (row-major) for scalar-first quaternions."""
    if quaternion.ndim == 1:
        quaternion = quaternion.reshape(1, -1)
    rotation = Rotation.from_quat(quaternion, scalar_first=True)
    rotation_matrices = rotation.as_matrix()
    return rotation_matrices.reshape(quaternion.shape[0], 9)
