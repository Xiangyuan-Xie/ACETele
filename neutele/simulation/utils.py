import math

import numpy as np
from scipy.spatial.transform import Rotation


class FirstOrderFilter:
    """
    First-order low-pass filter with different time constants for increasing and decreasing signals.

    This filter smooths the input signal with asymmetric response times:
    - Faster response when moving toward the target (accelerating)
    - Slower response when moving away from the target (decelerating)
    or vice versa, depending on the time constant values.
    """

    def __init__(self, time_constant_up: float, time_constant_down: float, initial_state: np.ndarray):
        """
        Initialize the first-order filter.

        Parameters:
        -----------
        time_constant_up : float
            Time constant for when the filter state is increasing (action > current state)
        time_constant_down : float
            Time constant for when the filter state is decreasing (action < current state)
        initial_state : np.ndarray
            Initial state of the filter
        """
        self._time_constant_up = time_constant_up
        self._time_constant_down = time_constant_down
        self._state = initial_state.copy()

    def update(self, action: np.ndarray, dt: float) -> np.ndarray:
        """
        Update the filter state with a new input action.

        Parameters:
        -----------
        action : np.ndarray
            Desired target value (input to the filter)
        dt : float
            Time step since last update

        Returns:
        --------
        np.ndarray
            Current filtered output state
        """
        # Calculate smoothing factors based on time constants
        alpha_up = np.exp(-dt / self._time_constant_up)
        alpha_down = np.exp(-dt / self._time_constant_down)

        # Determine which elements are accelerating (moving toward target)
        accelerating = action > self._state

        # Select appropriate alpha for each element based on direction
        alpha = np.where(accelerating, alpha_up, alpha_down)

        # Apply first-order filter equation: state = alpha*state + (1-alpha)*target
        self._state = alpha * self._state + (1.0 - alpha) * action

        return self._state


def body_to_world_velocity(v_body, quaternion, quat_format="wxyz"):
    """
    Transform velocity vector from body frame to world frame using quaternion rotation.

    This function uses the SciPy Rotation class to perform coordinate transformation
    of velocity vectors from body coordinate system to world coordinate system.

    Parameters
    ----------
    v_body : array_like
        Velocity vector in body frame. Shape can be (3,) for single vector
        or (N, 3) for multiple vectors.
    quaternion : array_like
        Quaternion representing the rotation from body frame to world frame.
        Shape can be (4,) for single rotation or (N, 4) for multiple rotations.
    quat_format : str, optional
        Format of the quaternion input. Options are:
        - 'wxyz': [w, x, y, z] (default)
        - 'xyzw': [x, y, z, w]

    Returns
    -------
    v_world : ndarray
        Velocity vector in world frame. Shape matches input v_body.

    Raises
    ------
    ValueError
        If quat_format is not 'wxyz' or 'xyzw'.

    Examples
    --------
    >>> v_body = [1.0, 0.0, 0.0]
    >>> # 45 degree rotation around Z-axis
    >>> q_wxyz = [np.cos(np.radians(22.5)), 0, 0, np.sin(np.radians(22.5))]
    >>> v_world = body_to_world_velocity(v_body, q_wxyz, 'wxyz')
    >>> print(v_world)
    [0.70710678 0.70710678 0.        ]
    """
    # Convert inputs to numpy arrays
    v_body = np.asarray(v_body, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)

    # Handle single vector case
    is_single_vector = v_body.ndim == 1
    if is_single_vector:
        v_body = v_body.reshape(1, 3)

    # Handle single quaternion case
    is_single_quat = quaternion.ndim == 1
    if is_single_quat:
        quaternion = quaternion.reshape(1, 4)

    # Validate quaternion format and create Rotation object
    if quat_format.lower() == "wxyz":
        R = Rotation.from_quat(quaternion, scalar_first=True)
    elif quat_format.lower() == "xyzw":
        R = Rotation.from_quat(quaternion)
    else:
        raise ValueError(f"Unsupported quat_format: {quat_format}. " "Please use 'wxyz' or 'xyzw'.")

    # Apply rotation: body frame -> world frame
    v_world = R.apply(v_body)

    # Return with same shape as input
    if is_single_vector:
        return v_world[0]
    return v_world


def world_to_body_velocity(v_world, quaternion, quat_format="wxyz"):
    """
    Transform velocity vector from world frame to body frame using quaternion rotation.

    This is the inverse transformation of body_to_world_velocity.

    Parameters
    ----------
    v_world : array_like
        Velocity vector in world frame. Shape can be (3,) for single vector
        or (N, 3) for multiple vectors.
    quaternion : array_like
        Quaternion representing the rotation from body frame to world frame.
        Shape can be (4,) for single rotation or (N, 4) for multiple rotations.
    quat_format : str, optional
        Format of the quaternion input. Options are:
        - 'wxyz': [w, x, y, z] (default)
        - 'xyzw': [x, y, z, w]

    Returns
    -------
    v_body : ndarray
        Velocity vector in body frame. Shape matches input v_world.

    Examples
    --------
    >>> v_world = [0.7071, 0.7071, 0.0]
    >>> q_wxyz = [np.cos(np.radians(22.5)), 0, 0, np.sin(np.radians(22.5))]
    >>> v_body = world_to_body_velocity(v_world, q_wxyz, 'wxyz')
    >>> print(v_body)
    [1. 0. 0.]
    """
    # Convert inputs to numpy arrays
    v_world = np.asarray(v_world, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)

    # Handle single vector case
    is_single_vector = v_world.ndim == 1
    if is_single_vector:
        v_world = v_world.reshape(1, 3)

    # Handle single quaternion case
    is_single_quat = quaternion.ndim == 1
    if is_single_quat:
        quaternion = quaternion.reshape(1, 4)

    # Create Rotation object based on quaternion format
    if quat_format.lower() == "wxyz":
        R = Rotation.from_quat(quaternion, scalar_first=True)
    elif quat_format.lower() == "xyzw":
        R = Rotation.from_quat(quaternion)
    else:
        raise ValueError(f"Unsupported quat_format: {quat_format}. " "Please use 'wxyz' or 'xyzw'.")

    # Apply inverse rotation: world frame -> body frame
    v_body = R.inv().apply(v_world)

    # Return with same shape as input
    if is_single_vector:
        return v_body[0]
    return v_body


def heading_to_world_velocity(v_heading, quaternion, quat_format="wxyz"):
    """
    Transform velocity vector from heading frame to world frame.

    The heading frame is a horizontal frame that only has yaw (heading) rotation
    with respect to the world frame. Its XY plane is always horizontal (parallel
    to the ground plane), and it only rotates around the world Z-axis.

    This function:
    1. Extracts the yaw angle from the input quaternion
    2. Constructs a heading quaternion with only yaw rotation (pitch=0, roll=0)
    3. Applies this heading quaternion to transform from heading frame to world frame

    Heading frame definition:
    - x-axis: Forward direction in the horizontal plane
    - y-axis: Right direction in the horizontal plane
    - z-axis: Up direction (vertical, same as world z-axis)

    Parameters
    ----------
    v_heading : array_like
        Velocity vector in heading frame. Shape can be (3,) for single vector
        or (N, 3) for multiple vectors.
    quaternion : array_like
        Quaternion representing rotation from body frame to world frame.
        Shape can be (4,) for single rotation or (N, 4) for multiple rotations.
    quat_format : str, optional
        Format of the quaternion input. Options are:
        - 'wxyz': [w, x, y, z] (default)
        - 'xyzw': [x, y, z, w]

    Returns
    -------
    v_world : ndarray
        Velocity vector in world frame. Shape matches input v_heading.

    Raises
    ------
    ValueError
        If quat_format is not 'wxyz' or 'xyzw'.

    Examples
    --------
    >>> v_heading = [1.0, 0.0, 0.0]  # 1 m/s forward in heading frame
    >>> # Quaternion for 45-degree rotation around Z-axis
    >>> q_wxyz = [np.cos(np.radians(22.5)), 0, 0, np.sin(np.radians(22.5))]
    >>> v_world = heading_to_world_velocity(v_heading, q_wxyz, 'wxyz')
    >>> print(v_world)
    [0.70710678 0.70710678 0.        ]

    >>> # Quaternion with pitch and roll
    >>> q_with_pitch_roll = Rotation.from_euler('xyz', [0.2, 0.3, 0.8],
    ...                                         degrees=False).as_quat(scalar_first=True)
    >>> v_world2 = heading_to_world_velocity(v_heading, q_with_pitch_roll, 'wxyz')
    >>> print(f"Result with pitch/roll quaternion: {v_world2}")
    >>> print(f"Z-component (should be 0): {v_world2[2]:.6f}")
    """
    # Convert inputs to numpy arrays
    v_heading = np.asarray(v_heading, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)

    # Handle single vector case
    is_single_vector = v_heading.ndim == 1
    if is_single_vector:
        v_heading = v_heading.reshape(1, 3)

    # Handle single quaternion case
    is_single_quat = quaternion.ndim == 1
    if is_single_quat:
        quaternion = quaternion.reshape(1, 4)

    # Check dimension consistency
    n_vectors = v_heading.shape[0]
    n_quats = quaternion.shape[0]

    if n_quats not in (1, n_vectors):
        raise ValueError(f"Number of quaternions ({n_quats}) must be 1 " f"or equal to number of vectors ({n_vectors})")

    # Create Rotation object based on quaternion format
    if quat_format.lower() == "wxyz":
        R = Rotation.from_quat(quaternion, scalar_first=True)
    elif quat_format.lower() == "xyzw":
        R = Rotation.from_quat(quaternion)
    else:
        raise ValueError(f"Unsupported quat_format: {quat_format}. " "Please use 'wxyz' or 'xyzw'.")

    # Extract Euler angles (ZYX convention, yaw is the first angle)
    euler_angles = R.as_euler("ZYX", degrees=False)

    # Extract yaw (heading) angle
    if n_quats == 1:
        yaw = euler_angles[0, 0]  # Single yaw for all vectors
    else:
        yaw = euler_angles[:, 0]  # Different yaw for each vector

    # Create heading rotation with only yaw (pitch=0, roll=0)
    R_heading = Rotation.from_euler("z", yaw, degrees=False)

    # Apply rotation: heading frame -> world frame
    v_world = R_heading.apply(v_heading)

    # Return with same shape as input
    if is_single_vector and n_quats == 1:
        return v_world[0]
    return v_world


def world_to_heading_velocity(v_world, quaternion, quat_format="wxyz"):
    """
    Transform velocity vector from world frame to heading frame.

    This is the inverse transformation of heading_to_world_velocity.

    Parameters
    ----------
    v_world : array_like
        Velocity vector in world frame. Shape can be (3,) for single vector
        or (N, 3) for multiple vectors.
    quaternion : array_like
        Quaternion representing rotation from body frame to world frame.
        Shape can be (4,) for single rotation or (N, 4) for multiple rotations.
    quat_format : str, optional
        Format of the quaternion input. Options are:
        - 'wxyz': [w, x, y, z] (default)
        - 'xyzw': [x, y, z, w]

    Returns
    -------
    v_heading : ndarray
        Velocity vector in heading frame. Shape matches input v_world.

    Raises
    ------
    ValueError
        If quat_format is not 'wxyz' or 'xyzw'.

    Examples
    --------
    >>> v_world = [0.7071, 0.7071, 0.0]
    >>> # Quaternion for 45-degree rotation around Z-axis
    >>> q_wxyz = [np.cos(np.radians(22.5)), 0, 0, np.sin(np.radians(22.5))]
    >>> v_heading = world_to_heading_velocity(v_world, q_wxyz, 'wxyz')
    >>> print(v_heading)
    [1. 0. 0.]
    """
    # Convert inputs to numpy arrays
    v_world = np.asarray(v_world, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)

    # Handle single vector case
    is_single_vector = v_world.ndim == 1
    if is_single_vector:
        v_world = v_world.reshape(1, 3)

    # Handle single quaternion case
    is_single_quat = quaternion.ndim == 1
    if is_single_quat:
        quaternion = quaternion.reshape(1, 4)

    # Check dimension consistency
    n_vectors = v_world.shape[0]
    n_quats = quaternion.shape[0]

    if n_quats not in (1, n_vectors):
        raise ValueError(f"Number of quaternions ({n_quats}) must be 1 " f"or equal to number of vectors ({n_vectors})")

    # Create Rotation object based on quaternion format
    if quat_format.lower() == "wxyz":
        R = Rotation.from_quat(quaternion, scalar_first=True)
    elif quat_format.lower() == "xyzw":
        R = Rotation.from_quat(quaternion)
    else:
        raise ValueError(f"Unsupported quat_format: {quat_format}. " "Please use 'wxyz' or 'xyzw'.")

    # Extract Euler angles (ZYX convention, yaw is the first angle)
    euler_angles = R.as_euler("ZYX", degrees=False)

    # Extract yaw (heading) angle
    if n_quats == 1:
        yaw = euler_angles[0, 0]  # Single yaw for all vectors
    else:
        yaw = euler_angles[:, 0]  # Different yaw for each vector

    # Create heading rotation with only yaw (pitch=0, roll=0)
    R_heading = Rotation.from_euler("z", yaw, degrees=False)

    # Apply inverse rotation: world frame -> heading frame
    v_heading = R_heading.inv().apply(v_world)

    # Return with same shape as input
    if is_single_vector and n_quats == 1:
        return v_heading[0]
    return v_heading


def calculate_slider_position(
    theta_rad: float, r: float = 0.02821, L: float = 0.0343, calibration_offset: float = 0.665
):
    """
    Calculate the position of the slider based on crank angle in a crank-slider mechanism.

    This function computes the displacement of the slider in an inline crank-slider
    mechanism using geometric relationships. The mechanism consists of a rotating crank
    connected to a linear slider via a connecting rod.

    Parameters:
    -----------
    theta_rad : float
        Crank angle in radians. Valid range: 0.665 to 2.388 radians (38.1° to 136.8°).
    r : float, optional
        Crank radius in meters. Default: 0.02821 m (28.21 mm).
    L : float, optional
        Connecting rod length in meters. Default: 0.0343 m (34.3 mm).
    calibration_offset : float, optional
        Calibration offset in radians. Default: 0.665 rad (38.1°).
        This value is subtracted from theta_rad before calculation.

    Returns:
    --------
    float
        Slider position in meters relative to the crank center.

    Notes:
    ------
    - The mechanism follows the standard crank-slider kinematic equation:
      x = r*cosθ + √(L² - (r*sinθ)²)
    - The slider position is measured from the crank center along the slider's path.
    - Ensure input angle stays within 0.665-2.388 radians (38.1°-136.8°) for valid results.
    """
    # Apply calibration offset to adjust the angle
    adjusted_theta = theta_rad + calibration_offset

    # Calculate slider position using crank-slider equation
    # x = crank_radius * cosθ + √(rod_length² - (crank_radius * sinθ)²)
    x = r * math.cos(adjusted_theta) + math.sqrt(L**2 - (r * math.sin(adjusted_theta)) ** 2)

    return np.clip(x - 0.00778, 0, 0.04225)


if __name__ == "__main__":
    quat = [9.9999e-01, 3.3120e-06, 3.7672e-06, -3.6372e-03]
    lin_vel_b_1 = [3.0330e-04, 1.1572e-05, 1.3792e-05]
    lin_vel_w_1 = body_to_world_velocity(lin_vel_b_1, quat)
    lin_vel_w_2 = [3.0338e-04, 9.3649e-06, 1.3789e-05]
    lin_vel_b_2 = world_to_body_velocity(lin_vel_w_2, quat)
    print("linear velocity error:")
    print("body->world:", lin_vel_b_1 - lin_vel_b_2)
    print("world->body:", lin_vel_w_1 - lin_vel_w_2)

    ang_vel_b_1 = [4.1883e-05, -9.9674e-03, -2.3948e-05]
    ang_vel_w_1 = body_to_world_velocity(ang_vel_b_1, quat)
    ang_vel_w_2 = [-3.0624e-05, -9.9674e-03, -2.4014e-05]
    ang_vel_b_2 = world_to_body_velocity(ang_vel_w_1, quat)
    print("angular velocity error:")
    print("body->world:", ang_vel_b_1 - ang_vel_b_2)
    print("world->body:", ang_vel_w_1 - ang_vel_w_2)
