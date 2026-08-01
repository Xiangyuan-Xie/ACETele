import numpy as np

from acetele.model.joint_angle import unwrap_near, wrap_to_pi


def test_wrap_to_pi_uses_half_open_range():
    values = np.array([-3 * np.pi, -np.pi, -np.pi - 0.1, 0.0, np.pi, np.pi + 0.1, 3 * np.pi])

    wrapped = wrap_to_pi(values)

    np.testing.assert_allclose(
        wrapped,
        np.array([-np.pi, -np.pi, np.pi - 0.1, 0.0, -np.pi, -np.pi + 0.1, -np.pi]),
    )


def test_unwrap_near_selects_equivalent_angle_closest_to_reference():
    values = np.array([-np.pi + 0.01, np.pi - 0.01, 0.2])
    references = np.array([np.pi - 0.02, -np.pi + 0.02, 2 * np.pi + 0.1])

    unwrapped = unwrap_near(values, references)

    np.testing.assert_allclose(
        unwrapped,
        np.array([np.pi + 0.01, -np.pi - 0.01, 2 * np.pi + 0.2]),
    )
