from __future__ import annotations

import numpy as np
import pytest

from acetele.control import LeaderEffortController
from acetele.core import JointState
from acetele.model import ArmDynamics


class _SyntheticDynamics(ArmDynamics):
    def __init__(self, jacobian, gravity=None, *, limit=10.0):
        self._jacobian = np.asarray(jacobian, dtype=float)
        count = self._jacobian.shape[1]
        self._joint_names = tuple(f"joint_{index}" for index in range(count))
        self._lower = np.full(count, -limit)
        self._upper = np.full(count, limit)
        self._effort = np.full(count, 10.0)
        self._gravity = (
            np.zeros(count) if gravity is None else np.asarray(gravity, dtype=float)
        )

    @property
    def joint_names(self):
        return self._joint_names

    @property
    def lower_limits(self):
        return self._lower.copy()

    @property
    def upper_limits(self):
        return self._upper.copy()

    @property
    def effort_limits(self):
        return self._effort.copy()

    def inverse_dynamics(self, positions, velocities):
        return self._gravity.copy()

    def full_jacobian(self, positions):
        return self._jacobian.copy()


def _state(dynamics, positions, velocities=None):
    positions = np.asarray(positions, dtype=float)
    velocities = np.zeros_like(positions) if velocities is None else velocities
    return JointState(
        dynamics.joint_names,
        positions,
        velocities,
        np.zeros_like(positions),
        0,
        0,
    )


def test_full_rank_four_dof_has_exactly_zero_posture_torque():
    jacobian = np.vstack((np.eye(4), np.zeros((2, 4))))
    dynamics = _SyntheticDynamics(jacobian, gravity=[0.1, 0.2, 0.3, 0.4])
    controller = LeaderEffortController(
        dynamics,
        gravity_compensation=True,
        redundancy_posture=True,
        rest_posture_rad=np.zeros(4),
    )

    result = controller.compute(_state(dynamics, np.ones(4)), now_ns=1)

    assert result.diagnostics.jacobian_rank == 4
    assert result.diagnostics.nullity == 0
    np.testing.assert_array_equal(result.diagnostics.rest_nm, np.zeros(4))
    np.testing.assert_allclose(result.efforts_nm, [0.085, 0.17, 0.255, 0.34])


def test_seven_dof_posture_torque_stays_in_full_pose_null_space():
    jacobian = np.hstack((np.eye(6), np.zeros((6, 1))))
    dynamics = _SyntheticDynamics(jacobian)
    controller = LeaderEffortController(
        dynamics,
        gravity_compensation=False,
        redundancy_posture=True,
        rest_posture_rad=np.zeros(7),
    )

    result = controller.compute(
        _state(dynamics, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        now_ns=1,
    )

    assert result.diagnostics.jacobian_rank == 6
    assert result.diagnostics.nullity == 1
    np.testing.assert_allclose(jacobian @ result.diagnostics.rest_nm, np.zeros(6))
    assert result.diagnostics.rest_nm[-1] == pytest.approx(-0.15)


def test_joint_limit_spring_points_inward_and_effort_output_is_immutable():
    dynamics = _SyntheticDynamics(np.vstack((np.eye(1), np.zeros((5, 1)))), limit=1.0)
    controller = LeaderEffortController(
        dynamics,
        gravity_compensation=False,
        redundancy_posture=False,
        rest_posture_rad=None,
    )

    lower = controller.compute(_state(dynamics, [-0.95]), now_ns=1)
    controller.reset()
    upper = controller.compute(_state(dynamics, [0.95]), now_ns=2)

    assert lower.efforts_nm[0] > 0.0
    assert upper.efforts_nm[0] < 0.0
    with pytest.raises(ValueError):
        upper.efforts_nm[0] = 0.0
