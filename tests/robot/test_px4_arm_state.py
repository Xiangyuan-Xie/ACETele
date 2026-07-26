from types import SimpleNamespace

import numpy as np
import pytest

from acetele.equipment.joint_device import JointDeviceState
from acetele.robot.ace_follower.px4_arm_state import (
    PX4_ARM_JOINT_CAPACITY,
    encode_px4_arm_joint_states,
)


def _state(positions, velocities=None):
    positions_array = np.asarray(positions, dtype=float)
    if velocities is None:
        velocities = np.zeros_like(positions_array)
    return JointDeviceState(
        public_positions=positions_array,
        raw_positions=positions_array,
        velocities=velocities,
        motor_torque_magnitude=np.zeros_like(positions_array),
        motor_torque_signed=np.zeros_like(positions_array),
    )


@pytest.mark.parametrize("joint_count", (4, 7, 14))
def test_encode_px4_arm_joint_states_zero_pads_single_arm(joint_count):
    positions = np.arange(joint_count, dtype=float) / 10.0
    velocities = positions + 1.0

    encoded = encode_px4_arm_joint_states((_state(positions, velocities),))

    assert encoded.joint_count == joint_count
    assert encoded.positions[:joint_count] == tuple(positions)
    assert encoded.velocities[:joint_count] == tuple(velocities)
    assert encoded.positions[joint_count:] == (0.0,) * (PX4_ARM_JOINT_CAPACITY - joint_count)
    assert encoded.velocities[joint_count:] == (0.0,) * (PX4_ARM_JOINT_CAPACITY - joint_count)


def test_encode_px4_arm_joint_states_preserves_dual_arm_assembly_order():
    left = _state(np.arange(7, dtype=float), np.arange(7, dtype=float) + 10.0)
    right = _state(np.arange(7, dtype=float) + 20.0, np.arange(7, dtype=float) + 30.0)

    encoded = encode_px4_arm_joint_states((left, right))

    assert encoded.joint_count == 14
    assert encoded.positions == tuple(left.public_positions) + tuple(right.public_positions)
    assert encoded.velocities == tuple(left.velocities) + tuple(right.velocities)


def test_encode_px4_arm_joint_states_rejects_empty_or_oversized_state():
    with pytest.raises(ValueError, match="at least one"):
        encode_px4_arm_joint_states(())

    with pytest.raises(ValueError, match="between 1 and 14"):
        encode_px4_arm_joint_states((_state(np.zeros(15)),))


@pytest.mark.parametrize(
    "state",
    (
        SimpleNamespace(public_positions=[0.0, 1.0], velocities=[0.0]),
        SimpleNamespace(public_positions=[0.0, np.nan], velocities=[0.0, 0.0]),
        SimpleNamespace(public_positions=[0.0, 1.0], velocities=[0.0, np.inf]),
    ),
)
def test_encode_px4_arm_joint_states_rejects_invalid_values(state):
    with pytest.raises(ValueError):
        encode_px4_arm_joint_states((state,))
