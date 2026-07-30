from __future__ import annotations

import numpy as np
import pytest

from acetele.core import JointCommand, JointState, JointUnit, SensorState


def test_joint_state_owns_read_only_vectors():
    positions = np.array([0.1, 0.2])
    state = JointState(
        names=("joint_1", "joint_2"),
        positions=positions,
        velocities=[0.0, 0.0],
        efforts=[1.0, 2.0],
        timestamp_ns=1,
        sequence=0,
    )

    positions[0] = 9.0

    assert state.positions.tolist() == [0.1, 0.2]
    with pytest.raises(ValueError):
        state.positions[0] = 1.0


def test_joint_command_rejects_invalid_deadline_and_limits():
    with pytest.raises(ValueError, match="deadline"):
        JointCommand(("joint_1",), [0.0], 2, 1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        JointCommand(
            ("joint_1",),
            [0.0],
            1,
            2,
            0,
            velocity_limits=[-1.0],
        )
    with pytest.raises(ValueError, match="non-negative"):
        JointCommand(
            ("joint_1",),
            [0.0],
            1,
            2,
            0,
            effort_limits=[-0.1],
        )


def test_joint_contract_rejects_duplicate_names_and_wrong_unit():
    with pytest.raises(ValueError, match="unique"):
        JointState(("joint", "joint"), [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0, 0)
    with pytest.raises(ValueError, match="JointUnit"):
        JointCommand(("joint",), [0.0], 0, 1, 0, unit="radian")

    command = JointCommand(("gripper",), [0.5], 0, 1, 0, unit=JointUnit.NORMALIZED)
    assert command.unit == JointUnit.NORMALIZED


def test_sensor_state_recursively_owns_and_freezes_mapping_values():
    source = {
        "servo": {
            "firmware_version": 316,
            "temperatures": [30.0, 31.0],
        }
    }
    state = SensorState("bus", source, 1, 0)

    source["servo"]["firmware_version"] = 999
    source["servo"]["temperatures"][0] = 100.0

    assert state.values["servo"]["firmware_version"] == 316
    assert state.values["servo"]["temperatures"].tolist() == [30.0, 31.0]
    with pytest.raises(TypeError):
        state.values["servo"]["firmware_version"] = 999
    with pytest.raises(ValueError):
        state.values["servo"]["temperatures"][0] = 100.0
