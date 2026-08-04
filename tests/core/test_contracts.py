from __future__ import annotations

import numpy as np
import pytest

from acetele.core import (
    EndEffectorPose,
    JointCommand,
    JointEffortCommand,
    JointState,
    JointUnit,
    SensorState,
)
from acetele.specification import Backend


def test_backend_is_a_specification_not_a_domain_contract():
    import acetele.core as core

    assert not hasattr(core, "Backend")
    assert Backend.PHYSICAL.value == "physical"


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


def test_joint_effort_command_owns_nm_vector_and_rejects_nonfinite_values():
    source = np.array([0.1, -0.2])
    command = JointEffortCommand(("joint_1", "joint_2"), source, 1, 2, 0)
    source[0] = 9.0

    assert command.efforts_nm.tolist() == [0.1, -0.2]
    with pytest.raises(ValueError):
        command.efforts_nm[0] = 0.0
    with pytest.raises(ValueError, match="finite"):
        JointEffortCommand(("joint_1",), [float("nan")], 1, 2, 0)


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


def test_end_effector_pose_normalizes_and_owns_vectors():
    position = np.array([1.0, 2.0, 3.0])
    pose = EndEffectorPose(1, "base", position, [0.0, 0.0, 0.0, -2.0])
    position[0] = 9.0

    assert pose.position_m.tolist() == [1.0, 2.0, 3.0]
    assert pose.quaternion_xyzw.tolist() == [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        pose.position_m[0] = 0.0


@pytest.mark.parametrize(
    "arguments",
    (
        (-1, "base", [0.0] * 3, [0.0, 0.0, 0.0, 1.0]),
        (0, "", [0.0] * 3, [0.0, 0.0, 0.0, 1.0]),
        (0, "base", [0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        (0, "base", [0.0] * 3, [0.0, 0.0, 0.0, 0.0]),
    ),
)
def test_end_effector_pose_rejects_invalid_contract(arguments):
    with pytest.raises(ValueError):
        EndEffectorPose(*arguments)
