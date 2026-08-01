from __future__ import annotations

from pathlib import Path

import pytest

from acetele.config import load_robot_spec
from acetele.specification import Backend, BusType, ParallelGripperSpec

project_root = Path(__file__).resolve().parents[2]


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "robot.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_explicit_bus_and_joint_schema(tmp_path):
    path = _write_config(
        tmp_path,
        """[basic]
model = "ace_follower"
backend = "physical"
urdf_path = "robot.urdf"

[buses.arm]
type = "feetech_packet"
port = "/dev/ttyUSB0"
baudrate = 1000000
cycle_hz = 100
physical_layer = "ttl"
family = "hls"

[arms.single]
bus = "arm"
tool_frame = "link_2"

[[arms.single.joints]]
name = "joint_1"
servo_id = 0
servo_model = "HL3960"
expected_model_number = 123
direction = 1
home_position_rad = 0.0

[arms.single.end_effector]
kind = "parallel_gripper"
bus = "arm"
travel_range_rad = 0.7

[arms.single.end_effector.joint]
name = "joint_2"
servo_id = 1
servo_model = "HL3915"
expected_model_number = 124
direction = -1
home_position_rad = 0.0
""",
    )

    spec = load_robot_spec(path)

    assert spec.backend == Backend.PHYSICAL
    assert spec.buses[0].type == BusType.FEETECH_PACKET
    assert spec.urdf_path == str((tmp_path / "robot.urdf").resolve())
    assert spec.arms[0].joints[0].expected_model_number == 123
    assert spec.arms[0].tool_frame == "link_2"
    assert isinstance(spec.arms[0].end_effector, ParallelGripperSpec)


def test_packaged_hls_ttl_configs_preserve_physical_topology():
    leader = load_robot_spec(
        project_root / "acetele/config/presets/ace_leader/feetech_hls_ttl.toml"
    )
    follower = load_robot_spec(
        project_root / "acetele/config/presets/ace_follower/feetech_hls_ttl.toml"
    )

    assert leader.model == "ace_leader"
    assert leader.buses[0].port == "/dev/ttyUSB0"
    assert leader.buses[0].physical_layer == "ttl"
    assert leader.buses[0].family == "hls"
    assert leader.buses[0].allow_unverified_identity
    assert tuple(joint.servo_id for joint in leader.arms[0].joints) == (0, 1, 2, 3)
    assert tuple(joint.servo_model for joint in leader.arms[0].joints) == (
        "HL3915",
        "HL3915",
        "HL3915",
        "HL3915",
    )
    assert tuple(joint.direction for joint in leader.arms[0].joints) == (1, -1, -1, -1)
    assert leader.arms[0].control.adaptive_position
    assert leader.arms[0].tool_frame == "link_5"
    assert not leader.arms[0].control.gravity_position
    assert isinstance(leader.arms[0].end_effector, ParallelGripperSpec)
    assert leader.arms[0].end_effector.travel_range_rad == pytest.approx(0.7853981633974483)

    assert follower.model == "ace_follower"
    assert follower.buses[0].port == "/dev/ttyCH341USB0"
    assert follower.buses[0].physical_layer == "ttl"
    assert follower.buses[0].family == "hls"
    assert follower.buses[0].allow_unverified_identity
    assert follower.arms[0].tool_frame == "link_5"
    assert tuple(joint.servo_id for joint in follower.arms[0].joints) == (0, 1, 2, 3)
    assert tuple(joint.servo_model for joint in follower.arms[0].joints) == (
        "HL3960",
        "HL3950",
        "HL3930",
        "HL3915",
    )
    assert tuple(joint.direction for joint in follower.arms[0].joints) == (1, 1, 1, -1)
    assert tuple(joint.home_position_rad for joint in follower.arms[0].joints) == (
        -1.57,
        3.14,
        0.0,
        0.0,
    )
    assert isinstance(follower.arms[0].end_effector, ParallelGripperSpec)
    assert follower.arms[0].end_effector.travel_range_rad == pytest.approx(
        1.3744467859455345
    )
    assert follower.arms[0].end_effector.joint.home_position_rad == pytest.approx(
        1.3744467859455345
    )


def test_rejects_legacy_parallel_joint_arrays(tmp_path):
    path = _write_config(
        tmp_path,
        """[basic]
model = "ace_follower"
[buses.arm]
type = "fashionstar_rs485"
port = "/dev/ttyUSB0"
baudrate = 115200
cycle_hz = 50
[arms.single]
bus = "arm"
joint_ids = [1]
""",
    )

    with pytest.raises(ValueError, match="legacy field"):
        load_robot_spec(path)


@pytest.mark.parametrize(
    ("line", "path"),
    (
        ("max_utilisation = 0.5", "buses.arm"),
        ("adaptive_compensaton = true", "arms.single.control"),
    ),
)
def test_rejects_unknown_fields_in_strict_tables(tmp_path, line, path):
    control = ""
    bus_extra = ""
    if path == "buses.arm":
        bus_extra = line
    else:
        control = f"[arms.single.control]\n{line}"
    config = f"""[basic]
model = "ace_follower"

[buses.arm]
type = "feetech_packet"
port = "/dev/ttyUSB0"
baudrate = 1000000
cycle_hz = 100
physical_layer = "ttl"
family = "hls"
{bus_extra}

[arms.single]
bus = "arm"

[[arms.single.joints]]
name = "joint_1"
servo_id = 0
servo_model = "HL3960"
expected_model_number = 123
direction = 1
home_position_rad = 0.0

{control}
"""

    with pytest.raises(ValueError, match=path):
        load_robot_spec(_write_config(tmp_path, config))
