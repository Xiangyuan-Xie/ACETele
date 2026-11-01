"""Tests EAA physical presets and their packaged model resources."""

import pathlib
from unittest import mock
from xml.etree import ElementTree

import pytest

from acetele import config
from acetele import runtime
from acetele import specification

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MODEL_NAMES = ("eaa_leader", "eaa_follower")
_JOINT_NAMES = ("Joint_1", "Joint_2", "Joint_3", "Joint_4")


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
def test_eaa_physical_preset_loads_expected_topology(model_name: str) -> None:
    """Checks the preset role, hardware backend, and four-joint topology.

    Args:
        model_name: EAA leader or follower model identifying the preset.
    """
    preset = config.packaged_robot_spec(model_name, "fashionstar_rs485.toml")
    spec = config.load_robot_spec(preset)

    assert spec.model == model_name
    assert spec.backend == specification.Backend.PHYSICAL
    assert len(spec.arms) == 1
    assert len(spec.buses) == 1
    assert spec.buses[0].type == specification.BusType.FASHIONSTAR_RS485

    arm = spec.arms[0]
    assert arm.bus == spec.buses[0].name
    assert arm.tool_frame == "Link_4"
    assert tuple(joint.name for joint in arm.joints) == _JOINT_NAMES
    expected_urdf = (
        _PROJECT_ROOT
        / "acetele/model/robots"
        / model_name
        / "description"
        / f"{model_name}.urdf"
    )
    assert spec.urdf_path is not None
    assert pathlib.Path(spec.urdf_path) == expected_urdf


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
def test_eaa_preflight_never_opens_serial(model_name: str) -> None:
    """Validates the physical preset without constructing a serial connection.

    Args:
        model_name: EAA leader or follower model identifying the preset.
    """
    preset = config.packaged_robot_spec(model_name, "fashionstar_rs485.toml")
    spec = config.load_robot_spec(preset)

    with mock.patch(
        "serial.Serial",
        side_effect=AssertionError("Static preflight must not open serial"),
    ) as serial_constructor:
        robot_runtime = runtime.RobotRuntime(spec)
        assert not robot_runtime.connected
        metadata = robot_runtime.preflight.arms[spec.arms[0].name]
        assert metadata.joint_names == _JOINT_NAMES
        serial_constructor.assert_not_called()


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
def test_eaa_urdf_references_existing_meshes(model_name: str) -> None:
    """Checks that every URDF mesh reference resolves inside the model package.

    Args:
        model_name: EAA leader or follower model identifying the preset.
    """
    preset = config.packaged_robot_spec(model_name, "fashionstar_rs485.toml")
    spec = config.load_robot_spec(preset)
    assert spec.urdf_path is not None
    urdf_path = pathlib.Path(spec.urdf_path)
    model_root = urdf_path.parent.parent.resolve()
    mesh_names = {
        mesh.attrib["filename"]
        for mesh in ElementTree.parse(urdf_path).iter("mesh")
    }

    assert mesh_names, "The EAA model must reference its display meshes"
    for mesh_name in sorted(mesh_names):
        mesh_path = (urdf_path.parent / mesh_name).resolve()
        assert mesh_path.is_relative_to(model_root)
        assert mesh_path.is_file(), f"Missing EAA mesh: {mesh_path}"
        assert mesh_path.stat().st_size > 0, f"Empty EAA mesh: {mesh_path}"
