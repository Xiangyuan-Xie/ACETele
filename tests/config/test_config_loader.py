import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from acetele.config.config_loader import ConfigLoader
from acetele.config.robot_config import (
    ArmAssemblyConfig,
    ArmConfig,
    FeeTechGripperConfig,
    MockJointConfig,
    O6DexterousHandConfig,
    RobotConfig,
)
from acetele.robot.base_robot import BaseRobot


class RobotForTest(BaseRobot):
    def act(self):
        raise NotImplementedError


@pytest.mark.parametrize("robot_type", ["ace_leader", "ace_follower"])
def test_builtin_robot_configs_default_to_physical_backend(robot_type):
    loader = ConfigLoader(Path(f"acetele/config/{robot_type}.toml"))

    assert loader.get_backend() == "physical"


def test_config_loader_accepts_direct_robot_config(tmp_path):
    config_path = Path(tmp_path, "ace_follower.toml")
    config_path.write_text(
        """
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "ros2"

[arms.single]
port = "/dev/test"
joint_ids = [1]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]

[arms.single.end_effector]
kind = "gripper"
port = "/dev/test"
joint_id = 2
joint_name = "joint_5"
joint_sign = 1
home_pose = 0.0
servo_model = "HL3915"
travel_range_rad = 0.7853981633974483
"""
    )

    loader = ConfigLoader(config_path)
    config = loader.get_robot_config()

    assert config.robot_type == "ace_follower"
    assert config.backend == "physical"
    assert config.runtime == "ros2"
    assert config.arm_assemblies[0].arm.port == "/dev/test"
    assert config.arm_assemblies[0].arm.joint_names == ("joint_1",)
    assert config.arm_assemblies[0].end_effector.joint_ids == (2,)
    assert config.arm_assemblies[0].end_effector.joint_names == ("joint_5",)


def test_config_loader_can_override_runtime_without_changing_device_backend(tmp_path):
    config_path = Path(tmp_path, "default.toml")
    robot_config_path = Path(tmp_path, "ace_leader.toml")
    config_path.write_text('[basic]\nconfig_file = "ace_leader.toml"\n')
    robot_config_path.write_text(
        """
[basic]
robot_type = "ace_leader"
backend = "mock"
runtime = "standalone"

[arms.single]
port = "/dev/test"
joint_ids = [1]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]

[arms.single.end_effector]
kind = "gripper"
port = "/dev/test"
joint_id = 2
joint_name = "joint_5"
joint_sign = 1
home_pose = 0.0
servo_model = "HL3915"
travel_range_rad = 0.7853981633974483
"""
    )

    loader = ConfigLoader(config_path, runtime_override="ros2")

    assert loader.get_robot_type() == "ace_leader"
    assert loader.get_backend() == "mock"
    assert loader.get_runtime() == "ros2"


def test_config_loader_preserves_explicit_compensation_booleans(tmp_path):
    config_path = tmp_path / "boolean_controls.toml"
    config_path.write_text(
        """
[basic]
robot_type = "ace_follower"
backend = "mock"
runtime = "standalone"

[arms.single]
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
enable_gravity_compensation = false
enable_adaptive_compensation = true
"""
    )

    arm = ConfigLoader(config_path).get_robot_config().arm_assemblies[0].arm

    assert not arm.enable_gravity_compensation
    assert arm.enable_adaptive_compensation


@pytest.mark.parametrize(
    "field_name",
    ["enable_gravity_compensation", "enable_adaptive_compensation"],
)
@pytest.mark.parametrize("invalid_value", ['"false"', "0", "1", "0.0", "[]"])
def test_config_loader_rejects_non_boolean_compensation_flags(
    tmp_path,
    field_name,
    invalid_value,
):
    config_path = tmp_path / "invalid_boolean_control.toml"
    config_path.write_text(
        f"""
[basic]
robot_type = "ace_follower"
backend = "mock"
runtime = "standalone"

[arms.single]
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
{field_name} = {invalid_value}
"""
    )

    with pytest.raises(
        ValueError,
        match=rf"arms\.single\.{field_name}.*boolean",
    ):
        ConfigLoader(config_path)


@pytest.mark.parametrize(
    ("original", "replacement", "match"),
    [
        (
            "home_poses = [0.0]",
            "home_poses = [true]",
            r"arms\.single\.home_poses\[0\]",
        ),
        (
            "home_poses = [0.0]",
            'home_poses = ["0.0"]',
            r"arms\.single\.home_poses\[0\]",
        ),
        (
            "control_period = 0.004",
            "control_period = true",
            r"arms\.single\.control_period",
        ),
        (
            "control_period = 0.004",
            'control_period = "0.004"',
            r"arms\.single\.control_period",
        ),
        (
            "home_pose = 0.0",
            "home_pose = true",
            r"arms\.single\.end_effector\.home_pose",
        ),
        (
            "home_pose = 0.0",
            'home_pose = "0.0"',
            r"arms\.single\.end_effector\.home_pose",
        ),
        (
            "travel_range_rad = 0.7853981633974483",
            "travel_range_rad = true",
            r"arms\.single\.end_effector\.travel_range_rad",
        ),
        (
            "travel_range_rad = 0.7853981633974483",
            'travel_range_rad = "0.7853981633974483"',
            r"arms\.single\.end_effector\.travel_range_rad",
        ),
    ],
)
def test_config_loader_rejects_implicitly_convertible_numeric_fields(
    tmp_path,
    original,
    replacement,
    match,
):
    config_text = """
[basic]
robot_type = "ace_follower"
backend = "mock"
runtime = "standalone"

[arms.single]
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
control_period = 0.004

[arms.single.end_effector]
kind = "gripper"
joint_id = 4
joint_name = "joint_5"
joint_sign = 1
home_pose = 0.0
servo_model = "HL3915"
travel_range_rad = 0.7853981633974483
"""
    config_path = tmp_path / "invalid_numeric_field.toml"
    config_path.write_text(config_text.replace(original, replacement))

    with pytest.raises(ValueError, match=match):
        ConfigLoader(config_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("joint_names", "[1]", r"arms\.single\.joint_names\[0\]"),
        ("servo_models", "[3915]", r"arms\.single\.servo_models\[0\]"),
        ("port", "42", r"arms\.single\.port"),
    ],
)
def test_config_loader_rejects_non_string_device_metadata(
    tmp_path,
    field,
    value,
    match,
):
    fields = {
        "joint_names": '["joint_1"]',
        "servo_models": '["HL3915"]',
        "port": '"/dev/test"',
    }
    fields[field] = value
    config_path = tmp_path / "invalid_strings.toml"
    config_path.write_text(
        f"""
[basic]
robot_type = "ace_follower"
backend = "mock"
runtime = "standalone"

[arms.single]
port = {fields['port']}
joint_ids = [0]
joint_names = {fields['joint_names']}
joint_signs = [1]
home_poses = [0.0]
servo_models = {fields['servo_models']}
"""
    )

    with pytest.raises(ValueError, match=match):
        ConfigLoader(config_path)


@pytest.mark.parametrize("invalid_sign", ["1.5", '"1"', "true", "0", "2"])
def test_config_loader_rejects_invalid_arm_joint_signs(tmp_path, invalid_sign):
    config_path = tmp_path / "invalid_arm_sign.toml"
    config_path.write_text(
        f"""
[basic]
robot_type = "ace_follower"
backend = "mock"
runtime = "standalone"

[arms.single]
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [{invalid_sign}]
home_poses = [0.0]
"""
    )

    with pytest.raises(ValueError, match=r"arms\.single\.joint_signs\[0\]"):
        ConfigLoader(config_path)


@pytest.mark.parametrize("invalid_sign", ["1.5", '"1"', "true", "0", "2"])
def test_config_loader_rejects_invalid_gripper_joint_sign(tmp_path, invalid_sign):
    config_path = tmp_path / "invalid_gripper_sign.toml"
    config_path.write_text(
        f"""
[basic]
robot_type = "ace_follower"
backend = "mock"
runtime = "standalone"

[arms.single]
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]

[arms.single.end_effector]
kind = "gripper"
joint_id = 4
joint_name = "joint_5"
joint_sign = {invalid_sign}
home_pose = 0.0
servo_model = "HL3915"
travel_range_rad = 0.7853981633974483
"""
    )

    with pytest.raises(ValueError, match=r"arms\.single\.end_effector\.joint_sign"):
        ConfigLoader(config_path)


@pytest.mark.parametrize("invalid_sign", [1.5, "1", True, 0, 2, np.array(1)])
def test_typed_device_configs_reject_invalid_joint_signs(invalid_sign):
    with pytest.raises(ValueError, match="arm joint_signs"):
        ArmConfig(
            port=None,
            joint_ids=(0,),
            joint_names=("joint_1",),
            joint_signs=(invalid_sign,),
            home_poses=(0.0,),
            servo_models=(),
        )

    with pytest.raises(ValueError, match="gripper joint sign"):
        FeeTechGripperConfig(
            port=None,
            joint_id=4,
            joint_name="joint_5",
            joint_sign=invalid_sign,
            home_pose=0.0,
            servo_model="HL3915",
            travel_range_rad=0.7853981633974483,
        )


def test_typed_device_configs_accept_numpy_integer_joint_signs():
    arm = ArmConfig(
        port=None,
        joint_ids=(0,),
        joint_names=("joint_1",),
        joint_signs=(np.int64(-1),),
        home_poses=(0.0,),
        servo_models=(),
    )
    gripper = FeeTechGripperConfig(
        port=None,
        joint_id=4,
        joint_name="joint_5",
        joint_sign=np.int64(1),
        home_pose=0.0,
        servo_model="HL3915",
        travel_range_rad=0.7853981633974483,
    )

    assert arm.joint_signs == (-1,)
    assert gripper.joint_sign == 1


@pytest.mark.parametrize(
    "field_name",
    ["enable_gravity_compensation", "enable_adaptive_compensation"],
)
@pytest.mark.parametrize(
    "invalid_value",
    ["false", 0, 1, 0.0, [], {}, np.bool_(True)],
)
def test_typed_arm_config_rejects_non_boolean_compensation_flags(
    field_name,
    invalid_value,
):
    kwargs = {field_name: invalid_value}

    with pytest.raises(ValueError, match=rf"arm {field_name}.*boolean"):
        ArmConfig(
            port=None,
            joint_ids=(0,),
            joint_names=("joint_1",),
            joint_signs=(1,),
            home_poses=(0.0,),
            servo_models=(),
            **kwargs,
        )


@pytest.mark.parametrize("value", [False, True])
def test_typed_arm_config_accepts_python_boolean_compensation_flags(value):
    arm = ArmConfig(
        port=None,
        joint_ids=(0,),
        joint_names=("joint_1",),
        joint_signs=(1,),
        home_poses=(0.0,),
        servo_models=(),
        enable_gravity_compensation=value,
        enable_adaptive_compensation=value,
    )

    assert arm.enable_gravity_compensation is value
    assert arm.enable_adaptive_compensation is value


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("home_poses", (True,)),
        ("home_poses", ("0.0",)),
        ("control_period", True),
        ("control_period", "0.004"),
    ],
)
def test_typed_arm_config_rejects_implicitly_convertible_numeric_fields(
    field_name,
    invalid_value,
):
    kwargs = {
        "port": None,
        "joint_ids": (0,),
        "joint_names": ("joint_1",),
        "joint_signs": (1,),
        "home_poses": (0.0,),
        "servo_models": (),
        "control_period": 0.004,
    }
    kwargs[field_name] = invalid_value

    with pytest.raises(ValueError, match=field_name):
        ArmConfig(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("home_pose", True),
        ("home_pose", "0.0"),
        ("travel_range_rad", True),
        ("travel_range_rad", "0.5"),
    ],
)
def test_typed_gripper_config_rejects_implicitly_convertible_numeric_fields(
    field_name,
    invalid_value,
):
    kwargs = {
        "port": None,
        "joint_id": 4,
        "joint_name": "joint_5",
        "joint_sign": 1,
        "home_pose": 0.0,
        "servo_model": "HL3915",
        "travel_range_rad": 0.5,
    }
    kwargs[field_name] = invalid_value

    with pytest.raises(ValueError, match=field_name):
        FeeTechGripperConfig(**kwargs)


@pytest.mark.parametrize(
    "invalid_value",
    [True, "0.0", np.array(0.0)],
)
@pytest.mark.parametrize(
    "field_name",
    ["initial_position", "lower_limit", "upper_limit", "max_velocity"],
)
def test_mock_joint_config_rejects_implicitly_convertible_numeric_fields(
    field_name,
    invalid_value,
):
    kwargs = {
        "name": "joint_1",
        "joint_id": 0,
        "initial_position": 0.0,
        "lower_limit": -1.0,
        "upper_limit": 1.0,
        "max_velocity": 2.0,
    }
    kwargs[field_name] = invalid_value

    with pytest.raises(ValueError, match=field_name):
        MockJointConfig(**kwargs)


def test_typed_configs_copy_mutable_sequence_inputs():
    joint_names = ["joint_1"]
    home_poses = [0.25]
    servo_models = ["HL3915"]
    arm = ArmConfig(
        port=None,
        joint_ids=[0],
        joint_names=joint_names,
        joint_signs=[1],
        home_poses=home_poses,
        servo_models=servo_models,
    )
    assembly = ArmAssemblyConfig("single", arm)
    arm_assemblies = [assembly]
    robot = RobotConfig(
        robot_type="ace_follower",
        backend="mock",
        runtime="standalone",
        arm_assemblies=arm_assemblies,
    )

    joint_names[0] = "changed"
    home_poses[0] = 9.0
    servo_models[0] = "changed"
    arm_assemblies.clear()

    assert arm.joint_names == ("joint_1",)
    assert arm.home_poses == (0.25,)
    assert arm.servo_models == ("HL3915",)
    assert robot.arm_assemblies == (assembly,)


@pytest.mark.parametrize("name", [None, 1, b"joint_1", ""])
def test_mock_joint_config_requires_a_non_empty_string_name(name):
    with pytest.raises(ValueError, match="non-empty string"):
        MockJointConfig(name, 0, 0.0, -1.0, 1.0, 2.0)


def test_typed_configs_copy_numpy_sequence_inputs():
    names = np.array(["joint_1"])
    homes = np.array([0.25])
    models = np.array(["HL3915"])
    arm = ArmConfig(
        port=None,
        joint_ids=np.array([0]),
        joint_names=names,
        joint_signs=np.array([1]),
        home_poses=homes,
        servo_models=models,
    )

    names[0] = "changed"
    homes[0] = 9.0
    models[0] = "changed"

    assert arm.joint_ids == (0,)
    assert arm.joint_names == ("joint_1",)
    assert arm.joint_signs == (1,)
    assert arm.home_poses == (0.25,)
    assert arm.servo_models == ("HL3915",)


def test_o6_config_copies_source_joint_id_list_and_derives_names():
    joint_ids = list(range(10, 16))
    hand = O6DexterousHandConfig("left", joint_ids)

    joint_ids.clear()

    assert hand.joint_ids == tuple(range(10, 16))
    assert hand.joint_names[0] == "lh_thumb_cmc_yaw"


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("joint_names", "joint_1"),
        ("home_poses", "0.0"),
        ("servo_models", "HL3915"),
    ],
)
def test_typed_arm_config_rejects_strings_as_sequence_fields(
    field_name,
    field_value,
):
    kwargs = {
        "port": None,
        "joint_ids": (0,),
        "joint_names": ("joint_1",),
        "joint_signs": (1,),
        "home_poses": (0.0,),
        "servo_models": (),
    }
    kwargs[field_name] = field_value

    with pytest.raises(ValueError, match="must be a sequence"):
        ArmConfig(**kwargs)


def test_o6_and_robot_configs_reject_strings_as_sequence_fields():
    with pytest.raises(ValueError, match="joint_ids"):
        O6DexterousHandConfig("left", "lh_thumb_cmc_yaw")
    with pytest.raises(ValueError, match="must be a sequence"):
        RobotConfig("ace_follower", "mock", "standalone", "single")


def test_typed_robot_config_rejects_mutable_nested_config_values():
    class MutableConfig:
        pass

    with pytest.raises(ValueError, match="ArmAssemblyConfig"):
        RobotConfig(
            "ace_follower",
            "mock",
            "standalone",
            [MutableConfig()],
        )


def test_arm_assembly_rejects_end_effector_without_joint_metadata():
    @dataclass(frozen=True)
    class MissingJointMetadata:
        model: str = "test"

    arm = ArmConfig(
        port=None,
        joint_ids=(0,),
        joint_names=("joint_1",),
        joint_signs=(1,),
        home_poses=(0.0,),
        servo_models=(),
    )

    with pytest.raises(ValueError, match="must define joint_names and joint_ids"):
        ArmAssemblyConfig("single", arm, MissingJointMetadata())


def test_arm_assembly_rejects_shallow_frozen_end_effector_config():
    @dataclass(frozen=True)
    class MutableJointMetadata:
        joint_names: tuple[str, ...]
        joint_ids: tuple[int, ...]
        limits: list[float]

    arm = ArmConfig(
        port=None,
        joint_ids=(0,),
        joint_names=("joint_1",),
        joint_signs=(1,),
        home_poses=(0.0,),
        servo_models=(),
    )

    with pytest.raises(ValueError, match="deeply immutable and hashable"):
        ArmAssemblyConfig(
            "single",
            arm,
            MutableJointMetadata(("joint_5",), (4,), [0.0, 1.0]),
        )


def test_config_loader_rejects_legacy_equipment_schema(tmp_path):
    config_path = Path(tmp_path, "legacy.toml")
    config_path.write_text(
        """
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "standalone"

[linker.single]
joint_ids = [0]
"""
    )

    with pytest.raises(ValueError, match="legacy robot configuration"):
        ConfigLoader(config_path)


@pytest.mark.parametrize(
    "mock_field",
    [
        "mock_joints",
        "wrap_public_positions",
        "initial_positions",
        "lower_limits",
        "upper_limits",
        "max_velocities",
    ],
)
def test_config_loader_rejects_mock_only_arm_fields(tmp_path, mock_field):
    config_path = tmp_path / "mock_parameter.toml"
    config_path.write_text(
        f"""
[basic]
robot_type = "ace_follower"
backend = "mock"
runtime = "standalone"

[arms.single]
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
{mock_field} = [0.0]
"""
    )

    with pytest.raises(ValueError, match=rf"mock-only.*arms\.single\.{mock_field}"):
        ConfigLoader(config_path)


@pytest.mark.parametrize(
    "mock_field",
    [
        "mock_joints",
        "wrap_public_positions",
        "initial_positions",
        "lower_limits",
        "upper_limits",
        "max_velocities",
    ],
)
def test_config_loader_rejects_mock_only_gripper_fields(tmp_path, mock_field):
    config_path = tmp_path / "mock_gripper_parameter.toml"
    config_path.write_text(
        f"""
[basic]
robot_type = "ace_follower"
backend = "mock"
runtime = "standalone"

[arms.single]
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]

[arms.single.end_effector]
kind = "gripper"
joint_id = 4
joint_name = "joint_5"
joint_sign = 1
home_pose = 0.0
servo_model = "HL3915"
travel_range_rad = 0.7853981633974483
{mock_field} = [0.0]
"""
    )

    with pytest.raises(
        ValueError,
        match=rf"mock-only.*arms\.single\.end_effector\.{mock_field}",
    ):
        ConfigLoader(config_path)


def test_config_loader_derives_o6_names_without_mock_parameters(tmp_path):
    config_path = tmp_path / "o6.toml"
    config_path.write_text(
        """
[basic]
robot_type = "ace_follower"
backend = "mock"
runtime = "standalone"

[arms.single]
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]

[arms.single.end_effector]
kind = "dexterous_hand"
model = "o6"
side = "left"
joint_ids = [10, 11, 12, 13, 14, 15]
"""
    )

    hand = ConfigLoader(config_path).get_robot_config().arm_assemblies[0].end_effector

    assert isinstance(hand, O6DexterousHandConfig)
    assert hand.joint_ids == (10, 11, 12, 13, 14, 15)
    assert hand.joint_names[0] == "lh_thumb_cmc_yaw"


@pytest.mark.parametrize(
    ("device_body", "match"),
    [
        (
            """
[arms.single]
joint_ids = [0]
joint_signs = [1]
home_poses = [0.0]
""",
            r"arms\.single\.joint_names.*hardware bus addresses",
        ),
        (
            """
[arms.single]
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]

[arms.single.end_effector]
kind = "gripper"
joint_id = 4
joint_sign = 1
home_pose = 0.0
servo_model = "HL3915"
travel_range_rad = 0.7853981633974483
""",
            r"arms\.single\.end_effector\.joint_name.*hardware bus addresses",
        ),
    ],
)
def test_config_loader_requires_explicit_kinematic_joint_names(tmp_path, device_body, match):
    config_path = tmp_path / "missing_joint_name.toml"
    config_path.write_text(
        """
[basic]
robot_type = "ace_follower"
backend = "mock"
runtime = "standalone"
"""
        + device_body
    )

    with pytest.raises(ValueError, match=match):
        ConfigLoader(config_path)


@pytest.mark.parametrize(
    ("joint_ids", "joint_names", "count", "match"),
    [
        ([0], [], 1, "match joint_ids"),
        ([0, 1], ["joint_1", "joint_1"], 2, "unique"),
        ([0], [""], 1, "non-empty string"),
    ],
)
def test_config_loader_rejects_invalid_explicit_arm_joint_names(
    tmp_path,
    joint_ids,
    joint_names,
    count,
    match,
):
    config_path = tmp_path / "invalid_joint_names.toml"
    config_path.write_text(
        f"""
[basic]
robot_type = "ace_follower"
backend = "mock"
runtime = "standalone"

[arms.single]
joint_ids = {joint_ids}
joint_names = {joint_names}
joint_signs = {[1] * count}
home_poses = {[0.0] * count}
"""
    )

    with pytest.raises(ValueError, match=match):
        ConfigLoader(config_path)


@pytest.mark.parametrize(
    "device_body",
    [
        """
[arms.single]
port = "/dev/test"
joint_ids = [0.0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]
""",
        """
[arms.single]
port = "/dev/test"
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]

[arms.single.end_effector]
kind = "gripper"
port = "/dev/test"
joint_id = 4.0
joint_name = "joint_5"
joint_sign = 1
home_pose = 0.0
servo_model = "HL3915"
travel_range_rad = 0.7853981633974483
""",
        """
[arms.single]
port = "/dev/test"
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]

[arms.single.end_effector]
kind = "dexterous_hand"
model = "o6"
side = "left"
joint_ids = [10.0, 11, 12, 13, 14, 15]
""",
    ],
)
def test_config_loader_rejects_non_integer_joint_ids(tmp_path, device_body):
    config_path = tmp_path / "invalid_joint_id.toml"
    config_path.write_text(
        """
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "standalone"
"""
        + device_body
    )

    with pytest.raises(ValueError, match="must be an integer"):
        ConfigLoader(config_path)


def test_ace_follower_pin_model_path_exists_and_loads():
    import pinocchio as pin

    loader = ConfigLoader(Path("acetele/config/ace_follower.toml"))
    robot = RobotForTest(loader)

    model = robot.get_pin_model()

    assert robot._urdf_model_path.endswith("ace_follower/description/ace_follower.urdf")
    assert model.nv == 5
    assert model.getFrameId("link_5") < len(model.frames)
    reduced_model = pin.buildReducedModel(model, [model.getJointId("joint_5")], pin.neutral(model))
    assert reduced_model.nv == 4


@pytest.mark.parametrize("robot_type", ["ace_leader", "ace_follower"])
def test_robot_extracts_ordered_arm_joint_limits_from_urdf(robot_type):
    robot = RobotForTest(ConfigLoader(Path(f"acetele/config/{robot_type}.toml")))

    lower, upper = robot._get_joint_position_limits(
        ["joint_1", "joint_2", "joint_3", "joint_4"]
    )

    np.testing.assert_allclose(lower, np.array([-2.6485, 0.0, -2.6485, -3.1416]))
    np.testing.assert_allclose(upper, np.array([2.6485, 3.4907, 2.6485, 3.1416]))


@pytest.mark.parametrize(
    ("urdf", "match"),
    [
        ("<robot name='test'><joint name='joint_2' type='revolute'/></robot>", "joint_1"),
        ("<robot name='test'><joint name='joint_1' type='revolute'/></robot>", "limit"),
        (
            "<robot name='test'><joint name='joint_1' type='revolute'><limit lower='1' upper='0'/></joint></robot>",
            "lower",
        ),
    ],
)
def test_robot_rejects_missing_or_invalid_urdf_joint_limits(tmp_path, urdf, match):
    urdf_path = tmp_path / "test.urdf"
    urdf_path.write_text(urdf)
    robot = RobotForTest.__new__(RobotForTest)
    robot._urdf_model_path = str(urdf_path)

    with pytest.raises(ValueError, match=match):
        robot._get_joint_position_limits(["joint_1"])


def test_robot_requires_urdf_when_reading_joint_limits():
    robot = RobotForTest.__new__(RobotForTest)
    robot._urdf_model_path = None

    with pytest.raises(RuntimeError, match="URDF"):
        robot._get_joint_position_limits(["joint_1"])


@pytest.mark.parametrize("joint_type", ["fixed", "prismatic"])
def test_robot_rejects_non_angular_configured_urdf_joint(tmp_path, joint_type):
    urdf_path = tmp_path / "test.urdf"
    urdf_path.write_text(
        f"""
<robot name="test">
  <link name="base_link"/>
  <link name="link_1"/>
  <joint name="joint_1" type="{joint_type}">
    <parent link="base_link"/>
    <child link="link_1"/>
  </joint>
</robot>
"""
    )
    robot = RobotForTest.__new__(RobotForTest)
    robot._urdf_model_path = str(urdf_path)

    with pytest.raises(ValueError, match="one-DOF angular"):
        robot._validate_urdf_joint_mapping(("joint_1",))


def test_robot_accepts_arm_and_gripper_in_urdf_kinematic_order(tmp_path):
    urdf_path = tmp_path / "test.urdf"
    urdf_path.write_text(
        """
<robot name="test">
  <link name="base_link"/>
  <link name="link_1"/>
  <link name="gripper"/>
  <joint name="arm_joint" type="revolute">
    <parent link="base_link"/>
    <child link="link_1"/>
  </joint>
  <joint name="gripper_joint" type="revolute">
    <parent link="link_1"/>
    <child link="gripper"/>
  </joint>
</robot>
"""
    )
    robot = RobotForTest.__new__(RobotForTest)
    robot._urdf_model_path = str(urdf_path)

    robot._validate_urdf_joint_mapping(("arm_joint",), ("gripper_joint",))

    with pytest.raises(ValueError, match="kinematic order"):
        robot._validate_urdf_joint_mapping(("gripper_joint",), ("arm_joint",))


def test_robot_uses_explicit_toml_joint_names_after_servo_ids_are_remapped(tmp_path):
    config_path = tmp_path / "remapped_follower.toml"
    config_path.write_text(
        """
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "standalone"

[arms.single]
port = "/dev/test"
joint_ids = [10, 11, 12, 13]
joint_names = ["joint_1", "joint_2", "joint_3", "joint_4"]
joint_signs = [1, 1, 1, -1]
home_poses = [-1.57, 3.14, 0.0, 0.0]
servo_models = ["HL3960", "HL3950", "HL3930", "HL3915"]
"""
    )
    config = ConfigLoader(config_path).get_robot_config()
    arm = config.arm_assemblies[0].arm
    robot = RobotForTest(config)

    lower, upper = robot._get_joint_position_limits(arm.joint_names)

    assert arm.joint_ids == (10, 11, 12, 13)
    assert arm.joint_names == ("joint_1", "joint_2", "joint_3", "joint_4")
    np.testing.assert_allclose(lower, np.array([-2.6485, 0.0, -2.6485, -3.1416]))
    np.testing.assert_allclose(upper, np.array([2.6485, 3.4907, 2.6485, 3.1416]))


def test_robot_builds_arm_only_pin_model_from_configured_joint_names():
    robot = RobotForTest(ConfigLoader(Path("acetele/config/ace_follower.toml")))

    model = robot._get_pin_model_for_joint_names(
        ["joint_1", "joint_2", "joint_3", "joint_4"]
    )

    assert model.nv == 4
    assert tuple(model.names[1:]) == ("joint_1", "joint_2", "joint_3", "joint_4")


@pytest.mark.parametrize(
    ("joint_names", "match"),
    [
        (("joint_1", "joint_missing"), "missing"),
        (("joint_1", "joint_1"), "unique"),
        (("joint_2", "joint_1"), "model order"),
    ],
)
def test_robot_rejects_invalid_pin_model_joint_name_layout(joint_names, match):
    robot = RobotForTest(ConfigLoader(Path("acetele/config/ace_follower.toml")))

    with pytest.raises(ValueError, match=match):
        robot._get_pin_model_for_joint_names(joint_names)


@pytest.mark.parametrize("backend", ["physical", "mock"])
@pytest.mark.parametrize("runtime", ["standalone", "ros2"])
def test_robot_name_contains_backend_and_runtime(backend, runtime):
    config = ConfigLoader(Path("acetele/config/ace_follower.toml")).get_robot_config()

    robot = RobotForTest(replace(config, backend=backend, runtime=runtime))

    assert robot.name == f"ace_follower_{backend}_{runtime}"


def test_ace_follower_urdf_is_arm_only_x500_model():
    urdf_path = Path("acetele/robot/ace_follower/description/ace_follower.urdf")
    root = ET.parse(urdf_path).getroot()

    assert root.attrib["name"] == "ace_follower"

    link_names = {link.attrib["name"] for link in root.findall("link")}
    joint_names = {joint.attrib["name"] for joint in root.findall("joint")}
    assert link_names == {"base_link", "link_1", "link_2", "link_3", "link_4", "link_5"}
    assert joint_names == {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5"}

    base_link = root.find("./link[@name='base_link']")
    assert base_link is not None
    assert base_link.find("inertial") is None
    assert base_link.find("visual") is None
    assert base_link.find("collision") is None

    mesh_filenames = {mesh.attrib["filename"] for mesh in root.findall(".//mesh")}
    assert mesh_filenames == {f"meshes/link_{index}.STL" for index in range(1, 6)}
