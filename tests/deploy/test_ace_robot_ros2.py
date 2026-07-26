import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from acetele.config.robot_config import (
    ArmAssemblyConfig,
    ArmConfig,
    FeeTechGripperConfig,
    O6DexterousHandConfig,
    RobotConfig,
)
from acetele.equipment.joint_device import JointDeviceState


def make_ros2_robot_config(robot_type, end_effector=None):
    arm = ArmConfig(
        port=None,
        joint_ids=tuple(range(4)),
        joint_names=tuple(f"joint_{index + 1}" for index in range(4)),
        joint_signs=(1, 1, 1, 1),
        home_poses=(0.0, 0.0, 0.0, 0.0),
        servo_models=(),
    )
    return RobotConfig(
        robot_type=robot_type,
        backend="mock",
        runtime="ros2",
        arm_assemblies=(
            ArmAssemblyConfig(
                name="single",
                arm=arm,
                end_effector=end_effector,
            ),
        ),
    )


def make_o6_dexterous_hand_config():
    return O6DexterousHandConfig("left", tuple(range(10, 16)))


def make_feetech_gripper_config():
    return FeeTechGripperConfig(
        port=None,
        joint_id=4,
        joint_name="joint_5",
        joint_sign=-1,
        home_pose=1.0,
        servo_model="HL3915",
        travel_range_rad=1.0,
    )


class FakeRosArm:
    def __init__(self, positions, velocities=None, effort=None, raw_positions=None, signed_effort=None):
        self.public_positions = np.asarray(positions, dtype=float)
        self.raw_positions = (
            self.public_positions.copy() if raw_positions is None else np.asarray(raw_positions, dtype=float)
        )
        self.velocities = (
            np.zeros_like(self.public_positions) if velocities is None else np.asarray(velocities, dtype=float)
        )
        self.motor_torque_magnitude = (
            np.zeros_like(self.public_positions) if effort is None else np.asarray(effort, dtype=float)
        )
        self.motor_torque_signed = (
            self.motor_torque_magnitude.copy() if signed_effort is None else np.asarray(signed_effort, dtype=float)
        )
        self.ids = np.arange(len(self.public_positions))
        self.joint_names = tuple(f"joint_{int(joint_id) + 1}" for joint_id in self.ids)
        self.position_calls = []
        self.position_validation_error = None

    def get_state(self):
        return JointDeviceState(
            public_positions=self.public_positions,
            raw_positions=self.raw_positions,
            velocities=self.velocities,
            motor_torque_magnitude=self.motor_torque_magnitude,
            motor_torque_signed=self.motor_torque_signed,
        )

    def act(self):
        return self.public_positions, self.velocities, self.motor_torque_magnitude

    def set_position(self, positions, **kwargs):
        self.position_calls.append((list(positions), kwargs))

    def validate_position_command(self, positions, **kwargs):
        if self.position_validation_error is not None:
            raise self.position_validation_error


class FakeRosGripper:
    def __init__(self, position=0.0, velocity=0.0, effort=0.0):
        self.position = float(position)
        self.velocity = float(velocity)
        self.effort = float(effort)
        self.position_calls = []
        self.state_reads = 0
        self.ids = np.array([4])
        self.joint_names = ("joint_5",)
        self.position_validation_error = None

    def get_state(self):
        self.state_reads += 1
        return JointDeviceState(
            public_positions=np.array([self.position]),
            raw_positions=np.array([self.position]),
            velocities=np.array([self.velocity]),
            motor_torque_magnitude=np.array([self.effort]),
            motor_torque_signed=np.array([self.effort]),
        )

    def act(self):
        return np.array([self.position]), np.array([self.velocity]), np.array([self.effort])

    def set_position(self, positions, **kwargs):
        self.position_calls.append((np.asarray(positions, dtype=float).tolist(), kwargs))

    def validate_position_command(self, positions, **kwargs):
        if self.position_validation_error is not None:
            raise self.position_validation_error


def install_arm_assembly(robot, arm, end_effector=None):
    robot.arm_assemblies = (
        types.SimpleNamespace(
            arm=arm,
            end_effector=end_effector,
            config=types.SimpleNamespace(end_effector=None),
        ),
    )


def make_joint_device_state(positions, velocities=None, effort=None):
    positions = np.asarray(positions, dtype=float)
    velocities = (
        np.zeros_like(positions)
        if velocities is None
        else np.asarray(velocities, dtype=float)
    )
    effort = (
        np.zeros_like(positions)
        if effort is None
        else np.asarray(effort, dtype=float)
    )
    return JointDeviceState(
        public_positions=positions,
        raw_positions=positions,
        velocities=velocities,
        motor_torque_magnitude=effort,
        motor_torque_signed=effort,
    )


def install_follower_sync(
    robot,
    sync,
    mode,
    status,
    *,
    last_command_ns=None,
    heartbeat_timeout_ns=int(1e9),
):
    robot._sync = sync.FollowerSyncController(heartbeat_timeout_ns)
    robot._sync.mode = mode
    robot._sync.status = status
    robot._sync.last_command_ns = last_command_ns
    robot._heartbeat_lost = False
    robot._latest_arm_command = None
    robot._latest_gripper_command = None
    robot._latest_arm_state = None
    robot._latest_gripper_state = None


def install_leader_devices(robot, arm_positions, gripper_position=None):
    arm = FakeRosArm(arm_positions)
    robot.act = arm.act
    robot.ids = np.array(list(arm.ids) + ([] if gripper_position is None else [4]))
    robot.gripper_id = None if gripper_position is None else 4
    robot.gripper_index = None if gripper_position is None else len(arm.ids)
    gripper = None if gripper_position is None else FakeRosGripper(position=gripper_position)
    install_arm_assembly(robot, arm, gripper)
    return arm, gripper


def install_robot_level_state(robot, positions, velocities=None, effort=None):
    positions = np.asarray(positions, dtype=float)
    velocities = np.zeros_like(positions) if velocities is None else np.asarray(velocities, dtype=float)
    effort = np.zeros_like(positions) if effort is None else np.asarray(effort, dtype=float)
    arm = FakeRosArm(positions, velocities=velocities, effort=effort)
    robot.ids = np.arange(len(positions))
    robot.act = lambda: (positions, velocities, effort)
    robot.get_robot_state = lambda: types.SimpleNamespace(
        public_positions=positions,
        raw_positions=positions,
        velocities=velocities,
        motor_torque_magnitude=effort,
        motor_torque_signed=effort,
    )
    install_arm_assembly(robot, arm)
    return arm


def install_fake_ros_modules(monkeypatch):
    class FakeNode:
        def __init__(self, *_args, **_kwargs):
            pass

        def declare_parameter(self, *_args, **_kwargs):
            pass

        def get_parameter(self, *_args, **_kwargs):
            pass

        def create_publisher(self, *_args, **_kwargs):
            pass

        def create_subscription(self, *_args, **_kwargs):
            pass

        def create_timer(self, *_args, **_kwargs):
            pass

        def get_logger(self, *_args, **_kwargs):
            pass

    rclpy_module = types.ModuleType("rclpy")
    rclpy_module.init = lambda: None
    rclpy_module.spin = lambda _node: None
    rclpy_module.shutdown = lambda: None

    rclpy_node_module = types.ModuleType("rclpy.node")
    rclpy_node_module.Node = FakeNode

    rclpy_qos_module = types.ModuleType("rclpy.qos")
    rclpy_qos_module.QoSProfile = lambda depth: ("qos", depth)
    rclpy_qos_module.qos_profile_sensor_data = ("qos", "sensor_data")

    std_msgs_module = types.ModuleType("std_msgs")
    std_msgs_msg_module = types.ModuleType("std_msgs.msg")

    class FakeString:
        def __init__(self):
            self.data = ""

    std_msgs_msg_module.String = FakeString

    sensor_msgs_module = types.ModuleType("sensor_msgs")
    sensor_msgs_msg_module = types.ModuleType("sensor_msgs.msg")

    class FakeJointState:
        def __init__(self):
            self.header = types.SimpleNamespace(stamp=None)
            self.name = []
            self.position = []
            self.velocity = []
            self.effort = []

    sensor_msgs_msg_module.JointState = FakeJointState

    px4_msgs_module = types.ModuleType("px4_msgs")
    px4_msgs_msg_module = types.ModuleType("px4_msgs.msg")

    class FakeArmJointState:
        MAX_JOINTS = 14

        @classmethod
        def get_fields_and_field_types(cls):
            return {
                "timestamp": "uint64",
                "timestamp_sample": "uint64",
                "sequence": "uint32",
                "joint_count": "uint8",
                "arm_velocity_valid": "boolean",
                "arm_position": "float[14]",
                "arm_velocity": "float[14]",
            }

        def __init__(self):
            self.timestamp = 0
            self.timestamp_sample = 0
            self.sequence = 0
            self.joint_count = 0
            self.arm_velocity_valid = False
            self.arm_position = []
            self.arm_velocity = []

    px4_msgs_msg_module.ArmJointState = FakeArmJointState
    px4_msgs_msg_module.VehicleLandDetected = type("VehicleLandDetected", (), {})

    monkeypatch.setitem(sys.modules, "rclpy", rclpy_module)
    monkeypatch.setitem(sys.modules, "rclpy.node", rclpy_node_module)
    monkeypatch.setitem(sys.modules, "rclpy.qos", rclpy_qos_module)
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs_module)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg_module)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs_module)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg_module)
    monkeypatch.setitem(sys.modules, "px4_msgs", px4_msgs_module)
    monkeypatch.setitem(sys.modules, "px4_msgs.msg", px4_msgs_msg_module)
    return FakeNode


def test_follower_ros2_rejects_stale_px4_arm_joint_state_schema(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))

    class OldArmJointState:
        MAX_JOINTS = 5

        @classmethod
        def get_fields_and_field_types(cls):
            return {
                "timestamp": "uint64",
                "timestamp_sample": "uint64",
                "sequence": "uint32",
                "joint_count": "uint8",
                "arm_position": "float[5]",
                "arm_velocity": "float[5]",
            }

    monkeypatch.setattr(module, "ArmJointState", OldArmJointState)

    with pytest.raises(RuntimeError, match="schema mismatch"):
        module.AceFollowerROS2Robot._validate_px4_arm_joint_state_schema()


@pytest.mark.parametrize(
    ("module_name", "class_name", "robot_base_name", "robot_type"),
    (
        (
            "acetele.robot.ace_follower.ace_follower_ros2",
            "AceFollowerROS2Robot",
            "AceFollowerRobot",
            "ace_follower",
        ),
        (
            "acetele.robot.ace_leader.ace_leader_ros2",
            "AceLeaderROS2Robot",
            "AceLeaderRobot",
            "ace_leader",
        ),
    ),
)
def test_ros2_rejects_unsupported_end_effector_before_resource_construction(
    monkeypatch,
    module_name,
    class_name,
    robot_base_name,
    robot_type,
):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module(module_name))
    joint_robot_module = importlib.import_module("acetele.robot.joint_robot")
    construction_calls = []

    monkeypatch.setattr(
        module.Node,
        "__init__",
        lambda *_args, **_kwargs: construction_calls.append("node"),
    )
    monkeypatch.setattr(
        getattr(module, robot_base_name),
        "__init__",
        lambda *_args, **_kwargs: construction_calls.append("robot"),
    )
    monkeypatch.setattr(
        joint_robot_module,
        "FeeTechDriver",
        lambda *_args, **_kwargs: construction_calls.append("driver"),
    )
    config = make_ros2_robot_config(
        robot_type,
        end_effector=make_o6_dexterous_hand_config(),
    )

    with pytest.raises(RuntimeError, match="supports only a normalized gripper"):
        getattr(module, class_name)(config)

    assert construction_calls == []


@pytest.mark.parametrize(
    ("module_name", "class_name", "robot_base_name", "robot_type"),
    (
        (
            "acetele.robot.ace_follower.ace_follower_ros2",
            "AceFollowerROS2Robot",
            "AceFollowerRobot",
            "ace_follower",
        ),
        (
            "acetele.robot.ace_leader.ace_leader_ros2",
            "AceLeaderROS2Robot",
            "AceLeaderRobot",
            "ace_leader",
        ),
    ),
)
def test_ros2_constructor_failure_closes_robot_and_destroys_node(
    monkeypatch,
    module_name,
    class_name,
    robot_base_name,
    robot_type,
):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module(module_name))
    events = []
    initialization_error = RuntimeError("ROS interface setup failed")
    close_error = RuntimeError("robot close failed")

    monkeypatch.setattr(
        module.Node,
        "__init__",
        lambda *_args, **_kwargs: events.append("node_init"),
    )
    monkeypatch.setattr(
        module.Node,
        "destroy_node",
        lambda _self: events.append("node_destroy"),
        raising=False,
    )
    robot_base = getattr(module, robot_base_name)
    monkeypatch.setattr(
        robot_base,
        "__init__",
        lambda *_args, **_kwargs: events.append("robot_init"),
    )

    def fail_close(_self):
        events.append("robot_close")
        raise close_error

    monkeypatch.setattr(robot_base, "close", fail_close)
    monkeypatch.setattr(
        getattr(module, class_name),
        "_initialize_ros_interfaces",
        lambda _self: (_ for _ in ()).throw(initialization_error),
    )

    with pytest.raises(RuntimeError) as exc_info:
        getattr(module, class_name)(make_ros2_robot_config(robot_type))

    assert exc_info.value is initialization_error
    assert exc_info.value.__cause__ is close_error
    assert events == ["node_init", "robot_init", "robot_close", "node_destroy"]


def test_ace_robot_node_uses_config_path_parameter(monkeypatch, tmp_path):
    FakeNode = install_fake_ros_modules(monkeypatch)
    module = importlib.reload(
        importlib.import_module("acetele.deploy.ace_robot_ros2.ace_robot_ros2.ace_robot_node")
    )
    config_path = tmp_path / "robot.toml"
    config_path.write_text("[basic]\n")
    captured = {}

    class FakeParameter:
        value = str(config_path)

    class FakeParameterNode(FakeNode):
        def declare_parameter(self, name, default):
            captured["declared"] = (name, default)
            return FakeParameter()

        def destroy_node(self):
            captured["parameter_destroyed"] = True

    class FakeRobotNode(FakeParameterNode):
        def close(self):
            captured["closed"] = True

        def destroy_node(self):
            captured["destroyed"] = True

    monkeypatch.setattr(module, "Node", FakeParameterNode)
    monkeypatch.setattr(module, "make_robot", lambda loader: FakeRobotNode())

    class FakeConfigLoader:
        def __init__(self, path, runtime_override=None):
            captured["config_path"] = path
            captured["runtime_override"] = runtime_override

    monkeypatch.setattr(module, "ConfigLoader", FakeConfigLoader)

    module.main()

    assert captured["declared"] == ("config_path", "")
    assert captured["config_path"] == config_path
    assert captured["runtime_override"] == "ros2"
    assert captured["closed"]
    assert captured["destroyed"]


def test_ace_robot_node_attempts_all_cleanup_when_robot_close_fails(monkeypatch):
    FakeNode = install_fake_ros_modules(monkeypatch)
    module = importlib.reload(
        importlib.import_module(
            "acetele.deploy.ace_robot_ros2.ace_robot_ros2.ace_robot_node"
        )
    )
    events = []
    close_error = RuntimeError("robot close failed")

    class ParameterNode(FakeNode):
        def declare_parameter(self, _name, _default):
            return types.SimpleNamespace(value="")

        def destroy_node(self):
            events.append("parameter_destroy")

    class RobotNode(ParameterNode):
        def close(self):
            events.append("robot_close")
            raise close_error

        def destroy_node(self):
            events.append("robot_destroy")

    monkeypatch.setattr(module, "Node", ParameterNode)
    monkeypatch.setattr(module, "ConfigLoader", lambda **_kwargs: object())
    monkeypatch.setattr(module, "make_robot", lambda _loader: RobotNode())
    monkeypatch.setattr(
        module.rclpy,
        "shutdown",
        lambda: events.append("shutdown"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        module.main()

    assert exc_info.value is close_error
    assert events == [
        "parameter_destroy",
        "robot_close",
        "robot_destroy",
        "shutdown",
    ]


def test_ace_robot_node_preserves_non_node_error_and_shuts_down(monkeypatch):
    FakeNode = install_fake_ros_modules(monkeypatch)
    module = importlib.reload(
        importlib.import_module(
            "acetele.deploy.ace_robot_ros2.ace_robot_ros2.ace_robot_node"
        )
    )
    events = []

    class ParameterNode(FakeNode):
        def declare_parameter(self, _name, _default):
            return types.SimpleNamespace(value="")

        def destroy_node(self):
            events.append("parameter_destroy")

    monkeypatch.setattr(module, "Node", ParameterNode)
    monkeypatch.setattr(module, "ConfigLoader", lambda **_kwargs: object())
    monkeypatch.setattr(module, "make_robot", lambda _loader: object())
    monkeypatch.setattr(
        module.rclpy,
        "shutdown",
        lambda: events.append("shutdown"),
    )

    with pytest.raises(ValueError, match="Robot node must be") as exc_info:
        module.main()

    assert exc_info.value.__cause__ is None
    assert events == ["parameter_destroy", "shutdown"]


def test_ace_robot_node_guards_cleanup_attribute_access(monkeypatch):
    FakeNode = install_fake_ros_modules(monkeypatch)
    module = importlib.reload(
        importlib.import_module(
            "acetele.deploy.ace_robot_ros2.ace_robot_ros2.ace_robot_node"
        )
    )
    events = []
    lookup_error = RuntimeError("destroy lookup failed")

    class ParameterNode(FakeNode):
        def declare_parameter(self, _name, _default):
            return types.SimpleNamespace(value="")

        def destroy_node(self):
            events.append("parameter_destroy")

    class NonNodeRobot:
        def close(self):
            events.append("robot_close")

        @property
        def destroy_node(self):
            events.append("robot_destroy_lookup")
            raise lookup_error

    monkeypatch.setattr(module, "Node", ParameterNode)
    monkeypatch.setattr(module, "ConfigLoader", lambda **_kwargs: object())
    monkeypatch.setattr(module, "make_robot", lambda _loader: NonNodeRobot())
    monkeypatch.setattr(
        module.rclpy,
        "shutdown",
        lambda: events.append("shutdown"),
    )

    with pytest.raises(ValueError, match="Robot node must be") as exc_info:
        module.main()

    assert exc_info.value.__cause__ is lookup_error
    assert events == [
        "parameter_destroy",
        "robot_close",
        "robot_destroy_lookup",
        "shutdown",
    ]


def test_ace_robot_ros2_data_files_exist(monkeypatch):
    setup_path = Path("acetele/deploy/ace_robot_ros2/setup.py")
    setup_source = setup_path.read_text()
    captured = {}

    def fake_setup(**kwargs):
        captured.update(kwargs)

    fake_setuptools = types.ModuleType("setuptools")
    fake_setuptools.setup = fake_setup
    fake_setuptools.find_packages = lambda: []
    monkeypatch.setitem(sys.modules, "setuptools", fake_setuptools)

    exec(setup_source, {})

    package_root = setup_path.parent
    missing_files = [
        source
        for _destination, sources in captured["data_files"]
        for source in sources
        if not (package_root / source).is_file()
    ]

    assert missing_files == []


def test_ace_robot_ros2_default_params_use_profile_sync_keys():
    params_path = Path("acetele/deploy/ace_robot_ros2/config/ace_robot_params.yaml")
    params = yaml.safe_load(params_path.read_text())["ace_robot_node"]["ros__parameters"]

    assert params["sync_profile_velocity"] == 2.0
    assert params["sync_profile_acceleration"] == 3.0
    assert "sync_move_step_size" not in params
    assert "sync_move_max_steps" not in params


def test_follower_holds_pose_during_sync_request(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.SYNC_REQUEST,
        sync.FollowerSyncStatus.READY,
    )
    robot._heartbeat_lost = True
    published = {"state": [], "px4": [], "status": []}

    now_ns = [1_000_000_000]

    class FakeClock:
        @property
        def nanoseconds(self):
            return now_ns[0]

        def to_msg(self):
            return f"stamp-{now_ns[0]}"

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeClock())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None, warn=lambda _message: None)
    install_robot_level_state(robot, [1.0, 2.0])
    robot._arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["state"].append(msg))
    robot._px4_arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["px4"].append(msg))
    robot._sync_status_pub = types.SimpleNamespace(publish=lambda msg: published["status"].append(msg))
    robot._warned_invalid_px4_arm_state = False

    msg = types.SimpleNamespace(position=[1.0, 2.0], velocity=[99.0, 99.0])

    module.AceFollowerROS2Robot._arm_command_callback(robot, msg)

    assert robot._latest_arm_command is None
    assert not hasattr(robot, "_latest_arm_command_velocity")
    assert robot._sync.last_command_ns is None
    assert robot._heartbeat_lost

    module.AceFollowerROS2Robot._control_loop(robot)

    assert robot._sync.status == sync.FollowerSyncStatus.READY
    assert robot._latest_arm_state is not None
    assert published == {"state": [], "px4": [], "status": []}

    now_ns[0] = 1_200_000_000
    module.AceFollowerROS2Robot._control_loop(robot)

    assert robot._sync.status == sync.FollowerSyncStatus.READY

    module.AceFollowerROS2Robot._publish_state_loop(robot)

    assert len(published["state"]) == 1
    assert published["state"][0].position == [1.0, 2.0]
    assert published["status"][0].data == "ready"


def test_follower_ignores_commands_until_tracking_mode(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.IDLE,
        sync.FollowerSyncStatus.IDLE,
    )

    now_ns = [1_000_000_000]

    class FakeNow:
        @property
        def nanoseconds(self):
            return now_ns[0]

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    arm = install_robot_level_state(robot, [0.5, 0.6])

    command = types.SimpleNamespace(position=[0.5, 0.6], velocity=[0.0, 0.0])
    module.AceFollowerROS2Robot._arm_command_callback(robot, command)
    module.AceFollowerROS2Robot._control_loop(robot)

    assert arm.position_calls == []
    assert robot._sync.status == sync.FollowerSyncStatus.IDLE

    module.AceFollowerROS2Robot._sync_mode_callback(
        robot, types.SimpleNamespace(data=sync.LeaderSyncMode.SYNC_REQUEST.value)
    )
    module.AceFollowerROS2Robot._arm_command_callback(robot, command)
    module.AceFollowerROS2Robot._control_loop(robot)

    assert arm.position_calls == []
    assert robot._sync.status == sync.FollowerSyncStatus.READY

    now_ns[0] = 1_200_000_000
    module.AceFollowerROS2Robot._control_loop(robot)

    assert arm.position_calls == []
    assert robot._sync.status == sync.FollowerSyncStatus.READY

    module.AceFollowerROS2Robot._sync_mode_callback(
        robot, types.SimpleNamespace(data=sync.LeaderSyncMode.TRACKING.value)
    )
    module.AceFollowerROS2Robot._arm_command_callback(robot, command)
    module.AceFollowerROS2Robot._control_loop(robot)

    assert arm.position_calls == [([0.5, 0.6], {})]
    assert robot._sync.status == sync.FollowerSyncStatus.TRACKING


@pytest.mark.parametrize(
    ("positions", "names"),
    [
        ([], []),
        ([0.1], []),
        ([0.1, 0.2, 0.3], []),
        ([0.1, np.nan], []),
        ([0.1, np.inf], []),
        ([0.1, 0.2], ["joint_2", "joint_1"]),
        ([0.1, 0.2], ["joint_1", "joint_missing"]),
    ],
)
def test_follower_rejects_invalid_arm_command_before_heartbeat(
    monkeypatch,
    positions,
    names,
):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    robot._sync = sync.FollowerSyncController(heartbeat_timeout_ns=int(1e9))
    robot._sync.set_mode(sync.LeaderSyncMode.SYNC_REQUEST)
    robot._sync.set_mode(sync.LeaderSyncMode.TRACKING)
    robot._heartbeat_lost = False
    robot._latest_arm_command = None
    logged = []
    robot.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=1_000_000_000)
    )
    robot.get_logger = lambda: types.SimpleNamespace(warn=lambda message: logged.append(message))
    install_arm_assembly(robot, FakeRosArm([0.0, 0.0]))

    module.AceFollowerROS2Robot._arm_command_callback(
        robot,
        types.SimpleNamespace(position=positions, name=names),
    )

    assert robot._sync.status == sync.FollowerSyncStatus.READY
    assert robot._sync.last_command_ns is None
    assert robot._latest_arm_command is None
    assert logged


def test_follower_accepts_exact_or_empty_arm_command_names(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")

    for names in ([], ["joint_1", "joint_2"]):
        robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
        robot._sync = sync.FollowerSyncController(heartbeat_timeout_ns=int(1e9))
        robot._sync.set_mode(sync.LeaderSyncMode.SYNC_REQUEST)
        robot._sync.set_mode(sync.LeaderSyncMode.TRACKING)
        robot._heartbeat_lost = False
        robot._latest_arm_command = None
        robot.get_clock = lambda: types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(nanoseconds=1_000_000_000)
        )
        robot.get_logger = lambda: types.SimpleNamespace(warn=lambda _message: None)
        install_arm_assembly(robot, FakeRosArm([0.0, 0.0]))

        module.AceFollowerROS2Robot._arm_command_callback(
            robot,
            types.SimpleNamespace(position=[0.1, 0.2], name=names),
        )

        assert robot._sync.status == sync.FollowerSyncStatus.TRACKING
        assert robot._sync.last_command_ns == 1_000_000_000
        assert robot._latest_arm_command == [0.1, 0.2]


def test_follower_rejects_device_invalid_arm_command_before_heartbeat(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    robot._sync = sync.FollowerSyncController(heartbeat_timeout_ns=int(1e9))
    robot._sync.set_mode(sync.LeaderSyncMode.SYNC_REQUEST)
    robot._sync.set_mode(sync.LeaderSyncMode.TRACKING)
    robot._heartbeat_lost = False
    robot._latest_arm_command = None
    logged = []
    robot.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=1_000_000_000)
    )
    robot.get_logger = lambda: types.SimpleNamespace(warn=lambda message: logged.append(message))
    arm = FakeRosArm([0.0, 0.0])
    arm.position_validation_error = ValueError("encoded positions exceed the register range")
    install_arm_assembly(robot, arm)

    module.AceFollowerROS2Robot._arm_command_callback(
        robot,
        types.SimpleNamespace(position=[100.0, 0.0], name=[]),
    )

    assert robot._sync.status == sync.FollowerSyncStatus.READY
    assert robot._sync.last_command_ns is None
    assert robot._latest_arm_command is None
    assert logged == [
        "Ignoring invalid arm command: encoded positions exceed the register range"
    ]


@pytest.mark.parametrize(
    ("positions", "names"),
    [
        ([], []),
        ([0.1, 0.2], []),
        ([np.nan], []),
        ([np.inf], []),
        ([0.1], ["joint_missing"]),
    ],
)
def test_follower_rejects_invalid_gripper_command_before_caching(
    monkeypatch,
    positions,
    names,
):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    robot._sync = sync.FollowerSyncController(heartbeat_timeout_ns=int(1e9))
    robot._sync.set_mode(sync.LeaderSyncMode.SYNC_REQUEST)
    robot._sync.set_mode(sync.LeaderSyncMode.TRACKING)
    robot._latest_gripper_command = None
    logged = []
    robot.get_logger = lambda: types.SimpleNamespace(warn=lambda message: logged.append(message))
    install_arm_assembly(robot, FakeRosArm([0.0, 0.0]), FakeRosGripper())

    module.AceFollowerROS2Robot._gripper_command_callback(
        robot,
        types.SimpleNamespace(position=positions, name=names),
    )

    assert robot._sync.status == sync.FollowerSyncStatus.READY
    assert robot._sync.last_command_ns is None
    assert robot._latest_gripper_command is None
    assert logged


def test_follower_accepts_exact_gripper_command_name_without_establishing_heartbeat(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    robot._sync = sync.FollowerSyncController(heartbeat_timeout_ns=int(1e9))
    robot._sync.set_mode(sync.LeaderSyncMode.SYNC_REQUEST)
    robot._sync.set_mode(sync.LeaderSyncMode.TRACKING)
    robot._latest_gripper_command = None
    robot.get_logger = lambda: types.SimpleNamespace(warn=lambda _message: None)
    install_arm_assembly(robot, FakeRosArm([0.0, 0.0]), FakeRosGripper())

    module.AceFollowerROS2Robot._gripper_command_callback(
        robot,
        types.SimpleNamespace(position=[0.4], name=["joint_5"]),
    )

    assert robot._sync.status == sync.FollowerSyncStatus.READY
    assert robot._sync.last_command_ns is None
    assert robot._latest_gripper_command == [0.4]


def test_follower_requires_fresh_arm_command_after_resync(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    robot._sync = sync.FollowerSyncController(heartbeat_timeout_ns=int(1e9))
    robot._sync.mode = sync.LeaderSyncMode.TRACKING
    robot._sync.status = sync.FollowerSyncStatus.TRACKING
    robot._sync.last_command_ns = 500_000_000
    robot._heartbeat_lost = False
    robot._latest_arm_command = [9.0, 9.0]
    robot._latest_gripper_command = [0.9]
    robot._latest_arm_state = None

    class FakeNow:
        nanoseconds = 1_000_000_000

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None, warn=lambda _message: None)
    arm = FakeRosArm([0.5, 0.6])
    gripper = FakeRosGripper(position=0.4)
    install_arm_assembly(robot, arm, gripper)

    for mode in (
        sync.LeaderSyncMode.SYNC_REQUEST,
        sync.LeaderSyncMode.READY,
        sync.LeaderSyncMode.TRACKING,
    ):
        module.AceFollowerROS2Robot._sync_mode_callback(
            robot,
            types.SimpleNamespace(data=mode.value),
        )

    assert robot._latest_arm_command is None
    assert robot._latest_gripper_command is None

    module.AceFollowerROS2Robot._gripper_command_callback(
        robot,
        types.SimpleNamespace(position=[0.7]),
    )
    module.AceFollowerROS2Robot._control_loop(robot)

    assert arm.position_calls == []
    assert gripper.position_calls == []
    assert robot._sync.status == sync.FollowerSyncStatus.READY

    module.AceFollowerROS2Robot._arm_command_callback(
        robot,
        types.SimpleNamespace(position=[0.7, 0.8]),
    )
    module.AceFollowerROS2Robot._control_loop(robot)

    assert arm.position_calls == [([0.7, 0.8], {})]
    assert gripper.position_calls == [([0.7], {})]
    assert robot._sync.status == sync.FollowerSyncStatus.TRACKING
    assert robot._sync.last_command_ns == FakeNow.nanoseconds


def test_follower_command_timeout_enters_lost_and_requires_resync(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.TRACKING,
        sync.FollowerSyncStatus.TRACKING,
        last_command_ns=0,
    )
    robot._latest_arm_command = [0.5, 0.6]
    robot._latest_gripper_command = [0.7]

    class FakeNow:
        nanoseconds = 2_000_000_001

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    install_robot_level_state(robot, [0.5, 0.6])

    module.AceFollowerROS2Robot._control_loop(robot)

    assert robot._sync.status == sync.FollowerSyncStatus.LOST
    assert robot._heartbeat_lost
    assert robot._latest_arm_command is None
    assert robot._latest_gripper_command is None


def test_follower_sync_request_keeps_ready_when_command_changes(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.SYNC_REQUEST,
        sync.FollowerSyncStatus.READY,
    )

    class FakeNow:
        nanoseconds = 1_100_000_000

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    install_robot_level_state(robot, [0.5, 0.6])

    module.AceFollowerROS2Robot._arm_command_callback(
        robot, types.SimpleNamespace(position=[0.7, 0.6], velocity=[0.0, 0.0])
    )
    module.AceFollowerROS2Robot._control_loop(robot)

    assert robot._sync.status == sync.FollowerSyncStatus.READY


def test_follower_rejects_invalid_sync_mode(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.IDLE,
        sync.FollowerSyncStatus.IDLE,
    )
    logged = []
    robot.get_logger = lambda: types.SimpleNamespace(warn=lambda message: logged.append(message))

    module.AceFollowerROS2Robot._sync_mode_callback(robot, types.SimpleNamespace(data="bad-mode"))

    assert robot._sync.mode == sync.LeaderSyncMode.IDLE
    assert "Ignoring invalid sync mode" in logged[0]


def test_follower_sync_request_logs_hold_still_prompt(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.IDLE,
        sync.FollowerSyncStatus.IDLE,
    )
    logged = []
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda message: logged.append(message))

    module.AceFollowerROS2Robot._sync_mode_callback(
        robot, types.SimpleNamespace(data=sync.LeaderSyncMode.SYNC_REQUEST.value)
    )

    assert robot._sync.status == sync.FollowerSyncStatus.READY
    assert "Holding follower arm pose for leader synchronization." in logged


def test_follower_uses_100hz_control_and_publish_timers(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    timers = []
    declared = []
    parameters = {
        "control_rate": 100.0,
        "publish_rate": 100.0,
        "heartbeat_timeout": 1.0,
    }

    monkeypatch.setattr(module.AceFollowerRobot, "__init__", lambda self, _config_loader: None)
    monkeypatch.setattr(
        module.AceFollowerRobot,
        "act",
        lambda self: (np.zeros(5), np.zeros(5), np.zeros(5)),
    )
    monkeypatch.setattr(module.AceFollowerRobot, "get_robot_state", lambda self: types.SimpleNamespace(
        public_positions=np.zeros(5),
        raw_positions=np.zeros(5),
        velocities=np.zeros(5),
        motor_torque_magnitude=np.zeros(5),
        motor_torque_signed=np.zeros(5),
    ))
    monkeypatch.setattr(module.AceFollowerRobot, "set_position", lambda self, _positions, **_kwargs: None)
    monkeypatch.setattr(
        module.AceFollowerROS2Robot,
        "declare_parameter",
        lambda self, name, default: declared.append((name, default)),
    )
    monkeypatch.setattr(
        module.AceFollowerROS2Robot,
        "get_parameter",
        lambda self, name: types.SimpleNamespace(value=parameters[name]),
    )
    publishers = []
    subscriptions = []
    monkeypatch.setattr(
        module.AceFollowerROS2Robot,
        "create_publisher",
        lambda self, msg_type, topic, qos: publishers.append((msg_type, topic, qos))
        or types.SimpleNamespace(publish=lambda _msg: None),
    )
    monkeypatch.setattr(
        module.AceFollowerROS2Robot,
        "create_subscription",
        lambda self, msg_type, topic, callback, qos: subscriptions.append((msg_type, topic, callback.__name__, qos))
        or object(),
    )
    monkeypatch.setattr(
        module.AceFollowerROS2Robot,
        "create_timer",
        lambda self, period, callback: timers.append((period, callback.__name__)),
    )
    monkeypatch.setattr(
        module.AceFollowerROS2Robot,
        "get_logger",
        lambda self: types.SimpleNamespace(info=lambda _message: None),
    )

    robot = module.AceFollowerROS2Robot(
        config_loader=make_ros2_robot_config(
            "ace_follower",
            end_effector=make_feetech_gripper_config(),
        )
    )

    assert hasattr(robot, "_control_timer")
    assert hasattr(robot, "_state_publish_timer")
    assert hasattr(robot, "_arm_state_pub")
    assert hasattr(robot, "_arm_command_sub")
    assert not hasattr(robot, "_state_pub")
    assert not hasattr(robot, "_command_sub")
    assert not hasattr(robot, "_latest_state")
    assert not hasattr(robot, "_latest_command")
    assert not hasattr(robot, "_timer")
    assert not hasattr(robot, "_publish_timer")
    assert not hasattr(robot, "_update_sync_state")
    assert not hasattr(robot, "_heartbeat_timeout")
    assert not hasattr(robot, "_sync_velocity_tolerance")
    assert not hasattr(robot, "_sync_stable_duration")
    assert not hasattr(robot, "_sync_position_tolerance")
    assert not hasattr(robot, "_sync_stable_duration_ns")
    assert not hasattr(robot, "_publish_state")
    assert not hasattr(robot, "_update_external_estimate")
    assert not hasattr(robot, "_external_estimate_executor")
    assert not hasattr(robot, "_external_estimate_future")
    assert ("publish_rate", 100.0) in declared
    assert ("sync_position_tolerance", 0.03) not in declared
    assert ("sync_velocity_tolerance", 0.05) not in declared
    assert ("control_rate", 100.0) in declared
    assert robot._control_rate == 100.0
    assert robot._publish_rate == 100.0
    assert (1.0 / 100.0, "_control_loop") in timers
    assert (1.0 / 100.0, "_publish_state_loop") in timers
    publisher_topics = {topic for _msg_type, topic, _qos in publishers}
    subscription_topics = {topic for _msg_type, topic, _callback, _qos in subscriptions}
    assert {
        "/ace_follower/arm/state",
        "/ace_follower/arm/sync_status",
        "/ace_follower/gripper/state",
    }.issubset(publisher_topics)
    assert "/ace_follower/arm/" + "external_" + "joint_torque" not in publisher_topics
    assert "/ace_follower/arm/" + "external_" + "wrench" not in publisher_topics
    assert "/ace_follower/gripper/" + "force_" + "state" not in publisher_topics
    assert {"/ace_leader/arm/command", "/ace_leader/gripper/command", "/ace_leader/arm/sync_mode"}.issubset(
        subscription_topics
    )
    assert all(not topic.startswith("/arm/") for topic in publisher_topics | subscription_topics)


def test_follower_tracking_sends_arm_command_to_single_arm(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.TRACKING,
        sync.FollowerSyncStatus.TRACKING,
        last_command_ns=1_000_000_000,
    )
    robot._latest_arm_command = [0.5, 0.6, 0.7]
    arm = None

    class FakeNow:
        nanoseconds = 1_100_000_000

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    arm = install_robot_level_state(robot, [0.8, 0.1, 0.2])

    module.AceFollowerROS2Robot._control_loop(robot)

    assert arm.position_calls == [([0.5, 0.6, 0.7], {})]
    assert robot._sync.status == sync.FollowerSyncStatus.TRACKING


def test_follower_tracking_sends_arm_command_before_reading_state(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.TRACKING,
        sync.FollowerSyncStatus.TRACKING,
        last_command_ns=1_000_000_000,
    )
    robot._latest_arm_command = [0.5, 0.6]
    events = []

    class FakeNow:
        nanoseconds = 1_100_000_000

    class FakeArm:
        ids = np.array([0, 1])
        joint_names = ("joint_1", "joint_2")

        def set_position(self, positions):
            events.append(("set_position", list(positions)))

        def get_state(self):
            events.append(("get_state", None))
            return JointDeviceState(
                public_positions=np.array([0.1, 0.2]),
                raw_positions=np.array([0.1, 0.2]),
                velocities=np.zeros(2),
                motor_torque_magnitude=np.zeros(2),
                motor_torque_signed=np.zeros(2),
            )

    install_arm_assembly(robot, FakeArm())
    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)

    module.AceFollowerROS2Robot._control_loop(robot)

    assert events[:2] == [("set_position", [0.5, 0.6]), ("get_state", None)]
    assert robot._sync.status == sync.FollowerSyncStatus.TRACKING


def test_follower_tracking_keeps_arm_command_arm_only(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.TRACKING,
        sync.FollowerSyncStatus.TRACKING,
        last_command_ns=1_000_000_000,
    )
    robot._latest_arm_command = [0.4, 0.5, 0.6]
    arm = None

    class FakeNow:
        nanoseconds = 1_100_000_000

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    arm = install_robot_level_state(robot, [0.0, 0.0, 0.0])

    module.AceFollowerROS2Robot._control_loop(robot)

    assert arm.position_calls == [([0.4, 0.5, 0.6], {})]


def test_follower_gripper_command_callback_only_caches_command(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.TRACKING,
        sync.FollowerSyncStatus.READY,
    )
    gripper = FakeRosGripper(position=0.2)
    install_arm_assembly(robot, FakeRosArm([]), gripper)

    module.AceFollowerROS2Robot._gripper_command_callback(robot, types.SimpleNamespace(position=[0.8]))

    assert robot._latest_gripper_command == [0.8]
    assert gripper.position_calls == []


def test_follower_tracking_sends_cached_gripper_command(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.TRACKING,
        sync.FollowerSyncStatus.TRACKING,
        last_command_ns=1_000_000_000,
    )
    robot._latest_gripper_command = [0.8]
    gripper = FakeRosGripper(position=0.2)

    class FakeNow:
        nanoseconds = 1_100_000_000

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    arm = install_robot_level_state(robot, [0.0, 0.0])
    install_arm_assembly(robot, arm, gripper)

    module.AceFollowerROS2Robot._control_loop(robot)

    assert gripper.position_calls == [([0.8], {})]
    assert robot._sync.status == sync.FollowerSyncStatus.TRACKING


def test_follower_tracking_sends_gripper_command_without_cached_state(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.TRACKING,
        sync.FollowerSyncStatus.TRACKING,
        last_command_ns=1_000_000_000,
    )
    robot._latest_gripper_command = [0.8]
    gripper = FakeRosGripper(position=0.2)

    class FakeNow:
        nanoseconds = 1_100_000_000

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    arm = install_robot_level_state(robot, [0.0, 0.0])
    install_arm_assembly(robot, arm, gripper)

    module.AceFollowerROS2Robot._control_loop(robot)

    assert gripper.position_calls == [([0.8], {})]
    assert robot._latest_gripper_state.public_positions[0] == pytest.approx(0.2)
    assert gripper.state_reads == 1


def test_follower_ready_does_not_send_cached_gripper_command(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.READY,
        sync.FollowerSyncStatus.READY,
    )
    robot._latest_gripper_command = [0.8]
    gripper = FakeRosGripper(position=0.2)

    class FakeNow:
        nanoseconds = 1_100_000_000

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    arm = install_robot_level_state(robot, [0.0, 0.0])
    install_arm_assembly(robot, arm, gripper)

    module.AceFollowerROS2Robot._control_loop(robot)

    assert gripper.position_calls == []
    assert robot._sync.status == sync.FollowerSyncStatus.READY


def test_ros2_classes_do_not_define_robot_arm_boundary_helpers(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    leader_module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    follower_module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))

    assert "_arm_ids" not in leader_module.AceLeaderROS2Robot.__dict__
    assert "_arm_values_from_public" not in leader_module.AceLeaderROS2Robot.__dict__
    assert "_arm_values_from_public" not in follower_module.AceFollowerROS2Robot.__dict__


def test_leader_uses_100hz_control_and_publish_timers(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    timers = []
    declared = []
    parameters = {
        "control_rate": 100.0,
        "publish_rate": 100.0,
        "sync_status_timeout": 0.5,
        "follower_state_timeout": 0.5,
        "sync_position_tolerance": 0.03,
        "sync_stable_duration": 0.2,
        "sync_profile_velocity": 2.0,
        "sync_profile_acceleration": 3.0,
        "ready_lock_rate": 20.0,
        "ready_resync_threshold": 0.08,
    }

    monkeypatch.setattr(module.AceLeaderRobot, "__init__", lambda self, _config_loader: None)
    monkeypatch.setattr(
        module.AceLeaderROS2Robot,
        "declare_parameter",
        lambda self, name, default: declared.append((name, default)),
    )
    monkeypatch.setattr(
        module.AceLeaderROS2Robot,
        "get_parameter",
        lambda self, name: types.SimpleNamespace(value=parameters[name]),
    )
    publishers = []
    subscriptions = []
    monkeypatch.setattr(
        module.AceLeaderROS2Robot,
        "create_publisher",
        lambda self, msg_type, topic, qos: publishers.append((msg_type, topic, qos))
        or types.SimpleNamespace(publish=lambda _msg: None),
    )
    monkeypatch.setattr(
        module.AceLeaderROS2Robot,
        "create_subscription",
        lambda self, msg_type, topic, callback, qos: subscriptions.append((msg_type, topic, callback.__name__, qos))
        or object(),
    )
    monkeypatch.setattr(
        module.AceLeaderROS2Robot,
        "create_timer",
        lambda self, period, callback: timers.append((period, callback.__name__)),
    )
    monkeypatch.setattr(
        module.AceLeaderROS2Robot,
        "get_logger",
        lambda self: types.SimpleNamespace(info=lambda _message: None),
    )

    robot = module.AceLeaderROS2Robot(
        config_loader=make_ros2_robot_config("ace_leader")
    )

    assert hasattr(robot, "_control_timer")
    assert hasattr(robot, "_sync_mode_publish_timer")
    assert hasattr(robot, "_arm_command_pub")
    assert hasattr(robot, "_latest_arm_command")
    assert not hasattr(robot, "_arm_command_publish_timer")
    assert not hasattr(robot, "_command_publish_timer")
    assert not hasattr(robot, "_command_pub")
    assert not hasattr(robot, "_latest_command")
    assert not hasattr(robot, "_timer")
    assert not hasattr(robot, "_publish_timer")
    assert not hasattr(robot, "_publish_command")
    assert not hasattr(robot, "_sync_status_timeout")
    assert ("publish_rate", 100.0) in declared
    assert ("sync_move_step_size", 0.2) not in declared
    assert ("sync_move_max_steps", 20) not in declared
    assert ("sync_profile_velocity", 2.0) in declared
    assert ("sync_profile_acceleration", 3.0) in declared
    assert ("ready_lock_rate", 20.0) in declared
    assert ("ready_resync_threshold", 0.08) in declared
    assert robot._control_rate == 100.0
    assert robot._publish_rate == 100.0
    assert not hasattr(robot, "_sync_move_step_size")
    assert not hasattr(robot, "_sync_move_max_steps")
    assert robot._sync_profile_velocity == 2.0
    assert robot._sync_profile_acceleration == 3.0
    assert robot._ready_lock_period_ns == int(0.05e9)
    assert robot._ready_resync_threshold == 0.08
    assert (1.0 / 100.0, "_control_loop") in timers
    assert (1.0 / 100.0, "_publish_sync_mode_loop") in timers
    publisher_topics = {topic for _msg_type, topic, _qos in publishers}
    subscription_topics = {topic for _msg_type, topic, _callback, _qos in subscriptions}
    assert {
        "/ace_leader/arm/command",
        "/ace_leader/arm/sync_mode",
        "/ace_leader/gripper/command",
    }.issubset(publisher_topics)
    assert {"/ace_follower/arm/state", "/ace_follower/arm/sync_status"}.issubset(subscription_topics)
    assert all(not topic.startswith("/arm/") for topic in publisher_topics | subscription_topics)


def test_leader_control_loop_publishes_tracking_commands_directly(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._sync_mode = sync.LeaderSyncMode.TRACKING
    robot._follower_sync_status = sync.FollowerSyncStatus.TRACKING
    robot._last_follower_sync_status_ns = None
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._last_follower_state_ns = None
    robot._is_landed = False
    robot._latest_arm_command = None
    robot._latest_gripper_command = None
    published = {"mode": [], "arm": [], "gripper": []}

    joint_pos = np.array([0.1, 0.2])
    joint_vel = np.array([1.1, 1.2])
    joint_effort = np.array([2.1, 2.2])

    robot.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(
            nanoseconds=1_000_000_000,
            to_msg=lambda: "stamp",
        )
    )
    robot.ids = np.array([0, 1])
    install_arm_assembly(
        robot,
        FakeRosArm(joint_pos, velocities=joint_vel, effort=joint_effort),
    )
    robot._sync_mode_pub = types.SimpleNamespace(publish=lambda msg: published["mode"].append(msg))
    robot._arm_command_pub = types.SimpleNamespace(publish=lambda msg: published["arm"].append(msg))
    robot._gripper_command_pub = types.SimpleNamespace(publish=lambda msg: published["gripper"].append(msg))

    module.AceLeaderROS2Robot._control_loop(robot)

    assert published["mode"][0].data == "tracking"
    assert published["arm"][0].position == [0.1, 0.2]
    assert published["arm"][0].velocity == [1.1, 1.2]
    assert published["arm"][0].effort == [2.1, 2.2]
    assert published["gripper"] == []
    cached_pos, cached_vel, cached_effort = robot._latest_arm_command
    np.testing.assert_allclose(cached_pos, joint_pos)
    np.testing.assert_allclose(cached_vel, joint_vel)
    np.testing.assert_allclose(cached_effort, joint_effort)


def test_leader_control_loop_does_not_publish_commands_outside_tracking(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._sync_mode = sync.LeaderSyncMode.READY
    robot._follower_sync_status = sync.FollowerSyncStatus.READY
    robot._last_follower_sync_status_ns = 1_000_000_000
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._follower_state_timeout_ns = int(0.5e9)
    robot._sync_position_tolerance = 0.03
    robot._sync_profile_velocity = 2.0
    robot._sync_profile_acceleration = 3.0
    robot._ready_resync_threshold = 0.08
    robot._sync_target_position = [0.1, 0.2]
    robot._last_ready_lock_ns = 1_000_000_000
    robot._ready_lock_period_ns = int(0.05e9)
    robot._last_follower_state_ns = 1_000_000_000
    robot._is_landed = False
    robot._latest_arm_command = None
    robot._latest_gripper_command = None
    published = {"mode": [], "arm": [], "gripper": []}

    robot.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(
            nanoseconds=1_100_000_000,
            to_msg=lambda: "stamp",
        )
    )
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    robot.set_position = lambda *_args, **_kwargs: None
    robot.ids = np.array([0, 1, 4])
    gripper = FakeRosGripper(position=0.5)
    install_arm_assembly(robot, FakeRosArm([0.1, 0.2]), gripper)
    robot._sync_mode_pub = types.SimpleNamespace(publish=lambda msg: published["mode"].append(msg))
    robot._arm_command_pub = types.SimpleNamespace(publish=lambda msg: published["arm"].append(msg))
    robot._gripper_command_pub = types.SimpleNamespace(publish=lambda msg: published["gripper"].append(msg))

    module.AceLeaderROS2Robot._control_loop(robot)

    assert published["mode"][0].data == "ready"
    assert published["arm"] == []
    assert published["gripper"] == []


def test_leader_control_loop_publishes_gripper_command_separately(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    published = {"arm": [], "gripper": [], "mode": []}
    robot._sync_mode = sync.LeaderSyncMode.TRACKING
    robot._follower_sync_status = sync.FollowerSyncStatus.TRACKING
    robot._last_follower_sync_status_ns = None
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._last_follower_state_ns = None
    robot._is_landed = False
    robot._latest_arm_command = None
    robot._latest_gripper_command = None
    robot.gripper_id = 4
    install_arm_assembly(
        robot,
        FakeRosArm([0.1, 0.2], velocities=[1.1, 1.2], effort=[2.1, 2.2]),
        FakeRosGripper(position=0.9, velocity=0.3, effort=0.4),
    )
    robot.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(
            nanoseconds=1_000_000_000,
            to_msg=lambda: "stamp",
        )
    )
    robot._sync_mode_pub = types.SimpleNamespace(publish=lambda msg: published["mode"].append(msg))
    robot._arm_command_pub = types.SimpleNamespace(publish=lambda msg: published["arm"].append(msg))
    robot._gripper_command_pub = types.SimpleNamespace(publish=lambda msg: published["gripper"].append(msg))

    module.AceLeaderROS2Robot._control_loop(robot)

    assert published["mode"][0].data == "tracking"
    assert published["arm"][0].position == [0.1, 0.2]
    assert published["gripper"][0].name == ["joint_5"]
    assert published["gripper"][0].position == [0.9]


def test_leader_auto_aligns_to_follower_state_without_gripper_axis(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._sync_mode = sync.LeaderSyncMode.IDLE
    robot._follower_sync_status = sync.FollowerSyncStatus.IDLE
    robot._last_follower_sync_status_ns = None
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._follower_state_timeout_ns = int(0.5e9)
    robot._sync_position_tolerance = 0.03
    robot._sync_stable_duration_ns = int(0.2e9)
    robot._sync_profile_velocity = 2.0
    robot._sync_profile_acceleration = 3.0
    robot._ready_lock_period_ns = int(0.05e9)
    robot._ready_resync_threshold = 0.08
    robot._last_ready_lock_ns = None
    robot._sync_stable_since_ns = None
    robot._sync_target_position = None
    robot._last_follower_state_ns = 1_000_000_000
    robot._is_landed = False
    robot._latest_arm_command = None
    robot._latest_follower_state = ([0.2, 0.4], [0.0, 0.0], [0.0, 0.0])
    robot.set_calls = []
    logged = []

    class FakeNow:
        nanoseconds = 1_000_000_000

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda message: logged.append(message))
    robot.set_position = lambda positions, ids=None, **kwargs: robot.set_calls.append(
        (list(positions), list(ids), kwargs)
    )
    robot.ids = np.array([0, 1, 4])
    robot.gripper_index = 2
    arm = FakeRosArm([0.1, 0.3])
    install_arm_assembly(robot, arm)

    module.AceLeaderROS2Robot._control_loop(robot)

    assert robot._sync_mode == sync.LeaderSyncMode.SYNC_REQUEST
    assert robot._latest_arm_command is not None
    assert robot._sync_target_position == [0.2, 0.4]
    assert robot.set_calls == [([0.2, 0.4], [0, 1], {"velocities": 2.0, "accelerations": 3.0})]

    arm.public_positions = np.array([0.2, 0.4])
    arm.raw_positions = arm.public_positions.copy()
    module.AceLeaderROS2Robot._control_loop(robot)
    assert robot._sync_mode == sync.LeaderSyncMode.SYNC_REQUEST

    class LaterNow:
        nanoseconds = 1_200_000_000

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: LaterNow())
    module.AceLeaderROS2Robot._control_loop(robot)

    assert robot._sync_mode == sync.LeaderSyncMode.READY
    assert robot.set_calls[-1] == ([0.2, 0.4], [0, 1], {"velocities": 2.0, "accelerations": 3.0})


def test_leader_sync_request_uses_shortest_angle_error_for_stability(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._sync_mode = sync.LeaderSyncMode.SYNC_REQUEST
    robot._follower_sync_status = sync.FollowerSyncStatus.READY
    robot._last_follower_sync_status_ns = None
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._follower_state_timeout_ns = int(0.5e9)
    robot._sync_position_tolerance = 0.03
    robot._sync_stable_duration_ns = int(0.2e9)
    robot._sync_profile_velocity = 2.0
    robot._sync_profile_acceleration = 3.0
    robot._ready_lock_period_ns = int(0.05e9)
    robot._ready_resync_threshold = 0.08
    robot._sync_stable_since_ns = 1_000_000_000
    robot._last_ready_lock_ns = None
    robot._sync_target_position = [-np.pi + 0.01, 0.4]
    robot._last_follower_state_ns = 1_200_000_000
    robot._is_landed = False
    robot._latest_arm_command = None
    robot._latest_follower_state = ([-np.pi + 0.01, 0.4], [], [])
    robot.set_calls = []

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=1_200_000_000))
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    robot.set_position = lambda positions, ids=None, **kwargs: robot.set_calls.append(
        (list(positions), list(ids), kwargs)
    )
    robot.ids = np.array([0, 1, 4])
    robot.gripper_index = 2
    install_arm_assembly(robot, FakeRosArm([np.pi + 0.01, 0.4]))

    module.AceLeaderROS2Robot._control_loop(robot)

    assert robot._sync_mode == sync.LeaderSyncMode.READY
    assert robot.set_calls == [([-np.pi + 0.01, 0.4], [0, 1], {"velocities": 2.0, "accelerations": 3.0})]


def test_leader_waits_for_gripper_one_before_tracking(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._sync_mode = sync.LeaderSyncMode.READY
    robot._follower_sync_status = sync.FollowerSyncStatus.READY
    robot._last_follower_sync_status_ns = 1_000_000_000
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._follower_state_timeout_ns = int(0.5e9)
    robot._sync_position_tolerance = 0.03
    robot._sync_stable_duration_ns = int(0.2e9)
    robot._sync_profile_velocity = 2.0
    robot._sync_profile_acceleration = 3.0
    robot._sync_stable_since_ns = 1_000_000_000
    robot._sync_target_position = [0.2, 0.4]
    robot._ready_lock_period_ns = int(0.05e9)
    robot._ready_resync_threshold = 0.08
    robot._last_ready_lock_ns = 1_000_000_000
    robot._last_follower_state_ns = 1_000_000_000
    robot._is_landed = False
    robot._latest_arm_command = None
    robot._latest_follower_state = ([0.2, 0.4], [], [])
    robot.set_calls = []
    robot.set_torque_enable_calls = []
    logged = []

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=1_100_000_000))
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda message: logged.append(message))
    robot.set_position = lambda positions, ids=None, **kwargs: robot.set_calls.append(
        (list(positions), list(ids), kwargs)
    )
    robot.set_torque_enable = lambda enable, ids=None: robot.set_torque_enable_calls.append((enable, list(ids)))
    robot.ids = np.array([0, 1, 4])
    robot.gripper_id = 4
    robot.gripper_index = 2
    gripper = FakeRosGripper(position=0.5)
    install_arm_assembly(robot, FakeRosArm([0.2, 0.4]), gripper)

    module.AceLeaderROS2Robot._control_loop(robot)

    assert robot._sync_mode == sync.LeaderSyncMode.READY
    assert robot.set_calls == [([0.2, 0.4], [0, 1], {"velocities": 2.0, "accelerations": 3.0})]

    gripper.position = 1.0
    module.AceLeaderROS2Robot._control_loop(robot)

    assert robot._sync_mode == sync.LeaderSyncMode.TRACKING
    assert robot.set_torque_enable_calls == [(module.TorqueEnable.Disable, [0, 1])]


def test_leader_without_gripper_enters_tracking_without_release_gate(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._sync_mode = sync.LeaderSyncMode.READY
    robot._follower_sync_status = sync.FollowerSyncStatus.READY
    robot._last_follower_sync_status_ns = 1_000_000_000
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._follower_state_timeout_ns = int(0.5e9)
    robot._sync_position_tolerance = 0.03
    robot._sync_stable_duration_ns = int(0.2e9)
    robot._sync_profile_velocity = 2.0
    robot._sync_profile_acceleration = 3.0
    robot._sync_stable_since_ns = 1_000_000_000
    robot._sync_target_position = [0.2, 0.4]
    robot._ready_lock_period_ns = int(0.05e9)
    robot._ready_resync_threshold = 0.08
    robot._last_ready_lock_ns = 1_000_000_000
    robot._last_follower_state_ns = 1_000_000_000
    robot._is_landed = False
    robot._latest_arm_command = None
    robot._latest_follower_state = ([0.2, 0.4], [], [])
    robot.set_calls = []
    robot.set_torque_enable_calls = []

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=1_100_000_000))
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    robot.set_position = lambda positions, ids=None, **kwargs: robot.set_calls.append(
        (list(positions), list(ids), kwargs)
    )
    robot.set_torque_enable = lambda enable, ids=None: robot.set_torque_enable_calls.append((enable, list(ids)))
    robot.ids = np.array([0, 1])
    robot.gripper_index = None
    install_arm_assembly(robot, FakeRosArm([0.2, 0.4]))

    module.AceLeaderROS2Robot._control_loop(robot)

    assert robot._sync_mode == sync.LeaderSyncMode.TRACKING
    assert robot.set_torque_enable_calls == [(module.TorqueEnable.Disable, [0, 1])]


def test_leader_ready_hold_waits_for_lock_period(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._sync_mode = sync.LeaderSyncMode.READY
    robot._follower_sync_status = sync.FollowerSyncStatus.READY
    robot._last_follower_sync_status_ns = 1_000_000_000
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._follower_state_timeout_ns = int(0.5e9)
    robot._sync_position_tolerance = 0.03
    robot._sync_profile_velocity = 2.0
    robot._sync_profile_acceleration = 3.0
    robot._ready_lock_period_ns = int(0.05e9)
    robot._ready_resync_threshold = 0.08
    robot._last_ready_lock_ns = 1_000_000_000
    robot._sync_target_position = [0.2, 0.4]
    robot._last_follower_state_ns = 1_000_000_000
    robot._is_landed = False
    robot._latest_follower_state = ([0.2, 0.4], [], [])
    robot._latest_arm_command = None
    robot.set_calls = []
    robot.set_torque_enable_calls = []

    now_ns = [1_020_000_000, 1_060_000_000]
    robot.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=now_ns.pop(0))
    )
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    robot.set_position = lambda positions, ids=None, **kwargs: robot.set_calls.append(
        (list(positions), list(ids), kwargs)
    )
    robot.set_torque_enable = lambda enable, ids=None: robot.set_torque_enable_calls.append((enable, list(ids)))
    robot.ids = np.array([0, 1, 4])
    robot.gripper_id = 4
    robot.gripper_index = 2
    install_arm_assembly(robot, FakeRosArm([0.2, 0.4]), FakeRosGripper(position=0.5))

    module.AceLeaderROS2Robot._control_loop(robot)
    assert robot.set_calls == []

    module.AceLeaderROS2Robot._control_loop(robot)
    assert robot.set_calls == [([0.2, 0.4], [0, 1], {"velocities": 2.0, "accelerations": 3.0})]


def test_leader_ready_large_error_returns_to_sync_request_without_hold(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._sync_mode = sync.LeaderSyncMode.READY
    robot._follower_sync_status = sync.FollowerSyncStatus.READY
    robot._last_follower_sync_status_ns = 1_000_000_000
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._follower_state_timeout_ns = int(0.5e9)
    robot._sync_position_tolerance = 0.03
    robot._sync_profile_velocity = 2.0
    robot._sync_profile_acceleration = 3.0
    robot._ready_lock_period_ns = int(0.05e9)
    robot._ready_resync_threshold = 0.08
    robot._last_ready_lock_ns = 1_000_000_000
    robot._sync_target_position = [0.2, 0.4]
    robot._last_follower_state_ns = 1_000_000_000
    robot._is_landed = False
    robot._latest_follower_state = ([0.2, 0.4, 0.0], [], [])
    robot._latest_arm_command = None
    robot.set_calls = []

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=1_100_000_000))
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    robot.set_position = (
        lambda *_args, **_kwargs: pytest.fail("READY large error must resync instead of holding")
    )
    robot.set_torque_enable = lambda *_args, **_kwargs: None
    robot.ids = np.array([0, 1, 4])
    robot.gripper_index = 2
    install_arm_assembly(robot, FakeRosArm([0.6, 0.4]))

    module.AceLeaderROS2Robot._control_loop(robot)

    assert robot._sync_mode == sync.LeaderSyncMode.SYNC_REQUEST
    assert robot._sync_target_position is None


def test_leader_publishes_idle_mode_without_command_before_follower_state(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._sync_mode = sync.LeaderSyncMode.IDLE
    robot._follower_sync_status = sync.FollowerSyncStatus.IDLE
    robot._last_follower_sync_status_ns = None
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._follower_state_timeout_ns = int(0.5e9)
    robot._latest_follower_state = None
    robot._last_follower_state_ns = None
    robot._is_landed = False
    robot._latest_arm_command = None
    published_modes = []
    published_commands = []

    robot.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(
            nanoseconds=1_000_000_000,
            to_msg=lambda: "stamp",
        )
    )
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    install_arm_assembly(robot, FakeRosArm([0.1, 0.2, 0.3]))
    robot._sync_mode_pub = types.SimpleNamespace(publish=lambda msg: published_modes.append(msg.data))
    robot._arm_command_pub = types.SimpleNamespace(publish=lambda msg: published_commands.append(msg))

    module.AceLeaderROS2Robot._control_loop(robot)

    assert robot._sync_mode == sync.LeaderSyncMode.IDLE
    assert robot._latest_arm_command is not None
    assert published_modes == ["idle"]
    assert published_commands == []


def test_leader_landing_publishes_stop_mode_without_command(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._sync_mode = sync.LeaderSyncMode.TRACKING
    robot._follower_sync_status = sync.FollowerSyncStatus.TRACKING
    robot._last_follower_sync_status_ns = 1_000_000_000
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._is_landed = True
    robot._latest_arm_command = (np.zeros(3), np.zeros(3), np.zeros(3))
    published_modes = []
    published_commands = []

    robot.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(
            nanoseconds=1_000_000_000,
            to_msg=lambda: "stamp",
        )
    )
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    install_arm_assembly(robot, FakeRosArm([0.1, 0.2, -0.1]))
    robot._sync_mode_pub = types.SimpleNamespace(publish=lambda msg: published_modes.append(msg.data))
    robot._arm_command_pub = types.SimpleNamespace(publish=lambda msg: published_commands.append(msg))

    module.AceLeaderROS2Robot._control_loop(robot)

    assert robot._sync_mode == sync.LeaderSyncMode.STOP
    assert robot._latest_arm_command is not None
    assert published_modes == ["stop"]
    assert published_commands == []


def test_leader_status_timeout_returns_to_sync_request(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._sync_mode = sync.LeaderSyncMode.TRACKING
    robot._follower_sync_status = sync.FollowerSyncStatus.TRACKING
    robot._last_follower_sync_status_ns = 0
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._is_landed = False
    robot._latest_arm_command = None

    class FakeNow:
        nanoseconds = 700_000_001

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    install_arm_assembly(robot, FakeRosArm([0.1, -0.2, -0.1]))

    module.AceLeaderROS2Robot._control_loop(robot)

    assert robot._sync_mode == sync.LeaderSyncMode.SYNC_REQUEST


def test_leader_lost_status_returns_to_sync_request_immediately(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._sync_mode = sync.LeaderSyncMode.TRACKING
    robot._follower_sync_status = sync.FollowerSyncStatus.LOST
    robot._last_follower_sync_status_ns = 1_000_000_000
    robot._sync_status_timeout_ns = int(0.5e9)
    robot._is_landed = False
    robot._latest_arm_command = None

    class FakeNow:
        nanoseconds = 1_000_000_000

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    install_arm_assembly(robot, FakeRosArm([0.1, -0.2, -0.1]))

    module.AceLeaderROS2Robot._control_loop(robot)

    assert robot._sync_mode == sync.LeaderSyncMode.SYNC_REQUEST


def test_leader_rejects_invalid_sync_status(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._follower_sync_status = sync.FollowerSyncStatus.IDLE
    robot._last_follower_sync_status_ns = None
    logged = []
    robot.get_logger = lambda: types.SimpleNamespace(warn=lambda message: logged.append(message))

    module.AceLeaderROS2Robot._sync_status_callback(robot, types.SimpleNamespace(data="bad-status"))

    assert robot._follower_sync_status == sync.FollowerSyncStatus.IDLE
    assert robot._last_follower_sync_status_ns is None
    assert "Ignoring invalid sync status" in logged[0]


def test_sync_mode_and_status_are_published_from_publish_loops(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    leader_module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    follower_module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    sync = importlib.import_module("acetele.utils.teleop_sync")

    leader = leader_module.AceLeaderROS2Robot.__new__(leader_module.AceLeaderROS2Robot)
    follower = follower_module.AceFollowerROS2Robot.__new__(follower_module.AceFollowerROS2Robot)
    published_modes = []
    published_commands = []
    published_states = []
    published_px4_states = []
    published_statuses = []
    leader._sync_mode = sync.LeaderSyncMode.SYNC_REQUEST
    install_follower_sync(
        follower,
        sync,
        sync.LeaderSyncMode.READY,
        sync.FollowerSyncStatus.READY,
    )
    leader._latest_arm_command = (np.zeros(2), np.zeros(2), np.zeros(2))
    follower._latest_arm_state = make_joint_device_state(np.zeros(2))
    install_arm_assembly(follower, FakeRosArm(np.zeros(2)))
    leader.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(to_msg=lambda: "leader-stamp")
    )
    follower.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(
            nanoseconds=1_000_000_000,
            to_msg=lambda: "follower-stamp",
        )
    )
    follower.get_logger = lambda: types.SimpleNamespace(warn=lambda _message: None)
    leader.ids = np.array([0, 1])
    follower.ids = np.array([0, 1])
    follower._warned_invalid_px4_arm_state = False
    leader._sync_mode_pub = types.SimpleNamespace(publish=lambda msg: published_modes.append(msg.data))
    leader._arm_command_pub = types.SimpleNamespace(publish=lambda msg: published_commands.append(msg))
    follower._arm_state_pub = types.SimpleNamespace(publish=lambda msg: published_states.append(msg))
    follower._px4_arm_state_pub = types.SimpleNamespace(publish=lambda msg: published_px4_states.append(msg))
    follower._sync_status_pub = types.SimpleNamespace(publish=lambda msg: published_statuses.append(msg.data))

    leader_module.AceLeaderROS2Robot._publish_sync_mode_loop(leader)
    follower_module.AceFollowerROS2Robot._publish_state_loop(follower)

    assert published_modes == ["sync_request"]
    assert published_commands == []
    assert len(published_states) == 1
    assert published_px4_states == []
    assert published_statuses == ["ready"]


def test_teleop_sync_exports_only_state_enums():
    sync = importlib.import_module("acetele.utils.teleop_sync")

    assert hasattr(sync, "LeaderSyncMode")
    assert hasattr(sync, "FollowerSyncStatus")
    assert not hasattr(sync, "is_valid_leader_sync_mode")
    assert not hasattr(sync, "is_valid_follower_sync_status")


def test_follower_publish_state_loop_publishes_padded_arm_only_px4_state(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    published = {"state": [], "px4": []}
    logged = []

    class FakeNow:
        nanoseconds = 1_234_567_890

        def to_msg(self):
            return "stamp"

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(warn=lambda message: logged.append(message))
    robot.ids = np.array([0, 1, 2, 3, 4])
    robot._arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["state"].append(msg))
    robot._px4_arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["px4"].append(msg))
    robot._sync_status_pub = types.SimpleNamespace(publish=lambda msg: published.setdefault("status", []).append(msg))
    sync = importlib.import_module("acetele.utils.teleop_sync")
    install_follower_sync(
        robot,
        sync,
        sync.LeaderSyncMode.TRACKING,
        sync.FollowerSyncStatus.TRACKING,
        last_command_ns=1_000_000_000,
    )
    robot._warned_invalid_px4_arm_state = False

    joint_pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    joint_vel = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    joint_effort = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
    robot._latest_arm_state = make_joint_device_state(
        joint_pos,
        joint_vel,
        joint_effort,
    )
    install_arm_assembly(robot, FakeRosArm(joint_pos))

    module.AceFollowerROS2Robot._publish_state_loop(robot)

    assert len(published["state"]) == 1
    state_msg = published["state"][0]
    assert state_msg.header.stamp == "stamp"
    assert state_msg.name == ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
    assert state_msg.position == joint_pos.tolist()
    assert state_msg.velocity == joint_vel.tolist()
    assert state_msg.effort == joint_effort.tolist()

    assert len(published["px4"]) == 1
    px4_msg = published["px4"][0]
    assert px4_msg.timestamp == 1_234_567
    assert px4_msg.timestamp_sample == 1_234_567
    assert px4_msg.sequence == 0
    assert px4_msg.joint_count == 5
    assert px4_msg.arm_velocity_valid
    assert px4_msg.arm_position == joint_pos.tolist() + [0.0] * 9
    assert px4_msg.arm_velocity == joint_vel.tolist() + [0.0] * 9
    assert published["status"][0].data == "tracking"
    assert logged == []
    assert not hasattr(robot, "_publish_state")


def test_follower_publish_state_loop_publishes_gripper_state_and_px4_adapter(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    published = {"state": [], "gripper": [], "px4": []}

    class FakeNow:
        nanoseconds = 1_234_567_890

        def to_msg(self):
            return "stamp"

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(warn=lambda _message: None)
    install_arm_assembly(
        robot,
        FakeRosArm([0.1, 0.2, 0.3, 0.4]),
        FakeRosGripper(position=0.9),
    )
    robot.gripper_id = 4
    robot._latest_arm_state = make_joint_device_state([0.1, 0.2, 0.3, 0.4])
    robot._latest_gripper_state = make_joint_device_state([0.9], [0.5], [0.6])
    robot._arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["state"].append(msg))
    robot._gripper_state_pub = types.SimpleNamespace(publish=lambda msg: published["gripper"].append(msg))
    robot._px4_arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["px4"].append(msg))
    robot._sync_status_pub = types.SimpleNamespace(publish=lambda _msg: None)
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot._sync = sync.FollowerSyncController(int(1e9))
    robot._sync.mode = sync.LeaderSyncMode.TRACKING
    robot._sync.status = sync.FollowerSyncStatus.TRACKING
    robot._warned_invalid_px4_arm_state = False

    module.AceFollowerROS2Robot._publish_state_loop(robot)

    assert published["state"][0].position == [0.1, 0.2, 0.3, 0.4]
    assert published["gripper"][0].name == ["joint_5"]
    assert published["gripper"][0].position == [0.9]
    px4_msg = published["px4"][0]
    assert px4_msg.timestamp == 1_234_567
    assert px4_msg.timestamp_sample == 1_234_567
    assert px4_msg.sequence == 0
    assert px4_msg.joint_count == 4
    assert px4_msg.arm_velocity_valid
    assert px4_msg.arm_position == [0.1, 0.2, 0.3, 0.4] + [0.0] * 10
    assert px4_msg.arm_velocity == [0.0] * 14


def test_follower_publish_state_loop_rejects_nonfinite_px4_arm_state(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    published = {"state": [], "px4": []}
    logged = []
    now_ns = [1_000_000_000, 1_100_000_000]

    class FakeNow:
        def __init__(self, nanoseconds):
            self.nanoseconds = nanoseconds

        def to_msg(self):
            return f"stamp-{self.nanoseconds}"

    robot.get_clock = lambda: types.SimpleNamespace(
        now=lambda: FakeNow(now_ns[len(published["state"])])
    )
    robot.get_logger = lambda: types.SimpleNamespace(warn=lambda message: logged.append(message))
    robot.ids = np.array([0, 1, 2, 3, 4])
    robot._arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["state"].append(msg))
    robot._px4_arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["px4"].append(msg))
    robot._sync_status_pub = types.SimpleNamespace(publish=lambda _msg: None)
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot._sync = sync.FollowerSyncController(int(1e9))
    robot._sync.mode = sync.LeaderSyncMode.TRACKING
    robot._sync.status = sync.FollowerSyncStatus.TRACKING
    robot._latest_gripper_state = None
    robot._warned_invalid_px4_arm_state = False

    joint_pos = np.zeros(5)
    joint_effort = np.zeros(5)
    install_arm_assembly(robot, FakeRosArm(joint_pos))

    robot._latest_arm_state = make_joint_device_state(
        joint_pos,
        [0.2, 0.4, 2.0, np.nan, -2.0],
        joint_effort,
    )
    module.AceFollowerROS2Robot._publish_state_loop(robot)
    robot._latest_arm_state = make_joint_device_state(
        joint_pos,
        [0.8, 1.0, 2.0, 0.4, -2.0],
        joint_effort,
    )
    module.AceFollowerROS2Robot._publish_state_loop(robot)

    np.testing.assert_allclose(
        published["state"][0].velocity,
        [0.2, 0.4, 2.0, np.nan, -2.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        published["state"][1].velocity,
        [0.8, 1.0, 2.0, 0.4, -2.0],
    )
    assert [msg.arm_position for msg in published["px4"]] == [
        joint_pos.tolist() + [0.0] * 9,
    ]
    assert [msg.timestamp for msg in published["px4"]] == [1_100_000]
    assert [msg.timestamp_sample for msg in published["px4"]] == [1_100_000]
    assert [msg.sequence for msg in published["px4"]] == [0]
    assert [msg.joint_count for msg in published["px4"]] == [5]
    np.testing.assert_allclose(
        published["px4"][0].arm_velocity,
        [0.8, 1.0, 2.0, 0.4, -2.0] + [0.0] * 9,
    )
    assert len(logged) == 1
    assert "must be finite" in logged[0]


def test_follower_publish_state_loop_does_not_publish_removed_external_topics(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    published = {"state": [], "px4": []}

    class FakeNow:
        nanoseconds = 1_234_567_890

        def to_msg(self):
            return "stamp"

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(warn=lambda _message: None)
    robot.ids = np.array([0, 1, 2, 3, 4])
    robot._arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["state"].append(msg))
    robot._px4_arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["px4"].append(msg))
    robot._sync_status_pub = types.SimpleNamespace(publish=lambda _msg: None)
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot._sync = sync.FollowerSyncController(int(1e9))
    robot._sync.mode = sync.LeaderSyncMode.TRACKING
    robot._sync.status = sync.FollowerSyncStatus.TRACKING
    robot._latest_gripper_state = None
    robot._warned_invalid_px4_arm_state = False
    robot._latest_arm_state = make_joint_device_state(
        np.zeros(5),
        np.zeros(5),
        [9.0, 8.0, 7.0, 6.0, 5.0],
    )
    install_arm_assembly(robot, FakeRosArm(np.zeros(5)))

    module.AceFollowerROS2Robot._publish_state_loop(robot)

    assert published["state"][0].effort == [9.0, 8.0, 7.0, 6.0, 5.0]
    assert not hasattr(robot, "_external_" + "joint_torque_pub")
    assert not hasattr(robot, "_external_" + "wrench_pub")
    assert not hasattr(robot, "_gripper_" + "force_" + "state_pub")


def test_follower_publish_state_loop_skips_px4_arm_state_below_policy_joint_count(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    published = {"state": [], "px4": []}
    logged = []

    class FakeNow:
        nanoseconds = 1_234_567_890

        def to_msg(self):
            return "stamp"

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeNow())
    robot.get_logger = lambda: types.SimpleNamespace(warn=lambda message: logged.append(message))
    robot.ids = np.array([0, 1])
    robot._arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["state"].append(msg))
    robot._px4_arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["px4"].append(msg))
    robot._sync_status_pub = types.SimpleNamespace(publish=lambda _msg: None)
    sync = importlib.import_module("acetele.utils.teleop_sync")
    robot._sync = sync.FollowerSyncController(int(1e9))
    robot._sync.mode = sync.LeaderSyncMode.TRACKING
    robot._sync.status = sync.FollowerSyncStatus.TRACKING
    robot._latest_gripper_state = None
    robot._warned_invalid_px4_arm_state = False
    install_arm_assembly(robot, FakeRosArm([0.1, 0.2]))

    robot._latest_arm_state = make_joint_device_state(
        [0.1, 0.2],
        [1.1, 1.2],
        [2.1, 2.2],
    )
    module.AceFollowerROS2Robot._publish_state_loop(robot)
    robot._latest_arm_state = make_joint_device_state(
        [0.3, 0.4],
        [1.3, 1.4],
        [2.3, 2.4],
    )
    module.AceFollowerROS2Robot._publish_state_loop(robot)

    assert len(published["state"]) == 2
    assert published["px4"] == []
    assert len(logged) == 1
    assert "requires at least 4 arm joints" in logged[0]
