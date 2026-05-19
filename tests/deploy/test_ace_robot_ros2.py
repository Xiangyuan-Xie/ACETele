import importlib
import sys
import types
from pathlib import Path

import numpy as np


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

    rclpy_node_module = types.ModuleType("rclpy.node")
    rclpy_node_module.Node = FakeNode

    rclpy_qos_module = types.ModuleType("rclpy.qos")
    rclpy_qos_module.QoSProfile = lambda depth: ("qos", depth)
    rclpy_qos_module.qos_profile_sensor_data = ("qos", "sensor_data")

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
        def __init__(self):
            self.timestamp = 0
            self.arm_position = []

    px4_msgs_msg_module.ArmJointState = FakeArmJointState
    px4_msgs_msg_module.VehicleLandDetected = type("VehicleLandDetected", (), {})

    monkeypatch.setitem(sys.modules, "rclpy", rclpy_module)
    monkeypatch.setitem(sys.modules, "rclpy.node", rclpy_node_module)
    monkeypatch.setitem(sys.modules, "rclpy.qos", rclpy_qos_module)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs_module)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg_module)
    monkeypatch.setitem(sys.modules, "px4_msgs", px4_msgs_module)
    monkeypatch.setitem(sys.modules, "px4_msgs.msg", px4_msgs_msg_module)
    return FakeNode


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
        def __init__(self, path, backend_override=None):
            captured["config_path"] = path
            captured["backend_override"] = backend_override

    monkeypatch.setattr(module, "ConfigLoader", FakeConfigLoader)

    module.main()

    assert captured["declared"] == ("config_path", "")
    assert captured["config_path"] == config_path
    assert captured["backend_override"] == "ros2"
    assert captured["closed"]
    assert captured["destroyed"]


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


def test_follower_command_callback_defers_initial_sync(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    robot = module.AceFollowerROS2Robot.__new__(module.AceFollowerROS2Robot)
    robot._is_synced = False
    robot._last_command_ns = None
    robot._heartbeat_lost = True
    robot._heartbeat_timeout_ns = int(1e9)
    robot._latest_state = None
    robot.move_calls = []
    robot.set_calls = []
    robot.state_published = False

    class FakeClock:
        @property
        def nanoseconds(self):
            return 123

    robot.get_clock = lambda: types.SimpleNamespace(now=lambda: FakeClock())
    robot.get_logger = lambda: types.SimpleNamespace(info=lambda _message: None)
    robot.move_position = lambda positions: robot.move_calls.append(list(positions))
    robot.set_position = lambda positions: robot.set_calls.append(list(positions))
    robot.act = lambda: (np.zeros(2), np.zeros(2), np.zeros(2))
    robot._publish_state = lambda *_args: setattr(robot, "state_published", True)

    msg = types.SimpleNamespace(position=[1.0, 2.0])

    module.AceFollowerROS2Robot._command_callback(robot, msg)

    assert robot.move_calls == []
    assert robot.set_calls == []
    assert robot._pending_sync_position == [1.0, 2.0]
    assert robot._last_command_ns == 123
    assert not robot._heartbeat_lost

    module.AceFollowerROS2Robot._control_loop(robot)

    assert robot.move_calls == [[1.0, 2.0]]
    assert robot._is_synced
    assert robot._pending_sync_position is None
    assert robot._latest_state is not None
    assert not robot.state_published

    module.AceFollowerROS2Robot._publish_state_loop(robot)

    assert robot.state_published


def test_follower_uses_100hz_publish_timer_without_changing_control_rate(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_follower.ace_follower_ros2"))
    timers = []
    declared = []
    parameters = {
        "control_rate": 250.0,
        "publish_rate": 100.0,
        "heartbeat_timeout": 1.0,
    }

    monkeypatch.setattr(module.AceFollowerRobot, "__init__", lambda self, _config_loader: None)
    monkeypatch.setattr(
        module.AceFollowerRobot,
        "act",
        lambda self: (np.zeros(5), np.zeros(5), np.zeros(5)),
    )
    monkeypatch.setattr(module.AceFollowerRobot, "move_position", lambda self, _positions: None)
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
    monkeypatch.setattr(
        module.AceFollowerROS2Robot,
        "create_publisher",
        lambda self, *_args: types.SimpleNamespace(publish=lambda _msg: None),
    )
    monkeypatch.setattr(
        module.AceFollowerROS2Robot,
        "create_subscription",
        lambda self, *_args: object(),
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

    robot = module.AceFollowerROS2Robot(config_loader=None)

    assert ("publish_rate", 100.0) in declared
    assert robot._control_rate == 250.0
    assert robot._publish_rate == 100.0
    assert (1.0 / 250.0, "_control_loop") in timers
    assert (1.0 / 100.0, "_publish_state_loop") in timers


def test_leader_uses_100hz_publish_timer_without_changing_control_rate(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    timers = []
    declared = []
    parameters = {
        "control_rate": 250.0,
        "publish_rate": 100.0,
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
    monkeypatch.setattr(
        module.AceLeaderROS2Robot,
        "create_publisher",
        lambda self, *_args: types.SimpleNamespace(publish=lambda _msg: None),
    )
    monkeypatch.setattr(
        module.AceLeaderROS2Robot,
        "create_subscription",
        lambda self, *_args: object(),
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

    robot = module.AceLeaderROS2Robot(config_loader=None)

    assert ("publish_rate", 100.0) in declared
    assert robot._control_rate == 250.0
    assert robot._publish_rate == 100.0
    assert (1.0 / 250.0, "_control_loop") in timers
    assert (1.0 / 100.0, "_publish_command_loop") in timers


def test_leader_control_loop_caches_command_for_publish_timer(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("acetele.robot.ace_leader.ace_leader_ros2"))
    robot = module.AceLeaderROS2Robot.__new__(module.AceLeaderROS2Robot)
    robot._is_started = True
    robot._is_landed = False
    robot._is_ended = False
    robot._latest_command = None
    robot.command_published = False

    joint_pos = np.array([0.1, 0.2])
    joint_vel = np.array([1.1, 1.2])
    joint_effort = np.array([2.1, 2.2])

    robot.act = lambda: (joint_pos, joint_vel, joint_effort)
    robot._publish_command = lambda *_args: setattr(robot, "command_published", True)

    module.AceLeaderROS2Robot._control_loop(robot)

    assert not robot.command_published
    cached_pos, cached_vel, cached_effort = robot._latest_command
    assert cached_pos is joint_pos
    assert cached_vel is joint_vel
    assert cached_effort is joint_effort

    module.AceLeaderROS2Robot._publish_command_loop(robot)

    assert robot.command_published


def test_follower_publish_state_dual_publishes_joint_state_and_px4_arm_state(monkeypatch):
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
    robot._equipments = types.SimpleNamespace(single_arm=types.SimpleNamespace(ids=[0, 1, 2, 3, 4]))
    robot._state_pub = types.SimpleNamespace(publish=lambda msg: published["state"].append(msg))
    robot._px4_arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["px4"].append(msg))
    robot._warned_invalid_px4_arm_state_length = False

    joint_pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    joint_vel = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    joint_effort = np.array([2.1, 2.2, 2.3, 2.4, 2.5])

    module.AceFollowerROS2Robot._publish_state(robot, joint_pos, joint_vel, joint_effort)

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
    assert px4_msg.arm_position == joint_pos.tolist()
    assert not hasattr(px4_msg, "arm_velocity")
    assert logged == []


def test_follower_publish_state_keeps_joint_state_velocity_off_px4_arm_state(monkeypatch):
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
    robot._equipments = types.SimpleNamespace(single_arm=types.SimpleNamespace(ids=[0, 1, 2, 3, 4]))
    robot._state_pub = types.SimpleNamespace(publish=lambda msg: published["state"].append(msg))
    robot._px4_arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["px4"].append(msg))
    robot._warned_invalid_px4_arm_state_length = False

    joint_pos = np.zeros(5)
    joint_effort = np.zeros(5)

    module.AceFollowerROS2Robot._publish_state(
        robot,
        joint_pos,
        np.array([0.2, 0.4, 2.0, np.nan, -2.0]),
        joint_effort,
    )
    module.AceFollowerROS2Robot._publish_state(
        robot,
        joint_pos,
        np.array([0.8, 1.0, 2.0, 0.4, -2.0]),
        joint_effort,
    )

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
        joint_pos.tolist(),
        joint_pos.tolist(),
    ]
    assert not hasattr(published["px4"][0], "arm_velocity")
    assert not hasattr(published["px4"][1], "arm_velocity")
    assert logged == []


def test_follower_publish_state_skips_px4_arm_state_when_not_five_axis(monkeypatch):
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
    robot._equipments = types.SimpleNamespace(single_arm=types.SimpleNamespace(ids=[0, 1]))
    robot._state_pub = types.SimpleNamespace(publish=lambda msg: published["state"].append(msg))
    robot._px4_arm_state_pub = types.SimpleNamespace(publish=lambda msg: published["px4"].append(msg))
    robot._warned_invalid_px4_arm_state_length = False

    module.AceFollowerROS2Robot._publish_state(
        robot,
        np.array([0.1, 0.2]),
        np.array([1.1, 1.2]),
        np.array([2.1, 2.2]),
    )
    module.AceFollowerROS2Robot._publish_state(
        robot,
        np.array([0.3, 0.4]),
        np.array([1.3, 1.4]),
        np.array([2.3, 2.4]),
    )

    assert len(published["state"]) == 2
    assert published["px4"] == []
    assert len(logged) == 1
    assert "expects 5 joints" in logged[0]
