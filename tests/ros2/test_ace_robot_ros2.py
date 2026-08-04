import importlib
import sys
import types
from pathlib import Path

import pytest
from ace_robot_ros2.spec_validation import validate_ros2_robot_spec

from acetele.specification import (
    ArmSpec,
    BusSpec,
    BusType,
    JointSpec,
    ParallelGripperSpec,
    RobotSpec,
)

project_root = Path(__file__).resolve().parents[2]


def _ros_spec(*, model="ace_follower", joint_count=4, end_effector_count=0):
    buses = []
    arms = []
    for arm_index in range(max(1, end_effector_count)):
        bus_name = f"bus_{arm_index}"
        buses.append(
            BusSpec(
                bus_name,
                BusType.FEETECH_PACKET,
                f"mock://{bus_name}",
                1_000_000,
                100.0,
                physical_layer="ttl",
                family="hls",
            )
        )
        count = joint_count if arm_index == 0 else 1
        joints = tuple(
            JointSpec(
                f"joint_{arm_index}_{index}",
                index,
                "HL3915",
                1,
                0.0,
            )
            for index in range(count)
        )
        end_effector = None
        if arm_index < end_effector_count:
            end_effector = ParallelGripperSpec(
                bus_name,
                JointSpec(
                    f"gripper_{arm_index}",
                    count,
                    "HL3915",
                    1,
                    0.0,
                ),
                1.0,
            )
        arms.append(ArmSpec(f"arm_{arm_index}", bus_name, joints, end_effector=end_effector))
    return RobotSpec(model, tuple(buses), tuple(arms))


def test_ros2_spec_validation_rejects_unsupported_topology_without_runtime():
    with pytest.raises(RuntimeError, match="at most one end effector"):
        validate_ros2_robot_spec(
            _ros_spec(end_effector_count=2),
            expected_model="ace_follower",
            arm_capacity=14,
        )


def test_follower_ros2_spec_validation_enforces_wire_capacity():
    with pytest.raises(RuntimeError, match="between 4 and 14"):
        validate_ros2_robot_spec(
            _ros_spec(joint_count=15),
            expected_model="ace_follower",
            arm_capacity=14,
            minimum_arm_joints=4,
        )

    with pytest.raises(RuntimeError, match="between 4 and 14"):
        validate_ros2_robot_spec(
            _ros_spec(joint_count=3),
            expected_model="ace_follower",
            arm_capacity=14,
            minimum_arm_joints=4,
        )


def test_robot_launch_defaults_to_packaged_leader_hls_ttl_spec():
    source = (
        project_root / "ros2/ace_robot_ros2/launch/ace_robot.launch.py"
    ).read_text(encoding="utf-8")

    assert '"ace_leader"' in source
    assert '"feetech_hls_ttl.toml"' in source
    assert 'packaged_robot_spec("ace_leader", "feetech_hls_ttl.toml")' in source


def test_follower_separates_motion_and_session_timeout_defaults():
    parameters = (
        project_root
        / "ros2/ace_robot_ros2/config/ace_robot_params.yaml"
    ).read_text(encoding="utf-8")
    source = (
        project_root
        / "ros2/ace_robot_ros2/ace_robot_ros2/runtime_follower_node.py"
    ).read_text(encoding="utf-8")

    assert "motion_timeout: 0.1" in parameters
    assert "session_timeout: 0.5" in parameters
    assert 'declare_parameter("motion_timeout", 0.1)' in source
    assert 'declare_parameter("session_timeout", 0.5)' in source


def test_follower_adaptive_diagnostics_are_rate_limited():
    parameters = (
        project_root
        / "ros2/ace_robot_ros2/config/ace_robot_params.yaml"
    ).read_text(encoding="utf-8")
    source = (
        project_root
        / "ros2/ace_robot_ros2/ace_robot_ros2/runtime_follower_node.py"
    ).read_text(encoding="utf-8")

    assert "adaptive_diagnostic_period: 2.0" in parameters
    assert 'declare_parameter("adaptive_diagnostic_period", 2.0)' in source
    assert "adaptive_saturated" in source


def test_follower_always_holds_measured_position_on_start():
    parameters = (
        project_root
        / "ros2/ace_robot_ros2/config/ace_robot_params.yaml"
    ).read_text(encoding="utf-8")
    source = (
        project_root
        / "ros2/ace_robot_ros2/ace_robot_ros2/runtime_follower_node.py"
    ).read_text(encoding="utf-8")
    launch = (
        project_root / "ros2/ace_robot_ros2/launch/ace_robot.launch.py"
    ).read_text(encoding="utf-8")

    assert "self._session.hold_position()" in source
    assert "hold_on_start" not in parameters
    assert "hold_on_start" not in source
    assert "hold_on_start" not in launch


def test_leader_end_effector_throttling_is_runtime_configured():
    parameters = (
        project_root
        / "ros2/ace_robot_ros2/config/ace_robot_params.yaml"
    ).read_text(encoding="utf-8")
    source = (
        project_root
        / "ros2/ace_robot_ros2/ace_robot_ros2/runtime_leader_node.py"
    ).read_text(encoding="utf-8")

    assert "end_effector_publish_threshold: 0.001" in parameters
    assert "end_effector_keepalive: 0.1" in parameters
    assert 'declare_parameter("end_effector_publish_threshold", 0.001)' in source
    assert 'declare_parameter("end_effector_keepalive", 0.1)' in source


def test_leader_does_not_subscribe_to_px4_landing_state():
    source = (
        project_root
        / "ros2/ace_robot_ros2/ace_robot_ros2/runtime_leader_node.py"
    ).read_text(encoding="utf-8")

    assert "VehicleLandDetected" not in source
    assert "/fmu/out/vehicle_land_detected" not in source
    assert "observe_landed" not in source


def test_cartesian_teleop_ros_interface_is_mode_selected_and_best_effort():
    parameters = (
        project_root
        / "ros2/ace_robot_ros2/config/ace_robot_params.yaml"
    ).read_text(encoding="utf-8")
    leader = (
        project_root
        / "ros2/ace_robot_ros2/ace_robot_ros2/runtime_leader_node.py"
    ).read_text(encoding="utf-8")
    follower = (
        project_root
        / "ros2/ace_robot_ros2/ace_robot_ros2/runtime_follower_node.py"
    ).read_text(encoding="utf-8")
    package = (
        project_root / "ros2/ace_robot_ros2/package.xml"
    ).read_text(encoding="utf-8")

    assert "teleop_mode: joint" in parameters
    assert "translation_scale: 2.0" in parameters
    assert '"/ace_teleop/arm/ee_pose/command"' in leader
    assert '"/ace_teleop/arm/ee_pose/command"' in follower
    assert '"/ace_follower/arm/ee_pose/state"' in follower
    assert "ReliabilityPolicy.BEST_EFFORT" in leader
    assert "ReliabilityPolicy.BEST_EFFORT" in follower
    assert '"/fmu/in/arm_joint_state",\n            px4_qos,' in follower
    assert "lifespan=Duration(seconds=command_lifespan)" in leader
    assert "lifespan=Duration(seconds=command_lifespan)" in follower
    assert "<depend>geometry_msgs</depend>" in package
    assert "<depend>std_srvs</depend>" in package


def test_ros_nodes_expose_explicit_emergency_stop_services():
    leader = (
        project_root
        / "ros2/ace_robot_ros2/ace_robot_ros2/runtime_leader_node.py"
    ).read_text(encoding="utf-8")
    follower = (
        project_root
        / "ros2/ace_robot_ros2/ace_robot_ros2/runtime_follower_node.py"
    ).read_text(encoding="utf-8")

    assert '"/ace_leader/emergency_stop"' in leader
    assert '"/ace_leader/authorize_alignment"' in leader
    assert '"/ace_leader/start_tracking"' in leader
    assert '"/ace_follower/emergency_stop"' in follower
    assert "self._session.stop()" in leader
    assert "self._session.set_mode(LeaderSyncMode.STOP)" in follower


def test_leader_automatic_control_errors_hold_instead_of_publishing_emergency_stop():
    source = (
        project_root
        / "ros2/ace_robot_ros2/ace_robot_ros2/runtime_leader_node.py"
    ).read_text(encoding="utf-8")

    control_error = source.split("def _control_loop", 1)[1].split(
        "def _report_mode_transition", 1
    )[0]
    assert "self._session.hold()" in control_error
    assert "self._session.stop()" not in control_error


def test_follower_fault_cleanup_preserves_the_holding_target():
    source = (
        project_root
        / "ros2/ace_robot_ros2/ace_robot_ros2/runtime_follower_node.py"
    ).read_text(encoding="utf-8")
    close_method = source.split("def close(self)", 1)[1]

    assert "RuntimeSafetyState.FAULT" in close_method
    assert "self._session.close(preserve_hold=preserve_hold)" in close_method


def _load_entry_module(monkeypatch, *, config_path="robot.toml"):
    class FakeNode:
        def __init__(self, name="node"):
            self.name = name
            self.destroyed = False

        def declare_parameter(self, _name, _default):
            return types.SimpleNamespace(value=config_path)

        def destroy_node(self):
            self.destroyed = True

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda: None
    rclpy.spin = lambda _node: None
    rclpy.shutdown = lambda: None
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = FakeNode
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.node", rclpy_node)
    module_name = "ace_robot_ros2.ace_robot_node"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name), FakeNode, rclpy


def test_robot_node_requires_explicit_new_schema_config(monkeypatch):
    module, _node_type, _rclpy = _load_entry_module(monkeypatch)

    with pytest.raises(ValueError, match="config_path"):
        module._make_robot_node("")


@pytest.mark.parametrize(
    ("model", "module_suffix", "class_name"),
    (
        ("ace_follower", "runtime_follower_node", "RuntimeFollowerNode"),
        ("ace_leader", "runtime_leader_node", "RuntimeLeaderNode"),
    ),
)
def test_robot_node_dispatches_only_to_runtime_adapters(
    monkeypatch,
    model,
    module_suffix,
    class_name,
):
    module, _node_type, _rclpy = _load_entry_module(monkeypatch)
    spec = types.SimpleNamespace(model=model)
    sentinel = object()
    monkeypatch.setattr(module, "load_robot_spec", lambda _path: spec)
    adapter_module_name = (
        "ace_robot_ros2." + module_suffix
    )
    adapter_module = types.ModuleType(adapter_module_name)
    setattr(adapter_module, class_name, lambda received: sentinel if received is spec else None)
    monkeypatch.setitem(sys.modules, adapter_module_name, adapter_module)

    assert module._make_robot_node("robot.toml") is sentinel


def test_main_preserves_spin_error_and_cleans_runtime(monkeypatch):
    module, node_type, rclpy = _load_entry_module(monkeypatch)
    events = []

    class RobotNode(node_type):
        def close(self):
            events.append("close")

        def destroy_node(self):
            events.append("destroy")

    robot = RobotNode("robot")
    monkeypatch.setattr(module, "_make_robot_node", lambda _path: robot)
    rclpy.init = lambda: events.append("init")
    rclpy.spin = lambda _node: (_ for _ in ()).throw(RuntimeError("spin failed"))
    rclpy.shutdown = lambda: events.append("shutdown")

    with pytest.raises(RuntimeError, match="spin failed"):
        module.main()

    assert events == ["init", "close", "destroy", "shutdown"]
