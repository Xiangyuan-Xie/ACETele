"""Exercise EAA timers with real rclpy and a frozen, rewound ROS clock.

Domain 78 is isolated from the live robots (domain 77). Real serial/runtime
connection paths are blocked; sessions supply immutable in-memory samples.
"""
from __future__ import annotations

import contextlib
import time
from types import SimpleNamespace
from unittest import mock

import pytest
import rclpy
from rclpy.clock import ClockType
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time

from acetele.hardware.buses import BusActor, SerialTransport
from acetele.runtime import RobotRuntime, RuntimeSafetyState
from acetele.runtime.teleop import FollowerSyncStatus, LeaderSyncMode, TeleopMode
from eaa_robot_ros2 import runtime_follower_node as follower_module
from eaa_robot_ros2 import runtime_leader_node as leader_module

NAMES = ("Joint_1", "Joint_2", "Joint_3", "Joint_4")
SAMPLE = SimpleNamespace(joints={"single": SimpleNamespace(
    names=NAMES, positions=(0.0,) * 4, velocities=(0.0,) * 4,
    efforts=(0.0,) * 4,
)})


class MemoryRuntime:
    spec = SimpleNamespace(arms=(SimpleNamespace(name="single", end_effector=None),))
    joint_groups = {"single": None}

    def diagnostics(self):
        return SimpleNamespace(safety=SimpleNamespace(state=RuntimeSafetyState.HOLD))


class MemoryLeaderSession:
    def __init__(self, runtime, **kwargs):
        self.runtime = runtime
        self.teleop_mode = TeleopMode.JOINT
        self.mode = LeaderSyncMode.TRACKING
        self.follower_status = FollowerSyncStatus.READY
        self.arm_names = NAMES

    def connect(self):
        pass  # Explicitly in-memory; all real connection paths are patched to fail.

    def step(self, **kwargs):
        return SAMPLE

    def try_start_tracking(self, state):
        return False


class MemoryFollowerSession:
    runtime = MemoryRuntime()
    teleop_mode = TeleopMode.JOINT
    end_effector_names = ()
    status = FollowerSyncStatus.READY

    def read(self, **kwargs):
        return SAMPLE


class CapturedPublisher:
    def __init__(self):
        self.messages = []
        self.monotonic_times = []

    def publish(self, message):
        self.messages.append(message)
        self.monotonic_times.append(time.monotonic())


def initialize_node_clock(node, name, context):
    Node.__init__(node, name, context=context, parameter_overrides=[
        Parameter("use_sim_time", value=True),
    ])
    node._closed = True  # Teardown must never invoke a hardware-session close.


class LeaderProbe(leader_module.RuntimeLeaderNode):
    def __init__(self, context):
        initialize_node_clock(self, "eaa_clock_regression_leader", context)
        leader_module.RuntimeLeaderNode._initialize(self, MemoryRuntime())
        self.capture = CapturedPublisher()
        self._arm_command_pub = self.capture


class FollowerProbe(follower_module.RuntimeFollowerNode):
    def __init__(self, context):
        initialize_node_clock(self, "eaa_clock_regression_follower", context)
        self._session = MemoryFollowerSession()
        follower_module.RuntimeFollowerNode._initialize_interfaces(self)
        self.capture = CapturedPublisher()
        self._arm_state_pub = self.capture


@pytest.fixture
def isolated_ros(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "78")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    context = Context()
    forbidden = AssertionError("Clock regression test attempted hardware access")
    with contextlib.ExitStack() as stack:
        guards = [stack.enter_context(mock.patch.object(cls, "connect", side_effect=forbidden))
                  for cls in (RobotRuntime, BusActor, SerialTransport)]
        guards.append(stack.enter_context(mock.patch("serial.Serial", side_effect=forbidden)))
        stack.enter_context(mock.patch.object(
            Node, "create_publisher", side_effect=lambda *args, **kwargs: CapturedPublisher()
        ))
        stack.enter_context(mock.patch.object(leader_module, "LeaderTeleopSession", MemoryLeaderSession))
        rclpy.init(context=context, domain_id=78)
        try:
            yield context
        finally:
            context.try_shutdown()
            for guard in guards:
                guard.assert_not_called()


def spin_for(executor, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.01)


@pytest.mark.parametrize("probe_type", [LeaderProbe, FollowerProbe], ids=["leader", "follower"])
def test_eaa_control_survives_frozen_and_backward_ros_time(isolated_ros, probe_type):
    node = probe_type(isolated_ros)
    executor = SingleThreadedExecutor(context=isolated_ros)
    executor.add_node(node)
    try:
        assert node.get_clock().clock_type == ClockType.ROS_TIME
        assert node.get_clock().ros_time_is_active
        node.get_clock().set_ros_time_override(Time(seconds=10, clock_type=ClockType.ROS_TIME))
        spin_for(executor, 0.3)
        first_count = len(node.capture.messages)
        assert first_count >= 10, "Control stalled while the ROS clock was frozen"
        assert {m.header.stamp.sec for m in node.capture.messages} == {10}

        node.get_clock().set_ros_time_override(Time(seconds=5, clock_type=ClockType.ROS_TIME))
        spin_for(executor, 0.3)
        rewound_messages = node.capture.messages[first_count:]
        assert len(rewound_messages) >= 10, "Control stalled after the ROS clock rewound"
        assert {m.header.stamp.sec for m in rewound_messages} == {5}
        assert node.get_clock().now().nanoseconds == 5_000_000_000
        assert all(tuple(m.name) == NAMES for m in node.capture.messages)
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()


def test_default_ros_timer_reproduces_the_frozen_clock_failure(isolated_ros):
    """Negative control: the old default ROS-clock scheduler really stalls."""
    node = Node("eaa_clock_regression_negative_control", context=isolated_ros,
                parameter_overrides=[Parameter("use_sim_time", value=True)])
    executor = SingleThreadedExecutor(context=isolated_ros)
    executor.add_node(node)
    calls = []
    try:
        node.get_clock().set_ros_time_override(Time(seconds=10, clock_type=ClockType.ROS_TIME))
        node.create_timer(0.01, lambda: calls.append(time.monotonic()))
        spin_for(executor, 0.12)
        assert calls == []
        node.get_clock().set_ros_time_override(Time(seconds=5, clock_type=ClockType.ROS_TIME))
        spin_for(executor, 0.12)
        assert calls == []
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
