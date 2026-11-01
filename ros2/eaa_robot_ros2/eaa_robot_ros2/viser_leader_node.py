"""Operates an EAA ROS 2 follower through a Viser virtual joint leader.

The configuration supplies the model only; this node never constructs a robot
runtime or opens a serial port. The independently launched follower owns motion
limits, hardware, and the emergency stop. Use a mock follower for initial tests.

Example after building the ROS package:
    ros2 run eaa_robot_ros2 viser_leader_node --ros-args \
        -p config_path:=/absolute/path/to/eaa_follower.toml

Open http://127.0.0.1:8080 and explicitly press Start tracking. Each start waits
for a new follower synchronization acknowledgement and seeds the virtual arm
from measured feedback. Hold, browser disconnection, invalid feedback, or a
timeout revokes tracking. Reconnection never resumes motion automatically.

Joint values are radians. Timing uses a monotonic clock; only message headers
use the ROS clock. The existing /ace_leader and /ace_follower topic names remain
compatible with RuntimeFollowerNode and may be changed using ROS remappings.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import enum
import math
import pathlib
import threading
import time

import numpy as np
import rclpy
from rclpy import clock as ros_clock
from rclpy import duration
from rclpy import executors
from rclpy import node
from rclpy import qos
from sensor_msgs import msg as sensor_msgs
from std_msgs import msg as std_msgs
import viser
from viser import extras
import yourdfpy

from acetele import config

_JOINT_NAMES = ("Joint_1", "Joint_2", "Joint_3", "Joint_4")
_FOLLOWER_STATUSES = frozenset(
    ("idle", "ready", "tracking", "hold", "lost", "fault")
)
_ARM_COMMAND_TOPIC = "/ace_leader/arm/command"
_SYNC_MODE_TOPIC = "/ace_leader/arm/sync_mode"
_ARM_STATE_TOPIC = "/ace_follower/arm/state"
_SYNC_STATUS_TOPIC = "/ace_follower/arm/sync_status"


class _Phase(enum.Enum):
    """Identifies the local, explicitly armed synchronization phase."""

    HOLD = "hold"
    WAIT_BARRIER = "waiting for hold acknowledgement"
    WAIT_READY = "waiting for ready acknowledgement"
    TRACKING = "tracking"


@dataclasses.dataclass(frozen=True)
class _Action:
    """Carries one timer decision without performing transport or UI work.

    Attributes:
        mode: Optional follower mode to publish before any joint command.
        seed: Optional measured pose used to reset all virtual targets.
        command: Optional full joint command in canonical order, in radians.
    """

    mode: str | None = None
    seed: tuple[float, ...] | None = None
    command: tuple[float, ...] | None = None


class _LeaderController:
    """Checks feedback and synchronizes one leader without owning I/O.

    All methods run on the single ROS executor thread. Each Start first requests
    HOLD (or IDLE when the follower is already holding) and waits for that new
    acknowledgement. Only then can a newly received READY authorize tracking.
    This barrier prevents a retained READY sample from arming a new session.

    Attributes:
        phase: Current local synchronization phase.
        reason: Human-readable explanation of the current phase.
        feedback: Latest valid measured positions, or None after invalid input.
        status: Latest recognized follower synchronization status.
    """

    def __init__(
        self,
        limits: tuple[tuple[float, float], ...],
        feedback_timeout: float,
        sync_timeout: float,
    ) -> None:
        """Initializes a controller in HOLD.

        Args:
            limits: Four finite, increasing joint limit pairs, in radians.
            feedback_timeout: Maximum local feedback receipt age, in seconds.
            sync_timeout: Maximum acknowledgement wait, in seconds.

        Raises:
            ValueError: Limits or timeouts are invalid.
        """
        if len(limits) != len(_JOINT_NAMES) or any(
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower >= upper
            for lower, upper in limits
        ):
            raise ValueError("Expected four finite, increasing joint limits")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (feedback_timeout, sync_timeout)
        ):
            raise ValueError("Controller timeouts must be finite and positive")
        self.phase = _Phase.HOLD
        self.reason = "Connect a browser and press Start tracking"
        self.feedback: tuple[float, ...] | None = None
        self.status: str | None = None
        self._limits = limits
        self._feedback_timeout = feedback_timeout
        self._sync_timeout = sync_timeout
        self._feedback_time = -math.inf
        self._status_revision = 0
        self._barrier_revision = 0
        self._barrier_status = "hold"
        self._phase_time = 0.0
        self._last_tick: float | None = None
        self._tracking_confirmed = False
        self._input_error: str | None = None

    def valid_positions(self, positions: Sequence[float]) -> bool:
        """Returns whether a full joint vector is finite and within URDF limits.

        Args:
            positions: Candidate canonical joint positions, in radians.

        Returns:
            Whether every canonical position lies within its finite bounds.
        """
        return len(positions) == len(self._limits) and all(
            math.isfinite(value) and lower <= value <= upper
            for value, (lower, upper) in zip(positions, self._limits)
        )

    def feedback_is_fresh(self, now: float) -> bool:
        """Checks the local receipt age of the latest valid feedback sample.

        Args:
            now: Local monotonic time, in seconds.

        Returns:
            Whether a valid sample arrived within the feedback timeout.
        """
        return (
            self.feedback is not None
            and now - self._feedback_time <= self._feedback_timeout
        )

    def receive_feedback(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        now: float,
    ) -> None:
        """Validates and stores a sample using local monotonic receipt time.

        Args:
            names: Unique canonical joint names in any order.
            positions: Measured positions corresponding to names, in radians.
            now: Local monotonic receipt time, in seconds.
        """
        if (
            len(names) != len(_JOINT_NAMES)
            or len(positions) != len(_JOINT_NAMES)
            or set(names) != set(_JOINT_NAMES)
        ):
            self.feedback = None
            self._input_error = "Feedback has missing or duplicate joint names"
            return
        mapping = dict(zip(names, positions))
        ordered = tuple(float(mapping[name]) for name in _JOINT_NAMES)
        if not self.valid_positions(ordered):
            self.feedback = None
            self._input_error = "Feedback is non-finite or outside URDF limits"
            return
        self.feedback = ordered
        self._feedback_time = now

    def receive_status(self, status: str) -> None:
        """Stores a synchronization status and advances the receipt revision.

        Args:
            status: Follower status string; unknown values invalidate it.
        """
        self.status = status if status in _FOLLOWER_STATUSES else None
        self._status_revision += 1
        if self.status is None:
            self._input_error = "Follower sent an unknown status"
        elif self.phase == _Phase.TRACKING and (
            status not in ("ready", "tracking")
            or (status == "ready" and self._tracking_confirmed)
        ):
            self._input_error = (
                "Follower revoked tracking; manual restart required"
            )

    def hold(self, reason: str) -> _Action:
        """Revokes tracking and returns a HOLD publication request.

        Args:
            reason: Operator-facing reason for revoking command publication.

        Returns:
            A HOLD action containing no joint command.
        """
        self.phase = _Phase.HOLD
        self.reason = reason
        self._tracking_confirmed = False
        self._input_error = None
        return _Action(mode="hold")

    def step(
        self,
        now: float,
        target: Sequence[float],
        *,
        start: bool,
        hold: bool,
        connected: bool,
        peers_ready: bool,
    ) -> _Action:
        """Advances the handshake or produces one periodic tracking command.

        Args:
            now: Local monotonic timer time, in seconds.
            target: Latest complete virtual joint target, in radians.
            start: Whether the operator requested a new synchronization cycle.
            hold: Whether HOLD or browser disconnection was requested.
            connected: Whether a browser is currently connected.
            peers_ready: Whether exactly one compatible follower is discovered.

        Returns:
            Transport and UI actions to execute on the ROS executor thread.
        """
        delayed = (
            self._last_tick is not None
            and now - self._last_tick > self._feedback_timeout
        )
        self._last_tick = now
        if hold:
            return self.hold("Hold requested; press Start to synchronize again")
        active = self.phase != _Phase.HOLD
        if not active and not start:
            return _Action()
        if self._input_error is not None:
            return self.hold(self._input_error)
        if not connected:
            return self.hold("Browser disconnected; manual restart required")
        if not peers_ready:
            return self.hold("Waiting for exactly one follower and one leader")
        if delayed:
            return self.hold("Control loop stalled; manual restart required")
        if not self.feedback_is_fresh(now):
            return self.hold("Feedback is missing, invalid, or stale")
        if self.status is None or self.status == "fault":
            return self.hold("Follower status is unknown or faulted")
        if start and self.phase == _Phase.HOLD:
            # Choose a different status so a healthy follower must acknowledge.
            self._barrier_status = "idle" if self.status == "hold" else "hold"
            self._barrier_revision = self._status_revision
            self._phase_time = now
            self.phase = _Phase.WAIT_BARRIER
            self.reason = "Waiting for a new synchronization barrier"
            return _Action(mode=self._barrier_status)
        if self.phase in (_Phase.WAIT_BARRIER, _Phase.WAIT_READY):
            if now - self._phase_time > self._sync_timeout:
                return self.hold("Synchronization timed out; press Start again")
            if self._status_revision <= self._barrier_revision:
                return _Action()
            if self.phase == _Phase.WAIT_BARRIER:
                if self.status != self._barrier_status:
                    return _Action()
                self.phase = _Phase.WAIT_READY
                self._barrier_revision = self._status_revision
                self._phase_time = now
                self.reason = "Waiting for the follower READY acknowledgement"
                return _Action(mode="sync_request")
            if self.status != "ready":
                return _Action()
            self.phase = _Phase.TRACKING
            self._phase_time = now
            self.reason = "Tracking; use Hold before leaving the browser"
            # The first target is measured feedback, never an old slider value.
            return _Action(mode="tracking", seed=self.feedback)
        if self.phase == _Phase.TRACKING:
            if self.status == "tracking":
                self._tracking_confirmed = True
            elif (
                self.status != "ready"
                or self._tracking_confirmed
                or now - self._phase_time > self._sync_timeout
            ):
                return self.hold("Follower left tracking; press Start again")
            if not self.valid_positions(target):
                return self.hold("Target is invalid or outside URDF limits")
            return _Action(command=tuple(float(value) for value in target))
        return _Action()


class _Requests:
    """Stores bounded, thread-safe UI requests with HOLD taking priority."""

    def __init__(self) -> None:
        """Initializes an empty request mailbox."""
        self._lock = threading.Lock()
        self._start = False
        self._hold = False

    def start(self, event: viser.GuiEvent) -> None:
        """Queues Start without performing ROS work in the GUI thread.

        Args:
            event: Unused Viser button event.
        """
        del event
        with self._lock:
            if not self._hold:
                self._start = True

    def hold(self, event: viser.GuiEvent | viser.ClientHandle) -> None:
        """Queues HOLD and cancels pending Start, including on disconnection.

        Args:
            event: Unused button event or disconnected browser client.
        """
        del event
        with self._lock:
            self._hold = True
            self._start = False

    def take(self) -> tuple[bool, bool]:
        """Returns and clears the pending Start and Hold flags atomically."""
        with self._lock:
            result = (self._start, self._hold)
            self._start = False
            self._hold = False
            return result


class _Viewer:
    """Displays the virtual arm, feedback, and operator controls."""

    def __init__(
        self,
        server: viser.ViserServer,
        model: yourdfpy.URDF,
        limits: tuple[tuple[float, float], ...],
        requests: _Requests,
    ) -> None:
        """Builds the interface without starting robot or ROS work.

        Args:
            server: Viser server hosting the operator interface.
            model: Validated four-joint EAA display model.
            limits: Canonical joint position limits, in radians.
            requests: Bounded mailbox consumed by the ROS timer.
        """
        self._lock = threading.Lock()
        self._target = tuple(
            min(max(0.0, lower), upper) for lower, upper in limits
        )
        server.initial_camera.position = (0.9, 0.9, 0.7)
        server.initial_camera.look_at = (-0.1, 0.0, 0.15)
        self._robot = extras.ViserUrdf(server, model)
        server.gui.add_markdown(
            "**EAA ROS 2 virtual leader**\n\n"
            "Use a mock follower for initial testing. Start tracking first "
            "aligns the virtual arm to measured feedback. Hold or closing "
            "a browser revokes tracking. Angles are radians."
        )
        server.gui.add_button("Start tracking").on_click(requests.start)
        server.gui.add_button("Hold").on_click(requests.hold)
        self._feedback = server.gui.add_markdown("Waiting for ROS feedback...")
        self._sliders = [
            server.gui.add_slider(
                f"{name} (rad)",
                min=lower,
                max=upper,
                step=0.001,
                initial_value=value,
                disabled=True,
            )
            for name, (lower, upper), value in zip(
                _JOINT_NAMES, limits, self._target
            )
        ]
        for slider in self._sliders:
            slider.on_update(self._update_target)
        server.on_client_disconnect(requests.hold)
        self._robot.update_cfg(np.asarray(self._target))

    def _update_target(self, event: viser.GuiEvent) -> None:
        """Copies slider values into the latest target without transport work.

        Args:
            event: Unused Viser slider update event.
        """
        del event
        with self._lock:
            self._target = tuple(float(item.value) for item in self._sliders)

    def target(self) -> tuple[float, ...]:
        """Returns the complete latest target under the UI mailbox lock."""
        with self._lock:
            return self._target

    def seed(self, positions: tuple[float, ...]) -> None:
        """Resets all sliders before admitting user tracking commands.

        Args:
            positions: Validated measured joint positions, in radians.
        """
        for slider, value in zip(self._sliders, positions):
            slider.value = value
        with self._lock:
            self._target = positions

    def render(self, controller: _LeaderController) -> None:
        """Updates the target model, control availability, and feedback table.

        Args:
            controller: Controller whose state is owned by the ROS timer.
        """
        target = self.target()
        self._robot.update_cfg(np.asarray(target))
        for slider in self._sliders:
            slider.disabled = controller.phase != _Phase.TRACKING
        fresh = controller.feedback_is_fresh(time.monotonic())
        status = controller.status or "unknown"
        feedback_heading = "Feedback (rad)"
        if not fresh:
            status = f"offline/stale (last received: {status})"
            feedback_heading = "Last feedback (rad; stale)"
        rows = [
            f"**Leader: {controller.phase.value}** | Follower: {status}",
            "",
            controller.reason,
            "",
            f"| Joint | Target (rad) | {feedback_heading} |",
            "| --- | ---: | ---: |",
        ]
        for index, name in enumerate(_JOINT_NAMES):
            measured = (
                f"{controller.feedback[index]:.4f}"
                if controller.feedback is not None
                else "unavailable"
            )
            rows.append(f"| {name} | {target[index]:.4f} | {measured} |")
        self._feedback.content = "\n".join(rows)


def _load_model(
    config_path: str,
) -> tuple[yourdfpy.URDF, tuple[tuple[float, float], ...]]:
    """Loads and validates the display model without constructing a runtime.

    Args:
        config_path: Explicit EAA RobotSpec TOML path.

    Returns:
        The display model and canonical joint position limits, in radians.

    Raises:
        ValueError: The path, topology, joint units, or limits are invalid.
    """
    if not config_path:
        raise ValueError("Pass an explicit config_path ROS parameter")
    path = pathlib.Path(config_path).expanduser().resolve()
    spec = config.load_robot_spec(path)
    if spec.model != "eaa_follower" or len(spec.arms) != 1:
        raise ValueError("Expected an EAA follower specification with one arm")
    if tuple(item.name for item in spec.arms[0].joints) != _JOINT_NAMES:
        raise ValueError(f"Expected canonical joint order {_JOINT_NAMES}")
    if not spec.urdf_path:
        raise ValueError("The EAA specification must provide urdf_path")
    model = yourdfpy.URDF.load(spec.urdf_path)
    if tuple(model.actuated_joint_names) != _JOINT_NAMES:
        raise ValueError("URDF actuated joint order must match the EAA arm")
    limits = []
    for name in _JOINT_NAMES:
        joint = model.joint_map[name]
        if joint.type != "revolute" or joint.limit is None:
            raise ValueError(f"{name} requires bounded revolute joint limits")
        lower, upper = joint.limit.lower, joint.limit.upper
        if lower is None or upper is None:
            raise ValueError(f"{name} has missing position limits")
        limits.append((float(lower), float(upper)))
    return model, tuple(limits)


class ViserLeaderNode(node.Node):
    """Connects the virtual leader UI to the EAA follower ROS topics."""

    def __init__(self) -> None:
        """Creates ROS interfaces and a local Viser server, initially in HOLD.

        Raises:
            ValueError: A model or ROS parameter is invalid.
        """
        super().__init__("viser_leader_node")
        self._server: viser.ViserServer | None = None
        self._closed = False
        try:
            self._initialize()
        except BaseException:
            try:
                self.close()
            finally:
                self.destroy_node()
            raise

    def _initialize(self) -> None:
        """Validates configuration and creates bounded ROS/UI interfaces."""
        defaults = {
            "config_path": "",
            "host": "127.0.0.1",
            "port": 8080,
            "command_rate": 25.0,
            "display_rate": 5.0,
            "feedback_timeout": 0.5,
            "sync_timeout": 2.0,
            "command_lifespan": 0.05,
            "state_lifespan": 0.5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        values = {
            name: self.get_parameter(name).value for name in defaults
        }
        for name in (
            "command_rate", "display_rate", "feedback_timeout",
            "sync_timeout", "command_lifespan", "state_lifespan",
        ):
            value = float(values[name])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 10.0 <= values["command_rate"] <= 100.0:
            raise ValueError("command_rate must be between 10 and 100 Hz")
        if values["display_rate"] > values["command_rate"]:
            raise ValueError("display_rate must not exceed command_rate")
        if not 1 <= values["port"] <= 65535:
            raise ValueError("port must be between 1 and 65535")
        model, limits = _load_model(str(values["config_path"]))
        self._controller = _LeaderController(
            limits, float(values["feedback_timeout"]),
            float(values["sync_timeout"]),
        )
        command_qos = qos.QoSProfile(
            depth=1,
            history=qos.HistoryPolicy.KEEP_LAST,
            reliability=qos.ReliabilityPolicy.BEST_EFFORT,
            durability=qos.DurabilityPolicy.VOLATILE,
            lifespan=duration.Duration(seconds=values["command_lifespan"]),
        )
        state_qos = qos.QoSProfile(
            depth=1,
            history=qos.HistoryPolicy.KEEP_LAST,
            reliability=qos.ReliabilityPolicy.BEST_EFFORT,
            durability=qos.DurabilityPolicy.VOLATILE,
            lifespan=duration.Duration(seconds=values["state_lifespan"]),
        )
        sync_qos = qos.QoSProfile(
            depth=1,
            history=qos.HistoryPolicy.KEEP_LAST,
            reliability=qos.ReliabilityPolicy.RELIABLE,
            durability=qos.DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._command_pub = self.create_publisher(
            sensor_msgs.JointState, _ARM_COMMAND_TOPIC, command_qos
        )
        self._mode_pub = self.create_publisher(
            std_msgs.String, _SYNC_MODE_TOPIC, sync_qos
        )
        self._state_sub = self.create_subscription(
            sensor_msgs.JointState, _ARM_STATE_TOPIC,
            self._receive_feedback, state_qos,
        )
        self._status_sub = self.create_subscription(
            std_msgs.String, _SYNC_STATUS_TOPIC,
            self._receive_status, sync_qos,
        )
        self._requests = _Requests()
        self._display_period = 1.0 / values["display_rate"]
        self._last_display = -math.inf
        self._server = viser.ViserServer(
            host=str(values["host"]), port=int(values["port"])
        )
        self._viewer = _Viewer(self._server, model, limits, self._requests)
        self._steady_clock = ros_clock.Clock(
            clock_type=ros_clock.ClockType.STEADY_TIME
        )
        self._timer = self.create_timer(
            1.0 / values["command_rate"], self._tick,
            clock=self._steady_clock,
        )
        self._publish_mode("hold")
        self.get_logger().info(
            f"Viser leader: http://{values['host']}:{values['port']}; "
            "waiting for feedback and an explicit Start."
        )

    def _receive_feedback(self, message: sensor_msgs.JointState) -> None:
        """Checks feedback without trusting the remote timestamp for freshness.

        Args:
            message: Complete measured EAA joint state.
        """
        self._controller.receive_feedback(
            message.name, message.position, time.monotonic()
        )

    def _receive_status(self, message: std_msgs.String) -> None:
        """Records a follower acknowledgement on the single executor thread.

        Args:
            message: Follower synchronization status.
        """
        self._controller.receive_status(message.data)

    def _peers_ready(self) -> bool:
        """Checks for one follower and no competing leader."""
        return (
            self._command_pub.get_subscription_count() >= 1
            and self._mode_pub.get_subscription_count() >= 1
            and self.count_publishers(self._state_sub.topic_name) == 1
            and self.count_publishers(self._status_sub.topic_name) == 1
            and self.count_publishers(self._command_pub.topic_name) == 1
            and self.count_publishers(self._mode_pub.topic_name) == 1
        )

    def _publish_mode(self, mode: str) -> None:
        """Publishes a reliable synchronization transition.

        Args:
            mode: Valid follower mode selected by the controller.
        """
        message = std_msgs.String()
        message.data = mode
        self._mode_pub.publish(message)

    def _tick(self) -> None:
        """Owns synchronization and periodic command publication."""
        if self._closed or self._server is None:
            return
        now = time.monotonic()
        start, hold = self._requests.take()
        old_reason = self._controller.reason
        action = self._controller.step(
            now, self._viewer.target(), start=start, hold=hold,
            connected=bool(self._server.get_clients()),
            peers_ready=self._peers_ready(),
        )
        if action.seed is not None:
            self._viewer.seed(action.seed)
        if action.mode is not None:
            self._publish_mode(action.mode)
        if action.command is not None:
            message = sensor_msgs.JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = list(_JOINT_NAMES)
            message.position = list(action.command)
            self._command_pub.publish(message)
        if old_reason != self._controller.reason:
            self.get_logger().info(self._controller.reason)
        if (
            now - self._last_display >= self._display_period
            or old_reason != self._controller.reason
        ):
            self._viewer.render(self._controller)
            self._last_display = now

    def close(self) -> None:
        """Cancels commands, requests HOLD, and stops Viser.

        Delivery during shutdown is best effort. The follower's existing remote
        heartbeat timeout remains responsible for abrupt process termination.
        """
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "_timer"):
            self._timer.cancel()
        try:
            if self.context.ok() and hasattr(self, "_mode_pub"):
                self._publish_mode("hold")
        finally:
            if self._server is not None:
                self._server.stop()


def main(args: list[str] | None = None) -> None:
    """Runs the virtual leader with a single-threaded ROS executor.

    Args:
        args: ROS command-line arguments, or None to use process arguments.
    """
    rclpy.init(args=args)
    leader = None
    try:
        leader = ViserLeaderNode()
        rclpy.spin(leader)
    except (KeyboardInterrupt, executors.ExternalShutdownException):
        pass
    finally:
        try:
            if leader is not None:
                try:
                    leader.close()
                finally:
                    leader.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
