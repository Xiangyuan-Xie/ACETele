"""Low-latency ROS 2 adapter for :class:`FollowerTeleopSession`."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import ArmJointState
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState as ROSJointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from acetele.hardware.devices import AutomaticFaultAction
from acetele.runtime import FollowerTeleopSession, RobotRuntime, RuntimeSafetyState
from acetele.runtime.teleop import FollowerSyncStatus, LeaderSyncMode, TeleopMode
from acetele.specification import DexterousHandSpec, RobotSpec

from .pose_messages import pose_from_message, pose_message
from .spec_validation import validate_ros2_robot_spec


class RuntimeFollowerNode(Node):
    """ROS 2 message adapter composed around the vendor-neutral follower runtime."""

    def __init__(self, spec: RobotSpec) -> None:
        # Validate wire schema and topology before Node construction or serial startup;
        # unsupported deployments fail without leaving ROS entities or hardware open.
        self._validate_px4_schema()
        validate_ros2_robot_spec(
            spec,
            expected_model="ace_follower",
            arm_capacity=int(ArmJointState.MAX_JOINTS),
            minimum_arm_joints=4,
        )
        node_initialized = False
        session_connected = False
        self._closed = False
        try:
            super().__init__("ace_follower_robot")
            node_initialized = True
            self.declare_parameter("motion_timeout", 0.1)
            self.declare_parameter("session_timeout", 0.5)
            self.declare_parameter("teleop_mode", TeleopMode.JOINT.value)
            self.declare_parameter("translation_scale", 2.0)
            self.declare_parameter("rotation_scale", 1.0)
            motion_timeout = float(self.get_parameter("motion_timeout").value)
            session_timeout = float(self.get_parameter("session_timeout").value)
            if any(
                not np.isfinite(value) or value <= 0.0
                for value in (motion_timeout, session_timeout)
            ):
                raise ValueError("follower timeouts must be finite and positive")
            if session_timeout < motion_timeout:
                raise ValueError("session_timeout must not be shorter than motion_timeout")
            motion_timeout_ns = round(motion_timeout * 1e9)
            session_timeout_ns = round(session_timeout * 1e9)
            try:
                teleop_mode = TeleopMode(str(self.get_parameter("teleop_mode").value))
            except ValueError as exc:
                raise ValueError("teleop_mode must be 'joint' or 'ee_pose'") from exc
            translation_scale = float(self.get_parameter("translation_scale").value)
            rotation_scale = float(self.get_parameter("rotation_scale").value)
            if any(
                not np.isfinite(value) or value <= 0.0
                for value in (translation_scale, rotation_scale)
            ):
                raise ValueError("Cartesian teleop scales must be finite and positive")
            runtime = RobotRuntime(
                spec,
                command_timeout_ns=motion_timeout_ns,
            )
            self._session = FollowerTeleopSession(
                runtime,
                session_timeout_ns=session_timeout_ns,
                teleop_mode=teleop_mode,
                translation_scale=translation_scale,
                rotation_scale=rotation_scale,
            )
            self._session.connect()
            session_connected = True
            self._session.hold_position()
            self._initialize_interfaces()
        except BaseException as initialization_error:
            cleanup_error: Optional[BaseException] = None
            if session_connected:
                try:
                    self._session.close()
                except BaseException as exc:
                    cleanup_error = exc
            if node_initialized:
                try:
                    self.destroy_node()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if cleanup_error is not None:
                raise initialization_error from cleanup_error
            raise

    def _initialize_interfaces(self) -> None:
        """Create QoS-separated data/sync interfaces after the runtime is connected."""

        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("command_lifespan", 0.05)
        self.declare_parameter("state_lifespan", 0.5)
        publish_rate = float(self.get_parameter("publish_rate").value)
        command_lifespan = float(self.get_parameter("command_lifespan").value)
        state_lifespan = float(self.get_parameter("state_lifespan").value)
        if not np.isfinite(publish_rate) or publish_rate <= 0.0:
            raise ValueError("publish_rate must be finite and positive")
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in (command_lifespan, state_lifespan)
        ):
            raise ValueError("follower timing parameters must be finite and positive")
        self._last_runtime_fault_message: Optional[str] = None

        # Commands and state are latest-value streams: retransmitting an old sample adds
        # latency and is less useful than the next frame. Sync transitions must arrive.
        command_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            lifespan=Duration(seconds=command_lifespan),
        )
        state_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            lifespan=Duration(seconds=state_lifespan),
        )
        px4_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        sync_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            # Sync topics carry current state, not an event log. Transient-local depth
            # one lets a restarted peer receive that state without periodic republishing.
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._end_effector_groups = tuple(self._session.end_effector_names)
        uses_dexterous_hand = any(
            isinstance(arm.end_effector, DexterousHandSpec)
            for arm in self._session.runtime.spec.arms
        )
        state_topic = (
            "/ace_follower/end_effector/state"
            if uses_dexterous_hand
            else "/ace_follower/gripper/state"
        )
        command_topic = (
            "/ace_leader/end_effector/command"
            if uses_dexterous_hand
            else "/ace_leader/gripper/command"
        )
        self._arm_state_pub = self.create_publisher(
            ROSJointState,
            "/ace_follower/arm/state",
            state_qos,
        )
        self._ee_pose_state_pub = None
        if self._session.teleop_mode == TeleopMode.EE_POSE:
            self._ee_pose_state_pub = self.create_publisher(
                PoseStamped,
                "/ace_follower/arm/ee_pose/state",
                state_qos,
            )
        self._end_effector_state_pub = self.create_publisher(
            ROSJointState,
            state_topic,
            state_qos,
        )
        self._sync_status_pub = self.create_publisher(
            String,
            "/ace_follower/arm/sync_status",
            sync_qos,
        )
        self._last_published_sync_status: Optional[FollowerSyncStatus] = None
        self._px4_state_pub = self.create_publisher(
            ArmJointState,
            "/fmu/in/arm_joint_state",
            px4_qos,
        )
        self._arm_command_sub = None
        self._ee_pose_command_sub = None
        if self._session.teleop_mode == TeleopMode.JOINT:
            self._arm_command_sub = self.create_subscription(
                ROSJointState,
                "/ace_leader/arm/command",
                self._arm_command_callback,
                command_qos,
            )
        else:
            self._ee_pose_command_sub = self.create_subscription(
                PoseStamped,
                "/ace_teleop/arm/ee_pose/command",
                self._ee_pose_command_callback,
                command_qos,
            )
        self._end_effector_command_sub = self.create_subscription(
            ROSJointState,
            command_topic,
            self._end_effector_command_callback,
            command_qos,
        )
        self._sync_mode_sub = self.create_subscription(
            String,
            "/ace_leader/arm/sync_mode",
            self._sync_mode_callback,
            sync_qos,
        )
        self._emergency_stop_service = self.create_service(
            Trigger,
            "/ace_follower/emergency_stop",
            self._emergency_stop_callback,
        )
        self._sequence = 0
        self._timer = self.create_timer(1.0 / publish_rate, self._publish_state)
        self._publish_sync_status()
        safety_state = self._session.runtime.diagnostics().safety.state.value
        self.get_logger().info(
            f"Follower RobotRuntime ROS 2 node started in {safety_state}."
        )

    def _arm_command_callback(self, message: ROSJointState) -> None:
        """Validate and replace the target consumed by the local motion cycle."""

        # The callback never clocks hardware. This removes network jitter from the bus
        # watchdog while preserving latest-value semantics and bounded callback time.
        names = tuple(message.name) or self._session.arm_names
        positions = self._finite_positions(message, len(self._session.arm_names), "arm")
        if positions is None:
            return
        try:
            self._session.write_arm(
                names,
                positions,
                now_ns=time.monotonic_ns(),
            )
        except (RuntimeError, ValueError) as exc:
            self._report_runtime_error("Ignoring invalid arm command", exc)
        self._publish_sync_status()

    def _ee_pose_command_callback(self, message: PoseStamped) -> None:
        """Convert and submit one generic Cartesian source sample without timer delay."""

        now_ns = time.monotonic_ns()
        try:
            pose = pose_from_message(message, timestamp_ns=now_ns)
            self._session.write_arm_pose(pose, now_ns=now_ns)
        except (RuntimeError, ValueError) as exc:
            self._report_runtime_error("Ignoring invalid end-effector pose command", exc)
        self._publish_sync_status()

    def _end_effector_command_callback(self, message: ROSJointState) -> None:
        """Submit end-effector motion only through the session's heartbeat gate."""

        if not self._end_effector_groups:
            return
        group_name = self._end_effector_groups[0]
        expected_names = self._session.end_effector_names[group_name]
        names = tuple(message.name) or expected_names
        positions = self._finite_positions(
            message,
            len(expected_names),
            "end effector",
        )
        if positions is None:
            return
        try:
            self._session.write_end_effector(
                group_name,
                names,
                positions,
                now_ns=time.monotonic_ns(),
            )
        except (RuntimeError, ValueError) as exc:
            self._report_runtime_error("Ignoring invalid end-effector command", exc)
        self._publish_sync_status()

    def _sync_mode_callback(self, message: String) -> None:
        """Translate a reliable sync-mode message into one session transition."""

        try:
            mode = LeaderSyncMode(message.data)
            self._session.set_mode(mode)
        except (RuntimeError, ValueError) as exc:
            self._report_runtime_error("Ignoring invalid sync mode", exc)
        self._publish_sync_status()

    def _publish_state(self) -> None:
        """Publish the coherent state snapshot maintained by the local motion loop."""

        now = self.get_clock().now()
        try:
            state = self._session.read(now_ns=time.monotonic_ns())
        except RuntimeError as exc:
            self._report_runtime_error("Follower runtime state failed", exc)
            self._publish_sync_status()
            return
        # Flatten arm assemblies in RobotSpec order. End effectors remain on their own
        # topic and never consume slots in the PX4 arm-only message.
        arm_states = tuple(state.joints[arm.name] for arm in self._session.runtime.spec.arms)
        arm_names = tuple(name for item in arm_states for name in item.names)
        arm_positions = tuple(float(value) for item in arm_states for value in item.positions)
        arm_velocities = tuple(float(value) for item in arm_states for value in item.velocities)
        arm_efforts = tuple(float(value) for item in arm_states for value in item.efforts)
        self._publish_joint_state(
            self._arm_state_pub,
            now,
            arm_names,
            arm_positions,
            arm_velocities,
            arm_efforts,
        )
        if self._session.teleop_mode == TeleopMode.EE_POSE:
            pose = self._session.end_effector_pose(
                state,
                timestamp_ns=time.monotonic_ns(),
            )
            if self._ee_pose_state_pub is None:
                raise RuntimeError("ee_pose state publisher was not initialized")
            self._ee_pose_state_pub.publish(pose_message(pose, now.to_msg()))

        if self._end_effector_groups:
            end_state = state.joints[self._end_effector_groups[0]]
            self._publish_joint_state(
                self._end_effector_state_pub,
                now,
                end_state.names,
                tuple(end_state.positions),
                tuple(end_state.velocities),
                tuple(end_state.efforts),
            )

        # ``read()`` advances the heartbeat state machine. This call is cheap when the
        # enum is unchanged and emits DDS traffic only for a real transition.
        self._publish_sync_status()
        self._publish_px4(now, arm_positions, arm_velocities)

    def _publish_sync_status(self, *, force: bool = False) -> None:
        """Publish only synchronization transitions instead of mirroring state at 100 Hz."""

        current = self._session.status
        if not force and self._last_published_sync_status == current:
            return
        if (
            current == FollowerSyncStatus.LOST
            and self._last_published_sync_status != FollowerSyncStatus.LOST
        ):
            self.get_logger().warn(
                "No valid arm command arrived before the session timeout; the follower "
                "remains powered in HOLD and requires synchronization before motion resumes."
            )
        status = String()
        status.data = current.value
        self._sync_status_pub.publish(status)
        self._last_published_sync_status = current

    def _report_runtime_error(self, context: str, error: BaseException) -> None:
        """Report a latched runtime fault once, including the first bus-level cause."""

        diagnostics = self._session.runtime.diagnostics()
        if diagnostics.safety.state != RuntimeSafetyState.FAULT:
            self.get_logger().warn(f"{context}: {error}")
            return
        bus_faults = tuple(
            f"{name}: {bus.fault}"
            for name, bus in diagnostics.buses.items()
            if bus.fault
        )
        reason = diagnostics.safety.fault_reason or str(error)
        if diagnostics.fault_action == AutomaticFaultAction.DISABLE:
            action = "the runtime requested protocol-level torque disable"
        elif diagnostics.fault_action == AutomaticFaultAction.EXTERNAL_ESTOP:
            action = (
                "software motion is held, but this device requires the independent "
                "hardware emergency stop"
            )
        else:
            action = "the runtime retained the last trustworthy holding target"
        detail = f"Follower runtime fault: {reason}; {action}."
        if bus_faults:
            detail += "; bus fault: " + "; ".join(bus_faults)
        if detail == self._last_runtime_fault_message:
            return
        self._last_runtime_fault_message = detail
        self.get_logger().error(detail)

    def _emergency_stop_callback(self, _request, response):
        """Expose the follower's strongest profile-supported stop as an operator API."""

        try:
            self._session.set_mode(LeaderSyncMode.STOP)
        except (RuntimeError, ValueError) as exc:
            response.success = False
            response.message = f"Follower emergency stop failed: {exc}"
            self.get_logger().error(response.message)
        else:
            response.success = True
            response.message = "Follower emergency stop latched."
        self._publish_sync_status(force=True)
        return response

    @staticmethod
    def _publish_joint_state(
        publisher,
        now,
        names,
        positions,
        velocities,
        efforts,
    ) -> None:
        """Convert one immutable core sample into a ROS JointState message."""

        message = ROSJointState()
        message.header.stamp = now.to_msg()
        message.name = list(names)
        message.position = list(positions)
        message.velocity = list(velocities)
        message.effort = list(efforts)
        publisher.publish(message)

    def _publish_px4(self, now, positions, velocities) -> None:
        """Zero-pad finite arm state into the negotiated fixed-capacity PX4 message."""

        count = len(positions)
        capacity = int(ArmJointState.MAX_JOINTS)
        if not 4 <= count <= capacity:
            return
        message = ArmJointState()
        timestamp_us = now.nanoseconds // 1000
        message.timestamp = timestamp_us
        message.timestamp_sample = timestamp_us
        message.sequence = self._sequence
        message.joint_count = count
        message.arm_velocity_valid = bool(np.all(np.isfinite(velocities)))
        message.arm_position = list(positions) + [0.0] * (capacity - count)
        message.arm_velocity = list(velocities) + [0.0] * (capacity - count)
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        self._px4_state_pub.publish(message)

    def _finite_positions(
        self,
        message: ROSJointState,
        expected_count: int,
        label: str,
    ) -> Optional[tuple[float, ...]]:
        """Return one finite command vector or warn and reject it without side effects."""

        try:
            positions = np.asarray(message.position, dtype=float)
        except (TypeError, ValueError):
            positions = np.asarray([], dtype=float)
        if positions.shape != (expected_count,) or not np.all(np.isfinite(positions)):
            self.get_logger().warn(
                f"Ignoring invalid {label} command: expected {expected_count} finite positions."
            )
            return None
        return tuple(float(value) for value in positions)

    @staticmethod
    def _validate_px4_schema() -> None:
        """Reject stale generated px4_msgs before creating any runtime resources."""

        fields = ArmJointState.get_fields_and_field_types()
        required_capacity = 14
        expected = {
            "joint_count": "uint8",
            "arm_velocity_valid": "boolean",
            "arm_position": f"float[{required_capacity}]",
            "arm_velocity": f"float[{required_capacity}]",
        }
        if any(fields.get(name) != value for name, value in expected.items()) or (
            getattr(ArmJointState, "MAX_JOINTS", None) != required_capacity
        ):
            raise RuntimeError(
                "px4_msgs ArmJointState does not match the required 14-joint schema"
            )

    def close(self) -> None:
        """Disconnect while preserving a faulted follower's last holding target."""

        if self._closed:
            return
        self._closed = True
        # A runtime fault has already closed motion admission and requested HOLD. Do not
        # turn subsequent ROS cleanup (including Ctrl+C after a fault) into an implicit
        # emergency stop that releases a gravity-loaded arm.
        try:
            diagnostics = self._session.runtime.diagnostics()
            preserve_hold = (
                diagnostics.safety.state == RuntimeSafetyState.FAULT
                and diagnostics.fault_action != AutomaticFaultAction.DISABLE
            )
        except BaseException:
            # Diagnostics are best effort during teardown. If safety state cannot be
            # established, retaining the last servo target is the conservative payload
            # support policy; independent hardware E-stop remains available.
            preserve_hold = True
        self._session.close(preserve_hold=preserve_hold)


__all__ = ["RuntimeFollowerNode"]
