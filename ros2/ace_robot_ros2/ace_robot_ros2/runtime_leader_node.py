"""ROS 2 adapter for leader alignment, state sampling, and command publication."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from geometry_msgs.msg import PoseStamped
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

from acetele.runtime import LeaderTeleopSession, RobotRuntime
from acetele.runtime.teleop import FollowerSyncStatus, LeaderSyncMode, TeleopMode
from acetele.specification import DexterousHandSpec, RobotSpec

from .pose_messages import pose_message
from .spec_validation import validate_ros2_robot_spec


class RuntimeLeaderNode(Node):
    """ROS 2 adapter composed around the pure Python leader session."""

    def __init__(self, spec: RobotSpec) -> None:
        # Topology validation precedes Node and runtime resource creation. The cleanup
        # boundary below handles failures after either side has been initialized.
        validate_ros2_robot_spec(spec, expected_model="ace_leader")
        runtime = RobotRuntime(spec)
        node_initialized = False
        self._closed = False
        try:
            super().__init__("ace_leader_robot")
            node_initialized = True
            self._initialize(runtime)
        except BaseException as initialization_error:
            cleanup_error: Optional[BaseException] = None
            if hasattr(self, "_session"):
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

    def _initialize(self, runtime: RobotRuntime) -> None:
        """Validate runtime parameters, connect the session, and create ROS entities."""

        self.declare_parameter("control_rate", 100.0)
        self.declare_parameter("follower_state_timeout", 0.5)
        self.declare_parameter("sync_position_tolerance", 0.03)
        self.declare_parameter("sync_stable_duration", 0.2)
        self.declare_parameter("sync_profile_velocity", 2.0)
        self.declare_parameter("sync_profile_acceleration", 3.0)
        self.declare_parameter("end_effector_publish_threshold", 0.001)
        self.declare_parameter("end_effector_keepalive", 0.1)
        self.declare_parameter("command_lifespan", 0.05)
        self.declare_parameter("state_lifespan", 0.5)
        self.declare_parameter("teleop_mode", TeleopMode.JOINT.value)
        control_rate = float(self.get_parameter("control_rate").value)
        follower_timeout = float(self.get_parameter("follower_state_timeout").value)
        sync_tolerance = float(self.get_parameter("sync_position_tolerance").value)
        sync_stable = float(self.get_parameter("sync_stable_duration").value)
        sync_velocity = float(self.get_parameter("sync_profile_velocity").value)
        sync_acceleration = float(
            self.get_parameter("sync_profile_acceleration").value
        )
        end_effector_publish_threshold = float(
            self.get_parameter("end_effector_publish_threshold").value
        )
        end_effector_keepalive = float(
            self.get_parameter("end_effector_keepalive").value
        )
        command_lifespan = float(self.get_parameter("command_lifespan").value)
        state_lifespan = float(self.get_parameter("state_lifespan").value)
        try:
            teleop_mode = TeleopMode(str(self.get_parameter("teleop_mode").value))
        except ValueError as exc:
            raise ValueError("teleop_mode must be 'joint' or 'ee_pose'") from exc
        values = (
            control_rate,
            follower_timeout,
            sync_tolerance,
            sync_stable,
            sync_velocity,
            sync_acceleration,
            end_effector_publish_threshold,
            end_effector_keepalive,
            command_lifespan,
            state_lifespan,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("leader ROS 2 parameters must be finite and positive")
        self._end_effector_publish_threshold = end_effector_publish_threshold
        self._end_effector_keepalive_ns = round(end_effector_keepalive * 1e9)
        self._session = LeaderTeleopSession(
            runtime,
            follower_timeout_ns=round(follower_timeout * 1e9),
            sync_tolerance_rad=sync_tolerance,
            sync_stable_ns=round(sync_stable * 1e9),
            sync_velocity_limit_rad_s=sync_velocity,
            sync_acceleration_limit_rad_s2=sync_acceleration,
            teleop_mode=teleop_mode,
        )
        self._session.connect()

        # High-rate data is replaceable; synchronization state is not. Keeping each
        # history at depth one prevents DDS queues from replaying stale teleoperation.
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
        sync_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._end_effector_groups = tuple(
            name for name in runtime.joint_groups if name.endswith(".end_effector")
        )
        end_effectors = tuple(
            arm.end_effector
            for arm in runtime.spec.arms
            if arm.end_effector is not None
        )
        uses_dexterous_hand = any(
            isinstance(end_effector, DexterousHandSpec)
            for end_effector in end_effectors
        )
        command_topic = (
            "/ace_leader/end_effector/command"
            if uses_dexterous_hand
            else "/ace_leader/gripper/command"
        )
        self._arm_command_pub = None
        self._ee_pose_command_pub = None
        if teleop_mode == TeleopMode.JOINT:
            self._arm_command_pub = self.create_publisher(
                ROSJointState,
                "/ace_leader/arm/command",
                command_qos,
            )
        else:
            self._ee_pose_command_pub = self.create_publisher(
                PoseStamped,
                "/ace_teleop/arm/ee_pose/command",
                command_qos,
            )
        self._end_effector_command_pub = self.create_publisher(
            ROSJointState,
            command_topic,
            command_qos,
        )
        self._sync_mode_pub = self.create_publisher(
            String,
            "/ace_leader/arm/sync_mode",
            sync_qos,
        )
        self._last_published_mode: Optional[LeaderSyncMode] = None
        self._state_sub = self.create_subscription(
            ROSJointState,
            "/ace_follower/arm/state",
            self._state_callback,
            state_qos,
        )
        self._sync_status_sub = self.create_subscription(
            String,
            "/ace_follower/arm/sync_status",
            self._status_callback,
            sync_qos,
        )
        self._emergency_stop_service = self.create_service(
            Trigger,
            "/ace_leader/emergency_stop",
            self._emergency_stop_callback,
        )
        self._authorize_alignment_service = self.create_service(
            Trigger,
            "/ace_leader/authorize_alignment",
            self._authorize_alignment_callback,
        )
        self._start_tracking_service = self.create_service(
            Trigger,
            "/ace_leader/start_tracking",
            self._start_tracking_callback,
        )
        self._last_end_effector_positions: Optional[np.ndarray] = None
        self._last_end_effector_publish_ns: Optional[int] = None
        self._timer = self.create_timer(1.0 / control_rate, self._control_loop)
        self._publish_mode()
        self.get_logger().info(
            "Leader RobotRuntime ROS 2 node started; waiting for follower state."
        )

    def _state_callback(self, message: ROSJointState) -> None:
        """Record follower feedback only when names and positions match the model."""

        names = tuple(message.name) or self._session.arm_names
        try:
            self._session.observe_follower_state(
                names,
                message.position,
                now_ns=time.monotonic_ns(),
            )
        except (TypeError, ValueError) as exc:
            self.get_logger().warn(f"Ignoring invalid follower state: {exc}")

    def _status_callback(self, message: String) -> None:
        """Record one reliable follower synchronization status sample."""

        try:
            status = FollowerSyncStatus(message.data)
            self._session.observe_follower_status(
                status,
                now_ns=time.monotonic_ns(),
            )
        except ValueError as exc:
            self.get_logger().warn(f"Ignoring invalid follower status: {exc}")

    def _control_loop(self) -> None:
        """Advance synchronization and publish the latest leader state when tracking."""

        now = self.get_clock().now()
        previous_mode = self._session.mode
        try:
            state = self._session.step(now_ns=time.monotonic_ns())
            self._session.try_start_tracking(state)
        except (RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Leader runtime control failed: {exc}")
            try:
                self._session.hold()
            except (RuntimeError, ValueError) as hold_exc:
                self.get_logger().error(f"Leader fault hold failed: {hold_exc}")
            self._report_mode_transition(previous_mode)
            if self._session.mode != previous_mode:
                self._publish_mode()
            return
        self._report_mode_transition(previous_mode)
        if self._session.mode != previous_mode:
            self._publish_mode()
        if self._session.mode != LeaderSyncMode.TRACKING:
            return
        if self._session.teleop_mode == TeleopMode.EE_POSE:
            pose = self._session.end_effector_pose(
                state,
                timestamp_ns=time.monotonic_ns(),
            )
            if self._ee_pose_command_pub is None:
                raise RuntimeError("ee_pose publisher was not initialized")
            self._ee_pose_command_pub.publish(pose_message(pose, now.to_msg()))
        else:
            arms = tuple(state.joints[arm.name] for arm in self._session.runtime.spec.arms)
            self._publish_joint_state(
                self._arm_command_pub,
                now,
                tuple(name for item in arms for name in item.names),
                tuple(float(value) for item in arms for value in item.positions),
                tuple(float(value) for item in arms for value in item.velocities),
                tuple(float(value) for item in arms for value in item.efforts),
            )
        self._publish_gripper(state, now)

    def _report_mode_transition(self, previous_mode: LeaderSyncMode) -> None:
        """Emit one actionable message for each operator-visible session transition."""

        current_mode = self._session.mode
        if current_mode == previous_mode:
            return
        if current_mode == LeaderSyncMode.SYNC_REQUEST:
            if self._session.follower_status == FollowerSyncStatus.FAULT:
                self.get_logger().error(
                    "Follower reported a hardware fault; leader torque remains released "
                    "while waiting for a healthy follower."
                )
                return
            if self._session.follower_status == FollowerSyncStatus.LOST:
                self.get_logger().warn(
                    "Follower command heartbeat was lost; leader torque remains released "
                    "until synchronization can restart."
                )
                return
            message = (
                "Follower detected; leader arm torque is enabled for automatic alignment."
                if self._session.uses_start_trigger
                else (
                    "Follower detected; waiting for explicit operator authorization "
                    "because no gripper start trigger is configured."
                )
            )
        elif current_mode == LeaderSyncMode.READY:
            if self._session.uses_start_trigger:
                message = (
                    "Leader aligned; move the gripper from fully released to at least "
                    f"{self._session.start_trigger_threshold:.0%} to start teleoperation."
                )
            else:
                message = "Leader aligned; waiting for an explicit start command."
        elif current_mode == LeaderSyncMode.TRACKING:
            message = "Teleoperation started; leader arm torque is released."
        elif current_mode == LeaderSyncMode.HOLD:
            if self._session.torque_released:
                message = (
                    "Teleoperation is holding the follower; leader torque is released. "
                    "Release then close the gripper to request a fresh synchronization."
                )
            else:
                self.get_logger().error(
                    "Teleoperation entered HOLD, but leader torque release could not be "
                    "confirmed. Use the hardware emergency stop before handling the arm."
                )
                return
        elif current_mode == LeaderSyncMode.STOP:
            self.get_logger().error("Teleoperation stopped by a safety fault.")
            return
        else:
            message = f"Teleoperation mode changed to {current_mode.value}."
        self.get_logger().info(message)

    def _publish_mode(self, *, force: bool = False) -> None:
        """Publish only transitions; reliable DDS handles delivery of each event."""

        if not force and self._last_published_mode == self._session.mode:
            return
        message = String()
        message.data = self._session.mode.value
        self._sync_mode_pub.publish(message)
        self._last_published_mode = self._session.mode

    def _emergency_stop_callback(self, _request, response):
        """Latch local STOP and publish it even when hardware stop dispatch fails."""

        try:
            self._session.stop()
        except (RuntimeError, ValueError) as exc:
            response.success = False
            response.message = f"Leader emergency stop failed: {exc}"
            self.get_logger().error(response.message)
        else:
            response.success = True
            response.message = "Leader emergency stop latched."
        finally:
            self._publish_mode(force=True)
        return response

    def _authorize_alignment_callback(self, _request, response):
        """Provide a deliberate start control for leaders without a gripper trigger."""

        try:
            self._session.authorize_alignment()
        except (RuntimeError, ValueError) as exc:
            response.success = False
            response.message = f"Alignment authorization rejected: {exc}"
        else:
            response.success = True
            response.message = "Powered leader alignment authorized."
            self.get_logger().warn(response.message)
        return response

    def _start_tracking_callback(self, _request, response):
        """Explicitly release the aligned leader and enter TRACKING."""

        try:
            self._session.start_tracking()
        except (RuntimeError, ValueError) as exc:
            response.success = False
            response.message = f"Tracking start rejected: {exc}"
        else:
            response.success = True
            response.message = "Teleoperation tracking started."
            self.get_logger().info(response.message)
            self._publish_mode()
        return response

    def _publish_gripper(self, state, now) -> None:
        """Rate-reduce end-effector traffic while preserving a periodic keepalive."""

        if not self._end_effector_groups:
            return
        end_effector = state.joints[self._end_effector_groups[0]]
        positions = np.asarray(end_effector.positions, dtype=float)
        # Threshold suppresses encoder chatter; keepalive guarantees eventual recovery
        # after a best-effort packet is lost.
        changed = (
            self._last_end_effector_positions is None
            or np.max(np.abs(positions - self._last_end_effector_positions))
            > self._end_effector_publish_threshold
        )
        keepalive = (
            self._last_end_effector_publish_ns is None
            or now.nanoseconds - self._last_end_effector_publish_ns
            >= self._end_effector_keepalive_ns
        )
        if not changed and not keepalive:
            return
        self._publish_joint_state(
            self._end_effector_command_pub,
            now,
            end_effector.names,
            tuple(end_effector.positions),
            tuple(end_effector.velocities),
            tuple(end_effector.efforts),
        )
        self._last_end_effector_positions = positions.copy()
        self._last_end_effector_publish_ns = now.nanoseconds

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

    def close(self) -> None:
        """Idempotently disconnect the leader session and runtime."""

        if self._closed:
            return
        self._closed = True
        self._session.close()


__all__ = ["RuntimeLeaderNode"]
