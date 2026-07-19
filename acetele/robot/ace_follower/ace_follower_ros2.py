from __future__ import annotations

from typing import Optional

import numpy as np
from px4_msgs.msg import ArmJointState
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from acetele.config.config_loader import ConfigLoader
from acetele.config.robot_config import FeeTechGripperConfig, RobotConfig
from acetele.equipment.joint_device import JointDevice, JointDeviceState
from acetele.robot.ace_follower.ace_follower import AceFollowerRobot
from acetele.utils.teleop_sync import (
    FollowerSyncController,
    FollowerSyncStatus,
    LeaderSyncMode,
)


class AceFollowerROS2Robot(Node, AceFollowerRobot):
    def __init__(self, config_loader: ConfigLoader | RobotConfig):
        robot_config = (
            config_loader
            if isinstance(config_loader, RobotConfig)
            else config_loader.get_robot_config()
        )
        configured_end_effector = robot_config.arm_assemblies[0].end_effector
        if configured_end_effector is not None and not isinstance(
            configured_end_effector,
            FeeTechGripperConfig,
        ):
            raise RuntimeError("ACE follower ROS 2 currently supports only a normalized gripper end effector")
        node_initialized = False
        robot_initialized = False
        try:
            Node.__init__(self, "ace_follower_robot_node")
            node_initialized = True
            AceFollowerRobot.__init__(self, robot_config)
            robot_initialized = True
            self._initialize_ros_interfaces()
        except BaseException as initialization_error:
            cleanup_error: Optional[BaseException] = None
            if robot_initialized:
                try:
                    AceFollowerRobot.close(self)
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

    def _initialize_ros_interfaces(self) -> None:
        self.declare_parameter("control_rate", 100.0)
        self._control_rate = self.get_parameter("control_rate").value
        self.declare_parameter("publish_rate", 100.0)
        self._publish_rate = self.get_parameter("publish_rate").value
        self.declare_parameter("heartbeat_timeout", 1.0)
        self._heartbeat_timeout_ns = int(self.get_parameter("heartbeat_timeout").value * 1e9)
        self._sync = FollowerSyncController(self._heartbeat_timeout_ns)
        qos = QoSProfile(depth=10)
        self._arm_state_pub = self.create_publisher(
            JointState,
            "/ace_follower/arm/state",
            qos,
        )
        self._validate_px4_arm_joint_state_schema()
        self._px4_arm_state_sequence = 0
        self._px4_arm_state_pub = self.create_publisher(
            ArmJointState,
            "/fmu/in/arm_joint_state",
            qos,
        )
        self._sync_status_pub = self.create_publisher(
            String,
            "/ace_follower/arm/sync_status",
            qos,
        )
        self._gripper_state_pub = self.create_publisher(
            JointState,
            "/ace_follower/gripper/state",
            qos,
        )
        self._arm_command_sub = self.create_subscription(
            JointState,
            "/ace_leader/arm/command",
            self._arm_command_callback,
            qos,
        )
        self._gripper_command_sub = self.create_subscription(
            JointState,
            "/ace_leader/gripper/command",
            self._gripper_command_callback,
            qos,
        )
        self._sync_mode_sub = self.create_subscription(
            String,
            "/ace_leader/arm/sync_mode",
            self._sync_mode_callback,
            qos,
        )

        control_period = 1.0 / self._control_rate
        self._control_timer = self.create_timer(control_period, self._control_loop)
        publish_period = 1.0 / self._publish_rate
        self._state_publish_timer = self.create_timer(publish_period, self._publish_state_loop)

        current_pos, _, _ = self.act()
        self.set_position(current_pos)
        self._heartbeat_lost = False
        self._latest_arm_command: Optional[list[float]] = None
        self._latest_gripper_command: Optional[list[float]] = None
        self._latest_arm_state: Optional[JointDeviceState] = None
        self._latest_gripper_state: Optional[JointDeviceState] = None
        self._warned_invalid_px4_arm_state_length = False

        self.get_logger().info("Follower arm controller node started.")

    def _clear_cached_commands(self) -> None:
        self._latest_arm_command = None
        self._latest_gripper_command = None

    def _validate_joint_command(
        self,
        msg: JointState,
        device: Optional[JointDevice],
        label: str,
    ) -> Optional[list[float]]:
        if device is None:
            self.get_logger().warn(f"Ignoring {label} command because no device is configured.")
            return None
        try:
            positions = np.asarray(msg.position, dtype=float)
        except (TypeError, ValueError):
            positions = np.asarray([], dtype=float)
        expected_count = len(device.ids)
        if positions.shape != (expected_count,) or not np.all(np.isfinite(positions)):
            self.get_logger().warn(
                f"Ignoring invalid {label} command: expected {expected_count} finite position values."
            )
            return None

        names = tuple(getattr(msg, "name", ()) or ())
        expected_names = tuple(device.joint_names)
        if names and names != expected_names:
            self.get_logger().warn(
                f"Ignoring invalid {label} command joint order: expected {expected_names}, got {names}."
            )
            return None
        try:
            device.validate_position_command(positions)
        except ValueError as exc:
            self.get_logger().warn(f"Ignoring invalid {label} command: {exc}")
            return None
        return positions.tolist()

    def _arm_command_callback(self, msg: JointState):
        if not self._sync.command_allowed:
            return
        positions = self._validate_joint_command(msg, self.arm, "arm")
        if positions is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        if not self._sync.accept_command(now_ns):
            return
        self._heartbeat_lost = False
        self._latest_arm_command = positions

    def _gripper_command_callback(self, msg: JointState):
        if not self._sync.command_allowed:
            return
        positions = self._validate_joint_command(
            msg,
            self.end_effector,
            "gripper",
        )
        if positions is None:
            return
        self._latest_gripper_command = positions

    def _sync_mode_callback(self, msg: String):
        try:
            sync_mode = LeaderSyncMode(msg.data)
        except ValueError:
            self.get_logger().warn(f"Ignoring invalid sync mode: {msg.data}")
            return
        previous_mode = self._sync.mode
        self._sync.set_mode(sync_mode)
        if sync_mode != LeaderSyncMode.TRACKING or previous_mode != LeaderSyncMode.TRACKING:
            self._clear_cached_commands()
        if sync_mode == LeaderSyncMode.SYNC_REQUEST:
            self.get_logger().info("Holding follower arm pose for leader synchronization.")

    @staticmethod
    def _validate_px4_arm_joint_state_schema():
        fields = ArmJointState.get_fields_and_field_types()
        required_fields = {
            "timestamp",
            "timestamp_sample",
            "sequence",
            "arm_position",
            "arm_velocity",
        }
        if not required_fields.issubset(fields):
            raise RuntimeError(
                "px4_msgs.msg.ArmJointState schema mismatch: expected fields "
                "timestamp, timestamp_sample, sequence, arm_position, arm_velocity. "
                f"Got {fields}. Rebuild/source the workspace px4_msgs package."
            )

    def _control_loop(self):
        now_ns = self.get_clock().now().nanoseconds
        previous_status = self._sync.status
        self._sync.update(now_ns)
        if (
            previous_status == FollowerSyncStatus.TRACKING
            and self._sync.status == FollowerSyncStatus.LOST
        ):
            if not self._heartbeat_lost:
                self.get_logger().info("Heartbeat lost. Entering lost sync state.")
            self._heartbeat_lost = True
            self._clear_cached_commands()
        arm = self.arm
        gripper = self.end_effector
        if self._sync.mode == LeaderSyncMode.STOP:
            self._clear_cached_commands()
        elif self._sync.mode == LeaderSyncMode.IDLE:
            self._clear_cached_commands()
        elif self._sync.mode in (LeaderSyncMode.SYNC_REQUEST, LeaderSyncMode.READY):
            self._clear_cached_commands()
        elif self._sync.mode == LeaderSyncMode.TRACKING:
            has_current_arm_command = (
                self._sync.status == FollowerSyncStatus.TRACKING
                and self._sync.last_command_ns is not None
            )
            if has_current_arm_command and self._latest_arm_command is not None:
                arm.set_position(self._latest_arm_command)
            if (
                has_current_arm_command
                and self._latest_gripper_command is not None
                and gripper is not None
            ):
                gripper.set_position(self._latest_gripper_command)
        self._latest_arm_state = arm.get_state()
        if gripper is not None:
            self._latest_gripper_state = gripper.get_state()
        else:
            self._latest_gripper_state = None

    def _publish_state_loop(self):
        if self._latest_arm_state is None:
            return

        arm_state = self._latest_arm_state
        now = self.get_clock().now()
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = list(self.arm.joint_names)
        msg.position = arm_state.public_positions.tolist()
        msg.velocity = arm_state.velocities.tolist()
        msg.effort = arm_state.motor_torque_magnitude.tolist()
        self._arm_state_pub.publish(msg)

        status_msg = String()
        status_msg.data = self._sync.status.value
        self._sync_status_pub.publish(status_msg)

        gripper_state = self._latest_gripper_state
        if gripper_state is not None:
            gripper_msg = JointState()
            gripper_msg.header.stamp = now.to_msg()
            gripper_msg.name = list(self.end_effector.joint_names)
            gripper_msg.position = gripper_state.public_positions.tolist()
            gripper_msg.velocity = gripper_state.velocities.tolist()
            gripper_msg.effort = gripper_state.motor_torque_magnitude.tolist()
            self._gripper_state_pub.publish(gripper_msg)

        px4_positions = list(msg.position)
        if gripper_state is not None:
            px4_positions += gripper_state.public_positions.tolist()

        if len(px4_positions) != 5:
            if not self._warned_invalid_px4_arm_state_length:
                self.get_logger().warn(
                    "Skipping PX4 arm joint state publish: ArmJointState expects 5 joints."
                )
                self._warned_invalid_px4_arm_state_length = True
            return

        px4_velocities = list(msg.velocity)
        if gripper_state is not None:
            px4_velocities += gripper_state.velocities.tolist()

        if len(px4_velocities) != 5:
            px4_velocities = [0.0] * 5

        px4_msg = ArmJointState()
        timestamp_us = now.nanoseconds // 1000
        px4_msg.timestamp = timestamp_us
        px4_msg.timestamp_sample = timestamp_us
        sequence = getattr(self, "_px4_arm_state_sequence", 0)
        px4_msg.sequence = sequence
        px4_msg.arm_position = px4_positions
        px4_msg.arm_velocity = px4_velocities
        self._px4_arm_state_sequence = (sequence + 1) & 0xFFFFFFFF
        self._px4_arm_state_pub.publish(px4_msg)

    def close(self):
        AceFollowerRobot.close(self)
