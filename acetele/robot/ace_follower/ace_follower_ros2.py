from typing import Optional, Sequence, Tuple

import numpy as np
from px4_msgs.msg import ArmJointState
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from acetele.config.config_loader import ConfigLoader
from acetele.robot.ace_follower.ace_follower import AceFollowerRobot
from acetele.utils.teleop_sync import (
    FollowerSyncStatus,
    LeaderSyncMode,
)


class AceFollowerROS2Robot(Node, AceFollowerRobot):
    def __init__(self, config_loader: ConfigLoader):
        Node.__init__(self, "ace_follower_robot_node")
        AceFollowerRobot.__init__(self, config_loader)
        self.declare_parameter("control_rate", 100.0)
        self._control_rate = self.get_parameter("control_rate").value
        self.declare_parameter("publish_rate", 100.0)
        self._publish_rate = self.get_parameter("publish_rate").value
        self.declare_parameter("heartbeat_timeout", 1.0)
        self._heartbeat_timeout_ns = int(self.get_parameter("heartbeat_timeout").value * 1e9)
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
        self._sync_mode = LeaderSyncMode.IDLE
        self._sync_status = FollowerSyncStatus.IDLE
        self._last_command_ns = None
        self._heartbeat_lost = False
        self._latest_arm_command: Optional[list[float]] = None
        self._latest_gripper_command: Optional[list[float]] = None
        self._latest_arm_state: Optional[
            Tuple[Sequence[float], Sequence[float], Sequence[float]]
        ] = None
        self._latest_gripper_state: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None
        self._latest_gripper_device_state = None
        self._warned_invalid_px4_arm_state_length = False

        self.get_logger().info("Follower arm controller node started.")

    def _arm_command_callback(self, msg: JointState):
        if self._sync_mode != LeaderSyncMode.TRACKING:
            return
        self._last_command_ns = self.get_clock().now().nanoseconds
        self._heartbeat_lost = False
        self._latest_arm_command = list(msg.position)

    def _gripper_command_callback(self, msg: JointState):
        if self._sync_mode != LeaderSyncMode.TRACKING:
            return
        if not msg.position:
            return
        self._latest_gripper_command = list(msg.position)

    def _sync_mode_callback(self, msg: String):
        try:
            sync_mode = LeaderSyncMode(msg.data)
        except ValueError:
            self.get_logger().warn(f"Ignoring invalid sync mode: {msg.data}")
            return
        self._sync_mode = sync_mode
        if sync_mode == LeaderSyncMode.IDLE:
            self._sync_status = FollowerSyncStatus.IDLE
        elif sync_mode == LeaderSyncMode.SYNC_REQUEST:
            self.get_logger().info("Holding follower arm pose for leader synchronization.")
            self._sync_status = FollowerSyncStatus.READY
        elif sync_mode == LeaderSyncMode.READY:
            self._sync_status = FollowerSyncStatus.READY
        elif sync_mode == LeaderSyncMode.STOP:
            self._sync_status = FollowerSyncStatus.LOST

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
        if self._sync_status == FollowerSyncStatus.TRACKING and self._last_command_ns is not None:
            if now_ns - self._last_command_ns > self._heartbeat_timeout_ns:
                if not self._heartbeat_lost:
                    self.get_logger().info("Heartbeat lost. Entering lost sync state.")
                self._sync_status = FollowerSyncStatus.LOST
                self._heartbeat_lost = True
        single_arm = getattr(getattr(self, "_equipments", None), "single_arm", None)
        gripper = getattr(getattr(self, "_equipments", None), "gripper", None)
        if self._sync_mode == LeaderSyncMode.STOP:
            self._sync_status = FollowerSyncStatus.LOST
        elif self._sync_mode == LeaderSyncMode.IDLE:
            self._sync_status = FollowerSyncStatus.IDLE
        elif self._sync_mode in (LeaderSyncMode.SYNC_REQUEST, LeaderSyncMode.READY):
            self._sync_status = FollowerSyncStatus.READY
        elif self._sync_mode == LeaderSyncMode.TRACKING:
            command_allowed = self._sync_status in (FollowerSyncStatus.READY, FollowerSyncStatus.TRACKING)
            command_applied = False
            if command_allowed and self._latest_arm_command is not None and single_arm is not None:
                single_arm.set_position(self._latest_arm_command)
                command_applied = True
            latest_gripper_command = getattr(self, "_latest_gripper_command", None)
            if command_allowed and latest_gripper_command is not None and gripper is not None:
                command_position = float(latest_gripper_command[0])
                gripper.set_position(command_position)
                command_applied = True
            if command_applied:
                self._sync_status = FollowerSyncStatus.TRACKING
        if single_arm is not None and hasattr(single_arm, "get_linker_state"):
            arm_state = single_arm.get_linker_state()
            joint_pos = arm_state.public_positions
            joint_vel = arm_state.velocities
            joint_effort = arm_state.motor_torque_magnitude
            self._latest_arm_state = (joint_pos, joint_vel, joint_effort)
        else:
            joint_pos, joint_vel, joint_effort = self.act()
            self._latest_arm_state = (joint_pos, joint_vel, joint_effort)
        if gripper is not None:
            gripper_state = gripper.get_state()
            self._latest_gripper_device_state = gripper_state
            self._latest_gripper_state = (
                np.array([gripper_state.public_position], dtype=float),
                np.array([gripper_state.velocity], dtype=float),
                np.array([gripper_state.motor_torque_magnitude], dtype=float),
            )
        else:
            self._latest_gripper_device_state = None
            self._latest_gripper_state = None

    def _publish_state_loop(self):
        if self._latest_arm_state is None:
            return

        joint_pos, joint_vel, joint_effort = self._latest_arm_state
        now = self.get_clock().now()
        msg = JointState()
        msg.header.stamp = now.to_msg()
        single_arm = getattr(getattr(self, "_equipments", None), "single_arm", None)
        arm_ids = np.asarray(getattr(single_arm, "ids", getattr(self, "ids", range(len(joint_pos))))).astype(int)
        msg.name = [f"joint_{i+1}" for i in arm_ids]
        msg.position = joint_pos.tolist()
        msg.velocity = joint_vel.tolist()
        msg.effort = joint_effort.tolist()
        self._arm_state_pub.publish(msg)

        status_msg = String()
        status_msg.data = self._sync_status.value
        self._sync_status_pub.publish(status_msg)

        gripper_state = getattr(self, "_latest_gripper_state", None)
        if gripper_state is not None:
            gripper_pos, gripper_vel, gripper_effort = gripper_state
            gripper_msg = JointState()
            gripper_msg.header.stamp = now.to_msg()
            gripper_id = getattr(self, "gripper_id", None)
            gripper_msg.name = [] if gripper_id is None else [f"joint_{int(gripper_id) + 1}"]
            gripper_msg.position = gripper_pos.tolist()
            gripper_msg.velocity = gripper_vel.tolist()
            gripper_msg.effort = gripper_effort.tolist()
            self._gripper_state_pub.publish(gripper_msg)

        px4_positions = list(msg.position)
        if gripper_state is not None:
            px4_positions = px4_positions + list(np.asarray(gripper_state[0], dtype=float))

        if len(px4_positions) != 5:
            if not self._warned_invalid_px4_arm_state_length:
                self.get_logger().warn(
                    "Skipping PX4 arm joint state publish: ArmJointState expects 5 joints."
                )
                self._warned_invalid_px4_arm_state_length = True
            return

        px4_velocities = list(msg.velocity)
        if gripper_state is not None:
            px4_velocities = px4_velocities + list(np.asarray(gripper_state[1], dtype=float))

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
