from typing import Optional, Sequence, Tuple

import numpy as np
from geometry_msgs.msg import WrenchStamped
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
        self._state_pub = self.create_publisher(
            JointState,
            "/arm/state",
            qos,
        )
        self._external_joint_torque_pub = self.create_publisher(
            JointState,
            "/arm/external_joint_torque",
            qos,
        )
        self._external_wrench_pub = self.create_publisher(
            WrenchStamped,
            "/arm/external_wrench",
            qos,
        )
        self._px4_arm_state_pub = self.create_publisher(
            ArmJointState,
            "/fmu/in/arm_joint_state",
            qos,
        )
        self._sync_status_pub = self.create_publisher(
            String,
            "/arm/sync_status",
            qos,
        )
        self._command_sub = self.create_subscription(
            JointState,
            "/arm/command",
            self._command_callback,
            qos,
        )
        self._sync_mode_sub = self.create_subscription(
            String,
            "/arm/sync_mode",
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
        self._latest_command: Optional[list[float]] = None
        self._latest_state: Optional[
            Tuple[Sequence[float], Sequence[float], Sequence[float]]
        ] = None
        self._last_external_estimate_ns: Optional[int] = None
        self._external_joint_torque: Optional[np.ndarray] = None
        self._external_wrench: Optional[np.ndarray] = None
        single_arm = getattr(getattr(self, "_equipments", None), "single_arm", None)
        self._external_wrench_frame_id = getattr(
            single_arm,
            "_external_wrench_frame_name",
            "link_5",
        )
        self._warned_invalid_px4_arm_state_length = False

        self.get_logger().info("Follower arm controller node started.")

    def _command_callback(self, msg: JointState):
        if self._sync_mode != LeaderSyncMode.TRACKING:
            return
        self._last_command_ns = self.get_clock().now().nanoseconds
        self._heartbeat_lost = False
        self._latest_command = list(msg.position)

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

    def _control_loop(self):
        now_ns = self.get_clock().now().nanoseconds
        if self._sync_status == FollowerSyncStatus.TRACKING and self._last_command_ns is not None:
            if now_ns - self._last_command_ns > self._heartbeat_timeout_ns:
                if not self._heartbeat_lost:
                    self.get_logger().info("Heartbeat lost. Entering lost sync state.")
                self._sync_status = FollowerSyncStatus.LOST
                self._heartbeat_lost = True
        single_arm = getattr(getattr(self, "_equipments", None), "single_arm", None)
        if single_arm is not None and hasattr(single_arm, "get_linker_state"):
            state = single_arm.get_linker_state()
            joint_pos = state.public_positions
            joint_vel = state.velocities
            joint_effort = state.motor_torque_magnitude
            self._latest_state = (joint_pos, joint_vel, joint_effort)
            self._update_external_estimate(now_ns, state)
        else:
            joint_pos, joint_vel, joint_effort = self.act()
            self._latest_state = (joint_pos, joint_vel, joint_effort)
        if self._sync_mode == LeaderSyncMode.STOP:
            self._sync_status = FollowerSyncStatus.LOST
            return
        if self._sync_mode == LeaderSyncMode.IDLE:
            self._sync_status = FollowerSyncStatus.IDLE
            return
        if self._sync_mode in (LeaderSyncMode.SYNC_REQUEST, LeaderSyncMode.READY):
            self._sync_status = FollowerSyncStatus.READY
            return
        if self._sync_mode == LeaderSyncMode.TRACKING:
            if self._latest_command is None:
                return
            if self._sync_status in (FollowerSyncStatus.READY, FollowerSyncStatus.TRACKING):
                self.set_position(self._latest_command)
                self._sync_status = FollowerSyncStatus.TRACKING

    def _update_external_estimate(self, now_ns: int, state):
        linker = self._equipments.single_arm
        if not getattr(linker, "_enable_estimate_external_torque", False):
            self._external_joint_torque = None
            self._external_wrench = None
            return
        if self._last_external_estimate_ns is None:
            dt = 0.0
        else:
            dt = (now_ns - self._last_external_estimate_ns) * 1e-9
        self._last_external_estimate_ns = now_ns
        external_joint_torque = linker.estimate_joint_external_torque(
            state.raw_positions,
            state.velocities,
            state.motor_torque_signed,
            dt,
        )
        self._external_joint_torque = np.asarray(external_joint_torque, dtype=float)
        self._external_wrench = np.asarray(
            linker.external_wrench_from_joint_torque(state.raw_positions, self._external_joint_torque),
            dtype=float,
        )

    def _publish_state_loop(self):
        if self._latest_state is None:
            return

        joint_pos, joint_vel, joint_effort = self._latest_state
        now = self.get_clock().now()
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = [f"joint_{i+1}" for i in self._equipments.single_arm.ids]
        msg.position = joint_pos.tolist()
        msg.velocity = joint_vel.tolist()
        msg.effort = joint_effort.tolist()
        self._state_pub.publish(msg)

        external_joint_torque = getattr(self, "_external_joint_torque", None)
        if external_joint_torque is not None:
            external_msg = JointState()
            external_msg.header.stamp = now.to_msg()
            external_msg.name = msg.name
            external_msg.effort = external_joint_torque.tolist()
            self._external_joint_torque_pub.publish(external_msg)

        external_wrench = getattr(self, "_external_wrench", None)
        if external_wrench is not None:
            wrench_msg = WrenchStamped()
            wrench_msg.header.stamp = now.to_msg()
            wrench_msg.header.frame_id = getattr(self, "_external_wrench_frame_id", "link_5")
            wrench_msg.wrench.force.x = float(external_wrench[0])
            wrench_msg.wrench.force.y = float(external_wrench[1])
            wrench_msg.wrench.force.z = float(external_wrench[2])
            wrench_msg.wrench.torque.x = float(external_wrench[3])
            wrench_msg.wrench.torque.y = float(external_wrench[4])
            wrench_msg.wrench.torque.z = float(external_wrench[5])
            self._external_wrench_pub.publish(wrench_msg)

        status_msg = String()
        status_msg.data = self._sync_status.value
        self._sync_status_pub.publish(status_msg)

        if len(msg.position) != 5:
            if not self._warned_invalid_px4_arm_state_length:
                self.get_logger().warn(
                    "Skipping PX4 arm joint state publish: ArmJointState expects 5 joints."
                )
                self._warned_invalid_px4_arm_state_length = True
            return

        px4_msg = ArmJointState()
        px4_msg.timestamp = now.nanoseconds // 1000
        px4_msg.arm_position = msg.position
        self._px4_arm_state_pub.publish(px4_msg)
