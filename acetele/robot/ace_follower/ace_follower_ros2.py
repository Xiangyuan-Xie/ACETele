from typing import Optional

from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState

from acetele.config.config_loader import ConfigLoader
from acetele.robot.ace_follower.ace_follower import AceFollowerRobot


class AceFollowerROS2Robot(Node, AceFollowerRobot):
    def __init__(self, config_loader: ConfigLoader):
        Node.__init__(self, "ace_follower_robot_node")
        AceFollowerRobot.__init__(self, config_loader)
        self.declare_parameter("control_rate", 250.0)
        self._control_rate = self.get_parameter("control_rate").value
        self.declare_parameter("heartbeat_timeout", 1.0)
        self._heartbeat_timeout = self.get_parameter("heartbeat_timeout").value
        self._heartbeat_timeout_ns = int(self._heartbeat_timeout * 1e9)

        qos = QoSProfile(depth=10)
        self._state_pub = self.create_publisher(
            JointState,
            "/arm/state",
            qos,
        )
        self._command_sub = self.create_subscription(
            JointState,
            "/arm/command",
            self._command_callback,
            qos,
        )

        period = 1.0 / self._control_rate
        self._timer = self.create_timer(period, self._control_loop)

        current_pos, _, _ = self.act()
        self.move_position(current_pos)
        self._is_synced = False
        self._last_command_ns = None
        self._heartbeat_lost = False
        self._pending_sync_position: Optional[list[float]] = None

        self.get_logger().info("Follower arm controller node started.")

    def _command_callback(self, msg: JointState):
        self._last_command_ns = self.get_clock().now().nanoseconds
        self._heartbeat_lost = False
        if self._is_synced:
            self.set_position(msg.position)
        else:
            self._pending_sync_position = list(msg.position)

    def _control_loop(self):
        if self._is_synced and self._last_command_ns is not None:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_command_ns > self._heartbeat_timeout_ns:
                if not self._heartbeat_lost:
                    self.get_logger().info("Heartbeat lost. Resetting sync state.")
                self._is_synced = False
                self._heartbeat_lost = True
        if not self._is_synced and self._pending_sync_position is not None:
            self.get_logger().info("Synchronizing to the leader arm...")
            self.get_logger().info("Please keep the leader arm still.")
            self.move_position(self._pending_sync_position)
            self._pending_sync_position = None
            self._is_synced = True
            self.get_logger().info("Synchronization completed.")
        joint_pos, joint_vel, joint_effort = self.act()
        self._publish_state(joint_pos, joint_vel, joint_effort)

    def _publish_state(self, joint_pos, joint_vel, joint_effort):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint_{i+1}" for i in self._equipments.single_arm.ids]
        msg.position = joint_pos.tolist()
        msg.velocity = joint_vel.tolist()
        msg.effort = joint_effort.tolist()
        self._state_pub.publish(msg)
