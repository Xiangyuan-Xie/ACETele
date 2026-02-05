from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState

from acetele.station.ace_follower.ace_follower import AceFollowerStation


class AceFollowerROS2Station(Node, AceFollowerStation):
    def __init__(self):
        Node.__init__(self, "ace_follower_station_node")
        self.declare_parameter("control_rate", 250.0)
        self._control_rate = self.get_parameter("control_rate").value

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

        self.get_logger().info("Follower arm controller node started.")

    def _command_callback(self, msg: JointState):
        if self._is_synced:
            self.set_position(msg.position)
        else:
            self.get_logger().info("Synchronizing to the leader arm...")
            self.get_logger().info("Please keep the leader arm still.")
            self.move_position(msg.position)
            self._is_synced = True
            self.get_logger().info("Synchronization completed.")

    def _control_loop(self):
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
