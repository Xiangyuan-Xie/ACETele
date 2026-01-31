import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState

from acetele.config.config_loader import ConfigLoader
from acetele.equipment.feetech.linker import Linker


class FollowerArmControllerNode(Node, Linker):
    def __init__(self):
        Node.__init__(self, "follower_arm_controller_node")
        self.config_loader = ConfigLoader()
        station_name = self.config_loader.get_station_type()
        linker_config = self.config_loader.get_linker_config()
        Linker.__init__(self, station_name, linker_config)
        self.declare_parameter("control_rate", 500.0)
        self.control_rate = self.get_parameter("control_rate").value

        qos = QoSProfile(depth=10)
        self.state_pub = self.create_publisher(
            JointState,
            "/arm/state",
            qos,
        )
        self.command_sub = self.create_subscription(
            JointState,
            "/arm/command",
            self.command_callback,
            qos,
        )

        period = 1.0 / self.control_rate
        self.timer = self.create_timer(period, self._control_loop)

        self.move_position(self._home_poses)
        self.is_synced = False

        self.get_logger().info("Follower arm controller node started.")

    def command_callback(self, msg: JointState):
        if self.is_synced:
            self.set_position(msg.position)
        else:
            self.get_logger().info("Synchronizing to the master arm...")
            self.get_logger().info("Please keep the master arm still.")
            self.move_position(msg.position)
            self.is_synced = True
            self.get_logger().info("Synchronization completed.")

    def _control_loop(self):
        joint_pos, joint_vel, joint_effort = self.act()
        self.publish_state(joint_pos, joint_vel, joint_effort)

    def publish_state(self, joint_pos, joint_vel, joint_effort):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint_{i+1}" for i in self._ids]
        msg.position = joint_pos.tolist()
        msg.velocity = joint_vel.tolist()
        msg.effort = joint_effort.tolist()
        self.state_pub.publish(msg)


def main():
    rclpy.init()
    node = FollowerArmControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
