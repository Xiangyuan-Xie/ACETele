from typing import Any, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState

from neutele.config.config_loader import ConfigLoader
from neutele.equipment.feetech.linker import Linker


class ArmControllNode(Node, Linker):
    def __init__(self, station_type: str, config: Dict[str, Any]):
        Node.__init__(self, "arm_controller")
        Linker.__init__(self, station_type, config)
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

        if self._dynamic_enable:
            self._stop_flag.set()
            self._control_thread.join()
        period = 1.0 / self.control_rate
        self.timer = self.create_timer(period, self._control_loop)

        self.get_logger().info("Arm controller node started.")

    def command_callback(self, msg: JointState):
        self.set_position(msg.position)

    def _control_loop(self):
        pos, vel = self.act(encode_gripper=False)
        self.publish_state(pos, vel)

    def publish_state(self, joint_pos, joint_vel):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint_{i}" for i in self._ids]
        msg.position = joint_pos.tolist()
        msg.velocity = joint_vel.tolist()
        self.state_pub.publish(msg)


def main():
    rclpy.init()
    config_loader = ConfigLoader()
    node = ArmControllNode(config_loader.config["basic"]["station_type"], config_loader.config["linker"]["single"])
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
