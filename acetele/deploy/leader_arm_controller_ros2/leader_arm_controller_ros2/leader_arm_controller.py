import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState

from acetele.config.config_loader import ConfigLoader
from acetele.equipment.feetech.linker import Linker


class LeaderArmControllerNode(Node, Linker):
    def __init__(self):
        Node.__init__(self, "leader_arm_controller_node")
        self.config_loader = ConfigLoader(config_name="leader.toml")
        station_type, _ = self.config_loader.get_station_info()
        linker_config = self.config_loader.get_linker_config()
        Linker.__init__(self, station_type, linker_config)
        self.declare_parameter("control_rate", 500.0)
        self.control_rate = self.get_parameter("control_rate").value

        qos = QoSProfile(depth=10)
        self.command_pub = self.create_publisher(
            JointState,
            "/arm/command",
            qos,
        )
        self.state_sub = self.create_subscription(
            JointState,
            "/arm/state",
            self.state_callback,
            qos,
        )

        period = 1.0 / self.control_rate
        self.timer = self.create_timer(period, self._control_loop)

        self.get_logger().info("Leader arm controller node started.")

    def state_callback(self, msg: JointState):
        self.external_torque = msg.effort

    def _control_loop(self):
        joint_pos, joint_vel, joint_effort = self.act(encode_gripper=False)
        tau_n = self._null_space_regulation(joint_pos, joint_vel)  # 零空间投影
        tau_g = self._gravity_compensation(joint_pos, joint_vel)  # 重力补偿
        tau_ss = self._friction_compensation(tau_g, joint_vel)  # 摩擦力补偿
        tau_fb = self._torque_feedback(joint_vel)  # 力反馈
        tau = tau_n + tau_g + tau_ss + tau_fb
        self.set_torque(tau)
        joint_pos = self._encode_gripper(joint_pos)
        self.publish_command(joint_pos, joint_vel, joint_effort)
        # tau_ext = self.estimate_joint_external_torque(joint_pos, joint_vel, joint_effort, 1 / self.control_rate)
        # print("Joint ext torque:", tau_ext)

    def publish_command(self, joint_pos, joint_vel, joint_effort):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint_{i+1}" for i in self._ids]
        msg.position = joint_pos.tolist()
        msg.velocity = joint_vel.tolist()
        msg.effort = joint_effort.tolist()
        self.command_pub.publish(msg)


def main():
    rclpy.init()
    node = LeaderArmControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
