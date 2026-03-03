from px4_msgs.msg import VehicleLandDetected
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState

from acetele.config.config_loader import ConfigLoader
from acetele.robot.ace_leader.ace_leader import AceLeaderRobot


class AceLeaderROS2Robot(Node, AceLeaderRobot):
    def __init__(self, config_loader: ConfigLoader):
        Node.__init__(self, "ace_leader_robot_node")
        AceLeaderRobot.__init__(self, config_loader)
        self.declare_parameter("control_rate", 250.0)
        self._control_rate = self.get_parameter("control_rate").value

        qos = QoSProfile(depth=10)
        self._command_pub = self.create_publisher(
            JointState,
            "/arm/command",
            qos,
        )
        self._state_sub = self.create_subscription(
            JointState,
            "/arm/state",
            self._state_callback,
            qos,
        )
        self._land_detected_sub = self.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self._landed_callback,
            qos_profile_sensor_data,
        )

        period = 1.0 / self._control_rate
        self._timer = self.create_timer(period, self._control_loop)

        self._is_started = False
        self._is_landed = False
        self._is_ended = False

        self.get_logger().info("Leader arm controller node started.")

    def _state_callback(self, msg: JointState):
        self._external_torque = msg.effort

    def _landed_callback(self, msg: VehicleLandDetected):
        self._is_landed = msg.landed

    def _control_loop(self):
        joint_pos, joint_vel, joint_effort = self.act()
        if self._is_started:
            if self._is_ended:
                return
            if self._is_landed:
                self._is_ended = True
                self.get_logger().info("Landing detected. Teleoperation command publishing stopped.")
                return
            self._publish_command(joint_pos, joint_vel, joint_effort)
        else:
            if not self._is_ended and joint_pos[-1] <= 0.0:
                self._is_started = True
                self.get_logger().info("Leader arm control started.")

    def _publish_command(self, joint_pos, joint_vel, joint_effort):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint_{i+1}" for i in self._equipments.single_arm.ids]
        msg.position = joint_pos.tolist()
        msg.velocity = joint_vel.tolist()
        msg.effort = joint_effort.tolist()
        self._command_pub.publish(msg)
