from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState

from acetele.station.ace_leader.ace_leader import AceLeaderStation


class AceLeaderROS2Station(Node, AceLeaderStation):
    def __init__(self):
        Node.__init__(self, "ace_leader_station_node")
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

        period = 1.0 / self._control_rate
        self._timer = self.create_timer(period, self._control_loop)

        self._is_synced = False
        self._is_started = False

        self.get_logger().info("Leader arm controller node started.")

    def _state_callback(self, msg: JointState):
        if not self._is_started:
            if self._is_synced:
                self.set_position(ids=self._equipments.single_arm.ids[:-1], positions=msg.position[:-1])
            else:
                self.get_logger().info("Synchronizing to the follower arm...")
                self.move_position(ids=self._equipments.single_arm.ids[:-1], positions=msg.position[:-1])
                self._is_synced = True
                self._equipments.single_arm.start_control_loop()
                self.get_logger().info("Synchronization completed.")
        self._external_torque = msg.effort

    def _control_loop(self):
        if self._is_synced:
            joint_pos, joint_vel, joint_effort = self.act()
            if self._is_started:
                self._publish_command(joint_pos, joint_vel, joint_effort)
            else:
                if joint_pos[-1] <= 0.0:
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
