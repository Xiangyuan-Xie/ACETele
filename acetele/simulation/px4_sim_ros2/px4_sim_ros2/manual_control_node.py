import math

import numpy as np
import rclpy
from px4_msgs.msg import ManualControlSetpoint
from rclpy.node import Node

from acetele.equipment.joystick.joystick_driver import JDKFPVDriver


class ManualControlNode(Node):
    def __init__(self) -> None:
        super().__init__("manual_control_from_joystick")
        self._driver = JDKFPVDriver()
        self._publisher = self.create_publisher(
            ManualControlSetpoint,
            "/fmu/in/manual_control_input",
            10,
        )
        rate_hz = self.declare_parameter("publish_rate_hz", 50.0).value
        period = 1.0 / max(rate_hz, 1.0)
        self._timer = self.create_timer(period, self._publish_once)

    def _publish_once(self) -> None:
        data = self._driver.act()
        msg = ManualControlSetpoint()
        now = self.get_clock().now().nanoseconds // 1000
        msg.timestamp = now
        msg.timestamp_sample = now
        msg.valid = False
        msg.data_source = ManualControlSetpoint.SOURCE_UNKNOWN
        msg.roll = math.nan
        msg.pitch = math.nan
        msg.yaw = math.nan
        msg.throttle = math.nan
        msg.flaps = 0.0
        msg.aux1 = 0.0
        msg.aux2 = 0.0
        msg.aux3 = 0.0
        msg.aux4 = 0.0
        msg.aux5 = 0.0
        msg.aux6 = 0.0
        msg.sticks_moving = False
        msg.buttons = 0
        if data and data.get("connected"):
            mapped = data.get("mapped", {})
            roll = float(mapped.get("Roll", 0.0))
            pitch = float(mapped.get("Pitch", 0.0))
            yaw = float(mapped.get("Yaw", 0.0))
            throttle = float(mapped.get("Throttle", -1.0))
            roll = float(np.clip(roll, -1.0, 1.0))
            pitch = float(np.clip(pitch, -1.0, 1.0))
            yaw = float(np.clip(yaw, -1.0, 1.0))
            throttle = float(np.clip(throttle, -1.0, 1.0))
            msg.roll = roll
            msg.pitch = pitch
            msg.yaw = yaw
            msg.throttle = throttle
            msg.aux1 = float(mapped.get("Aux1", 0.0))
            msg.aux2 = float(mapped.get("Aux2", 0.0))
            msg.aux3 = float(mapped.get("Aux3", 0.0))
            msg.aux4 = float(mapped.get("Aux4", 0.0))
            msg.aux5 = 0.0
            msg.aux6 = 0.0
            msg.valid = True
            msg.data_source = ManualControlSetpoint.SOURCE_MAVLINK_0
            msg.sticks_moving = any(abs(v) > 0.01 for v in (roll, pitch, yaw, throttle))
        self._publisher.publish(msg)


def main():
    rclpy.init()
    node = ManualControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
