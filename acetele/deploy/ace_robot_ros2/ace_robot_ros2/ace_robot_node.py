from pathlib import Path

import rclpy
from rclpy.node import Node

from acetele.config.config_loader import ConfigLoader
from acetele.core.make_robot import make_robot


def main():
    rclpy.init()
    parameter_node = Node("ace_robot_node")
    parameter = parameter_node.declare_parameter("config_path", "")
    config_path = parameter.value
    parameter_node.destroy_node()

    config_loader = (
        ConfigLoader(Path(config_path), backend_override="ros2")
        if config_path
        else ConfigLoader(backend_override="ros2")
    )
    robot_node = make_robot(config_loader)
    if not isinstance(robot_node, Node):
        raise ValueError("Robot node must be a rclpy.node.Node instance")
    try:
        rclpy.spin(robot_node)
    except KeyboardInterrupt:
        pass
    finally:
        robot_node.close()
        robot_node.destroy_node()
