import rclpy
from rclpy.node import Node

from acetele.core.make_robot import make_robot


def main():
    rclpy.init()
    robot_node = make_robot()
    print(type(robot_node))
    if not isinstance(robot_node, Node):
        raise ValueError("Robot node must be a rclpy.node.Node instance")
    try:
        rclpy.spin(robot_node)
    except KeyboardInterrupt:
        pass
    finally:
        robot_node.close()
        robot_node.destroy_node()
