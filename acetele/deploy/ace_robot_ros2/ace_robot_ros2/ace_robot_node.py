from pathlib import Path

import rclpy
from rclpy.node import Node

from acetele.config.config_loader import ConfigLoader
from acetele.core.make_robot import make_robot


def main():
    initialized = False
    parameter_node = None
    robot_node = None
    primary_error = None
    try:
        rclpy.init()
        initialized = True
        parameter_node = Node("ace_robot_node")
        parameter = parameter_node.declare_parameter("config_path", "")
        config_path = parameter.value
        parameter_node.destroy_node()
        parameter_node = None

        config_loader = (
            ConfigLoader(Path(config_path), runtime_override="ros2")
            if config_path
            else ConfigLoader(runtime_override="ros2")
        )
        robot_node = make_robot(config_loader)
        if not isinstance(robot_node, Node):
            raise ValueError("Robot node must be a rclpy.node.Node instance")
        rclpy.spin(robot_node)
    except KeyboardInterrupt:
        pass
    except BaseException as exc:
        primary_error = exc
    finally:
        cleanup_error = None
        if robot_node is not None:
            for method_name in ("close", "destroy_node"):
                try:
                    cleanup = getattr(robot_node, method_name, None)
                    if callable(cleanup):
                        cleanup()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
        if parameter_node is not None:
            try:
                parameter_node.destroy_node()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if initialized:
            try:
                rclpy.shutdown()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc

    if primary_error is not None:
        if cleanup_error is not None:
            raise primary_error from cleanup_error
        raise primary_error
    if cleanup_error is not None:
        raise cleanup_error
