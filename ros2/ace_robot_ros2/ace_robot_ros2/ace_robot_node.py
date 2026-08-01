"""ROS 2 executable that selects and owns one runtime-backed robot node."""

from pathlib import Path

import rclpy
from rclpy.node import Node

from acetele.config import load_robot_spec


def _make_robot_node(config_path: str):
    if not config_path:
        raise ValueError("config_path must point to a RobotSpec TOML file")
    spec = load_robot_spec(Path(config_path))
    if spec.model == "ace_follower":
        from .runtime_follower_node import RuntimeFollowerNode

        return RuntimeFollowerNode(spec)
    if spec.model == "ace_leader":
        from .runtime_leader_node import RuntimeLeaderNode

        return RuntimeLeaderNode(spec)
    raise ValueError(f"ROS 2 RobotRuntime does not support model '{spec.model}'")


def main():
    """Run the configured node and attempt every cleanup step on all exit paths."""

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

        robot_node = _make_robot_node(config_path)
        if not isinstance(robot_node, Node):
            raise ValueError("Robot node must be a rclpy.node.Node instance")
        rclpy.spin(robot_node)
    except KeyboardInterrupt:
        pass
    except BaseException as exc:
        primary_error = exc
    finally:
        # Cleanup is intentionally independent: a malformed candidate or one failed
        # close method must not prevent node destruction and rclpy shutdown.
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
