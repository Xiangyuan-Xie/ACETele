import os
from importlib.resources import files

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch one runtime-backed robot with an overridable RobotSpec path."""

    config = os.path.join(
        get_package_share_directory("ace_robot_ros2"),
        "config",
        "ace_robot_params.yaml",
    )
    default_robot_config = str(
        files("acetele.config").joinpath(
            "ace_leader",
            "feetech_hls_ttl.toml",
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value=default_robot_config,
                description="Path to an ACETele RobotSpec TOML file.",
            ),
            Node(
                package="ace_robot_ros2",
                executable="ace_robot_node",
                name="ace_robot_node",
                output="screen",
                parameters=[config, {"config_path": LaunchConfiguration("config_path")}],
            ),
        ]
    )
