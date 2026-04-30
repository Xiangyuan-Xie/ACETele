import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("ace_robot_ros2"),
        "config",
        "ace_robot_params.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value="",
                description="Path to the ACETele robot TOML config.",
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
