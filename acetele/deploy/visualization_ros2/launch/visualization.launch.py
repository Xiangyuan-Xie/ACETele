import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory("visualization_ros2"), "config", "visualization_params.yaml")

    return LaunchDescription(
        [
            Node(
                package="visualization_ros2",
                executable="visualization",
                name="visualization",
                output="screen",
                parameters=[config],
            )
        ]
    )
