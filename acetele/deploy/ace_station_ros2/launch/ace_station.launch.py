import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("ace_station_ros2"),
        "config",
        "ace_station_params.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="ace_station_ros2",
                executable="ace_station_node",
                name="ace_station_node",
                output="screen",
                parameters=[config],
            ),
        ]
    )
