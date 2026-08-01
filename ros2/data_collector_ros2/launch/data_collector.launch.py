import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch the bag recorder with its package-shared parameter file."""

    config = os.path.join(
        get_package_share_directory("data_collector_ros2"),
        "config",
        "data_collector_params.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="data_collector_ros2",
                executable="data_collector_node",
                name="data_collector_node",
                output="screen",
                parameters=[config],
            ),
        ]
    )
