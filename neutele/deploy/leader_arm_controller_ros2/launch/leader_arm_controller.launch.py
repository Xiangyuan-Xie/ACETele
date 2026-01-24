import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("leader_arm_controller_ros2"),
        "config",
        "leader_arm_controller_params.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="leader_arm_controller_ros2",
                executable="leader_arm_controller",
                name="leader_arm_controller",
                output="screen",
                parameters=[config],
            )
        ]
    )
