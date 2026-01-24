import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("follower_arm_controller_ros2"),
        "config",
        "follower_arm_controller_params.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="follower_arm_controller_ros2",
                executable="follower_arm_controller",
                name="follower_arm_controller",
                output="screen",
                parameters=[config],
            )
        ]
    )
