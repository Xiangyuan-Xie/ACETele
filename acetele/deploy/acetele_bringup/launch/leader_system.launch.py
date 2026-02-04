import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    leader_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("leader_arm_controller_ros2"),
                "launch",
                "leader_arm_controller.launch.py",
            )
        )
    )

    visualization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("visualization_ros2"),
                "launch",
                "visualization.launch.py",
            )
        )
    )

    return LaunchDescription(
        [
            leader_launch,
            visualization_launch,
        ]
    )
