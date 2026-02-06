import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    leader_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ace_station_ros2"),
                "launch",
                "ace_station.launch.py",
            )
        ),
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
