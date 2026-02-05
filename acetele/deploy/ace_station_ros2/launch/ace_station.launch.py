import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("ace_station_ros2"),
        "config",
        "ace_station_params.yaml",
    )

    station_type_arg = DeclareLaunchArgument(
        "station_type", default_value="ace_leader", description="Type of the station: ace_leader or ace_follower"
    )

    return LaunchDescription(
        [
            station_type_arg,
            Node(
                package="ace_station_ros2",
                executable="ace_station_node",
                name="ace_station_node",
                output="screen",
                parameters=[config, {"config_name": [LaunchConfiguration("station_type"), ".toml"]}],
            ),
        ]
    )
