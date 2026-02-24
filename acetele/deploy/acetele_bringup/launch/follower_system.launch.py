import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    follower_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ace_robot_ros2"),
                "launch",
                "ace_robot.launch.py",
            )
        ),
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py",
            )
        ),
        launch_arguments={
            "camera_name": "front",
            "rgb_camera.color_profile": "424x240x30",
            "rgb_camera.enable_auto_exposure": "false",
            "depth_module.depth_profile": "480x270x30",
            "depth_module.enable_auto_exposure": "false",
            "enable_sync": "true",
            "decimation_filter.enable": "true",
            "align_depth.enable": "true",
        }.items(),
    )

    odometry_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros2_px4_odometry"),
                "launch",
                "generic_odometry.launch.py",
            )
        ),
        launch_arguments={
            "config_file": "nokov_mocap_config.yaml",
        }.items(),
    )

    return LaunchDescription(
        [
            follower_launch,
            realsense_launch,
            odometry_launch,
        ]
    )
