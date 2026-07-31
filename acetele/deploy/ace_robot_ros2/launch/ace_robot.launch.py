import os
from importlib.resources import files

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Launch one runtime-backed robot with an overridable RobotSpec path."""

    config = os.path.join(
        get_package_share_directory("ace_robot_ros2"),
        "config",
        "ace_robot_params.yaml",
    )
    default_robot_config = str(
        files("acetele.config").joinpath(
            "ace_leader",
            "feetech_hls_ttl.toml",
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value=default_robot_config,
                description="Path to an ACETele RobotSpec TOML file.",
            ),
            DeclareLaunchArgument(
                "teleop_mode",
                default_value="joint",
                description="Arm command mode: joint or ee_pose.",
            ),
            DeclareLaunchArgument(
                "translation_scale",
                default_value="2.0",
                description="Relative source-to-follower translation scale.",
            ),
            DeclareLaunchArgument(
                "rotation_scale",
                default_value="1.0",
                description="Relative source-to-follower rotation scale.",
            ),
            Node(
                package="ace_robot_ros2",
                executable="ace_robot_node",
                name="ace_robot_node",
                output="screen",
                parameters=[
                    config,
                    {
                        "config_path": LaunchConfiguration("config_path"),
                        "teleop_mode": ParameterValue(
                            LaunchConfiguration("teleop_mode"),
                            value_type=str,
                        ),
                        "translation_scale": ParameterValue(
                            LaunchConfiguration("translation_scale"),
                            value_type=float,
                        ),
                        "rotation_scale": ParameterValue(
                            LaunchConfiguration("rotation_scale"),
                            value_type=float,
                        ),
                    },
                ],
            ),
        ]
    )
