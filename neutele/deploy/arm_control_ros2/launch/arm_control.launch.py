from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="arm_control_ros2",
                executable="arm_control",
                name="arm_control",
                output="screen",
                parameters=[
                    {
                        "control_rate": 500.0,
                    }
                ],
            )
        ]
    )
