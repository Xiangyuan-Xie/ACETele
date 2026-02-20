from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def _load_px4_repo_path() -> str:
    share_dir = Path(get_package_share_directory("px4_sim_ros2"))
    cfg_path = share_dir / "config" / "px4_sim_config.yaml"
    default_path = "acetele/simulation/PX4-Autopilot"
    if cfg_path.is_file():
        params = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        px4_repo_path = params.get("px4_repo_path")
        if isinstance(px4_repo_path, str) and px4_repo_path.strip():
            return px4_repo_path.strip()
    return default_path


def generate_launch_description():
    px4_repo_path = _load_px4_repo_path()
    micro_xrce_agent = ExecuteProcess(
        cmd=["MicroXRCEAgent", "udp4", "-p", "8888"],
        output="screen",
    )
    px4_sitl = ExecuteProcess(
        cmd=["bash", "-lc", "make px4_sitl none"],
        cwd=px4_repo_path,
        output="screen",
    )
    joystick_node = Node(
        package="px4_sim_ros2",
        executable="manual_control",
        name="manual_control",
        output="screen",
    )
    return LaunchDescription(
        [
            micro_xrce_agent,
            px4_sitl,
            joystick_node,
        ]
    )
