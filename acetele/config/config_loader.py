from pathlib import Path
from typing import Any, Dict, Tuple

import tomli

_ROBOT_MAP = {
    "ace_leader": {
        "default": ("acetele.robot.ace_leader.ace_leader", "AceLeaderRobot"),
        "ros2": ("acetele.robot.ace_leader.ace_leader_ros2", "AceLeaderROS2Robot"),
        "mock": ("acetele.robot.ace_leader.ace_leader_mock", "AceLeaderMockRobot"),
    },
    "ace_follower": {
        "default": ("acetele.robot.ace_follower.ace_follower", "AceFollowerRobot"),
        "ros2": ("acetele.robot.ace_follower.ace_follower_ros2", "AceFollowerROS2Robot"),
        "mock": ("acetele.robot.ace_follower.ace_follower_mock", "AceFollowerMockRobot"),
    },
}

__all__ = ["ConfigLoader"]


class ConfigLoader:
    def __init__(self, config_dir: Path = Path(__file__).parent, entry_config_name: str = "default.toml"):
        self._config_dir = Path(config_dir).expanduser().resolve()
        entry_config_path = self._config_dir / entry_config_name

        with open(entry_config_path, "rb") as f:
            self._entry_config = tomli.load(f)

        if "config_file" in self._entry_config["basic"]:
            robot_config_path = self._config_dir / self._entry_config["basic"]["config_file"]
            with open(robot_config_path, "rb") as f:
                self._robot_config = tomli.load(f)
        else:
            raise RuntimeError("Robot config file not specified.")

    def get_robot_type(self) -> str:
        return self._robot_config["basic"]["robot_type"]

    def get_backend(self) -> str:
        return self._robot_config["basic"].get("backend", "default")

    def get_robot_info(self) -> Tuple[str, str]:
        robot_type = self.get_robot_type()
        backend = self.get_backend()
        if robot_type in _ROBOT_MAP:
            if backend in _ROBOT_MAP[robot_type]:
                return _ROBOT_MAP[robot_type][backend]
            else:
                raise ValueError(f"Backend '{backend}' not supported for robot type '{robot_type}'")
        else:
            raise ValueError(f"Robot type '{robot_type}' not supported")

    def get_linker_config(self) -> Tuple[Dict[str, Any], ...]:
        if "single" in self._robot_config["linker"]:
            return (self._robot_config["linker"]["single"],)
        elif "dual" in self._robot_config["linker"]:
            return tuple(self._robot_config["linker"]["dual"].values())
        else:
            raise ValueError("Linker type not supported")
