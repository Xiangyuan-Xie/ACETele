from pathlib import Path
from typing import Any, Dict, Tuple, Union

import tomli

_STATION_MAP = {
    "ace_leader": ("acetele.station.ace_leader.ace_leader", "AceLeaderStation"),
    "ace_leader_ros2": ("acetele.station.ace_leader.ace_leader_ros2", "AceLeaderROS2Station"),
    "ace_follower": ("acetele.station.ace_follower.ace_follower", "AceFollowerStation"),
    "ace_follower_ros2": ("acetele.station.ace_follower.ace_follower_ros2", "AceFollowerROS2Station"),
}

__all__ = ["ConfigLoader"]


class ConfigLoader:
    def __init__(self, config_dir: Path = Path(__file__).parent, entry_config_name: str = "default.toml"):
        self._config_dir = Path(config_dir).expanduser().resolve()
        entry_config_path = self._config_dir / entry_config_name

        with open(entry_config_path, "rb") as f:
            self._entry_config = tomli.load(f)

        if "config_file" in self._entry_config["basic"]:
            station_config_path = self._config_dir / self._entry_config["basic"]["config_file"]
            with open(station_config_path, "rb") as f:
                self._station_config = tomli.load(f)
        else:
            raise RuntimeError("Station config file not specified.")

    def get_station_type(self) -> str:
        return self._station_config["basic"]["station_type"]

    def get_station_info(self) -> Tuple[str, str]:
        return _STATION_MAP[self.get_station_type()]

    def get_linker_config(self) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        if "single" in self._station_config["linker"]:
            return self._station_config["linker"]["single"]
        elif "dual" in self._station_config["linker"]:
            return tuple(self._station_config["linker"]["dual"].values())
        else:
            raise ValueError("Linker type not supported")
