from pathlib import Path
from typing import Any, Dict, Tuple

import tomli

_STATION_MAP = {
    "ace_leader": {
        "default": ("acetele.station.ace_leader.ace_leader", "AceLeaderStation"),
        "ros2": ("acetele.station.ace_leader.ace_leader_ros2", "AceLeaderROS2Station"),
        "mock": ("acetele.station.ace_leader.ace_leader_mock", "AceLeaderMockStation"),
    },
    "ace_follower": {
        "default": ("acetele.station.ace_follower.ace_follower", "AceFollowerStation"),
        "ros2": ("acetele.station.ace_follower.ace_follower_ros2", "AceFollowerROS2Station"),
        "mock": ("acetele.station.ace_follower.ace_follower_mock", "AceFollowerMockStation"),
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
            station_config_path = self._config_dir / self._entry_config["basic"]["config_file"]
            with open(station_config_path, "rb") as f:
                self._station_config = tomli.load(f)
        else:
            raise RuntimeError("Station config file not specified.")

    def get_station_type(self) -> str:
        return self._station_config["basic"]["station_type"]

    def get_backend(self) -> str:
        return self._station_config["basic"].get("backend", "default")

    def get_station_info(self) -> Tuple[str, str]:
        station_type = self.get_station_type()
        backend = self.get_backend()
        if station_type in _STATION_MAP:
            if backend in _STATION_MAP[station_type]:
                return _STATION_MAP[station_type][backend]
            else:
                raise ValueError(f"Backend '{backend}' not supported for station type '{station_type}'")
        else:
            raise ValueError(f"Station type '{station_type}' not supported")

    def get_linker_config(self) -> Tuple[Dict[str, Any], ...]:
        if "single" in self._station_config["linker"]:
            return (self._station_config["linker"]["single"],)
        elif "dual" in self._station_config["linker"]:
            return tuple(self._station_config["linker"]["dual"].values())
        else:
            raise ValueError("Linker type not supported")
