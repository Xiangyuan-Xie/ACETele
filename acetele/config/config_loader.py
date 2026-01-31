from pathlib import Path
from typing import Any, Dict, Tuple, Union

import tomli

STATION_MAP = {
    "leader": ("acetele.station.leader.leader", "LeaderStation"),
    "follower": ("acetele.station.follower.follower", "FollowerStation"),
}


class ConfigLoader:
    def __init__(self, config_dir: Path = Path(__file__).parent, config_name: str = "default.toml"):
        config_path = Path(config_dir / config_name).expanduser().resolve()
        with open(config_path, "rb") as f:
            self.config = tomli.load(f)

    def get_station_info(self) -> Tuple[str, str]:
        return STATION_MAP[self.config["basic"]["station_type"]]

    def get_linker_config(self) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        if "single" in self.config["linker"]:
            return self.config["linker"]["single"]
        elif "dual" in self.config["linker"]:
            return tuple(self.config["linker"]["dual"].values())
        else:
            raise ValueError("Linker type not supported")
