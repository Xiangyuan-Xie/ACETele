from pathlib import Path
from typing import Optional, Tuple, Union

import tomli

STATION_MAP = {
    "leader": ("neutele.station.leader.leader", "LeaderStation"),
    "follower": ("neutele.station.follower.follower", "FollowerStation"),
}


class ConfigLoader:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        config_path = Path(config_path or (Path(__file__).parent / "default.toml")).expanduser().resolve()
        with open(config_path, "rb") as f:
            self.config = tomli.load(f)

    def get_station_info(self) -> Tuple[str, str]:
        return STATION_MAP[self.config["basic"]["station_type"]]
