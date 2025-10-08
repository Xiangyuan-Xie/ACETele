import tomllib
from pathlib import Path
from typing import Optional, Tuple, Union

STATION_MAP = {
    "flying_hand": ("neutele.station.flying_hand.flying_hand", "FlyingHandStation"),
}


class ConfigLoader:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        config_path = Path(config_path or Path(__file__).parent / "default.toml").expanduser().resolve()
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

    def get_station_info(self) -> Tuple[str, str]:
        return STATION_MAP[self.config["basic"]["station_type"]]
