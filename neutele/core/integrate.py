import importlib
import time
from typing import Optional, Sequence

from config.config_loader import ConfigLoader
from station.base_station import BaseStation


class TeleCore:
    def __init__(self, config_path: Optional[str] = None):
        self._config_loader = ConfigLoader(config_path)
        module_name, class_name = self._config_loader.get_station_info()
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        self._station: BaseStation = cls(self._config_loader)

    def act(self) -> Sequence[float]:
        return self._station.act()

    def calibrate(self) -> bool:
        return self._station.calibrate()

    def close(self):
        self._station.close()


if __name__ == "__main__":
    tele_core = TeleCore()
    while True:
        # tele_core.act()
        print(tele_core.act())
        time.sleep(0.05)
