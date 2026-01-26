import importlib
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from acetele.config.config_loader import ConfigLoader
from acetele.station.base_station import BaseStation

np.set_printoptions(suppress=True)


class TeleCore:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self._config_loader = ConfigLoader(Path(config_path)) if config_path is not None else ConfigLoader()
        module_name, class_name = self._config_loader.get_station_info()
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        self._station: BaseStation = cls(self._config_loader)

    def act(self) -> Sequence[float]:
        return self._station.act()

    def close(self):
        self._station.close()

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        self._station.apply_torque_feedback(external_torque)


if __name__ == "__main__":
    tele_core = TeleCore()
    while True:
        # tele_core.act()
        print(np.around(tele_core.act(), 4))
        time.sleep(0.05)
