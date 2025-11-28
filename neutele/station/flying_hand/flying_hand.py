import time
from typing import Sequence

import numpy as np
from equipment.xbox.xbox_driver import Xbox360Driver

from neutele.config.config_loader import ConfigLoader
from neutele.equipment.feetech.linker import Linker
from neutele.station.base_station import BaseStation


class FlyingHandStation(BaseStation):
    def __init__(self, config_loader: ConfigLoader):
        super().__init__(config_loader)
        self._equipments["feetech_arm"] = Linker(
            self._config_loader.config["basic"]["station_type"], self._config_loader.config["linker"]["single"]
        )
        self._equipments["xbox"] = Xbox360Driver()

    def act(self) -> Sequence[float]:
        pos, _ = self._equipments["feetech_arm"].act()
        channel = self._equipments["xbox"].act()
        channel = np.array(
            [
                -channel["Axis"]["MainY"],
                -channel["Axis"]["MainX"],
                channel["Axis"]["Left"],
                channel["Axis"]["Right"],
                -channel["Axis"]["SubY"],
            ]
        )

        return np.concatenate([pos, [0], channel])

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        self._equipments["feetech_arm"].apply_torque_feedback(external_torque)


if __name__ == "__main__":
    config_loader = ConfigLoader()
    station = FlyingHandStation(config_loader)
    try:
        while True:
            print(station.act())
            time.sleep(0.5)
    except KeyboardInterrupt:
        station.close()
