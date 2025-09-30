from typing import Sequence

from neutele.config.config_loader import ConfigLoader
from neutele.equipment.feetech.linker import Linker
from neutele.station.base_station import BaseStation


class FlyingHandStation(BaseStation):
    def __init__(self, config_loader: ConfigLoader):
        super().__init__(config_loader)
        self._equipments["feetech_arm"] = Linker(self._config_loader.config["linker"]["single"])

    def act(self) -> Sequence[float]:
        return self._equipments["feetech_arm"].act()

    def calibrate(self) -> bool:
        return self._equipments["feetech_arm"].calibrate()
