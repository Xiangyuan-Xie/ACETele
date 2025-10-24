from typing import Sequence

from neutele.config.config_loader import ConfigLoader
from neutele.equipment.feetech.linker import Linker
from neutele.station.base_station import BaseStation


class FlyingHandStation(BaseStation):
    def __init__(self, config_loader: ConfigLoader):
        super().__init__(config_loader)
        self._equipments["feetech_arm"] = Linker(
            self._config_loader.config["basic"]["station_type"], self._config_loader.config["linker"]["single"]
        )

    def act(self) -> Sequence[float]:
        pos, _ = self._equipments["feetech_arm"].act()
        return pos

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        self._equipments["feetech_arm"].apply_torque_feedback(external_torque)
