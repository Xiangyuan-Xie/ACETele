import time
from dataclasses import dataclass
from typing import Sequence

from acetele.config.config_loader import ConfigLoader
from acetele.equipment.feetech.linker import Linker
from acetele.station.base_station import BaseEquipmentLibrary, BaseStation


@dataclass
class FollowerEquipmentLibrary(BaseEquipmentLibrary):
    single_arm: Linker


class FollowerStation(BaseStation):
    def __init__(self, config_loader: ConfigLoader):
        super().__init__(config_loader)
        self._equipments: FollowerEquipmentLibrary = FollowerEquipmentLibrary(
            single_arm=Linker(
                self._config_loader.config["basic"]["station_type"],
                self._config_loader.config["linker"]["single"],
            ),
        )

    def act(self) -> Sequence[float]:
        pos, _, _ = self._equipments.single_arm.act()
        return pos

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        self._equipments.single_arm.apply_torque_feedback(external_torque)

    def set_position(self, target_pos: Sequence[float]):
        self._equipments.single_arm.set_position(positions=target_pos)

    def move_position(self, target_pos: Sequence[float]):
        self._equipments.single_arm.move_position(positions=target_pos)


if __name__ == "__main__":
    config_loader = ConfigLoader()
    hardware = FollowerStation(config_loader)
    try:
        while True:
            print(hardware.act())
            time.sleep(0.5)
    except KeyboardInterrupt:
        hardware.close()
