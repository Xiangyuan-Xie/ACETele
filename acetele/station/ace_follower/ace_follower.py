import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, cast

from acetele.config.config_loader import ConfigLoader
from acetele.equipment.feetech.linker import Linker
from acetele.station.base_station import BaseEquipmentLibrary, BaseStation, make_station


@dataclass
class AceFollowerEquipmentLibrary(BaseEquipmentLibrary):
    single_arm: Linker


class AceFollowerStation(BaseStation):
    def __init__(self, config_loader: ConfigLoader):
        super().__init__(config_loader)
        self._equipments: AceFollowerEquipmentLibrary = AceFollowerEquipmentLibrary(
            single_arm=Linker(
                config=cast(dict, self._config_loader.get_linker_config()),
            ),
        )

    def act(self) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        return self._equipments.single_arm.act()

    def set_position(self, positions: Sequence[float], ids: Optional[Sequence[int]] = None):
        self._equipments.single_arm.set_position(positions=positions, ids=ids)

    def move_position(self, positions: Sequence[float], ids: Optional[Sequence[int]] = None):
        self._equipments.single_arm.move_position(positions=positions, ids=ids)


if __name__ == "__main__":
    config_loader = ConfigLoader()
    hardware = make_station()
    try:
        while True:
            print(hardware.act())
            time.sleep(0.5)
    except KeyboardInterrupt:
        hardware.close()
