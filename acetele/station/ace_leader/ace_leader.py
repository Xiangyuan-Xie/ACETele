import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from acetele.config.config_loader import ConfigLoader
from acetele.core.make_station import make_station
from acetele.equipment.feetech.linker import Linker
from acetele.station.base_station import BaseEquipmentLibrary, BaseStation


@dataclass
class AceLeaderEquipmentLibrary(BaseEquipmentLibrary):
    single_arm: Linker


class AceLeaderStation(BaseStation):
    def __init__(self, config_loader: ConfigLoader):
        super().__init__(config_loader)
        (single_arm_config,) = self._config_loader.get_linker_config()
        self._equipments: AceLeaderEquipmentLibrary = AceLeaderEquipmentLibrary(
            single_arm=Linker(
                config=single_arm_config,
                pin_model=self.get_pin_model(),
            ),
        )

    def act(self) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        return self._equipments.single_arm.act()

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        self._equipments.single_arm.apply_torque_feedback(external_torque)

    def set_position(self, positions: Sequence[float], ids: Optional[Sequence[int]] = None):
        self._equipments.single_arm.set_position(positions=positions, ids=ids)

    def move_position(self, positions: Sequence[float], ids: Optional[Sequence[int]] = None):
        self._equipments.single_arm.move_position(positions=positions, ids=ids)


if __name__ == "__main__":
    config_loader = ConfigLoader()
    hardware = make_station()
    try:
        with np.printoptions(suppress=True):
            while True:
                print(hardware.act())
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        hardware.close()
