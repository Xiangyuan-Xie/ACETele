import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from acetele.config.config_loader import ConfigLoader
from acetele.core.make_robot import make_robot
from acetele.equipment.feetech.linker import Linker
from acetele.robot.base_robot import BaseEquipmentLibrary, BaseRobot


@dataclass
class AceFollowerEquipmentLibrary(BaseEquipmentLibrary):
    single_arm: Linker


class AceFollowerRobot(BaseRobot):
    def __init__(self, config_loader: ConfigLoader):
        super().__init__(config_loader)
        (single_arm_config,) = self._config_loader.get_linker_config()
        self._equipments: AceFollowerEquipmentLibrary = AceFollowerEquipmentLibrary(
            single_arm=Linker(
                config=single_arm_config,
            ),
        )

    def act(self) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        return self._equipments.single_arm.act()

    def set_position(self, positions: Sequence[float], ids: Optional[Sequence[int]] = None):
        self._equipments.single_arm.set_position(positions=positions, ids=ids)

    def move_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        torque: Optional[Sequence[float]] = None,
    ):
        self._equipments.single_arm.move_position(positions=positions, ids=ids, torque=torque)


if __name__ == "__main__":
    config_loader = ConfigLoader()
    hardware = make_robot()
    try:
        with np.printoptions(suppress=True):
            while True:
                print(hardware.act())
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        hardware.close()
