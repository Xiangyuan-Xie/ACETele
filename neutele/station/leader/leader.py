import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from neutele.config.config_loader import ConfigLoader
from neutele.equipment.feetech.linker import Linker
from neutele.equipment.xbox.xbox_driver import Xbox360Driver
from neutele.station.base_station import BaseEquipmentLibrary, BaseStation


@dataclass
class LeaderEquipmentLibrary(BaseEquipmentLibrary):
    single_arm: Linker
    xbox360: Xbox360Driver


class LeaderStation(BaseStation):
    def __init__(self, config_loader: ConfigLoader):
        super().__init__(config_loader)
        self._equipments: LeaderEquipmentLibrary = LeaderEquipmentLibrary(
            single_arm=Linker(
                self._config_loader.config["basic"]["station_type"],
                self._config_loader.config["linker"]["single"],
            ),
            xbox360=Xbox360Driver(),
        )

    def act(self) -> Sequence[float]:
        pos, _ = self._equipments.single_arm.act()
        # pos = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        # pos = np.array([-1.5708, 3.1416, 0.0, 0.0, 0.0])  # v2
        # pos = np.array([0.0, -0.785, -0.785, 0.0, 0.0])  # v1
        channel = np.zeros(5)
        # channel = self._equipments.xbox360.act()
        # channel = np.array(
        #     [
        #         -channel["Axis"]["MainY"],
        #         -channel["Axis"]["MainX"],
        #         channel["Axis"]["Left"],
        #         channel["Axis"]["Right"],
        #         -channel["Axis"]["SubY"],
        #     ]
        # )

        return np.concatenate([pos, channel])

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        self._equipments.single_arm.apply_torque_feedback(external_torque)

    def set_position(self, target_pos: Sequence[float]):
        self._equipments.single_arm.set_position(positions=target_pos)

    def move_position(self, target_pos: Sequence[float]):
        self._equipments.single_arm.move_position(positions=target_pos)


if __name__ == "__main__":
    config_loader = ConfigLoader()
    hardware = LeaderStation(config_loader)
    try:
        while True:
            print(hardware.act())
            time.sleep(0.5)
    except KeyboardInterrupt:
        hardware.close()
