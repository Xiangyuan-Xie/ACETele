import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from neutele.config.config_loader import ConfigLoader
from neutele.equipment.base_equipment import BaseEquipment
from neutele.equipment.feetech.feetech_driver import FeeTechDriver


class Linker(BaseEquipment):
    def __init__(self, config: Dict[str, Any], driver: Optional[FeeTechDriver] = None):
        super().__init__()
        self._ids = np.array(config["joint_ids"])
        self._signs = np.array(config["joint_signs"])
        self._home_poses = np.array(config["home_poses"])
        self._driver = driver if driver is not None else FeeTechDriver(self._ids, config["port"])

    def calibrate(self) -> bool:
        home_poses = (self._home_poses * 2048 / np.pi).astype(int)
        return self._driver.calibrate(self._ids, home_poses)

    def act(self) -> Sequence[float]:
        pos, _ = self._driver.get_pos_and_vel()
        pos = np.array([v for k, v in pos.items() if k in self._ids]) * self._signs * np.pi / 2048.0
        return pos

    def get_frequency(self) -> float:
        return self._driver.get_frequency()

    def close(self):
        self._driver.close()


if __name__ == "__main__":
    config_loader = ConfigLoader()
    linker = Linker(config_loader.config["linker"]["single"])
    while True:
        print(linker.act())
        time.sleep(0.05)
