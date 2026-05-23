from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from acetele.config.config_loader import ConfigLoader
from acetele.core.make_robot import make_robot
from acetele.equipment.feetech.feetech_driver import TorqueEnable
from acetele.equipment.feetech.linker import Linker
from acetele.robot.base_robot import BaseEquipmentLibrary, BaseRobot


@dataclass
class AceLeaderEquipmentLibrary(BaseEquipmentLibrary):
    single_arm: Linker


class AceLeaderRobot(BaseRobot):
    def __init__(self, config_loader: ConfigLoader):
        super().__init__(config_loader)
        (single_arm_config,) = self._config_loader.get_linker_config()
        self._equipments: AceLeaderEquipmentLibrary = AceLeaderEquipmentLibrary(
            single_arm=Linker(
                config=single_arm_config,
                pin_model=self.get_pin_model(),
            ),
        )

    def act(
        self,
        encode_gripper: bool = True,
    ) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        return self._equipments.single_arm.act(
            encode_gripper=encode_gripper,
        )

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        self._equipments.single_arm.apply_torque_feedback(external_torque)

    def set_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ):
        self._equipments.single_arm.set_position(
            positions=positions,
            ids=ids,
            velocities=velocities,
            accelerations=accelerations,
            torque=torque,
        )

    def set_torque_enable(self, enable: TorqueEnable, ids: Optional[Sequence[int]] = None):
        self._equipments.single_arm.set_torque_enable(enable=enable, ids=ids)


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
