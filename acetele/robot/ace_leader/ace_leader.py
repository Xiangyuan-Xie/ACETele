import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from acetele.config.config_loader import ConfigLoader
from acetele.core.make_robot import make_robot
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
        cal_torque_sign: bool = False,
    ) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        return self._equipments.single_arm.act(
            encode_gripper=encode_gripper,
            cal_torque_sign=cal_torque_sign,
        )

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        self._equipments.single_arm.apply_torque_feedback(external_torque)

    def set_position(self, positions: Sequence[float], ids: Optional[Sequence[int]] = None):
        self._equipments.single_arm.set_position(positions=positions, ids=ids)

    def set_position_and_torque(
        self,
        positions: Sequence[float],
        torques: Sequence[float],
        ids: Optional[Sequence[int]] = None,
    ):
        self._equipments.single_arm.set_position_and_torque(positions=positions, torques=torques, ids=ids)

    def move_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        torque: Optional[Sequence[float]] = None,
    ):
        self._equipments.single_arm.move_position(
            positions=positions,
            ids=ids,
            torque=torque,
        )


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
