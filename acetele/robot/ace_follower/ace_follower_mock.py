from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import numpy as np

from acetele.config.config_loader import ConfigLoader
from acetele.robot.base_robot import BaseRobot


class AceFollowerMockRobot(BaseRobot):
    def __init__(self, config_loader: ConfigLoader):
        super().__init__(config_loader)
        (single_arm_config,) = self._config_loader.get_linker_config()
        gripper_configs = self._config_loader.get_gripper_config()
        self.ids = np.array(
            tuple(single_arm_config["joint_ids"]) + tuple(int(config["joint_id"]) for config in gripper_configs)
        )
        self.dof = len(self.ids)
        self.current_positions = np.zeros(self.dof)
        self.current_velocities = np.zeros(self.dof)
        self.current_currents = np.zeros(self.dof)
        print(f"[{self.name}_robot] Initialized with {self.dof} DOFs")

    def act(self) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        return self.current_positions, self.current_velocities, self.current_currents

    def set_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ):
        del velocities, accelerations, torque
        if ids is None:
            self.current_positions = np.asarray(positions)
            return
        next_positions = self.current_positions.copy()
        index_by_id = {int(ft_id): index for index, ft_id in enumerate(self.ids)}
        indices = [index_by_id[int(ft_id)] for ft_id in ids]
        next_positions[indices] = np.asarray(positions)
        self.current_positions = next_positions


if __name__ == "__main__":
    config_loader = ConfigLoader()
    hardware = AceFollowerMockRobot(config_loader)
    try:
        with np.printoptions(suppress=True):
            while True:
                print(hardware.act())
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        hardware.close()
