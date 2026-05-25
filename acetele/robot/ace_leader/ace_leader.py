from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from acetele.config.config_loader import ConfigLoader
from acetele.core.make_robot import make_robot
from acetele.equipment.feetech.feetech_driver import FeeTechDriver, TorqueEnable
from acetele.equipment.feetech.gripper import Gripper
from acetele.equipment.feetech.linker import Linker
from acetele.robot.base_robot import BaseEquipmentLibrary, BaseRobot


@dataclass
class AceLeaderEquipmentLibrary(BaseEquipmentLibrary):
    single_arm: Linker
    gripper: Optional[Gripper] = None


class AceLeaderRobot(BaseRobot):
    def __init__(self, config_loader: ConfigLoader):
        super().__init__(config_loader)
        (single_arm_config,) = self._config_loader.get_linker_config()
        gripper_configs = self._config_loader.get_gripper_config()
        gripper_config = gripper_configs[0] if gripper_configs else None
        arm_joint_ids = list(single_arm_config["joint_ids"])
        robot_joint_ids = list(arm_joint_ids)
        gripper_id: Optional[int] = None
        if gripper_config is not None:
            if "joint_id" not in gripper_config:
                raise ValueError("gripper.single.joint_id must be specified.")
            if "port" not in gripper_config:
                raise ValueError("gripper.single.port must be specified.")
            gripper_id = int(gripper_config["joint_id"])
            robot_joint_ids.append(gripper_id)
        self._drivers: tuple[FeeTechDriver, ...]
        gripper_driver: Optional[FeeTechDriver]
        if gripper_config is None:
            self._driver = FeeTechDriver(arm_joint_ids, single_arm_config["port"])
            gripper_driver = None
            self._drivers = (self._driver,)
        elif gripper_config["port"] == single_arm_config["port"]:
            self._driver = FeeTechDriver(robot_joint_ids, single_arm_config["port"])
            gripper_driver = self._driver
            self._drivers = (self._driver,)
        else:
            self._driver = FeeTechDriver(arm_joint_ids, single_arm_config["port"])
            assert gripper_id is not None
            gripper_driver = FeeTechDriver([gripper_id], gripper_config["port"])
            self._drivers = (self._driver, gripper_driver)
        arm_pin_model = None
        if single_arm_config["enable_gravity_compensation"]:
            fixed_joint_names = []
            if gripper_id is not None:
                fixed_joint_names = [f"joint_{gripper_id + 1}"]
            arm_pin_model = self._get_pin_model_with_fixed_joints(fixed_joint_names)
        gripper = None
        if gripper_config is not None:
            assert gripper_driver is not None
            gripper = Gripper(gripper_config, driver=gripper_driver)
        self._equipments: AceLeaderEquipmentLibrary = AceLeaderEquipmentLibrary(
            single_arm=Linker(
                config=single_arm_config,
                driver=self._driver,
                pin_model=arm_pin_model,
            ),
            gripper=gripper,
        )
        self.ids = np.array(robot_joint_ids)
        self.gripper_id = gripper_id
        self.gripper_index = None if gripper_config is None else len(arm_joint_ids)

    def act(self) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        arm_pos, arm_vel, arm_effort = self._equipments.single_arm.act()
        gripper = self._equipments.gripper
        if gripper is None:
            return arm_pos, arm_vel, arm_effort
        gripper_pos, gripper_vel, gripper_effort = gripper.act()
        combined_values = []
        for arm_values, gripper_values in (
            (arm_pos, gripper_pos),
            (arm_vel, gripper_vel),
            (arm_effort, gripper_effort),
        ):
            combined_values.append(
                np.concatenate(
                    (
                        np.asarray(arm_values, dtype=float),
                        np.array([float(np.asarray(gripper_values, dtype=float)[0])]),
                    )
                )
            )
        return (
            combined_values[0],
            combined_values[1],
            combined_values[2],
        )

    def set_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ):
        ids_array = self.ids if ids is None else np.asarray(ids)
        positions_array = np.asarray(positions, dtype=float)
        arm_joint_ids = np.asarray(self._equipments.single_arm.ids)
        arm_mask = np.isin(ids_array, arm_joint_ids)
        profile_values: dict[str, Any] = {
            "velocities": velocities,
            "accelerations": accelerations,
            "torque": torque,
        }
        for name, values in profile_values.items():
            if values is None:
                continue
            values_array = np.asarray(values, dtype=float)
            if values_array.ndim == 0 or len(values_array) == len(ids_array):
                profile_values[name] = values_array
            else:
                raise ValueError(f"{name} must be scalar or match ids length.")
        if np.any(arm_mask):
            arm_profile_kwargs: dict[str, Any] = {}
            for name, values in profile_values.items():
                if values is None:
                    arm_profile_kwargs[name] = None
                    continue
                values_array = np.asarray(values, dtype=float)
                if values_array.ndim == 0:
                    arm_profile_kwargs[name] = float(values_array)
                elif len(values_array) == len(ids_array):
                    arm_profile_kwargs[name] = values_array[arm_mask]
                else:
                    arm_profile_kwargs[name] = values
            self._equipments.single_arm.set_position(
                positions=positions_array[arm_mask],
                ids=ids_array[arm_mask],
                **arm_profile_kwargs,
            )
        gripper_index = None
        if self._equipments.gripper is not None and self.gripper_id in ids_array:
            gripper_index = int(np.where(ids_array == self.gripper_id)[0][0])
        if gripper_index is not None:
            gripper = self._equipments.gripper
            assert gripper is not None
            gripper_profile_kwargs: dict[str, float] = {}
            for command_name, gripper_name in (
                ("velocities", "velocity"),
                ("accelerations", "acceleration"),
                ("torque", "torque"),
            ):
                values = profile_values[command_name]
                if values is None:
                    continue
                values_array = np.asarray(values, dtype=float)
                gripper_profile_kwargs[gripper_name] = (
                    float(values_array)
                    if values_array.ndim == 0
                    else float(values_array[gripper_index])
                    if len(values_array) == len(ids_array)
                    else float(values_array)
                )
            gripper.set_position(
                float(positions_array[gripper_index]),
                **gripper_profile_kwargs,
            )

    def set_torque_enable(self, enable: TorqueEnable, ids: Optional[Sequence[int]] = None):
        ids_array = self.ids if ids is None else np.asarray(ids)
        arm_joint_ids = np.asarray(self._equipments.single_arm.ids)
        arm_mask = np.isin(ids_array, arm_joint_ids)
        if np.any(arm_mask):
            self._equipments.single_arm.set_torque_enable(enable=enable, ids=ids_array[arm_mask])
        if self._equipments.gripper is not None and self.gripper_id in ids_array:
            self._equipments.gripper.set_torque_enable(enable)

    def close(self):
        for equipment in (self._equipments.single_arm, self._equipments.gripper):
            if equipment is not None:
                equipment.close()
        for driver in self._drivers if hasattr(self, "_drivers") else (self._driver,):
            driver.close()


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
