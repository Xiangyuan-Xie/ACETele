from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import tomli

from acetele.config.config_loader import ConfigLoader
from acetele.equipment.feetech.feetech_driver import FeeTechDriver
from acetele.utils.gripper import decode_normalized_gripper_home_pose


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArmCalibrationResult:
    arm_index: int
    port: str
    ids: Tuple[int, ...]
    encoded_home_poses: Tuple[int, ...]
    positions_after_calibration: Dict[int, int]


@dataclass(frozen=True)
class _CalibrationGroup:
    arm_index: int
    port: str
    ids: Tuple[int, ...]
    encoded_home_poses: Tuple[int, ...]
    label: str


class Calibration:
    def __init__(
        self,
        config_path: Optional[Path] = None,
        driver_factory: Callable[[Any, str], FeeTechDriver] = FeeTechDriver,
    ):
        self._config_loader = None if config_path is not None else ConfigLoader()
        self._robot_type = None
        self._backend = None
        self._linker_config, self._gripper_config = self._load_robot_config(config_path)
        self._driver_factory = driver_factory

    def calibrate(self):
        results = []
        for i, linker_config in enumerate(self._linker_config):
            gripper_config = self._gripper_config[i] if i < len(self._gripper_config) else None
            results.extend(self._calibrate_arm(i, linker_config, gripper_config))
        return tuple(results)

    def _calibrate_arm(self, arm_index, linker_config, gripper_config=None):
        if gripper_config is not None and "joint_id" not in gripper_config:
            raise CalibrationError("gripper.single.joint_id must be specified.")
        if gripper_config is not None and "port" not in gripper_config:
            raise CalibrationError("gripper.single.port must be specified.")

        if gripper_config is None or gripper_config["port"] == linker_config["port"]:
            try:
                group = _CalibrationGroup(
                    arm_index=arm_index,
                    port=linker_config["port"],
                    ids=self._calibration_ids(linker_config, gripper_config),
                    encoded_home_poses=tuple(
                        int(pose) for pose in self._encode_home_poses(linker_config, gripper_config)
                    ),
                    label="arm",
                )
            except Exception as exc:
                raise CalibrationError(
                    f"Failed to calibrate arm {arm_index} on port {linker_config['port']}: {exc}"
                ) from exc
            return (self._calibrate_group(group),)

        try:
            arm_group = _CalibrationGroup(
                arm_index=arm_index,
                port=linker_config["port"],
                ids=self._calibration_ids(linker_config),
                encoded_home_poses=tuple(int(pose) for pose in self._encode_home_poses(linker_config)),
                label="arm",
            )
        except Exception as exc:
            raise CalibrationError(
                f"Failed to calibrate arm {arm_index} on port {linker_config['port']}: {exc}"
            ) from exc

        try:
            gripper_group = _CalibrationGroup(
                arm_index=arm_index,
                port=gripper_config["port"],
                ids=(self._gripper_joint_id(gripper_config),),
                encoded_home_poses=tuple(int(pose) for pose in self._encode_single_gripper_home_pose(gripper_config)),
                label="gripper",
            )
        except Exception as exc:
            raise CalibrationError(
                f"Failed to calibrate arm {arm_index} gripper on port {gripper_config['port']}: {exc}"
            ) from exc

        return tuple(self._calibrate_group(group) for group in (arm_group, gripper_group))

    def _calibrate_group(self, group: _CalibrationGroup):
        driver = self._driver_factory(group.ids, group.port)
        try:
            home_poses = np.asarray(group.encoded_home_poses, dtype=int)
            driver.calibrate(group.ids, home_poses)
            pos, _, _ = driver.get_state()
            result = ArmCalibrationResult(
                arm_index=group.arm_index,
                port=group.port,
                ids=group.ids,
                encoded_home_poses=group.encoded_home_poses,
                positions_after_calibration=dict(pos),
            )
            label_text = "夹爪" if group.label == "gripper" else ""
            print(f"臂{group.arm_index}{label_text}标定完成，当前姿态：{np.array(list(pos.values()))}.")
            return result
        except Exception as exc:
            label_text = " gripper" if group.label == "gripper" else ""
            raise CalibrationError(
                f"Failed to calibrate arm {group.arm_index}{label_text} on port {group.port}: {exc}"
            ) from exc
        finally:
            driver.close()

    @staticmethod
    def _calibration_ids(linker_config, gripper_config=None):
        ids = tuple(linker_config["joint_ids"])
        if gripper_config is None:
            return ids
        return ids + (Calibration._gripper_joint_id(gripper_config),)

    @staticmethod
    def _encode_home_poses(linker_config, gripper_config=None):
        if gripper_config is not None:
            if "port" not in gripper_config:
                raise ValueError("gripper.single.port must be specified.")
            home_poses = np.asarray(tuple(linker_config["home_poses"]) + (float(gripper_config.get("home_pose", 0.0)),))
            joint_signs = np.asarray(
                tuple(linker_config["joint_signs"]) + (float(gripper_config["joint_sign"]),),
                dtype=float,
            )
            gripper_home_pose = home_poses[-1]
            if not 0.0 <= gripper_home_pose <= 1.0:
                raise ValueError("Gripper home pose must be between 0.0 and 1.0.")
            gripper_scale = decode_normalized_gripper_home_pose(
                [gripper_home_pose],
                [Calibration._gripper_joint_id(gripper_config)],
                Calibration._gripper_joint_id(gripper_config),
                gripper_config["gripper_type"],
            )[0]
            home_poses[-1] = gripper_scale
            return np.rint(home_poses * joint_signs * 2048.0 / np.pi).astype(int)

        home_poses = np.asarray(linker_config["home_poses"], dtype=float)
        joint_signs = np.array(linker_config["joint_signs"], dtype=float)
        return np.rint(home_poses * joint_signs * 2048.0 / np.pi).astype(int)

    @staticmethod
    def _encode_single_gripper_home_pose(gripper_config):
        if "port" not in gripper_config:
            raise ValueError("gripper.single.port must be specified.")
        gripper_home_pose = float(gripper_config.get("home_pose", 0.0))
        if not 0.0 <= gripper_home_pose <= 1.0:
            raise ValueError("Gripper home pose must be between 0.0 and 1.0.")
        gripper_scale = decode_normalized_gripper_home_pose(
            [gripper_home_pose],
            [Calibration._gripper_joint_id(gripper_config)],
            Calibration._gripper_joint_id(gripper_config),
            gripper_config["gripper_type"],
        )[0]
        return np.rint(
            np.asarray([gripper_scale], dtype=float) * float(gripper_config["joint_sign"]) * 2048.0 / np.pi
        ).astype(int)

    @staticmethod
    def _gripper_joint_id(gripper_config):
        if "joint_id" not in gripper_config:
            raise ValueError("gripper.single.joint_id must be specified.")
        return int(gripper_config["joint_id"])

    def _load_robot_config(self, config_path):
        if config_path is None:
            return self._config_loader.get_linker_config(), self._config_loader.get_gripper_config()

        config_path = Path(config_path).expanduser().resolve()
        with open(config_path, "rb") as f:
            config = tomli.load(f)

        if "config_file" in config.get("basic", {}):
            self._config_loader = ConfigLoader(config_path)
            return self._config_loader.get_linker_config(), self._config_loader.get_gripper_config()

        self._robot_type = config.get("basic", {}).get("robot_type")
        self._backend = config.get("basic", {}).get("backend")
        linker_config = ConfigLoader._get_equipment_config(config, "linker")
        if not linker_config:
            raise ValueError("Linker type not supported")
        return linker_config, ConfigLoader._get_equipment_config(config, "gripper")

    def get_robot_type(self):
        if self._config_loader is not None:
            return self._config_loader.get_robot_type()
        return self._robot_type

    def get_backend(self):
        if self._config_loader is not None:
            return self._config_loader.get_backend()
        return self._backend


def main():
    calibration = Calibration()
    calibration.calibrate()


if __name__ == "__main__":
    main()
