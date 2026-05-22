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


class Calibration:
    def __init__(
        self,
        config_path: Optional[Path] = None,
        driver_factory: Callable[[Any, str], FeeTechDriver] = FeeTechDriver,
    ):
        self._config_loader = None if config_path is not None else ConfigLoader()
        self._robot_type = None
        self._backend = None
        self._config = self._load_linker_config(config_path)
        self._driver_factory = driver_factory

    def calibrate(self):
        results = []
        for i, linker_config in enumerate(self._config):
            ids = tuple(linker_config["joint_ids"])
            port = linker_config["port"]
            driver = self._driver_factory(ids, port)
            try:
                home_poses = self._encode_home_poses(linker_config)
                driver.calibrate(ids, home_poses)
                pos, _, _ = driver.get_state()
                results.append(
                    ArmCalibrationResult(
                        arm_index=i,
                        port=port,
                        ids=ids,
                        encoded_home_poses=tuple(int(pose) for pose in home_poses),
                        positions_after_calibration=dict(pos),
                    )
                )
                print(f"臂{i}标定完成，当前姿态：{np.array(list(pos.values()))}.")
            except Exception as exc:
                raise CalibrationError(f"Failed to calibrate arm {i} on port {port}: {exc}") from exc
            finally:
                driver.close()
        return tuple(results)

    @staticmethod
    def _encode_home_poses(linker_config):
        home_poses = decode_normalized_gripper_home_pose(
            linker_config["home_poses"],
            linker_config["joint_ids"],
            linker_config["gripper_id"],
            linker_config["gripper_type"],
        )
        joint_signs = np.array(linker_config["joint_signs"], dtype=float)
        return np.rint(home_poses * joint_signs * 2048.0 / np.pi).astype(int)

    def _load_linker_config(self, config_path):
        if config_path is None:
            return self._config_loader.get_linker_config()

        config_path = Path(config_path).expanduser().resolve()
        with open(config_path, "rb") as f:
            config = tomli.load(f)

        if "config_file" in config.get("basic", {}):
            self._config_loader = ConfigLoader(config_path)
            return self._config_loader.get_linker_config()

        self._robot_type = config.get("basic", {}).get("robot_type")
        self._backend = config.get("basic", {}).get("backend")
        linker_config = config["linker"]
        if "single" in linker_config:
            return (linker_config["single"],)
        if "dual" in linker_config:
            return tuple(linker_config["dual"].values())
        raise ValueError("Linker type not supported")

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
