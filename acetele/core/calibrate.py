from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from acetele.config.config_loader import ConfigLoader
from acetele.config.robot_config import ArmConfig, FeeTechGripperConfig
from acetele.equipment.feetech.feetech_driver import (
    FEETECH_SIGNED_15_BIT_MAX,
    FeeTechDriver,
    normalize_feetech_servo_ids,
)
from acetele.equipment.feetech.servo_specs import validate_feetech_servo_models


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
        self._config_loader = ConfigLoader() if config_path is None else ConfigLoader(config_path)
        self._robot_config = self._config_loader.get_robot_config()
        self._driver_factory = driver_factory

    def calibrate(self) -> tuple[ArmCalibrationResult, ...]:
        if self._robot_config.backend != "physical":
            raise CalibrationError(
                "Calibration requires backend='physical'; "
                f"got backend='{self._robot_config.backend}'"
            )

        groups: list[_CalibrationGroup] = []
        for index, assembly in enumerate(self._robot_config.arm_assemblies):
            end_effector = assembly.end_effector
            if end_effector is not None and not isinstance(end_effector, FeeTechGripperConfig):
                raise CalibrationError(
                    f"Calibration does not support {type(end_effector).__name__} on arm {index}"
                )
            try:
                validate_feetech_servo_models(
                    assembly.arm.servo_models,
                    context=f"arm {index}",
                )
                if isinstance(end_effector, FeeTechGripperConfig):
                    validate_feetech_servo_models(
                        (end_effector.servo_model,),
                        context=f"gripper on arm {index}",
                    )
            except ValueError as exc:
                raise CalibrationError(str(exc)) from exc
            groups.extend(self._build_calibration_groups(index, assembly.arm, end_effector))

        return tuple(self._calibrate_group(group) for group in groups)

    def _build_calibration_groups(
        self,
        arm_index: int,
        arm: ArmConfig,
        gripper: Optional[FeeTechGripperConfig] = None,
    ) -> tuple[_CalibrationGroup, ...]:
        if arm.port is None:
            raise CalibrationError(f"arm {arm_index} requires a serial port for calibration")
        if gripper is not None and gripper.port is None:
            raise CalibrationError(f"gripper on arm {arm_index} requires a serial port for calibration")

        if gripper is None or gripper.port == arm.port:
            group = self._make_calibration_group(
                arm_index=arm_index,
                port=arm.port,
                ids=self._calibration_ids(arm, gripper),
                encoded_home_poses=self._encode_home_poses(arm, gripper),
                label="arm",
            )
            return (group,)

        arm_group = self._make_calibration_group(
            arm_index=arm_index,
            port=arm.port,
            ids=self._calibration_ids(arm),
            encoded_home_poses=self._encode_home_poses(arm),
            label="arm",
        )
        assert gripper.port is not None
        gripper_group = self._make_calibration_group(
            arm_index=arm_index,
            port=gripper.port,
            ids=gripper.joint_ids,
            encoded_home_poses=self._encode_gripper_home_pose(gripper),
            label="gripper",
        )
        return arm_group, gripper_group

    @staticmethod
    def _make_calibration_group(
        *,
        arm_index: int,
        port: str,
        ids: Tuple[int, ...],
        encoded_home_poses: Sequence[float],
        label: str,
    ) -> _CalibrationGroup:
        encoded = np.asarray(encoded_home_poses, dtype=float)
        label_text = " gripper" if label == "gripper" else ""
        context = f"arm {arm_index}{label_text} on port {port}"
        try:
            normalized_ids = normalize_feetech_servo_ids(
                ids,
                field_name=f"servo IDs for {context}",
            )
        except ValueError as exc:
            raise CalibrationError(str(exc)) from exc
        if encoded.ndim != 1 or len(encoded) != len(ids):
            raise CalibrationError(
                f"Encoded home poses for {context} must match the configured joint count"
            )
        if not np.all(np.isfinite(encoded)) or np.any(
            np.abs(encoded) > FEETECH_SIGNED_15_BIT_MAX
        ):
            raise CalibrationError(
                f"Encoded home poses for {context} must be within the signed 15-bit range "
                f"[-{FEETECH_SIGNED_15_BIT_MAX}, {FEETECH_SIGNED_15_BIT_MAX}]"
            )
        return _CalibrationGroup(
            arm_index=arm_index,
            port=port,
            ids=normalized_ids,
            encoded_home_poses=tuple(int(value) for value in encoded),
            label=label,
        )

    def _calibrate_group(self, group: _CalibrationGroup) -> ArmCalibrationResult:
        driver = self._driver_factory(group.ids, group.port)
        result: Optional[ArmCalibrationResult] = None
        calibration_error: Optional[BaseException] = None
        try:
            home_poses = np.asarray(group.encoded_home_poses, dtype=int)
            driver.calibrate(group.ids, home_poses)
            positions, _, _ = driver.get_state()
            result = ArmCalibrationResult(
                arm_index=group.arm_index,
                port=group.port,
                ids=group.ids,
                encoded_home_poses=group.encoded_home_poses,
                positions_after_calibration=dict(positions),
            )
        except BaseException as exc:
            calibration_error = exc

        close_error: Optional[BaseException] = None
        try:
            driver.close()
        except BaseException as exc:
            close_error = exc

        label_text = " gripper" if group.label == "gripper" else ""
        if calibration_error is not None:
            if not isinstance(calibration_error, Exception):
                raise calibration_error
            close_detail = (
                ""
                if close_error is None
                else f"; additionally failed to close driver: {close_error}"
            )
            raise CalibrationError(
                f"Failed to calibrate arm {group.arm_index}{label_text} on port "
                f"{group.port}: {calibration_error}{close_detail}"
            ) from calibration_error
        if close_error is not None:
            if not isinstance(close_error, Exception):
                raise close_error
            raise CalibrationError(
                f"Calibration completed for arm {group.arm_index}{label_text} on port "
                f"{group.port}, but failed to close driver: {close_error}"
            ) from close_error

        assert result is not None
        label_text_zh = "夹爪" if group.label == "gripper" else ""
        print(
            f"臂{group.arm_index}{label_text_zh}标定完成，"
            f"当前姿态：{np.array(list(result.positions_after_calibration.values()))}."
        )
        return result

    @staticmethod
    def _calibration_ids(
        arm: ArmConfig,
        gripper: Optional[FeeTechGripperConfig] = None,
    ) -> Tuple[int, ...]:
        return arm.joint_ids if gripper is None else arm.joint_ids + gripper.joint_ids

    @staticmethod
    def _encode_home_poses(
        arm: ArmConfig,
        gripper: Optional[FeeTechGripperConfig] = None,
    ) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            arm_home_poses = np.rint(
                np.asarray(arm.home_poses, dtype=float)
                * np.asarray(arm.joint_signs, dtype=float)
                * 2048.0
                / np.pi
            )
        if gripper is None:
            return arm_home_poses
        gripper_home_pose = np.asarray(
            [
                int(np.rint(gripper.home_pose * gripper.travel_range_counts))
                * gripper.joint_sign
            ],
            dtype=int,
        )
        return np.concatenate((arm_home_poses, gripper_home_pose))

    @staticmethod
    def _encode_gripper_home_pose(gripper: FeeTechGripperConfig) -> np.ndarray:
        return np.asarray(
            [
                int(np.rint(gripper.home_pose * gripper.travel_range_counts))
                * gripper.joint_sign
            ],
            dtype=int,
        )

    def get_robot_type(self) -> str:
        return self._robot_config.robot_type

    def get_backend(self) -> str:
        return self._robot_config.backend

    def get_runtime(self) -> str:
        return self._robot_config.runtime


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate configured physical FEETECH joints")
    parser.add_argument(
        "--config",
        type=Path,
        help="path to a robot TOML config or an entry config containing basic.config_file",
    )
    args = parser.parse_args(argv)
    calibration = Calibration(config_path=args.config)
    calibration.calibrate()


if __name__ == "__main__":
    main()
