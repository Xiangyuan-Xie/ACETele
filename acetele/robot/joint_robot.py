from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from acetele.config.config_loader import ConfigLoader
from acetele.config.robot_config import (
    ArmAssemblyConfig,
    ArmConfig,
    FeeTechGripperConfig,
    MockJointConfig,
    O6DexterousHandConfig,
    RobotConfig,
)
from acetele.equipment.end_effector_factory import create_end_effector
from acetele.equipment.feetech.arm import FeeTechArm
from acetele.equipment.feetech.feetech_driver import (
    FeeTechDriver,
    TorqueEnable,
    normalize_feetech_servo_ids,
)
from acetele.equipment.feetech.servo_specs import validate_feetech_servo_models
from acetele.equipment.joint_device import (
    CompositeJointDevice,
    JointDevice,
    JointDeviceState,
    MockJointDevice,
)
from acetele.robot.base_robot import BaseRobot
from acetele.utils.angle import unwrap_near


@dataclass(frozen=True)
class ArmAssembly:
    config: ArmAssemblyConfig
    arm: JointDevice
    end_effector: Optional[JointDevice] = None


@dataclass(frozen=True)
class _PhysicalAssemblyMetadata:
    position_limits: Optional[tuple[tuple[float, ...], tuple[float, ...]]]
    pin_model: Any = None


class JointRobot(BaseRobot):
    ROBOT_TYPE = ""

    def __init__(self, config_loader: ConfigLoader | RobotConfig):
        super().__init__(config_loader)
        if self.robot_config.robot_type != self.ROBOT_TYPE:
            raise ValueError(
                f"{self.__class__.__name__} requires robot_type '{self.ROBOT_TYPE}', "
                f"got '{self.robot_config.robot_type}'"
            )
        if self.robot_config.backend == "physical" and len(self.robot_config.arm_assemblies) > 1:
            raise RuntimeError("physical dual follower backend is not implemented")

        self._preflight_urdf_joint_mappings()
        self._physical_build_metadata = (
            self._preflight_physical_assemblies()
            if self.robot_config.backend == "physical"
            else ()
        )

        self._drivers: tuple[FeeTechDriver, ...] = ()
        self._closed = False
        self.arm_assemblies: tuple[ArmAssembly, ...] = ()
        try:
            self.arm_assemblies = self._build_arm_assemblies()
            devices = self._devices_from_assemblies(self.arm_assemblies)
            self._joint_device = CompositeJointDevice(devices)
            self.ids = self._joint_device.ids
            self.joint_names = self._joint_device.joint_names
            self.dof = len(self.ids)

            first_end_effector = self.arm_assemblies[0].config.end_effector
            self.gripper_id: Optional[int] = None
            self.gripper_index: Optional[int] = None
            if isinstance(first_end_effector, FeeTechGripperConfig):
                self.gripper_id = first_end_effector.joint_id
                self.gripper_index = sum(
                    len(assembly.config.arm.joint_ids) for assembly in self.arm_assemblies
                )
        except BaseException as construction_error:
            cleanup_error = self._close_resources(
                self._devices_from_assemblies(self.arm_assemblies),
                self._drivers,
            )
            self._drivers = ()
            if cleanup_error is not None:
                raise construction_error from cleanup_error
            raise

    @property
    def arm(self) -> JointDevice:
        return self.arm_assemblies[0].arm

    @property
    def end_effector(self) -> Optional[JointDevice]:
        return self.arm_assemblies[0].end_effector

    def _build_arm_assemblies(self) -> tuple[ArmAssembly, ...]:
        if self.robot_config.backend == "mock":
            return self._build_mock_assemblies()

        drivers = self._create_feetech_drivers()
        self._drivers = tuple(drivers.values())
        built: list[ArmAssembly] = []
        pending_arm: Optional[JointDevice] = None
        pending_end_effector: Optional[JointDevice] = None
        try:
            for assembly_config, metadata in zip(
                self.robot_config.arm_assemblies,
                self._physical_build_metadata,
            ):
                arm_config = assembly_config.arm
                if arm_config.port is None:
                    raise RuntimeError(
                        f"physical arm '{assembly_config.name}' requires a serial port"
                    )

                pending_arm = FeeTechArm(
                    arm_config,
                    driver=drivers[arm_config.port],
                    pin_model=metadata.pin_model,
                    position_limits=metadata.position_limits,
                )
                pending_end_effector = self._build_physical_end_effector(assembly_config, drivers)
                built.append(
                    ArmAssembly(
                        assembly_config,
                        pending_arm,
                        pending_end_effector,
                    )
                )
                pending_arm = None
                pending_end_effector = None
        except BaseException as construction_error:
            pending_devices = tuple(
                device
                for device in (pending_end_effector, pending_arm)
                if device is not None
            )
            cleanup_error = self._close_resources(
                pending_devices + self._devices_from_assemblies(tuple(built)),
                self._drivers,
            )
            self._drivers = ()
            if cleanup_error is not None:
                raise construction_error from cleanup_error
            raise
        return tuple(built)

    def _build_mock_assemblies(self) -> tuple[ArmAssembly, ...]:
        built: list[ArmAssembly] = []
        try:
            for config in self.robot_config.arm_assemblies:
                built.append(self._build_mock_assembly(config))
        except BaseException as construction_error:
            cleanup_error = self._close_resources(
                self._devices_from_assemblies(tuple(built)),
                (),
            )
            if cleanup_error is not None:
                raise construction_error from cleanup_error
            raise
        return tuple(built)

    def _preflight_physical_assemblies(
        self,
    ) -> tuple[_PhysicalAssemblyMetadata, ...]:
        metadata: list[_PhysicalAssemblyMetadata] = []
        for assembly in self.robot_config.arm_assemblies:
            arm_config = assembly.arm
            normalize_feetech_servo_ids(
                arm_config.joint_ids,
                field_name=f"arm '{assembly.name}' servo IDs",
            )
            validate_feetech_servo_models(
                arm_config.servo_models,
                context=f"arm '{assembly.name}'",
            )

            end_effector = assembly.end_effector
            if isinstance(end_effector, FeeTechGripperConfig):
                normalize_feetech_servo_ids(
                    end_effector.joint_ids,
                    field_name=f"gripper on arm '{assembly.name}' servo IDs",
                )
                validate_feetech_servo_models(
                    (end_effector.servo_model,),
                    context=f"gripper on arm '{assembly.name}'",
                )
            elif isinstance(end_effector, O6DexterousHandConfig):
                raise RuntimeError("O6 dexterous hand physical backend is not implemented")

            lower, upper = self._get_joint_position_limits(arm_config.joint_names)
            position_limits = (tuple(lower), tuple(upper))

            pin_model = None
            if arm_config.enable_gravity_compensation:
                pin_model = self._get_pin_model_for_joint_names(arm_config.joint_names)
                if pin_model.nv != len(arm_config.joint_ids):
                    raise ValueError(
                        f"Pinocchio model nv ({pin_model.nv}) must match arm joint count "
                        f"({len(arm_config.joint_ids)})"
                    )

            metadata.append(
                _PhysicalAssemblyMetadata(
                    position_limits=position_limits,
                    pin_model=pin_model,
                )
            )
        return tuple(metadata)

    @staticmethod
    def _devices_from_assemblies(
        assemblies: Sequence[ArmAssembly],
    ) -> tuple[JointDevice, ...]:
        return tuple(assembly.arm for assembly in assemblies) + tuple(
            assembly.end_effector
            for assembly in assemblies
            if assembly.end_effector is not None
        )

    @staticmethod
    def _close_resources(
        devices: Sequence[JointDevice],
        drivers: Sequence[FeeTechDriver],
    ) -> Optional[BaseException]:
        closed: set[int] = set()
        first_error: Optional[BaseException] = None
        for resource in tuple(devices) + tuple(drivers):
            resource_id = id(resource)
            if resource_id in closed:
                continue
            try:
                resource.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                closed.add(resource_id)
        return first_error

    def _preflight_urdf_joint_mappings(self) -> None:
        if self._urdf_model_path is None:
            dual_mock_without_urdf = (
                self.robot_config.robot_type == "ace_follower_dual"
                and self.robot_config.backend == "mock"
                and self.robot_config.runtime == "standalone"
            )
            if dual_mock_without_urdf:
                return
            raise RuntimeError(
                f"URDF model is required to validate joint mappings for "
                f"'{self.robot_config.robot_type}'"
            )
        for assembly in self.robot_config.arm_assemblies:
            end_effector_names: tuple[str, ...] = ()
            if isinstance(assembly.end_effector, FeeTechGripperConfig):
                end_effector_names = assembly.end_effector.joint_names
            self._validate_urdf_joint_mapping(
                assembly.arm.joint_names,
                end_effector_names,
            )

    def _create_feetech_drivers(self) -> dict[str, FeeTechDriver]:
        ids_by_port: dict[str, list[int]] = {}
        for assembly in self.robot_config.arm_assemblies:
            arm = assembly.arm
            if arm.port is None:
                raise RuntimeError(f"physical arm '{assembly.name}' requires a serial port")
            ids_by_port.setdefault(arm.port, []).extend(arm.joint_ids)
            end_effector = assembly.end_effector
            if isinstance(end_effector, FeeTechGripperConfig):
                if end_effector.port is None:
                    raise RuntimeError(
                        f"physical gripper on arm '{assembly.name}' requires a serial port"
                    )
                ids_by_port.setdefault(end_effector.port, []).append(end_effector.joint_id)
        drivers: dict[str, FeeTechDriver] = {}
        try:
            for port, joint_ids in ids_by_port.items():
                drivers[port] = FeeTechDriver(joint_ids, port)
        except BaseException as construction_error:
            cleanup_error = self._close_resources((), tuple(drivers.values()))
            if cleanup_error is not None:
                raise construction_error from cleanup_error
            raise
        return drivers

    @staticmethod
    def _build_physical_end_effector(
        assembly: ArmAssemblyConfig,
        drivers: dict[str, FeeTechDriver],
    ) -> Optional[JointDevice]:
        config = assembly.end_effector
        if config is None:
            return None
        driver = None
        if isinstance(config, FeeTechGripperConfig):
            if config.port is None:
                raise RuntimeError("physical FEETECH gripper requires a serial port")
            driver = drivers[config.port]
        return create_end_effector(config, backend="physical", driver=driver)

    def _build_mock_assembly(self, assembly: ArmAssemblyConfig) -> ArmAssembly:
        arm = MockJointDevice(
            self._default_mock_arm_joints(assembly.arm),
            wrap_public_positions=assembly.arm.wrap_public_positions,
        )
        try:
            end_effector = (
                None
                if assembly.end_effector is None
                else create_end_effector(assembly.end_effector, backend="mock")
            )
        except BaseException as construction_error:
            cleanup_error = self._close_resources((arm,), ())
            if cleanup_error is not None:
                raise construction_error from cleanup_error
            raise
        return ArmAssembly(assembly, arm, end_effector)

    def _default_mock_arm_joints(self, config: ArmConfig) -> tuple[MockJointConfig, ...]:
        if config.mock_joints:
            return config.mock_joints
        lower_limits = [-2.0 * math.pi] * len(config.joint_ids)
        upper_limits = [2.0 * math.pi] * len(config.joint_ids)
        if self._urdf_model_path is not None:
            lower_limits, upper_limits = self._get_joint_position_limits(config.joint_names)
        return tuple(
            MockJointConfig(
                name,
                joint_id,
                float(
                    home
                    if lower <= home <= upper
                    else unwrap_near(home, 0.5 * (lower + upper))
                ),
                lower,
                upper,
                10.0,
            )
            for name, joint_id, home, lower, upper in zip(
                config.joint_names,
                config.joint_ids,
                config.home_poses,
                lower_limits,
                upper_limits,
            )
        )

    def get_robot_state(self) -> JointDeviceState:
        return self._joint_device.get_state()

    def act(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._joint_device.act()

    def set_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ) -> None:
        self._joint_device.set_position(
            positions,
            ids=ids,
            velocities=velocities,
            accelerations=accelerations,
            torque=torque,
        )

    def set_torque_enable(
        self,
        enable: TorqueEnable,
        ids: Optional[Sequence[int]] = None,
    ) -> None:
        self._joint_device.set_torque_enable(enable, ids=ids)

    def close(self) -> None:
        if self._closed:
            return
        first_error = self._close_resources(
            self._devices_from_assemblies(self.arm_assemblies),
            self._drivers,
        )
        self._drivers = ()
        self._closed = True
        if first_error is not None:
            raise first_error


__all__ = ["ArmAssembly", "JointRobot"]
