"""Immutable robot, joint, and end-effector topology specifications.

These types validate static facts only and never open hardware. Joint names identify
kinematics; servo IDs identify devices on one bus. Keeping those namespaces separate is
what permits bus addresses to change without changing URDF or ROS interfaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

from acetele.specification.backend import Backend
from acetele.specification.bus import BusSpec, BusType
from acetele.specification.control import ControlSpec


def _name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class JointSpec:
    """Bind one canonical joint name to one vendor device and direction."""

    name: str
    servo_id: int
    servo_model: str
    direction: int
    home_position_rad: float
    expected_model_number: Optional[int] = None
    firmware_version: Optional[int] = None

    def __post_init__(self) -> None:
        _name(self.name, field_name="joint name")
        _name(self.servo_model, field_name=f"joint '{self.name}' servo_model")
        if type(self.servo_id) is not int or not 0 <= self.servo_id <= 254:
            raise ValueError(f"joint '{self.name}' servo_id must be an integer in [0, 254]")
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ValueError(f"joint '{self.name}' direction must be -1 or 1")
        if not math.isfinite(self.home_position_rad):
            raise ValueError(f"joint '{self.name}' home_position_rad must be finite")
        if self.expected_model_number is not None and (
            type(self.expected_model_number) is not int
            or not 0 <= self.expected_model_number <= 0xFFFF
        ):
            raise ValueError(
                f"joint '{self.name}' expected_model_number must be a uint16 or omitted"
            )
        if self.firmware_version is not None and (
            type(self.firmware_version) is not int or self.firmware_version <= 0
        ):
            raise ValueError(
                f"joint '{self.name}' firmware_version must be a positive integer or omitted"
            )


@dataclass(frozen=True)
class ParallelGripperSpec:
    """Single-servo gripper exposed as a normalized joint in ``[0, 1]``."""

    bus: str
    joint: JointSpec
    travel_range_rad: float
    kinematic_joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _name(self.bus, field_name="parallel gripper bus")
        if (
            not math.isfinite(self.travel_range_rad)
            or not 0.0 < self.travel_range_rad < math.pi
        ):
            raise ValueError("parallel gripper travel_range_rad must be in (0, pi)")
        kinematic_joint_names = tuple(self.kinematic_joint_names)
        if any(not isinstance(name, str) or not name.strip() for name in kinematic_joint_names):
            raise ValueError("parallel gripper kinematic joint names must be non-empty strings")
        if len(set(kinematic_joint_names)) != len(kinematic_joint_names):
            raise ValueError("parallel gripper kinematic joint names must be unique")
        object.__setattr__(self, "kinematic_joint_names", kinematic_joint_names)


@dataclass(frozen=True)
class DexterousHandSpec:
    """Vendor hand addressed as one bus device with model-defined channels."""

    bus: str
    vendor: str
    model: str
    side: str
    slave_id: int

    def __post_init__(self) -> None:
        _name(self.bus, field_name="dexterous hand bus")
        _name(self.vendor, field_name="dexterous hand vendor")
        _name(self.model, field_name="dexterous hand model")
        if self.side not in ("left", "right"):
            raise ValueError("dexterous hand side must be 'left' or 'right'")
        if type(self.slave_id) is not int or not 1 <= self.slave_id <= 247:
            raise ValueError("dexterous hand slave_id must be an integer in [1, 247]")


EndEffectorSpec = Union[ParallelGripperSpec, DexterousHandSpec]


@dataclass(frozen=True)
class ArmSpec:
    """Ordered arm joints plus an optional end effector."""

    name: str
    bus: str
    joints: tuple[JointSpec, ...]
    control: ControlSpec = ControlSpec()
    end_effector: Optional[EndEffectorSpec] = None
    tool_frame: Optional[str] = None

    def __post_init__(self) -> None:
        _name(self.name, field_name="arm name")
        _name(self.bus, field_name=f"arm '{self.name}' bus")
        joints = tuple(self.joints)
        if not joints:
            raise ValueError(f"arm '{self.name}' requires at least one joint")
        if len({joint.name for joint in joints}) != len(joints):
            raise ValueError(f"arm '{self.name}' joint names must be unique")
        if len({joint.servo_id for joint in joints}) != len(joints):
            raise ValueError(f"arm '{self.name}' servo IDs must be unique")
        if self.tool_frame is not None:
            _name(self.tool_frame, field_name=f"arm '{self.name}' tool_frame")
        rest_posture = self.control.rest_posture_rad
        if rest_posture is not None and len(rest_posture) != len(joints):
            raise ValueError(
                f"arm '{self.name}' rest posture must match its joint count"
            )
        object.__setattr__(self, "joints", joints)


@dataclass(frozen=True)
class RobotSpec:
    """Complete immutable topology consumed by ``RobotRuntime``.

    Global names and per-bus addresses are made unique here so runtime routing cannot
    become ambiguous after hardware is connected.
    """

    model: str
    buses: tuple[BusSpec, ...]
    arms: tuple[ArmSpec, ...]
    backend: Backend = Backend.PHYSICAL
    urdf_path: Optional[str] = None

    def __post_init__(self) -> None:
        _name(self.model, field_name="robot model")
        if not isinstance(self.backend, Backend):
            raise ValueError("robot backend must be a Backend")
        if self.urdf_path is not None:
            _name(self.urdf_path, field_name="robot urdf_path")
        buses = tuple(self.buses)
        arms = tuple(self.arms)
        if not buses or not arms:
            raise ValueError("robot spec requires at least one bus and one arm")
        bus_names = tuple(bus.name for bus in buses)
        bus_ports = tuple(bus.port for bus in buses)
        arm_names = tuple(arm.name for arm in arms)
        if len(set(bus_names)) != len(bus_names):
            raise ValueError("robot bus names must be unique")
        if len(set(bus_ports)) != len(bus_ports):
            raise ValueError(
                "robot bus ports must be unique; devices on one physical port "
                "must share one bus"
            )
        if len(set(arm_names)) != len(arm_names):
            raise ValueError("robot arm names must be unique")
        known_buses = set(bus_names)
        bus_by_name = {bus.name: bus for bus in buses}
        # Joint names form the public kinematic namespace, while device IDs need only
        # be unique on their physical bus. Validate both domains independently.
        used_ids: dict[str, set[int]] = {name: set() for name in bus_names}
        used_joint_names: set[str] = set()
        for arm in arms:
            if arm.bus not in known_buses:
                raise ValueError(f"arm '{arm.name}' references unknown bus '{arm.bus}'")
            if arm.end_effector is not None and arm.end_effector.bus not in known_buses:
                raise ValueError(
                    f"arm '{arm.name}' end effector references unknown bus "
                    f"'{arm.end_effector.bus}'"
                )
            arm_bus = bus_by_name[arm.bus]
            if arm_bus.type == BusType.LINKER_HAND_RS485:
                raise ValueError(f"arm '{arm.name}' cannot use a Linker Hand bus")
            for joint in arm.joints:
                if joint.name in used_joint_names:
                    raise ValueError(f"robot contains duplicate joint name '{joint.name}'")
                used_joint_names.add(joint.name)
                self._validate_servo_id(arm_bus, joint.servo_id, context=joint.name)
                if joint.servo_id in used_ids[arm.bus]:
                    raise ValueError(
                        f"bus '{arm.bus}' contains duplicate device ID {joint.servo_id}"
                    )
                used_ids[arm.bus].add(joint.servo_id)
            end_effector = arm.end_effector
            if isinstance(end_effector, ParallelGripperSpec):
                gripper_bus = bus_by_name[end_effector.bus]
                if gripper_bus.type == BusType.LINKER_HAND_RS485:
                    raise ValueError("parallel gripper cannot use a Linker Hand bus")
                joint = end_effector.joint
                if joint.name in used_joint_names:
                    raise ValueError(f"robot contains duplicate joint name '{joint.name}'")
                used_joint_names.add(joint.name)
                self._validate_servo_id(gripper_bus, joint.servo_id, context=joint.name)
                if joint.servo_id in used_ids[end_effector.bus]:
                    raise ValueError(
                        f"bus '{end_effector.bus}' contains duplicate device ID "
                        f"{joint.servo_id}"
                    )
                used_ids[end_effector.bus].add(joint.servo_id)
            elif isinstance(end_effector, DexterousHandSpec):
                hand_bus = bus_by_name[end_effector.bus]
                if hand_bus.type != BusType.LINKER_HAND_RS485:
                    raise ValueError("Linker dexterous hands require a linker_hand_rs485 bus")
                if end_effector.vendor.lower() != "linker":
                    raise ValueError(
                        f"unsupported dexterous hand vendor '{end_effector.vendor}'"
                    )
                if end_effector.slave_id in used_ids[end_effector.bus]:
                    raise ValueError(
                        f"bus '{end_effector.bus}' contains duplicate device ID "
                        f"{end_effector.slave_id}"
                    )
                used_ids[end_effector.bus].add(end_effector.slave_id)
        object.__setattr__(self, "buses", buses)
        object.__setattr__(self, "arms", arms)

    @staticmethod
    def _validate_servo_id(bus: BusSpec, servo_id: int, *, context: str) -> None:
        """Apply protocol address ranges before a transport can be opened."""

        if bus.type == BusType.FEETECH_PACKET:
            valid = 0 <= servo_id <= 252
            expected = "[0, 252]"
        elif bus.type == BusType.FASHIONSTAR_RS485:
            valid = 0 <= servo_id <= 254
            expected = "[0, 254]"
        else:
            valid = 1 <= servo_id <= 247
            expected = "[1, 247]"
        if not valid:
            raise ValueError(
                f"{context} servo_id {servo_id} is invalid for bus '{bus.name}'; "
                f"expected {expected}"
            )


__all__ = [
    "ArmSpec",
    "DexterousHandSpec",
    "EndEffectorSpec",
    "JointSpec",
    "ParallelGripperSpec",
    "RobotSpec",
]
