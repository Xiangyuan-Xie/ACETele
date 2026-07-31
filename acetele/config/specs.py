"""Immutable hardware topology and control specifications.

These types validate static facts only and never open hardware. Joint names identify
kinematics; servo IDs identify devices on one bus. Keeping those namespaces separate is
what permits bus addresses to change without changing URDF or ROS interfaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from acetele.core import Backend


class BusType(str, Enum):
    """Supported protocol families and physical transport semantics."""

    FEETECH_PACKET = "feetech_packet"
    FEETECH_MODBUS_RTU = "feetech_modbus_rtu"
    FASHIONSTAR_RS485 = "fashionstar_rs485"
    LINKER_HAND_RS485 = "linker_hand_rs485"


class DirectionControl(str, Enum):
    """How an RS485 adapter switches between transmit and receive."""

    AUTO = "auto"
    RTS = "rts"


def _name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class BusSpec:
    """Static configuration for one physical serial port and its bus actor."""

    name: str
    type: BusType
    port: str
    baudrate: int
    cycle_hz: float
    direction_control: DirectionControl = DirectionControl.AUTO
    physical_layer: Optional[str] = None
    family: Optional[str] = None
    max_utilization: float = 0.70
    external_estop: bool = False
    allow_unverified_identity: bool = False

    def __post_init__(self) -> None:
        _name(self.name, field_name="bus name")
        if not isinstance(self.type, BusType):
            raise ValueError("bus type must be a BusType")
        _name(self.port, field_name=f"bus '{self.name}' port")
        if type(self.baudrate) is not int or self.baudrate <= 0:
            raise ValueError(f"bus '{self.name}' baudrate must be a positive integer")
        if not math.isfinite(self.cycle_hz) or self.cycle_hz <= 0.0:
            raise ValueError(f"bus '{self.name}' cycle_hz must be finite and positive")
        if not isinstance(self.direction_control, DirectionControl):
            raise ValueError(f"bus '{self.name}' direction_control must be a DirectionControl")
        if not math.isfinite(self.max_utilization) or not 0.0 < self.max_utilization <= 1.0:
            raise ValueError(f"bus '{self.name}' max_utilization must be in (0, 1]")
        if type(self.external_estop) is not bool:
            raise ValueError(f"bus '{self.name}' external_estop must be a boolean")
        if type(self.allow_unverified_identity) is not bool:
            raise ValueError(
                f"bus '{self.name}' allow_unverified_identity must be a boolean"
            )
        if self.type == BusType.FEETECH_PACKET:
            if self.physical_layer not in ("ttl", "rs485"):
                raise ValueError("feetech_packet buses require physical_layer='ttl' or 'rs485'")
            if self.family not in ("hls", "sms"):
                raise ValueError("feetech_packet buses require family='hls' or 'sms'")
            if self.family == "hls" and self.physical_layer != "ttl":
                raise ValueError("FEETECH HLS requires the TTL physical layer")
            if self.family == "sms" and self.physical_layer != "rs485":
                raise ValueError("FEETECH SMS requires the RS485 physical layer")
        elif self.physical_layer not in (None, "rs485"):
            raise ValueError(f"bus '{self.name}' uses the RS485 physical layer")


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
class PositionControlTuning:
    """Numerical policy for quasi-static adaptive position compensation.

    These values form one coherent control-law specification. Keeping them together
    makes alternate tuning explicit without exposing unrelated knobs in the TOML
    schema or scattering magic numbers through the implementation.
    """

    adaptive_deadband_rad: float = 0.02
    adaptation_rate_per_s: float = 8.0
    offset_filter_bandwidth_per_s: float = 4.0
    maximum_adaptive_offset_rad: float = 0.10
    stable_time_s: float = 0.20
    target_stable_threshold_rad: float = 0.008
    target_reset_threshold_rad: float = 0.05
    target_direction_threshold_rad: float = 0.002
    velocity_threshold_rad_s: float = 0.05
    minimum_dt_s: float = 0.001
    maximum_dt_s: float = 0.05

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"position control tuning {name} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(
                    f"position control tuning {name} must be finite and positive"
                )
            object.__setattr__(self, name, normalized)
        if self.minimum_dt_s > self.maximum_dt_s:
            raise ValueError("position control minimum_dt_s cannot exceed maximum_dt_s")
        if self.target_stable_threshold_rad >= self.target_reset_threshold_rad:
            raise ValueError(
                "position control target_stable_threshold_rad must be smaller than "
                "target_reset_threshold_rad"
            )
        if self.target_direction_threshold_rad >= self.target_reset_threshold_rad:
            raise ValueError(
                "position control target_direction_threshold_rad must be smaller than "
                "target_reset_threshold_rad"
            )


@dataclass(frozen=True)
class ControlSpec:
    """Opt-in model-based position conditioning for one arm."""

    adaptive_position: bool = False
    gravity_position: bool = False
    gravity_compliance_rad_per_nm: Optional[tuple[float, ...]] = None
    position_tuning: PositionControlTuning = field(
        default_factory=PositionControlTuning
    )

    def __post_init__(self) -> None:
        if type(self.adaptive_position) is not bool or type(self.gravity_position) is not bool:
            raise ValueError("control feature switches must be booleans")
        if not isinstance(self.position_tuning, PositionControlTuning):
            raise ValueError("position_tuning must be a PositionControlTuning")
        compliance = self.gravity_compliance_rad_per_nm
        if compliance is not None:
            if isinstance(compliance, (str, bytes)):
                raise ValueError("gravity compliance must be a numeric sequence")
            compliance = tuple(float(value) for value in compliance)
            if not compliance or any(
                not math.isfinite(value) or value <= 0.0 for value in compliance
            ):
                raise ValueError("gravity compliance values must be finite and positive")
            object.__setattr__(self, "gravity_compliance_rad_per_nm", compliance)
        if self.gravity_position and compliance is None:
            raise ValueError(
                "gravity position compensation requires per-joint compliance calibration"
            )


@dataclass(frozen=True)
class ParallelGripperSpec:
    """Single-servo gripper exposed as a normalized joint in ``[0, 1]``."""

    bus: str
    joint: JointSpec
    travel_range_rad: float

    def __post_init__(self) -> None:
        _name(self.bus, field_name="parallel gripper bus")
        if (
            not math.isfinite(self.travel_range_rad)
            or not 0.0 < self.travel_range_rad < math.pi
        ):
            raise ValueError("parallel gripper travel_range_rad must be in (0, pi)")


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
        compliance = self.control.gravity_compliance_rad_per_nm
        if compliance is not None and len(compliance) != len(joints):
            raise ValueError(
                f"arm '{self.name}' gravity compliance must match its joint count"
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
    "BusSpec",
    "BusType",
    "ControlSpec",
    "DexterousHandSpec",
    "DirectionControl",
    "EndEffectorSpec",
    "JointSpec",
    "ParallelGripperSpec",
    "PositionControlTuning",
    "RobotSpec",
]
