from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Optional, Protocol, Tuple

from acetele.utils.joint import (
    normalize_joint_id,
    normalize_joint_ids,
    normalize_joint_sign,
    normalize_joint_signs,
)


def _immutable_tuple(values: Any, *, field_name: str) -> tuple:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence, not a string")
    try:
        return tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence") from exc


def _normalize_finite_real(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite real number")
    return normalized


def _normalize_finite_real_sequence(
    values: Any,
    *,
    field_name: str,
) -> Tuple[float, ...]:
    items = _immutable_tuple(values, field_name=field_name)
    return tuple(
        _normalize_finite_real(value, field_name=f"{field_name}[{index}]")
        for index, value in enumerate(items)
    )


@dataclass(frozen=True)
class MockJointConfig:
    name: str
    joint_id: int
    initial_position: float
    lower_limit: float
    upper_limit: float
    max_velocity: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("mock joint name must be a non-empty string")
        object.__setattr__(
            self,
            "joint_id",
            normalize_joint_id(self.joint_id, field_name=f"mock joint '{self.name}' id"),
        )
        if self.joint_id < 0:
            raise ValueError(f"mock joint '{self.name}' id must be non-negative")
        for field_name in (
            "initial_position",
            "lower_limit",
            "upper_limit",
            "max_velocity",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_finite_real(
                    getattr(self, field_name),
                    field_name=f"mock joint '{self.name}' {field_name}",
                ),
            )
        if self.lower_limit > self.upper_limit:
            raise ValueError(f"mock joint '{self.name}' lower limit must not exceed upper limit")
        if not self.lower_limit <= self.initial_position <= self.upper_limit:
            raise ValueError(f"mock joint '{self.name}' initial position is outside its limits")
        if self.max_velocity <= 0.0:
            raise ValueError(f"mock joint '{self.name}' max velocity must be positive")


@dataclass(frozen=True)
class ArmConfig:
    port: Optional[str]
    joint_ids: Tuple[int, ...]
    joint_names: Tuple[str, ...]
    joint_signs: Tuple[int, ...]
    home_poses: Tuple[float, ...]
    servo_models: Tuple[str, ...]
    enable_gravity_compensation: bool = False
    enable_adaptive_compensation: bool = False
    control_period: float = 0.004
    mock_joints: Tuple[MockJointConfig, ...] = ()
    wrap_public_positions: bool = True

    def __post_init__(self) -> None:
        if self.port is not None and (
            not isinstance(self.port, str) or not self.port.strip()
        ):
            raise ValueError("arm port must be a non-empty string or None")
        object.__setattr__(
            self,
            "joint_names",
            _immutable_tuple(self.joint_names, field_name="arm joint_names"),
        )
        object.__setattr__(
            self,
            "home_poses",
            _normalize_finite_real_sequence(
                self.home_poses,
                field_name="arm home_poses",
            ),
        )
        object.__setattr__(
            self,
            "servo_models",
            _immutable_tuple(self.servo_models, field_name="arm servo_models"),
        )
        object.__setattr__(
            self,
            "mock_joints",
            _immutable_tuple(self.mock_joints, field_name="arm mock_joints"),
        )
        object.__setattr__(
            self,
            "control_period",
            _normalize_finite_real(
                self.control_period,
                field_name="arm control_period",
            ),
        )
        for field_name in (
            "enable_gravity_compensation",
            "enable_adaptive_compensation",
            "wrap_public_positions",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"arm {field_name} must be a boolean")
        object.__setattr__(
            self,
            "joint_ids",
            normalize_joint_ids(self.joint_ids, field_name="arm joint_ids"),
        )
        object.__setattr__(
            self,
            "joint_signs",
            normalize_joint_signs(self.joint_signs, field_name="arm joint_signs"),
        )
        count = len(self.joint_ids)
        if count == 0:
            raise ValueError("arm must contain at least one joint")
        for field_name, values in (
            ("joint_names", self.joint_names),
            ("joint_signs", self.joint_signs),
            ("home_poses", self.home_poses),
        ):
            if len(values) != count:
                raise ValueError(f"arm {field_name} must match joint_ids")
        if self.servo_models and len(self.servo_models) != count:
            raise ValueError("arm servo_models must match joint_ids")
        if self.mock_joints:
            if len(self.mock_joints) != count or not all(
                isinstance(joint, MockJointConfig) for joint in self.mock_joints
            ):
                raise ValueError("arm mock_joints must contain one MockJointConfig per joint")
            if tuple(joint.name for joint in self.mock_joints) != self.joint_names:
                raise ValueError("arm mock_joints names must match joint_names")
            if tuple(joint.joint_id for joint in self.mock_joints) != self.joint_ids:
                raise ValueError("arm mock_joints ids must match joint_ids")
        if len(set(self.joint_ids)) != count or len(set(self.joint_names)) != count:
            raise ValueError("arm joint ids and names must be unique")
        if any(joint_id < 0 for joint_id in self.joint_ids):
            raise ValueError("arm joint ids must be non-negative")
        if any(not isinstance(name, str) or not name.strip() for name in self.joint_names):
            raise ValueError("arm joint names must not be empty")
        if any(not isinstance(model, str) or not model.strip() for model in self.servo_models):
            raise ValueError("arm servo models must be non-empty strings")
        if self.control_period <= 0.0:
            raise ValueError("arm control period must be positive")


class EndEffectorConfig(Protocol):
    @property
    def joint_names(self) -> Tuple[str, ...]: ...

    @property
    def joint_ids(self) -> Tuple[int, ...]: ...


@dataclass(frozen=True)
class FeeTechGripperConfig:
    port: Optional[str]
    joint_id: int
    joint_name: str
    joint_sign: int
    home_pose: float
    servo_model: str
    travel_range_rad: float

    def __post_init__(self) -> None:
        if self.port is not None and (
            not isinstance(self.port, str) or not self.port.strip()
        ):
            raise ValueError("gripper port must be a non-empty string or None")
        object.__setattr__(
            self,
            "joint_id",
            normalize_joint_id(self.joint_id, field_name="gripper joint_id"),
        )
        object.__setattr__(
            self,
            "joint_sign",
            normalize_joint_sign(self.joint_sign, field_name="gripper joint sign"),
        )
        object.__setattr__(
            self,
            "home_pose",
            _normalize_finite_real(
                self.home_pose,
                field_name="gripper home_pose",
            ),
        )
        object.__setattr__(
            self,
            "travel_range_rad",
            _normalize_finite_real(
                self.travel_range_rad,
                field_name="gripper travel_range_rad",
            ),
        )
        if (
            self.joint_id < 0
            or not isinstance(self.joint_name, str)
            or not self.joint_name.strip()
        ):
            raise ValueError("gripper joint id and name must be valid")
        if not 0.0 <= self.home_pose <= 1.0:
            raise ValueError("gripper home pose must be between 0.0 and 1.0")
        if not isinstance(self.servo_model, str) or not self.servo_model.strip():
            raise ValueError("gripper servo model must not be empty")
        if self.travel_range_rad <= 0.0:
            raise ValueError("gripper travel_range_rad must be positive")
        if not 1 <= self.travel_range_counts <= 2047:
            raise ValueError(
                "gripper travel_range_rad must encode to between 1 and 2047 "
                "FEETECH position counts"
            )

    @property
    def joint_names(self) -> Tuple[str, ...]:
        return (self.joint_name,)

    @property
    def joint_ids(self) -> Tuple[int, ...]:
        return (self.joint_id,)

    @property
    def travel_range_counts(self) -> int:
        return int(round(self.travel_range_rad * 2048.0 / math.pi))


@dataclass(frozen=True)
class O6DexterousHandConfig:
    side: str
    joint_ids: Tuple[int, ...]
    mock_joints: Tuple[MockJointConfig, ...] = ()

    _ACTIVE_SUFFIXES = (
        "thumb_cmc_yaw",
        "thumb_cmc_pitch",
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
    )

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError("O6 dexterous hand side must be 'left' or 'right'")
        object.__setattr__(
            self,
            "joint_ids",
            normalize_joint_ids(
                self.joint_ids,
                field_name=f"{self.side} O6 dexterous hand joint_ids",
            ),
        )
        object.__setattr__(
            self,
            "mock_joints",
            _immutable_tuple(self.mock_joints, field_name=f"{self.side} O6 mock_joints"),
        )
        if len(self.joint_ids) != len(self._ACTIVE_SUFFIXES):
            raise ValueError(
                f"{self.side} O6 dexterous hand requires "
                f"{len(self._ACTIVE_SUFFIXES)} joint IDs"
            )
        if len(set(self.joint_ids)) != len(self.joint_ids):
            raise ValueError(f"{self.side} O6 dexterous hand joint IDs must be unique")
        if any(joint_id < 0 for joint_id in self.joint_ids):
            raise ValueError(f"{self.side} O6 dexterous hand joint IDs must be non-negative")
        if self.mock_joints:
            if len(self.mock_joints) != len(self.joint_ids) or not all(
                isinstance(joint, MockJointConfig) for joint in self.mock_joints
            ):
                raise ValueError(f"{self.side} O6 mock_joints must contain one MockJointConfig per joint")
            if tuple(joint.name for joint in self.mock_joints) != self.joint_names:
                raise ValueError(f"{self.side} O6 mock_joints names must match the O6 joint names")
            if tuple(joint.joint_id for joint in self.mock_joints) != self.joint_ids:
                raise ValueError(f"{self.side} O6 mock_joints ids must match joint_ids")

    @property
    def joint_names(self) -> Tuple[str, ...]:
        prefix = "lh" if self.side == "left" else "rh"
        return tuple(f"{prefix}_{suffix}" for suffix in self._ACTIVE_SUFFIXES)


@dataclass(frozen=True)
class ArmAssemblyConfig:
    name: str
    arm: ArmConfig
    end_effector: Optional[EndEffectorConfig] = None
    _end_effector_joint_names: Tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
        default=(),
    )
    _end_effector_joint_ids: Tuple[int, ...] = field(
        init=False,
        repr=False,
        compare=False,
        default=(),
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("arm assembly name must not be empty")
        if not isinstance(self.arm, ArmConfig):
            raise ValueError("arm assembly arm must be an ArmConfig")
        if self.end_effector is not None:
            dataclass_params = getattr(
                type(self.end_effector),
                "__dataclass_params__",
                None,
            )
            if dataclass_params is None or not dataclass_params.frozen:
                raise ValueError("arm assembly end effector config must be a frozen dataclass")
            try:
                joint_names = self.end_effector.joint_names
                joint_ids = self.end_effector.joint_ids
            except (AttributeError, TypeError) as exc:
                raise ValueError(
                    "arm assembly end effector config must define joint_names and joint_ids"
                ) from exc
            if not isinstance(joint_names, tuple) or not isinstance(joint_ids, tuple):
                raise ValueError(
                    "arm assembly end effector joint_names and joint_ids must be immutable tuples"
                )
            if not joint_names or len(joint_names) != len(joint_ids):
                raise ValueError(
                    "arm assembly end effector joint_names and joint_ids must be non-empty and equal in length"
                )
            if any(not isinstance(name, str) or not name.strip() for name in joint_names):
                raise ValueError("arm assembly end effector joint names must be non-empty strings")
            normalized_ids = normalize_joint_ids(
                joint_ids,
                field_name="arm assembly end effector joint_ids",
            )
            if any(joint_id < 0 for joint_id in normalized_ids):
                raise ValueError("arm assembly end effector joint ids must be non-negative")
            if len(set(joint_names)) != len(joint_names) or len(set(normalized_ids)) != len(
                normalized_ids
            ):
                raise ValueError("arm assembly end effector joint names and ids must be unique")
            try:
                hash(self.end_effector)
            except TypeError as exc:
                raise ValueError(
                    "arm assembly end effector config must be deeply immutable and hashable"
                ) from exc
            object.__setattr__(self, "_end_effector_joint_names", tuple(joint_names))
            object.__setattr__(self, "_end_effector_joint_ids", normalized_ids)

    @property
    def end_effector_joint_names(self) -> Tuple[str, ...]:
        return self._end_effector_joint_names

    @property
    def end_effector_joint_ids(self) -> Tuple[int, ...]:
        return self._end_effector_joint_ids


@dataclass(frozen=True)
class RobotConfig:
    robot_type: str
    backend: str
    runtime: str
    arm_assemblies: Tuple[ArmAssemblyConfig, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arm_assemblies",
            _immutable_tuple(self.arm_assemblies, field_name="robot arm_assemblies"),
        )
        if any(
            not isinstance(assembly, ArmAssemblyConfig)
            for assembly in self.arm_assemblies
        ):
            raise ValueError("robot arm_assemblies must contain ArmAssemblyConfig values")
        expected_arm_count = {
            "ace_leader": 1,
            "ace_follower": 1,
            "ace_follower_dual": 2,
        }.get(self.robot_type)
        if expected_arm_count is None:
            raise ValueError(f"unsupported robot type '{self.robot_type}'")
        if len(self.arm_assemblies) != expected_arm_count:
            raise ValueError(f"{self.robot_type} requires exactly {expected_arm_count} arm assembly(s)")
        if self.backend not in ("physical", "mock"):
            raise ValueError(f"unsupported device backend '{self.backend}'")
        if self.runtime not in ("standalone", "ros2"):
            raise ValueError(f"unsupported runtime '{self.runtime}'")
        if len({assembly.name for assembly in self.arm_assemblies}) != len(self.arm_assemblies):
            raise ValueError("arm assembly names must be unique")
        if self.backend == "physical":
            for assembly in self.arm_assemblies:
                if not assembly.arm.port:
                    raise ValueError(f"physical arm '{assembly.name}' requires a port")
                if not assembly.arm.servo_models:
                    raise ValueError(f"physical arm '{assembly.name}' requires servo_models")
                end_effector = assembly.end_effector
                if isinstance(end_effector, FeeTechGripperConfig) and not end_effector.port:
                    raise ValueError(f"physical gripper on arm '{assembly.name}' requires a port")

        joint_names = self.joint_names
        joint_ids = self.joint_ids
        if len(set(joint_names)) != len(joint_names) or len(set(joint_ids)) != len(joint_ids):
            raise ValueError("robot joint names and ids must be globally unique")

    @property
    def joint_names(self) -> Tuple[str, ...]:
        arm_names = tuple(name for assembly in self.arm_assemblies for name in assembly.arm.joint_names)
        end_effector_names = tuple(
            name
            for assembly in self.arm_assemblies
            for name in assembly.end_effector_joint_names
        )
        return arm_names + end_effector_names

    @property
    def joint_ids(self) -> Tuple[int, ...]:
        arm_ids = tuple(joint_id for assembly in self.arm_assemblies for joint_id in assembly.arm.joint_ids)
        end_effector_ids = tuple(
            joint_id
            for assembly in self.arm_assemblies
            for joint_id in assembly.end_effector_joint_ids
        )
        return arm_ids + end_effector_ids


__all__ = [
    "ArmAssemblyConfig",
    "ArmConfig",
    "EndEffectorConfig",
    "FeeTechGripperConfig",
    "MockJointConfig",
    "O6DexterousHandConfig",
    "RobotConfig",
]
