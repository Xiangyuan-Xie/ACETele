"""Vendor-neutral, immutable contracts shared by every ACETele layer.

Hardware addresses and register units deliberately stop below this boundary. Runtime,
control, and ROS adapters exchange named joints in SI units (or an explicitly declared
normalized unit) so vendor protocols cannot leak into application logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np


class JointUnit(str, Enum):
    """Unit carried by every joint state and command."""

    RADIAN = "radian"
    NORMALIZED = "normalized"


def _names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    """Freeze and validate a canonical, non-ambiguous joint-name order."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of names")
    result = tuple(values)
    if not result or any(not isinstance(name, str) or not name.strip() for name in result):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must be unique")
    return result


def _readonly_vector(
    values: Sequence[float] | np.ndarray,
    *,
    field_name: str,
    length: Optional[int] = None,
    finite: bool = True,
) -> np.ndarray:
    """Copy numeric input into an immutable vector owned by the contract."""

    array = np.asarray(values, dtype=float).copy()
    if array.ndim != 1 or (length is not None and len(array) != length):
        expected = "one-dimensional" if length is None else f"one-dimensional with length {length}"
        raise ValueError(f"{field_name} must be {expected}")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    array.setflags(write=False)
    return array


def _immutable_sensor_value(value: Any) -> Any:
    """Recursively detach mutable sensor payloads from their producer."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _immutable_sensor_value(item) for key, item in value.items()}
        )
    if isinstance(value, np.ndarray):
        array = value.copy()
        array.setflags(write=False)
        return array
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        array = np.asarray(value)
        if not array.dtype.hasobject:
            array = array.copy()
            array.setflags(write=False)
            return array
        return tuple(_immutable_sensor_value(item) for item in value)
    return value


@dataclass(frozen=True)
class JointState:
    """One coherent, immutable joint sample in canonical name order."""

    names: tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray
    efforts: np.ndarray
    timestamp_ns: int
    sequence: int
    unit: JointUnit = JointUnit.RADIAN

    def __post_init__(self) -> None:
        names = _names(self.names, field_name="joint state names")
        count = len(names)
        object.__setattr__(self, "names", names)
        object.__setattr__(
            self,
            "positions",
            _readonly_vector(self.positions, field_name="joint state positions", length=count),
        )
        object.__setattr__(
            self,
            "velocities",
            _readonly_vector(self.velocities, field_name="joint state velocities", length=count),
        )
        object.__setattr__(
            self,
            "efforts",
            _readonly_vector(self.efforts, field_name="joint state efforts", length=count),
        )
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("joint state timestamp_ns must be a non-negative integer")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("joint state sequence must be a non-negative integer")
        if not isinstance(self.unit, JointUnit):
            raise ValueError("joint state unit must be a JointUnit")


@dataclass(frozen=True)
class EndEffectorPose:
    """Immutable Cartesian pose expressed in one named reference frame.

    The quaternion uses ROS-compatible ``(x, y, z, w)`` order. It is normalized and
    canonicalized so equivalent inputs with opposite signs have one stable
    representation at transport and diagnostic boundaries.
    """

    timestamp_ns: int
    frame_id: str
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("end-effector pose timestamp_ns must be a non-negative integer")
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("end-effector pose frame_id must be a non-empty string")
        position = _readonly_vector(
            self.position_m,
            field_name="end-effector pose position_m",
            length=3,
        )
        quaternion = np.asarray(self.quaternion_xyzw, dtype=float).copy()
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError(
                "end-effector pose quaternion_xyzw must contain four finite values"
            )
        norm = float(np.linalg.norm(quaternion))
        if norm <= np.finfo(float).eps:
            raise ValueError("end-effector pose quaternion_xyzw must have non-zero norm")
        quaternion /= norm
        if quaternion[3] < 0.0:
            quaternion *= -1.0
        quaternion.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "quaternion_xyzw", quaternion)


@dataclass(frozen=True)
class JointCommand:
    """A bounded-lifetime joint target tied to one runtime generation.

    ``generation`` invalidates queued motion after a safety transition, while
    ``deadline_ns`` prevents delayed transport frames from replaying stale targets.
    """

    names: tuple[str, ...]
    positions: np.ndarray
    submitted_at_ns: int
    deadline_ns: int
    generation: int
    unit: JointUnit = JointUnit.RADIAN
    velocity_limits: Optional[np.ndarray] = None
    acceleration_limits: Optional[np.ndarray] = None
    effort_limits: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        names = _names(self.names, field_name="joint command names")
        count = len(names)
        object.__setattr__(self, "names", names)
        object.__setattr__(
            self,
            "positions",
            _readonly_vector(self.positions, field_name="joint command positions", length=count),
        )
        for field_name in ("velocity_limits", "acceleration_limits", "effort_limits"):
            value = getattr(self, field_name)
            if value is None:
                continue
            array = _readonly_vector(value, field_name=f"joint command {field_name}", length=count)
            if np.any(array < 0.0):
                raise ValueError(f"joint command {field_name} must be non-negative")
            object.__setattr__(self, field_name, array)
        if type(self.submitted_at_ns) is not int or self.submitted_at_ns < 0:
            raise ValueError("joint command submitted_at_ns must be a non-negative integer")
        if type(self.deadline_ns) is not int or self.deadline_ns < self.submitted_at_ns:
            raise ValueError("joint command deadline_ns must not precede submitted_at_ns")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("joint command generation must be a non-negative integer")
        if not isinstance(self.unit, JointUnit):
            raise ValueError("joint command unit must be a JointUnit")


@dataclass(frozen=True)
class JointEffortCommand:
    """A bounded-lifetime joint effort target expressed strictly in newton-metres."""

    names: tuple[str, ...]
    efforts_nm: np.ndarray
    submitted_at_ns: int
    deadline_ns: int
    generation: int

    def __post_init__(self) -> None:
        names = _names(self.names, field_name="joint effort command names")
        object.__setattr__(self, "names", names)
        object.__setattr__(
            self,
            "efforts_nm",
            _readonly_vector(
                self.efforts_nm,
                field_name="joint effort command efforts_nm",
                length=len(names),
            ),
        )
        if type(self.submitted_at_ns) is not int or self.submitted_at_ns < 0:
            raise ValueError(
                "joint effort command submitted_at_ns must be a non-negative integer"
            )
        if type(self.deadline_ns) is not int or self.deadline_ns < self.submitted_at_ns:
            raise ValueError(
                "joint effort command deadline_ns must not precede submitted_at_ns"
            )
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError(
                "joint effort command generation must be a non-negative integer"
            )


@dataclass(frozen=True)
class SensorState:
    """Immutable vendor-neutral snapshot of non-joint telemetry."""

    name: str
    values: Mapping[str, Any]
    timestamp_ns: int
    sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("sensor state name must be a non-empty string")
        normalized: dict[str, Any] = {}
        for key, value in self.values.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("sensor state field names must be non-empty strings")
            normalized[key] = _immutable_sensor_value(value)
        object.__setattr__(self, "values", MappingProxyType(normalized))
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("sensor state timestamp_ns must be a non-negative integer")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sensor state sequence must be a non-negative integer")


def _immutable_mapping(values: Mapping[str, Any], *, field_name: str) -> Mapping[str, Any]:
    """Copy a named collection and expose it through a read-only proxy."""

    result = dict(values)
    if any(not isinstance(name, str) or not name.strip() for name in result):
        raise ValueError(f"{field_name} keys must be non-empty strings")
    return MappingProxyType(result)


@dataclass(frozen=True)
class RobotState:
    """Atomic application-facing collection of joint and auxiliary sensor groups."""

    joints: Mapping[str, JointState]
    sensors: Mapping[str, SensorState]

    def __post_init__(self) -> None:
        object.__setattr__(self, "joints", _immutable_mapping(self.joints, field_name="robot joints"))
        object.__setattr__(self, "sensors", _immutable_mapping(self.sensors, field_name="robot sensors"))


@dataclass(frozen=True)
class RobotCommand:
    """Joint commands staged as one logical robot update."""

    joints: Mapping[str, JointCommand]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joints",
            _immutable_mapping(self.joints, field_name="robot command joints"),
        )


@dataclass(frozen=True)
class RobotEffortCommand:
    """Joint-effort groups staged as one logical robot update."""

    joints: Mapping[str, JointEffortCommand]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joints",
            _immutable_mapping(
                self.joints,
                field_name="robot effort command joints",
            ),
        )


__all__ = [
    "EndEffectorPose",
    "JointCommand",
    "JointEffortCommand",
    "JointState",
    "JointUnit",
    "RobotCommand",
    "RobotEffortCommand",
    "RobotState",
    "SensorState",
]
