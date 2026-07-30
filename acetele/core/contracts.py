"""Vendor-neutral, immutable contracts shared by every ACETele layer.

Hardware addresses and register units deliberately stop below this boundary. Runtime,
control, and ROS adapters exchange named joints in SI units (or an explicitly declared
normalized unit) so vendor protocols cannot leak into application logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


class Backend(str, Enum):
    """Select whether a specification creates physical or deterministic mock buses."""

    PHYSICAL = "physical"
    MOCK = "mock"


class JointUnit(str, Enum):
    """Unit carried by every joint state and command."""

    RADIAN = "radian"
    NORMALIZED = "normalized"


class SafetyAction(str, Enum):
    """Safety operations a device can implement without vendor-specific arguments."""

    HOLD = "hold"
    DISABLE = "disable"
    DAMPING = "damping"
    ZERO_EFFORT = "zero_effort"


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
class DeviceCapabilities:
    """Static capabilities used to reject unsupported operations before I/O."""

    units: tuple[JointUnit, ...]
    safety_actions: tuple[SafetyAction, ...]
    max_update_hz: float
    supports_velocity: bool = False
    supports_effort: bool = False
    supports_temperature: bool = False
    supports_fault_state: bool = False
    supports_tactile: bool = False
    supports_verified_disable: bool = False

    def __post_init__(self) -> None:
        units = tuple(self.units)
        actions = tuple(self.safety_actions)
        if not units or any(not isinstance(unit, JointUnit) for unit in units):
            raise ValueError("device capabilities units must contain JointUnit values")
        if any(not isinstance(action, SafetyAction) for action in actions):
            raise ValueError("device capabilities safety_actions must contain SafetyAction values")
        if not np.isfinite(self.max_update_hz) or self.max_update_hz <= 0.0:
            raise ValueError("device capabilities max_update_hz must be finite and positive")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "safety_actions", actions)


@runtime_checkable
class JointHardware(Protocol):
    """Minimal explicit-lifecycle interface for a named joint device."""

    @property
    def connected(self) -> bool:
        """Return whether the device owns live hardware resources."""

        ...

    @property
    def capabilities(self) -> DeviceCapabilities:
        """Return immutable capabilities known before commands are issued."""

        ...

    def connect(self) -> None:
        """Acquire hardware resources after static validation."""

        ...

    def read(self) -> JointState:
        """Return the latest coherent state sample."""

        ...

    def write(self, command: JointCommand) -> None:
        """Submit a validated, live command."""

        ...

    def hold(self) -> None:
        """Hold the latest trustworthy position."""

        ...

    def set_enabled(self, enabled: bool) -> None:
        """Change actuator enable state through a safety transaction."""

        ...

    def emergency_stop(self) -> None:
        """Execute and latch the strongest supported stop action."""

        ...

    def disconnect(self) -> None:
        """Release all resources with bounded shutdown."""

        ...


__all__ = [
    "Backend",
    "DeviceCapabilities",
    "JointCommand",
    "JointHardware",
    "JointState",
    "JointUnit",
    "RobotCommand",
    "RobotState",
    "SafetyAction",
    "SensorState",
]
