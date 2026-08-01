"""Deterministic in-memory implementation of the bus protocol contract."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from acetele.hardware.buses import MotionEnvelope, RecoverableBusError


@dataclass(frozen=True)
class MockDeviceDefinition:
    """Static initial state for one simulated bus device."""

    initial_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        positions = tuple(float(value) for value in self.initial_positions)
        if not positions:
            raise ValueError("mock device requires at least one position")
        object.__setattr__(self, "initial_positions", positions)


@dataclass(frozen=True)
class MockMotion:
    """Model-ordered mock target positions."""

    positions: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.positions)
        if not values:
            raise ValueError("mock motion requires at least one position")
        object.__setattr__(self, "positions", values)


@dataclass(frozen=True)
class MockDeviceState:
    """Deterministic state sample returned by the mock bus."""

    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    efforts: tuple[float, ...]
    timestamp_ns: int


class MockBusProtocol:
    """No-dynamics protocol used to test routing and lifecycle without hardware."""

    def __init__(
        self,
        devices: Mapping[int, MockDeviceDefinition],
        *,
        clock_ns=time.monotonic_ns,
    ) -> None:
        devices = dict(devices)
        if not devices:
            raise ValueError("mock bus requires at least one device")
        if any(type(device_id) is not int or device_id < 0 for device_id in devices):
            raise ValueError("mock device IDs must be non-negative integers")
        self._devices = devices
        self._positions = {
            device_id: definition.initial_positions
            for device_id, definition in devices.items()
        }
        self._clock_ns = clock_ns
        self._connected = False
        self._enabled = False

    def connect(self) -> None:
        """Reset the in-memory bus to a connected, disabled state."""

        self._connected = True
        self._enabled = False

    def disconnect(self) -> None:
        """Disable and disconnect the in-memory bus."""

        self._enabled = False
        self._connected = False

    def cancel(self) -> None:
        """No-op because mock reads never block."""

    def execute_safety(self, label: str, payload) -> object:
        """Model only software enable and stop behavior needed by runtime tests."""

        if label == "set_enabled":
            if type(payload) is not bool:
                raise ValueError("mock set_enabled payload must be a boolean")
            self._enabled = payload
            return True
        if label in ("hold", "emergency_stop"):
            if label == "emergency_stop":
                self._enabled = False
            return True
        raise ValueError(f"unsupported mock safety task '{label}'")

    def write_motion(self, targets: Sequence[MotionEnvelope]) -> None:
        """Apply a validated target set atomically to in-memory positions."""

        if not self._enabled:
            raise RecoverableBusError("mock motion is blocked while disabled")
        values: dict[int, tuple[float, ...]] = {}
        for target in targets:
            definition = self._devices.get(target.device_id)
            if definition is None or not isinstance(target.payload, MockMotion):
                raise RecoverableBusError("invalid mock motion target")
            if len(target.payload.positions) != len(definition.initial_positions):
                raise RecoverableBusError("mock motion position count is invalid")
            values[target.device_id] = target.payload.positions
        self._positions.update(values)

    def read_fast_state(
        self,
        *,
        deadline_ns: Optional[int] = None,
    ) -> Mapping[int, MockDeviceState]:
        """Return coherent zero-dynamics state for every mock device."""

        timestamp_ns = self._clock_ns()
        return {
            device_id: MockDeviceState(
                positions,
                (0.0,) * len(positions),
                (0.0,) * len(positions),
                timestamp_ns,
            )
            for device_id, positions in self._positions.items()
        }

    def read_slow_state(
        self,
        *,
        deadline_ns: Optional[int] = None,
    ) -> Mapping[int, dict[str, object]]:
        """Return no slow telemetry for the deterministic mock."""

        return {}


__all__ = [
    "MockBusProtocol",
    "MockDeviceDefinition",
    "MockDeviceState",
    "MockMotion",
]
