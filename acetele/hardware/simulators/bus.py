"""Deterministic in-memory implementation of the bus protocol contract."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from acetele.hardware.buses import (
    MotionEnvelope,
    RecoverableBusError,
    resolve_device_enable_request,
)


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
class MockEffort:
    """Scalar mock effort used to exercise runtime routing and mode safety."""

    effort_nm: float

    def __post_init__(self) -> None:
        value = float(self.effort_nm)
        if not math.isfinite(value):
            raise ValueError("mock effort must be finite")
        object.__setattr__(self, "effort_nm", value)


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
        self._enabled_ids: set[int] = set()
        self._effort_mode_ids: set[int] = set()
        self._efforts = {device_id: 0.0 for device_id in devices}

    @property
    def operation_timeout_ns(self) -> int:
        """Return a bounded transaction budget matching physical protocol defaults."""

        return 50_000_000

    def connect(self) -> None:
        """Reset the in-memory bus to a connected, disabled state."""

        self._connected = True
        self._enabled_ids.clear()
        self._effort_mode_ids.clear()
        self._efforts = {device_id: 0.0 for device_id in self._devices}

    def disconnect(self) -> None:
        """Disable and disconnect the in-memory bus."""

        self._enabled_ids.clear()
        self._effort_mode_ids.clear()
        self._connected = False

    def cancel(self) -> None:
        """No-op because mock reads never block."""

    def execute_safety(self, label: str, payload) -> object:
        """Model only software enable and stop behavior needed by runtime tests."""

        if label == "set_enabled":
            enabled, device_ids = resolve_device_enable_request(
                payload,
                self._devices,
                context="mock",
            )
            if enabled:
                if any(device_id in self._effort_mode_ids for device_id in device_ids):
                    raise ValueError(
                        "mock effort-mode devices cannot be enabled through position safety"
                    )
                self._enabled_ids.update(device_ids)
            else:
                self._enabled_ids.difference_update(device_ids)
            return True
        if label in ("hold", "emergency_stop"):
            for device_id in self._effort_mode_ids:
                self._efforts[device_id] = 0.0
            self._enabled_ids.difference_update(self._effort_mode_ids)
            if label == "emergency_stop":
                self._enabled_ids.clear()
                self._effort_mode_ids.clear()
            return True
        if label == "activate_effort":
            targets = tuple(payload)
            ids = tuple(target.device_id for target in targets)
            if not targets or any(
                device_id not in self._devices
                or not isinstance(target.payload, MockEffort)
                for device_id, target in zip(ids, targets)
            ):
                raise ValueError("invalid mock effort activation")
            self._effort_mode_ids.update(ids)
            self._enabled_ids.update(ids)
            for target in targets:
                self._efforts[target.device_id] = target.payload.effort_nm
            return True
        if label == "deactivate_effort":
            ids = tuple(payload)
            for device_id in ids:
                if device_id not in self._devices:
                    raise ValueError(f"unknown mock device ID {device_id}")
                self._efforts[device_id] = 0.0
            self._enabled_ids.difference_update(ids)
            self._effort_mode_ids.difference_update(ids)
            return True
        raise ValueError(f"unsupported mock safety task '{label}'")

    def write_motion(self, targets: Sequence[MotionEnvelope]) -> None:
        """Apply a validated target set atomically to in-memory positions."""

        values: dict[int, tuple[float, ...]] = {}
        efforts: dict[int, float] = {}
        for target in targets:
            if target.device_id not in self._enabled_ids:
                raise RecoverableBusError(
                    f"mock motion for device {target.device_id} is blocked while disabled"
                )
            definition = self._devices.get(target.device_id)
            if definition is None or not isinstance(
                target.payload, (MockMotion, MockEffort)
            ):
                raise RecoverableBusError("invalid mock motion target")
            if isinstance(target.payload, MockEffort):
                if target.device_id not in self._effort_mode_ids:
                    raise RecoverableBusError("mock effort target requires effort mode")
                efforts[target.device_id] = target.payload.effort_nm
            else:
                if target.device_id in self._effort_mode_ids:
                    raise RecoverableBusError("mock position target requires position mode")
                if len(target.payload.positions) != len(definition.initial_positions):
                    raise RecoverableBusError("mock motion position count is invalid")
                values[target.device_id] = target.payload.positions
        self._positions.update(values)
        self._efforts.update(efforts)

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
                (self._efforts[device_id],) * len(positions),
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
    "MockEffort",
    "MockMotion",
]
