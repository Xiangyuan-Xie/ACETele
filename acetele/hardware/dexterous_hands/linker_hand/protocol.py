"""Linker Hand Modbus protocol with normalized joint semantics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Optional, Sequence

from acetele.hardware.dexterous_hands.linker_hand.profiles import LinkerHandProfile
from acetele.hardware.serial import (
    MotionEnvelope,
    RecoverableBusError,
    SerialTransport,
    decode_read_input_registers,
    decode_write_registers_response,
    encode_read_input_registers,
    encode_write_registers,
)


@dataclass(frozen=True)
class LinkerHandMotion:
    """Model-ordered normalized hand positions in ``[0, 1]``."""

    positions: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(self.positions)
        if not values or any(
            isinstance(value, bool) or not isinstance(value, Real) for value in values
        ):
            raise ValueError("Linker Hand positions must be a numeric sequence")
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in values
        ):
            raise ValueError("Linker Hand positions must be normalized to [0, 1]")
        object.__setattr__(self, "positions", tuple(float(value) for value in values))


@dataclass(frozen=True)
class LinkerHandFastState:
    """Fast normalized positions, efforts, and velocities."""

    positions: tuple[float, ...]
    efforts: tuple[float, ...]
    velocities: tuple[float, ...]
    timestamp_ns: int


@dataclass(frozen=True)
class LinkerHandSlowState:
    """Low-rate temperatures, per-joint errors, and firmware identity."""

    temperatures_c: tuple[int, ...]
    errors: tuple[int, ...]
    version: tuple[int, ...]
    timestamp_ns: int


class LinkerHandModbusProtocol:
    """Actor-owned Modbus transport for all documented Linker Hand RS485 profiles."""

    def __init__(
        self,
        transport: SerialTransport,
        devices: Mapping[int, LinkerHandProfile],
        *,
        operation_timeout_s: float = 0.05,
        clock_ns=time.monotonic_ns,
    ) -> None:
        devices = dict(devices)
        if not devices:
            raise ValueError("Linker Hand protocol requires at least one device")
        if any(type(slave) is not int or not 1 <= slave <= 247 for slave in devices):
            raise ValueError("Linker Hand slave IDs must be integers in [1, 247]")
        if operation_timeout_s <= 0.0:
            raise ValueError("Linker Hand operation timeout must be positive")
        self._transport = transport
        self._devices = devices
        self._operation_timeout_ns = round(operation_timeout_s * 1e9)
        self._clock_ns = clock_ns
        self._last_frame_ns: Optional[int] = None
        self._last_fast: dict[int, LinkerHandFastState] = {}
        self._versions: dict[int, tuple[int, ...]] = {}

    def connect(self) -> None:
        """Open the bus and verify each hand's reported joint count."""

        self._last_fast.clear()
        self._versions.clear()
        self._last_frame_ns = None
        self._transport.connect()
        try:
            # The first version register encodes joint count in the documented SDK.
            # Checking it prevents applying an O6/L6/L7/L10 register layout blindly.
            for slave_id, profile in self._devices.items():
                version = self._read_registers(
                    slave_id,
                    profile.version_address,
                    profile.version_count,
                )
                if not version or version[0] != profile.joint_count:
                    raise RuntimeError(
                        f"Linker Hand {slave_id} identity reports {version[0] if version else 'no'} "
                        f"joints, expected {profile.joint_count} for {profile.model}"
                    )
                self._versions[slave_id] = version
        except BaseException:
            self._transport.disconnect()
            raise

    def cancel(self) -> None:
        """Interrupt pending transport I/O for actor shutdown."""

        self._transport.cancel()

    def disconnect(self) -> None:
        """Release the serial transport."""

        self._transport.disconnect()

    def execute_safety(self, label: str, payload) -> object:
        """Execute the strongest documented hand safety behavior."""

        # The documented RS485 map has no verifiable torque-disable command. Holding
        # the last valid pose is therefore the strongest software action available.
        if label in ("hold", "emergency_stop"):
            if not self._last_fast:
                raise RuntimeError("Linker Hand cannot hold before a valid state is available")
            for slave_id, state in self._last_fast.items():
                self._write_positions(slave_id, state.positions)
            return "hold_only"
        if label == "set_enabled" and payload is True:
            return True
        raise RuntimeError(
            "Linker Hand RS485 profile has no documented verifiable torque-disable command"
        )

    def write_motion(self, targets: Sequence[MotionEnvelope]) -> None:
        """Write normalized targets while respecting model-specific frame gaps."""

        try:
            deadline_ns = min(target.deadline_ns for target in targets)
            for target in targets:
                profile = self._devices.get(target.device_id)
                if profile is None:
                    raise ValueError(f"unknown Linker Hand slave ID {target.device_id}")
                if not isinstance(target.payload, LinkerHandMotion):
                    raise ValueError("Linker Hand motion payload must be LinkerHandMotion")
                if len(target.payload.positions) != profile.joint_count:
                    raise ValueError(
                        f"Linker Hand {profile.model} requires {profile.joint_count} positions"
                    )
                self._write_positions(
                    target.device_id,
                    target.payload.positions,
                    deadline_ns=deadline_ns,
                )
        except (TimeoutError, OSError, ValueError) as exc:
            raise RecoverableBusError("Linker Hand motion write failed") from exc

    def read_fast_state(
        self,
        *,
        deadline_ns: int | None = None,
    ) -> Mapping[int, LinkerHandFastState]:
        """Read contiguous position, effort, and velocity registers."""

        states: dict[int, LinkerHandFastState] = {}
        try:
            for slave_id, profile in self._devices.items():
                registers = self._read_registers(
                    slave_id,
                    profile.position_address,
                    profile.fast_register_count,
                    deadline_ns=deadline_ns,
                )
                offset = profile.position_address
                positions = self._normalize_block(
                    registers,
                    profile.position_address - offset,
                    profile.joint_count,
                )
                efforts = self._normalize_block(
                    registers,
                    profile.torque_address - offset,
                    profile.joint_count,
                )
                velocities = self._normalize_block(
                    registers,
                    profile.speed_address - offset,
                    profile.joint_count,
                )
                states[slave_id] = LinkerHandFastState(
                    positions,
                    efforts,
                    velocities,
                    self._clock_ns(),
                )
        except (TimeoutError, OSError, ValueError) as exc:
            raise RecoverableBusError("Linker Hand fast-state read failed") from exc
        self._last_fast = states
        return states

    def read_slow_state(
        self,
        *,
        deadline_ns: int | None = None,
    ) -> Mapping[int, LinkerHandSlowState]:
        """Read temperatures, errors, and cached identity outside the fast path."""

        states: dict[int, LinkerHandSlowState] = {}
        try:
            for slave_id, profile in self._devices.items():
                count = profile.error_address + profile.joint_count - profile.temperature_address
                registers = self._read_registers(
                    slave_id,
                    profile.temperature_address,
                    count,
                    deadline_ns=deadline_ns,
                )
                error_offset = profile.error_address - profile.temperature_address
                states[slave_id] = LinkerHandSlowState(
                    tuple(registers[: profile.joint_count]),
                    tuple(registers[error_offset : error_offset + profile.joint_count]),
                    self._versions[slave_id],
                    self._clock_ns(),
                )
        except (TimeoutError, OSError, ValueError) as exc:
            raise RecoverableBusError("Linker Hand slow-state read failed") from exc
        return states

    def _write_positions(
        self,
        slave_id: int,
        positions: Sequence[float],
        *,
        deadline_ns: int | None = None,
    ) -> None:
        """Quantize normalized positions and verify the Modbus write response."""

        profile = self._devices[slave_id]
        raw = tuple(round(float(value) * 255.0) for value in positions)
        request = encode_write_registers(slave_id, profile.position_address, raw)
        self._exchange_write(
            request,
            slave_id,
            profile.position_address,
            len(raw),
            profile,
            deadline_ns=deadline_ns,
        )

    def _read_registers(
        self,
        slave_id: int,
        address: int,
        count: int,
        *,
        deadline_ns: int | None = None,
    ) -> tuple[int, ...]:
        """Read one register range while enforcing the model-specific frame gap."""

        profile = self._devices[slave_id]
        self._wait_frame_gap(profile)
        deadline_ns = self._deadline(deadline_ns)
        self._transport.write(
            encode_read_input_registers(slave_id, address, count),
            deadline_ns=deadline_ns,
        )
        prefix = self._transport.read_exact(3, deadline_ns=deadline_ns)
        # Modbus exception responses have a fixed two-byte tail; normal read responses
        # declare their data byte count in the third prefix byte.
        remaining = 2 if prefix[1] & 0x80 else prefix[2] + 2
        frame = prefix + self._transport.read_exact(remaining, deadline_ns=deadline_ns)
        self._last_frame_ns = self._clock_ns()
        return decode_read_input_registers(
            frame,
            expected_slave=slave_id,
            expected_count=count,
        )

    def _exchange_write(
        self,
        request: bytes,
        slave_id: int,
        address: int,
        count: int,
        profile: LinkerHandProfile,
        *,
        deadline_ns: int | None = None,
    ) -> None:
        """Write registers and require the slave to echo address and count."""

        self._wait_frame_gap(profile)
        deadline_ns = self._deadline(deadline_ns)
        self._transport.write(request, deadline_ns=deadline_ns)
        prefix = self._transport.read_exact(2, deadline_ns=deadline_ns)
        remaining = 3 if prefix[1] & 0x80 else 6
        response = prefix + self._transport.read_exact(remaining, deadline_ns=deadline_ns)
        self._last_frame_ns = self._clock_ns()
        decode_write_registers_response(
            response,
            expected_slave=slave_id,
            expected_address=address,
            expected_count=count,
        )

    def _deadline(self, deadline_ns: int | None) -> int:
        operation_deadline_ns = self._clock_ns() + self._operation_timeout_ns
        if deadline_ns is None:
            return operation_deadline_ns
        if type(deadline_ns) is not int or deadline_ns < 0:
            raise ValueError("Linker Hand deadline must be a non-negative integer")
        return min(operation_deadline_ns, deadline_ns)

    def _wait_frame_gap(self, profile: LinkerHandProfile) -> None:
        if self._last_frame_ns is None or profile.frame_gap_s <= 0.0:
            return
        remaining_ns = round(profile.frame_gap_s * 1e9) - (
            self._clock_ns() - self._last_frame_ns
        )
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1e9)

    @staticmethod
    def _normalize_block(
        registers: Sequence[int],
        start: int,
        count: int,
    ) -> tuple[float, ...]:
        return tuple(float(value) / 255.0 for value in registers[start : start + count])


__all__ = [
    "LinkerHandFastState",
    "LinkerHandModbusProtocol",
    "LinkerHandMotion",
    "LinkerHandSlowState",
]
