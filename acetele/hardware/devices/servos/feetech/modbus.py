"""Actor-owned FEETECH Modbus-RTU transactions and SI conversions."""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from acetele.hardware.buses import (
    MotionEnvelope,
    RecoverableBusError,
    SerialTransport,
    decode_read_holding_registers,
    decode_write_registers_response,
    encode_read_holding_registers,
    encode_write_registers,
    resolve_device_enable_request,
)
from acetele.hardware.devices.servos.feetech.profile import FeetechModbusServoProfile


@dataclass(frozen=True)
class FeetechModbusMotion:
    """SI position target with optional Modbus profile limits."""

    position_rad: float
    velocity_rad_s: float | None = None
    acceleration_rad_s2: float | None = None
    torque_limit_ratio: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.position_rad):
            raise ValueError("FEETECH Modbus position must be finite")
        for field_name in ("velocity_rad_s", "acceleration_rad_s2"):
            value = getattr(self, field_name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(
                    f"FEETECH Modbus {field_name} must be finite and non-negative"
                )
        if self.torque_limit_ratio is not None and (
            not math.isfinite(self.torque_limit_ratio)
            or not 0.0 <= self.torque_limit_ratio <= 1.0
        ):
            raise ValueError("FEETECH Modbus torque limit must be in [0, 1]")


@dataclass(frozen=True)
class FeetechModbusFastState:
    """Fast FEETECH Modbus telemetry converted to SI units."""

    position_rad: float
    velocity_rad_s: float
    current_a: float
    effort_ratio: float
    voltage_v: float
    temperature_c: int
    status: int
    timestamp_ns: int


@dataclass(frozen=True)
class FeetechModbusSlowState:
    """Low-rate identity and health telemetry."""

    model: str
    firmware_version: int
    servo_version: int
    status: int
    voltage_v: float
    temperature_c: int
    timestamp_ns: int


class FeetechModbusBusProtocol:
    """Actor-owned FEETECH Modbus-RTU protocol for documented MB profiles."""

    identity_address = 0
    identity_count = 2
    goal_address = 128
    torque_enable_address = 129
    fast_state_address = 256
    fast_state_count = 8

    def __init__(
        self,
        transport: SerialTransport,
        devices: Mapping[int, FeetechModbusServoProfile],
        *,
        operation_timeout_s: float = 0.05,
        clock_ns=time.monotonic_ns,
    ) -> None:
        devices = dict(devices)
        if not devices:
            raise ValueError("FEETECH Modbus bus requires at least one servo")
        if any(type(slave_id) is not int or not 1 <= slave_id <= 247 for slave_id in devices):
            raise ValueError("FEETECH Modbus slave IDs must be integers in [1, 247]")
        if operation_timeout_s <= 0.0:
            raise ValueError("FEETECH Modbus operation timeout must be positive")
        self._transport = transport
        self._devices = devices
        self._operation_timeout_ns = round(operation_timeout_s * 1e9)
        self._clock_ns = clock_ns
        self._enabled_ids: set[int] = set()
        self._last_fast: dict[int, FeetechModbusFastState] = {}

    def connect(self) -> None:
        """Verify firmware/profile identity and leave every servo disabled."""

        self._last_fast.clear()
        self._enabled_ids.clear()
        self._transport.connect()
        try:
            # Firmware and servo versions jointly identify the documented register
            # contract; both must match before any control register is written.
            for slave_id, profile in self._devices.items():
                firmware, servo_version = self._read_registers(
                    slave_id,
                    self.identity_address,
                    self.identity_count,
                )
                if (firmware, servo_version) != (
                    profile.firmware_version,
                    profile.servo_version,
                ):
                    raise RuntimeError(
                        f"FEETECH Modbus slave {slave_id} reports firmware/servo "
                        f"{firmware}/{servo_version}, expected "
                        f"{profile.firmware_version}/{profile.servo_version} for "
                        f"{profile.model}"
                    )
            # Complete every identity check before issuing the first per-device write,
            # so a later mismatch cannot leave only part of the bus configured.
            for slave_id in self._devices:
                self._write_registers(slave_id, self.torque_enable_address, (0,))
            self._enabled_ids.clear()
        except BaseException:
            self._enabled_ids.clear()
            self._transport.disconnect()
            raise

    def cancel(self) -> None:
        """Interrupt pending transport I/O for bounded shutdown."""

        self._transport.cancel()

    def disconnect(self) -> None:
        """Release the serial transport."""

        self._enabled_ids.clear()
        self._transport.disconnect()

    def execute_safety(self, label: str, payload) -> object:
        """Execute ordered enable, hold, or emergency-disable transactions."""

        if label == "set_enabled":
            enabled, slave_ids = resolve_device_enable_request(
                payload,
                self._devices,
                context="FEETECH Modbus",
            )
            if enabled:
                # Seed goals from measured positions before torque-on, preventing motion
                # toward stale values retained in the servo registers.
                if not set(slave_ids).issubset(self._last_fast):
                    raise RuntimeError(
                        "FEETECH Modbus torque cannot be enabled before a complete "
                        "state sample for selected devices"
                    )
                for slave_id in slave_ids:
                    state = self._last_fast[slave_id]
                    self._write_registers(
                        slave_id,
                        self.goal_address,
                        self._encode_motion(
                            self._devices[slave_id],
                            FeetechModbusMotion(state.position_rad),
                        ),
                    )
            try:
                for slave_id in slave_ids:
                    self._write_registers(
                        slave_id,
                        self.torque_enable_address,
                        (int(enabled),),
                    )
            except BaseException:
                self._enabled_ids.difference_update(slave_ids)
                raise
            if enabled:
                self._enabled_ids.update(slave_ids)
            else:
                self._enabled_ids.difference_update(slave_ids)
            return True
        if label == "emergency_stop":
            self._enabled_ids.clear()
            for slave_id in self._devices:
                self._write_registers(slave_id, self.torque_enable_address, (0,))
            return True
        if label == "hold":
            slave_ids = tuple(sorted(self._enabled_ids))
            if not slave_ids:
                return True
            if not set(slave_ids).issubset(self._last_fast):
                raise RuntimeError("FEETECH Modbus cannot hold before state is available")
            for slave_id in slave_ids:
                state = self._last_fast[slave_id]
                values = self._encode_motion(
                    self._devices[slave_id],
                    FeetechModbusMotion(state.position_rad),
                )
                self._write_registers(slave_id, self.goal_address, values)
            return True
        raise ValueError(f"unsupported FEETECH Modbus safety task '{label}'")

    def write_motion(self, targets: Sequence[MotionEnvelope]) -> None:
        """Encode and write each latest SI target within its shared deadline."""

        seen: set[int] = set()
        try:
            deadline_ns = min(target.deadline_ns for target in targets)
            encoded: list[tuple[int, tuple[int, ...]]] = []
            for target in targets:
                if target.device_id not in self._enabled_ids:
                    raise ValueError(
                        f"FEETECH Modbus slave ID {target.device_id} is software-disabled"
                    )
                profile = self._devices.get(target.device_id)
                if profile is None:
                    raise ValueError(
                        f"unknown FEETECH Modbus slave ID {target.device_id}"
                    )
                if target.device_id in seen:
                    raise ValueError("FEETECH Modbus motion contains duplicate slave IDs")
                if not isinstance(target.payload, FeetechModbusMotion):
                    raise ValueError("FEETECH Modbus payload must be FeetechModbusMotion")
                seen.add(target.device_id)
                encoded.append(
                    (target.device_id, self._encode_motion(profile, target.payload))
                )
            # Validation and encoding finish before the first request. Modbus cannot
            # make wire writes atomic, but argument errors cannot cause partial output.
            for device_id, values in encoded:
                self._write_registers(
                    device_id,
                    self.goal_address,
                    values,
                    deadline_ns=deadline_ns,
                )
        except (TimeoutError, OSError, ValueError) as exc:
            raise RecoverableBusError("FEETECH Modbus motion write failed") from exc

    def read_fast_state(
        self,
        *,
        deadline_ns: int | None = None,
    ) -> Mapping[int, FeetechModbusFastState]:
        """Read motion-critical state from every configured slave."""

        states: dict[int, FeetechModbusFastState] = {}
        try:
            # Decode each response against the exact profile selected during preflight;
            # no register layout is inferred from the returned payload.
            for slave_id, profile in self._devices.items():
                values = self._read_registers(
                    slave_id,
                    self.fast_state_address,
                    self.fast_state_count,
                    deadline_ns=deadline_ns,
                )
                states[slave_id] = self._decode_fast_state(
                    profile,
                    values,
                    self._clock_ns(),
                )
        except (TimeoutError, OSError, ValueError) as exc:
            raise RecoverableBusError("FEETECH Modbus state read failed") from exc
        self._last_fast = states
        return states

    def read_slow_state(
        self,
        *,
        deadline_ns: int | None = None,
    ) -> Mapping[int, FeetechModbusSlowState]:
        """Read low-rate identity and health fields."""

        timestamp_ns = self._clock_ns()
        return {
            slave_id: FeetechModbusSlowState(
                model=self._devices[slave_id].model,
                firmware_version=self._devices[slave_id].firmware_version,
                servo_version=self._devices[slave_id].servo_version,
                status=state.status,
                voltage_v=state.voltage_v,
                temperature_c=state.temperature_c,
                timestamp_ns=timestamp_ns,
            )
            for slave_id, state in self._last_fast.items()
        }

    def _encode_motion(
        self,
        profile: FeetechModbusServoProfile,
        motion: FeetechModbusMotion,
    ) -> tuple[int, ...]:
        """Convert one SI command into the profile's signed Modbus registers."""

        position = round(
            motion.position_rad * profile.counts_per_revolution / (2.0 * math.pi)
        )
        if not -0x8000 <= position <= 0x7FFF:
            raise ValueError("FEETECH Modbus position exceeds the int16 register")
        velocity = (
            profile.default_velocity_raw
            if motion.velocity_rad_s is None
            else round(motion.velocity_rad_s / profile.velocity_unit_rad_s)
        )
        acceleration = (
            profile.default_acceleration_raw
            if motion.acceleration_rad_s2 is None
            else round(motion.acceleration_rad_s2 / profile.acceleration_unit_rad_s2)
        )
        torque_limit = (
            profile.default_torque_limit_raw
            if motion.torque_limit_ratio is None
            else round(motion.torque_limit_ratio * 1000.0)
        )
        for name, value in (
            ("velocity", velocity),
            ("acceleration", acceleration),
            ("torque limit", torque_limit),
        ):
            if not 0 <= value <= 0xFFFF:
                raise ValueError(f"FEETECH Modbus {name} exceeds the uint16 register")
        return (position & 0xFFFF, 1, acceleration, velocity, torque_limit)

    @staticmethod
    def _decode_fast_state(
        profile: FeetechModbusServoProfile,
        values: Sequence[int],
        timestamp_ns: int,
    ) -> FeetechModbusFastState:
        """Interpret two's-complement telemetry registers and convert them to SI."""

        status, position, velocity, pwm, voltage, temperature, _, current = values
        position_signed = struct.unpack(">h", struct.pack(">H", position))[0]
        velocity_signed = struct.unpack(">h", struct.pack(">H", velocity))[0]
        pwm_signed = struct.unpack(">h", struct.pack(">H", pwm))[0]
        current_signed = struct.unpack(">h", struct.pack(">H", current))[0]
        return FeetechModbusFastState(
            position_rad=(
                position_signed * 2.0 * math.pi / profile.counts_per_revolution
            ),
            velocity_rad_s=velocity_signed * profile.velocity_unit_rad_s,
            current_a=current_signed * profile.current_unit_a,
            effort_ratio=pwm_signed / 1000.0,
            voltage_v=voltage / 10.0,
            temperature_c=temperature,
            status=status,
            timestamp_ns=timestamp_ns,
        )

    def _read_registers(
        self,
        slave_id: int,
        address: int,
        count: int,
        *,
        deadline_ns: int | None = None,
    ) -> tuple[int, ...]:
        """Perform one bounded holding-register request and validate its response."""

        deadline_ns = self._deadline(deadline_ns)
        self._transport.write(
            encode_read_holding_registers(slave_id, address, count),
            deadline_ns=deadline_ns,
        )
        prefix = self._transport.read_exact(3, deadline_ns=deadline_ns)
        # Exception frames have a fixed CRC tail; successful reads advertise payload
        # length, avoiding timeout-based frame delimiting.
        remaining = 2 if prefix[1] & 0x80 else prefix[2] + 2
        frame = prefix + self._transport.read_exact(remaining, deadline_ns=deadline_ns)
        return decode_read_holding_registers(
            frame,
            expected_slave=slave_id,
            expected_count=count,
        )

    def _write_registers(
        self,
        slave_id: int,
        address: int,
        values: Sequence[int],
        *,
        deadline_ns: int | None = None,
    ) -> None:
        """Write a contiguous block and require an exact address/count echo."""

        values = tuple(values)
        deadline_ns = self._deadline(deadline_ns)
        self._transport.write(
            encode_write_registers(slave_id, address, values),
            deadline_ns=deadline_ns,
        )
        prefix = self._transport.read_exact(2, deadline_ns=deadline_ns)
        remaining = 3 if prefix[1] & 0x80 else 6
        frame = prefix + self._transport.read_exact(remaining, deadline_ns=deadline_ns)
        decode_write_registers_response(
            frame,
            expected_slave=slave_id,
            expected_address=address,
            expected_count=len(values),
        )

    def _deadline(self, deadline_ns: int | None) -> int:
        operation_deadline_ns = self._clock_ns() + self._operation_timeout_ns
        if deadline_ns is None:
            return operation_deadline_ns
        if type(deadline_ns) is not int or deadline_ns < 0:
            raise ValueError("FEETECH Modbus deadline must be a non-negative integer")
        return min(operation_deadline_ns, deadline_ns)


__all__ = [
    "FeetechModbusBusProtocol",
    "FeetechModbusFastState",
    "FeetechModbusMotion",
    "FeetechModbusSlowState",
]
