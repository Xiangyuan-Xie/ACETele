"""Actor-owned FEETECH HLS/SMS packet transactions and SI conversions."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from acetele.hardware.buses import (
    MotionEnvelope,
    RecoverableBusError,
    SerialTransport,
    resolve_device_enable_request,
)
from acetele.hardware.devices.servos.feetech.codec import (
    FeetechInstruction,
    FeetechPacketCodec,
    FeetechPacketError,
    FeetechStatusPacket,
)
from acetele.hardware.devices.servos.feetech.profile import (
    FeetechPacketFamily,
    FeetechPacketServoProfile,
)


def nearest_multiturn_position_target(
    current_position: int,
    goal_position: int,
    position_period: int,
) -> int:
    """Choose the nearest equivalent count without overflowing signed 15-bit range."""

    if type(current_position) is not int or type(goal_position) is not int:
        raise ValueError("FEETECH position counts must be integers")
    if type(position_period) is not int or position_period <= 0 or position_period % 2:
        raise ValueError("FEETECH position period must be a positive even integer")
    half_period = position_period // 2
    shortest_delta = (
        goal_position - current_position + half_period
    ) % position_period - half_period
    if shortest_delta == -half_period and current_position < 0:
        shortest_delta = half_period
    adjusted = current_position + shortest_delta
    maximum = (1 << 15) - 1
    if not -maximum <= adjusted <= maximum:
        raise ValueError(
            "FEETECH multi-turn position range is exhausted: "
            f"current={current_position}, goal={goal_position}, nearest={adjusted}"
        )
    return adjusted


@dataclass(frozen=True)
class FeetechPacketMotion:
    """One SI position target and optional profile limits."""

    position_rad: float
    velocity_rad_s: float | None = None
    acceleration_rad_s2: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.position_rad):
            raise ValueError("FEETECH position must be finite")
        for field_name in (
            "velocity_rad_s",
            "acceleration_rad_s2",
        ):
            value = getattr(self, field_name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"FEETECH {field_name} must be finite and non-negative")


@dataclass(frozen=True)
class FeetechPacketEffort:
    """One prevalidated signed HLS current-register target."""

    current_raw: int

    def __post_init__(self) -> None:
        if type(self.current_raw) is not int or not -0x7FFF <= self.current_raw <= 0x7FFF:
            raise ValueError("FEETECH effort current must be a signed-15-bit integer")


@dataclass(frozen=True)
class FeetechPacketFastState:
    """Fast HLS/SMS telemetry converted to SI units."""

    position_rad: float
    velocity_rad_s: float
    current_a: float
    load_ratio: float
    voltage_v: float
    temperature_c: int
    status: int
    timestamp_ns: int


@dataclass(frozen=True)
class FeetechPacketSlowState:
    """Low-rate identity and health telemetry for diagnostics."""

    model: str
    model_number: int
    voltage_v: float
    temperature_c: int
    status: int
    timestamp_ns: int


class FeetechPacketBusProtocol:
    """Blocking FEETECH packet protocol owned by one serial bus actor."""

    model_address = 3
    mode_address = 33
    torque_enable_address = 40
    goal_start_address = 41
    goal_torque_address = 44
    # Addresses 56..70 contain position, speed, load, voltage, temperature,
    # reserved bytes, moving, and current. Fault bits come from the packet error byte;
    # address 65 is reserved and must never be interpreted as hardware status.
    present_state_address = 56
    present_state_length = 15

    def __init__(
        self,
        transport: SerialTransport,
        devices: Mapping[int, FeetechPacketServoProfile],
        *,
        expected_model_numbers: Optional[Mapping[int, int]] = None,
        operation_timeout_s: float = 0.05,
        clock_ns=time.monotonic_ns,
    ) -> None:
        devices = dict(devices)
        if not devices:
            raise ValueError("FEETECH packet bus requires at least one servo")
        for servo_id in devices:
            FeetechPacketCodec.validate_servo_id(servo_id)
        families = {profile.family for profile in devices.values()}
        if len(families) != 1:
            raise ValueError("FEETECH HLS and SMS profiles cannot share one packet bus")
        expected_model_numbers = dict(expected_model_numbers or {})
        if set(expected_model_numbers) - set(devices):
            raise ValueError("FEETECH expected model map contains unknown servo IDs")
        resolved_models: dict[int, int] = {}
        for servo_id, profile in devices.items():
            model_number = expected_model_numbers.get(servo_id, profile.model_number)
            if model_number is not None:
                if type(model_number) is not int or not 0 <= model_number <= 0xFFFF:
                    raise ValueError("FEETECH expected model numbers must be uint16 values")
                resolved_models[servo_id] = model_number
        if operation_timeout_s <= 0.0:
            raise ValueError("FEETECH packet operation timeout must be positive")
        # The protocol owns no thread. BusActor is the sole caller after
        # connection, which keeps packet parsing and response ordering deterministic.
        self._transport = transport
        self._devices = devices
        self._expected_model_numbers = resolved_models
        self._observed_model_numbers: dict[int, int] = {}
        self._operation_timeout_ns = round(operation_timeout_s * 1e9)
        self._clock_ns = clock_ns
        self._enabled_ids: set[int] = set()
        self._effort_mode_ids: set[int] = set()
        self._last_fast: dict[int, FeetechPacketFastState] = {}
        self._hold_positions_rad: dict[int, float] = {}

    def connect(self) -> None:
        """Verify model identities, disable torque, and select position mode."""

        self._last_fast.clear()
        self._hold_positions_rad.clear()
        self._observed_model_numbers.clear()
        self._enabled_ids.clear()
        self._effort_mode_ids.clear()
        self._transport.connect()
        try:
            # Read identity before changing actuator state. Exact model checks prevent
            # applying a memory map merely because another servo looks similar.
            for servo_id in self._devices:
                model_number = FeetechPacketCodec.decode_word(
                    self._read_register(servo_id, self.model_address, 2)
                )
                self._observed_model_numbers[servo_id] = model_number
                expected = self._expected_model_numbers.get(servo_id)
                if expected is not None and model_number != expected:
                    raise RuntimeError(
                        f"FEETECH servo ID {servo_id} reports model {model_number}, "
                        f"expected {expected} for {self._devices[servo_id].model}"
                    )
            # Torque-off must precede mode selection; changing operating mode under
            # load is not a safe initialization sequence.
            self._sync_write(
                self.torque_enable_address,
                {servo_id: b"\x00" for servo_id in self._devices},
            )
            self._sync_write(
                self.mode_address,
                {servo_id: b"\x00" for servo_id in self._devices},
            )
            self._enabled_ids.clear()
            self._effort_mode_ids.clear()
        except BaseException:
            self._enabled_ids.clear()
            self._effort_mode_ids.clear()
            self._observed_model_numbers.clear()
            self._transport.disconnect()
            raise

    def cancel(self) -> None:
        """Interrupt pending transport I/O for bounded shutdown."""

        self._transport.cancel()

    def disconnect(self) -> None:
        """Release transport and clear cached feedback."""

        self._enabled_ids.clear()
        self._effort_mode_ids.clear()
        self._hold_positions_rad.clear()
        self._transport.disconnect()

    def execute_safety(self, label: str, payload) -> object:
        """Execute ordered enable, hold, disable, or calibration operations."""

        if label == "set_enabled":
            enabled, servo_ids = resolve_device_enable_request(
                payload,
                self._devices,
                context="FEETECH packet",
            )
            if enabled:
                if set(servo_ids) & self._effort_mode_ids:
                    raise RuntimeError(
                        "FEETECH effort-mode servos must be enabled by activate_effort"
                    )
                # Seed selected goals with measured position before enabling torque. This
                # avoids a jump toward a stale power-on goal register.
                if not set(servo_ids).issubset(self._last_fast):
                    raise RuntimeError(
                        "FEETECH torque cannot be enabled before a complete state sample "
                        "for selected devices is available"
                    )
                hold_values = {
                    servo_id: self._encode_motion(
                        self._devices[servo_id],
                        FeetechPacketMotion(self._last_fast[servo_id].position_rad),
                        reference_position_rad=self._last_fast[servo_id].position_rad,
                    )
                    for servo_id in servo_ids
                }
                self._sync_write(self.goal_start_address, hold_values)
                self._hold_positions_rad.update(
                    {
                        servo_id: self._last_fast[servo_id].position_rad
                        for servo_id in servo_ids
                    }
                )
            try:
                self._sync_write(
                    self.torque_enable_address,
                    {servo_id: bytes((int(enabled),)) for servo_id in servo_ids},
                )
            except BaseException:
                # A failed broadcast leaves the addressed hardware state unknown.
                self._enabled_ids.difference_update(servo_ids)
                raise
            if enabled:
                self._enabled_ids.update(servo_ids)
            else:
                self._enabled_ids.difference_update(servo_ids)
            return True
        if label == "activate_effort":
            targets = tuple(payload)
            if not targets:
                raise ValueError("FEETECH effort activation requires preload targets")
            effort_values: dict[int, bytes] = {}
            for target in targets:
                profile = self._devices.get(target.device_id)
                if profile is None:
                    raise ValueError(f"unknown FEETECH servo ID {target.device_id}")
                if profile.family != FeetechPacketFamily.HLS:
                    raise ValueError("FEETECH Torque mode is supported only by HLS profiles")
                if target.device_id in effort_values or not isinstance(
                    target.payload, FeetechPacketEffort
                ):
                    raise ValueError("invalid FEETECH effort activation payload")
                effort_values[target.device_id] = self._encode_effort(target.payload)
            servo_ids = tuple(effort_values)
            try:
                # Mode changes are one strict transaction: torque-off, mode select,
                # preload, then enable. Any failure triggers best-effort zero current
                # and torque-off before the error is allowed to escape the worker.
                self._sync_write(
                    self.torque_enable_address,
                    {servo_id: b"\x00" for servo_id in servo_ids},
                )
                self._enabled_ids.difference_update(servo_ids)
                self._sync_write(
                    self.mode_address,
                    {servo_id: b"\x02" for servo_id in servo_ids},
                )
                self._sync_write(self.goal_torque_address, effort_values)
                self._sync_write(
                    self.torque_enable_address,
                    {servo_id: b"\x01" for servo_id in servo_ids},
                )
            except BaseException:
                self._best_effort_zero_and_disable(servo_ids)
                try:
                    self._sync_write(
                        self.mode_address,
                        {servo_id: b"\x00" for servo_id in servo_ids},
                    )
                except BaseException:
                    pass
                self._effort_mode_ids.difference_update(servo_ids)
                raise
            self._effort_mode_ids.update(servo_ids)
            self._enabled_ids.update(servo_ids)
            return True
        if label == "deactivate_effort":
            servo_ids = tuple(payload)
            if not servo_ids or any(
                type(servo_id) is not int or servo_id not in self._devices
                for servo_id in servo_ids
            ):
                raise ValueError("invalid FEETECH effort deactivation IDs")
            first_error = self._attempt_safety_writes(
                (
                    (
                        self.goal_torque_address,
                        {servo_id: b"\x00\x00" for servo_id in servo_ids},
                    ),
                    (
                        self.torque_enable_address,
                        {servo_id: b"\x00" for servo_id in servo_ids},
                    ),
                    (
                        self.mode_address,
                        {servo_id: b"\x00" for servo_id in servo_ids},
                    ),
                )
            )
            # Never retain an optimistic enabled cache after an unload attempt. Keep
            # Torque-mode ownership on failure so a later position enable cannot cross
            # an uncertain mode transition.
            self._enabled_ids.difference_update(servo_ids)
            if first_error is not None:
                raise first_error
            self._effort_mode_ids.difference_update(servo_ids)
            return True
        if label == "emergency_stop":
            effort_ids = tuple(self._effort_mode_ids)
            operations = []
            if effort_ids:
                operations.append(
                    (
                        self.goal_torque_address,
                        {servo_id: b"\x00\x00" for servo_id in effort_ids},
                    )
                )
            operations.append(
                (
                    self.torque_enable_address,
                    {servo_id: b"\x00" for servo_id in self._devices},
                )
            )
            if effort_ids:
                operations.append(
                    (
                        self.mode_address,
                        {servo_id: b"\x00" for servo_id in effort_ids},
                    )
                )
            first_error = self._attempt_safety_writes(tuple(operations))
            self._enabled_ids.clear()
            if first_error is not None:
                raise first_error
            self._effort_mode_ids.difference_update(effort_ids)
            return True
        if label == "hold":
            effort_ids = tuple(sorted(self._enabled_ids & self._effort_mode_ids))
            first_error = None
            if effort_ids:
                first_error = self._attempt_safety_writes(
                    (
                        (
                            self.goal_torque_address,
                            {servo_id: b"\x00\x00" for servo_id in effort_ids},
                        ),
                        (
                            self.torque_enable_address,
                            {servo_id: b"\x00" for servo_id in effort_ids},
                        ),
                    )
                )
                self._enabled_ids.difference_update(effort_ids)
            position_ids = tuple(sorted(self._enabled_ids - self._effort_mode_ids))
            try:
                if position_ids:
                    if not set(position_ids).issubset(self._hold_positions_rad):
                        raise RuntimeError("FEETECH bus has no trustworthy hold target")
                    hold_values = {
                        servo_id: self._encode_motion(
                            self._devices[servo_id],
                            FeetechPacketMotion(self._hold_positions_rad[servo_id]),
                            reference_position_rad=self._hold_positions_rad[servo_id],
                        )
                        for servo_id in position_ids
                    }
                    self._sync_write(self.goal_start_address, hold_values)
                    self._sync_write(
                        self.torque_enable_address,
                        {servo_id: b"\x01" for servo_id in position_ids},
                    )
            except BaseException as exc:
                first_error = first_error or exc
            if first_error is not None:
                raise first_error
            return True
        if label == "calibrate_offset":
            if self._enabled_ids:
                raise RuntimeError("FEETECH offset calibration requires disabled torque")
            values: dict[int, int] = dict(payload)
            if not values:
                raise ValueError("FEETECH calibration payload cannot be empty")
            for servo_id, raw_position in values.items():
                if servo_id not in self._devices:
                    raise ValueError(f"unknown FEETECH calibration servo ID {servo_id}")
                if type(raw_position) is not int or not -0x7FFF <= raw_position <= 0x7FFF:
                    raise ValueError(
                        "FEETECH calibration positions must be signed-15-bit integers"
                    )
                # Calibration is intentionally per-servo because the instruction has
                # a status response that must be checked before proceeding. Unlike
                # motion registers, the official reOfsCal SDK writes this parameter as
                # a little-endian two's-complement int16, not signed magnitude.
                self._discard_stale_input()
                self._send(
                    FeetechPacketCodec.encode_instruction(
                        servo_id,
                        FeetechInstruction.OFFSET_CALIBRATION,
                        raw_position.to_bytes(2, "little", signed=True),
                    )
                )
                status = self._read_status(expected_servo_id=servo_id, parameter_count=0)
                self._raise_status_error(status)
            return True
        raise ValueError(f"unsupported FEETECH packet safety task '{label}'")

    def write_motion(self, targets: Sequence[MotionEnvelope]) -> None:
        """Merge latest per-ID targets into one packet-family sync write."""

        disabled_ids = tuple(
            target.device_id
            for target in targets
            if target.device_id not in self._enabled_ids
        )
        if disabled_ids:
            raise RecoverableBusError(
                "FEETECH motion is blocked for software-disabled servo IDs: "
                + ", ".join(str(servo_id) for servo_id in disabled_ids)
            )
        position_values: dict[int, bytes] = {}
        effort_values: dict[int, bytes] = {}
        try:
            # Validate and encode the entire actor snapshot before emitting its single
            # broadcast frame; parameter errors cannot cause a partial bus update.
            for target in targets:
                profile = self._devices.get(target.device_id)
                if profile is None:
                    raise ValueError(f"unknown FEETECH servo ID {target.device_id}")
                if target.device_id in position_values or target.device_id in effort_values:
                    raise ValueError("FEETECH motion contains duplicate servo IDs")
                if isinstance(target.payload, FeetechPacketEffort):
                    if target.device_id not in self._effort_mode_ids:
                        raise ValueError("FEETECH effort target requires Torque mode")
                    effort_values[target.device_id] = self._encode_effort(target.payload)
                elif isinstance(target.payload, FeetechPacketMotion):
                    if target.device_id in self._effort_mode_ids:
                        raise ValueError("FEETECH position target requires Position mode")
                    current_state = self._last_fast.get(target.device_id)
                    if current_state is None:
                        raise ValueError(
                            f"FEETECH servo ID {target.device_id} has no current position"
                        )
                    position_values[target.device_id] = self._encode_motion(
                        profile,
                        target.payload,
                        reference_position_rad=current_state.position_rad,
                    )
                else:
                    raise ValueError("unsupported FEETECH motion payload")
            deadline_ns = min(target.deadline_ns for target in targets)
            if position_values:
                self._sync_write(
                    self.goal_start_address,
                    position_values,
                    deadline_ns=deadline_ns,
                )
            if effort_values:
                self._sync_write(
                    self.goal_torque_address,
                    effort_values,
                    deadline_ns=deadline_ns,
                )
            # Only a successfully transmitted target is eligible for a later HOLD.
            # Raw telemetry remains diagnostic input and can never overwrite this cache.
            self._hold_positions_rad.update(
                {
                    target.device_id: target.payload.position_rad
                    for target in targets
                    if isinstance(target.payload, FeetechPacketMotion)
                }
            )
        except (FeetechPacketError, TimeoutError, OSError, ValueError) as exc:
            raise RecoverableBusError("FEETECH packet motion write failed") from exc

    def read_fast_state(
        self,
        *,
        deadline_ns: int | None = None,
    ) -> Mapping[int, FeetechPacketFastState]:
        """Sync-read the contiguous fast-state block and convert it to SI."""

        servo_ids = tuple(self._devices)
        try:
            # A timed-out sync read may leave a partial or delayed status packet in the
            # kernel buffer. Starting the next request from that suffix can mix samples
            # or repeatedly report duplicate IDs, so establish a clean frame boundary.
            self._discard_stale_input()
            self._send(
                FeetechPacketCodec.encode_sync_read(
                    servo_ids,
                    self.present_state_address,
                    self.present_state_length,
                ),
                deadline_ns=deadline_ns,
            )
            packets: dict[int, FeetechStatusPacket] = {}
            # Sync-read responses can arrive in any ID order. Collect by ID and reject
            # duplicates so the returned mapping always represents one complete cycle.
            for _ in servo_ids:
                packet = self._read_status(
                    parameter_count=self.present_state_length,
                    deadline_ns=deadline_ns,
                )
                if packet.servo_id not in self._devices or packet.servo_id in packets:
                    raise FeetechPacketError(
                        f"unexpected FEETECH sync-read servo ID {packet.servo_id}"
                    )
                packets[packet.servo_id] = packet
            timestamp_ns = self._clock_ns()
            states = {
                servo_id: self._decode_fast_state(
                    self._devices[servo_id],
                    packets[servo_id].parameters,
                    timestamp_ns,
                    status=packets[servo_id].error,
                )
                for servo_id in servo_ids
            }
        except (FeetechPacketError, TimeoutError, OSError, ValueError) as exc:
            raise RecoverableBusError("FEETECH packet state read failed") from exc
        self._last_fast = states
        return states

    def read_slow_state(
        self,
        *,
        deadline_ns: int | None = None,
    ) -> Mapping[int, FeetechPacketSlowState]:
        """Return low-rate identity and health telemetry."""

        timestamp_ns = self._clock_ns()
        return {
            servo_id: FeetechPacketSlowState(
                model=self._devices[servo_id].model,
                model_number=self._observed_model_numbers[servo_id],
                voltage_v=state.voltage_v,
                temperature_c=state.temperature_c,
                status=state.status,
                timestamp_ns=timestamp_ns,
            )
            for servo_id, state in self._last_fast.items()
        }

    def _encode_motion(
        self,
        profile: FeetechPacketServoProfile,
        motion: FeetechPacketMotion,
        *,
        reference_position_rad: float,
    ) -> bytes:
        """Encode one SI target using the nearest representable multi-turn branch."""

        goal_position = round(
            motion.position_rad * profile.counts_per_revolution / (2.0 * math.pi)
        )
        current_position = round(
            reference_position_rad
            * profile.counts_per_revolution
            / (2.0 * math.pi)
        )
        # A wrapped public angle has infinitely many raw representations. Choosing the
        # nearest branch avoids an unnecessary full revolution during teleoperation.
        position = nearest_multiturn_position_target(
            current_position,
            goal_position,
            profile.counts_per_revolution,
        )
        raw_position = FeetechPacketCodec.encode_signed_magnitude(position)
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
        if not 0 <= velocity <= 0x7FFF:
            raise ValueError("FEETECH velocity exceeds the signed-magnitude register")
        if not 0 <= acceleration <= 0xFF:
            raise ValueError("FEETECH acceleration exceeds the uint8 register")
        # Addresses 44..45 are HLS GOAL_TORQUE but SMS GOAL_TIME. HLS position mode
        # needs a nonzero torque value to hold a target. The raw value is a frozen
        # model profile captured before commands can contaminate SRAM; it remains
        # distinct from an SI effort limit and the torque-limit register at address 48.
        goal_torque = profile.default_goal_torque_raw
        if profile.family == FeetechPacketFamily.HLS:
            if goal_torque is None:
                raise RuntimeError("FEETECH HLS profile has no goal torque")
            auxiliary = FeetechPacketCodec.word(goal_torque)
        else:
            auxiliary = b"\x00\x00"
        return (
            bytes((acceleration,))
            + FeetechPacketCodec.word(raw_position)
            + auxiliary
            + FeetechPacketCodec.word(velocity)
        )

    @staticmethod
    def _encode_effort(effort: FeetechPacketEffort) -> bytes:
        """Encode an HLS signed-magnitude current target for GOAL_TORQUE."""

        return FeetechPacketCodec.word(
            FeetechPacketCodec.encode_signed_magnitude(effort.current_raw)
        )

    def _best_effort_zero_and_disable(self, servo_ids: Sequence[int]) -> None:
        """Attempt the strongest Torque-mode containment without masking its caller."""

        servo_ids = tuple(servo_ids)
        if not servo_ids:
            return
        try:
            self._sync_write(
                self.goal_torque_address,
                {servo_id: b"\x00\x00" for servo_id in servo_ids},
            )
        except BaseException:
            pass
        try:
            self._sync_write(
                self.torque_enable_address,
                {servo_id: b"\x00" for servo_id in servo_ids},
            )
        except BaseException:
            pass
        self._enabled_ids.difference_update(servo_ids)

    def _attempt_safety_writes(
        self,
        operations: Sequence[tuple[int, Mapping[int, bytes]]],
    ) -> BaseException | None:
        """Attempt every containment write and return the first transport failure."""

        first_error = None
        for address, values in operations:
            try:
                self._sync_write(address, values)
            except BaseException as exc:
                first_error = first_error or exc
        return first_error

    @staticmethod
    def _nearest_multiturn_position_target(
        current_position: int,
        goal_position: int,
        position_period: int,
    ) -> int:
        return nearest_multiturn_position_target(
            current_position,
            goal_position,
            position_period,
        )

    def _decode_fast_state(
        self,
        profile: FeetechPacketServoProfile,
        data: bytes,
        timestamp_ns: int,
        *,
        status: int,
    ) -> FeetechPacketFastState:
        """Decode the contiguous packet-family telemetry block into SI units."""

        position = FeetechPacketCodec.decode_signed_magnitude(
            FeetechPacketCodec.decode_word(data[0:2])
        )
        velocity = FeetechPacketCodec.decode_signed_magnitude(
            FeetechPacketCodec.decode_word(data[2:4])
        )
        load = FeetechPacketCodec.decode_signed_magnitude(
            FeetechPacketCodec.decode_word(data[4:6]), sign_bit=10
        )
        current = FeetechPacketCodec.decode_signed_magnitude(
            FeetechPacketCodec.decode_word(data[13:15])
        )
        return FeetechPacketFastState(
            position_rad=position * 2.0 * math.pi / profile.counts_per_revolution,
            velocity_rad_s=velocity * profile.velocity_unit_rad_s,
            current_a=current * profile.current_unit_a,
            load_ratio=load / 1000.0,
            voltage_v=data[6] / 10.0,
            temperature_c=data[7],
            # A complete state response remains usable as telemetry even when its
            # packet error byte reports a device alarm. Runtime interprets that alarm
            # and schedules the appropriate disable action; malformed packets and
            # transport errors still fail before reaching this point.
            status=status,
            timestamp_ns=timestamp_ns,
        )

    def _read_register(
        self,
        servo_id: int,
        address: int,
        length: int,
        *,
        deadline_ns: int | None = None,
    ) -> bytes:
        """Perform one addressed read and validate its status response."""

        self._discard_stale_input()
        self._send(
            FeetechPacketCodec.encode_read(servo_id, address, length),
            deadline_ns=deadline_ns,
        )
        packet = self._read_status(
            expected_servo_id=servo_id,
            parameter_count=length,
            deadline_ns=deadline_ns,
        )
        self._raise_status_error(packet)
        return packet.parameters

    def _discard_stale_input(self) -> None:
        """Discard bytes left by an abandoned response before sending a new request."""

        discard_input = getattr(self._transport, "discard_input", None)
        if callable(discard_input):
            discard_input()

    def _sync_write(
        self,
        address: int,
        values: Mapping[int, bytes],
        *,
        deadline_ns: int | None = None,
    ) -> None:
        """Send one broadcast write; the protocol provides no per-servo ACK."""

        self._send(
            FeetechPacketCodec.encode_sync_write(address, values),
            deadline_ns=deadline_ns,
        )

    def _send(self, frame: bytes, *, deadline_ns: int | None = None) -> None:
        self._transport.write(frame, deadline_ns=self._deadline(deadline_ns))

    def _read_status(
        self,
        *,
        expected_servo_id: int | None = None,
        parameter_count: int | None = None,
        deadline_ns: int | None = None,
    ) -> FeetechStatusPacket:
        """Resynchronize on the header, then read exactly one bounded status frame."""

        deadline_ns = self._deadline(deadline_ns)
        header = bytearray()
        while bytes(header) != FeetechPacketCodec.header:
            header.extend(self._transport.read_exact(1, deadline_ns=deadline_ns))
            if len(header) > 2:
                del header[0]
        identifier_and_length = self._transport.read_exact(2, deadline_ns=deadline_ns)
        length = identifier_and_length[1]
        if length < 2:
            raise FeetechPacketError("FEETECH status length is smaller than two")
        suffix = self._transport.read_exact(length, deadline_ns=deadline_ns)
        return FeetechPacketCodec.decode_status(
            bytes(header) + identifier_and_length + suffix,
            expected_servo_id=expected_servo_id,
            expected_parameter_count=parameter_count,
        )

    def _deadline(self, deadline_ns: int | None) -> int:
        operation_deadline_ns = self._clock_ns() + self._operation_timeout_ns
        if deadline_ns is None:
            return operation_deadline_ns
        if type(deadline_ns) is not int or deadline_ns < 0:
            raise ValueError("FEETECH packet deadline must be a non-negative integer")
        return min(operation_deadline_ns, deadline_ns)

    @staticmethod
    def _raise_status_error(packet: FeetechStatusPacket) -> None:
        if packet.error:
            raise FeetechPacketError(
                f"FEETECH servo {packet.servo_id} reported error 0x{packet.error:02x}"
            )


__all__ = [
    "FeetechPacketBusProtocol",
    "FeetechPacketFastState",
    "FeetechPacketEffort",
    "FeetechPacketMotion",
    "FeetechPacketSlowState",
    "nearest_multiturn_position_target",
]
