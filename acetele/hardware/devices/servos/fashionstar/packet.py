"""Actor-owned FashionStar RS485 protocol with firmware capability negotiation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence

from acetele.hardware.buses import (
    MotionEnvelope,
    RecoverableBusError,
    SerialTransport,
    resolve_device_enable_request,
)
from acetele.hardware.devices.servos.fashionstar.codec import (
    FashionStarCommand,
    FashionStarMonitorState,
    FashionStarPacket,
    FashionStarPacketCodec,
    FashionStarProtocolError,
    FashionStarStopMode,
)
from acetele.hardware.devices.servos.fashionstar.profile import FashionStarServoProfile


@dataclass(frozen=True)
class FashionStarMotion:
    """Multi-turn SI target and bounded vendor motion parameters."""

    position_rad: float
    interval_ms: int = 10
    acceleration_ms: int = 20
    deceleration_ms: int = 20
    power_mw: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.position_rad):
            raise ValueError("FashionStar motion position must be finite")
        raw_position = round(math.degrees(self.position_rad) * 10.0)
        if not -3_686_400 <= raw_position <= 3_686_400:
            raise ValueError("FashionStar multi-turn position exceeds the protocol range")
        for field_name, maximum in (
            ("interval_ms", 0xFFFFFFFF),
            ("acceleration_ms", 0xFFFF),
            ("deceleration_ms", 0xFFFF),
            ("power_mw", 0xFFFF),
        ):
            value = getattr(self, field_name)
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError(
                    f"FashionStar motion {field_name} must be an integer in [0, {maximum}]"
                )


class FashionStarBusProtocol:
    """FashionStar packet protocol with Actor-owned serial I/O and deadlines."""

    def __init__(
        self,
        transport: SerialTransport,
        devices: Mapping[int, FashionStarServoProfile],
        *,
        firmware_versions: Optional[Mapping[int, int]] = None,
        operation_timeout_s: float = 0.05,
        clock_ns=time.monotonic_ns,
    ) -> None:
        devices = dict(devices)
        if not devices:
            raise ValueError("FashionStar bus requires at least one servo")
        for servo_id in devices:
            FashionStarPacketCodec.validate_servo_id(servo_id)
        firmware_versions = dict(firmware_versions or {})
        if set(firmware_versions) - set(devices):
            raise ValueError("FashionStar firmware map contains unknown servo IDs")
        if any(type(version) is not int or version <= 0 for version in firmware_versions.values()):
            raise ValueError("FashionStar firmware versions must be positive integers")
        if operation_timeout_s <= 0.0:
            raise ValueError("FashionStar operation timeout must be positive")
        self._transport = transport
        self._devices = devices
        self._firmware_versions = firmware_versions
        self._observed_firmware_versions: dict[int, int] = {}
        self._operation_timeout_ns = round(operation_timeout_s * 1e9)
        self._clock_ns = clock_ns
        self._last_command_ns: Optional[int] = None
        self._enabled_ids: set[int] = set()
        self._sync_capable = False
        self._last_states: dict[int, FashionStarMonitorState] = {}

    @property
    def operation_timeout_ns(self) -> int:
        """Return the deadline budget for one complete packet transaction."""

        return self._operation_timeout_ns

    @property
    def synchronized(self) -> bool:
        """Return whether every device passed firmware and sync-monitor probing."""

        return self._sync_capable

    def connect(self) -> None:
        """Verify IDs and firmware, then negotiate synchronous monitoring."""

        self._last_states.clear()
        self._observed_firmware_versions.clear()
        self._last_command_ns = None
        self._enabled_ids.clear()
        self._sync_capable = False
        self._transport.connect()
        try:
            # Ping, response mode, and firmware are negotiated before enable. In
            # particular, response_switch must permit latest-command interruption.
            for servo_id in self._devices:
                response = self._request_response(
                    FashionStarPacketCodec.encode_request(
                        FashionStarCommand.PING,
                        bytes((servo_id,)),
                    ),
                    expected_command=FashionStarCommand.PING,
                )
                if response.parameters != bytes((servo_id,)):
                    raise RuntimeError(
                        f"FashionStar ping response does not match servo ID {servo_id}"
                    )
                response_switch = FashionStarPacketCodec.decode_read_data(
                    self._request_response(
                        FashionStarPacketCodec.encode_read_data(
                            servo_id,
                            FashionStarPacketCodec.response_switch_data_id,
                        ),
                        expected_command=FashionStarCommand.READ_DATA,
                    ),
                    servo_id=servo_id,
                    data_id=FashionStarPacketCodec.response_switch_data_id,
                    size=1,
                )[0]
                if response_switch != 0:
                    raise RuntimeError(
                        f"FashionStar servo {servo_id} response_switch must be 0 for "
                        "interruptible latest-command control; configure it with the "
                        "official servo tool before connecting"
                    )
                firmware_version = int.from_bytes(
                    FashionStarPacketCodec.decode_read_data(
                        self._request_response(
                            FashionStarPacketCodec.encode_read_data(
                                servo_id,
                                FashionStarPacketCodec.firmware_version_data_id,
                            ),
                            expected_command=FashionStarCommand.READ_DATA,
                        ),
                        servo_id=servo_id,
                        data_id=FashionStarPacketCodec.firmware_version_data_id,
                        size=2,
                    ),
                    byteorder="little",
                    signed=False,
                )
                if firmware_version <= 0:
                    raise RuntimeError(
                        f"FashionStar servo {servo_id} reported invalid firmware version "
                        f"{firmware_version}"
                    )
                expected_firmware = self._firmware_versions.get(servo_id)
                if (
                    expected_firmware is not None
                    and firmware_version != expected_firmware
                ):
                    raise RuntimeError(
                        f"FashionStar servo {servo_id} reports firmware "
                        f"{firmware_version}, expected {expected_firmware}"
                    )
                self._observed_firmware_versions[servo_id] = firmware_version
            # Firmware claims are insufficient on their own; probe the actual sync
            # monitor path and fall back to bounded unicast when it is unavailable.
            self._sync_capable = self._probe_sync_monitor()
            self._stop_all(FashionStarStopMode.RELEASE)
            self._enabled_ids.clear()
        except BaseException:
            self._observed_firmware_versions.clear()
            self._transport.disconnect()
            raise

    def cancel(self) -> None:
        """Interrupt pending transport I/O for bounded shutdown."""

        self._transport.cancel()

    def disconnect(self) -> None:
        """Release transport and clear negotiated capabilities."""

        self._enabled_ids.clear()
        self._sync_capable = False
        self._observed_firmware_versions.clear()
        self._transport.disconnect()

    def execute_safety(self, label: str, payload) -> object:
        """Execute hold, damping, release, enable, or calibration barriers."""

        if label == "hold":
            self._stop_devices(self._enabled_ids, FashionStarStopMode.HOLD)
            return True
        if label == "emergency_stop":
            self._enabled_ids.clear()
            self._stop_all(FashionStarStopMode.RELEASE)
            return True
        if label == "set_enabled":
            enabled, servo_ids = resolve_device_enable_request(
                payload,
                self._devices,
                context="FashionStar",
            )
            if not enabled:
                self._enabled_ids.difference_update(servo_ids)
                self._stop_devices(servo_ids, FashionStarStopMode.RELEASE)
            else:
                self._stop_devices(servo_ids, FashionStarStopMode.HOLD)
                self._enabled_ids.update(servo_ids)
            return True
        if label in ("set_origin", "reset_multi_turn"):
            # These operations change the persistent angle frame. The software latch
            # prevents any queued motion from racing calibration.
            if self._enabled_ids:
                raise RuntimeError(
                    f"FashionStar {label} requires the software-disabled calibration state"
                )
            servo_ids = self._safety_servo_ids(payload)
            command = (
                FashionStarCommand.SET_ORIGIN
                if label == "set_origin"
                else FashionStarCommand.RESET_MULTI_TURN_ANGLE
            )
            for servo_id in servo_ids:
                parameters = bytes((servo_id, 0)) if label == "set_origin" else bytes((servo_id,))
                self._send(FashionStarPacketCodec.encode_request(command, parameters))
            return True
        raise ValueError(f"unsupported FashionStar safety task '{label}'")

    def write_motion(self, targets: Sequence[MotionEnvelope]) -> None:
        """Use a sync frame when negotiated, otherwise bounded per-servo writes."""

        commands: list[tuple[int, FashionStarMotion]] = []
        # Validate the complete mailbox snapshot before selecting a wire strategy.
        for target in targets:
            if target.device_id not in self._enabled_ids:
                raise RecoverableBusError(
                    f"FashionStar servo ID {target.device_id} is software-disabled"
                )
            if target.device_id not in self._devices:
                raise RecoverableBusError(f"unknown FashionStar servo ID {target.device_id}")
            if not isinstance(target.payload, FashionStarMotion):
                raise RecoverableBusError("FashionStar payload must be FashionStarMotion")
            commands.append((target.device_id, target.payload))
        try:
            deadline_ns = min(target.deadline_ns for target in targets)
            if len(commands) > 1 and self.synchronized:
                # Newer firmware provides one batch frame. Older firmware is handled
                # below with deadline-bounded unicast writes.
                self._send(
                    FashionStarPacketCodec.encode_sync_multi_turn_positions(
                        (
                            servo_id,
                            motion.position_rad,
                            motion.interval_ms,
                            motion.acceleration_ms,
                            motion.deceleration_ms,
                            motion.power_mw,
                        )
                        for servo_id, motion in commands
                    ),
                    deadline_ns=deadline_ns,
                )
            else:
                for servo_id, motion in commands:
                    self._send(
                        FashionStarPacketCodec.encode_multi_turn_position(
                            servo_id,
                            motion.position_rad,
                            interval_ms=motion.interval_ms,
                            power_mw=motion.power_mw,
                        ),
                        deadline_ns=deadline_ns,
                    )
        except (FashionStarProtocolError, TimeoutError, OSError, ValueError) as exc:
            raise RecoverableBusError("FashionStar motion write failed") from exc

    def read_fast_state(
        self,
        *,
        deadline_ns: int | None = None,
    ) -> Mapping[int, FashionStarMonitorState]:
        """Read monitor state through the negotiated or compatibility path."""

        states: dict[int, FashionStarMonitorState] = {}
        try:
            if self.synchronized and len(self._devices) > 1:
                # Responses arrive one packet per servo in unspecified order; collect
                # by ID and reject duplicate or foreign packets.
                self._send(
                    FashionStarPacketCodec.encode_sync_monitor(self._devices),
                    deadline_ns=deadline_ns,
                )
                for _ in self._devices:
                    state = FashionStarPacketCodec.decode_monitor(
                        self._read_packet(deadline_ns=deadline_ns)
                    )
                    if state.servo_id not in self._devices or state.servo_id in states:
                        raise FashionStarProtocolError(
                            f"unexpected FashionStar monitor servo ID {state.servo_id}"
                        )
                    states[state.servo_id] = state
            else:
                for servo_id in self._devices:
                    packet = self._request_response(
                        FashionStarPacketCodec.encode_query_monitor(servo_id),
                        expected_command=FashionStarCommand.QUERY_MONITOR,
                        deadline_ns=deadline_ns,
                    )
                    state = FashionStarPacketCodec.decode_monitor(packet)
                    if state.servo_id != servo_id:
                        raise FashionStarProtocolError(
                            f"FashionStar monitor ID {state.servo_id} does not match {servo_id}"
                        )
                    states[servo_id] = state
        except (FashionStarProtocolError, TimeoutError, OSError, ValueError) as exc:
            raise RecoverableBusError("FashionStar monitor read failed") from exc
        timestamp_ns = self._clock_ns()
        states = {
            servo_id: replace(state, timestamp_ns=timestamp_ns)
            for servo_id, state in states.items()
        }
        self._last_states = states
        return states

    def read_slow_state(
        self,
        *,
        deadline_ns: int | None = None,
    ) -> Mapping[int, dict[str, float | int]]:
        """Return cached firmware capabilities without fast-path traffic."""

        return {
            servo_id: {
                "voltage_v": state.voltage_v,
                "current_a": state.current_a,
                "power_w": state.power_w,
                "temperature_raw": state.temperature_raw,
                "status": state.status,
                "firmware_version": self._observed_firmware_versions.get(servo_id, 0),
            }
            for servo_id, state in self._last_states.items()
        }

    def _stop_all(self, mode: FashionStarStopMode) -> None:
        """Apply a stop behavior to every configured servo in deterministic order."""

        self._stop_devices(self._devices, mode)

    def _stop_devices(
        self,
        servo_ids: Iterable[int],
        mode: FashionStarStopMode,
    ) -> None:
        """Apply one stop behavior to a selected deterministic device set."""

        for servo_id in sorted(servo_ids):
            self._send(FashionStarPacketCodec.encode_stop(servo_id, mode))

    def _probe_sync_monitor(self) -> bool:
        """Verify synchronous monitor support without trusting firmware metadata."""

        if len(self._devices) < 2:
            return False
        try:
            self._send(FashionStarPacketCodec.encode_sync_monitor(self._devices))
            observed = {
                FashionStarPacketCodec.decode_monitor(self._read_packet()).servo_id
                for _ in self._devices
            }
            return observed == set(self._devices)
        except (FashionStarProtocolError, TimeoutError, OSError, ValueError):
            # A failed probe may leave a partial packet in the receive buffer. Flush it
            # before compatibility-mode unicast requests begin.
            discard_input = getattr(self._transport, "discard_input", None)
            if callable(discard_input):
                discard_input()
            return False

    def _send(self, frame: bytes, *, deadline_ns: int | None = None) -> None:
        """Respect the strictest active profile's inter-command interval."""

        self._wait_command_interval()
        self._transport.write(frame, deadline_ns=self._deadline(deadline_ns))
        self._last_command_ns = self._clock_ns()

    def _request_response(
        self,
        frame: bytes,
        *,
        expected_command: FashionStarCommand,
        deadline_ns: int | None = None,
    ) -> FashionStarPacket:
        """Send one request and reject a response with the wrong command code."""

        self._send(frame, deadline_ns=deadline_ns)
        packet = self._read_packet(deadline_ns=deadline_ns)
        if packet.command != expected_command:
            raise FashionStarProtocolError(
                f"FashionStar response {packet.command.name} does not match {expected_command.name}"
            )
        return packet

    def _read_packet(self, *, deadline_ns: int | None = None) -> FashionStarPacket:
        """Resynchronize on the header and read one checksum-bounded response."""

        deadline_ns = self._deadline(deadline_ns)
        header = bytearray()
        while bytes(header) != FashionStarPacketCodec.response_header:
            header.extend(self._transport.read_exact(1, deadline_ns=deadline_ns))
            if len(header) > 2:
                del header[0]
        code_and_size = self._transport.read_exact(2, deadline_ns=deadline_ns)
        payload_size = code_and_size[1]
        suffix = self._transport.read_exact(payload_size + 1, deadline_ns=deadline_ns)
        return FashionStarPacketCodec.decode_response(bytes(header) + code_and_size + suffix)

    def _deadline(self, deadline_ns: int | None) -> int:
        """Intersect the caller's cycle deadline with the protocol timeout."""

        operation_deadline_ns = self._clock_ns() + self._operation_timeout_ns
        if deadline_ns is None:
            return operation_deadline_ns
        if type(deadline_ns) is not int or deadline_ns < 0:
            raise ValueError("FashionStar deadline must be a non-negative integer")
        return min(operation_deadline_ns, deadline_ns)

    def _wait_command_interval(self) -> None:
        """Sleep only for the remaining model-required frame interval."""

        if self._last_command_ns is None:
            return
        minimum_s = max(
            profile.minimum_command_interval_s for profile in self._devices.values()
        )
        remaining_ns = round(minimum_s * 1e9) - (self._clock_ns() - self._last_command_ns)
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1e9)

    def _safety_servo_ids(self, payload) -> tuple[int, ...]:
        """Normalize IDs used by persistent calibration actions."""

        servo_ids = tuple(payload)
        if not servo_ids or any(
            type(servo_id) is not int or servo_id not in self._devices
            for servo_id in servo_ids
        ):
            raise ValueError("FashionStar safety payload contains unknown servo IDs")
        return servo_ids


__all__ = ["FashionStarBusProtocol", "FashionStarMotion"]
