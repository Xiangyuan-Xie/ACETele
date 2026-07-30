"""Pure FashionStar UART/RS485 packet codec based on the documented wire format."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


class FashionStarProtocolError(ValueError):
    """A FashionStar response violates framing or request correlation."""


class FashionStarCommand(IntEnum):
    """Supported FashionStar command identifiers."""

    PING = 0x01
    READ_DATA = 0x03
    WRITE_DATA = 0x04
    SET_ANGLE = 0x08
    SET_DAMPING = 0x09
    QUERY_ANGLE = 0x0A
    SET_ANGLE_BY_INTERVAL = 0x0B
    SET_ANGLE_BY_VELOCITY = 0x0C
    SET_MULTI_TURN_ANGLE = 0x0D
    SET_MULTI_TURN_ANGLE_BY_INTERVAL = 0x0E
    SET_MULTI_TURN_ANGLE_BY_VELOCITY = 0x0F
    QUERY_MULTI_TURN_ANGLE = 0x10
    RESET_MULTI_TURN_ANGLE = 0x11
    QUERY_MONITOR = 0x16
    SET_ORIGIN = 0x17
    STOP_CONTROL = 0x18
    SYNC = 0x19


class FashionStarStopMode(IntEnum):
    """Vendor stop modes ordered from released to actively damped."""

    RELEASE = 0x10
    HOLD = 0x11
    DAMPING = 0x12


@dataclass(frozen=True)
class FashionStarPacket:
    """Decoded FashionStar response payload."""

    command: FashionStarCommand
    parameters: bytes


@dataclass(frozen=True)
class FashionStarMonitorState:
    """Monitor response converted to physical units."""

    servo_id: int
    voltage_v: float
    current_a: float
    power_w: float
    temperature_raw: int
    status: int
    position_rad: float
    turns: int
    timestamp_ns: int = 0


class FashionStarPacketCodec:
    """Stateless frame conversion; the bus protocol owns deadlines and retries."""

    request_header = b"\x12\x4c"
    response_header = b"\x05\x1c"
    max_parameters = 0xFF
    response_switch_data_id = 33
    firmware_version_data_id = 6

    @classmethod
    def validate_servo_id(cls, servo_id: int) -> None:
        """Validate a unicast FashionStar bus address."""

        cls._servo_id(servo_id)

    @classmethod
    def encode_request(cls, command: FashionStarCommand, parameters: bytes = b"") -> bytes:
        """Encode one checksummed request frame."""

        command = cls._command(command)
        if not isinstance(parameters, bytes):
            raise ValueError("FashionStar parameters must be bytes")
        if len(parameters) > cls.max_parameters:
            raise ValueError("FashionStar parameters cannot exceed 255 bytes")
        body = cls.request_header + bytes((command, len(parameters))) + parameters
        return body + bytes((sum(body) & 0xFF,))

    @classmethod
    def encode_read_data(cls, servo_id: int, data_id: int) -> bytes:
        """Encode a persistent-data read request."""

        cls._servo_id(servo_id)
        cls._uint8(data_id, "data_id")
        return cls.encode_request(
            FashionStarCommand.READ_DATA,
            bytes((servo_id, data_id)),
        )

    @classmethod
    def decode_read_data(
        cls,
        packet: FashionStarPacket,
        *,
        servo_id: int,
        data_id: int,
        size: int,
    ) -> bytes:
        """Correlate a data response with its requested servo and data ID."""

        cls._servo_id(servo_id)
        cls._uint8(data_id, "data_id")
        if type(size) is not int or size <= 0:
            raise ValueError("FashionStar data size must be a positive integer")
        if packet.command != FashionStarCommand.READ_DATA:
            raise FashionStarProtocolError("FashionStar packet is not a data response")
        expected_prefix = bytes((servo_id, data_id))
        if (
            len(packet.parameters) != len(expected_prefix) + size
            or packet.parameters[:2] != expected_prefix
        ):
            raise FashionStarProtocolError(
                "FashionStar data response does not match the requested servo and data ID"
            )
        return packet.parameters[2:]

    @classmethod
    def decode_response(cls, frame: bytes) -> FashionStarPacket:
        """Validate and decode one complete response frame."""

        if not isinstance(frame, bytes) or len(frame) < 5:
            raise FashionStarProtocolError("FashionStar response is too short")
        if frame[:2] != cls.response_header:
            raise FashionStarProtocolError("FashionStar response header is invalid")
        size = frame[3]
        if len(frame) != size + 5:
            raise FashionStarProtocolError("FashionStar response length is invalid")
        if frame[-1] != sum(frame[:-1]) & 0xFF:
            raise FashionStarProtocolError("FashionStar response checksum is invalid")
        try:
            command = FashionStarCommand(frame[2])
        except ValueError as exc:
            raise FashionStarProtocolError(
                f"FashionStar response command 0x{frame[2]:02x} is unknown"
            ) from exc
        return FashionStarPacket(command, frame[4:-1])

    @classmethod
    def encode_single_turn_position(
        cls,
        servo_id: int,
        position_rad: float,
        *,
        interval_ms: int,
        power_mw: int = 0,
    ) -> bytes:
        """Encode a bounded single-turn position command."""

        cls._servo_id(servo_id)
        raw_position = cls._angle_raw(position_rad, multi_turn=False)
        cls._uint16(interval_ms, "interval_ms")
        cls._uint16(power_mw, "power_mw")
        parameters = struct.pack("<BhHH", servo_id, raw_position, interval_ms, power_mw)
        return cls.encode_request(FashionStarCommand.SET_ANGLE, parameters)

    @classmethod
    def encode_multi_turn_position(
        cls,
        servo_id: int,
        position_rad: float,
        *,
        interval_ms: int,
        power_mw: int = 0,
    ) -> bytes:
        """Encode a continuous multi-turn position command."""

        cls._servo_id(servo_id)
        raw_position = cls._angle_raw(position_rad, multi_turn=True)
        cls._uint32(interval_ms, "interval_ms")
        cls._uint16(power_mw, "power_mw")
        parameters = struct.pack("<BiIH", servo_id, raw_position, interval_ms, power_mw)
        return cls.encode_request(FashionStarCommand.SET_MULTI_TURN_ANGLE, parameters)

    @classmethod
    def encode_sync_multi_turn_positions(
        cls,
        commands: Iterable[tuple[int, float, int, int, int, int]],
    ) -> bytes:
        """Encode same-cycle multi-turn targets for supported firmware."""

        encoded = []
        for servo_id, position_rad, interval_ms, acceleration_ms, deceleration_ms, power_mw in commands:
            cls._servo_id(servo_id)
            cls._uint32(interval_ms, "interval_ms")
            cls._uint16(acceleration_ms, "acceleration_ms")
            cls._uint16(deceleration_ms, "deceleration_ms")
            cls._uint16(power_mw, "power_mw")
            encoded.append(
                struct.pack(
                    "<BiIHHH",
                    servo_id,
                    cls._angle_raw(position_rad, multi_turn=True),
                    interval_ms,
                    acceleration_ms,
                    deceleration_ms,
                    power_mw,
                )
            )
        if not encoded:
            raise ValueError("FashionStar sync position command cannot be empty")
        item_size = struct.calcsize("<BiIHHH")
        maximum_items = (cls.max_parameters - 3) // item_size
        if len(encoded) > maximum_items:
            raise ValueError(
                f"FashionStar sync position command cannot exceed {maximum_items} servos"
            )
        parameters = bytes(
            (
                FashionStarCommand.SET_MULTI_TURN_ANGLE_BY_INTERVAL,
                item_size,
                len(encoded),
            )
        ) + b"".join(encoded)
        return cls.encode_request(FashionStarCommand.SYNC, parameters)

    @classmethod
    def encode_sync_monitor(cls, servo_ids: Iterable[int]) -> bytes:
        """Encode one synchronous monitor request for unique IDs."""

        servo_ids = tuple(servo_ids)
        maximum_items = cls.max_parameters - 3
        if not servo_ids or len(servo_ids) > maximum_items:
            raise ValueError(
                f"FashionStar sync monitor requires 1 to {maximum_items} servo IDs"
            )
        for servo_id in servo_ids:
            cls._servo_id(servo_id)
        parameters = bytes((FashionStarCommand.QUERY_MONITOR, 1, len(servo_ids), *servo_ids))
        return cls.encode_request(FashionStarCommand.SYNC, parameters)

    @classmethod
    def encode_stop(
        cls,
        servo_id: int,
        mode: FashionStarStopMode,
        *,
        power_mw: int = 0,
    ) -> bytes:
        """Encode hold, damping, or torque release for selected servos."""

        cls._servo_id(servo_id)
        if not isinstance(mode, FashionStarStopMode):
            raise ValueError("FashionStar stop mode must be a FashionStarStopMode")
        cls._uint16(power_mw, "power_mw")
        return cls.encode_request(
            FashionStarCommand.STOP_CONTROL,
            struct.pack("<BBH", servo_id, mode, power_mw),
        )

    @classmethod
    def encode_query_monitor(cls, servo_id: int) -> bytes:
        """Encode one compatibility monitor request."""

        cls._servo_id(servo_id)
        return cls.encode_request(FashionStarCommand.QUERY_MONITOR, bytes((servo_id,)))

    @classmethod
    def decode_monitor(cls, packet: FashionStarPacket) -> FashionStarMonitorState:
        """Decode monitor fields and convert documented units to SI."""

        if packet.command != FashionStarCommand.QUERY_MONITOR:
            raise FashionStarProtocolError("FashionStar packet is not a monitor response")
        if len(packet.parameters) != struct.calcsize("<BHHHHBih"):
            raise FashionStarProtocolError("FashionStar monitor payload length is invalid")
        servo_id, voltage, current, power, temperature, status, angle, turns = struct.unpack(
            "<BHHHHBih", packet.parameters
        )
        if angle == -235_929_599 and turns == 0:
            raise FashionStarProtocolError("FashionStar monitor returned its invalid-angle sentinel")
        return FashionStarMonitorState(
            servo_id=servo_id,
            voltage_v=voltage / 1000.0,
            current_a=current / 1000.0,
            power_w=power / 1000.0,
            temperature_raw=temperature,
            status=status,
            position_rad=math.radians(angle / 10.0),
            turns=turns,
        )

    @staticmethod
    def _command(command: FashionStarCommand) -> FashionStarCommand:
        if not isinstance(command, FashionStarCommand):
            raise ValueError("FashionStar command must be a FashionStarCommand")
        return command

    @staticmethod
    def _servo_id(servo_id: int) -> None:
        if type(servo_id) is not int or not 0 <= servo_id <= 0xFE:
            raise ValueError("FashionStar servo ID must be an integer in [0, 254]")

    @staticmethod
    def _uint8(value: int, name: str) -> None:
        if type(value) is not int or not 0 <= value <= 0xFF:
            raise ValueError(f"FashionStar {name} must be an integer in [0, 255]")

    @staticmethod
    def _angle_raw(position_rad: float, *, multi_turn: bool) -> int:
        if not math.isfinite(position_rad):
            raise ValueError("FashionStar position must be finite")
        raw = round(math.degrees(position_rad) * 10.0)
        lower, upper = (-3_686_400, 3_686_400) if multi_turn else (-1_800, 1_800)
        if not lower <= raw <= upper:
            kind = "multi-turn" if multi_turn else "single-turn"
            raise ValueError(f"FashionStar {kind} position is outside [{lower}, {upper}] raw")
        return raw

    @staticmethod
    def _uint16(value: int, name: str) -> None:
        if type(value) is not int or not 0 <= value <= 0xFFFF:
            raise ValueError(f"FashionStar {name} must be an integer in [0, 65535]")

    @staticmethod
    def _uint32(value: int, name: str) -> None:
        if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
            raise ValueError(f"FashionStar {name} must be an integer in [0, 4294967295]")


__all__ = [
    "FashionStarCommand",
    "FashionStarMonitorState",
    "FashionStarPacket",
    "FashionStarPacketCodec",
    "FashionStarProtocolError",
    "FashionStarStopMode",
]
