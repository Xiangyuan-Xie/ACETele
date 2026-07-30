"""FEETECH packet codec with strict frame, ID, and register-value validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Sequence


class FeetechPacketError(ValueError):
    """A FEETECH status packet is malformed or reports a device error."""


class FeetechInstruction(IntEnum):
    """Instructions shared by the supported HLS and SMS packet families."""

    PING = 0x01
    READ = 0x02
    WRITE = 0x03
    REG_WRITE = 0x04
    ACTION = 0x05
    RESET = 0x0A
    OFFSET_CALIBRATION = 0x0B
    SYNC_READ = 0x82
    SYNC_WRITE = 0x83


@dataclass(frozen=True)
class FeetechStatusPacket:
    """Decoded status frame returned by one servo."""

    servo_id: int
    error: int
    parameters: bytes


class FeetechPacketCodec:
    """Pure packet encoder/decoder with no serial ownership or retry policy."""

    header = b"\xff\xff"
    broadcast_id = 0xFE
    max_servo_id = 0xFC
    max_packet_bytes = 250

    @classmethod
    def encode_instruction(
        cls,
        servo_id: int,
        instruction: FeetechInstruction,
        parameters: bytes = b"",
        *,
        allow_broadcast: bool = False,
    ) -> bytes:
        """Encode one validated FEETECH instruction frame."""

        cls.validate_servo_id(servo_id, allow_broadcast=allow_broadcast)
        if not isinstance(instruction, FeetechInstruction):
            raise ValueError("FEETECH instruction must be a FeetechInstruction")
        if not isinstance(parameters, bytes):
            raise ValueError("FEETECH instruction parameters must be bytes")
        length = len(parameters) + 2
        if length > 0xFF:
            raise ValueError("FEETECH instruction parameters are too long")
        body = bytes((servo_id, length, instruction)) + parameters
        frame = cls.header + body + bytes((cls.checksum(body),))
        if len(frame) > cls.max_packet_bytes:
            raise ValueError("FEETECH instruction exceeds the 250-byte packet limit")
        return frame

    @classmethod
    def encode_ping(cls, servo_id: int) -> bytes:
        """Encode a unicast identity ping."""

        return cls.encode_instruction(servo_id, FeetechInstruction.PING)

    @classmethod
    def encode_read(cls, servo_id: int, address: int, length: int) -> bytes:
        """Encode a contiguous register read."""

        cls._byte(address, "read address")
        cls._byte(length, "read length", minimum=1)
        return cls.encode_instruction(
            servo_id,
            FeetechInstruction.READ,
            bytes((address, length)),
        )

    @classmethod
    def encode_write(cls, servo_id: int, address: int, data: bytes) -> bytes:
        """Encode a unicast contiguous register write."""

        cls._byte(address, "write address")
        if not isinstance(data, bytes) or not data:
            raise ValueError("FEETECH write data must be non-empty bytes")
        return cls.encode_instruction(
            servo_id,
            FeetechInstruction.WRITE,
            bytes((address,)) + data,
        )

    @classmethod
    def encode_sync_read(
        cls,
        servo_ids: Sequence[int],
        address: int,
        length: int,
    ) -> bytes:
        """Encode one broadcast read for unique device IDs."""

        servo_ids = tuple(servo_ids)
        if not servo_ids or len(set(servo_ids)) != len(servo_ids):
            raise ValueError("FEETECH sync read requires unique servo IDs")
        for servo_id in servo_ids:
            cls.validate_servo_id(servo_id)
        cls._byte(address, "sync read address")
        cls._byte(length, "sync read length", minimum=1)
        return cls.encode_instruction(
            cls.broadcast_id,
            FeetechInstruction.SYNC_READ,
            bytes((address, length, *servo_ids)),
            allow_broadcast=True,
        )

    @classmethod
    def encode_sync_write(
        cls,
        address: int,
        values: Mapping[int, bytes],
    ) -> bytes:
        """Encode equal-width per-device values as one broadcast write."""

        values = dict(values)
        if not values:
            raise ValueError("FEETECH sync write requires at least one servo")
        if any(not isinstance(value, bytes) for value in values.values()):
            raise ValueError("FEETECH sync write values must be bytes")
        cls._byte(address, "sync write address")
        lengths = {len(value) for value in values.values()}
        if len(lengths) != 1:
            raise ValueError("FEETECH sync write values must be equal-length bytes")
        data_length = next(iter(lengths))
        cls._byte(data_length, "sync write data length", minimum=1)
        parameters = bytearray((address, data_length))
        for servo_id, data in values.items():
            cls.validate_servo_id(servo_id)
            parameters.append(servo_id)
            parameters.extend(data)
        return cls.encode_instruction(
            cls.broadcast_id,
            FeetechInstruction.SYNC_WRITE,
            bytes(parameters),
            allow_broadcast=True,
        )

    @classmethod
    def decode_status(
        cls,
        frame: bytes,
        *,
        expected_servo_id: int | None = None,
        expected_parameter_count: int | None = None,
    ) -> FeetechStatusPacket:
        """Validate and decode one complete status frame."""

        if not isinstance(frame, bytes) or len(frame) < 6:
            raise FeetechPacketError("FEETECH status packet is too short")
        if frame[:2] != cls.header:
            raise FeetechPacketError("FEETECH status packet header is invalid")
        servo_id, length = frame[2], frame[3]
        cls.validate_servo_id(servo_id)
        if len(frame) != length + 4 or length < 2:
            raise FeetechPacketError("FEETECH status packet length is invalid")
        body = frame[2:-1]
        if frame[-1] != cls.checksum(body):
            raise FeetechPacketError("FEETECH status packet checksum is invalid")
        if expected_servo_id is not None and servo_id != expected_servo_id:
            raise FeetechPacketError(
                f"FEETECH status ID {servo_id} does not match {expected_servo_id}"
            )
        parameters = frame[5:-1]
        if (
            expected_parameter_count is not None
            and len(parameters) != expected_parameter_count
        ):
            raise FeetechPacketError(
                "FEETECH status packet parameter count is invalid"
            )
        return FeetechStatusPacket(servo_id, frame[4], parameters)

    @staticmethod
    def checksum(body: bytes) -> int:
        """Calculate the inverted FEETECH byte-sum checksum."""

        if not isinstance(body, bytes):
            raise ValueError("FEETECH checksum input must be bytes")
        return (~sum(body)) & 0xFF

    @classmethod
    def validate_servo_id(cls, servo_id: int, *, allow_broadcast: bool = False) -> None:
        """Validate a unicast ID, optionally accepting the broadcast address."""

        maximum = cls.broadcast_id if allow_broadcast else cls.max_servo_id
        if type(servo_id) is not int or not 0 <= servo_id <= maximum:
            raise ValueError(f"FEETECH servo ID must be an integer in [0, {maximum}]")
        if servo_id == 0xFD or (servo_id == cls.broadcast_id and not allow_broadcast):
            raise ValueError("FEETECH servo ID is reserved")

    @staticmethod
    def encode_signed_magnitude(value: int, *, sign_bit: int = 15) -> int:
        """Encode a signed value as direction-bit magnitude."""

        maximum = (1 << sign_bit) - 1
        if type(value) is not int or not -maximum <= value <= maximum:
            raise ValueError(
                f"FEETECH signed value must be an integer in [{-maximum}, {maximum}]"
            )
        return abs(value) | ((1 << sign_bit) if value < 0 else 0)

    @staticmethod
    def decode_signed_magnitude(value: int, *, sign_bit: int = 15) -> int:
        """Decode a direction-bit magnitude into a signed integer."""

        maximum_encoded = (1 << (sign_bit + 1)) - 1
        if type(value) is not int or not 0 <= value <= maximum_encoded:
            raise ValueError("FEETECH encoded signed value is outside its bit width")
        magnitude = value & ((1 << sign_bit) - 1)
        return -magnitude if value & (1 << sign_bit) else magnitude

    @staticmethod
    def word(value: int) -> bytes:
        """Encode one unsigned little-endian register word."""

        if type(value) is not int or not 0 <= value <= 0xFFFF:
            raise ValueError("FEETECH word must be an integer in [0, 65535]")
        return value.to_bytes(2, "little")

    @staticmethod
    def decode_word(data: bytes) -> int:
        """Decode one unsigned little-endian register word."""

        if not isinstance(data, bytes) or len(data) != 2:
            raise ValueError("FEETECH word data must contain exactly two bytes")
        return int.from_bytes(data, "little")

    @staticmethod
    def _byte(value: int, label: str, *, minimum: int = 0) -> None:
        if type(value) is not int or not minimum <= value <= 0xFF:
            raise ValueError(f"FEETECH {label} must be an integer in [{minimum}, 255]")


__all__ = [
    "FeetechInstruction",
    "FeetechPacketCodec",
    "FeetechPacketError",
    "FeetechStatusPacket",
]
