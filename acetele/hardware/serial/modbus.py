"""Minimal Modbus-RTU codec used by FEETECH and Linker RS485 protocols."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Sequence


class ModbusProtocolError(ValueError):
    """A received frame violates the expected Modbus transaction."""


def crc16(data: bytes) -> int:
    """Calculate the Modbus CRC-16 value for ``data``."""

    if not isinstance(data, bytes):
        raise ValueError("Modbus CRC input must be bytes")
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def append_crc(payload: bytes) -> bytes:
    """Append a little-endian Modbus CRC to a request payload."""

    return payload + struct.pack("<H", crc16(payload))


def verify_crc(frame: bytes) -> None:
    """Reject a frame whose trailing CRC does not match its payload."""

    if not isinstance(frame, bytes) or len(frame) < 4:
        raise ModbusProtocolError("Modbus frame is too short")
    expected = struct.unpack("<H", frame[-2:])[0]
    actual = crc16(frame[:-2])
    if actual != expected:
        raise ModbusProtocolError(
            f"Modbus CRC mismatch: received 0x{expected:04x}, calculated 0x{actual:04x}"
        )


def encode_read_input_registers(slave_id: int, address: int, count: int) -> bytes:
    """Encode function 0x04 for a contiguous input-register block."""

    return _encode_read_registers(slave_id, 0x04, address, count)


def encode_read_holding_registers(slave_id: int, address: int, count: int) -> bytes:
    """Encode function 0x03 for a contiguous holding-register block."""

    return _encode_read_registers(slave_id, 0x03, address, count)


def _encode_read_registers(
    slave_id: int,
    function: int,
    address: int,
    count: int,
) -> bytes:
    _validate_slave(slave_id)
    _validate_register(address, "read address")
    if type(count) is not int or not 1 <= count <= 125:
        raise ValueError("Modbus read count must be an integer in [1, 125]")
    return append_crc(struct.pack(">BBHH", slave_id, function, address, count))


def encode_write_registers(slave_id: int, address: int, values: Sequence[int]) -> bytes:
    """Encode function 0x10 after validating every 16-bit register value."""

    _validate_slave(slave_id)
    _validate_register(address, "write address")
    values = tuple(values)
    if not values or len(values) > 123:
        raise ValueError("Modbus register write requires between 1 and 123 values")
    if any(type(value) is not int or not 0 <= value <= 0xFFFF for value in values):
        raise ValueError("Modbus register values must be integers in [0, 65535]")
    body = struct.pack(">BBHHB", slave_id, 0x10, address, len(values), len(values) * 2)
    body += struct.pack(f">{len(values)}H", *values)
    return append_crc(body)


def decode_read_input_registers(
    frame: bytes,
    *,
    expected_slave: int,
    expected_count: int,
) -> tuple[int, ...]:
    """Decode and verify an input-register response."""

    return _decode_read_registers(
        frame,
        expected_slave=expected_slave,
        expected_count=expected_count,
        function=0x04,
    )


def decode_read_holding_registers(
    frame: bytes,
    *,
    expected_slave: int,
    expected_count: int,
) -> tuple[int, ...]:
    """Decode and verify a holding-register response."""

    return _decode_read_registers(
        frame,
        expected_slave=expected_slave,
        expected_count=expected_count,
        function=0x03,
    )


def _decode_read_registers(
    frame: bytes,
    *,
    expected_slave: int,
    expected_count: int,
    function: int,
) -> tuple[int, ...]:
    verify_crc(frame)
    _validate_response_header(frame, expected_slave, function)
    byte_count = frame[2]
    if byte_count != expected_count * 2 or len(frame) != byte_count + 5:
        raise ModbusProtocolError("Modbus read response has an unexpected payload length")
    return struct.unpack(f">{expected_count}H", frame[3:-2])


def decode_write_registers_response(
    frame: bytes,
    *,
    expected_slave: int,
    expected_address: int,
    expected_count: int,
) -> None:
    """Verify that a write response echoes the requested range."""

    verify_crc(frame)
    _validate_response_header(frame, expected_slave, 0x10)
    if len(frame) != 8:
        raise ModbusProtocolError("Modbus write response must contain 8 bytes")
    address, count = struct.unpack(">HH", frame[2:6])
    if address != expected_address or count != expected_count:
        raise ModbusProtocolError("Modbus write response does not match the request")


@dataclass(frozen=True)
class ModbusExceptionResponse:
    """Structured fields from a Modbus exception response."""

    slave_id: int
    function: int
    exception_code: int


def _validate_response_header(frame: bytes, expected_slave: int, function: int) -> None:
    if frame[0] != expected_slave:
        raise ModbusProtocolError(
            f"Modbus response slave {frame[0]} does not match {expected_slave}"
        )
    if frame[1] == function | 0x80:
        raise ModbusProtocolError(
            f"Modbus function 0x{function:02x} failed with exception 0x{frame[2]:02x}"
        )
    if frame[1] != function:
        raise ModbusProtocolError(
            f"Modbus response function 0x{frame[1]:02x} does not match 0x{function:02x}"
        )


def _validate_slave(slave_id: int) -> None:
    if type(slave_id) is not int or not 1 <= slave_id <= 247:
        raise ValueError("Modbus slave ID must be an integer in [1, 247]")


def _validate_register(value: int, label: str) -> None:
    if type(value) is not int or not 0 <= value <= 0xFFFF:
        raise ValueError(f"Modbus {label} must be an integer in [0, 65535]")


__all__ = [
    "ModbusProtocolError",
    "append_crc",
    "crc16",
    "decode_read_holding_registers",
    "decode_read_input_registers",
    "decode_write_registers_response",
    "encode_read_holding_registers",
    "encode_read_input_registers",
    "encode_write_registers",
    "verify_crc",
]
