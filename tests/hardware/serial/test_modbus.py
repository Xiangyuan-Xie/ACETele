from __future__ import annotations

import pytest

from acetele.hardware.serial import (
    ModbusProtocolError,
    append_crc,
    crc16,
    decode_read_holding_registers,
    decode_read_input_registers,
    encode_read_holding_registers,
    encode_read_input_registers,
    encode_write_registers,
)


def test_modbus_crc_matches_standard_read_input_register_frame():
    frame = encode_read_input_registers(1, 0, 6)

    assert frame == bytes.fromhex("01 04 00 00 00 06 70 08")
    assert crc16(frame[:-2]) == 0x0870


def test_modbus_read_response_decodes_big_endian_registers():
    frame = append_crc(bytes.fromhex("01 04 04 00 7f 00 ff"))

    assert decode_read_input_registers(frame, expected_slave=1, expected_count=2) == (
        127,
        255,
    )


def test_modbus_holding_register_codec_uses_function_three():
    request = encode_read_holding_registers(1, 256, 2)
    response = append_crc(bytes.fromhex("01 03 04 00 0a 01 02"))

    assert request == bytes.fromhex("01 03 01 00 00 02 c5 f7")
    assert decode_read_holding_registers(
        response,
        expected_slave=1,
        expected_count=2,
    ) == (0x000A, 0x0102)


def test_modbus_crc_and_raw_register_values_are_strict():
    frame = bytearray(append_crc(bytes.fromhex("01 04 02 00 01")))
    frame[-1] ^= 0xFF
    with pytest.raises(ModbusProtocolError, match="CRC"):
        decode_read_input_registers(bytes(frame), expected_slave=1, expected_count=1)

    with pytest.raises(ValueError, match="register values"):
        encode_write_registers(1, 0, [1.0])
