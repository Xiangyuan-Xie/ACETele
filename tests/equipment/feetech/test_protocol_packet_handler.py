import pytest

from acetele.equipment.feetech.feetech_sdk.hls import (
    HLS_PRESENT_POSITION_L,
    HLS_PRESENT_SPEED_L,
    hls,
)
from acetele.equipment.feetech.feetech_sdk.protocol_packet_handler import (
    protocol_packet_handler,
)


@pytest.mark.parametrize(
    ("encoded", "decoded"),
    (
        (0x0000, 0),
        (0x0001, 1),
        (0x7FFF, 32767),
        (0x8000, 0),
        (0x8001, -1),
        (0xFFFF, -32767),
    ),
)
def test_signed_magnitude_values_decode_without_negative_bit_extension(
    encoded,
    decoded,
):
    handler = protocol_packet_handler(None, 0)

    assert handler.scs_tohost(encoded, 15) == decoded


@pytest.mark.parametrize("value", (-32767, -1, 0, 1, 32767))
def test_signed_magnitude_encoding_round_trips(value):
    handler = protocol_packet_handler(None, 0)

    assert handler.scs_tohost(handler.scs_toscs(value, 15), 15) == value


def test_makeword_preserves_raw_sign_bit_for_field_specific_decoding():
    handler = protocol_packet_handler(None, 0)

    assert handler.scs_makeword(0x01, 0x80) == 0x8001
    assert handler.scs_tohost(handler.scs_makeword(0x01, 0x80), 15) == -1


@pytest.mark.parametrize(
    ("method_name", "address", "raw_value"),
    (
        ("ReadPos", HLS_PRESENT_POSITION_L, 0x8001),
        ("ReadSpeed", HLS_PRESENT_SPEED_L, 0x8001),
    ),
)
def test_hls_sdk_accessors_use_signed_magnitude_format(
    method_name,
    address,
    raw_value,
):
    handler = hls(None)
    handler.read2ByteTxRx = lambda _servo_id, actual_address: (
        raw_value if actual_address == address else 0,
        0,
        0,
    )

    value, comm_result, error = getattr(handler, method_name)(1)

    assert (value, comm_result, error) == (-1, 0, 0)


def test_hls_position_writer_uses_signed_magnitude_format():
    handler = hls(None)
    captured = {}

    def write_tx_rx(servo_id, address, length, payload):
        captured.update(
            servo_id=servo_id,
            address=address,
            length=length,
            payload=payload,
        )
        return 0, 0

    handler.writeTxRx = write_tx_rx

    handler.WritePosEx(1, -1, speed=10, acc=2, torque=100)

    assert captured["payload"][1:3] == [0x01, 0x80]
