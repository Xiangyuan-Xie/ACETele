import pytest

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
