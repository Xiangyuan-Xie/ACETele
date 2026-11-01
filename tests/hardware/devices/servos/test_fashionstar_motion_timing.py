"""Checks advanced motion timing against the pinned official SDK 1.3.12.

The SDK's set_servo_angle clamps multi-turn interval to at least t_acc +
t_dec. Source: https://pypi.org/project/fashionstar-uart-sdk/1.3.12/.
Its wheel checksum is pinned in the FashionStar profile registry. The golden
frame below independently specifies ID 1, 10 degrees, 40 ms, and 20 ms ramps.
"""

from __future__ import annotations

import math

import pytest

from acetele.hardware.devices.servos.fashionstar import codec
from acetele.hardware.devices.servos.fashionstar import packet


def test_default_motion_has_time_for_both_ramps() -> None:
    """The runtime default is compatible with advanced synchronous control."""
    motion = packet.FashionStarMotion(math.radians(10.0))
    assert motion.interval_ms >= motion.acceleration_ms + motion.deceleration_ms
    frame = codec.FashionStarPacketCodec.encode_sync_multi_turn_positions(
        [
            (
                1,
                motion.position_rad,
                motion.interval_ms,
                motion.acceleration_ms,
                motion.deceleration_ms,
                motion.power_mw,
            )
        ]
    )
    assert frame == bytes.fromhex(
        "12 4c 19 12 0e 0f 01 01 64 00 00 00 28 00 00 00 "
        "14 00 14 00 00 00 5c"
    )


@pytest.mark.parametrize("interval_ms", [0, 10, 39])
def test_impossible_sync_ramps_are_rejected(interval_ms: int) -> None:
    """Invalid advanced profiles cannot become wire frames.

    Args:
        interval_ms: A duration shorter than the two default ramp phases.
    """
    with pytest.raises(ValueError, match="must cover"):
        codec.FashionStarPacketCodec.encode_sync_multi_turn_positions(
            [(1, 0.0, interval_ms, 20, 20, 0)]
        )


def test_all_sync_members_are_validated_before_returning_a_frame() -> None:
    """An invalid second member cannot produce a partial motion frame."""
    with pytest.raises(ValueError, match="must cover"):
        codec.FashionStarPacketCodec.encode_sync_multi_turn_positions(
            [(1, 0.0, 40, 20, 20, 0), (2, 0.0, 39, 20, 20, 0)]
        )
