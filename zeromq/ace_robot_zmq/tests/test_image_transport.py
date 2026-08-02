from __future__ import annotations

import socket
import time

import pytest
from ace_robot_zmq.image_transport import (
    ImageCodec,
    ImageFrame,
    ImagePublisher,
    ImageSubscriber,
    ImageTransportError,
    ImageTransportOptions,
)


def _port() -> int:
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _frame(sequence: int = 1, payload: bytes = b"jpeg") -> ImageFrame:
    return ImageFrame(
        "front",
        "color",
        sequence,
        100,
        200,
        "jpeg",
        payload,
        {"width": 640, "height": 480},
    )


def test_image_codec_round_trips_and_detaches_metadata():
    metadata = {"nested": {"values": [1, 2]}}
    frame = ImageFrame("front", "color", 1, 2, 3, "jpeg", b"data", metadata)
    metadata["nested"]["values"].append(3)

    decoded = ImageCodec().decode(ImageCodec().encode(frame), received_at_ns=9)

    assert decoded.payload == b"data"
    assert decoded.received_at_ns == 9
    assert decoded.metadata["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        decoded.metadata["new"] = True


@pytest.mark.parametrize(
    "values",
    (
        ("side", "color", "jpeg"),
        ("front", "raw", "jpeg"),
        ("front", "depth", "jpeg"),
    ),
)
def test_image_frame_rejects_non_image_wire_values(values):
    with pytest.raises(ImageTransportError):
        ImageFrame(values[0], values[1], 0, 1, 2, values[2], b"x")


def test_image_codec_rejects_oversized_frame():
    codec = ImageCodec(1024)
    with pytest.raises(ImageTransportError, match="limit"):
        codec.encode(_frame(payload=b"x" * 2000))


def test_image_pub_sub_exchanges_snapshot():
    endpoint = f"tcp://127.0.0.1:{_port()}"
    publisher = ImagePublisher(ImageTransportOptions(endpoint, True))
    subscriber = ImageSubscriber(ImageTransportOptions(endpoint, False))
    publisher.open()
    subscriber.open()
    try:
        deadline = time.monotonic() + 2.0
        received = None
        while time.monotonic() < deadline:
            publisher.publish(_frame())
            received = subscriber.receive(timeout_ms=20)
            if received is not None:
                break
        assert received is not None
        assert received.camera == "front"
        assert received.stream == "color"
        assert received.received_at_ns > 0
    finally:
        subscriber.close()
        publisher.close()
