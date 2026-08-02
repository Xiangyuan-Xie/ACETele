from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
from ace_robot_zmq.image_transport import ImageFrame, ImageTransportDiagnostics
from ace_robot_zmq.operator import OperatorImageOptions, ZmqOperatorSource


class Subscriber:
    def __init__(self, frames=()):
        self.frames = list(frames)
        self.opened = False

    def open(self):
        self.opened = True

    def receive(self, *, timeout_ms=0):
        return self.frames.pop(0) if self.frames else None

    def diagnostics(self):
        return ImageTransportDiagnostics(received_frames=3)

    def close(self):
        self.opened = False


def test_operator_source_decodes_camera_preview_only(monkeypatch):
    class Decompressor:
        def decompress(self, payload, *, max_output_size):
            assert max_output_size == 12
            return payload

    monkeypatch.setitem(
        sys.modules,
        "zstandard",
        SimpleNamespace(ZstdDecompressor=Decompressor),
    )
    depth = np.arange(6, dtype=np.uint16).tobytes()
    frames = (
        ImageFrame(
            "front",
            "metadata",
            1,
            2,
            3,
            "json",
            json.dumps({"depth_scale_m": 0.001}).encode(),
        ),
        ImageFrame(
            "front",
            "depth",
            2,
            3,
            4,
            "zstd-raw",
            depth,
            {"width": 3, "height": 2},
        ),
    )
    subscriber = Subscriber(frames)
    source = ZmqOperatorSource(
        OperatorImageOptions("follower"),
        subscriber=subscriber,
        clock_ns=lambda: 10,
        wall_clock_ns=lambda: 14,
    )

    source.open()
    snapshot = source.snapshot()
    source.close()

    assert snapshot.images["front_depth"].shape == (2, 3)
    assert "depth_scale_m" in snapshot.metadata["front"]
    assert snapshot.health["front_metadata"].startswith("ONLINE")
    assert snapshot.metrics["image.received"] == "3"
    assert snapshot.joints.names == ()
    assert not subscriber.opened
