from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ace_robot_zmq.camera import (
    CameraApplication,
    CameraFrameSet,
    CameraOptions,
)


class Sink:
    def __init__(self):
        self.frames = []
        self.opened = False

    def open(self):
        self.opened = True

    def publish(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        self.opened = False


class Source:
    def __init__(self):
        self.opened = False

    def open(self):
        self.opened = True

    def capture(self, camera):
        return CameraFrameSet(
            camera,
            f"serial-{camera}",
            7,
            12.5,
            100,
            np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
            np.arange(6, dtype=np.uint16).reshape(2, 3),
            {"depth_scale_m": 0.001},
        )

    def close(self):
        self.opened = False


@pytest.mark.parametrize(
    "serials",
    ({"front": "one"}, {"front": "same", "wrist": "same"}),
)
def test_camera_options_require_two_distinct_assignments(serials):
    with pytest.raises(ValueError):
        CameraOptions(serials)


def test_camera_application_publishes_only_compressed_preview(monkeypatch):
    class Compressor:
        def __init__(self, *, level):
            assert level == 1

        def compress(self, payload):
            return b"zstd" + payload

    monkeypatch.setitem(sys.modules, "zstandard", SimpleNamespace(ZstdCompressor=Compressor))
    source = Source()
    preview = Sink()
    app = CameraApplication(
        source,
        preview,
        CameraOptions({"front": "one", "wrist": "two"}, width=160, height=160),
        wall_clock_ns=lambda: 200,
    )
    app.open()
    try:
        sample = app.publish_once("front")
    finally:
        app.close()

    assert sample.color_bgr8.flags.writeable is False
    assert [frame.encoding for frame in preview.frames] == ["json", "jpeg", "zstd-raw"]
    assert preview.frames[2].payload.startswith(b"zstd")
    assert not source.opened and not preview.opened


def test_camera_open_failure_closes_partially_initialized_source(monkeypatch):
    class FailingSource(Source):
        def __init__(self):
            super().__init__()
            self.close_calls = 0

        def open(self):
            self.opened = True
            raise RuntimeError("camera initialization failed")

        def close(self):
            self.close_calls += 1
            super().close()

    class Compressor:
        def __init__(self, *, level):
            pass

    monkeypatch.setitem(sys.modules, "zstandard", SimpleNamespace(ZstdCompressor=Compressor))
    source = FailingSource()
    preview = Sink()
    app = CameraApplication(
        source,
        preview,
        CameraOptions({"front": "one", "wrist": "two"}),
    )

    with pytest.raises(RuntimeError, match="camera initialization failed"):
        app.open()

    assert source.close_calls == 1
    assert not source.opened and not preview.opened
