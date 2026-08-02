"""Image-only ZeroMQ source for the shared operator window."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from ace_robot_zmq.image_transport import (
    ImageFrame,
    ImageSubscriber,
    ImageTransportOptions,
)
from ace_robot_zmq.options import CurveCredentials


@dataclass(frozen=True)
class OperatorImageOptions:
    """Endpoint for read-only camera previews from the follower host."""

    follower_host: str
    camera_port: int = 5562
    curve: Optional[CurveCredentials] = None
    endpoint: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.follower_host, str) or not self.follower_host.strip():
            raise ValueError("follower_host must be a non-empty host")
        if "://" in self.follower_host:
            raise ValueError("follower_host must not include a transport scheme")
        if type(self.camera_port) is not int or not 1 <= self.camera_port <= 65_535:
            raise ValueError("camera_port must be in [1, 65535]")
        if self.endpoint is not None and not self.endpoint.startswith(
            ("tcp://", "ipc://")
        ):
            raise ValueError("image endpoint must use tcp:// or ipc://")


class ZmqOperatorSource:
    """Decode latest RGB-D previews without touching robot control sockets."""

    def __init__(
        self,
        options: OperatorImageOptions,
        *,
        subscriber: Optional[ImageSubscriber] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        wall_clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.options = options
        self._clock_ns = clock_ns
        self._wall_clock_ns = wall_clock_ns
        endpoint = options.endpoint or (
            f"tcp://{options.follower_host}:{options.camera_port}"
        )
        self._subscriber = subscriber or ImageSubscriber(
            ImageTransportOptions(
                endpoint,
                False,
                curve=None if endpoint.startswith("ipc://") else options.curve,
            )
        )
        self._images: dict[str, np.ndarray] = {}
        self._metadata: dict[str, str] = {}
        self._health: dict[str, tuple[int, float]] = {}
        self._camera_online = {"front": False, "wrist": False}
        self._opened = False
        self._depth_decompressor = None

    def open(self) -> None:
        if self._opened:
            return
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError("ZMQ image display requires zstandard") from exc
        self._subscriber.open()
        self._depth_decompressor = zstandard.ZstdDecompressor()
        self._opened = True

    def snapshot(self):
        """Drain available frames and return one detached UI snapshot."""

        if not self._opened:
            self.open()
        while True:
            frame = self._subscriber.receive(timeout_ms=0)
            if frame is None:
                break
            try:
                self._consume(frame)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        now_ns = self._clock_ns()
        health = {
            name: (
                f"ONLINE ({latency_ms:.1f}ms)"
                if now_ns - observed_ns < 2_000_000_000
                else "OFFLINE"
            )
            for name, (observed_ns, latency_ms) in self._health.items()
        }
        for name in (
            "front_color",
            "front_depth",
            "front_metadata",
            "wrist_color",
            "wrist_depth",
            "wrist_metadata",
        ):
            health.setdefault(name, "OFFLINE")
        diagnostics = self._subscriber.diagnostics()
        from ace_operator_ui import OperatorSnapshot

        return OperatorSnapshot(
            images=self._images,
            metadata=self._metadata,
            health=health,
            metrics={
                "image.received": str(diagnostics.received_frames),
                "image.dropped": str(diagnostics.dropped_frames),
                "image.rejected": str(diagnostics.rejected_frames),
            },
        )

    def close(self) -> None:
        self._opened = False
        self._depth_decompressor = None
        self._subscriber.close()

    def _consume(self, frame: ImageFrame) -> None:
        """Decode one validated component into the latest display state."""

        observed_ns = self._clock_ns()
        latency_ms = max(0.0, (self._wall_clock_ns() - frame.sent_at_ns) / 1e6)
        if frame.stream == "status":
            document = json.loads(frame.payload.decode("utf-8"))
            self._camera_online[frame.camera] = bool(document.get("online", False))
            if not self._camera_online[frame.camera]:
                for stream in ("color", "depth", "metadata"):
                    self._health.pop(f"{frame.camera}_{stream}", None)
            return
        if frame.stream == "metadata":
            document = json.loads(frame.payload.decode("utf-8"))
            self._metadata[frame.camera] = json.dumps(
                document,
                indent=2,
                sort_keys=True,
            )
            self._health[f"{frame.camera}_metadata"] = (observed_ns, latency_ms)
            return
        if frame.stream == "color":
            encoded = np.frombuffer(frame.payload, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("invalid JPEG preview")
        else:
            if self._depth_decompressor is None:
                raise RuntimeError("depth decompressor is not initialized")
            width = int(frame.metadata["width"])
            height = int(frame.metadata["height"])
            raw = self._depth_decompressor.decompress(
                frame.payload,
                max_output_size=width * height * 2,
            )
            image = np.frombuffer(raw, dtype=np.uint16).reshape(height, width).copy()
        key = f"{frame.camera}_{frame.stream}"
        self._images[key] = image
        self._health[key] = (observed_ns, latency_ms)


__all__ = ["OperatorImageOptions", "ZmqOperatorSource"]
