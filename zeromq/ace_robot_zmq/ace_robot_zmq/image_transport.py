"""Best-effort image transport isolated from the teleoperation control sockets."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol

import msgpack
import zmq
from ace_robot_zmq.options import CurveCredentials
from ace_robot_zmq.security import CurveAuthenticator


class ImageTransportError(ValueError):
    """Raised when an image frame violates the wire contract."""


def _unsigned(value: object, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= (1 << 64) - 1:
        raise ImageTransportError(f"{field_name} must be an unsigned integer")
    return value


def _metadata_value(value: object, *, depth: int = 0) -> Any:
    """Detach metadata into finite, deterministic MessagePack primitives."""

    if depth > 4:
        raise ImageTransportError("image metadata nesting is too deep")
    if value is None or isinstance(value, (str, bytes, bool)):
        return value
    if type(value) is int:
        if not -(1 << 63) <= value <= (1 << 64) - 1:
            raise ImageTransportError("image metadata integer is out of range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ImageTransportError("image metadata must be finite")
        return float(value)
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 96:
                raise ImageTransportError("image metadata keys must be non-empty strings")
            normalized[key] = _metadata_value(item, depth=depth + 1)
        return MappingProxyType(normalized)
    if isinstance(value, (tuple, list)):
        return tuple(_metadata_value(item, depth=depth + 1) for item in value)
    raise ImageTransportError(
        f"unsupported image metadata value: {type(value).__name__}"
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ImageFrame:
    """One compressed camera component with source and capture timestamps."""

    camera: str
    stream: str
    sequence: int
    captured_at_ns: int
    sent_at_ns: int
    encoding: str
    payload: bytes
    metadata: Mapping[str, Any] = field(default_factory=dict)
    received_at_ns: int = 0
    version: int = 1

    def __post_init__(self) -> None:
        if self.camera not in ("front", "wrist"):
            raise ImageTransportError("camera must be front or wrist")
        encodings = {
            "color": "jpeg",
            "depth": "zstd-raw",
            "metadata": "json",
            "status": "json",
        }
        if self.stream not in encodings:
            raise ImageTransportError("stream must be color, depth, metadata, or status")
        if self.encoding != encodings[self.stream]:
            raise ImageTransportError(
                f"{self.stream} stream must use {encodings[self.stream]} encoding"
            )
        if self.version != 1:
            raise ImageTransportError(f"unsupported image protocol version {self.version}")
        for name in ("sequence", "captured_at_ns", "sent_at_ns", "received_at_ns"):
            _unsigned(getattr(self, name), name)
        if not isinstance(self.payload, bytes):
            raise ImageTransportError("image payload must be immutable bytes")
        if not isinstance(self.metadata, Mapping):
            raise ImageTransportError("image metadata must be a mapping")
        object.__setattr__(self, "metadata", _metadata_value(self.metadata))

    def received(self, timestamp_ns: int) -> "ImageFrame":
        """Return a receiver-stamped copy without mutating publisher data."""

        _unsigned(timestamp_ns, "received timestamp")
        return replace(self, received_at_ns=timestamp_ns)


class ImageCodec:
    """Encode one bounded image component as a strict MessagePack map."""

    def __init__(self, maximum_frame_bytes: int = 4 * 1024 * 1024) -> None:
        if type(maximum_frame_bytes) is not int or not 1024 <= maximum_frame_bytes <= 64 * 1024 * 1024:
            raise ValueError("maximum_frame_bytes must be in [1024, 67108864]")
        self.maximum_frame_bytes = maximum_frame_bytes

    def encode(self, frame: ImageFrame) -> bytes:
        if not isinstance(frame, ImageFrame):
            raise ImageTransportError("encode requires ImageFrame")
        payload = msgpack.packb(
            {
                "version": frame.version,
                "camera": frame.camera,
                "stream": frame.stream,
                "sequence": frame.sequence,
                "captured_at_ns": frame.captured_at_ns,
                "sent_at_ns": frame.sent_at_ns,
                "received_at_ns": frame.received_at_ns,
                "encoding": frame.encoding,
                "metadata": _plain(frame.metadata),
                "payload": frame.payload,
            },
            use_bin_type=True,
            strict_types=True,
        )
        if len(payload) > self.maximum_frame_bytes:
            raise ImageTransportError(
                f"image frame is {len(payload)} bytes; limit is {self.maximum_frame_bytes}"
            )
        return payload

    def decode(self, payload: bytes, *, received_at_ns: int = 0) -> ImageFrame:
        if not isinstance(payload, bytes) or not payload:
            raise ImageTransportError("image wire payload must be non-empty bytes")
        if len(payload) > self.maximum_frame_bytes:
            raise ImageTransportError("image frame exceeds maximum_frame_bytes")
        try:
            document = msgpack.unpackb(
                payload,
                raw=False,
                strict_map_key=True,
                ext_hook=lambda _code, _data: (_ for _ in ()).throw(
                    ImageTransportError("MessagePack extension values are forbidden")
                ),
            )
        except (TypeError, ValueError, msgpack.ExtraData, msgpack.FormatError, msgpack.StackError) as exc:
            if isinstance(exc, ImageTransportError):
                raise
            raise ImageTransportError(f"invalid image MessagePack: {exc}") from exc
        expected = {
            "version",
            "camera",
            "stream",
            "sequence",
            "captured_at_ns",
            "sent_at_ns",
            "received_at_ns",
            "encoding",
            "metadata",
            "payload",
        }
        if not isinstance(document, dict) or set(document) != expected:
            raise ImageTransportError("image document has missing or unknown fields")
        frame = ImageFrame(
            document["camera"],
            document["stream"],
            document["sequence"],
            document["captured_at_ns"],
            document["sent_at_ns"],
            document["encoding"],
            document["payload"],
            document["metadata"],
            document["received_at_ns"],
            document["version"],
        )
        return frame if received_at_ns == 0 else frame.received(received_at_ns)


class ImageSink(Protocol):
    def open(self) -> None: ...

    def publish(self, frame: ImageFrame) -> bool: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ImageTransportOptions:
    endpoint: str
    bind: bool
    maximum_frame_bytes: int = 4 * 1024 * 1024
    high_water_mark: int = 8
    curve: Optional[CurveCredentials] = None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str) or not self.endpoint.startswith(
            ("tcp://", "ipc://")
        ):
            raise ValueError("image endpoint must use tcp:// or ipc://")
        if type(self.bind) is not bool:
            raise ValueError("image bind must be bool")
        if type(self.high_water_mark) is not int or not 1 <= self.high_water_mark <= 4096:
            raise ValueError("image high_water_mark must be in [1, 4096]")
        ImageCodec(self.maximum_frame_bytes)


@dataclass(frozen=True)
class ImageTransportDiagnostics:
    sent_frames: int = 0
    received_frames: int = 0
    dropped_frames: int = 0
    rejected_frames: int = 0
    last_sequence: Optional[int] = None
    last_receive_ns: Optional[int] = None
    last_error: Optional[str] = None


class ImagePublisher:
    """Publish without blocking camera capture or robot control."""

    def __init__(
        self,
        options: ImageTransportOptions,
        *,
        context_factory: Callable[[], zmq.Context] = zmq.Context,
    ) -> None:
        self.options = options
        self.codec = ImageCodec(options.maximum_frame_bytes)
        self._context_factory = context_factory
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._authenticator: Optional[CurveAuthenticator] = None
        self._diagnostics = ImageTransportDiagnostics()

    def open(self) -> None:
        if self._socket is not None:
            return
        context = self._context_factory()
        socket = context.socket(zmq.PUB)
        authenticator = None
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDHWM, self.options.high_water_mark)
            socket.setsockopt(zmq.IMMEDIATE, 1)
            if self.options.curve is not None:
                authenticator = CurveAuthenticator(context, self.options.curve)
                if self.options.bind:
                    authenticator.configure_server(socket)
                else:
                    authenticator.configure_client(socket)
            if self.options.bind:
                socket.bind(self.options.endpoint)
            else:
                socket.connect(self.options.endpoint)
        except BaseException:
            socket.close(linger=0)
            if authenticator is not None:
                authenticator.close()
            context.term()
            raise
        self._context = context
        self._socket = socket
        self._authenticator = authenticator

    def publish(self, frame: ImageFrame) -> bool:
        if self._socket is None:
            raise RuntimeError("image publisher is not open")
        try:
            self._socket.send(self.codec.encode(frame), flags=zmq.NOBLOCK)
        except zmq.Again:
            self._diagnostics = replace(
                self._diagnostics,
                dropped_frames=self._diagnostics.dropped_frames + 1,
            )
            return False
        self._diagnostics = replace(
            self._diagnostics,
            sent_frames=self._diagnostics.sent_frames + 1,
            last_sequence=frame.sequence,
        )
        return True

    def diagnostics(self) -> ImageTransportDiagnostics:
        return self._diagnostics

    def close(self) -> None:
        self._close_resources()

    def _close_resources(self) -> None:
        socket, context, authenticator = self._socket, self._context, self._authenticator
        self._socket = None
        self._context = None
        self._authenticator = None
        first_error: Optional[BaseException] = None
        callbacks = (
            (() if socket is None else (lambda: socket.close(linger=0),))
            + (() if authenticator is None else (authenticator.close,))
            + (() if context is None else (context.term,))
        )
        for callback in callbacks:
            try:
                callback()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


class ImageSubscriber:
    """Receive bounded image components and reject stale per-camera sequences."""

    def __init__(
        self,
        options: ImageTransportOptions,
        *,
        context_factory: Callable[[], zmq.Context] = zmq.Context,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.options = options
        self.codec = ImageCodec(options.maximum_frame_bytes)
        self._context_factory = context_factory
        self._clock_ns = clock_ns
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._poller: Optional[zmq.Poller] = None
        self._authenticator: Optional[CurveAuthenticator] = None
        self._diagnostics = ImageTransportDiagnostics()
        self._last_sequences: dict[str, int] = {}

    def open(self) -> None:
        if self._socket is not None:
            return
        context = self._context_factory()
        socket = context.socket(zmq.SUB)
        authenticator = None
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVHWM, self.options.high_water_mark)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            if self.options.curve is not None:
                authenticator = CurveAuthenticator(context, self.options.curve)
                if self.options.bind:
                    authenticator.configure_server(socket)
                else:
                    authenticator.configure_client(socket)
            if self.options.bind:
                socket.bind(self.options.endpoint)
            else:
                socket.connect(self.options.endpoint)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
        except BaseException:
            socket.close(linger=0)
            if authenticator is not None:
                authenticator.close()
            context.term()
            raise
        self._context = context
        self._socket = socket
        self._poller = poller
        self._authenticator = authenticator

    def receive(self, *, timeout_ms: int = 0) -> Optional[ImageFrame]:
        if type(timeout_ms) is not int or timeout_ms < 0:
            raise ValueError("timeout_ms must be a non-negative integer")
        if self._socket is None or self._poller is None:
            raise RuntimeError("image subscriber is not open")
        if self._socket not in dict(self._poller.poll(timeout_ms)):
            return None
        received_ns = self._clock_ns()
        try:
            frame = self.codec.decode(
                self._socket.recv(flags=zmq.NOBLOCK),
                received_at_ns=received_ns,
            )
        except (ImageTransportError, zmq.ZMQError) as exc:
            self._diagnostics = replace(
                self._diagnostics,
                rejected_frames=self._diagnostics.rejected_frames + 1,
                last_error=str(exc),
            )
            return None
        previous = self._last_sequences.get(frame.camera)
        if previous is not None and frame.sequence <= previous:
            self._diagnostics = replace(
                self._diagnostics,
                rejected_frames=self._diagnostics.rejected_frames + 1,
                last_error=(
                    f"out-of-order image sequence for {frame.camera}: "
                    f"received {frame.sequence} after {previous}"
                ),
            )
            return None
        dropped = 0 if previous is None else max(0, frame.sequence - previous - 1)
        self._last_sequences[frame.camera] = frame.sequence
        self._diagnostics = replace(
            self._diagnostics,
            received_frames=self._diagnostics.received_frames + 1,
            dropped_frames=self._diagnostics.dropped_frames + dropped,
            last_sequence=frame.sequence,
            last_receive_ns=received_ns,
        )
        return frame

    def diagnostics(self) -> ImageTransportDiagnostics:
        return self._diagnostics

    def close(self) -> None:
        socket, context, authenticator = self._socket, self._context, self._authenticator
        self._socket = None
        self._context = None
        self._poller = None
        self._authenticator = None
        self._last_sequences.clear()
        first_error: Optional[BaseException] = None
        callbacks = (
            (() if socket is None else (lambda: socket.close(linger=0),))
            + (() if authenticator is None else (authenticator.close,))
            + (() if context is None else (context.term,))
        )
        for callback in callbacks:
            try:
                callback()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


__all__ = [
    "ImageCodec",
    "ImageFrame",
    "ImagePublisher",
    "ImageSink",
    "ImageSubscriber",
    "ImageTransportDiagnostics",
    "ImageTransportError",
    "ImageTransportOptions",
]
