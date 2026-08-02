"""Direct RealSense RGB-D capture isolated from teleoperation control."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol

import numpy as np
from ace_robot_zmq.image_transport import ImageFrame, ImageSink


class CameraError(RuntimeError):
    """Raised for camera discovery, capture, or encoding failures."""


def _plain_metadata(value: Any) -> Any:
    """Detach SDK metadata into JSON-safe primitives for image transport."""

    if isinstance(value, Mapping):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class CameraDeviceInfo:
    serial: str
    name: str
    firmware: str


@dataclass(frozen=True)
class CameraOptions:
    """Explicit physical camera assignment and stream/preview policy."""

    serials: Mapping[str, str]
    width: int = 640
    height: int = 480
    fps: int = 30
    jpeg_quality: int = 85
    depth_zstd_level: int = 1
    frame_timeout_ms: int = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.serials, Mapping) or set(self.serials) != {"front", "wrist"}:
            raise ValueError("camera serials must map exactly 'front' and 'wrist'")
        serials = dict(self.serials)
        if any(not isinstance(value, str) or not value.strip() for value in serials.values()):
            raise ValueError("camera serial numbers must be non-empty strings")
        if len(set(serials.values())) != 2:
            raise ValueError("front and wrist must use different RealSense devices")
        object.__setattr__(self, "serials", MappingProxyType(serials))
        for name in ("width", "height"):
            value = getattr(self, name)
            if type(value) is not int or not 160 <= value <= 4096:
                raise ValueError(f"camera {name} must be an integer in [160, 4096]")
        if type(self.fps) is not int or not 1 <= self.fps <= 90:
            raise ValueError("camera fps must be an integer in [1, 90]")
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if type(self.depth_zstd_level) is not int or not -5 <= self.depth_zstd_level <= 22:
            raise ValueError("depth_zstd_level must be in [-5, 22]")
        if type(self.frame_timeout_ms) is not int or not 1 <= self.frame_timeout_ms <= 10_000:
            raise ValueError("frame_timeout_ms must be in [1, 10000]")


@dataclass(frozen=True)
class CameraFrameSet:
    """One aligned color/depth pair detached from SDK-owned buffers."""

    camera: str
    serial: str
    frame_number: int
    device_time_ms: float
    captured_at_ns: int
    color_bgr8: np.ndarray
    depth_u16: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.camera not in ("front", "wrist"):
            raise ValueError("camera must be front or wrist")
        if not isinstance(self.serial, str) or not self.serial:
            raise ValueError("camera serial must be non-empty")
        if type(self.frame_number) is not int or self.frame_number < 0:
            raise ValueError("camera frame number must be non-negative")
        if not math.isfinite(self.device_time_ms) or self.device_time_ms < 0.0:
            raise ValueError("camera device timestamp must be finite and non-negative")
        if type(self.captured_at_ns) is not int or self.captured_at_ns < 0:
            raise ValueError("camera captured_at_ns must be non-negative")
        color = np.asarray(self.color_bgr8, dtype=np.uint8).copy()
        depth = np.asarray(self.depth_u16, dtype=np.uint16).copy()
        if color.ndim != 3 or color.shape[2] != 3:
            raise ValueError("camera color frame must be HxWx3 BGR8")
        if depth.ndim != 2 or depth.shape != color.shape[:2]:
            raise ValueError("camera depth frame must match color height and width")
        color.setflags(write=False)
        depth.setflags(write=False)
        object.__setattr__(self, "color_bgr8", color)
        object.__setattr__(self, "depth_u16", depth)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class CameraSource(Protocol):
    def open(self) -> None: ...

    def capture(self, camera: str) -> CameraFrameSet: ...

    def close(self) -> None: ...


def discover_realsense_devices() -> tuple[CameraDeviceInfo, ...]:
    """Enumerate devices without retaining an SDK context or opening streams."""

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraError(
            "RealSense capture requires pyrealsense2/librealsense"
        ) from exc
    devices = []
    for device in rs.context().query_devices():
        def value(field) -> str:
            return str(device.get_info(field)) if device.supports(field) else ""

        devices.append(
            CameraDeviceInfo(
                value(rs.camera_info.serial_number),
                value(rs.camera_info.name),
                value(rs.camera_info.firmware_version),
            )
        )
    return tuple(sorted(devices, key=lambda item: item.serial))


class RealSenseCameraSource:
    """Own two librealsense pipelines in a camera-only process."""

    def __init__(
        self,
        options: CameraOptions,
        *,
        wall_clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.options = options
        self._wall_clock_ns = wall_clock_ns
        self._rs = None
        self._pipelines: dict[str, Any] = {}
        self._aligners: dict[str, Any] = {}
        self._metadata: dict[str, Mapping[str, Any]] = {}

    def open(self) -> None:
        if self._pipelines:
            return
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraError(
                "RealSense capture requires pyrealsense2/librealsense"
            ) from exc
        available = {device.serial for device in discover_realsense_devices()}
        missing = sorted(set(self.options.serials.values()) - available)
        if missing:
            raise CameraError("RealSense serials are not connected: " + ", ".join(missing))
        opened: list[Any] = []
        try:
            for camera, serial in self.options.serials.items():
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(
                    rs.stream.color,
                    self.options.width,
                    self.options.height,
                    rs.format.bgr8,
                    self.options.fps,
                )
                config.enable_stream(
                    rs.stream.depth,
                    self.options.width,
                    self.options.height,
                    rs.format.z16,
                    self.options.fps,
                )
                profile = pipeline.start(config)
                opened.append(pipeline)
                self._pipelines[camera] = pipeline
                self._aligners[camera] = rs.align(rs.stream.color)
                self._metadata[camera] = self._profile_metadata(rs, profile, serial)
        except BaseException as exc:
            for pipeline in reversed(opened):
                try:
                    pipeline.stop()
                except BaseException:
                    pass
            self._pipelines.clear()
            self._aligners.clear()
            self._metadata.clear()
            raise CameraError(f"could not start RealSense streams: {exc}") from exc
        self._rs = rs

    def capture(self, camera: str) -> CameraFrameSet:
        try:
            pipeline = self._pipelines[camera]
            aligner = self._aligners[camera]
        except KeyError as exc:
            raise CameraError(f"RealSense camera '{camera}' is not open") from exc
        try:
            frames = aligner.process(
                pipeline.wait_for_frames(self.options.frame_timeout_ms)
            )
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise CameraError(f"RealSense {camera} returned an incomplete frameset")
            color = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).copy()
            return CameraFrameSet(
                camera,
                self.options.serials[camera],
                int(color_frame.get_frame_number()),
                float(color_frame.get_timestamp()),
                self._wall_clock_ns(),
                color,
                depth,
                self._metadata[camera],
            )
        except CameraError:
            raise
        except BaseException as exc:
            raise CameraError(f"RealSense {camera} capture failed: {exc}") from exc

    def close(self) -> None:
        pipelines = tuple(self._pipelines.values())
        self._pipelines.clear()
        self._aligners.clear()
        self._metadata.clear()
        self._rs = None
        first_error: Optional[BaseException] = None
        for pipeline in pipelines:
            try:
                pipeline.stop()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise CameraError(f"could not stop RealSense pipeline: {first_error}") from first_error

    @staticmethod
    def _profile_metadata(rs: Any, profile: Any, serial: str) -> Mapping[str, Any]:
        """Extract calibration by named stream instead of SDK enum ordinals."""

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color = color_profile.get_intrinsics()
        depth = depth_profile.get_intrinsics()
        extrinsics = depth_profile.get_extrinsics_to(color_profile)
        sensor = profile.get_device().first_depth_sensor()

        def intrinsics(value: Any) -> Mapping[str, Any]:
            return {
                "width": int(value.width),
                "height": int(value.height),
                "fx": float(value.fx),
                "fy": float(value.fy),
                "ppx": float(value.ppx),
                "ppy": float(value.ppy),
                "model": str(value.model),
                "coeffs": tuple(float(item) for item in value.coeffs),
            }

        return MappingProxyType(
            {
                "serial": serial,
                "color_intrinsics": intrinsics(color),
                "depth_intrinsics": intrinsics(depth),
                "depth_to_color_rotation": tuple(float(item) for item in extrinsics.rotation),
                "depth_to_color_translation_m": tuple(
                    float(item) for item in extrinsics.translation
                ),
                "depth_scale_m": float(sensor.get_depth_scale()),
            }
        )


class CameraApplication:
    """Capture two cameras and publish compressed previews outside control."""

    def __init__(
        self,
        source: CameraSource,
        preview_sink: ImageSink,
        options: CameraOptions,
        *,
        wall_clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.source = source
        self.preview_sink = preview_sink
        self.options = options
        self._wall_clock_ns = wall_clock_ns
        self._preview_sequences = {"front": 0, "wrist": 0}
        self._opened = False
        self._cv2 = None
        self._depth_compressor = None
        self._camera_status: dict[str, bool] = {}

    def open(self) -> None:
        if self._opened:
            return
        self.preview_sink.open()
        try:
            import cv2
            import zstandard
            self._cv2 = cv2
            self._depth_compressor = zstandard.ZstdCompressor(
                level=self.options.depth_zstd_level
            )
        except ImportError as exc:
            try:
                self.preview_sink.close()
            finally:
                raise CameraError(
                    "camera preview requires OpenCV and zstandard"
                ) from exc
        try:
            self.source.open()
        except BaseException:
            # A backend may allocate native handles before its open call fails.
            for callback in (
                self.source.close,
                self.preview_sink.close,
            ):
                try:
                    callback()
                except BaseException:
                    pass
            raise
        self._opened = True

    def publish_once(self, camera: str) -> CameraFrameSet:
        frame = self.source.capture(camera)
        sent_at = self._wall_clock_ns()
        common = {
            "camera": frame.camera,
            "serial": frame.serial,
            "frame_number": frame.frame_number,
            "device_time_ms": frame.device_time_ms,
            "width": int(frame.color_bgr8.shape[1]),
            "height": int(frame.color_bgr8.shape[0]),
        }
        color_metadata = {
            **common,
            "dtype": "uint8",
            "pixel_format": "bgr8",
            "stride": int(frame.color_bgr8.strides[0]),
        }
        depth_metadata = {
            **common,
            "dtype": "uint16",
            "pixel_format": "16UC1",
            "stride": int(frame.depth_u16.strides[0]),
        }
        metadata_payload = json.dumps(
            _plain_metadata(frame.metadata), separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        self.preview_sink.publish(
            ImageFrame(
                camera,
                "metadata",
                self._next_sequence(self._preview_sequences, camera),
                frame.captured_at_ns,
                sent_at,
                "json",
                metadata_payload,
                common,
            )
        )
        self._publish_preview(frame, sent_at, color_metadata, depth_metadata)
        return frame

    def run(self, should_stop: Callable[[], bool]) -> None:
        self.open()
        while not should_stop():
            for camera in ("front", "wrist"):
                if should_stop():
                    return
                try:
                    self.publish_once(camera)
                except CameraError as exc:
                    self._publish_status(camera, False, str(exc))
                    try:
                        self.source.close()
                    except BaseException:
                        pass
                    deadline = time.monotonic() + 1.0
                    while not should_stop() and time.monotonic() < deadline:
                        time.sleep(0.05)
                    if should_stop():
                        return
                    try:
                        self.source.open()
                    except CameraError:
                        break
                else:
                    self._publish_status(camera, True, None)

    def close(self) -> None:
        if not self._opened:
            return
        self._opened = False
        self._cv2 = None
        self._depth_compressor = None
        self._camera_status.clear()
        first_error: Optional[BaseException] = None
        for callback in (self.source.close, self.preview_sink.close):
            try:
                callback()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _publish_preview(
        self,
        frame: CameraFrameSet,
        sent_at_ns: int,
        color_metadata: Mapping[str, Any],
        depth_metadata: Mapping[str, Any],
    ) -> None:
        if self._cv2 is None or self._depth_compressor is None:
            raise CameraError("camera preview encoder is not initialized")
        ok, encoded_color = self._cv2.imencode(
            ".jpg",
            frame.color_bgr8,
            (self._cv2.IMWRITE_JPEG_QUALITY, self.options.jpeg_quality),
        )
        if not ok:
            raise CameraError(f"could not JPEG-encode {frame.camera} color frame")
        encoded_depth = self._depth_compressor.compress(
            frame.depth_u16.tobytes(order="C")
        )
        for channel, encoding, payload, metadata in (
            ("color", "jpeg", encoded_color.tobytes(), color_metadata),
            ("depth", "zstd-raw", encoded_depth, depth_metadata),
        ):
            self.preview_sink.publish(
                ImageFrame(
                    frame.camera,
                    channel,
                    self._next_sequence(self._preview_sequences, frame.camera),
                    frame.captured_at_ns,
                    sent_at_ns,
                    encoding,
                    payload,
                    metadata,
                )
            )

    def _publish_status(
        self,
        camera: str,
        online: bool,
        error: Optional[str],
    ) -> None:
        """Report camera degradation without coupling it to robot safety state."""

        if self._camera_status.get(camera) is online:
            return
        self._camera_status[camera] = online
        observed_at_ns = self._wall_clock_ns()
        payload = json.dumps(
            {"online": online, "error": error},
            separators=(",", ":"),
        ).encode("utf-8")
        frame = ImageFrame(
            camera,
            "status",
            self._next_sequence(self._preview_sequences, camera),
            observed_at_ns,
            observed_at_ns,
            "json",
            payload,
            {"online": online},
        )
        try:
            self.preview_sink.publish(frame)
        except Exception:
            pass

    @staticmethod
    def _next_sequence(sequences: dict[str, int], camera: str) -> int:
        """Advance one source-local sequence without coupling the two cameras."""

        sequence = sequences[camera]
        sequences[camera] = sequence + 1
        return sequence


__all__ = [
    "CameraApplication",
    "CameraDeviceInfo",
    "CameraError",
    "CameraFrameSet",
    "CameraOptions",
    "CameraSource",
    "RealSenseCameraSource",
    "discover_realsense_devices",
]
