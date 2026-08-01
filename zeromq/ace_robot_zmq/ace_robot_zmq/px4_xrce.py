"""Managed native XRCE-DDS publication for the ZMQ follower.

The Python process owns lifecycle and validation while a small native sidecar owns
the Micro XRCE-DDS C API. A Unix datagram carries the exact fixed-size CDR payload,
keeping native middleware failures outside the process that controls robot hardware.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import select
import signal
import socket
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Callable, Mapping, Optional

import numpy as np

from acetele.core import JointUnit, RobotState
from acetele.specification import RobotSpec


class Px4XrceError(RuntimeError):
    """Raised when the mandatory PX4 XRCE publication path is unavailable."""


@dataclass(frozen=True)
class ArmJointStateSchema:
    """Canonical PX4 wire contract mirrored from ``px4_msgs``."""

    maximum_joints: int = 14
    minimum_joints: int = 4
    serialized_size: int = 136
    topic_name: str = "rt/fmu/in/arm_joint_state"
    type_name: str = "px4_msgs::msg::dds_::ArmJointState_"

    @property
    def message_path(self) -> Path:
        return Path(__file__).with_name("ArmJointState.msg")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.message_path.read_bytes()).hexdigest()

    @property
    def payload_struct(self) -> struct.Struct:
        # The explicit two-byte pad matches PX4's generated micro-CDR serializer.
        return struct.Struct(f"<QQIB?2x{self.maximum_joints}f{self.maximum_joints}f")

    def __post_init__(self) -> None:
        if self.payload_struct.size != self.serialized_size:
            raise ValueError("ArmJointState schema and serialized size disagree")


class ArmJointStateEncoder:
    """Flatten measured arm groups and encode the exact PX4 CDR payload."""

    def __init__(
        self,
        spec: RobotSpec,
        *,
        schema: ArmJointStateSchema = ArmJointStateSchema(),
    ) -> None:
        self.schema = schema
        self._groups = tuple(
            (arm.name, tuple(joint.name for joint in arm.joints)) for arm in spec.arms
        )
        count = sum(len(names) for _, names in self._groups)
        if not schema.minimum_joints <= count <= schema.maximum_joints:
            raise ValueError(
                "PX4 ArmJointState requires between "
                f"{schema.minimum_joints} and {schema.maximum_joints} arm joints; got {count}"
            )

    def encode(
        self,
        state: RobotState,
        *,
        sequence: int,
        monotonic_now_ns: int,
        wall_now_ns: int,
        timestamp_us: Optional[int] = None,
        timestamp_sample_us: Optional[int] = None,
    ) -> bytes:
        """Return one finite, zero-padded little-endian CDR sample."""

        if type(sequence) is not int or not 0 <= sequence <= 0xFFFFFFFF:
            raise ValueError("PX4 sequence must be an unsigned 32-bit integer")
        if type(monotonic_now_ns) is not int or monotonic_now_ns < 0:
            raise ValueError("monotonic_now_ns must be a non-negative integer")
        if type(wall_now_ns) is not int or wall_now_ns < 0:
            raise ValueError("wall_now_ns must be a non-negative integer")

        positions: list[float] = []
        velocities: list[float] = []
        sample_timestamps: list[int] = []
        for group_name, expected_names in self._groups:
            try:
                sample = state.joints[group_name]
            except KeyError as exc:
                raise ValueError(f"robot state is missing arm group '{group_name}'") from exc
            if sample.names != expected_names:
                raise ValueError(
                    f"arm group '{group_name}' names do not match RobotSpec order"
                )
            if sample.unit != JointUnit.RADIAN:
                raise ValueError(f"arm group '{group_name}' must use radians")
            if sample.timestamp_ns > monotonic_now_ns:
                raise ValueError(f"arm group '{group_name}' timestamp is in the future")
            positions.extend(float(value) for value in sample.positions)
            velocities.extend(float(value) for value in sample.velocities)
            sample_timestamps.append(sample.timestamp_ns)

        position_array = np.asarray(positions, dtype=float)
        velocity_array = np.asarray(velocities, dtype=float)
        if not np.all(np.isfinite(position_array)):
            raise ValueError("PX4 arm positions must be finite")
        velocity_valid = bool(np.all(np.isfinite(velocity_array)))
        if not velocity_valid:
            velocity_array = np.zeros_like(position_array)

        count = len(position_array)
        padding = self.schema.maximum_joints - count
        if (timestamp_us is None) != (timestamp_sample_us is None):
            raise ValueError("explicit PX4 timestamps must be supplied together")
        if timestamp_us is None:
            timestamp_us = wall_now_ns // 1000
            sample_age_ns = monotonic_now_ns - min(sample_timestamps)
            timestamp_sample_us = max(0, timestamp_us - sample_age_ns // 1000)
        elif (
            type(timestamp_us) is not int
            or type(timestamp_sample_us) is not int
            or timestamp_us <= 0
            or timestamp_sample_us <= 0
            or timestamp_sample_us > timestamp_us
        ):
            raise ValueError(
                "explicit PX4 timestamps must be positive integers with sample <= publish"
            )
        payload = self.schema.payload_struct.pack(
            timestamp_us,
            timestamp_sample_us,
            sequence,
            count,
            velocity_valid,
            *(position_array.tolist() + [0.0] * padding),
            *(velocity_array.tolist() + [0.0] * padding),
        )
        if len(payload) != self.schema.serialized_size:
            raise AssertionError("encoded ArmJointState has an unexpected size")
        return payload


@dataclass(frozen=True)
class Px4XrceOptions:
    """Validated process, endpoint, and compatibility settings for one bridge."""

    prefix: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "ACETELE_XRCE_PREFIX",
                Path.home() / ".local" / "lib" / "acetele" / "xrce-2.4.2",
            )
        )
    )
    agent_port: int = 8888
    domain_id: int = 0
    namespace: str = ""
    client_key: int = 0xACED0001
    startup_timeout_s: float = 3.0
    ack_timeout_ns: int = 100_000_000
    stop_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        prefix = Path(self.prefix).expanduser().resolve()
        object.__setattr__(self, "prefix", prefix)
        if type(self.agent_port) is not int or not 1 <= self.agent_port <= 65_535:
            raise ValueError("XRCE agent port must be an integer in [1, 65535]")
        if type(self.domain_id) is not int or not 0 <= self.domain_id <= 232:
            raise ValueError("XRCE domain ID must be an integer in [0, 232]")
        if not isinstance(self.namespace, str):
            raise ValueError("XRCE namespace must be a string")
        namespace = self.namespace.strip("/")
        if "//" in namespace or len(namespace) > 64:
            raise ValueError("XRCE namespace must be at most 64 characters without empty parts")
        object.__setattr__(self, "namespace", namespace)
        if type(self.client_key) is not int or not 1 <= self.client_key <= 0xFFFFFFFF:
            raise ValueError("XRCE client key must be a non-zero unsigned 32-bit integer")
        if self.client_key == 1:
            raise ValueError(
                "XRCE publisher client key must differ from PX4's default UXRCE_DDS_KEY=1"
            )
        if not math.isfinite(self.startup_timeout_s) or self.startup_timeout_s <= 0.0:
            raise ValueError("XRCE startup timeout must be finite and positive")
        if type(self.ack_timeout_ns) is not int or self.ack_timeout_ns <= 0:
            raise ValueError("XRCE ACK timeout must be a positive integer")
        if not math.isfinite(self.stop_timeout_s) or self.stop_timeout_s <= 0.0:
            raise ValueError("XRCE stop timeout must be finite and positive")

    @property
    def agent_executable(self) -> Path:
        return self.prefix / "bin" / "MicroXRCEAgent"

    @property
    def publisher_executable(self) -> Path:
        return self.prefix / "bin" / "ace-px4-xrce-publisher"

    @property
    def manifest_path(self) -> Path:
        return self.prefix / "share" / "ace-px4-xrce" / "manifest.json"


@dataclass(frozen=True)
class Px4XrceDiagnostics:
    """Read-only process and latest-value publication diagnostics."""

    sent_samples: int = 0
    acknowledged_samples: int = 0
    dropped_samples: int = 0
    last_sent_sequence: Optional[int] = None
    last_acknowledged_sequence: Optional[int] = None
    acknowledgement_age_ns: Optional[int] = None


class Px4XrceBridge:
    """Own a pinned Agent and native publisher without opening robot hardware."""

    def __init__(
        self,
        spec: RobotSpec,
        options: Px4XrceOptions,
        *,
        schema: ArmJointStateSchema = ArmJointStateSchema(),
        clock_ns: Callable[[], int] = time.monotonic_ns,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.options = options
        self.schema = schema
        self.encoder = ArmJointStateEncoder(spec, schema=schema)
        self._clock_ns = clock_ns
        self._wall_clock_ns = wall_clock_ns
        self._popen = popen_factory
        self._runtime_directory: Optional[tempfile.TemporaryDirectory[str]] = None
        self._ack_socket: Optional[socket.socket] = None
        self._state_socket: Optional[socket.socket] = None
        self._agent: Optional[subprocess.Popen] = None
        self._publisher: Optional[subprocess.Popen] = None
        self._log_files: list[BinaryIO] = []
        self._started = False
        self._sent_samples = 0
        self._acknowledged_samples = 0
        self._dropped_samples = 0
        self._last_sent_sequence: Optional[int] = None
        self._last_sent_ns: Optional[int] = None
        self._last_explicit_timestamp_us: Optional[int] = None
        self._first_unacknowledged_sent_ns: Optional[int] = None
        self._last_acknowledged_sequence: Optional[int] = None
        self._last_ack_ns: Optional[int] = None
        self._ipc_blocked_since_ns: Optional[int] = None

    def start(self) -> None:
        """Validate the pinned stack and establish XRCE entities before hardware."""

        if self._started:
            return
        # A bridge may be restarted after an orderly close. Link-health timestamps
        # belong to one native session and must not poison the replacement session.
        self._last_sent_sequence = None
        self._last_sent_ns = None
        self._first_unacknowledged_sent_ns = None
        self._last_acknowledged_sequence = None
        self._last_ack_ns = None
        self._ipc_blocked_since_ns = None
        self._validate_installation()
        self._require_agent_port_available()
        runtime_directory = tempfile.TemporaryDirectory(prefix="acetele-xrce-")
        self._runtime_directory = runtime_directory
        root = Path(runtime_directory.name)
        state_path = root / "state.sock"
        ack_path = root / "ack.sock"
        ack_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        ack_socket.bind(str(ack_path))
        ack_socket.setblocking(False)
        self._ack_socket = ack_socket
        try:
            agent_log = (root / "agent.log").open("wb")
            publisher_log = (root / "publisher.log").open("wb")
            self._log_files.extend((agent_log, publisher_log))
            self._agent = self._popen(
                (
                    str(self.options.agent_executable),
                    "udp4",
                    "-p",
                    str(self.options.agent_port),
                    "-v",
                    "4",
                ),
                stdin=subprocess.DEVNULL,
                stdout=agent_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._publisher = self._popen(
                (
                    str(self.options.publisher_executable),
                    "--agent-host",
                    "127.0.0.1",
                    "--agent-port",
                    str(self.options.agent_port),
                    "--client-key",
                    hex(self.options.client_key),
                    "--domain-id",
                    str(self.options.domain_id),
                    "--namespace",
                    self.options.namespace,
                    "--state-socket",
                    str(state_path),
                    "--ack-socket",
                    str(ack_path),
                    "--startup-timeout-ms",
                    str(round(self.options.startup_timeout_s * 1000)),
                ),
                stdin=subprocess.DEVNULL,
                stdout=publisher_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._wait_until_ready()
            state_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            state_socket.setblocking(False)
            state_socket.connect(str(state_path))
            self._state_socket = state_socket
            self._started = True
        except BaseException as exc:
            cleanup_error = self._close_resources()
            if cleanup_error is not None:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise exc from cleanup_error
                raise Px4XrceError(
                    f"could not start PX4 XRCE bridge: {exc}; "
                    f"cleanup also failed: {cleanup_error}"
                ) from exc
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, Px4XrceError):
                raise
            raise Px4XrceError(f"could not start PX4 XRCE bridge: {exc}") from exc

    def publish(
        self,
        state: RobotState,
        *,
        sequence: int,
        timestamp_us: Optional[int] = None,
        timestamp_sample_us: Optional[int] = None,
    ) -> bool:
        """Send one latest measured state without blocking the control loop."""

        self._require_healthy()
        now_ns = self._clock_ns()
        self._drain_acknowledgements(now_ns)
        self._require_fresh_ack(now_ns)
        if (
            timestamp_us is not None
            and self._last_explicit_timestamp_us is not None
            and timestamp_us <= self._last_explicit_timestamp_us
        ):
            raise Px4XrceError("explicit PX4 timestamp must increase monotonically")
        try:
            payload = self.encoder.encode(
                state,
                sequence=sequence,
                monotonic_now_ns=now_ns,
                wall_now_ns=self._wall_clock_ns(),
                timestamp_us=timestamp_us,
                timestamp_sample_us=timestamp_sample_us,
            )
        except ValueError as exc:
            raise Px4XrceError(f"PX4 arm state is invalid: {exc}") from exc
        if self._state_socket is None:
            raise Px4XrceError("PX4 XRCE state socket is not connected")
        try:
            written = self._state_socket.send(payload)
        except BlockingIOError:
            self._dropped_samples += 1
            if self._ipc_blocked_since_ns is None:
                self._ipc_blocked_since_ns = now_ns
            elif now_ns - self._ipc_blocked_since_ns > self.options.ack_timeout_ns:
                raise Px4XrceError("PX4 XRCE state IPC remained blocked")
            return False
        except OSError as exc:
            raise Px4XrceError(f"PX4 XRCE state IPC failed: {exc}") from exc
        if written != self.schema.serialized_size:
            raise Px4XrceError(
                f"PX4 XRCE state IPC wrote {written} of {self.schema.serialized_size} bytes"
            )
        self._ipc_blocked_since_ns = None
        if self._last_ack_ns is None and self._first_unacknowledged_sent_ns is None:
            self._first_unacknowledged_sent_ns = now_ns
        self._sent_samples += 1
        self._last_sent_sequence = sequence
        self._last_sent_ns = now_ns
        if timestamp_us is not None:
            self._last_explicit_timestamp_us = timestamp_us
        return True

    def diagnostics(self) -> Px4XrceDiagnostics:
        """Return an immutable snapshot without exposing process or socket objects."""

        now_ns = self._clock_ns()
        if self._started:
            self._drain_acknowledgements(now_ns)
        age = None if self._last_ack_ns is None else max(0, now_ns - self._last_ack_ns)
        return Px4XrceDiagnostics(
            self._sent_samples,
            self._acknowledged_samples,
            self._dropped_samples,
            self._last_sent_sequence,
            self._last_acknowledged_sequence,
            age,
        )

    def close(self) -> None:
        """Stop the publisher before the Agent and release every local IPC resource."""

        error = self._close_resources()
        if error is not None:
            raise Px4XrceError(f"PX4 XRCE bridge cleanup failed: {error}") from error

    def _validate_installation(self) -> None:
        for label, path in (
            ("XRCE Agent", self.options.agent_executable),
            ("XRCE publisher", self.options.publisher_executable),
            ("XRCE manifest", self.options.manifest_path),
        ):
            if not path.is_file():
                raise Px4XrceError(f"{label} is missing from prefix: {path}")
        try:
            manifest = json.loads(self.options.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise Px4XrceError("XRCE manifest is unreadable") from exc
        expected = {
            "agent_version": "2.4.2",
            "client_version": "2.4.0",
            "schema_sha256": self.schema.sha256,
        }
        if not isinstance(manifest, dict) or any(
            manifest.get(name) != value for name, value in expected.items()
        ):
            raise Px4XrceError(
                "XRCE prefix is incompatible; expected "
                + ", ".join(f"{name}={value}" for name, value in expected.items())
            )
        for path in (self.options.agent_executable, self.options.publisher_executable):
            if not os.access(path, os.X_OK):
                raise Px4XrceError(f"XRCE executable is not executable: {path}")

    def _require_agent_port_available(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind(("0.0.0.0", self.options.agent_port))
        except OSError as exc:
            raise Px4XrceError(
                f"XRCE Agent UDP port {self.options.agent_port} is already in use"
            ) from exc
        finally:
            probe.close()

    def _wait_until_ready(self) -> None:
        if self._ack_socket is None:
            raise Px4XrceError("PX4 XRCE acknowledgement socket is unavailable")
        deadline = time.monotonic() + self.options.startup_timeout_s
        while True:
            self._require_processes_running(starting=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise Px4XrceError(
                    "timed out waiting for PX4 XRCE entities"
                    + self._native_log_summary()
                )
            readable, _, _ = select.select((self._ack_socket,), (), (), remaining)
            if not readable:
                continue
            message = self._ack_socket.recv(512).decode("ascii", errors="replace").strip()
            fields = message.split()
            if fields == ["READY", self.schema.sha256, "2.4.0"]:
                return
            if fields and fields[0] == "ERROR":
                raise Px4XrceError("native XRCE publisher failed: " + " ".join(fields[1:]))
            raise Px4XrceError(f"unexpected native XRCE readiness message: {message!r}")

    def _drain_acknowledgements(self, now_ns: int) -> None:
        if self._ack_socket is None:
            return
        while True:
            try:
                message = self._ack_socket.recv(512).decode("ascii", errors="replace").strip()
            except BlockingIOError:
                return
            except OSError as exc:
                raise Px4XrceError(f"PX4 XRCE acknowledgement IPC failed: {exc}") from exc
            fields = message.split()
            if len(fields) == 3 and fields[0] == "ACK":
                try:
                    sequence = int(fields[1])
                    acknowledged_ms = int(fields[2])
                except ValueError as exc:
                    raise Px4XrceError(f"invalid XRCE acknowledgement: {message!r}") from exc
                if not 0 <= sequence <= 0xFFFFFFFF or acknowledged_ms < 0:
                    raise Px4XrceError(f"invalid XRCE acknowledgement: {message!r}")
                acknowledged_ns = acknowledged_ms * 1_000_000
                if acknowledged_ns > now_ns + 1_000_000:
                    raise Px4XrceError("XRCE acknowledgement timestamp is in the future")
                self._last_acknowledged_sequence = sequence
                self._last_ack_ns = acknowledged_ns
                self._first_unacknowledged_sent_ns = None
                self._acknowledged_samples += 1
            elif fields and fields[0] == "ERROR":
                raise Px4XrceError("native XRCE publisher failed: " + " ".join(fields[1:]))

    def _require_fresh_ack(self, now_ns: int) -> None:
        if self._last_sent_ns is None:
            return
        reference = (
            self._last_ack_ns
            if self._last_ack_ns is not None
            else self._first_unacknowledged_sent_ns
        )
        if reference is None:
            return
        if now_ns - reference > self.options.ack_timeout_ns:
            raise Px4XrceError("PX4 XRCE publisher acknowledgement is stale")

    def _require_healthy(self) -> None:
        if not self._started:
            raise Px4XrceError("PX4 XRCE bridge is not started")
        self._require_processes_running(starting=False)

    def _require_processes_running(self, *, starting: bool) -> None:
        for label, process in (("Agent", self._agent), ("publisher", self._publisher)):
            if process is None:
                raise Px4XrceError(f"PX4 XRCE {label} was not created")
            return_code = process.poll()
            if return_code is not None:
                phase = "during startup" if starting else "during operation"
                raise Px4XrceError(
                    f"PX4 XRCE {label} exited with status {return_code} {phase}"
                    + self._native_log_summary()
                )

    def _native_log_summary(self) -> str:
        """Return bounded native logs before startup cleanup removes the temp files."""

        runtime_directory = self._runtime_directory
        if runtime_directory is None:
            return ""
        sections = []
        for label in ("agent", "publisher"):
            path = Path(runtime_directory.name) / f"{label}.log"
            try:
                content = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if content:
                sections.append(f"{label}: {content[-2048:]}")
        return "" if not sections else "\nNative logs:\n" + "\n".join(sections)

    def _close_resources(self) -> Optional[BaseException]:
        first_error: Optional[BaseException] = None
        self._started = False
        for item_name in ("_state_socket", "_ack_socket"):
            item = getattr(self, item_name)
            if item is None:
                continue
            try:
                item.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            setattr(self, item_name, None)
        for process_name in ("_publisher", "_agent"):
            process = getattr(self, process_name)
            if process is None:
                continue
            try:
                self._stop_process(process)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            setattr(self, process_name, None)
        for log_file in self._log_files:
            try:
                log_file.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._log_files.clear()
        if self._runtime_directory is not None:
            try:
                self._runtime_directory.cleanup()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            self._runtime_directory = None
        return first_error

    def _stop_process(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=self.options.stop_timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=self.options.stop_timeout_s)


def xrce_manifest(options: Px4XrceOptions) -> Mapping[str, str]:
    """Expose the expected immutable native compatibility manifest for tooling."""

    schema = ArmJointStateSchema()
    return MappingProxyType(
        {
            "agent_version": "2.4.2",
            "client_version": "2.4.0",
            "schema_sha256": schema.sha256,
            "prefix": str(options.prefix),
        }
    )


__all__ = [
    "ArmJointStateEncoder",
    "ArmJointStateSchema",
    "Px4XrceBridge",
    "Px4XrceDiagnostics",
    "Px4XrceError",
    "Px4XrceOptions",
    "xrce_manifest",
]
