"""Single-threaded direct PUB/SUB transport with latest-value semantics."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

import zmq
from ace_robot_zmq.options import PeerRole, ZmqTeleopOptions
from ace_robot_zmq.security import CurveAuthenticator


def _close_resources(
    primary: Optional[BaseException],
    callbacks: tuple[Callable[[], None], ...],
) -> None:
    """Attempt every transport cleanup while retaining the operational failure."""

    cleanup_error: Optional[BaseException] = None
    for callback in callbacks:
        try:
            callback()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
    if primary is not None:
        if cleanup_error is not None:
            raise primary from cleanup_error
        raise primary
    if cleanup_error is not None:
        raise cleanup_error


@dataclass(frozen=True)
class TransportDiagnostics:
    """Read-only counters and timing from one direct peer connection."""

    sent_frames: int = 0
    received_frames: int = 0
    rejected_frames: int = 0
    sequence_gaps: int = 0
    session_changes: int = 0
    send_drops: int = 0
    last_receive_ns: Optional[int] = None
    last_rejection: Optional[str] = None
    last_send_duration_ns: Optional[int] = None
    last_receive_duration_ns: Optional[int] = None
    last_encode_duration_ns: Optional[int] = None
    last_receive_to_runtime_stage_ns: Optional[int] = None

    def receive_age_ns(self, *, now_ns: Optional[int] = None) -> Optional[int]:
        """Return local age without comparing clocks from different hosts."""

        if self.last_receive_ns is None:
            return None
        current = time.monotonic_ns() if now_ns is None else now_ns
        return max(0, current - self.last_receive_ns)


@dataclass(frozen=True)
class SequenceAdmission:
    """Result of validating one decoded frame against the active peer lease."""

    accepted: bool
    new_session: bool = False
    sequence_gap: int = 0
    reason: Optional[str] = None


class PeerSequenceGate:
    """Reject stale sessions and require heartbeat expiry before peer replacement."""

    def __init__(self, *, heartbeat_timeout_ns: int) -> None:
        if type(heartbeat_timeout_ns) is not int or heartbeat_timeout_ns <= 0:
            raise ValueError("heartbeat_timeout_ns must be a positive integer")
        self._heartbeat_timeout_ns = heartbeat_timeout_ns
        self._session_id: Optional[bytes] = None
        self._sequence: Optional[int] = None
        self._last_valid_ns: Optional[int] = None
        self._retired: list[bytes] = []

    @property
    def session_id(self) -> Optional[bytes]:
        return self._session_id

    def admit(self, session_id: bytes, sequence: int, *, now_ns: int) -> SequenceAdmission:
        if not isinstance(session_id, bytes) or len(session_id) != 16:
            return SequenceAdmission(False, reason="invalid session_id")
        if type(sequence) is not int or sequence < 0:
            return SequenceAdmission(False, reason="invalid sequence")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("sequence-gate time must be a non-negative integer")
        if session_id == self._session_id:
            if self._sequence is not None and sequence <= self._sequence:
                return SequenceAdmission(False, reason="duplicate or out-of-order sequence")
            gap = 0 if self._sequence is None else max(0, sequence - self._sequence - 1)
            self._sequence = sequence
            self._last_valid_ns = now_ns
            return SequenceAdmission(True, sequence_gap=gap)
        if session_id in self._retired:
            return SequenceAdmission(False, reason="retired session")
        if (
            self._session_id is not None
            and self._last_valid_ns is not None
            and now_ns - self._last_valid_ns <= self._heartbeat_timeout_ns
        ):
            return SequenceAdmission(False, reason="another peer session is active")
        if self._session_id is not None:
            self._retired.append(self._session_id)
            del self._retired[:-8]
        self._session_id = session_id
        self._sequence = sequence
        self._last_valid_ns = now_ns
        return SequenceAdmission(True, new_session=True)


class ZmqPeer:
    """Own exactly one publishing socket, one subscribing socket, and their context."""

    def __init__(
        self,
        options: ZmqTeleopOptions,
        *,
        context_factory: Callable[[], zmq.Context] = zmq.Context,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(options, ZmqTeleopOptions):
            raise ValueError("ZmqPeer requires ZmqTeleopOptions")
        self.options = options
        self._context_factory = context_factory
        self._clock_ns = clock_ns
        self._context: Optional[zmq.Context] = None
        self._publisher: Optional[zmq.Socket] = None
        self._subscriber: Optional[zmq.Socket] = None
        self._poller: Optional[zmq.Poller] = None
        self._authenticator: Optional[CurveAuthenticator] = None
        self._diagnostics = TransportDiagnostics()

    def open(self) -> None:
        """Create sockets only after all options and certificates have validated."""

        if self._context is not None:
            return
        context = self._context_factory()
        authenticator = None
        publisher = None
        subscriber = None
        try:
            if self.options.curve is not None:
                authenticator = CurveAuthenticator(context, self.options.curve)
            publisher = context.socket(zmq.PUB)
            subscriber = context.socket(zmq.SUB)
            for socket in (publisher, subscriber):
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.CONFLATE, 1)
                socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
                socket.setsockopt(zmq.IMMEDIATE, 1)
            subscriber.setsockopt(zmq.SUBSCRIBE, b"")
            if authenticator is not None:
                authenticator.configure_server(publisher)
                authenticator.configure_client(subscriber)
            if self.options.role == PeerRole.LEADER:
                publisher.bind(self.options.command_endpoint)
                subscriber.connect(self.options.state_endpoint)
            else:
                publisher.bind(self.options.state_endpoint)
                subscriber.connect(self.options.command_endpoint)
            poller = zmq.Poller()
            poller.register(subscriber, zmq.POLLIN)
        except BaseException as exc:
            callbacks: list[Callable[[], None]] = []
            if publisher is not None:
                callbacks.append(lambda: publisher.close(linger=0))
            if subscriber is not None:
                callbacks.append(lambda: subscriber.close(linger=0))
            if authenticator is not None:
                callbacks.append(authenticator.close)
            callbacks.append(context.term)
            _close_resources(exc, tuple(callbacks))
        self._context = context
        self._publisher = publisher
        self._subscriber = subscriber
        self._poller = poller
        self._authenticator = authenticator

    def send(self, payload: bytes) -> bool:
        """Publish one atomic snapshot without waiting for a subscriber or queue space."""

        if not isinstance(payload, bytes) or not payload:
            raise ValueError("ZMQ payload must be non-empty bytes")
        if len(payload) > self.options.maximum_frame_bytes:
            raise ValueError("ZMQ payload exceeds maximum_frame_bytes")
        publisher = self._require_publisher()
        started_ns = self._clock_ns()
        try:
            publisher.send(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            self._diagnostics = replace(
                self._diagnostics,
                send_drops=self._diagnostics.send_drops + 1,
                last_send_duration_ns=self._clock_ns() - started_ns,
            )
            return False
        self._diagnostics = replace(
            self._diagnostics,
            sent_frames=self._diagnostics.sent_frames + 1,
            last_send_duration_ns=self._clock_ns() - started_ns,
        )
        return True

    def receive(self, *, timeout_ms: int = 0) -> Optional[bytes]:
        """Return the newest complete frame, or ``None`` when the deadline expires."""

        if type(timeout_ms) is not int or timeout_ms < 0:
            raise ValueError("timeout_ms must be a non-negative integer")
        subscriber = self._require_subscriber()
        poller = self._poller
        if poller is None:
            raise RuntimeError("ZMQ peer is not open")
        started_ns = self._clock_ns()
        events = dict(poller.poll(timeout_ms))
        if subscriber not in events:
            return None
        payload = subscriber.recv(flags=zmq.NOBLOCK)
        received_ns = self._clock_ns()
        self._diagnostics = replace(
            self._diagnostics,
            received_frames=self._diagnostics.received_frames + 1,
            last_receive_ns=received_ns,
            last_receive_duration_ns=received_ns - started_ns,
        )
        return payload

    def record_admission(self, admission: SequenceAdmission) -> None:
        """Accumulate schema-independent session ordering diagnostics."""

        if admission.accepted:
            self._diagnostics = replace(
                self._diagnostics,
                sequence_gaps=self._diagnostics.sequence_gaps + admission.sequence_gap,
                session_changes=(
                    self._diagnostics.session_changes + int(admission.new_session)
                ),
            )
            return
        self.record_rejection(admission.reason or "frame rejected")

    def record_rejection(self, reason: str) -> None:
        self._diagnostics = replace(
            self._diagnostics,
            rejected_frames=self._diagnostics.rejected_frames + 1,
            last_rejection=str(reason),
        )

    def record_encode_duration(self, duration_ns: int) -> None:
        """Record local serialization cost without exposing mutable counters."""

        if type(duration_ns) is not int or duration_ns < 0:
            raise ValueError("encode duration must be a non-negative integer")
        self._diagnostics = replace(
            self._diagnostics,
            last_encode_duration_ns=duration_ns,
        )

    def record_runtime_stage_duration(self, duration_ns: int) -> None:
        """Record frame admission through latest-value mailbox staging."""

        if type(duration_ns) is not int or duration_ns < 0:
            raise ValueError("runtime stage duration must be a non-negative integer")
        self._diagnostics = replace(
            self._diagnostics,
            last_receive_to_runtime_stage_ns=duration_ns,
        )

    def diagnostics(self) -> TransportDiagnostics:
        return self._diagnostics

    def close(self) -> None:
        """Idempotently release sockets, authentication, and the private context."""

        publisher, subscriber = self._publisher, self._subscriber
        context, authenticator = self._context, self._authenticator
        self._publisher = None
        self._subscriber = None
        self._poller = None
        self._context = None
        self._authenticator = None
        callbacks: list[Callable[[], None]] = []
        if publisher is not None:
            callbacks.append(lambda: publisher.close(linger=0))
        if subscriber is not None:
            callbacks.append(lambda: subscriber.close(linger=0))
        if authenticator is not None:
            callbacks.append(authenticator.close)
        if context is not None:
            callbacks.append(context.term)
        _close_resources(None, tuple(callbacks))

    def _require_publisher(self) -> zmq.Socket:
        if self._publisher is None:
            raise RuntimeError("ZMQ peer is not open")
        return self._publisher

    def _require_subscriber(self) -> zmq.Socket:
        if self._subscriber is None:
            raise RuntimeError("ZMQ peer is not open")
        return self._subscriber


__all__ = [
    "PeerSequenceGate",
    "SequenceAdmission",
    "TransportDiagnostics",
    "ZmqPeer",
]
