from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest
from ace_robot_zmq.application import FollowerApplication, LeaderApplication
from ace_robot_zmq.options import PeerRole, ZmqTeleopOptions
from ace_robot_zmq.protocol import LeaderFrame
from ace_robot_zmq.px4_xrce import Px4XrceDiagnostics, Px4XrceError
from ace_robot_zmq.transport import TransportDiagnostics

from acetele.config import load_robot_spec
from acetele.runtime import RobotRuntime
from acetele.runtime.safety import RuntimeSafetyState
from acetele.runtime.teleop import FollowerSyncStatus, LeaderSyncMode, TeleopMode
from acetele.specification import Backend

repository_root = Path(__file__).resolve().parents[3]


class RecordingPeer:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.rejections: list[str] = []
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def send(self, payload: bytes) -> bool:
        self.sent.append(payload)
        return True

    def receive(self, *, timeout_ms: int = 0):
        return None

    def record_admission(self, _admission) -> None:
        pass

    def record_rejection(self, reason: str) -> None:
        self.rejections.append(reason)

    def record_encode_duration(self, _duration_ns: int) -> None:
        pass

    def record_runtime_stage_duration(self, _duration_ns: int) -> None:
        pass

    def diagnostics(self) -> TransportDiagnostics:
        return TransportDiagnostics()

    def close(self) -> None:
        self.opened = False


class RecordingXrceBridge:
    def __init__(self) -> None:
        self.started = False
        self.published: list[tuple[object, int]] = []
        self.timestamps: list[tuple[int | None, int | None]] = []
        self.publish_error = None

    def start(self) -> None:
        self.started = True

    def publish(
        self,
        state,
        *,
        sequence: int,
        timestamp_us=None,
        timestamp_sample_us=None,
    ) -> bool:
        if self.publish_error is not None:
            raise self.publish_error
        self.published.append((state, sequence))
        self.timestamps.append((timestamp_us, timestamp_sample_us))
        return True

    def diagnostics(self) -> Px4XrceDiagnostics:
        return Px4XrceDiagnostics(sent_samples=len(self.published))

    def close(self) -> None:
        self.started = False


def _spec(name: str):
    path = (
        repository_root
        / "acetele"
        / "config"
        / "presets"
        / name
        / "feetech_hls_ttl.toml"
    )
    spec = load_robot_spec(path)
    arms = tuple(replace(arm, end_effector=None) for arm in spec.arms)
    buses = tuple(replace(bus, port=f"mock://{name}/{bus.name}") for bus in spec.buses)
    return replace(spec, backend=Backend.MOCK, arms=arms, buses=buses)


def _publish_follower(application: FollowerApplication):
    deadline = time.monotonic() + 1.0
    while True:
        try:
            return application.publish_once(now_ns=time.monotonic_ns())
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.001)


def test_zmq_follower_holds_position_before_a_leader_connects():
    follower = FollowerApplication(
        _spec("ace_follower"),
        ZmqTeleopOptions(PeerRole.FOLLOWER, "127.0.0.1", "127.0.0.1"),
        peer=RecordingPeer(),
        xrce_bridge=RecordingXrceBridge(),
    )
    follower.connect()
    try:
        assert (
            follower.session.runtime.diagnostics().safety.state
            == RuntimeSafetyState.HOLD
        )
        assert follower.session.status == FollowerSyncStatus.IDLE
    finally:
        follower.close()


@pytest.mark.parametrize("teleop_mode", (TeleopMode.JOINT, TeleopMode.EE_POSE))
def test_applications_reuse_sessions_through_tracking(teleop_mode):
    leader_peer = RecordingPeer()
    follower_peer = RecordingPeer()
    xrce = RecordingXrceBridge()
    leader = LeaderApplication(
        _spec("ace_leader"),
        ZmqTeleopOptions(PeerRole.LEADER, "127.0.0.1", "127.0.0.1"),
        teleop_mode=teleop_mode,
        peer=leader_peer,
        session_id=b"l" * 16,
    )
    follower = FollowerApplication(
        _spec("ace_follower"),
        ZmqTeleopOptions(PeerRole.FOLLOWER, "127.0.0.1", "127.0.0.1"),
        teleop_mode=teleop_mode,
        peer=follower_peer,
        session_id=b"f" * 16,
        xrce_bridge=xrce,
    )
    leader.connect()
    follower.connect()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            _publish_follower(follower)
            assert leader.process_incoming(follower_peer.sent[-1])
            leader.publish_once(now_ns=time.monotonic_ns())
            assert follower.process_incoming(leader_peer.sent[-1])
            if follower.session.status == FollowerSyncStatus.TRACKING:
                break
            time.sleep(0.01)
        assert leader.session.mode == LeaderSyncMode.TRACKING
        assert follower.session.status == FollowerSyncStatus.TRACKING
        assert not leader_peer.rejections
        assert not follower_peer.rejections
        assert isinstance(leader.diagnostics(), TransportDiagnostics)
        assert isinstance(follower.diagnostics(), TransportDiagnostics)
        assert follower.xrce_diagnostics().sent_samples == len(xrce.published)
        assert tuple(sequence for _, sequence in xrce.published) == tuple(
            range(len(xrce.published))
        )
        expired = follower.publish_once(
            now_ns=time.monotonic_ns() + follower.options.heartbeat_timeout_ns + 1
        )
        assert expired.status == FollowerSyncStatus.LOST
    finally:
        leader.close()
        follower.close()
    assert not xrce.started


def test_follower_rejects_wrong_teleop_mode_without_refreshing_session_heartbeat():
    peer = RecordingPeer()
    follower = FollowerApplication(
        _spec("ace_follower"),
        ZmqTeleopOptions(PeerRole.FOLLOWER, "127.0.0.1", "127.0.0.1"),
        teleop_mode=TeleopMode.JOINT,
        peer=peer,
        xrce_bridge=RecordingXrceBridge(),
    )
    # Construct the invalid frame without connecting either runtime.
    payload = follower.codec.encode_leader(
        LeaderFrame(
            b"p" * 16,
            0,
            1,
            LeaderSyncMode.SYNC_REQUEST,
            TeleopMode.EE_POSE,
        )
    )

    assert not follower.process_incoming(payload, now_ns=1)
    assert follower.session.mode == LeaderSyncMode.IDLE
    assert "do not match" in peer.rejections[-1]


def test_follower_xrce_failure_holds_session_before_propagating():
    bridge = RecordingXrceBridge()
    follower = FollowerApplication(
        _spec("ace_follower"),
        ZmqTeleopOptions(PeerRole.FOLLOWER, "127.0.0.1", "127.0.0.1"),
        peer=RecordingPeer(),
        xrce_bridge=bridge,
    )
    follower.connect()
    try:
        follower.session.set_mode(LeaderSyncMode.SYNC_REQUEST)
        _publish_follower(follower)
        bridge.publish_error = Px4XrceError("publisher exited")
        with pytest.raises(Px4XrceError, match="publisher exited"):
            follower.publish_once(now_ns=time.monotonic_ns())
        assert follower.session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
        assert follower.session.status == FollowerSyncStatus.IDLE
    finally:
        follower.close()


def test_follower_uses_injected_px4_clock_timestamps():
    bridge = RecordingXrceBridge()
    follower = FollowerApplication(
        _spec("ace_follower"),
        ZmqTeleopOptions(PeerRole.FOLLOWER, "127.0.0.1", "127.0.0.1"),
        peer=RecordingPeer(),
        xrce_bridge=bridge,
        xrce_timestamp_provider=lambda _state: (5000, 4000),
    )
    follower.connect()
    try:
        _publish_follower(follower)
        assert bridge.timestamps[-1] == (5000, 4000)
    finally:
        follower.close()


def test_follower_establishes_xrce_and_zmq_before_connecting_hardware():
    events = []
    spec = _spec("ace_follower")

    class OrderedPeer(RecordingPeer):
        def open(self) -> None:
            events.append("zmq")
            super().open()

    class OrderedXrce(RecordingXrceBridge):
        def start(self) -> None:
            events.append("xrce")
            super().start()

    def runtime_factory(runtime_spec, **kwargs):
        runtime = RobotRuntime(runtime_spec, **kwargs)
        connect = runtime.connect

        def ordered_connect():
            events.append("hardware")
            connect()

        runtime.connect = ordered_connect
        return runtime

    follower = FollowerApplication(
        spec,
        ZmqTeleopOptions(PeerRole.FOLLOWER, "127.0.0.1", "127.0.0.1"),
        runtime_factory=runtime_factory,
        peer=OrderedPeer(),
        xrce_bridge=OrderedXrce(),
    )
    try:
        follower.connect()
        assert events == ["xrce", "zmq", "hardware"]
    finally:
        follower.close()


def test_xrce_startup_failure_prevents_network_and_hardware_connection():
    events = []
    spec = _spec("ace_follower")

    class FailingXrce(RecordingXrceBridge):
        def start(self) -> None:
            events.append("xrce")
            raise Px4XrceError("entity creation failed")

    class UnexpectedPeer(RecordingPeer):
        def open(self) -> None:
            events.append("zmq")

    def runtime_factory(runtime_spec, **kwargs):
        runtime = RobotRuntime(runtime_spec, **kwargs)

        def unexpected_connect():
            events.append("hardware")

        runtime.connect = unexpected_connect
        return runtime

    follower = FollowerApplication(
        spec,
        ZmqTeleopOptions(PeerRole.FOLLOWER, "127.0.0.1", "127.0.0.1"),
        runtime_factory=runtime_factory,
        peer=UnexpectedPeer(),
        xrce_bridge=FailingXrce(),
    )

    with pytest.raises(Px4XrceError, match="entity creation failed"):
        follower.connect()

    assert events == ["xrce"]
    follower.close()
