from __future__ import annotations

import socket
import time

import pytest
from ace_robot_zmq.options import CurveCredentials, PeerRole, ZmqTeleopOptions
from ace_robot_zmq.security import generate_curve_certificates
from ace_robot_zmq.transport import PeerSequenceGate, ZmqPeer


def _port() -> int:
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _options(role: PeerRole, command_port: int, state_port: int, *, curve=None):
    return ZmqTeleopOptions(
        role,
        "127.0.0.1",
        "127.0.0.1",
        command_port,
        state_port,
        curve=curve,
    )


def _receive(peer: ZmqPeer, expected: bytes) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        payload = peer.receive(timeout_ms=20)
        if payload == expected:
            return
    raise AssertionError(f"did not receive {expected!r}")


def test_direct_peers_exchange_latest_single_frames_in_both_directions():
    command_port, state_port = _port(), _port()
    leader = ZmqPeer(_options(PeerRole.LEADER, command_port, state_port))
    follower = ZmqPeer(_options(PeerRole.FOLLOWER, command_port, state_port))
    leader.open()
    follower.open()
    try:
        # PUB/SUB subscription propagation is asynchronous. Repeating current state is
        # part of the protocol and avoids a separate readiness handshake.
        for _ in range(20):
            leader.send(b"leader-latest")
            follower.send(b"follower-latest")
            if follower.receive(timeout_ms=10) == b"leader-latest" and leader.receive(
                timeout_ms=10
            ) == b"follower-latest":
                break
        else:
            pytest.fail("direct peers did not establish both subscriptions")
    finally:
        leader.close()
        follower.close()


def test_sequence_gate_rejects_replay_and_requires_expiry_for_new_session():
    gate = PeerSequenceGate(heartbeat_timeout_ns=100)
    first = b"a" * 16
    second = b"b" * 16

    assert gate.admit(first, 1, now_ns=0).new_session
    assert gate.admit(first, 3, now_ns=10).sequence_gap == 1
    assert not gate.admit(first, 3, now_ns=20).accepted
    assert not gate.admit(second, 0, now_ns=100).accepted
    assert gate.admit(second, 1, now_ns=111).new_session
    assert not gate.admit(first, 4, now_ns=500).accepted


def test_curve_peers_exchange_only_with_the_configured_certificates(tmp_path):
    leader_public, leader_secret = generate_curve_certificates(tmp_path, "leader")
    follower_public, follower_secret = generate_curve_certificates(tmp_path, "follower")
    command_port, state_port = _port(), _port()
    leader_curve = CurveCredentials(leader_secret, follower_public)
    follower_curve = CurveCredentials(follower_secret, leader_public)
    leader = ZmqPeer(
        _options(PeerRole.LEADER, command_port, state_port, curve=leader_curve)
    )
    follower = ZmqPeer(
        _options(PeerRole.FOLLOWER, command_port, state_port, curve=follower_curve)
    )
    leader.open()
    follower.open()
    try:
        for _ in range(40):
            leader.send(b"authenticated")
            if follower.receive(timeout_ms=20) == b"authenticated":
                break
        else:
            pytest.fail("CURVE peers did not authenticate")
    finally:
        leader.close()
        follower.close()


def test_curve_server_rejects_a_peer_with_an_unknown_key(tmp_path):
    leader_public, leader_secret = generate_curve_certificates(tmp_path, "leader")
    follower_public, _ = generate_curve_certificates(tmp_path, "follower")
    _, intruder_secret = generate_curve_certificates(tmp_path, "intruder")
    command_port, state_port = _port(), _port()
    leader = ZmqPeer(
        _options(
            PeerRole.LEADER,
            command_port,
            state_port,
            curve=CurveCredentials(leader_secret, follower_public),
        )
    )
    intruder = ZmqPeer(
        _options(
            PeerRole.FOLLOWER,
            command_port,
            state_port,
            curve=CurveCredentials(intruder_secret, leader_public),
        )
    )
    leader.open()
    intruder.open()
    try:
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            leader.send(b"must-not-arrive")
            assert intruder.receive(timeout_ms=10) is None
    finally:
        leader.close()
        intruder.close()


def test_close_attempts_every_resource_after_an_earlier_failure():
    calls = []

    class Socket:
        def __init__(self, name, *, fails=False):
            self.name = name
            self.fails = fails

        def close(self, *, linger):
            calls.append((self.name, linger))
            if self.fails:
                raise RuntimeError(f"{self.name} failed")

    class Context:
        def term(self):
            calls.append(("context", None))

    peer = ZmqPeer(_options(PeerRole.LEADER, _port(), _port()))
    peer._publisher = Socket("publisher", fails=True)  # type: ignore[assignment]
    peer._subscriber = Socket("subscriber")  # type: ignore[assignment]
    peer._context = Context()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="publisher failed"):
        peer.close()

    assert calls == [
        ("publisher", 0),
        ("subscriber", 0),
        ("context", None),
    ]
