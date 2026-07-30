from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from acetele.config.specs import ArmSpec, BusSpec, BusType, JointSpec, RobotSpec
from acetele.core import Backend
from acetele.runtime import LeaderTeleopSession, RobotRuntime, RuntimeSafetyState
from acetele.utils.teleop_sync import FollowerSyncStatus, LeaderSyncMode

urdf_path = (
    Path(__file__).resolve().parents[2]
    / "acetele"
    / "model"
    / "robots"
    / "ace_leader"
    / "description"
    / "ace_leader.urdf"
)


def _session(*, backend: Backend = Backend.MOCK) -> LeaderTeleopSession:
    spec = RobotSpec(
        "ace_leader",
        (
            BusSpec(
                "arm",
                BusType.FEETECH_PACKET,
                "mock://arm",
                1_000_000,
                100.0,
                physical_layer="ttl",
                family="hls",
                external_estop=backend == Backend.PHYSICAL,
            ),
        ),
        (
            ArmSpec(
                "single",
                "arm",
                (
                    JointSpec(
                        "joint_1",
                        0,
                        "HL3960",
                        1,
                        0.0,
                        expected_model_number=100,
                    ),
                ),
            ),
        ),
        backend=backend,
        urdf_path=str(urdf_path),
    )
    return LeaderTeleopSession(RobotRuntime(spec))


def _wait_for_state(session: LeaderTeleopSession) -> None:
    deadline = time.monotonic() + 1.0
    while True:
        try:
            session.runtime.read()
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.001)


def test_leader_session_aligns_then_releases_torque_for_tracking():
    session = _session()
    session.connect()
    try:
        _wait_for_state(session)
        now_ns = time.monotonic_ns()
        session.observe_follower_state(("joint_1",), (0.0,), now_ns=now_ns)
        session.observe_follower_status(
            FollowerSyncStatus.READY,
            now_ns=now_ns,
        )

        session.step(now_ns=now_ns)
        session.step(now_ns=now_ns + 200_000_001)
        assert session.mode == LeaderSyncMode.READY

        session.start_tracking()
        assert session.mode == LeaderSyncMode.TRACKING
        assert (
            session.runtime.diagnostics().safety.state
            == RuntimeSafetyState.SAFE_DISABLED
        )
    finally:
        session.close()


def test_leader_session_requests_resync_when_follower_heartbeat_expires():
    session = _session()
    session.connect()
    try:
        _wait_for_state(session)
        now_ns = time.monotonic_ns()
        session.observe_follower_state(("joint_1",), (0.0,), now_ns=now_ns)
        session.observe_follower_status(
            FollowerSyncStatus.READY,
            now_ns=now_ns,
        )
        session.step(now_ns=now_ns)
        session.step(now_ns=now_ns + 200_000_001)
        session.start_tracking()

        session.step(now_ns=now_ns + 500_000_001)
        assert session.mode == LeaderSyncMode.SYNC_REQUEST
    finally:
        session.close()


def test_leader_session_applies_explicit_sync_motion_limits():
    base = _session(backend=Backend.PHYSICAL)
    session = LeaderTeleopSession(
        base.runtime,
        sync_velocity_limit_rad_s=1.25,
        sync_acceleration_limit_rad_s2=2.5,
    )
    session._sync_target = np.array([0.0])  # noqa: SLF001

    command = session._target_command(1)  # noqa: SLF001
    joint_command = command.joints["single"]

    assert joint_command.velocity_limits.tolist() == [1.25]
    assert joint_command.acceleration_limits.tolist() == [2.5]


@pytest.mark.parametrize("value", (0.0, -1.0, float("inf"), float("nan")))
def test_leader_session_rejects_invalid_sync_motion_limits(value):
    with pytest.raises(ValueError, match="finite and positive"):
        LeaderTeleopSession(
            _session().runtime,
            sync_velocity_limit_rad_s=value,
        )


def test_leader_stop_latches_software_stop_before_hardware_failure(monkeypatch):
    session = _session()
    session.mode = LeaderSyncMode.TRACKING
    monkeypatch.setattr(
        session.runtime,
        "emergency_stop",
        lambda: (_ for _ in ()).throw(RuntimeError("hardware stop failed")),
    )

    with pytest.raises(RuntimeError, match="hardware stop failed"):
        session.stop()

    assert session.mode == LeaderSyncMode.STOP


def test_initial_landed_state_does_not_stop_leader_before_flight(monkeypatch):
    session = _session()
    stops = []
    monkeypatch.setattr(session, "stop", lambda: stops.append(True))

    assert not session.observe_landed(True)
    assert stops == []


def test_landing_edge_stops_leader_after_airborne_state(monkeypatch):
    session = _session()
    stops = []
    monkeypatch.setattr(session, "stop", lambda: stops.append(True))

    assert not session.observe_landed(False)
    assert session.observe_landed(True)
    assert stops == [True]
    assert not session.observe_landed(True)


@pytest.mark.parametrize("landed", (0, 1, None, "true"))
def test_landed_state_requires_a_real_boolean(landed):
    with pytest.raises(ValueError, match="boolean"):
        _session().observe_landed(landed)
