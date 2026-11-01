"""EAA mock follower tests following the shared follower-session test structure.

Run later from the repository root using the acetele Python environment:
    python -m pytest tests/runtime/teleop/test_eaa_follower.py -q

The TOML and URDF are loaded unchanged. The motion test requires finite positive
URDF velocity limits; it reports unfinished limits instead of replacing them.
Only in-memory mock devices are permitted. No dynamics or collisions are simulated.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from acetele.config import load_robot_spec
from acetele.control import StreamingPositionTuning
from acetele.model import load_urdf_model
from acetele.runtime import FollowerTeleopSession, RobotRuntime, RuntimeSafetyState
from acetele.runtime.teleop import FollowerSyncStatus, LeaderSyncMode
from acetele.specification import Backend, BusType

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    REPO_ROOT / "acetele/config/presets/eaa_follower/fashionstar_rs485.toml"
)
JOINT_NAMES = ("Joint_1", "Joint_2", "Joint_3", "Joint_4")
SESSION_TIMEOUT_NS = 500_000_000


@pytest.fixture
def eaa_spec():
    """Use the real EAA configuration without silently changing its backend."""

    spec = load_robot_spec(CONFIG_PATH)
    assert spec.backend == Backend.MOCK, (
        "EAA tests require backend = 'mock'; hardware is never opened."
    )
    return spec


@pytest.fixture
def eaa_runtime(eaa_spec):
    """Reject serial creation even if an adapter accidentally takes a physical path."""

    transport_factory = Mock(
        side_effect=AssertionError("A mock test attempted to create a serial port.")
    )
    runtime = RobotRuntime(
        eaa_spec,
        transport_factory=transport_factory,
        command_timeout_ns=500_000_000,
    )
    try:
        yield runtime
    finally:
        try:
            runtime.disconnect()
        finally:
            transport_factory.assert_not_called()


def _session(runtime, *, clock_ns=time.monotonic_ns) -> FollowerTeleopSession:
    """Compose the same session as test_follower.py, using the EAA runtime."""

    return FollowerTeleopSession(
        runtime,
        session_timeout_ns=SESSION_TIMEOUT_NS,
        motion_cycle_hz=None,  # Each test explicitly advances the local motion loop.
        clock_ns=clock_ns,
        motion_tuning=StreamingPositionTuning(
            velocity_limit_rad_s=0.2,
            acceleration_limit_rad_s2=0.5,
        ),
    )


def _read_until_available(session, *, clock_ns=time.monotonic_ns):
    """Wait for the first actor sample, as in the shared follower tests."""

    deadline = time.monotonic() + 2.0
    while True:
        try:
            return session.read(now_ns=clock_ns())
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


@pytest.fixture
def eaa_session(eaa_runtime):
    session = _session(eaa_runtime)
    try:
        session.connect()
        _read_until_available(session)
        yield session
    finally:
        session.close()


def _start_tracking(session):
    session.set_mode(LeaderSyncMode.SYNC_REQUEST)
    session.set_mode(LeaderSyncMode.TRACKING)


def test_eaa_configuration_matches_the_urdf(eaa_spec, eaa_runtime):
    assert eaa_spec.model == "eaa_follower"
    assert len(eaa_spec.arms) == 1
    arm = eaa_spec.arms[0]
    assert arm.name == "single"
    assert tuple(joint.name for joint in arm.joints) == JOINT_NAMES
    assert len({joint.servo_id for joint in arm.joints}) == 4
    assert all(bus.type == BusType.FASHIONSTAR_RS485 for bus in eaa_spec.buses)
    expected_urdf = (
        REPO_ROOT / "acetele/model/robots/eaa_follower/description/eaa_follower.urdf"
    )
    assert Path(eaa_spec.urdf_path).resolve() == expected_urdf.resolve()
    model = load_urdf_model(eaa_spec.urdf_path)
    assert model.root_link == "base_link"
    model.require_frame(arm.tool_frame)
    metadata = model.arm_metadata(JOINT_NAMES)
    assert metadata.joint_names == JOINT_NAMES
    home = np.asarray([joint.home_position_rad for joint in arm.joints])
    assert np.all(home >= metadata.lower_limits)
    assert np.all(home <= metadata.upper_limits)
    assert all(bus.budget.feasible for bus in eaa_runtime.preflight.buses.values())
    assert not eaa_runtime.connected


def test_eaa_follower_can_hold_its_measured_pose_without_a_leader(eaa_session):
    session = eaa_session
    measured = session.read(now_ns=time.monotonic_ns()).joints["single"]
    assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.SAFE_DISABLED
    session.hold_position()
    assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
    assert session.status == FollowerSyncStatus.IDLE
    after = session.read(now_ns=time.monotonic_ns()).joints["single"]
    np.testing.assert_allclose(after.positions, measured.positions)
    session.hold_position()
    assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD


def test_eaa_follower_requires_sync_before_accepting_arm_commands(eaa_session):
    session = eaa_session
    positions = session.read(now_ns=time.monotonic_ns()).joints["single"].positions
    assert not session.write_arm(JOINT_NAMES, positions, now_ns=time.monotonic_ns())
    session.set_mode(LeaderSyncMode.SYNC_REQUEST)
    assert session.status == FollowerSyncStatus.READY
    assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
    session.set_mode(LeaderSyncMode.TRACKING)
    assert session.write_arm(JOINT_NAMES, positions, now_ns=time.monotonic_ns())
    assert session.status == FollowerSyncStatus.TRACKING


@pytest.mark.parametrize(
    "invalid_names",
    [tuple(name.lower() for name in JOINT_NAMES), tuple(reversed(JOINT_NAMES))],
    ids=["wrong-case", "wrong-order"],
)
def test_eaa_follower_rejects_wrong_joint_names_without_heartbeat(
    eaa_session, invalid_names
):
    session = eaa_session
    positions = session.read(now_ns=time.monotonic_ns()).joints["single"].positions
    _start_tracking(session)
    with pytest.raises(ValueError, match="joint order"):
        session.write_arm(invalid_names, positions, now_ns=time.monotonic_ns())
    assert session.status == FollowerSyncStatus.READY


def test_eaa_mock_joint_feedback_converges_to_a_bounded_target(eaa_runtime):
    metadata = eaa_runtime.preflight.arms["single"]
    velocity = np.asarray(metadata.velocity_limits)
    assert np.all(np.isfinite(velocity) & (velocity > 0.0)), (
        "EAA URDF requires finite positive velocity limits (rad/s); "
        "velocity=0 prevents streaming motion. Update the model limits first."
    )
    session = _session(eaa_runtime)
    try:
        session.connect()
        initial = _read_until_available(session).joints["single"].positions
        lower = np.asarray(metadata.lower_limits)
        upper = np.asarray(metadata.upper_limits)
        # Move every joint a small distance toward the interior of its own range.
        midpoint = 0.5 * (lower + upper)
        delta = np.where(midpoint >= initial, 0.03, -0.03)
        target = np.clip(initial + delta, lower, upper)
        assert np.all(np.abs(target - initial) > 0.01)
        _start_tracking(session)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            assert session.write_arm(JOINT_NAMES, target, now_ns=time.monotonic_ns())
            state = session.read(now_ns=time.monotonic_ns()).joints["single"]
            assert np.all(state.positions >= lower - 1e-8)
            assert np.all(state.positions <= upper + 1e-8)
            if np.allclose(state.positions, target, atol=0.002, rtol=0.0):
                assert session.status == FollowerSyncStatus.TRACKING
                return
            time.sleep(0.01)
        np.testing.assert_allclose(state.positions, target, atol=0.002, rtol=0.0)
    finally:
        session.close()


def test_eaa_follower_loss_timeout_holds_and_requires_resynchronization(eaa_runtime):
    # Only the session lease uses this controllable clock; actors retain real time.
    clock = [time.monotonic_ns()]
    session = _session(eaa_runtime, clock_ns=lambda: clock[0])
    try:
        session.connect()
        positions = _read_until_available(
            session, clock_ns=lambda: clock[0]
        ).joints["single"].positions
        _start_tracking(session)
        assert session.write_arm(JOINT_NAMES, positions, now_ns=clock[0])
        clock[0] += SESSION_TIMEOUT_NS + 1
        assert session.update(now_ns=clock[0]) == FollowerSyncStatus.LOST
        assert not session.write_arm(JOINT_NAMES, positions, now_ns=clock[0])
        assert session.step_motion(now_ns=clock[0])
        clock[0] += 100_000_001
        assert session.step_motion(now_ns=clock[0])
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
        _start_tracking(session)
        assert session.write_arm(JOINT_NAMES, positions, now_ns=clock[0])
        assert session.status == FollowerSyncStatus.TRACKING
    finally:
        session.close()
