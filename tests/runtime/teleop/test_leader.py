from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from acetele.core import JointState, RobotState
from acetele.runtime import LeaderTeleopSession, RobotRuntime, RuntimeSafetyState
from acetele.runtime.teleop import FollowerSyncStatus, LeaderSyncMode, TeleopMode
from acetele.specification import (
    ArmSpec,
    Backend,
    BusSpec,
    BusType,
    JointSpec,
    ParallelGripperSpec,
    RobotSpec,
)

urdf_path = (
    Path(__file__).resolve().parents[3]
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


def test_leader_peer_reset_discards_alignment_and_releases_all_joints():
    session = _session()
    session.connect()
    try:
        _wait_for_state(session)
        now_ns = time.monotonic_ns()
        session.observe_follower_state(("joint_1",), (0.1,), now_ns=now_ns)
        session.step(now_ns=now_ns)

        session.reset_peer()

        assert session.mode == LeaderSyncMode.IDLE
        assert session.follower_status == FollowerSyncStatus.IDLE
        assert (
            session.runtime.diagnostics().safety.state
            == RuntimeSafetyState.SAFE_DISABLED
        )
    finally:
        session.close()


def test_leader_alignment_enables_arm_but_leaves_gripper_trigger_passive():
    joints = tuple(
        JointSpec(f"joint_{index}", index - 1, "HL3915", 1, 0.0)
        for index in range(1, 5)
    )
    gripper = ParallelGripperSpec(
        "arm",
        JointSpec("joint_5", 4, "HL3915", 1, 0.0),
        math.pi / 4.0,
    )
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
            ),
        ),
        (ArmSpec("single", "arm", joints, end_effector=gripper),),
        backend=Backend.MOCK,
        urdf_path=str(urdf_path),
    )
    session = LeaderTeleopSession(RobotRuntime(spec))
    session.connect()
    try:
        _wait_for_state(session)
        now_ns = time.monotonic_ns()
        session.observe_follower_state(
            tuple(joint.name for joint in joints),
            (0.0, 0.0, 0.0, 0.0),
            now_ns=now_ns,
        )
        state = session.step(now_ns=now_ns)

        actor = session.runtime._actors["arm"]  # noqa: SLF001
        protocol = actor._protocol  # noqa: SLF001
        assert protocol._enabled_ids == {0, 1, 2, 3}  # noqa: SLF001

        state = session.step(now_ns=now_ns + 200_000_001)
        assert session.mode == LeaderSyncMode.READY

        group_name = "single.end_effector"
        released = state.joints[group_name]
        closed = JointState(
            released.names,
            [0.9],
            released.velocities,
            released.efforts,
            released.timestamp_ns,
            released.sequence,
            released.unit,
        )
        triggered_state = RobotState(
            {**state.joints, group_name: closed},
            state.sensors,
        )
        assert not session.try_start_tracking(triggered_state)
        assert not session.try_start_tracking(state)
        assert session.try_start_tracking(triggered_state)
        assert session.mode == LeaderSyncMode.TRACKING
        assert protocol._enabled_ids == set()  # noqa: SLF001
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


def test_leader_cartesian_mode_exposes_tool_pose_from_current_state():
    joints = tuple(
        JointSpec(f"joint_{index}", index - 1, "HL3915", 1, 0.0)
        for index in range(1, 5)
    )
    spec = RobotSpec(
        "ace_leader",
        (
            BusSpec(
                "arm",
                BusType.FEETECH_PACKET,
                "mock://cartesian-leader",
                1_000_000,
                100.0,
                physical_layer="ttl",
                family="hls",
            ),
        ),
        (ArmSpec("single", "arm", joints, tool_frame="link_5"),),
        backend=Backend.MOCK,
        urdf_path=str(urdf_path),
    )
    session = LeaderTeleopSession(
        RobotRuntime(spec),
        teleop_mode=TeleopMode.EE_POSE,
    )
    session.connect()
    try:
        _wait_for_state(session)
        state = session.runtime.read()
        pose = session.end_effector_pose(state, timestamp_ns=7)

        assert pose.timestamp_ns == 7
        assert pose.frame_id == "base_link"
        assert np.all(np.isfinite(pose.position_m))
    finally:
        session.close()
