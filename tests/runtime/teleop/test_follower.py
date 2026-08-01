from __future__ import annotations

import time
from pathlib import Path

import pytest

from acetele.model import ArmKinematics
from acetele.runtime import (
    FollowerTeleopSession,
    RobotRuntime,
    RuntimeSafetyState,
)
from acetele.runtime.teleop import FollowerSyncStatus, LeaderSyncMode, TeleopMode
from acetele.specification import (
    ArmSpec,
    Backend,
    BusSpec,
    BusType,
    DexterousHandSpec,
    JointSpec,
    RobotSpec,
)

urdf_path = (
    Path(__file__).resolve().parents[3]
    / "acetele"
    / "model"
    / "robots"
    / "ace_follower"
    / "description"
    / "ace_follower.urdf"
)


def _session(
    *,
    heartbeat_timeout_ns: int = 100_000_000,
) -> FollowerTeleopSession:
    spec = RobotSpec(
        "ace_follower",
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
        backend=Backend.MOCK,
        urdf_path=str(urdf_path),
    )
    runtime = RobotRuntime(spec, command_timeout_ns=heartbeat_timeout_ns)
    return FollowerTeleopSession(
        runtime,
        heartbeat_timeout_ns=heartbeat_timeout_ns,
    )


def _read_until_available(session: FollowerTeleopSession) -> None:
    deadline = time.monotonic() + 1.0
    while True:
        try:
            session.read(now_ns=time.monotonic_ns())
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.001)


def _cartesian_session() -> FollowerTeleopSession:
    joints = tuple(
        JointSpec(f"joint_{index}", index - 1, "HL3915", 1, 0.0)
        for index in range(1, 5)
    )
    spec = RobotSpec(
        "ace_follower",
        (
            BusSpec(
                "arm",
                BusType.FEETECH_PACKET,
                "mock://cartesian-arm",
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
    return FollowerTeleopSession(
        RobotRuntime(spec),
        teleop_mode=TeleopMode.EE_POSE,
    )


def test_follower_can_hold_its_measured_pose_without_a_leader():
    session = _session()
    session.connect()
    try:
        session.hold_position()

        assert (
            session.runtime.diagnostics().safety.state
            == RuntimeSafetyState.HOLD
        )
        assert session.status == FollowerSyncStatus.IDLE
        session.hold_position()
    finally:
        session.close()


def test_follower_session_requires_current_sync_cycle_arm_heartbeat():
    session = _session()
    session.connect()
    try:
        _read_until_available(session)
        now_ns = time.monotonic_ns()
        assert not session.write_arm(("joint_1",), (0.1,), now_ns=now_ns)

        session.set_mode(LeaderSyncMode.SYNC_REQUEST)
        assert session.status == FollowerSyncStatus.READY
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
        session.set_mode(LeaderSyncMode.TRACKING)

        assert session.write_arm(
            ("joint_1",),
            (0.1,),
            now_ns=time.monotonic_ns(),
        )
        assert session.status == FollowerSyncStatus.TRACKING
    finally:
        session.close()


def test_follower_session_rejects_wrong_joint_order_without_heartbeat():
    session = _session()
    session.connect()
    try:
        _read_until_available(session)
        session.set_mode(LeaderSyncMode.SYNC_REQUEST)
        session.set_mode(LeaderSyncMode.TRACKING)

        with pytest.raises(ValueError, match="joint order"):
            session.write_arm(("wrong",), (0.1,), now_ns=time.monotonic_ns())

        assert session.status == FollowerSyncStatus.READY
    finally:
        session.close()


def test_follower_session_heartbeat_timeout_holds_runtime():
    session = _session()
    session.connect()
    try:
        _read_until_available(session)
        session.set_mode(LeaderSyncMode.SYNC_REQUEST)
        session.set_mode(LeaderSyncMode.TRACKING)
        now_ns = time.monotonic_ns()
        session.write_arm(("joint_1",), (0.1,), now_ns=now_ns)

        assert session.update(now_ns=now_ns + 100_000_001) == FollowerSyncStatus.LOST
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
    finally:
        session.close()


def test_follower_peer_reset_holds_and_requires_a_new_sync_cycle():
    session = _session()
    session.connect()
    try:
        _read_until_available(session)
        session.set_mode(LeaderSyncMode.SYNC_REQUEST)
        session.set_mode(LeaderSyncMode.TRACKING)
        assert session.write_arm(
            ("joint_1",),
            (0.1,),
            now_ns=time.monotonic_ns(),
        )

        session.reset_peer()

        assert session.mode == LeaderSyncMode.IDLE
        assert session.status == FollowerSyncStatus.IDLE
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
        assert not session.write_arm(
            ("joint_1",),
            (0.2,),
            now_ns=time.monotonic_ns(),
        )
    finally:
        session.close()


def test_follower_session_uses_the_same_timeout_as_runtime_safety():
    session = _session(heartbeat_timeout_ns=1_000_000_000)

    assert session.runtime.command_timeout_ns == 1_000_000_000


def test_follower_session_rejects_a_timeout_that_differs_from_runtime():
    session = _session()

    with pytest.raises(ValueError, match="must match"):
        FollowerTeleopSession(
            session.runtime,
            heartbeat_timeout_ns=1_000_000_000,
        )


def test_follower_session_reports_latched_runtime_fault():
    session = _session()
    session.connect()
    try:
        _read_until_available(session)
        session.runtime.emergency_stop()

        assert session.status == FollowerSyncStatus.FAULT
    finally:
        session.close()


def test_slow_end_effector_deadline_covers_its_bus_cycle_and_io_budget():
    arm_bus = BusSpec(
        "arm",
        BusType.FEETECH_PACKET,
        "mock://arm",
        1_000_000,
        100.0,
        physical_layer="ttl",
        family="hls",
    )
    hand_bus = BusSpec(
        "hand",
        BusType.LINKER_HAND_RS485,
        "mock://hand",
        1_000_000,
        10.0,
    )
    arm = ArmSpec(
        "single",
        "arm",
        (JointSpec("joint_1", 0, "HL3960", 1, 0.0),),
        end_effector=DexterousHandSpec("hand", "linker", "O6", "right", 1),
    )
    runtime = RobotRuntime(
        RobotSpec(
            "ace_follower",
            (arm_bus, hand_bus),
            (arm,),
            backend=Backend.MOCK,
            urdf_path=str(urdf_path),
        )
    )
    session = FollowerTeleopSession(runtime)
    group_name = "single.end_effector"
    names = session.end_effector_names[group_name]
    now_ns = 1_000_000_000

    command = session._commands_for_groups(  # noqa: SLF001
        {group_name: names},
        names,
        (0.0,) * len(names),
        now_ns=now_ns,
    )[group_name]

    assert command.deadline_ns - now_ns == runtime.command_lifetime_ns(group_name)
    assert command.deadline_ns - now_ns > 100_000_000
    assert runtime.command_lifetime_ns("single") == 50_000_000


def test_cartesian_follower_accepts_pose_only_after_tracking_and_resets_on_loss():
    session = _cartesian_session()
    session.connect()
    try:
        _read_until_available(session)
        kinematics = ArmKinematics(
            urdf_path,
            session.arm_names,
            "link_5",
        )
        pose = kinematics.forward([0.0, 0.0, 0.0, 0.0], timestamp_ns=1)
        now_ns = time.monotonic_ns()

        assert not session.write_arm_pose(pose, now_ns=now_ns)
        session.set_mode(LeaderSyncMode.SYNC_REQUEST)
        session.set_mode(LeaderSyncMode.TRACKING)
        assert not session.write_arm(session.arm_names, [0.0] * 4, now_ns=now_ns)
        assert session.write_arm_pose(pose, now_ns=now_ns)
        assert session.status == FollowerSyncStatus.TRACKING
        assert session.cartesian_diagnostics() is not None

        session.update(now_ns=now_ns + 100_000_001)
        assert session.status == FollowerSyncStatus.LOST
        assert session.cartesian_diagnostics() is None
    finally:
        session.close()
