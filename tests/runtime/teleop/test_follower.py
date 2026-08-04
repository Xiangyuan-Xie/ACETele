from __future__ import annotations

import time
from pathlib import Path

import pytest

from acetele.core import JointState, RobotState
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
    ParallelGripperSpec,
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
    motion_timeout_ns: int = 100_000_000,
    session_timeout_ns: int = 500_000_000,
    backend: Backend = Backend.MOCK,
    motion_cycle_hz: float | None = None,
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
    runtime = RobotRuntime(spec, command_timeout_ns=motion_timeout_ns)
    return FollowerTeleopSession(
        runtime,
        session_timeout_ns=session_timeout_ns,
        motion_cycle_hz=motion_cycle_hz,
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
        motion_cycle_hz=None,
    )


def _gripper_session() -> FollowerTeleopSession:
    bus = BusSpec(
        "arm",
        BusType.FEETECH_PACKET,
        "mock://gripper-arm",
        1_000_000,
        100.0,
        physical_layer="ttl",
        family="hls",
    )
    arm = ArmSpec(
        "single",
        "arm",
        (JointSpec("joint_1", 0, "HL3915", 1, 0.0),),
        end_effector=ParallelGripperSpec(
            "arm",
            JointSpec("joint_5", 4, "HL3915", 1, 0.0),
            0.75,
        ),
    )
    runtime = RobotRuntime(
        RobotSpec(
            "ace_follower",
            (bus,),
            (arm,),
            backend=Backend.MOCK,
            urdf_path=str(urdf_path),
        )
    )
    return FollowerTeleopSession(runtime, motion_cycle_hz=None)


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


def test_leader_idle_mode_keeps_the_follower_powered_in_hold():
    session = _session()
    session.connect()
    try:
        session.hold_position()
        session.set_mode(LeaderSyncMode.SYNC_REQUEST)
        session.set_mode(LeaderSyncMode.IDLE)

        assert session.status == FollowerSyncStatus.IDLE
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
        protocol = session.runtime._actors["arm"]._protocol  # noqa: SLF001
        assert protocol._enabled_ids == {0}  # noqa: SLF001
    finally:
        session.close()


def test_remote_idle_cannot_energize_a_safe_disabled_follower():
    session = _session()
    session.connect()
    try:
        session.set_mode(LeaderSyncMode.IDLE)

        assert session.status == FollowerSyncStatus.IDLE
        assert (
            session.runtime.diagnostics().safety.state
            == RuntimeSafetyState.SAFE_DISABLED
        )
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
        target = session._motion_targets["single"]  # noqa: SLF001
        assert target.velocity_limits.tolist() == [4.0]
        assert target.acceleration_limits.tolist() == [12.0]
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


def test_follower_session_loss_timeout_holds_runtime():
    session = _session()
    session.connect()
    try:
        _read_until_available(session)
        session.set_mode(LeaderSyncMode.SYNC_REQUEST)
        session.set_mode(LeaderSyncMode.TRACKING)
        now_ns = time.monotonic_ns()
        session.write_arm(("joint_1",), (0.1,), now_ns=now_ns)

        accepted_ns = session._sync.last_command_ns  # noqa: SLF001
        assert accepted_ns is not None
        timeout_ns = accepted_ns + 500_000_001
        assert session.update(now_ns=timeout_ns) == FollowerSyncStatus.LOST
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.READY

        assert session.step_motion(now_ns=timeout_ns)
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.ACTIVE
        assert session.step_motion(now_ns=timeout_ns + 100_000_001)
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
    finally:
        session.close()


def test_follower_session_loss_generates_a_measured_braking_target():
    session = _session()
    session.connect()
    try:
        _read_until_available(session)
        session.set_mode(LeaderSyncMode.SYNC_REQUEST)
        session.set_mode(LeaderSyncMode.TRACKING)
        assert session.write_arm(
            ("joint_1",),
            (1.0,),
            now_ns=time.monotonic_ns(),
        )
        session._latest_state = RobotState(  # noqa: SLF001
            {
                "single": JointState(
                    ("joint_1",),
                    (0.2,),
                    (0.6,),
                    (0.0,),
                    1,
                    1,
                )
            },
            {},
        )
        accepted_ns = session._sync.last_command_ns  # noqa: SLF001
        assert accepted_ns is not None

        session.update(now_ns=accepted_ns + 500_000_001)

        braking_target = session._motion_targets["single"]  # noqa: SLF001
        expected_distance = 0.6**2 / (2.0 * 12.0)
        assert braking_target.positions[0] == pytest.approx(0.2 + expected_distance)
        assert session.runtime.diagnostics().safety.state != RuntimeSafetyState.HOLD
    finally:
        session.close()


def test_follower_local_cycle_bridges_a_brief_network_gap_without_hardware_hold():
    session = _session(motion_timeout_ns=30_000_000)
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

        actor = session.runtime._actors["arm"]  # noqa: SLF001
        # No additional network command arrives for more than the 30 ms actuator
        # watchdog, but the follower-local 100 Hz loop keeps converging to the latest
        # target and therefore remains continuously ACTIVE.
        deadline = time.monotonic() + 0.12
        while time.monotonic() < deadline:
            session.read(now_ns=time.monotonic_ns())
            time.sleep(0.005)

        assert not actor.motion_watchdog_tripped
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.ACTIVE
        assert session.status == FollowerSyncStatus.TRACKING
        assert session.write_arm(
            ("joint_1",),
            (0.2,),
            now_ns=time.monotonic_ns(),
        )
        session.read(now_ns=time.monotonic_ns())
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.ACTIVE
        assert not actor.motion_watchdog_tripped
        assert session.status == FollowerSyncStatus.TRACKING
    finally:
        session.close()


def test_late_network_frame_cannot_recover_a_stalled_local_motion_cycle():
    session = _session(motion_timeout_ns=30_000_000)
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
        session.read(now_ns=time.monotonic_ns())

        actor = session.runtime._actors["arm"]  # noqa: SLF001
        deadline = time.monotonic() + 1.0
        while not actor.motion_watchdog_tripped:
            if time.monotonic() >= deadline:
                pytest.fail("actor motion watchdog did not detect the stalled local loop")
            time.sleep(0.001)

        # A late packet may replace the in-memory target, but the next local step sees
        # the actor generation fence and closes the synchronization cycle instead of
        # performing the old automatic HOLD-to-READY transition.
        assert session.write_arm(
            ("joint_1",),
            (0.2,),
            now_ns=time.monotonic_ns(),
        )
        with pytest.raises(RuntimeError, match="cannot move while hold"):
            session.read(now_ns=time.monotonic_ns())

        assert session.status == FollowerSyncStatus.HOLD
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
        assert actor.motion_watchdog_tripped
    finally:
        session.close()


def test_independent_motion_loop_survives_state_publisher_stall():
    session = _session(
        motion_timeout_ns=30_000_000,
        motion_cycle_hz=100.0,
    )
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
        deadline = time.monotonic() + 1.0
        while session.runtime.diagnostics().safety.state != RuntimeSafetyState.ACTIVE:
            if time.monotonic() >= deadline:
                pytest.fail("local motion loop did not submit the accepted target")
            time.sleep(0.001)

        # No transport-facing read or publish occurs for four watchdog periods.
        time.sleep(0.12)

        actor = session.runtime._actors["arm"]  # noqa: SLF001
        assert not actor.motion_watchdog_tripped
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.ACTIVE
    finally:
        session.close()


def test_local_motion_submission_failure_enters_hold_immediately(monkeypatch):
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

        def reject_motion(_command):
            raise ValueError("test local command failure")

        monkeypatch.setattr(session.runtime, "write", reject_motion)
        with pytest.raises(ValueError, match="test local command failure"):
            session.step_motion(now_ns=time.monotonic_ns())

        assert session.status == FollowerSyncStatus.HOLD
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
    finally:
        session.close()


def test_invalid_arm_frame_cannot_reenable_a_brief_motion_hold():
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
        session.runtime.hold()

        with pytest.raises(ValueError, match="joint order"):
            session.write_arm(
                ("wrong",),
                (0.2,),
                now_ns=time.monotonic_ns(),
            )

        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
    finally:
        session.close()


def test_follower_hold_mode_preserves_torque_and_rejects_motion():
    session = _session()
    session.connect()
    try:
        _read_until_available(session)
        session.set_mode(LeaderSyncMode.SYNC_REQUEST)
        session.set_mode(LeaderSyncMode.HOLD)

        assert session.status == FollowerSyncStatus.HOLD
        assert session.runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
        assert not session.write_arm(
            ("joint_1",),
            (0.1,),
            now_ns=time.monotonic_ns(),
        )
    finally:
        session.close()


def test_end_effector_commands_cannot_extend_the_arm_heartbeat():
    session = _gripper_session()
    session.connect()
    try:
        _read_until_available(session)
        session.set_mode(LeaderSyncMode.SYNC_REQUEST)
        session.set_mode(LeaderSyncMode.TRACKING)
        started_ns = time.monotonic_ns()
        assert session.write_arm(("joint_1",), (0.0,), now_ns=started_ns)
        heartbeat_ns = session._sync.last_command_ns  # noqa: SLF001
        group_name = "single.end_effector"

        assert session.write_end_effector(
            group_name,
            ("joint_5",),
            (0.5,),
            now_ns=started_ns + 50_000_000,
        )
        assert session._sync.last_command_ns == heartbeat_ns  # noqa: SLF001
        accepted_ns = session._sync.last_command_ns  # noqa: SLF001
        assert accepted_ns is not None
        assert not session.write_end_effector(
            group_name,
            ("joint_5",),
            (0.6,),
            now_ns=accepted_ns + 500_000_001,
        )
        assert session.status == FollowerSyncStatus.LOST
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


def test_follower_session_uses_actor_motion_timeout_and_separate_session_timeout():
    session = _session(
        motion_timeout_ns=100_000_000,
        session_timeout_ns=500_000_000,
    )

    assert session.runtime.command_timeout_ns == 100_000_000


def test_physical_follower_arm_targets_carry_bounded_streaming_profiles():
    session = _session(backend=Backend.PHYSICAL)

    commands = session._commands_for_groups(  # noqa: SLF001
        session._arm_names,  # noqa: SLF001
        session.arm_names,
        (0.0,),
        now_ns=1,
    )

    assert commands["single"].velocity_limits.tolist() == [4.0]
    assert commands["single"].acceleration_limits.tolist() == [12.0]


def test_follower_session_reports_latched_runtime_fault():
    session = _session()
    session.connect()
    try:
        _read_until_available(session)
        session.runtime.emergency_stop()

        assert session.status == FollowerSyncStatus.FAULT
    finally:
        session.close()


def test_follower_stop_retries_even_after_a_fault_or_previous_stop(monkeypatch):
    session = _session()
    attempts = []

    def stop():
        attempts.append("stop")

    monkeypatch.setattr(session.runtime, "emergency_stop", stop)
    session.runtime._safety.fault("test fault")  # noqa: SLF001

    session.set_mode(LeaderSyncMode.STOP)
    session.set_mode(LeaderSyncMode.STOP)

    assert attempts == ["stop", "stop"]


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

        accepted_ns = session._sync.last_command_ns  # noqa: SLF001
        assert accepted_ns is not None
        session.update(now_ns=accepted_ns + 500_000_001)
        assert session.status == FollowerSyncStatus.LOST
        assert session.cartesian_diagnostics() is None
    finally:
        session.close()
