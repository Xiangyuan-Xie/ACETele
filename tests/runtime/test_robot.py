from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from acetele.core import JointCommand, JointUnit, RobotCommand
from acetele.hardware.devices.servos.fashionstar import FashionStarMonitorState
from acetele.hardware.devices.servos.feetech import FeetechPacketFastState
from acetele.runtime import RobotRuntime, RuntimeSafetyState
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
    Path(__file__).resolve().parents[2]
    / "acetele"
    / "model"
    / "robots"
    / "ace_follower"
    / "description"
    / "ace_follower.urdf"
)


def _spec(
    *,
    model: str = "HL3960",
    backend: Backend = Backend.MOCK,
    external_estop: bool | None = None,
    cycle_hz: float = 100.0,
    expected_model_number: int | None = 100,
    allow_unverified_identity: bool = False,
) -> RobotSpec:
    if external_estop is None:
        external_estop = backend == Backend.PHYSICAL
    bus = BusSpec(
        "arm",
        BusType.FEETECH_PACKET,
        "mock://arm",
        1_000_000,
        cycle_hz,
        physical_layer="ttl",
        family="hls",
        external_estop=external_estop,
        allow_unverified_identity=allow_unverified_identity,
    )
    joint = JointSpec(
        "joint_1",
        0,
        model,
        -1,
        0.0,
        expected_model_number=expected_model_number,
    )
    return RobotSpec(
        "ace_follower",
        (bus,),
        (ArmSpec("single", "arm", (joint,)),),
        backend=backend,
        urdf_path=str(urdf_path),
    )


def _read_until_available(runtime: RobotRuntime):
    deadline = time.monotonic() + 1.0
    while True:
        try:
            return runtime.read()
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.001)


def test_runtime_constructor_preflights_without_creating_transport():
    calls = []

    def transport_factory(port, baudrate, direction):
        calls.append((port, baudrate, direction))
        raise AssertionError("transport should not be created during preflight")

    runtime = RobotRuntime(
        _spec(backend=Backend.PHYSICAL),
        transport_factory=transport_factory,
    )

    assert not runtime.connected
    assert not calls
    assert runtime.preflight.buses["arm"].budget.feasible
    assert not runtime.preflight.buses["arm"].supports_verified_disable


def test_physical_bus_without_verified_disable_requires_external_estop():
    calls = []

    with pytest.raises(ValueError, match="independent hardware emergency stop"):
        RobotRuntime(
            _spec(backend=Backend.PHYSICAL, external_estop=False),
            transport_factory=lambda *args: calls.append(args),
        )

    assert not calls


def test_physical_bus_requires_explicit_unverified_identity_acknowledgement():
    with pytest.raises(ValueError, match="allow_unverified_identity=true"):
        RobotRuntime(
            _spec(
                backend=Backend.PHYSICAL,
                expected_model_number=None,
            )
        )

    runtime = RobotRuntime(
        _spec(
            backend=Backend.PHYSICAL,
            expected_model_number=None,
            allow_unverified_identity=True,
        )
    )
    assert not runtime.preflight.buses["arm"].supports_verified_identity


def test_runtime_precomputes_home_calibration_without_hardware_io():
    calls = []
    runtime = RobotRuntime(
        _spec(backend=Backend.PHYSICAL),
        transport_factory=lambda *args: calls.append(args),
    )

    targets = runtime.home_calibration_targets()

    assert targets["arm"] == {0: 0}
    assert not calls


def test_unknown_profile_fails_before_transport_creation():
    calls = []

    with pytest.raises(ValueError, match="unsupported model"):
        RobotRuntime(
            _spec(model="UNKNOWN", backend=Backend.PHYSICAL),
            transport_factory=lambda *args: calls.append(args),
        )

    assert not calls


def test_mock_runtime_maps_direction_and_enforces_safety_state():
    runtime = RobotRuntime(_spec())
    runtime.connect()
    try:
        initial = _read_until_available(runtime)
        assert initial.joints["single"].positions.tolist() == [0.0]
        runtime.set_enabled(True)
        now = time.monotonic_ns()
        runtime.write(
            RobotCommand(
                {
                    "single": JointCommand(
                        ("joint_1",),
                        [0.5],
                        now,
                        now + 50_000_000,
                        runtime.diagnostics().safety.generation,
                        JointUnit.RADIAN,
                    )
                }
            )
        )

        deadline = time.monotonic() + 2.0
        while True:
            state = runtime.read().joints["single"]
            if state.positions[0] == pytest.approx(0.5):
                break
            if time.monotonic() >= deadline:
                pytest.fail("mock actor did not apply the latest command")
            now = time.monotonic_ns()
            runtime.write(
                RobotCommand(
                    {
                        "single": JointCommand(
                            ("joint_1",),
                            [0.5],
                            now,
                            now + 50_000_000,
                            runtime.diagnostics().safety.generation,
                            JointUnit.RADIAN,
                        )
                    }
                )
            )
            time.sleep(0.01)
        assert runtime.diagnostics().safety.state == RuntimeSafetyState.ACTIVE

        runtime.hold()
        assert runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
    finally:
        runtime.disconnect()


def test_runtime_mirrors_actor_watchdog_hold_and_requires_reenable():
    runtime = RobotRuntime(_spec(), command_timeout_ns=30_000_000)
    runtime.connect()
    try:
        _read_until_available(runtime)
        runtime.set_enabled(True)
        now_ns = time.monotonic_ns()
        runtime.write(
            RobotCommand(
                {
                    "single": JointCommand(
                        ("joint_1",),
                        (0.25,),
                        now_ns,
                        now_ns + 100_000_000,
                        runtime.generation,
                    )
                }
            )
        )
        deadline = time.monotonic() + 0.5
        while not runtime.diagnostics().buses["arm"].motion_watchdog_tripped:
            if time.monotonic() >= deadline:
                pytest.fail("actor watchdog did not autonomously hold motion")
            time.sleep(0.001)

        runtime.read()
        assert runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD
        with pytest.raises(RuntimeError, match="cannot move while hold"):
            runtime.write(
                RobotCommand(
                    {
                        "single": JointCommand(
                            ("joint_1",),
                            (0.3,),
                            time.monotonic_ns(),
                            time.monotonic_ns() + 100_000_000,
                            runtime.generation,
                        )
                    }
                )
            )

        runtime.set_enabled(True)
        assert runtime.diagnostics().safety.state == RuntimeSafetyState.READY
        assert not runtime.diagnostics().buses["arm"].motion_watchdog_tripped
    finally:
        runtime.disconnect()


def test_runtime_connect_waits_for_initial_state_snapshot():
    runtime = RobotRuntime(_spec())
    runtime.connect()
    try:
        assert runtime.read().joints["single"].positions.tolist() == [0.0]
    finally:
        runtime.disconnect()


def test_runtime_uses_each_bus_cycle_rate_for_state_freshness():
    runtime = RobotRuntime(_spec(cycle_hz=10.0))
    runtime.connect()
    try:
        runtime.read()
        time.sleep(0.06)
        runtime.read()
        assert runtime.diagnostics().safety.state == RuntimeSafetyState.SAFE_DISABLED
    finally:
        runtime.disconnect()


def test_hold_rejects_invalid_state_before_touching_hardware():
    class RecordingActor:
        connected = True

        def __init__(self):
            self.calls = []

        def submit_safety(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    runtime = RobotRuntime(_spec())
    actor = RecordingActor()
    runtime._actors = {"arm": actor}  # noqa: SLF001
    runtime._safety.connected()  # noqa: SLF001

    with pytest.raises(RuntimeError, match="SAFE_DISABLED|safe_disabled"):
        runtime.hold()

    assert actor.calls == []


def test_hold_closes_the_software_command_gate_before_touching_hardware():
    runtime = RobotRuntime(_spec())

    class StateObservingActor:
        connected = True

        def __init__(self):
            self.observed_states = []

        def submit_safety(self, *args, **kwargs):
            self.observed_states.append(runtime._safety.snapshot().state)  # noqa: SLF001

    actor = StateObservingActor()
    runtime._actors = {"arm": actor}  # noqa: SLF001
    runtime._safety.connected()  # noqa: SLF001
    runtime._safety.ready()  # noqa: SLF001

    runtime.hold()

    assert actor.observed_states == [RuntimeSafetyState.HOLD]
    assert runtime.generation == 1


def test_hold_cannot_overtake_an_accepted_motion_submission():
    release_motion = Event()
    motion_reached_actor = Event()

    class BlockingActor:
        connected = True

        def __init__(self):
            self.events = []

        @property
        def generation(self):
            motion_reached_actor.set()
            assert release_motion.wait(timeout=1.0)
            return 0

        def diagnostics(self):
            return SimpleNamespace(last_state_ns=time.monotonic_ns())

        def submit_motion(self, _targets):
            self.events.append("motion")

        def submit_safety(self, *args, **kwargs):
            self.events.append("hold")

    runtime = RobotRuntime(_spec())
    actor = BlockingActor()
    runtime._actors = {"arm": actor}  # noqa: SLF001
    runtime._safety.connected()  # noqa: SLF001
    runtime._safety.ready()  # noqa: SLF001
    now_ns = time.monotonic_ns()
    command = RobotCommand(
        {
            "single": JointCommand(
                ("joint_1",),
                [0.25],
                now_ns,
                now_ns + 1_000_000_000,
                runtime.generation,
            )
        }
    )
    errors = []

    def run(callable_):
        try:
            callable_()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    writer = Thread(target=run, args=(lambda: runtime.write(command),))
    holder = Thread(target=run, args=(runtime.hold,))
    writer.start()
    assert motion_reached_actor.wait(timeout=1.0)
    holder.start()
    assert actor.events == []

    release_motion.set()
    writer.join(timeout=1.0)
    holder.join(timeout=1.0)

    assert not writer.is_alive()
    assert not holder.is_alive()
    assert errors == []
    assert actor.events == ["motion", "hold"]
    assert runtime.diagnostics().safety.state == RuntimeSafetyState.HOLD


def test_disabling_a_linker_hand_uses_hold_instead_of_unsupported_disable():
    arm_bus = BusSpec(
        "arm",
        BusType.FEETECH_PACKET,
        "/dev/arm",
        1_000_000,
        100.0,
        physical_layer="ttl",
        family="hls",
        external_estop=True,
        allow_unverified_identity=True,
    )
    hand_bus = BusSpec(
        "hand",
        BusType.LINKER_HAND_RS485,
        "/dev/hand",
        1_000_000,
        10.0,
        external_estop=True,
        allow_unverified_identity=True,
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
            backend=Backend.PHYSICAL,
            urdf_path=str(urdf_path),
        )
    )

    class RecordingActor:
        connected = True

        def __init__(self):
            self.calls = []

        def submit_safety(self, label, payload, *, wait, clear_motion=True):
            self.calls.append((label, payload, wait, clear_motion))

    arm_actor = RecordingActor()
    hand_actor = RecordingActor()
    runtime._actors = {"arm": arm_actor, "hand": hand_actor}  # noqa: SLF001
    runtime._safety.connected()  # noqa: SLF001

    runtime.set_enabled(False)

    assert arm_actor.calls == [("set_enabled", False, True, True)]
    assert hand_actor.calls == [("hold", None, True, True)]


def test_hardware_status_latches_fault_before_state_is_published():
    class FaultActor:
        connected = True

        def __init__(self):
            self.calls = []

        def get_snapshot(self):
            return {0: SimpleNamespace(status=2)}

        def get_slow_snapshot(self):
            return {}

        def submit_safety(self, label, payload, *, wait, clear_motion=True):
            self.calls.append((label, payload, wait, clear_motion))

        def discard_motion(self):
            pass

    runtime = RobotRuntime(_spec())
    actor = FaultActor()
    runtime._actors = {"arm": actor}  # noqa: SLF001
    runtime._safety.connected()  # noqa: SLF001

    with pytest.raises(RuntimeError, match="hardware status 0x02"):
        runtime.read()

    assert runtime._safety.snapshot().state == RuntimeSafetyState.FAULT  # noqa: SLF001
    assert actor.calls == [("emergency_stop", None, False, True)]


def test_stale_hardware_state_faults_without_returning_the_old_snapshot():
    class StaleActor:
        connected = True

        def __init__(self):
            self.calls = []

        def get_snapshot(self):
            return {
                0: FeetechPacketFastState(
                    position_rad=0.0,
                    velocity_rad_s=0.0,
                    current_a=0.0,
                    load_ratio=0.0,
                    voltage_v=12.0,
                    temperature_c=30,
                    status=0,
                    timestamp_ns=1,
                )
            }

        def get_slow_snapshot(self):
            return {}

        def diagnostics(self):
            return SimpleNamespace(last_state_ns=1)

        def submit_safety(self, label, payload, *, wait, clear_motion=True):
            self.calls.append((label, payload, wait, clear_motion))

    runtime = RobotRuntime(_spec(), clock_ns=lambda: 1_000_000_000)
    actor = StaleActor()
    runtime._actors = {"arm": actor}  # noqa: SLF001
    runtime._safety.connected()  # noqa: SLF001

    with pytest.raises(RuntimeError, match="state is stale"):
        runtime.read()

    assert runtime._safety.snapshot().state == RuntimeSafetyState.FAULT  # noqa: SLF001
    assert actor.calls == [("emergency_stop", None, False, True)]


def test_gripper_normalized_motion_limits_are_converted_to_radians():
    bus = BusSpec(
        "arm",
        BusType.FEETECH_PACKET,
        "/dev/arm",
        1_000_000,
        100.0,
        physical_layer="ttl",
        family="hls",
        external_estop=True,
        allow_unverified_identity=True,
    )
    arm = ArmSpec(
        "single",
        "arm",
        (JointSpec("joint_1", 0, "HL3960", 1, 0.0),),
        end_effector=ParallelGripperSpec(
            "arm",
            JointSpec("joint_5", 4, "HL3915", 1, 0.0),
            0.5,
        ),
    )
    runtime = RobotRuntime(
        RobotSpec(
            "ace_follower",
            (bus,),
            (arm,),
            backend=Backend.PHYSICAL,
            urdf_path=str(urdf_path),
        )
    )
    runtime._position_references["arm"][4] = 0.25  # noqa: SLF001
    command = JointCommand(
        ("joint_5",),
        [0.5],
        0,
        1,
        0,
        JointUnit.NORMALIZED,
        velocity_limits=[2.0],
        acceleration_limits=[3.0],
    )

    envelope = runtime._encode_group(  # noqa: SLF001
        runtime._groups["single.end_effector"],  # noqa: SLF001
        command,
    )[0]

    assert envelope.payload.position_rad == pytest.approx(0.25)
    assert envelope.payload.velocity_rad_s == pytest.approx(1.0)
    assert envelope.payload.acceleration_rad_s2 == pytest.approx(1.5)


def test_packet_multiturn_overflow_is_rejected_before_actor_submission():
    class RecordingActor:
        connected = True
        generation = 0

        def __init__(self):
            self.submissions = []

        def diagnostics(self):
            return SimpleNamespace(last_state_ns=time.monotonic_ns())

        def submit_motion(self, targets):
            self.submissions.append(targets)

    runtime = RobotRuntime(_spec(backend=Backend.PHYSICAL))
    actor = RecordingActor()
    runtime._actors = {"arm": actor}  # noqa: SLF001
    runtime._safety.connected()  # noqa: SLF001
    runtime._safety.ready()  # noqa: SLF001
    runtime._position_references["arm"][0] = (  # noqa: SLF001
        -32680 * 2.0 * 3.141592653589793 / 4096
    )
    now_ns = time.monotonic_ns()

    with pytest.raises(ValueError, match="multi-turn position range is exhausted"):
        runtime.write(
            RobotCommand(
                {
                    "single": JointCommand(
                        ("joint_1",),
                        [0.5],
                        now_ns,
                        now_ns + 50_000_000,
                        runtime.generation,
                    )
                }
            )
        )

    assert actor.submissions == []
    assert runtime._safety.snapshot().state == RuntimeSafetyState.READY  # noqa: SLF001
    assert runtime._pipelines["single"]._last_output is None  # noqa: SLF001


def test_motion_submission_failure_faults_all_buses_without_committing_control_history():
    first_bus = BusSpec(
        "first",
        BusType.FEETECH_PACKET,
        "mock://first",
        1_000_000,
        100.0,
        physical_layer="ttl",
        family="hls",
    )
    second_bus = BusSpec(
        "second",
        BusType.FEETECH_PACKET,
        "mock://second",
        1_000_000,
        100.0,
        physical_layer="ttl",
        family="hls",
    )
    runtime = RobotRuntime(
        RobotSpec(
            "ace_follower",
            (first_bus, second_bus),
            (
                ArmSpec(
                    "first_arm",
                    "first",
                    (JointSpec("joint_1", 0, "HL3960", 1, 0.0),),
                ),
                ArmSpec(
                    "second_arm",
                    "second",
                    (JointSpec("joint_2", 1, "HL3950", 1, 0.0),),
                ),
            ),
            backend=Backend.MOCK,
            urdf_path=str(urdf_path),
        )
    )

    class RecordingActor:
        connected = True
        generation = 0

        def __init__(self, *, fail_motion: bool = False) -> None:
            self.fail_motion = fail_motion
            self.motion: list[object] = []
            self.safety: list[tuple[object, object, object, object]] = []

        def diagnostics(self):
            return SimpleNamespace(last_state_ns=time.monotonic_ns())

        def submit_motion(self, targets):
            self.motion.append(targets)
            if self.fail_motion:
                raise RuntimeError("motion mailbox failed")

        def submit_safety(self, label, payload, *, wait, clear_motion=True):
            self.safety.append((label, payload, wait, clear_motion))

        def discard_motion(self):
            self.motion.clear()

    first = RecordingActor()
    second = RecordingActor(fail_motion=True)
    runtime._actors = {"first": first, "second": second}  # noqa: SLF001
    runtime._safety.connected()  # noqa: SLF001
    runtime._safety.ready()  # noqa: SLF001
    now_ns = time.monotonic_ns()

    with pytest.raises(RuntimeError, match="motion mailbox failed"):
        runtime.write(
            RobotCommand(
                {
                    "first_arm": JointCommand(
                        ("joint_1",),
                        (0.1,),
                        now_ns,
                        now_ns + 50_000_000,
                        runtime.generation,
                    ),
                    "second_arm": JointCommand(
                        ("joint_2",),
                        (0.2,),
                        now_ns,
                        now_ns + 50_000_000,
                        runtime.generation,
                    ),
                }
            )
        )

    assert runtime._safety.snapshot().state == RuntimeSafetyState.FAULT  # noqa: SLF001
    assert first.motion[0][0].commit_gate.state is False
    assert first.safety == [("emergency_stop", None, False, True)]
    assert second.safety == [("emergency_stop", None, False, True)]
    assert runtime._pipelines["first_arm"]._last_output is None  # noqa: SLF001
    assert runtime._pipelines["second_arm"]._last_output is None  # noqa: SLF001


def test_runtime_rejects_any_group_before_submitting_motion():
    runtime = RobotRuntime(_spec())
    runtime.connect()
    try:
        _read_until_available(runtime)
        runtime.set_enabled(True)
        now = time.monotonic_ns()
        with pytest.raises(ValueError, match="unknown joint group"):
            runtime.write(
                RobotCommand(
                    {
                        "missing": JointCommand(
                            ("joint_1",),
                            [0.0],
                            now,
                            now + 50_000_000,
                            runtime.diagnostics().safety.generation,
                        )
                    }
                )
            )
        assert runtime.diagnostics().buses["arm"].pending_motion_count == 0
        assert runtime.diagnostics().safety.state == RuntimeSafetyState.READY
    finally:
        runtime.disconnect()


def test_runtime_rolls_back_all_buses_when_a_safety_transaction_fails():
    class RecordingActor:
        connected = True

        def __init__(self, *, fail_enable=False):
            self.fail_enable = fail_enable
            self.calls = []

        def submit_safety(
            self,
            label,
            payload,
            *,
            wait,
            clear_motion=True,
        ):
            self.calls.append((label, payload, wait, clear_motion))
            if self.fail_enable and label == "set_enabled" and payload is True:
                raise RuntimeError("enable failed")

    runtime = RobotRuntime(_spec())
    first = RecordingActor()
    second = RecordingActor(fail_enable=True)
    runtime._actors = {"first": first, "second": second}  # noqa: SLF001
    runtime._safety.connected()  # noqa: SLF001

    with pytest.raises(RuntimeError, match="enable failed"):
        runtime.set_enabled(True)

    assert runtime._safety.snapshot().state == RuntimeSafetyState.FAULT  # noqa: SLF001
    assert first.calls == [
        ("set_enabled", True, True, False),
        ("emergency_stop", None, False, True),
    ]
    assert second.calls == [
        ("set_enabled", True, True, False),
        ("emergency_stop", None, False, True),
    ]


def test_runtime_exposes_slow_bus_telemetry_as_sensor_state():
    runtime = RobotRuntime(_spec())

    sensors = runtime._sensor_states(  # noqa: SLF001 - verifies the snapshot adapter
        {
            "arm": {
                1: SimpleNamespace(timestamp_ns=123, temperature_c=31),
            }
        }
    )

    assert sensors["arm"].timestamp_ns == 123
    assert sensors["arm"].values["1"].temperature_c == 31
    with pytest.raises(TypeError):
        sensors["arm"].values["2"] = object()


def test_runtime_converts_timestamped_fashionstar_state():
    bus = BusSpec(
        "arm",
        BusType.FASHIONSTAR_RS485,
        "/dev/null",
        115_200,
        25.0,
        external_estop=True,
        allow_unverified_identity=True,
    )
    joint = JointSpec("joint_1", 1, "HX8-R50W-M", 1, 0.0, firmware_version=316)
    runtime = RobotRuntime(
        RobotSpec(
            "ace_follower",
            (bus,),
            (ArmSpec("single", "arm", (joint,)),),
            backend=Backend.PHYSICAL,
            urdf_path=str(urdf_path),
        )
    )

    state = runtime._read_group(  # noqa: SLF001 - verifies protocol/runtime contract
        runtime._groups["single"],  # noqa: SLF001
        {
            1: FashionStarMonitorState(
                servo_id=1,
                voltage_v=12.0,
                current_a=0.1,
                power_w=1.2,
                temperature_raw=30,
                status=0,
                position_rad=0.25,
                turns=0,
                timestamp_ns=123,
            )
        },
    )

    assert state.timestamp_ns == 123
    assert state.positions.tolist() == pytest.approx([0.25])


def test_actor_fault_latches_runtime_fault_and_stops_healthy_buses():
    class RecordingActor:
        def __init__(self, *, connected: bool) -> None:
            self.connected = connected
            self.calls: list[tuple[object, object, object, object]] = []
            self.discard_count = 0

        def submit_safety(self, label, payload, *, wait, clear_motion=True):
            self.calls.append((label, payload, wait, clear_motion))

        def discard_motion(self):
            self.discard_count += 1

    runtime = RobotRuntime(_spec())
    failed = RecordingActor(connected=False)
    healthy = RecordingActor(connected=True)
    runtime._actors = {"failed": failed, "healthy": healthy}  # noqa: SLF001
    runtime._safety.connected()  # noqa: SLF001

    with pytest.raises(RuntimeError, match="failed"):
        runtime._require_connected()  # noqa: SLF001

    safety = runtime._safety.snapshot()  # noqa: SLF001
    assert safety.state == RuntimeSafetyState.FAULT
    assert safety.fault_reason is not None and "failed" in safety.fault_reason
    assert failed.discard_count == 1
    assert healthy.calls == [("emergency_stop", None, False, True)]


def test_runtime_preserves_hls_current_based_torque_estimation():
    runtime = RobotRuntime(_spec(backend=Backend.PHYSICAL))
    group = runtime._groups["single"]  # noqa: SLF001 - verifies profile conversion
    state = runtime._read_group(  # noqa: SLF001 - no hardware is opened
        group,
        {
            0: FeetechPacketFastState(
                position_rad=0.0,
                velocity_rad_s=0.0,
                current_a=1.3,
                load_ratio=0.0,
                voltage_v=12.0,
                temperature_c=30,
                status=0,
                timestamp_ns=1,
            )
        },
    )

    assert state.efforts[0] == pytest.approx(14.84 * 0.0980665)


def test_runtime_maps_multiturn_feedback_to_the_urdf_joint_interval():
    runtime = RobotRuntime(_spec(backend=Backend.PHYSICAL))
    group = runtime._groups["single"]  # noqa: SLF001 - verifies public conversion

    state = runtime._read_group(  # noqa: SLF001 - no hardware is opened
        group,
        {
            0: FeetechPacketFastState(
                position_rad=-32623 * 2.0 * 3.141592653589793 / 4096,
                velocity_rad_s=0.0,
                current_a=0.0,
                load_ratio=0.0,
                voltage_v=12.0,
                temperature_c=30,
                status=0,
                timestamp_ns=1,
            )
        },
    )

    assert state.positions[0] == pytest.approx(-0.222427965, abs=1e-6)
    lower = runtime.preflight.arms["single"].lower_limits[0]
    upper = runtime.preflight.arms["single"].upper_limits[0]
    assert lower <= state.positions[0] <= upper


def test_runtime_emergency_stop_is_latched_until_explicit_reset():
    runtime = RobotRuntime(_spec())
    runtime.connect()
    try:
        _read_until_available(runtime)
        runtime.emergency_stop()

        with pytest.raises(RuntimeError, match="latched"):
            runtime.set_enabled(False)

        runtime.reset_emergency_stop()
        assert runtime.diagnostics().safety.state == RuntimeSafetyState.SAFE_DISABLED
    finally:
        runtime.disconnect()


def test_physical_packet_fault_reset_requires_external_confirmation():
    runtime = RobotRuntime(_spec(backend=Backend.PHYSICAL))
    runtime._actors = {  # noqa: SLF001 - exercises reset capability policy
        "arm": SimpleNamespace(
            connected=True,
            submit_safety=lambda *args, **kwargs: None,
            discard_motion=lambda: None,
        )
    }
    runtime._safety.connected()  # noqa: SLF001
    runtime._safety.emergency_stop()  # noqa: SLF001

    with pytest.raises(RuntimeError, match="external emergency-stop"):
        runtime.reset_emergency_stop()

    runtime.reset_emergency_stop(external_estop_reset=True)
    assert runtime._safety.snapshot().state == RuntimeSafetyState.SAFE_DISABLED  # noqa: SLF001
