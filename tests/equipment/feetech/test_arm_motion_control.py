import inspect
import time
import types
from collections import deque
from dataclasses import replace
from queue import Full, Queue
from threading import Event, Lock, RLock, Thread, current_thread

import numpy as np
import pytest

from acetele.config.robot_config import ArmConfig
from acetele.equipment.feetech.arm import FeeTechArm
from acetele.equipment.feetech.feetech_driver import (
    FEETECH_COMM_TIMEOUT,
    FEETECH_SIGNED_15_BIT_MAX,
    FeeTechCommandDispatchError,
    FeeTechCommandTimeoutError,
    FeeTechDriver,
    FeeTechStateSample,
    FeeTechStateTimeoutError,
    Mode,
    TorqueEnable,
    _CacheCommitGuard,
)
from acetele.equipment.feetech.feetech_sdk.scservo_def import COMM_SUCCESS, COMM_TX_FAIL
from acetele.equipment.feetech.servo_specs import (
    HLS_PROFILE_DEFAULTS_BY_SERVO,
    KT_MAPPING,
    NO_LOAD_CURRENT,
    PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2,
    PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
)
from acetele.equipment.feetech.state_estimator import FeeTechStateEstimator
from acetele.utils.angle import wrap_to_pi


def profile_payload(
    position_low,
    position_high,
    servo_model="HL3915",
    acceleration=None,
    current=None,
    velocity=None,
):
    defaults = HLS_PROFILE_DEFAULTS_BY_SERVO[servo_model]
    if acceleration is None:
        acceleration = defaults["acceleration"]
    if current is None:
        current = defaults["current"]
    if velocity is None:
        velocity = defaults["velocity"]
    return [
        acceleration,
        position_low,
        position_high,
        current & 0xFF,
        current >> 8,
        velocity & 0xFF,
        velocity >> 8,
    ]


class FakeFeeTechArm(FeeTechArm):
    def __init__(self, current_pos, ids=(0, 1, 2, 3)):
        self._ids = np.array(ids)
        self._dof = len(self._ids)
        self.current_pos = np.array(current_pos, dtype=float)
        self.position_commands = []
        self.torque_commands = []
        self._signs = np.ones(len(self._ids))
        self._servo_models = np.array(["HL3915"] * len(ids))
        self._profile_acceleration_defaults = np.zeros(len(ids), dtype=int)
        self._profile_current_defaults = np.full(len(ids), 500, dtype=int)
        self._profile_velocity_defaults = np.full(len(ids), 250, dtype=int)
        self._torque_current_mapping = np.ones(len(ids)) * (1000.0 / 9.3)
        self._no_load_current = np.ones(len(ids)) * 260
        self._driver = types.SimpleNamespace(
            set_position=lambda ids, positions, **kwargs: self.position_commands.append(
                (
                    np.array(ids),
                    np.array(positions),
                    kwargs,
                )
            )
        )

    def act(self):
        return self.current_pos.copy(), np.zeros(self._dof), np.zeros(self._dof)


def test_arm_set_position_converts_metric_velocity_to_raw_profile_velocity(monkeypatch):
    arm = FakeFeeTechArm(current_pos=[0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        "acetele.equipment.feetech.arm.time.sleep",
        lambda _duration: pytest.fail("profile set_position should not sleep"),
    )

    arm.set_position(
        positions=[-2.0, 0.0, 0.0, 0.0],
        velocities=26 * PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
    )

    assert len(arm.position_commands) == 1
    sent_ids, sent_positions, kwargs = arm.position_commands[0]
    np.testing.assert_array_equal(sent_ids, np.array([0, 1, 2, 3]))
    assert sent_positions[0] == pytest.approx(-1304, abs=1)
    assert "profile" not in kwargs
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.full(4, 26))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.zeros(4, dtype=int))
    np.testing.assert_array_equal(kwargs["currents_raw"], np.full(4, 500))


def test_set_position_profile_uses_single_profile_command(monkeypatch):
    arm = FakeFeeTechArm(current_pos=[0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        "acetele.equipment.feetech.arm.time.sleep",
        lambda _duration: pytest.fail("profile set_position should not sleep"),
    )

    arm.set_position(
        positions=[0.06, 0.0, 0.0, 0.0],
        accelerations=20 * PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2,
    )

    assert len(arm.position_commands) == 1
    assert arm.position_commands[0][1][0] == pytest.approx(39, abs=1)
    np.testing.assert_array_equal(arm.position_commands[0][2]["velocities_raw"], np.full(4, 250))
    np.testing.assert_array_equal(arm.position_commands[0][2]["accelerations_raw"], np.full(4, 20))
    np.testing.assert_array_equal(arm.position_commands[0][2]["currents_raw"], np.full(4, 500))


def test_set_position_explicit_none_velocity_and_acceleration_use_plain_position_write():
    arm = FakeFeeTechArm(current_pos=[0.0, 0.0, 0.0, 0.0])

    arm.set_position(
        positions=[0.06, 0.0, 0.0, 0.0],
        velocities=None,
        accelerations=None,
    )

    assert len(arm.position_commands) == 1
    _, _, kwargs = arm.position_commands[0]
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.full(4, 250))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.zeros(4, dtype=int))
    np.testing.assert_array_equal(kwargs["currents_raw"], np.full(4, 500))


def test_set_position_without_profile_uses_plain_position_write(monkeypatch):
    arm = FakeFeeTechArm(current_pos=[0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        "acetele.equipment.feetech.arm.time.sleep",
        lambda _duration: pytest.fail("plain set_position should not sleep"),
    )

    arm.set_position(
        positions=[0.01, 0.0, 0.0, 0.0],
    )

    assert len(arm.position_commands) == 1
    assert arm.position_commands[0][1][0] == pytest.approx(7, abs=1)
    np.testing.assert_array_equal(arm.position_commands[0][2]["velocities_raw"], np.full(4, 250))
    np.testing.assert_array_equal(arm.position_commands[0][2]["accelerations_raw"], np.zeros(4, dtype=int))
    np.testing.assert_array_equal(arm.position_commands[0][2]["currents_raw"], np.full(4, 500))


def test_set_position_does_not_accept_runtime_step_parameters():
    arm = FakeFeeTechArm(current_pos=[0.0, 0.0, 0.0, 0.0])

    with pytest.raises(TypeError):
        arm.set_position(positions=[0.0, 0.0, 0.0, 1.0], step_size=0.25)


def test_arm_set_position_does_not_accept_public_profile_or_current_parameters():
    arm = FakeFeeTechArm(current_pos=[0.0, 0.0, 0.0, 0.0])

    with pytest.raises(TypeError):
        arm.set_position([0.0, 0.0, 0.0, 1.0], profile=True)

    with pytest.raises(TypeError):
        arm.set_position([0.0, 0.0, 0.0, 1.0], current=1000)

    with pytest.raises(TypeError):
        arm.set_position([0.0, 0.0, 0.0, 1.0], velocity=1.0)

    with pytest.raises(TypeError):
        arm.set_position([0.0, 0.0, 0.0, 1.0], acceleration=1.0)


@pytest.mark.parametrize(
    ("positions", "profiles", "match"),
    [
        ([0.0, 0.0, np.nan, 0.0], {}, "finite"),
        ([0.0] * 4, {"velocities": -0.1}, "non-negative"),
        ([0.0] * 4, {"accelerations": -0.1}, "non-negative"),
        ([0.0] * 4, {"torque": np.inf}, "finite"),
        ([0.0] * 4, {"velocities": 1e9}, "encoded velocity"),
        ([0.0] * 4, {"accelerations": 1e9}, "encoded acceleration"),
        ([0.0] * 4, {"torque": 1e9}, "encoded current"),
        ([0.0] * 4, {"velocities": [1.0, 2.0]}, "match ids length"),
        ([100.0, 0.0, 0.0, 0.0], {}, "encoded positions"),
    ],
)
def test_arm_validates_complete_position_profile_without_dispatch(
    positions,
    profiles,
    match,
):
    arm = FakeFeeTechArm(current_pos=[0.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match=match):
        arm.validate_position_command(positions, **profiles)

    assert arm.position_commands == []


def test_control_loop_sleeps_to_maintain_control_period(monkeypatch):
    arm = FakeFeeTechArm(current_pos=[0.0, 0.0, 0.0, 0.0])
    arm._control_period = 0.004
    arm._stop_flag = type(
        "StopAfterOneIteration",
        (),
        {"calls": 0, "is_set": lambda self: setattr(self, "calls", self.calls + 1) or self.calls > 1},
    )()
    arm._null_space_regulation = lambda joint_pos, joint_vel: np.zeros(4)
    arm._gravity_compensation = lambda joint_pos, joint_vel: np.ones(4)
    arm._friction_compensation = lambda tau_g, joint_vel: np.zeros(4)
    arm.set_torque = lambda tau: arm.torque_commands.append(np.array(tau))

    perf_counter_values = iter([10.0, 10.001])
    sleep_calls = []
    monkeypatch.setattr("acetele.equipment.feetech.arm.time.perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr("acetele.equipment.feetech.arm.time.sleep", sleep_calls.append)

    arm._control_loop()

    assert len(arm.torque_commands) == 1
    assert sleep_calls == [pytest.approx(0.003)]


class FakeSyncWriteHandler:
    def __init__(self):
        self.params = []
        self.avail_flag = Event()
        self.avail_flag.set()

    def addParam(self, ft_id, data):
        self.params.append((ft_id, data))
        return True

    def txPacket(self):
        return 0

    def clearParam(self):
        self.avail_flag.set()


class CountingAvailFlag:
    def __init__(self):
        self.clear_calls = 0

    def is_set(self):
        return True

    def wait(self, _timeout):
        return True

    def clear(self):
        self.clear_calls += 1


class CountingSyncWriteHandler(FakeSyncWriteHandler):
    def __init__(self):
        super().__init__()
        self.tx_calls = []
        self.clear_calls = 0
        self.avail_flag = CountingAvailFlag()
        self.add_fail_ids = set()
        self.tx_result = COMM_SUCCESS

    def addParam(self, ft_id, data):
        if ft_id in self.add_fail_ids:
            return False
        return super().addParam(ft_id, data)

    def txPacket(self):
        self.tx_calls.append(list(self.params))
        return self.tx_result

    def clearParam(self):
        self.clear_calls += 1
        self.params = []


class FailOnceSyncWriteHandler(CountingSyncWriteHandler):
    def __init__(self):
        super().__init__()
        self._failed = False

    def txPacket(self):
        self.tx_calls.append(list(self.params))
        if not self._failed:
            self._failed = True
            return COMM_TX_FAIL
        return COMM_SUCCESS


class FakePacketHandler:
    @staticmethod
    def getTxRxResult(result):
        return f"communication result {result}"

    @staticmethod
    def scs_toscs(value, _bits):
        return int(value)

    @staticmethod
    def scs_lobyte(value):
        return int(value) & 0xFF

    @staticmethod
    def scs_hibyte(value):
        return (int(value) >> 8) & 0xFF


def make_driver_without_hardware(monkeypatch, ids):
    class FakePortHandler:
        def __init__(self, port):
            self.port = port
            self.is_open = False
            self.ser = None

        def openPort(self):
            self.is_open = True
            return True

        def setBaudRate(self, _baudrate):
            return True

        def closePort(self):
            self.is_open = False

        def getPortName(self):
            return self.port

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            del timeout

    monkeypatch.setattr("acetele.equipment.feetech.feetech_driver.PortHandler", FakePortHandler)
    monkeypatch.setattr("acetele.equipment.feetech.feetech_driver.hls", lambda _port_handler: FakePacketHandler())
    monkeypatch.setattr("acetele.equipment.feetech.feetech_driver.GroupSyncRead", lambda *_args: object())
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.GroupSyncWrite",
        lambda *_args: FakeSyncWriteHandler(),
    )
    monkeypatch.setattr("acetele.equipment.feetech.feetech_driver.Thread", FakeThread)
    monkeypatch.setattr("acetele.equipment.feetech.feetech_driver.tqdm", lambda iterable: iterable)
    monkeypatch.setattr(
        FeeTechDriver,
        "get_state",
        lambda self, timeout=1.0: (
            {int(ft_id): 0 for ft_id in self._ids},
            {int(ft_id): 0 for ft_id in self._ids},
            {int(ft_id): 0 for ft_id in self._ids},
        ),
    )
    return FeeTechDriver(ids, "/dev/test")


def install_driver_initialization_fakes(monkeypatch, *, close_error=None):
    records = {
        "ports": [],
        "threads": [],
    }

    class FakeSerial:
        def __init__(self):
            self.cancel_calls = []

        def cancel_read(self):
            self.cancel_calls.append("read")

        def cancel_write(self):
            self.cancel_calls.append("write")

    class FakePortHandler:
        def __init__(self, port):
            self.port = port
            self.is_open = False
            self.ser = FakeSerial()
            self.close_calls = 0
            records["ports"].append(self)

        def openPort(self):
            self.is_open = True
            return True

        def setBaudRate(self, _baudrate):
            return True

        def closePort(self):
            self.close_calls += 1
            self.is_open = False
            if close_error is not None:
                raise close_error

        def getPortName(self):
            return self.port

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon
            self.started = False
            self.join_timeouts = []
            records["threads"].append(self)

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

        def join(self, timeout=None):
            self.join_timeouts.append(timeout)
            self.started = False

    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.PortHandler",
        FakePortHandler,
    )
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.hls",
        lambda _port_handler: FakePacketHandler(),
    )
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.GroupSyncRead",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.GroupSyncWrite",
        lambda *_args: FakeSyncWriteHandler(),
    )
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.Thread",
        FakeThread,
    )
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.tqdm",
        lambda iterable: iterable,
    )
    return records


def test_driver_initialization_cleans_up_after_warmup_failure(monkeypatch):
    records = install_driver_initialization_fakes(monkeypatch)
    monkeypatch.setattr(
        FeeTechDriver,
        "get_state",
        lambda _self, timeout=1.0: (_ for _ in ()).throw(
            FeeTechStateTimeoutError("warmup failed")
        ),
    )

    with pytest.raises(FeeTechStateTimeoutError, match="warmup failed"):
        FeeTechDriver([1], "/dev/test")

    port = records["ports"][0]
    thread = records["threads"][0]
    assert port.close_calls == 1
    assert not port.is_open
    assert port.ser.cancel_calls == ["read", "write"]
    assert thread.join_timeouts == [pytest.approx(1.0)]
    assert not thread.is_alive()
    assert thread.target.__self__._stop_flag.is_set()


def test_driver_initialization_cleans_up_after_initial_mode_failure(monkeypatch):
    records = install_driver_initialization_fakes(monkeypatch)
    mode_calls = []
    monkeypatch.setattr(
        FeeTechDriver,
        "get_state",
        lambda self, timeout=1.0: (
            {int(ft_id): 0 for ft_id in self._ids},
            {int(ft_id): 0 for ft_id in self._ids},
            {int(ft_id): 0 for ft_id in self._ids},
        ),
    )

    def fail_initial_mode(_self, ids, modes, **kwargs):
        mode_calls.append((tuple(ids), tuple(modes), kwargs))
        raise FeeTechCommandDispatchError("initial mode failed")

    monkeypatch.setattr(FeeTechDriver, "_submit_mode_transaction", fail_initial_mode)

    with pytest.raises(FeeTechCommandDispatchError, match="initial mode failed"):
        FeeTechDriver([1], "/dev/test")

    assert records["ports"][0].close_calls == 1
    assert records["threads"][0].join_timeouts == [pytest.approx(1.0)]
    assert not records["threads"][0].is_alive()
    assert mode_calls == [
        ((1,), (Mode.Position,), {"force": True, "wait": True})
    ]


def test_driver_initialization_closes_port_when_handler_setup_fails(monkeypatch):
    records = install_driver_initialization_fakes(monkeypatch)
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.hls",
        lambda _port_handler: (_ for _ in ()).throw(
            RuntimeError("handler setup failed")
        ),
    )

    with pytest.raises(RuntimeError, match="handler setup failed"):
        FeeTechDriver([1], "/dev/test")

    assert records["ports"][0].close_calls == 1
    assert records["threads"] == []


def test_driver_initialization_preserves_error_when_cleanup_also_fails(monkeypatch):
    cleanup_error = RuntimeError("close failed")
    install_driver_initialization_fakes(
        monkeypatch,
        close_error=cleanup_error,
    )
    monkeypatch.setattr(
        FeeTechDriver,
        "get_state",
        lambda _self, timeout=1.0: (_ for _ in ()).throw(
            FeeTechStateTimeoutError("warmup failed")
        ),
    )

    with pytest.raises(FeeTechStateTimeoutError, match="warmup failed") as exc_info:
        FeeTechDriver([1], "/dev/test")

    assert exc_info.value.__cause__ is cleanup_error


@pytest.mark.parametrize(
    "invalid_ids",
    ([1.0], ["1"], [True], [[1]], np.array([[1]]), [253], [1, 1]),
)
def test_driver_rejects_invalid_servo_ids_before_opening_port(
    monkeypatch,
    invalid_ids,
):
    port_calls = []
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.PortHandler",
        lambda port: port_calls.append(port),
    )

    with pytest.raises(ValueError, match="driver ids"):
        FeeTechDriver(invalid_ids, "/dev/test")

    assert port_calls == []


@pytest.mark.parametrize(
    "invoke",
    [
        lambda driver, ids: driver.set_mode(ids, [Mode.Position] * len(ids)),
        lambda driver, ids: driver.set_torque_enable(
            ids,
            [TorqueEnable.Disable] * len(ids),
        ),
        lambda driver, ids: driver.set_current(ids, [0] * len(ids)),
        lambda driver, ids: driver.set_position(
            ids,
            [0] * len(ids),
            currents_raw=[0] * len(ids),
            velocities_raw=[0] * len(ids),
            accelerations_raw=[0] * len(ids),
        ),
        lambda driver, ids: driver.calibrate(ids, [0] * len(ids)),
    ],
)
@pytest.mark.parametrize(
    "invalid_ids",
    ([1.0], ["1"], [True], [[1]], np.array([[1]]), [2], [1, 1]),
)
def test_driver_public_commands_reject_invalid_ids_before_dispatch(
    invoke,
    invalid_ids,
):
    driver = BaseMethodDriver()

    with pytest.raises(ValueError, match="ids"):
        invoke(driver, invalid_ids)

    assert driver._comm_task_queue.empty()


def test_driver_rejects_raw_enum_values_before_dispatch():
    driver = BaseMethodDriver()

    with pytest.raises(ValueError, match="Mode"):
        driver.set_mode([1], [Mode.Position.value])
    with pytest.raises(ValueError, match="TorqueEnable"):
        driver.set_torque_enable([1], [TorqueEnable.Disable.value])

    assert driver._comm_task_queue.empty()


class PositionOnlyDriver(FeeTechDriver):
    def __init__(self):
        self._ids = [1]
        self._mode = {1: Mode.Position}
        self._torque_enable = {1: TorqueEnable.Enable}
        self._packetHandler = FakePacketHandler()
        self._groupSyncWriteModeHandler = FakeSyncWriteHandler()
        self._groupSyncWriteTorqueEnableHandler = FakeSyncWriteHandler()
        self._groupSyncWriteGoalCurrentHandler = FakeSyncWriteHandler()
        self._groupSyncWriteGoalPositionProfileHandler = FakeSyncWriteHandler()
        self._time_windows = deque(maxlen=100)
        self._lock = Lock()
        self._lifecycle_lock = RLock()
        self._stop_flag = Event()
        self._comm_fault = None
        self._comm_thread = types.SimpleNamespace(is_alive=lambda: True)
        self.queued_tasks = []

        class ImmediateQueue:
            def __init__(self, outer):
                self.outer = outer

            def put(self, task, timeout=None):
                del timeout
                self.outer.queued_tasks.append(task)
                task()

        self._comm_task_queue = ImmediateQueue(self)

    def get_state(self):
        return ({1: 5000}, {}, {})


class MultiIdPositionDriver(PositionOnlyDriver):
    def __init__(self):
        super().__init__()
        self._ids = [1, 2]
        self._mode = {1: Mode.Position, 2: Mode.Position}
        self._torque_enable = {1: TorqueEnable.Enable, 2: TorqueEnable.Enable}

    def get_state(self):
        return ({1: 5000, 2: 5000}, {}, {})


class BaseMethodDriver(FeeTechDriver):
    def __init__(self):
        self._ids = [1]
        self._mode = {1: Mode.Position}
        self._torque_enable = {1: TorqueEnable.Enable}
        self._packetHandler = FakePacketHandler()
        self._groupSyncWriteModeHandler = CountingSyncWriteHandler()
        self._groupSyncWriteTorqueEnableHandler = CountingSyncWriteHandler()
        self._groupSyncWriteGoalCurrentHandler = CountingSyncWriteHandler()
        self._groupSyncWriteGoalPositionProfileHandler = CountingSyncWriteHandler()
        self._comm_task_queue = Queue(maxsize=32)
        self._time_windows = deque(maxlen=100)
        self._stop_flag = Event()
        self._comm_fault = None
        self._comm_thread = types.SimpleNamespace(is_alive=lambda: True)
        self._portHandler = types.SimpleNamespace(is_open=True, ser=None)
        self._lock = Lock()
        self._lifecycle_lock = RLock()
        self._position = {1: 5000}
        self._velocity = {1: 0}
        self._current = {1: 0}
        self._state_timestamp = 1.25
        self._state_sequence = 4
        self.read_calls = 0
        self.executed_tasks = []

    def get_state(self):
        return ({1: 5000}, {}, {})

    def _read_state(self):
        self.read_calls += 1


class MixedServoPositionDriver(PositionOnlyDriver):
    def __init__(self):
        super().__init__()
        self._ids = [0, 1, 2]
        self._mode = {0: Mode.Position, 1: Mode.Position, 2: Mode.Position}
        self._torque_enable = {0: TorqueEnable.Enable, 1: TorqueEnable.Enable, 2: TorqueEnable.Enable}

    def get_state(self):
        return ({0: 5000, 1: 5000, 2: 5000}, {}, {})


def test_set_position_sends_multiturn_adjusted_position():
    driver = PositionOnlyDriver()

    driver.set_position([1], [100], velocities_raw=[250], accelerations_raw=[0], currents_raw=[500])

    assert driver._groupSyncWriteGoalPositionProfileHandler.params == [(1, profile_payload(100, 16))]


def test_hls_profile_defaults_match_official_initial_values():
    assert HLS_PROFILE_DEFAULTS_BY_SERVO == {
        "HL3960": {"acceleration": 0, "current": 1000, "velocity": 110},
        "HL3950": {"acceleration": 0, "current": 1000, "velocity": 110},
        "HL3930": {"acceleration": 250, "current": 1000, "velocity": 100},
        "HL3915": {"acceleration": 0, "current": 500, "velocity": 250},
    }


def test_hl3960_servo_specs_use_supplied_constants_and_hl3950_profile():
    assert KT_MAPPING["HL3960"] == pytest.approx(1.0 / 14.84)
    assert NO_LOAD_CURRENT["HL3960"] == 300
    assert HLS_PROFILE_DEFAULTS_BY_SERVO["HL3960"] == HLS_PROFILE_DEFAULTS_BY_SERVO["HL3950"]


def test_feetech_driver_does_not_accept_servo_models(monkeypatch):
    with pytest.raises(TypeError):
        FeeTechDriver([0, 1], "/dev/test", servo_models=["HL3915", "HL3915"])


def test_set_position_requires_full_raw_profile_fields():
    driver = PositionOnlyDriver()

    with pytest.raises(ValueError, match="velocities_raw, accelerations_raw, and currents_raw are required"):
        driver.set_position([1], [100], currents_raw=[33])


def test_set_position_profile_sends_acceleration_position_current_and_velocity():
    driver = PositionOnlyDriver()

    driver.set_position([1], [100], velocities_raw=[26], accelerations_raw=[20], currents_raw=[82])

    assert driver._groupSyncWriteGoalPositionProfileHandler.params == [(1, [20, 100, 16, 82, 0, 26, 0])]


def test_set_position_profile_rejects_out_of_range_profile_values():
    driver = PositionOnlyDriver()

    with pytest.raises(ValueError, match="accelerations must be between 0 and 255"):
        driver.set_position([1], [100], velocities_raw=[26], accelerations_raw=[256], currents_raw=[82])

    with pytest.raises(ValueError, match="velocities must be between 0 and 32767"):
        driver.set_position([1], [100], velocities_raw=[32768], accelerations_raw=[20], currents_raw=[82])

    with pytest.raises(ValueError, match="currents must be between 0 and 32767"):
        driver.set_position([1], [100], velocities_raw=[26], accelerations_raw=[20], currents_raw=[32768])


def test_set_position_profile_rejects_negative_raw_profile_values():
    driver = PositionOnlyDriver()

    with pytest.raises(ValueError, match="velocities must be between 0 and 32767"):
        driver.set_position([1], [100], velocities_raw=[-1], accelerations_raw=[20], currents_raw=[82])

    with pytest.raises(ValueError, match="accelerations must be between 0 and 255"):
        driver.set_position([1], [100], velocities_raw=[26], accelerations_raw=[-1], currents_raw=[82])

    with pytest.raises(ValueError, match="currents must be between 0 and 32767"):
        driver.set_position([1], [100], velocities_raw=[26], accelerations_raw=[20], currents_raw=[-1])


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("goal_positions_raw", [1.0]),
        ("goal_positions_raw", [True]),
        ("goal_positions_raw", ["1"]),
        ("goal_positions_raw", np.array([[1]])),
        ("goal_positions_raw", [FEETECH_SIGNED_15_BIT_MAX + 1]),
        ("velocities_raw", [1.0]),
        ("velocities_raw", [True]),
        ("accelerations_raw", [1.0]),
        ("accelerations_raw", np.array([[1]])),
        ("currents_raw", [1.0]),
        ("currents_raw", [True]),
    ],
)
def test_set_position_rejects_non_integer_raw_register_values_before_dispatch(
    field,
    invalid_value,
):
    driver = BaseMethodDriver()
    kwargs = {
        "goal_positions_raw": [100],
        "velocities_raw": [1],
        "accelerations_raw": [1],
        "currents_raw": [1],
    }
    kwargs[field] = invalid_value

    with pytest.raises(ValueError):
        driver.set_position([1], **kwargs)

    assert driver._comm_task_queue.empty()


def test_set_position_empty_ids_returns_without_profile_parameters():
    driver = PositionOnlyDriver()

    driver.set_position([], [], velocities_raw=None, accelerations_raw=None, currents_raw=None)

    assert driver._groupSyncWriteGoalPositionProfileHandler.params == []


def test_set_position_after_custom_profile_writes_full_initial_profile_registers():
    driver = PositionOnlyDriver()

    driver.set_position([1], [99], velocities_raw=[26], accelerations_raw=[20], currents_raw=[82])
    driver._groupSyncWriteGoalPositionProfileHandler.params = []
    driver.set_position([1], [100], velocities_raw=[250], accelerations_raw=[0], currents_raw=[500])

    assert driver._groupSyncWriteGoalPositionProfileHandler.params == [(1, profile_payload(100, 16))]


def test_set_position_always_writes_full_profile_for_all_ids():
    driver = MultiIdPositionDriver()

    driver.set_position(
        [1, 2],
        [100, 200],
        velocities_raw=[250, 250],
        accelerations_raw=[0, 0],
        currents_raw=[500, 500],
    )

    assert driver._groupSyncWriteGoalPositionProfileHandler.params == [
        (1, profile_payload(100, 16)),
        (2, profile_payload(200, 16)),
    ]


def test_set_position_uses_servo_model_defaults_for_selected_ids():
    driver = MixedServoPositionDriver()

    driver.set_position(
        [0, 1, 2],
        [100, 200, 250],
        velocities_raw=[110, 100, 250],
        accelerations_raw=[0, 250, 0],
        currents_raw=[1000, 1000, 500],
    )

    assert driver._groupSyncWriteGoalPositionProfileHandler.params == [
        (0, profile_payload(100, 16, servo_model="HL3950")),
        (1, profile_payload(200, 16, servo_model="HL3930")),
        (2, profile_payload(250, 16, servo_model="HL3915")),
    ]


def test_set_position_model_defaults_fill_only_omitted_profile_fields():
    arm = FakeFeeTechArm(current_pos=[0.0, 0.0, 0.0], ids=(0, 1, 2))
    arm._servo_models = np.array(["HL3950", "HL3930", "HL3915"])
    arm._profile_acceleration_defaults = np.array([0, 250, 0])
    arm._profile_current_defaults = np.array([1000, 1000, 500])
    arm._profile_velocity_defaults = np.array([110, 100, 250])

    arm.set_position([0.1, 0.2, 0.3], ids=[0, 1, 2], torque=[0.1, 0.1, 0.1])
    _, _, kwargs = arm.position_commands[-1]
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.array([110, 100, 250]))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.array([0, 250, 0]))

    arm.position_commands = []
    arm.set_position([0.1, 0.2, 0.3], ids=[0, 1, 2], velocities=[1.0, 1.0, 1.0])
    _, _, kwargs = arm.position_commands[-1]
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.array([0, 250, 0]))
    np.testing.assert_array_equal(kwargs["currents_raw"], np.array([1000, 1000, 500]))

    arm.position_commands = []
    arm.set_position([0.1, 0.2, 0.3], ids=[0, 1, 2], accelerations=[1.0, 1.0, 1.0])
    _, _, kwargs = arm.position_commands[-1]
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.array([110, 100, 250]))
    np.testing.assert_array_equal(kwargs["currents_raw"], np.array([1000, 1000, 500]))


def test_set_position_without_profile_arguments_uses_full_profile_write():
    driver = PositionOnlyDriver()

    driver.set_position([1], [100], velocities_raw=[250], accelerations_raw=[0], currents_raw=[500])

    assert driver._groupSyncWriteGoalPositionProfileHandler.params == [(1, profile_payload(100, 16))]


def test_set_position_switches_from_torque_to_position_mode():
    driver = BaseMethodDriver()
    driver._mode = {1: Mode.Torque}

    driver.set_position([1], [100], velocities_raw=[250], accelerations_raw=[0], currents_raw=[500])
    driver._comm_task_queue.get_nowait()()

    assert driver._groupSyncWriteTorqueEnableHandler.tx_calls == [
        [(1, [TorqueEnable.Disable.value])],
        [(1, [TorqueEnable.Enable.value])],
    ]
    assert driver._groupSyncWriteModeHandler.tx_calls == [[(1, [Mode.Position.value])]]
    assert driver._mode == {1: Mode.Position}
    assert driver._torque_enable == {1: TorqueEnable.Enable}


def test_set_mode_noop_transaction_does_not_touch_handler():
    driver = BaseMethodDriver()

    driver.set_mode([1], [Mode.Position])
    driver._comm_task_queue.get_nowait()()

    assert driver._comm_task_queue.empty()
    assert driver._groupSyncWriteModeHandler.tx_calls == []
    assert driver._groupSyncWriteModeHandler.avail_flag.clear_calls == 0


def test_set_torque_enable_noop_transaction_does_not_touch_handler():
    driver = BaseMethodDriver()

    driver.set_torque_enable([1], [TorqueEnable.Enable])
    driver._comm_task_queue.get_nowait()()

    assert driver._comm_task_queue.empty()
    assert driver._groupSyncWriteTorqueEnableHandler.tx_calls == []
    assert driver._groupSyncWriteTorqueEnableHandler.avail_flag.clear_calls == 0


def test_blocking_torque_enable_waits_for_exact_write_task():
    driver = BaseMethodDriver()
    driver._stop_flag = Event()

    def consume_one_task():
        driver._comm_task_queue.get(timeout=1.0)()

    driver._comm_thread = Thread(target=consume_one_task)
    driver._comm_thread.start()

    driver.set_torque_enable(
        [1],
        [TorqueEnable.Disable],
        force=True,
        wait=True,
        timeout=1.0,
    )
    driver._comm_thread.join()

    assert driver._groupSyncWriteTorqueEnableHandler.tx_calls == [[(1, [0])]]
    assert driver._torque_enable == {1: TorqueEnable.Disable}


def test_blocking_torque_cache_commit_preserves_worker_fifo_order():
    driver = BaseMethodDriver()
    driver._stop_flag = Event()
    disable_write_started = Event()
    later_enable_committed = Event()
    caller_errors = []

    original_write_sync_params = driver._write_sync_params

    def observed_write_sync_params(handler, params, label, *args, **kwargs):
        original_write_sync_params(handler, params, label, *args, **kwargs)
        if label == "groupSyncWriteTorqueEnable" and params == [(1, [0])]:
            disable_write_started.set()

    driver._write_sync_params = observed_write_sync_params
    original_update_cache = driver._update_torque_enable_cache
    blocking_caller = None

    def controlled_update_cache(ids, enables):
        if current_thread() is blocking_caller and tuple(enables) == (
            TorqueEnable.Disable,
        ):
            later_enable_committed.wait(timeout=0.5)
        original_update_cache(ids, enables)
        if tuple(enables) == (TorqueEnable.Enable,):
            later_enable_committed.set()

    driver._update_torque_enable_cache = controlled_update_cache

    def consume_two_tasks():
        for _ in range(2):
            driver._comm_task_queue.get(timeout=1.0)()

    def disable_torque_and_wait():
        try:
            driver.set_torque_enable(
                [1],
                [TorqueEnable.Disable],
                force=True,
                wait=True,
                timeout=1.0,
            )
        except Exception as exc:  # pragma: no cover - asserted below
            caller_errors.append(exc)

    driver._comm_thread = Thread(target=consume_two_tasks)
    blocking_caller = Thread(target=disable_torque_and_wait)
    driver._comm_thread.start()
    blocking_caller.start()

    assert disable_write_started.wait(timeout=1.0)
    driver.set_torque_enable([1], [TorqueEnable.Enable])

    blocking_caller.join(timeout=2.0)
    driver._comm_thread.join(timeout=2.0)

    assert not blocking_caller.is_alive()
    assert not driver._comm_thread.is_alive()
    assert caller_errors == []
    assert driver._groupSyncWriteTorqueEnableHandler.tx_calls == [
        [(1, [TorqueEnable.Disable.value])],
        [(1, [TorqueEnable.Enable.value])],
    ]
    assert driver._torque_enable == {1: TorqueEnable.Enable}

    driver._comm_thread = types.SimpleNamespace(is_alive=lambda: True)
    driver.set_mode([1], [Mode.Torque])
    driver._comm_task_queue.get_nowait()()

    assert driver._groupSyncWriteTorqueEnableHandler.tx_calls[-1] == [
        (1, [TorqueEnable.Disable.value])
    ]
    assert driver._groupSyncWriteModeHandler.tx_calls == [
        [(1, [Mode.Torque.value])]
    ]


def test_completed_waited_torque_task_does_not_time_out_or_erase_newer_cache(
    monkeypatch,
):
    driver = BaseMethodDriver()
    driver._stop_flag = Event()
    release_waiter = Event()
    disable_written = Event()

    class DelayedObservationEvent:
        def __init__(self):
            self._completed = Event()

        def set(self):
            self._completed.set()

        def is_set(self):
            release_waiter.wait(timeout=1.0)
            return self._completed.is_set()

        def wait(self, timeout=None):
            release_waiter.wait(timeout=timeout)
            return self._completed.is_set()

    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.Event",
        DelayedObservationEvent,
    )
    original_write_sync_params = driver._write_sync_params

    def observed_write_sync_params(handler, params, label, *args, **kwargs):
        original_write_sync_params(handler, params, label, *args, **kwargs)
        if label == "groupSyncWriteTorqueEnable" and params == [(1, [0])]:
            disable_written.set()

    driver._write_sync_params = observed_write_sync_params
    caller_errors = []

    def consume_two_tasks():
        for _ in range(2):
            driver._comm_task_queue.get(timeout=1.0)()
        time.sleep(0.03)
        release_waiter.set()

    def disable_and_wait():
        try:
            driver.set_torque_enable(
                [1],
                [TorqueEnable.Disable],
                force=True,
                wait=True,
                timeout=0.01,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            caller_errors.append(exc)

    driver._comm_thread = Thread(target=consume_two_tasks)
    waiting_caller = Thread(target=disable_and_wait)
    driver._comm_thread.start()
    waiting_caller.start()

    assert disable_written.wait(timeout=1.0)
    driver.set_torque_enable([1], [TorqueEnable.Enable])

    waiting_caller.join(timeout=2.0)
    driver._comm_thread.join(timeout=2.0)

    assert caller_errors == []
    assert driver._torque_enable == {1: TorqueEnable.Enable}


def test_blocking_torque_enable_rejects_stopped_comm_thread():
    driver = BaseMethodDriver()
    driver._stop_flag = Event()
    driver._comm_thread = types.SimpleNamespace(is_alive=lambda: False)

    with pytest.raises(FeeTechCommandDispatchError, match="thread is not running"):
        driver.set_torque_enable(
            [1],
            [TorqueEnable.Disable],
            force=True,
            wait=True,
        )

    assert driver._comm_task_queue.empty()
    assert driver._torque_enable == {1: None}


def test_blocking_torque_enable_reports_full_queue():
    driver = BaseMethodDriver()
    driver._stop_flag = Event()
    driver._comm_thread = types.SimpleNamespace(is_alive=lambda: True)

    class FullQueue:
        def put(self, _task, timeout=None):
            del timeout
            raise Full

    driver._comm_task_queue = FullQueue()

    with pytest.raises(FeeTechCommandDispatchError, match="queue is full"):
        driver.set_torque_enable(
            [1],
            [TorqueEnable.Disable],
            force=True,
            wait=True,
        )

    assert driver._torque_enable == {1: None}


def test_blocking_torque_enable_times_out_when_task_is_not_consumed():
    driver = BaseMethodDriver()
    driver._stop_flag = Event()
    driver._comm_thread = types.SimpleNamespace(is_alive=lambda: True)

    with pytest.raises(FeeTechCommandTimeoutError, match="Timed out"):
        driver.set_torque_enable(
            [1],
            [TorqueEnable.Disable],
            force=True,
            wait=True,
            timeout=0.01,
        )

    cancelled_task = driver._comm_task_queue.get_nowait()
    cancelled_task()

    assert driver._groupSyncWriteTorqueEnableHandler.tx_calls == []
    assert driver._torque_enable == {1: None}


def test_started_torque_task_timeout_invalidates_cache_and_forces_next_command():
    driver = BaseMethodDriver()
    driver._stop_flag = Event()
    task_started = Event()
    release_task = Event()

    def blocking_write(*_args):
        task_started.set()
        release_task.wait(timeout=1.0)

    driver._write_sync_params = blocking_write

    def consume_one_task():
        driver._comm_task_queue.get(timeout=1.0)()

    driver._comm_thread = Thread(target=consume_one_task)
    driver._comm_thread.start()

    with pytest.raises(FeeTechCommandTimeoutError, match="Timed out"):
        driver.set_torque_enable(
            [1],
            [TorqueEnable.Disable],
            force=True,
            wait=True,
            timeout=0.05,
        )

    assert task_started.is_set()
    assert driver._torque_enable == {1: None}
    release_task.set()
    driver._comm_thread.join()
    assert driver._torque_enable == {1: None}

    driver._comm_thread = types.SimpleNamespace(is_alive=lambda: True)
    driver.set_torque_enable([1], [TorqueEnable.Enable])

    assert driver._comm_task_queue.qsize() == 1


def test_timed_out_torque_task_cannot_erase_later_fifo_commit(monkeypatch):
    driver = BaseMethodDriver()
    driver._stop_flag = Event()
    first_write_started = Event()
    release_first_write = Event()
    abandon_started = Event()
    release_abandon = Event()
    later_enable_committed = Event()
    caller_errors = []

    original_write_sync_params = driver._write_sync_params

    def blocking_first_write(handler, params, label, *args, **kwargs):
        original_write_sync_params(handler, params, label, *args, **kwargs)
        if label == "groupSyncWriteTorqueEnable" and params == [(1, [0])]:
            first_write_started.set()
            release_first_write.wait(timeout=1.0)

    driver._write_sync_params = blocking_first_write
    original_update_cache = driver._update_torque_enable_cache

    def observed_update_cache(ids, enables):
        original_update_cache(ids, enables)
        if tuple(enables) == (TorqueEnable.Enable,):
            later_enable_committed.set()

    driver._update_torque_enable_cache = observed_update_cache
    original_invalidate = _CacheCommitGuard.invalidate

    def delayed_invalidate(self):
        abandon_started.set()
        release_abandon.wait(timeout=1.0)
        original_invalidate(self)

    monkeypatch.setattr(_CacheCommitGuard, "invalidate", delayed_invalidate)

    def consume_two_tasks():
        for _ in range(2):
            driver._comm_task_queue.get(timeout=1.0)()

    def disable_and_wait():
        try:
            driver.set_torque_enable(
                [1],
                [TorqueEnable.Disable],
                force=True,
                wait=True,
                timeout=0.02,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            caller_errors.append(exc)

    driver._comm_thread = Thread(target=consume_two_tasks)
    waiting_caller = Thread(target=disable_and_wait)
    driver._comm_thread.start()
    waiting_caller.start()

    assert first_write_started.wait(timeout=1.0)
    assert abandon_started.wait(timeout=1.0)
    driver.set_torque_enable([1], [TorqueEnable.Enable])
    release_first_write.set()

    newer_task_overtook_abandonment = later_enable_committed.wait(timeout=0.1)
    release_abandon.set()
    waiting_caller.join(timeout=2.0)
    driver._comm_thread.join(timeout=2.0)

    assert not waiting_caller.is_alive()
    assert not driver._comm_thread.is_alive()
    assert not newer_task_overtook_abandonment
    assert len(caller_errors) == 1
    assert isinstance(caller_errors[0], FeeTechCommandTimeoutError)
    assert driver._torque_enable == {1: TorqueEnable.Enable}


def test_blocking_torque_enable_surfaces_write_task_exception():
    driver = BaseMethodDriver()
    driver._stop_flag = Event()
    driver._write_sync_params = lambda *_args: (_ for _ in ()).throw(RuntimeError("write failed"))

    def consume_one_task():
        with pytest.raises(RuntimeError, match="write failed"):
            driver._comm_task_queue.get(timeout=1.0)()

    driver._comm_thread = Thread(target=consume_one_task)
    driver._comm_thread.start()

    with pytest.raises(FeeTechCommandDispatchError, match="Failed to execute") as exc_info:
        driver.set_torque_enable(
            [1],
            [TorqueEnable.Disable],
            force=True,
            wait=True,
            timeout=1.0,
        )
    driver._comm_thread.join()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert driver._torque_enable == {1: None}


def test_sync_write_handler_wait_is_bounded():
    driver = BaseMethodDriver()
    handler = CountingSyncWriteHandler()
    handler.avail_flag = Event()

    with pytest.raises(FeeTechCommandTimeoutError, match="handler to become available"):
        driver._write_sync_params(handler, [(1, [0])], "testSyncWrite", timeout=0.001)

    assert handler.tx_calls == []


def test_sync_write_handler_is_released_after_write_exception():
    driver = BaseMethodDriver()
    handler = CountingSyncWriteHandler()
    handler.txPacket = lambda: (_ for _ in ()).throw(RuntimeError("write failed"))

    with pytest.raises(RuntimeError, match="write failed"):
        driver._write_sync_params(handler, [(1, [0])], "testSyncWrite")

    assert handler.clear_calls == 1


def test_sync_write_aborts_without_transmitting_partial_parameters():
    driver = BaseMethodDriver()
    handler = CountingSyncWriteHandler()
    handler.add_fail_ids = {2}

    with pytest.raises(FeeTechCommandDispatchError, match=r"ID:2.*addParam failed"):
        driver._write_sync_params(
            handler,
            [(1, [0]), (2, [0])],
            "testSyncWrite",
        )

    assert handler.tx_calls == []
    assert handler.clear_calls == 1
    assert handler.params == []


def test_sync_write_rejects_unsuccessful_sdk_result_and_releases_handler():
    driver = BaseMethodDriver()
    handler = CountingSyncWriteHandler()
    handler.tx_result = COMM_TX_FAIL

    with pytest.raises(
        FeeTechCommandDispatchError,
        match=r"testSyncWrite.*communication result -2",
    ):
        driver._write_sync_params(handler, [(1, [0])], "testSyncWrite")

    assert handler.tx_calls == [[(1, [0])]]
    assert handler.clear_calls == 1
    assert handler.params == []


def test_blocking_torque_write_sdk_failure_invalidates_cache():
    driver = BaseMethodDriver()
    driver._stop_flag = Event()
    driver._groupSyncWriteTorqueEnableHandler.tx_result = COMM_TX_FAIL

    def consume_one_task():
        with pytest.raises(FeeTechCommandDispatchError):
            driver._comm_task_queue.get(timeout=1.0)()

    driver._comm_thread = Thread(target=consume_one_task)
    driver._comm_thread.start()

    with pytest.raises(FeeTechCommandDispatchError, match="Failed to execute"):
        driver.set_torque_enable(
            [1],
            [TorqueEnable.Disable],
            force=True,
            wait=True,
        )
    driver._comm_thread.join()

    assert driver._torque_enable == {1: None}


class LifecycleSerialPort:
    def __init__(self):
        self.events = []

    def cancel_read(self):
        self.events.append("cancel_read")

    def cancel_write(self):
        self.events.append("cancel_write")


class LifecyclePortHandler:
    def __init__(self):
        self.ser = LifecycleSerialPort()
        self.is_open = True

    def closePort(self):
        self.ser.events.append("close_port")
        self.is_open = False


class LifecycleThread:
    def __init__(self, *, stops_on_join):
        self._alive = True
        self._stops_on_join = stops_on_join
        self.join_timeouts = []

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)
        if self._stops_on_join:
            self._alive = False


def test_driver_close_cancels_io_closes_port_and_uses_bounded_join():
    driver = BaseMethodDriver()
    driver._portHandler = LifecyclePortHandler()
    driver._comm_thread = LifecycleThread(stops_on_join=True)

    driver.close(timeout=0.02)

    assert driver._portHandler.ser.events == ["cancel_read", "cancel_write", "close_port"]
    assert driver._comm_thread.join_timeouts == [0.02]
    assert driver._mode == {1: None}
    assert driver._torque_enable == {1: None}


def test_driver_close_times_out_without_unbounded_second_join():
    driver = BaseMethodDriver()
    driver._portHandler = LifecyclePortHandler()
    driver._comm_thread = LifecycleThread(stops_on_join=False)

    with pytest.raises(FeeTechCommandTimeoutError, match="thread to close"):
        driver.close(timeout=0.02)

    assert driver._comm_thread.join_timeouts == [0.02]
    assert driver._portHandler.ser.events == ["cancel_read", "cancel_write", "close_port"]
    assert driver._mode == {1: None}
    assert driver._torque_enable == {1: None}


def test_calibration_pause_keeps_serial_port_open():
    driver = BaseMethodDriver()
    driver._portHandler = LifecyclePortHandler()
    driver._comm_thread = LifecycleThread(stops_on_join=True)

    driver._pause_comm_worker(timeout=0.02)

    assert driver._comm_thread.join_timeouts == [0.02]
    assert driver._portHandler.is_open
    assert driver._portHandler.ser.events == []


def test_open_invalidates_mode_and_torque_cache(monkeypatch):
    driver = BaseMethodDriver()
    driver._comm_fault = RuntimeError("old fault")
    driver._comm_thread = types.SimpleNamespace(is_alive=lambda: False)
    started = []
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.Thread",
        lambda target, daemon: types.SimpleNamespace(start=lambda: started.append((target, daemon))),
    )

    driver.open()

    assert driver._mode == {1: None}
    assert driver._torque_enable == {1: None}
    assert driver._comm_fault is None
    assert started == [(driver._comm_worker, True)]


def test_open_waits_for_faulted_worker_before_restarting(monkeypatch):
    driver = BaseMethodDriver()
    driver._comm_fault = RuntimeError("old fault")
    previous_thread = LifecycleThread(stops_on_join=True)
    driver._comm_thread = previous_thread
    started = []
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.Thread",
        lambda target, daemon: types.SimpleNamespace(
            start=lambda: started.append((target, daemon))
        ),
    )

    driver.open()

    assert previous_thread.join_timeouts == [FEETECH_COMM_TIMEOUT]
    assert driver._comm_fault is None
    assert started == [(driver._comm_worker, True)]


def test_open_rejects_faulted_worker_that_does_not_stop():
    driver = BaseMethodDriver()
    fault = RuntimeError("old fault")
    driver._comm_fault = fault
    previous_thread = LifecycleThread(stops_on_join=False)
    driver._comm_thread = previous_thread

    with pytest.raises(FeeTechCommandTimeoutError, match="before reopening") as exc_info:
        driver.open()

    assert previous_thread.join_timeouts == [FEETECH_COMM_TIMEOUT]
    assert exc_info.value.__cause__ is fault
    assert driver._comm_fault is fault
    assert driver._mode == {1: None}
    assert driver._torque_enable == {1: None}


def test_set_position_with_position_mode_and_torque_enabled_enqueues_only_profile_position_task():
    driver = BaseMethodDriver()

    driver.set_position([1], [100], velocities_raw=[250], accelerations_raw=[0], currents_raw=[500])

    assert driver._comm_task_queue.qsize() == 1
    assert driver._groupSyncWriteModeHandler.avail_flag.clear_calls == 0
    assert driver._groupSyncWriteTorqueEnableHandler.avail_flag.clear_calls == 0
    task = driver._comm_task_queue.get_nowait()
    task()
    assert driver._groupSyncWriteGoalPositionProfileHandler.tx_calls == [[(1, profile_payload(100, 16))]]


def test_position_commands_default_to_fifo_for_position_streams():
    driver = BaseMethodDriver()

    driver.set_position([1], [100], velocities_raw=[250], accelerations_raw=[0], currents_raw=[500])
    driver.set_position([1], [101], velocities_raw=[250], accelerations_raw=[0], currents_raw=[500])
    driver.set_position([1], [102], velocities_raw=[250], accelerations_raw=[0], currents_raw=[500])

    assert driver._comm_task_queue.qsize() == 3
    while not driver._comm_task_queue.empty():
        driver._comm_task_queue.get_nowait()()
    assert driver._groupSyncWriteGoalPositionProfileHandler.tx_calls == [
        [(1, profile_payload(100, 16))],
        [(1, profile_payload(101, 16))],
        [(1, profile_payload(102, 16))],
    ]


def test_position_transaction_failure_aborts_dependents_and_next_command_recovers():
    driver = BaseMethodDriver()
    driver._mode = {1: Mode.Torque}
    driver._groupSyncWriteModeHandler = FailOnceSyncWriteHandler()

    driver.set_position([1], [100], velocities_raw=[250], accelerations_raw=[0], currents_raw=[500])
    driver.set_position([1], [101], velocities_raw=[250], accelerations_raw=[0], currents_raw=[500])

    with pytest.raises(FeeTechCommandDispatchError, match="groupSyncWriteMode"):
        driver._comm_task_queue.get_nowait()()

    assert driver._groupSyncWriteTorqueEnableHandler.tx_calls == [[(1, [TorqueEnable.Disable.value])]]
    assert driver._groupSyncWriteGoalPositionProfileHandler.tx_calls == []
    assert driver._mode == {1: None}
    assert driver._torque_enable == {1: None}

    driver._comm_task_queue.get_nowait()()

    assert driver._groupSyncWriteTorqueEnableHandler.tx_calls == [
        [(1, [TorqueEnable.Disable.value])],
        [(1, [TorqueEnable.Disable.value])],
        [(1, [TorqueEnable.Enable.value])],
    ]
    assert driver._groupSyncWriteModeHandler.tx_calls == [
        [(1, [Mode.Position.value])],
        [(1, [Mode.Position.value])],
    ]
    assert driver._groupSyncWriteGoalPositionProfileHandler.tx_calls == [
        [(1, profile_payload(101, 16))]
    ]
    assert driver._mode == {1: Mode.Position}
    assert driver._torque_enable == {1: TorqueEnable.Enable}


def test_current_transaction_failure_aborts_dependents_and_next_command_recovers():
    driver = BaseMethodDriver()
    driver._groupSyncWriteModeHandler = FailOnceSyncWriteHandler()

    driver.set_current([1], [32])
    driver.set_current([1], [33])

    with pytest.raises(FeeTechCommandDispatchError, match="groupSyncWriteMode"):
        driver._comm_task_queue.get_nowait()()

    assert driver._groupSyncWriteTorqueEnableHandler.tx_calls == [[(1, [TorqueEnable.Disable.value])]]
    assert driver._groupSyncWriteGoalCurrentHandler.tx_calls == []
    assert driver._mode == {1: None}
    assert driver._torque_enable == {1: None}

    driver._comm_task_queue.get_nowait()()

    assert driver._groupSyncWriteTorqueEnableHandler.tx_calls == [
        [(1, [TorqueEnable.Disable.value])],
        [(1, [TorqueEnable.Disable.value])],
        [(1, [TorqueEnable.Enable.value])],
    ]
    assert driver._groupSyncWriteModeHandler.tx_calls == [
        [(1, [Mode.Torque.value])],
        [(1, [Mode.Torque.value])],
    ]
    assert driver._groupSyncWriteGoalCurrentHandler.tx_calls == [[(1, [33, 0])]]
    assert driver._mode == {1: Mode.Torque}
    assert driver._torque_enable == {1: TorqueEnable.Enable}


def test_direct_torque_enable_rejects_unknown_mode_without_writing():
    driver = BaseMethodDriver()
    driver._mode = {1: None}
    driver._torque_enable = {1: TorqueEnable.Disable}

    driver.set_torque_enable([1], [TorqueEnable.Enable])

    with pytest.raises(FeeTechCommandDispatchError, match="confirmed mode is unknown"):
        driver._comm_task_queue.get_nowait()()

    assert driver._groupSyncWriteTorqueEnableHandler.tx_calls == []
    assert driver._mode == {1: None}
    assert driver._torque_enable == {1: None}


def test_direct_torque_disable_is_allowed_with_unknown_mode():
    driver = BaseMethodDriver()
    driver._mode = {1: None}
    driver._torque_enable = {1: None}

    driver.set_torque_enable([1], [TorqueEnable.Disable])
    driver._comm_task_queue.get_nowait()()

    assert driver._groupSyncWriteTorqueEnableHandler.tx_calls == [
        [(1, [TorqueEnable.Disable.value])]
    ]
    assert driver._mode == {1: None}
    assert driver._torque_enable == {1: TorqueEnable.Disable}


def test_comm_worker_executes_one_write_task_after_each_read(monkeypatch):
    driver = BaseMethodDriver()
    executed = []
    for index in range(3):
        driver._comm_task_queue.put(lambda index=index: executed.append(index))

    driver._stop_flag = Event()
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.time.sleep",
        lambda _duration: driver._stop_flag.set(),
    )

    FeeTechDriver._comm_worker(driver)

    assert driver.read_calls == 1
    assert executed == [0]
    assert driver._comm_task_queue.qsize() == 2


def test_comm_worker_does_not_execute_queued_write_after_pause_request(monkeypatch):
    driver = BaseMethodDriver()
    executed = []
    driver._comm_task_queue.put(lambda: executed.append(True))
    driver._stop_flag = Event()

    def read_then_pause():
        driver.read_calls += 1
        driver._stop_flag.set()

    driver._read_state = read_then_pause
    monkeypatch.setattr("acetele.equipment.feetech.feetech_driver.time.sleep", lambda _duration: None)

    FeeTechDriver._comm_worker(driver)

    assert driver.read_calls == 1
    assert executed == []
    assert driver._comm_task_queue.qsize() == 1


def test_comm_worker_recovers_after_async_dispatch_failure(monkeypatch):
    driver = BaseMethodDriver()
    executed = []

    def fail_write():
        raise FeeTechCommandDispatchError("write failed")

    driver._comm_task_queue.put(fail_write)
    driver._comm_task_queue.put(lambda: executed.append(True))
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.time.sleep",
        lambda _duration: driver._stop_flag.set() if executed else None,
    )

    FeeTechDriver._comm_worker(driver)

    assert driver._mode == {1: None}
    assert driver._torque_enable == {1: None}
    assert executed == [True]
    assert driver._comm_fault is None


def test_comm_worker_latches_state_read_failure_and_rejects_stale_state():
    driver = BaseMethodDriver()
    read_error = RuntimeError("read failed")
    driver._read_state = lambda: (_ for _ in ()).throw(read_error)

    FeeTechDriver._comm_worker(driver)

    assert driver._comm_fault is read_error
    assert driver._position == {}
    assert driver._velocity == {}
    assert driver._current == {}
    with pytest.raises(FeeTechCommandDispatchError, match="worker is faulted") as exc_info:
        FeeTechDriver.get_state(driver)
    assert exc_info.value.__cause__ is read_error


def test_async_dispatch_rejects_full_queue_with_bounded_wait():
    driver = BaseMethodDriver()

    class FullQueue:
        def put(self, _task, timeout=None):
            assert timeout == pytest.approx(0.001)
            raise Full

    driver._comm_task_queue = FullQueue()

    with pytest.raises(FeeTechCommandDispatchError, match="queue is full"):
        driver._submit_comm_task(lambda: None, timeout=0.001, label="test write")


def test_driver_does_not_expose_misleading_diagnostics_api():
    assert not hasattr(FeeTechDriver, "get_diagnostics")
    assert not hasattr(FeeTechDriver, "_update_queue_high_watermark")


class TorqueEnableRecordingDriver(PositionOnlyDriver):
    def __init__(self):
        super().__init__()
        self.torque_enable_calls = []

    def set_torque_enable(self, ids, enables):
        self.torque_enable_calls.append((list(ids), list(enables)))


@pytest.mark.parametrize("enable", [TorqueEnable.Disable, TorqueEnable.Enable])
def test_arm_set_torque_enable_for_selected_ids(enable):
    arm = make_arm_for_encoding(ids=(0, 1, 4))
    driver = TorqueEnableRecordingDriver()
    arm._driver = driver

    arm.set_torque_enable(enable, ids=[0, 1])

    assert driver.torque_enable_calls == [([0, 1], [enable, enable])]


def test_set_position_with_raw_current_switches_from_torque_to_position_mode():
    driver = BaseMethodDriver()
    driver._mode = {1: Mode.Torque}

    driver.set_position([1], [100], velocities_raw=[250], accelerations_raw=[0], currents_raw=[33])
    driver._comm_task_queue.get_nowait()()

    assert driver._groupSyncWriteModeHandler.tx_calls == [[(1, [Mode.Position.value])]]
    assert driver._groupSyncWriteGoalPositionProfileHandler.tx_calls == [
        [(1, profile_payload(100, 16, current=33))]
    ]


@pytest.mark.parametrize(
    "invalid_currents",
    (
        [1.0],
        ["1"],
        [True],
        [np.nan],
        [np.inf],
        np.array([[1]], dtype=int),
        [FEETECH_SIGNED_15_BIT_MAX + 1],
        [-FEETECH_SIGNED_15_BIT_MAX - 1],
    ),
)
def test_set_current_rejects_unencodable_values_before_dispatch(invalid_currents):
    driver = BaseMethodDriver()
    initial_mode = dict(driver._mode)
    initial_torque_enable = dict(driver._torque_enable)

    with pytest.raises(ValueError, match="goal currents"):
        driver.set_current([1], invalid_currents)

    assert driver._comm_task_queue.empty()
    assert driver._groupSyncWriteGoalCurrentHandler.tx_calls == []
    assert driver._mode == initial_mode
    assert driver._torque_enable == initial_torque_enable


@pytest.mark.parametrize(
    "current",
    (
        -FEETECH_SIGNED_15_BIT_MAX,
        0,
        FEETECH_SIGNED_15_BIT_MAX,
        np.int32(1),
    ),
)
def test_set_current_accepts_signed_15_bit_boundaries(current):
    driver = BaseMethodDriver()

    driver.set_current([1], [current])

    assert driver._comm_task_queue.qsize() == 1


def test_set_current_switches_to_torque_mode():
    driver = BaseMethodDriver()

    driver.set_current([1], [33])
    driver._comm_task_queue.get_nowait()()

    assert driver._groupSyncWriteModeHandler.tx_calls == [[(1, [Mode.Torque.value])]]
    assert driver._groupSyncWriteGoalCurrentHandler.tx_calls == [[(1, [33, 0])]]


def test_get_state_times_out_with_missing_ids(monkeypatch):
    driver = PositionOnlyDriver()
    driver._ids = [1, 2]
    driver._position = {1: 100}
    driver._velocity = {1: 0}
    driver._current = {1: 0}
    driver._lock = Lock()
    monkeypatch.setattr("acetele.equipment.feetech.feetech_driver.time.sleep", lambda _duration: None)

    with pytest.raises(FeeTechStateTimeoutError) as exc_info:
        FeeTechDriver.get_state(driver, timeout=0.0)

    assert "missing IDs: [2]" in str(exc_info.value)


def test_get_state_returns_one_atomic_copy_when_worker_updates_after_snapshot():
    driver = BaseMethodDriver()

    class UpdatingThread:
        def is_alive(self):
            with driver._lock:
                driver._position = {1: 6000}
                driver._velocity = {1: 1}
                driver._current = {1: 2}
            return True

    driver._comm_thread = UpdatingThread()

    position, velocity, current = FeeTechDriver.get_state(driver)

    assert position == {1: 5000}
    assert velocity == {1: 0}
    assert current == {1: 0}
    position[1] = -1
    velocity[1] = -1
    current[1] = -1
    assert driver._position == {1: 6000}
    assert driver._velocity == {1: 1}
    assert driver._current == {1: 2}


def test_get_state_sample_returns_atomic_metadata_and_independent_dicts():
    driver = BaseMethodDriver()

    sample = FeeTechDriver.get_state_sample(driver)

    assert sample.timestamp == pytest.approx(1.25)
    assert sample.sequence == 4
    assert sample.position == {1: 5000}
    sample.position[1] = -1
    assert driver._position == {1: 5000}


def test_read_state_stamps_and_sequences_complete_samples(monkeypatch):
    from acetele.equipment.feetech.feetech_sdk.protocol_packet_handler import (
        protocol_packet_handler,
    )

    driver = BaseMethodDriver()

    class ReadHandler:
        def addParam(self, _ft_id):
            return True

        def txRxPacket(self):
            return COMM_SUCCESS

        def isAvailable(self, _ft_id, _address, _length):
            return True, 0

        def getData(self, _ft_id, address, _length):
            values = {
                56: 0x8002,
                58: 0x8001,
                69: 45,
            }
            return values[address]

        def clearParam(self):
            pass

    driver._groupSyncReadHandler = ReadHandler()
    driver._packetHandler = protocol_packet_handler(None, 0)
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.time.monotonic",
        lambda: 9.5,
    )

    FeeTechDriver._read_state(driver)

    assert driver._position == {1: -2}
    assert driver._velocity == {1: -1}
    assert driver._current == {1: 45}
    assert driver._state_timestamp == pytest.approx(9.5)
    assert driver._state_sequence == 5


def test_hls_state_read_length_covers_position_to_current():
    import acetele.equipment.feetech.feetech_driver as driver_module

    assert driver_module.HLS_STATE_READ_LENGTH == (
        driver_module.HLS_PRESENT_CURRENT_L + 2 - driver_module.HLS_PRESENT_POSITION_L
    )


class CurrentRecordingDriver:
    def __init__(self):
        self.current_calls = []

    def set_current(self, ids, currents):
        self.current_calls.append((np.array(ids), np.array(currents)))


def make_arm_for_encoding(ids=(0, 1, 2, 3)):
    arm = FeeTechArm.__new__(FeeTechArm)
    arm._ids = np.array(ids)
    arm._joint_names = tuple(f"joint_{joint_id + 1}" for joint_id in ids)
    arm._dof = len(ids)
    arm._signs = np.ones(len(ids))
    arm._home_poses = np.zeros(len(ids))
    arm._servo_models = np.array(["HL3915"] * len(ids))
    arm._profile_acceleration_defaults = np.zeros(len(ids), dtype=int)
    arm._profile_current_defaults = np.full(len(ids), 500, dtype=int)
    arm._profile_velocity_defaults = np.full(len(ids), 250, dtype=int)
    arm._torque_current_mapping = np.ones(len(ids)) * (1000.0 / 9.3)
    arm._no_load_current = np.ones(len(ids)) * 260
    arm._pin_model = None
    arm._pin_data = None
    arm._enable_gravity_compensation = False
    arm._adaptive_compensation_enabled = False
    arm._state_estimator = FeeTechStateEstimator(len(ids))
    arm._fallback_state_sequence = 0
    return arm


def test_set_torque_sends_all_selected_arm_ids():
    arm = make_arm_for_encoding()
    driver = CurrentRecordingDriver()
    arm._driver = driver

    arm.set_torque([0.1, 0.2, 0.3, 0.4])

    sent_ids, sent_currents = driver.current_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([0, 1, 2, 3]))
    assert len(sent_currents) == 4


def test_set_torque_accepts_list_ids():
    arm = make_arm_for_encoding()
    driver = CurrentRecordingDriver()
    arm._driver = driver

    arm.set_torque([0.1, 0.5], ids=[0, 3])

    sent_ids, sent_currents = driver.current_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([0, 3]))
    assert len(sent_currents) == 2


@pytest.mark.parametrize("torque", (np.nan, np.inf, -np.inf))
def test_set_torque_rejects_non_finite_values_before_dispatch(torque):
    arm = make_arm_for_encoding(ids=(0,))
    driver = CurrentRecordingDriver()
    arm._driver = driver

    with pytest.raises(ValueError, match="finite one-dimensional"):
        arm.set_torque([torque])

    assert driver.current_calls == []


@pytest.mark.parametrize("torque", (1e9, -1e9))
def test_set_torque_rejects_unencodable_current_before_dispatch(torque):
    arm = make_arm_for_encoding(ids=(0,))
    driver = CurrentRecordingDriver()
    arm._driver = driver

    with pytest.raises(ValueError, match="encoded torque current"):
        arm.set_torque([torque])

    assert driver.current_calls == []


def test_set_torque_accepts_signed_15_bit_current_boundary():
    arm = make_arm_for_encoding(ids=(0,))
    driver = CurrentRecordingDriver()
    arm._driver = driver
    torque_at_current_limit = (
        (FEETECH_SIGNED_15_BIT_MAX * 6.5 - arm._no_load_current[0])
        / arm._torque_current_mapping[0]
        * 0.0981
    )

    arm.set_torque([torque_at_current_limit])

    _, currents = driver.current_calls[0]
    assert abs(int(currents[0])) == FEETECH_SIGNED_15_BIT_MAX


class PositionRecordingDriver:
    def __init__(self, state):
        self.state = state
        self.position_calls = []
        self.profile_calls = []

    def get_state(self):
        return self.state

    def set_position(self, ids, positions, **kwargs):
        call = (np.array(ids), np.array(positions), kwargs)
        if "velocities_raw" in kwargs or "accelerations_raw" in kwargs or "currents_raw" in kwargs:
            self.profile_calls.append(call)
        else:
            self.position_calls.append(call)


@pytest.mark.parametrize("invalid_ids", ([0.0], ["0"], [True]))
def test_arm_rejects_invalid_joint_ids_before_writing(invalid_ids):
    arm = make_arm_for_encoding(ids=(0,))
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    arm._driver = driver

    with pytest.raises(ValueError, match="arm joint ids"):
        arm.set_position([0.5], ids=invalid_ids)

    assert driver.profile_calls == []


class ImmediateProfileDriver(PositionOnlyDriver):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self._ids = tuple(state[0])
        self._mode = {ft_id: Mode.Position for ft_id in self._ids}
        self._torque_enable = {
            ft_id: TorqueEnable.Enable for ft_id in self._ids
        }

    def get_state(self):
        return self.state


class CloseRecordingDriver:
    def __init__(self):
        self.torque_enable_calls = []
        self.close_calls = 0
        self.events = []

    def set_torque_enable(self, ids, enables, **kwargs):
        self.events.append("torque_disable")
        self.torque_enable_calls.append((list(ids), list(enables), kwargs))

    def close(self):
        self.events.append("close")
        self.close_calls += 1


def minimal_arm_config():
    return ArmConfig(
        port="/dev/test",
        joint_ids=(0, 1),
        joint_names=("joint_1", "joint_2"),
        joint_signs=(1, 1),
        home_poses=(0.0, 0.0),
        servo_models=("HL3915", "HL3915"),
    )


def single_joint_arm_config(**overrides):
    config = ArmConfig(
        port="/dev/test",
        joint_ids=(0,),
        joint_names=("joint_1",),
        joint_signs=(1,),
        home_poses=(0.0,),
        servo_models=("HL3915",),
    )
    return replace(config, **overrides)


SINGLE_JOINT_POSITION_LIMITS = ([-1.0], [1.0])


class ManualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, duration):
        self.now += duration


def test_arm_close_does_not_close_external_driver():
    driver = CloseRecordingDriver()
    arm = FeeTechArm(minimal_arm_config(), driver=driver)

    arm.close()

    assert driver.torque_enable_calls == [
        (
            [0, 1],
            [TorqueEnable.Disable, TorqueEnable.Disable],
            {"force": True, "wait": True},
        )
    ]
    assert driver.events == ["torque_disable"]
    assert driver.close_calls == 0


def test_arm_close_closes_self_created_driver(monkeypatch):
    created = []

    class CapturingFeeTechDriver(CloseRecordingDriver):
        def __init__(self, ids, port):
            super().__init__()
            self.ids = list(ids)
            self.port = port
            created.append(self)

    monkeypatch.setattr("acetele.equipment.feetech.arm.FeeTechDriver", CapturingFeeTechDriver)
    arm = FeeTechArm(minimal_arm_config())

    arm.close()

    assert [(driver.ids, driver.port) for driver in created] == [([0, 1], "/dev/test")]
    assert created[0].torque_enable_calls == [
        (
            [0, 1],
            [TorqueEnable.Disable, TorqueEnable.Disable],
            {"force": True, "wait": True},
        )
    ]
    assert created[0].events == ["torque_disable", "close"]
    assert created[0].close_calls == 1


def test_arm_close_releases_self_created_driver_when_torque_disable_fails(monkeypatch):
    created = []

    class FailingFeeTechDriver(CloseRecordingDriver):
        def __init__(self, ids, port):
            super().__init__()
            del ids, port
            created.append(self)

        def set_torque_enable(self, ids, enables, **kwargs):
            del ids, enables, kwargs
            self.events.append("torque_disable")
            raise FeeTechCommandDispatchError("write failed")

    monkeypatch.setattr("acetele.equipment.feetech.arm.FeeTechDriver", FailingFeeTechDriver)
    arm = FeeTechArm(minimal_arm_config())

    with pytest.raises(FeeTechCommandDispatchError, match="write failed"):
        arm.close()

    assert created[0].events == ["torque_disable", "close"]
    assert created[0].close_calls == 1


def test_arm_close_preserves_disable_error_when_driver_close_also_fails(
    monkeypatch,
):
    disable_error = FeeTechCommandDispatchError("disable failed")
    close_error = RuntimeError("close failed")

    class FailingFeeTechDriver(CloseRecordingDriver):
        def __init__(self, ids, port):
            del ids, port
            super().__init__()

        def set_torque_enable(self, ids, enables, **kwargs):
            del ids, enables, kwargs
            raise disable_error

        def close(self):
            raise close_error

    monkeypatch.setattr(
        "acetele.equipment.feetech.arm.FeeTechDriver",
        FailingFeeTechDriver,
    )
    arm = FeeTechArm(minimal_arm_config())

    with pytest.raises(FeeTechCommandDispatchError) as exc_info:
        arm.close()

    assert exc_info.value is disable_error
    assert exc_info.value.__cause__ is close_error


def test_arm_requires_servo_models():
    config = replace(minimal_arm_config(), servo_models=())
    driver = PositionRecordingDriver(({0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}))

    with pytest.raises(ValueError, match="servo_models must be specified"):
        FeeTechArm(config, driver=driver)


def test_arm_rejects_invalid_servo_models():
    driver = PositionRecordingDriver(({0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}))

    with pytest.raises(ValueError, match="servo_models must match joint_ids"):
        ArmConfig(
            port="/dev/test",
            joint_ids=(0, 1, 2),
            joint_names=("joint_1", "joint_2", "joint_3"),
            joint_signs=(1, 1, 1),
            home_poses=(0.0, 0.0, 0.0),
            servo_models=("HL3915", "HL3915"),
        )

    config = ArmConfig(
        port="/dev/test",
        joint_ids=(0, 1, 2),
        joint_names=("joint_1", "joint_2", "joint_3"),
        joint_signs=(1, 1, 1),
        home_poses=(0.0, 0.0, 0.0),
        servo_models=("HL3915", "BAD", "HL3915"),
    )
    with pytest.raises(ValueError, match="unsupported servo models"):
        FeeTechArm(config, driver=driver)


def test_arm_constructs_feetech_driver_with_model_profiles(monkeypatch):
    captured = {}

    class CapturingFeeTechDriver:
        def __init__(self, ids, port):
            captured["ids"] = np.array(ids)
            captured["port"] = port

    monkeypatch.setattr("acetele.equipment.feetech.arm.FeeTechDriver", CapturingFeeTechDriver)
    config = ArmConfig(
        port="/dev/test",
        joint_ids=(0, 1, 2),
        joint_names=("joint_1", "joint_2", "joint_3"),
        joint_signs=(1, 1, 1),
        home_poses=(0.0, 0.0, 0.0),
        servo_models=("HL3950", "HL3930", "HL3915"),
    )

    arm = FeeTechArm(config)

    assert isinstance(arm._driver, CapturingFeeTechDriver)
    np.testing.assert_array_equal(captured["ids"], np.array([0, 1, 2]))
    assert captured["port"] == "/dev/test"


def test_arm_plain_position_fills_model_profile_defaults():
    driver = PositionRecordingDriver(({0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}))
    config = ArmConfig(
        port="/dev/test",
        joint_ids=(0, 1, 2),
        joint_names=("joint_1", "joint_2", "joint_3"),
        joint_signs=(1, 1, 1),
        home_poses=(0.0, 0.0, 0.0),
        servo_models=("HL3950", "HL3930", "HL3915"),
    )
    arm = FeeTechArm(config, driver=driver)

    arm.set_position([0.0, 0.0, 0.0])

    _, _, kwargs = driver.profile_calls[0]
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.array([110, 100, 250]))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.array([0, 250, 0]))
    np.testing.assert_array_equal(kwargs["currents_raw"], np.array([1000, 1000, 500]))


def test_arm_metric_profile_reaches_hls_payload():
    driver = ImmediateProfileDriver(({0: 5000}, {0: 0}, {0: 0}))
    config = single_joint_arm_config()
    arm = FeeTechArm(config, driver=driver)

    arm.set_position(
        [100.0 * np.pi / 2048.0],
        velocities=26 * PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
        accelerations=20 * PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2,
        torque=0.1,
    )

    expected_current = round(((1000.0 / 9.3) * (0.1 / 0.0981) + 260) / 6.5)
    assert driver._groupSyncWriteGoalPositionProfileHandler.params == [
        (0, profile_payload(100, 16, acceleration=20, current=expected_current, velocity=26))
    ]


def test_adaptive_compensation_disabled_keeps_encoded_target_unchanged():
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    arm = FeeTechArm(single_joint_arm_config(), driver=driver)

    arm.set_position([0.1])

    _, positions, _ = driver.profile_calls[0]
    assert positions[0] == pytest.approx(65, abs=1)
    np.testing.assert_allclose(arm.get_adaptive_compensation_state()["offset_rad"], np.array([0.0]))


def test_position_limits_apply_when_adaptive_compensation_is_disabled():
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    arm = FeeTechArm(
        single_joint_arm_config(),
        driver=driver,
        position_limits=([-0.5], [0.5]),
    )

    arm.set_position([0.8])

    _, positions, _ = driver.profile_calls[-1]
    assert positions[0] == pytest.approx(round(0.5 * 2048.0 / np.pi), abs=1)
    assert arm.get_adaptive_compensation_state()["enabled"] is False


def test_adaptive_compensation_enabled_requires_position_limits():
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    config = single_joint_arm_config(enable_adaptive_compensation=True)

    with pytest.raises(ValueError, match="position_limits"):
        FeeTechArm(config, driver=driver)


def test_adaptive_compensation_enabled_does_not_require_pin_model():
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    config = single_joint_arm_config(enable_adaptive_compensation=True)

    arm = FeeTechArm(config, driver=driver, position_limits=SINGLE_JOINT_POSITION_LIMITS)

    assert arm.get_adaptive_compensation_state()["enabled"] is True


def test_adaptive_compensation_waits_for_stable_target(monkeypatch):
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    config = single_joint_arm_config(enable_adaptive_compensation=True)
    arm = FeeTechArm(config, driver=driver, position_limits=SINGLE_JOINT_POSITION_LIMITS)
    clock = ManualClock()
    monkeypatch.setattr("acetele.equipment.feetech.arm.time.monotonic", clock)

    for _ in range(4):
        arm.set_position([0.1])
        clock.advance(0.05)

    state = arm.get_adaptive_compensation_state()
    assert state["active"].tolist() == [False]
    np.testing.assert_allclose(state["estimate_rad"], np.array([0.0]))

    arm.set_position([0.1])

    state = arm.get_adaptive_compensation_state()
    assert state["active"].tolist() == [True]
    assert state["estimate_rad"][0] > 0.0


def test_adaptive_compensation_does_not_learn_while_target_moves(monkeypatch):
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    config = single_joint_arm_config(enable_adaptive_compensation=True)
    arm = FeeTechArm(config, driver=driver, position_limits=SINGLE_JOINT_POSITION_LIMITS)
    clock = ManualClock()
    monkeypatch.setattr("acetele.equipment.feetech.arm.time.monotonic", clock)

    for target in np.arange(0.10, 0.22, 0.01):
        arm.set_position([target])
        clock.advance(0.05)

    state = arm.get_adaptive_compensation_state()
    assert state["active"].tolist() == [False]
    np.testing.assert_allclose(state["estimate_rad"], np.array([0.0]))


def test_adaptive_compensation_does_not_learn_above_velocity_threshold(monkeypatch):
    driver = PositionRecordingDriver(({0: 0}, {0: 1}, {0: 0}))
    config = single_joint_arm_config(enable_adaptive_compensation=True)
    arm = FeeTechArm(config, driver=driver, position_limits=SINGLE_JOINT_POSITION_LIMITS)
    clock = ManualClock()
    monkeypatch.setattr("acetele.equipment.feetech.arm.time.monotonic", clock)
    monkeypatch.setattr(
        "acetele.equipment.feetech.state_estimator.time.monotonic",
        clock,
    )

    for sample_index in range(8):
        driver.state = (
            {0: sample_index * 3},
            {0: 1},
            {0: 0},
        )
        arm.set_position([0.1])
        clock.advance(0.05)

    state = arm.get_adaptive_compensation_state()
    assert state["active"].tolist() == [False]
    np.testing.assert_allclose(state["estimate_rad"], np.array([0.0]))


@pytest.mark.parametrize(
    ("measured_position", "desired", "expected_sign"),
    [(0.0, 0.1, 1), (0.1, 0.0, -1)],
)
def test_adaptive_compensation_learns_static_error_direction(
    monkeypatch,
    measured_position,
    desired,
    expected_sign,
):
    measured_counts = round(measured_position * 2048.0 / np.pi)
    driver = PositionRecordingDriver(({0: measured_counts}, {0: 0}, {0: 0}))
    config = single_joint_arm_config(enable_adaptive_compensation=True)
    arm = FeeTechArm(config, driver=driver, position_limits=SINGLE_JOINT_POSITION_LIMITS)
    clock = ManualClock()
    monkeypatch.setattr("acetele.equipment.feetech.arm.time.monotonic", clock)

    for _ in range(5):
        arm.set_position([desired])
        clock.advance(0.05)

    state = arm.get_adaptive_compensation_state()
    assert np.sign(state["estimate_rad"][0]) == expected_sign
    assert np.sign(state["offset_rad"][0]) == expected_sign


def test_adaptive_compensation_holds_estimate_inside_deadband(monkeypatch):
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    config = single_joint_arm_config(enable_adaptive_compensation=True)
    arm = FeeTechArm(config, driver=driver, position_limits=SINGLE_JOINT_POSITION_LIMITS)
    clock = ManualClock()
    monkeypatch.setattr("acetele.equipment.feetech.arm.time.monotonic", clock)

    for _ in range(5):
        arm.set_position([0.1])
        clock.advance(0.05)
    before = arm.get_adaptive_compensation_state()
    measured_counts = round(0.1 * 2048.0 / np.pi)
    driver.state = ({0: measured_counts}, {0: 0}, {0: 0})

    arm.set_position([0.1])

    transitioning = arm.get_adaptive_compensation_state()
    assert transitioning["estimate_rad"][0] == pytest.approx(before["estimate_rad"][0])
    assert np.isnan(transitioning["last_error_rad"][0])

    for _ in range(2):
        clock.advance(0.01)
        arm.set_position([0.1])

    settled = arm.get_adaptive_compensation_state()
    assert settled["estimate_rad"][0] == pytest.approx(before["estimate_rad"][0])
    assert settled["offset_rad"][0] >= before["offset_rad"][0]
    assert abs(settled["last_error_rad"][0]) <= 0.02


def test_adaptive_compensation_filters_offset_when_estimate_saturates(monkeypatch):
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    config = single_joint_arm_config(enable_adaptive_compensation=True)
    arm = FeeTechArm(config, driver=driver, position_limits=SINGLE_JOINT_POSITION_LIMITS)
    clock = ManualClock()
    monkeypatch.setattr("acetele.equipment.feetech.arm.time.monotonic", clock)

    for _ in range(5):
        arm.set_position([0.8])
        clock.advance(0.05)

    state = arm.get_adaptive_compensation_state()
    assert state["estimate_rad"][0] == pytest.approx(0.10)
    assert 0.0 < state["offset_rad"][0] < state["estimate_rad"][0]
    assert state["last_limited"].tolist() == [True]


def test_adaptive_compensation_resets_estimate_and_filters_offset_on_target_change(monkeypatch):
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    config = single_joint_arm_config(enable_adaptive_compensation=True)
    arm = FeeTechArm(config, driver=driver, position_limits=SINGLE_JOINT_POSITION_LIMITS)
    clock = ManualClock()
    monkeypatch.setattr("acetele.equipment.feetech.arm.time.monotonic", clock)

    for _ in range(5):
        arm.set_position([0.1])
        clock.advance(0.05)
    before_offset = arm.get_adaptive_compensation_state()["offset_rad"][0]

    arm.set_position([0.2])

    state = arm.get_adaptive_compensation_state()
    np.testing.assert_allclose(state["estimate_rad"], np.array([0.0]))
    assert 0.0 < state["offset_rad"][0] < before_offset
    assert state["last_reset"].tolist() == [True]


def test_adaptive_compensation_resets_on_target_direction_reversal(monkeypatch):
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    config = single_joint_arm_config(enable_adaptive_compensation=True)
    arm = FeeTechArm(config, driver=driver, position_limits=SINGLE_JOINT_POSITION_LIMITS)
    clock = ManualClock()
    monkeypatch.setattr("acetele.equipment.feetech.arm.time.monotonic", clock)

    for _ in range(5):
        arm.set_position([0.1])
        clock.advance(0.05)
    arm.set_position([0.11])
    clock.advance(0.05)
    arm.set_position([0.10])

    state = arm.get_adaptive_compensation_state()

    np.testing.assert_allclose(state["estimate_rad"], np.array([0.0]))
    assert state["last_reset"].tolist() == [True]


def test_adaptive_compensation_rejects_non_finite_target():
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    arm = FeeTechArm(
        single_joint_arm_config(enable_adaptive_compensation=True),
        driver=driver,
        position_limits=SINGLE_JOINT_POSITION_LIMITS,
    )

    with pytest.raises(ValueError, match="finite"):
        arm.set_position([np.nan])


def test_adaptive_compensation_freezes_on_non_finite_feedback():
    driver = PositionRecordingDriver(({0: np.nan}, {0: 0}, {0: 0}))
    arm = FeeTechArm(
        single_joint_arm_config(enable_adaptive_compensation=True),
        driver=driver,
        position_limits=SINGLE_JOINT_POSITION_LIMITS,
    )

    arm.set_position([0.1])

    _, positions, _ = driver.profile_calls[-1]
    assert positions[0] == pytest.approx(round(0.1 * 2048.0 / np.pi), abs=1)
    state = arm.get_adaptive_compensation_state()
    assert state["active"].tolist() == [False]
    assert np.isnan(state["last_error_rad"][0])


def test_adaptive_compensation_clamps_target_to_joint_limits():
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    arm = FeeTechArm(
        single_joint_arm_config(enable_adaptive_compensation=True),
        driver=driver,
        position_limits=([-0.5], [0.5]),
    )

    arm.set_position([0.8])

    _, positions, _ = driver.profile_calls[-1]
    assert positions[0] == pytest.approx(round(0.5 * 2048.0 / np.pi), abs=1)
    state = arm.get_adaptive_compensation_state()
    assert state["compensated_target_rad"][0] == pytest.approx(0.5)
    assert state["command_limited"].tolist() == [True]


def test_adaptive_compensation_prevents_windup_at_upper_joint_limit(monkeypatch):
    measured_counts = round(0.4 * 2048.0 / np.pi)
    driver = PositionRecordingDriver(({0: measured_counts}, {0: 0}, {0: 0}))
    arm = FeeTechArm(
        single_joint_arm_config(enable_adaptive_compensation=True),
        driver=driver,
        position_limits=([-0.5], [0.5]),
    )
    clock = ManualClock()
    monkeypatch.setattr("acetele.equipment.feetech.arm.time.monotonic", clock)

    for _ in range(8):
        arm.set_position([0.5])
        clock.advance(0.05)

    state = arm.get_adaptive_compensation_state()
    assert state["estimate_rad"][0] == pytest.approx(0.0)
    assert state["offset_rad"][0] == pytest.approx(0.0)
    assert state["last_limited"].tolist() == [True]


def test_adaptive_compensation_uses_limit_for_selected_joint():
    driver = PositionRecordingDriver(({0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}))
    config = replace(minimal_arm_config(), enable_adaptive_compensation=True)
    arm = FeeTechArm(config, driver=driver, position_limits=([-1.0, -0.5], [1.0, 0.5]))

    arm.set_position([0.8], ids=[1])

    sent_ids, positions, _ = driver.profile_calls[-1]
    np.testing.assert_array_equal(sent_ids, np.array([1]))
    assert positions[0] == pytest.approx(round(0.5 * 2048.0 / np.pi), abs=1)


def test_adaptive_compensation_unwraps_public_target_before_applying_urdf_limits():
    measured = np.pi - 0.02
    measured_counts = round(measured * 2048.0 / np.pi)
    driver = PositionRecordingDriver(({0: measured_counts}, {0: 0}, {0: 0}))
    arm = FeeTechArm(
        single_joint_arm_config(enable_adaptive_compensation=True),
        driver=driver,
        position_limits=([0.0], [3.4907]),
    )

    arm.set_position([-np.pi + 0.01])

    _, positions, _ = driver.profile_calls[-1]
    sent_position = positions[0] * np.pi / 2048.0
    assert np.pi < sent_position < 3.4907


def test_adaptive_compensation_applies_limits_in_urdf_coordinate_after_multiple_turns():
    driver = PositionRecordingDriver(({0: 6144}, {0: 0}, {0: 0}))
    arm = FeeTechArm(
        single_joint_arm_config(enable_adaptive_compensation=True),
        driver=driver,
        position_limits=([-np.pi], [np.pi]),
    )

    arm.set_position([-2.8])

    _, positions, _ = driver.profile_calls[-1]
    sent_position = positions[0] * np.pi / 2048.0
    assert sent_position == pytest.approx(-2.8, abs=np.pi / 2048.0)


def test_adaptive_compensation_does_not_modify_input_and_state_returns_copies():
    driver = PositionRecordingDriver(({0: 0}, {0: 0}, {0: 0}))
    arm = FeeTechArm(
        single_joint_arm_config(enable_adaptive_compensation=True),
        driver=driver,
        position_limits=SINGLE_JOINT_POSITION_LIMITS,
    )
    positions = np.array([0.1])

    arm.set_position(positions)
    state = arm.get_adaptive_compensation_state()
    state["offset_rad"][0] = 99.0

    np.testing.assert_allclose(positions, np.array([0.1]))
    assert arm.get_adaptive_compensation_state()["offset_rad"][0] != 99.0
    assert "reference_rad" not in state
    assert {
        "active",
        "stable_target_rad",
        "compensated_target_rad",
        "command_limited",
    }.issubset(state)


def test_adaptive_compensation_reduces_quasi_static_sag_in_closed_loop(monkeypatch):
    class StaticSagDriver(PositionRecordingDriver):
        def __init__(self, sag, initial_position):
            self.sag = sag
            initial_counts = round(initial_position * 2048.0 / np.pi)
            super().__init__(({0: initial_counts}, {0: 0}, {0: 0}))

        def set_position(self, ids, positions, **kwargs):
            super().set_position(ids, positions, **kwargs)
            command = float(positions[0]) * np.pi / 2048.0
            measured = command - self.sag
            measured_counts = round(measured * 2048.0 / np.pi)
            self.state = ({0: measured_counts}, {0: 0}, {0: 0})

    desired = 0.2
    driver = StaticSagDriver(sag=0.05, initial_position=desired - 0.05)
    arm = FeeTechArm(
        single_joint_arm_config(enable_adaptive_compensation=True),
        driver=driver,
        position_limits=SINGLE_JOINT_POSITION_LIMITS,
    )
    clock = ManualClock()
    monkeypatch.setattr("acetele.equipment.feetech.arm.time.monotonic", clock)

    for _ in range(400):
        arm.set_position([desired])
        clock.advance(0.01)

    measured_counts = driver.state[0][0]
    measured = measured_counts * np.pi / 2048.0
    error = float(wrap_to_pi(np.array([desired - measured]))[0])

    assert abs(error) <= 0.02 + np.pi / 2048.0
    assert 0.0 < arm.get_adaptive_compensation_state()["offset_rad"][0] <= 0.10


def test_act_without_gripper_keeps_last_joint_in_radians():
    driver = PositionRecordingDriver(
        (
            {0: 0, 1: 1024},
            {0: 0, 1: 0},
            {0: 0, 1: 0},
        )
    )
    arm = make_arm_for_encoding(ids=(0, 1))
    arm._driver = driver

    positions, _, _ = arm.act()

    np.testing.assert_allclose(positions, np.array([0.0, np.pi / 2]))


def test_arm_public_methods_do_not_expose_encode_gripper():
    assert "encode_gripper" not in inspect.signature(FeeTechArm.act).parameters
    assert "encode_gripper" not in inspect.signature(FeeTechArm.get_state).parameters
    assert "encode_gripper" not in inspect.signature(FeeTechArm.set_position).parameters


def test_set_position_does_not_mutate_numpy_positions():
    driver = PositionRecordingDriver(
        (
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        )
    )
    arm = make_arm_for_encoding(ids=(0, 1, 2, 3))
    arm._driver = driver
    positions = np.array([0.0, 0.0, 0.0, 0.0])

    arm.set_position(positions)

    np.testing.assert_array_equal(positions, np.array([0.0, 0.0, 0.0, 0.0]))


def test_arm_set_position_profile_converts_only_explicit_metric_velocity():
    driver = PositionRecordingDriver(
        (
            {0: 0, 1: 0, 4: 0},
            {0: 0, 1: 0, 4: 0},
            {0: 0, 1: 0, 4: 0},
        )
    )
    arm = make_arm_for_encoding(ids=(0, 1, 2))
    arm._driver = driver

    arm.set_position([0.1, 0.0, 0.5], velocities=26 * PROFILE_VELOCITY_UNIT_RAD_PER_SEC)

    assert driver.position_calls == []
    sent_ids, positions, kwargs = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([0, 1, 2]))
    assert positions[0] == pytest.approx(65, abs=1)
    assert positions[2] == pytest.approx(326, abs=1)
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.array([26, 26, 26]))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.array([0, 0, 0]))
    np.testing.assert_array_equal(kwargs["currents_raw"], np.array([500, 500, 500]))
    assert "profile" not in kwargs


def test_arm_set_position_torque_overrides_profile_current():
    driver = PositionRecordingDriver(
        (
            {0: 0},
            {0: 0},
            {0: 0},
        )
    )
    arm = make_arm_for_encoding(ids=(0,))
    arm._driver = driver

    arm.set_position([0.1], velocities=26 * PROFILE_VELOCITY_UNIT_RAD_PER_SEC, torque=[0.1])

    _, _, kwargs = driver.profile_calls[0]
    expected_current = round(((1000.0 / 9.3) * (0.1 / 0.0981) + 260) / 6.5)
    assert kwargs["currents_raw"][0] == pytest.approx(expected_current)


def test_arm_set_position_torque_without_velocity_uses_profile_defaults():
    driver = PositionRecordingDriver(
        (
            {0: 0},
            {0: 0},
            {0: 0},
        )
    )
    arm = make_arm_for_encoding(ids=(0,))
    arm._driver = driver

    arm.set_position([0.1], torque=[0.1])

    assert driver.position_calls == []
    sent_ids, positions, kwargs = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([0]))
    assert positions[0] == pytest.approx(65, abs=1)
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.array([250]))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.array([0]))
    expected_current = round(((1000.0 / 9.3) * (0.1 / 0.0981) + 260) / 6.5)
    assert kwargs["currents_raw"][0] == pytest.approx(expected_current)


def test_arm_set_position_broadcasts_metric_profile_scalars_to_raw_arrays():
    driver = PositionRecordingDriver(
        (
            {0: 0, 1: 0},
            {0: 0, 1: 0},
            {0: 0, 1: 0},
        )
    )
    arm = make_arm_for_encoding(ids=(0, 1))
    arm._driver = driver

    arm.set_position(
        [0.1, 0.2],
        velocities=26 * PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
        accelerations=20 * PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2,
        torque=0.1,
    )

    _, _, kwargs = driver.profile_calls[0]
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.array([26, 26]))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.array([20, 20]))
    expected_current = round(((1000.0 / 9.3) * (0.1 / 0.0981) + 260) / 6.5)
    np.testing.assert_array_equal(kwargs["currents_raw"], np.array([expected_current, expected_current]))


def test_act_signed_torque_round_trips_set_torque_direction():
    arm = make_arm_for_encoding(ids=(0, 1))
    arm._signs = np.array([1, -1])
    current_driver = CurrentRecordingDriver()
    arm._driver = current_driver

    commanded_torque = np.array([0.1, -0.2])
    arm.set_torque(commanded_torque)
    sent_ids, sent_currents = current_driver.current_calls[0]

    state_driver = PositionRecordingDriver(
        (
            {int(ft_id): 0 for ft_id in sent_ids},
            {int(ft_id): 0 for ft_id in sent_ids},
            {int(ft_id): int(current) for ft_id, current in zip(sent_ids, sent_currents)},
        )
    )
    arm._driver = state_driver

    _, _, default_effort = arm.act()
    signed_effort = arm.get_state().motor_torque_signed

    assert np.all(default_effort >= 0.0)
    assert signed_effort[0] > 0.0
    assert signed_effort[1] < 0.0
    np.testing.assert_allclose(signed_effort, commanded_torque, atol=0.004)


def test_arm_state_keeps_raw_joint_angles_for_dynamics():
    driver = PositionRecordingDriver(
        (
            {0: 0, 4: 6144},
            {0: 0, 4: 0},
            {0: 0, 4: 0},
        )
    )
    arm = make_arm_for_encoding(ids=(0, 4))
    arm._driver = driver

    state = arm.get_state()
    positions, _, _ = arm.act()

    assert state.public_positions[0] == pytest.approx(0.0)
    assert state.public_positions[1] == pytest.approx(-np.pi)
    assert state.raw_positions[1] == pytest.approx(3 * np.pi)
    np.testing.assert_allclose(positions, state.public_positions)


def test_arm_state_wraps_public_joint_angles_to_half_open_pi_range():
    driver = PositionRecordingDriver(
        (
            {0: 3072, 1: 3072},
            {0: 0, 1: 0},
            {0: 0, 1: 0},
        )
    )
    arm = make_arm_for_encoding(ids=(0, 1))
    arm._signs = np.array([1, -1])
    arm._driver = driver

    state = arm.get_state()

    np.testing.assert_allclose(state.raw_positions, np.array([1.5 * np.pi, -1.5 * np.pi]))
    np.testing.assert_allclose(state.public_positions, np.array([-0.5 * np.pi, 0.5 * np.pi]))


def test_arm_state_estimator_rejects_velocity_register_spike():
    class VersionedDriver:
        def __init__(self):
            self.sample = FeeTechStateSample(
                position={0: 0},
                velocity={0: 0},
                current={0: 0},
                timestamp=0.0,
                sequence=1,
            )

        def get_state_sample(self):
            return self.sample

    driver = VersionedDriver()
    arm = FeeTechArm(single_joint_arm_config(), driver=driver)
    arm.get_state()
    driver.sample = FeeTechStateSample(
        position={0: 0},
        velocity={0: 32767},
        current={0: 0},
        timestamp=0.01,
        sequence=2,
    )

    state = arm.get_state()
    diagnostics = arm.get_state_estimator_diagnostics()

    assert abs(state.velocities[0]) < 1.0
    assert diagnostics["velocity_accepted"].tolist() == [False]
    assert diagnostics["velocity_rejection_count"].tolist() == [1]


def test_arm_state_estimator_protects_public_position_and_preserves_raw_spike():
    class VersionedDriver:
        def __init__(self):
            self.sample = FeeTechStateSample(
                position={0: 0},
                velocity={0: 0},
                current={0: 0},
                timestamp=0.0,
                sequence=1,
            )

        def get_state_sample(self):
            return self.sample

    driver = VersionedDriver()
    arm = FeeTechArm(single_joint_arm_config(), driver=driver)
    arm.get_state()
    driver.sample = FeeTechStateSample(
        position={0: 1304},
        velocity={0: 0},
        current={0: 0},
        timestamp=0.01,
        sequence=2,
    )

    state = arm.get_state()
    diagnostics = arm.get_state_estimator_diagnostics()

    assert state.raw_positions[0] == pytest.approx(1304 * np.pi / 2048.0)
    assert abs(state.public_positions[0]) < 0.1
    assert diagnostics["position_accepted"].tolist() == [False]


def test_arm_exposes_only_plain_arm_control_api():
    for name in (
        "external_" + "torque_estimation_enabled",
        "external_" + "wrench_frame_name",
        "reset_external_" + "torque_estimator",
        "update_momentum_observer",
        "estimate_joint_" + "external_torque",
        "external_" + "wrench_from_joint_torque",
        "apply_torque_" + "feedback",
        "_torque_" + "feedback",
    ):
        assert not hasattr(FeeTechArm, name)
