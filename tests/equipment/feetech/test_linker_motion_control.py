import types
from collections import deque
from queue import Queue
from threading import Lock

import numpy as np
import pytest

from acetele.equipment.feetech.feetech_driver import (
    FeeTechDriver,
    FeeTechStateTimeoutError,
    Mode,
    TorqueEnable,
)
from acetele.equipment.feetech.linker import HLS_PROFILE_DEFAULTS_BY_SERVO, Linker
from acetele.utils.gripper import GRIPPER_DECODING_SCALE, GRIPPER_ENCODING_SCALE

PROFILE_VELOCITY_UNIT_RAD_PER_SEC = 0.732 * np.pi / 30.0
PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2 = 8.7 * np.pi / 180.0


def profile_payload(
    position_low,
    position_high,
    servo_type="HL3915",
    acceleration=None,
    current=None,
    velocity=None,
):
    defaults = HLS_PROFILE_DEFAULTS_BY_SERVO[servo_type]
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


def install_driver_profile_defaults(driver, servo_types):
    driver._id_to_index = {int(ft_id): index for index, ft_id in enumerate(driver._ids)}


class FakeLinker(Linker):
    def __init__(self, current_pos, ids=(0, 1, 2, 3, 4), gripper_id=4):
        self._ids = np.array(ids)
        self._dof = len(self._ids)
        self._gripper_id = gripper_id
        self.current_pos = np.array(current_pos, dtype=float)
        self.position_commands = []
        self.torque_commands = []
        self._signs = np.ones(len(self._ids))
        self._gripper_type = "ace_leader"
        if self._gripper_id >= 0:
            self._gripper_encoding_scale = GRIPPER_ENCODING_SCALE[self._gripper_type]
            self._gripper_decoding_scale = GRIPPER_DECODING_SCALE[self._gripper_type]
        self._servo_types = np.array(["HL3915"] * len(ids))
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

    def act(self, encode_gripper=True, cal_torque_sign=False):
        return self.current_pos.copy(), np.zeros(self._dof), np.zeros(self._dof)


def test_linker_set_position_converts_metric_velocity_to_raw_profile_velocity(monkeypatch):
    linker = FakeLinker(current_pos=[0.0, 0.0, 0.0, 0.0, 0.5])
    monkeypatch.setattr(
        "acetele.equipment.feetech.linker.time.sleep",
        lambda _duration: pytest.fail("profile set_position should not sleep"),
    )

    linker.set_position(
        positions=[-2.0, 0.0, 0.0, 0.0, 0.5],
        velocities=26 * PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
    )

    assert len(linker.position_commands) == 1
    sent_ids, sent_positions, kwargs = linker.position_commands[0]
    np.testing.assert_array_equal(sent_ids, np.array([0, 1, 2, 3, 4]))
    assert sent_positions[0] == pytest.approx(-1304, abs=1)
    assert "profile" not in kwargs
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.full(5, 26))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.zeros(5, dtype=int))
    np.testing.assert_array_equal(kwargs["currents_raw"], np.full(5, 500))


def test_set_position_profile_encodes_gripper_without_relative_wrap(monkeypatch):
    current = [np.pi - 0.1, 0.0, 0.0, 0.0, 0.9]
    target = [-np.pi + 0.1, 0.0, 0.0, 0.0, 0.1]
    linker = FakeLinker(current_pos=current)
    monkeypatch.setattr(
        "acetele.equipment.feetech.linker.time.sleep",
        lambda _duration: pytest.fail("profile set_position should not sleep"),
    )

    linker.set_position(
        positions=target,
        velocities=26 * PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
    )

    final_position = linker.position_commands[-1][1]
    assert final_position[0] == pytest.approx(-1983, abs=1)
    assert final_position[-1] == pytest.approx(51, abs=1)


def test_set_position_profile_uses_single_profile_command(monkeypatch):
    linker = FakeLinker(current_pos=[0.0, 0.0, 0.0, 0.0, 0.5])
    monkeypatch.setattr(
        "acetele.equipment.feetech.linker.time.sleep",
        lambda _duration: pytest.fail("profile set_position should not sleep"),
    )

    linker.set_position(
        positions=[0.06, 0.0, 0.0, 0.0, 0.5],
        accelerations=20 * PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2,
    )

    assert len(linker.position_commands) == 1
    assert linker.position_commands[0][1][0] == pytest.approx(39, abs=1)
    np.testing.assert_array_equal(linker.position_commands[0][2]["velocities_raw"], np.full(5, 250))
    np.testing.assert_array_equal(linker.position_commands[0][2]["accelerations_raw"], np.full(5, 20))
    np.testing.assert_array_equal(linker.position_commands[0][2]["currents_raw"], np.full(5, 500))


def test_set_position_explicit_none_velocity_and_acceleration_use_plain_position_write():
    linker = FakeLinker(current_pos=[0.0, 0.0, 0.0, 0.0, 0.5])

    linker.set_position(
        positions=[0.06, 0.0, 0.0, 0.0, 0.5],
        velocities=None,
        accelerations=None,
    )

    assert len(linker.position_commands) == 1
    _, _, kwargs = linker.position_commands[0]
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.full(5, 250))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.zeros(5, dtype=int))
    np.testing.assert_array_equal(kwargs["currents_raw"], np.full(5, 500))


def test_set_position_without_profile_uses_plain_position_write(monkeypatch):
    linker = FakeLinker(current_pos=[0.0, 0.0, 0.0, 0.0, 0.5])
    monkeypatch.setattr(
        "acetele.equipment.feetech.linker.time.sleep",
        lambda _duration: pytest.fail("plain set_position should not sleep"),
    )

    linker.set_position(
        positions=[0.01, 0.0, 0.0, 0.0, 0.5],
    )

    assert len(linker.position_commands) == 1
    assert linker.position_commands[0][1][0] == pytest.approx(7, abs=1)
    np.testing.assert_array_equal(linker.position_commands[0][2]["velocities_raw"], np.full(5, 250))
    np.testing.assert_array_equal(linker.position_commands[0][2]["accelerations_raw"], np.zeros(5, dtype=int))
    np.testing.assert_array_equal(linker.position_commands[0][2]["currents_raw"], np.full(5, 500))


def test_set_position_does_not_accept_runtime_step_parameters():
    linker = FakeLinker(current_pos=[0.0, 0.0, 0.0, 0.0, 0.5])

    with pytest.raises(TypeError):
        linker.set_position(
            positions=[0.0, 0.0, 0.0, 1.0, 0.5],
            step_size=0.25,
        )


def test_linker_set_position_does_not_accept_public_profile_or_current_parameters():
    linker = FakeLinker(current_pos=[0.0, 0.0, 0.0, 0.0, 0.5])

    with pytest.raises(TypeError):
        linker.set_position([0.0, 0.0, 0.0, 1.0, 0.5], profile=True)

    with pytest.raises(TypeError):
        linker.set_position([0.0, 0.0, 0.0, 1.0, 0.5], current=1000)

    with pytest.raises(TypeError):
        linker.set_position([0.0, 0.0, 0.0, 1.0, 0.5], velocity=1.0)

    with pytest.raises(TypeError):
        linker.set_position([0.0, 0.0, 0.0, 1.0, 0.5], acceleration=1.0)


def test_control_loop_sleeps_to_maintain_control_period(monkeypatch):
    linker = FakeLinker(current_pos=[0.0, 0.0, 0.0, 0.0, 0.0])
    linker._control_period = 0.004
    linker._stop_flag = type(
        "StopAfterOneIteration",
        (),
        {"calls": 0, "is_set": lambda self: setattr(self, "calls", self.calls + 1) or self.calls > 1},
    )()
    linker._null_space_regulation = lambda joint_pos, joint_vel: np.zeros(5)
    linker._gravity_compensation = lambda joint_pos, joint_vel: np.ones(5)
    linker._friction_compensation = lambda tau_g, joint_vel: np.zeros(5)
    linker._torque_feedback = lambda joint_vel: np.zeros(5)
    linker.set_torque = lambda tau: linker.torque_commands.append(np.array(tau))

    perf_counter_values = iter([10.0, 10.001])
    sleep_calls = []
    monkeypatch.setattr("acetele.equipment.feetech.linker.time.perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr("acetele.equipment.feetech.linker.time.sleep", sleep_calls.append)

    linker._control_loop()

    assert len(linker.torque_commands) == 1
    assert sleep_calls == [pytest.approx(0.003)]


class FakeSyncWriteHandler:
    def __init__(self):
        self.params = []
        self.avail_flag = type("AlwaysAvailable", (), {"is_set": lambda self: True, "clear": lambda self: None})()

    def addParam(self, ft_id, data):
        self.params.append((ft_id, data))
        return True

    def txPacket(self):
        return 0

    def clearParam(self):
        pass


class CountingAvailFlag:
    def __init__(self):
        self.clear_calls = 0

    def is_set(self):
        return True

    def clear(self):
        self.clear_calls += 1


class CountingSyncWriteHandler(FakeSyncWriteHandler):
    def __init__(self):
        super().__init__()
        self.tx_calls = []
        self.clear_calls = 0
        self.avail_flag = CountingAvailFlag()

    def txPacket(self):
        self.tx_calls.append(list(self.params))
        return 0

    def clearParam(self):
        self.clear_calls += 1
        self.params = []


class FakePacketHandler:
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

        def openPort(self):
            return True

        def setBaudRate(self, _baudrate):
            return True

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self):
            pass

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


class PositionOnlyDriver(FeeTechDriver):
    def __init__(self):
        self._ids = [1]
        install_driver_profile_defaults(self, ["HL3915"])
        self._mode = {1: Mode.Position}
        self._torque_enable = {1: TorqueEnable.Enable}
        self._packetHandler = FakePacketHandler()
        self._groupSyncWriteGoalCurrentHandler = FakeSyncWriteHandler()
        self._groupSyncWriteGoalPositionProfileHandler = FakeSyncWriteHandler()
        self._time_windows = deque(maxlen=100)
        self.queued_tasks = []

        class ImmediateQueue:
            def __init__(self, outer):
                self.outer = outer

            def put(self, task):
                self.outer.queued_tasks.append(task)
                task()

        self._comm_task_queue = ImmediateQueue(self)

    def get_state(self):
        return ({1: 5000}, {}, {})

    def set_mode(self, ids, modes):
        pass

    def set_torque_enable(self, ids, enables):
        pass


class MultiIdPositionDriver(PositionOnlyDriver):
    def __init__(self):
        super().__init__()
        self._ids = [1, 2]
        install_driver_profile_defaults(self, ["HL3915", "HL3915"])
        self._mode = {1: Mode.Position, 2: Mode.Position}
        self._torque_enable = {1: TorqueEnable.Enable, 2: TorqueEnable.Enable}

    def get_state(self):
        return ({1: 5000, 2: 5000}, {}, {})


class BaseMethodDriver(FeeTechDriver):
    def __init__(self):
        self._ids = [1]
        install_driver_profile_defaults(self, ["HL3915"])
        self._mode = {1: Mode.Position}
        self._torque_enable = {1: TorqueEnable.Enable}
        self._packetHandler = FakePacketHandler()
        self._groupSyncWriteModeHandler = CountingSyncWriteHandler()
        self._groupSyncWriteTorqueEnableHandler = CountingSyncWriteHandler()
        self._groupSyncWriteGoalCurrentHandler = CountingSyncWriteHandler()
        self._groupSyncWriteGoalPositionProfileHandler = CountingSyncWriteHandler()
        self._comm_task_queue = Queue(maxsize=32)
        self._time_windows = deque(maxlen=100)
        self._stop_flag = types.SimpleNamespace(clear=lambda: None)
        self.read_calls = 0
        self.executed_tasks = []

    def get_state(self):
        return ({1: 5000}, {}, {})

    def _read_state(self):
        self.read_calls += 1


class ModeTrackingDriver(PositionOnlyDriver):
    def __init__(self):
        super().__init__()
        self._mode = {1: Mode.Torque}
        self.mode_calls = []

    def set_mode(self, ids, modes):
        self.mode_calls.append((list(ids), list(modes)))
        for ft_id, mode in zip(ids, modes):
            self._mode[ft_id] = mode


class MixedServoPositionDriver(PositionOnlyDriver):
    def __init__(self):
        super().__init__()
        self._ids = [0, 1, 2]
        install_driver_profile_defaults(self, ["HL3950", "HL3930", "HL3915"])
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
        "HL3950": {"acceleration": 0, "current": 1000, "velocity": 110},
        "HL3930": {"acceleration": 250, "current": 1000, "velocity": 100},
        "HL3915": {"acceleration": 0, "current": 500, "velocity": 250},
    }


def test_feetech_driver_does_not_accept_servo_types(monkeypatch):
    with pytest.raises(TypeError):
        FeeTechDriver([0, 1], "/dev/test", servo_types=["HL3915", "HL3915"])


def test_set_position_requires_full_raw_profile_fields():
    driver = PositionOnlyDriver()

    with pytest.raises(AssertionError, match="velocities_raw, accelerations_raw, and currents_raw are required"):
        driver.set_position([1], [100], currents_raw=[33])


def test_set_position_profile_sends_acceleration_position_current_and_velocity():
    driver = PositionOnlyDriver()

    driver.set_position([1], [100], velocities_raw=[26], accelerations_raw=[20], currents_raw=[82])

    assert driver._groupSyncWriteGoalPositionProfileHandler.params == [(1, [20, 100, 16, 82, 0, 26, 0])]


def test_set_position_profile_rejects_out_of_range_profile_values():
    driver = PositionOnlyDriver()

    with pytest.raises(AssertionError, match="accelerations must fit in one byte"):
        driver.set_position([1], [100], velocities_raw=[26], accelerations_raw=[256], currents_raw=[82])

    with pytest.raises(AssertionError, match="velocities must fit in 15 bits"):
        driver.set_position([1], [100], velocities_raw=[32768], accelerations_raw=[20], currents_raw=[82])

    with pytest.raises(AssertionError, match="currents must fit in 15 bits"):
        driver.set_position([1], [100], velocities_raw=[26], accelerations_raw=[20], currents_raw=[32768])


def test_set_position_profile_rejects_negative_raw_profile_values():
    driver = PositionOnlyDriver()

    with pytest.raises(AssertionError, match="velocities must be non-negative"):
        driver.set_position([1], [100], velocities_raw=[-1], accelerations_raw=[20], currents_raw=[82])

    with pytest.raises(AssertionError, match="accelerations must be non-negative"):
        driver.set_position([1], [100], velocities_raw=[26], accelerations_raw=[-1], currents_raw=[82])

    with pytest.raises(AssertionError, match="currents must be non-negative"):
        driver.set_position([1], [100], velocities_raw=[26], accelerations_raw=[20], currents_raw=[-1])


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
        (0, profile_payload(100, 16, servo_type="HL3950")),
        (1, profile_payload(200, 16, servo_type="HL3930")),
        (2, profile_payload(250, 16, servo_type="HL3915")),
    ]


def test_set_position_model_defaults_fill_only_omitted_profile_fields():
    linker = FakeLinker(current_pos=[0.0, 0.0, 0.0], ids=(0, 1, 2), gripper_id=-1)
    linker._servo_types = np.array(["HL3950", "HL3930", "HL3915"])
    linker._profile_acceleration_defaults = np.array([0, 250, 0])
    linker._profile_current_defaults = np.array([1000, 1000, 500])
    linker._profile_velocity_defaults = np.array([110, 100, 250])

    linker.set_position([0.1, 0.2, 0.3], ids=[0, 1, 2], torque=[0.1, 0.1, 0.1])
    _, _, kwargs = linker.position_commands[-1]
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.array([110, 100, 250]))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.array([0, 250, 0]))

    linker.position_commands = []
    linker.set_position([0.1, 0.2, 0.3], ids=[0, 1, 2], velocities=[1.0, 1.0, 1.0])
    _, _, kwargs = linker.position_commands[-1]
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.array([0, 250, 0]))
    np.testing.assert_array_equal(kwargs["currents_raw"], np.array([1000, 1000, 500]))

    linker.position_commands = []
    linker.set_position([0.1, 0.2, 0.3], ids=[0, 1, 2], accelerations=[1.0, 1.0, 1.0])
    _, _, kwargs = linker.position_commands[-1]
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.array([110, 100, 250]))
    np.testing.assert_array_equal(kwargs["currents_raw"], np.array([1000, 1000, 500]))


def test_set_position_without_profile_arguments_uses_full_profile_write():
    driver = PositionOnlyDriver()

    driver.set_position([1], [100], velocities_raw=[250], accelerations_raw=[0], currents_raw=[500])

    assert driver._groupSyncWriteGoalPositionProfileHandler.params == [(1, profile_payload(100, 16))]


def test_set_position_switches_from_torque_to_position_mode():
    driver = ModeTrackingDriver()

    driver.set_position([1], [100], velocities_raw=[250], accelerations_raw=[0], currents_raw=[500])

    assert driver.mode_calls == [([1], [Mode.Position])]


def test_set_mode_noop_does_not_enqueue_task_or_touch_handler():
    driver = BaseMethodDriver()

    driver.set_mode([1], [Mode.Position])

    assert driver._comm_task_queue.empty()
    assert driver._groupSyncWriteModeHandler.params == []
    assert driver._groupSyncWriteModeHandler.avail_flag.clear_calls == 0


def test_set_torque_enable_noop_does_not_enqueue_task_or_touch_handler():
    driver = BaseMethodDriver()

    driver.set_torque_enable([1], [TorqueEnable.Enable])

    assert driver._comm_task_queue.empty()
    assert driver._groupSyncWriteTorqueEnableHandler.params == []
    assert driver._groupSyncWriteTorqueEnableHandler.avail_flag.clear_calls == 0


def test_open_invalidates_mode_and_torque_cache(monkeypatch):
    driver = BaseMethodDriver()
    started = []
    monkeypatch.setattr(
        "acetele.equipment.feetech.feetech_driver.Thread",
        lambda target, daemon: types.SimpleNamespace(start=lambda: started.append((target, daemon))),
    )

    driver.open()

    assert driver._mode == {1: None}
    assert driver._torque_enable == {1: None}
    assert started == [(driver._comm_worker, True)]


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


def test_comm_worker_executes_one_write_task_after_each_read(monkeypatch):
    driver = BaseMethodDriver()
    executed = []
    for index in range(3):
        driver._comm_task_queue.put(lambda index=index: executed.append(index))

    class StopAfterOneLoop:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 1

    driver._stop_flag = StopAfterOneLoop()
    monkeypatch.setattr("acetele.equipment.feetech.feetech_driver.time.sleep", lambda _duration: None)

    FeeTechDriver._comm_worker(driver)

    assert driver.read_calls == 1
    assert executed == [0]
    assert driver._comm_task_queue.qsize() == 2


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
def test_linker_set_torque_enable_for_selected_ids(enable):
    linker = make_linker_for_encoding(ids=(0, 1, 4), gripper_id=4)
    driver = TorqueEnableRecordingDriver()
    linker._driver = driver

    linker.set_torque_enable(enable, ids=[0, 1])

    assert driver.torque_enable_calls == [([0, 1], [enable, enable])]


def test_set_position_with_raw_current_switches_from_torque_to_position_mode():
    driver = ModeTrackingDriver()

    driver.set_position([1], [100], velocities_raw=[250], accelerations_raw=[0], currents_raw=[33])

    assert driver.mode_calls == [([1], [Mode.Position])]


def test_set_current_switches_to_torque_mode():
    driver = ModeTrackingDriver()
    driver._mode = {1: Mode.Position}

    driver.set_current([1], [33])

    assert driver.mode_calls == [([1], [Mode.Torque])]


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


class CurrentRecordingDriver:
    def __init__(self):
        self.current_calls = []

    def set_current(self, ids, currents):
        self.current_calls.append((np.array(ids), np.array(currents)))


def make_linker_for_encoding(ids=(0, 1, 2, 3, 4), gripper_id=4, gripper_type="ace_leader"):
    linker = Linker.__new__(Linker)
    linker._ids = np.array(ids)
    linker._dof = len(ids)
    linker._signs = np.ones(len(ids))
    linker._home_poses = np.zeros(len(ids))
    linker._gripper_id = gripper_id
    linker._gripper_type = gripper_type
    linker._gripper_encoding_scale = GRIPPER_ENCODING_SCALE[linker._gripper_type]
    linker._gripper_decoding_scale = GRIPPER_DECODING_SCALE[linker._gripper_type]
    linker._servo_types = np.array(["HL3915"] * len(ids))
    linker._profile_acceleration_defaults = np.zeros(len(ids), dtype=int)
    linker._profile_current_defaults = np.full(len(ids), 500, dtype=int)
    linker._profile_velocity_defaults = np.full(len(ids), 250, dtype=int)
    linker._torque_current_mapping = np.ones(len(ids)) * (1000.0 / 9.3)
    linker._no_load_current = np.ones(len(ids)) * 260
    linker._pin_model = None
    linker._pin_data = None
    linker._enable_gravity_compensation = False
    linker._enable_estimate_external_torque = False
    return linker


def test_set_torque_excludes_only_configured_gripper():
    linker = make_linker_for_encoding()
    driver = CurrentRecordingDriver()
    linker._driver = driver

    linker.set_torque([0.1, 0.2, 0.3, 0.4, 0.5])

    sent_ids, sent_currents = driver.current_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([0, 1, 2, 3]))
    assert len(sent_currents) == 4


def test_set_torque_accepts_list_ids_when_excluding_configured_gripper():
    linker = make_linker_for_encoding()
    driver = CurrentRecordingDriver()
    linker._driver = driver

    linker.set_torque([0.1, 0.5], ids=[0, 4])

    sent_ids, sent_currents = driver.current_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([0]))
    assert len(sent_currents) == 1


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


class ImmediateProfileDriver(PositionOnlyDriver):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def get_state(self):
        return self.state


def test_linker_decodes_normalized_follower_gripper_home_pose():
    config = {
        "joint_ids": [0, 1, 4],
        "joint_signs": [1, 1, 1],
        "home_poses": [0.25, -0.5, 1.0],
        "port": "/dev/test",
        "gripper_id": 4,
        "gripper_type": "ace_follower",
        "enable_gravity_compensation": False,
        "enable_estimate_external_torque": False,
        "servo_types": ["HL3915", "HL3915", "HL3915"],
    }
    driver = PositionRecordingDriver(({0: 0, 1: 0, 4: 0}, {0: 0, 1: 0, 4: 0}, {0: 0, 1: 0, 4: 0}))

    linker = Linker(config, driver=driver)

    np.testing.assert_allclose(linker._home_poses, np.array([0.25, -0.5, 896.0 * np.pi / 2048.0]))


def test_linker_requires_servo_types():
    config = {
        "joint_ids": [0, 1, 2],
        "joint_signs": [1, 1, 1],
        "home_poses": [0.0, 0.0, 0.0],
        "port": "/dev/test",
        "gripper_id": -1,
        "gripper_type": "ace_leader",
        "enable_gravity_compensation": False,
        "enable_estimate_external_torque": False,
    }
    driver = PositionRecordingDriver(({0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}))

    with pytest.raises(ValueError, match="servo_types must be specified"):
        Linker(config, driver=driver)


def test_linker_rejects_invalid_servo_types():
    config = {
        "joint_ids": [0, 1, 2],
        "joint_signs": [1, 1, 1],
        "home_poses": [0.0, 0.0, 0.0],
        "port": "/dev/test",
        "gripper_id": -1,
        "gripper_type": "ace_leader",
        "enable_gravity_compensation": False,
        "enable_estimate_external_torque": False,
        "servo_types": ["HL3915", "HL3915"],
    }
    driver = PositionRecordingDriver(({0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}))

    with pytest.raises(ValueError, match="servo_types must have the same length"):
        Linker(config, driver=driver)

    config["servo_types"] = ["HL3915", "BAD", "HL3915"]
    with pytest.raises(ValueError, match="unsupported servo_types"):
        Linker(config, driver=driver)


def test_linker_constructs_feetech_driver_without_servo_types(monkeypatch):
    captured = {}

    class CapturingFeeTechDriver:
        def __init__(self, ids, port):
            captured["ids"] = np.array(ids)
            captured["port"] = port

    monkeypatch.setattr("acetele.equipment.feetech.linker.FeeTechDriver", CapturingFeeTechDriver)
    config = {
        "joint_ids": [0, 1, 2],
        "joint_signs": [1, 1, 1],
        "home_poses": [0.0, 0.0, 0.0],
        "port": "/dev/test",
        "gripper_id": -1,
        "gripper_type": "ace_leader",
        "enable_gravity_compensation": False,
        "enable_estimate_external_torque": False,
        "servo_types": ["HL3950", "HL3930", "HL3915"],
    }

    linker = Linker(config)

    assert isinstance(linker._driver, CapturingFeeTechDriver)
    np.testing.assert_array_equal(captured["ids"], np.array([0, 1, 2]))
    assert captured["port"] == "/dev/test"


def test_linker_plain_position_fills_model_profile_defaults():
    driver = PositionRecordingDriver(({0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}))
    config = {
        "joint_ids": [0, 1, 2],
        "joint_signs": [1, 1, 1],
        "home_poses": [0.0, 0.0, 0.0],
        "port": "/dev/test",
        "gripper_id": -1,
        "gripper_type": "ace_leader",
        "enable_gravity_compensation": False,
        "enable_estimate_external_torque": False,
        "servo_types": ["HL3950", "HL3930", "HL3915"],
    }
    linker = Linker(config, driver=driver)

    linker.set_position([0.0, 0.0, 0.0])

    _, _, kwargs = driver.profile_calls[0]
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.array([110, 100, 250]))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.array([0, 250, 0]))
    np.testing.assert_array_equal(kwargs["currents_raw"], np.array([1000, 1000, 500]))


def test_linker_metric_profile_reaches_hls_payload():
    driver = ImmediateProfileDriver(({0: 5000}, {0: 0}, {0: 0}))
    config = {
        "joint_ids": [0],
        "joint_signs": [1],
        "home_poses": [0.0],
        "port": "/dev/test",
        "gripper_id": -1,
        "gripper_type": "ace_leader",
        "enable_gravity_compensation": False,
        "enable_estimate_external_torque": False,
        "servo_types": ["HL3915"],
    }
    linker = Linker(config, driver=driver)

    linker.set_position(
        [100.0 * np.pi / 2048.0],
        velocities=26 * PROFILE_VELOCITY_UNIT_RAD_PER_SEC,
        accelerations=20 * PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2,
        torque=0.1,
    )

    expected_current = round(((1000.0 / 9.3) * (0.1 / 0.0981) + 260) / 6.5)
    assert driver._groupSyncWriteGoalPositionProfileHandler.params == [
        (0, profile_payload(100, 16, acceleration=20, current=expected_current, velocity=26))
    ]


def test_gripper_decoding_inverts_encoded_act_value():
    encoded_gripper_position = 512
    driver = PositionRecordingDriver(
        (
            {0: 0, 1: 0, 2: 0, 3: 0, 4: encoded_gripper_position},
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        )
    )
    linker = make_linker_for_encoding()
    linker._driver = driver

    positions, _, _ = linker.act()
    linker.set_position(positions)

    _, encoded_positions, _ = driver.profile_calls[0]
    assert encoded_positions[-1] == pytest.approx(encoded_gripper_position, abs=1)


def test_act_without_gripper_keeps_last_joint_in_radians():
    driver = PositionRecordingDriver(
        (
            {0: 0, 1: 1024},
            {0: 0, 1: 0},
            {0: 0, 1: 0},
        )
    )
    linker = make_linker_for_encoding(ids=(0, 1), gripper_id=-1)
    linker._driver = driver

    positions, _, _ = linker.act()

    np.testing.assert_allclose(positions, np.array([0.0, np.pi / 2]))


def test_act_uses_configured_gripper_id_not_last_joint():
    driver = PositionRecordingDriver(
        (
            {4: 512, 0: 1024, 1: 0},
            {4: 0, 0: 0, 1: 0},
            {4: 0, 0: 0, 1: 0},
        )
    )
    linker = make_linker_for_encoding(ids=(4, 0, 1), gripper_id=4)
    linker._driver = driver

    positions, _, _ = linker.act()

    assert positions[0] == pytest.approx(1.0)
    assert positions[1] == pytest.approx(np.pi / 2)
    assert positions[2] == pytest.approx(0.0)


def test_set_position_does_not_mutate_numpy_positions():
    driver = PositionRecordingDriver(
        (
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        )
    )
    linker = make_linker_for_encoding()
    linker._driver = driver
    positions = np.array([0.0, 0.0, 0.0, 0.0, 0.25])

    linker.set_position(positions)

    np.testing.assert_array_equal(positions, np.array([0.0, 0.0, 0.0, 0.0, 0.25]))


def test_set_position_accepts_list_ids_for_gripper_encoding():
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    linker = make_linker_for_encoding()
    linker._driver = driver

    linker.set_position([0.5], ids=[4])

    sent_ids, encoded_positions, _ = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([4]))
    assert encoded_positions[0] == pytest.approx(256, abs=1)


def test_set_position_uses_follower_gripper_travel():
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    linker = make_linker_for_encoding(ids=(4,), gripper_type="ace_follower")
    linker._driver = driver

    linker.set_position([0.5], ids=[4])

    sent_ids, encoded_positions, _ = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([4]))
    assert encoded_positions[0] == pytest.approx(448, abs=1)


def test_set_position_applies_gripper_sign_once():
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    linker = make_linker_for_encoding(ids=(4,), gripper_type="ace_follower")
    linker._signs = np.array([-1])
    linker._driver = driver

    linker.set_position([1.0], ids=[4])

    _, encoded_positions, _ = driver.profile_calls[0]
    assert encoded_positions[0] == pytest.approx(-896, abs=1)


def test_set_position_uses_configured_gripper_id_not_last_joint():
    driver = PositionRecordingDriver(({4: 0, 0: 0, 1: 0}, {4: 0, 0: 0, 1: 0}, {4: 0, 0: 0, 1: 0}))
    linker = make_linker_for_encoding(ids=(4, 0, 1), gripper_id=4)
    linker._driver = driver

    linker.set_position([0.5, np.pi / 2, 0.0])

    sent_ids, encoded_positions, _ = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([4, 0, 1]))
    assert encoded_positions[0] == pytest.approx(256, abs=1)
    assert encoded_positions[1] == pytest.approx(1024, abs=1)
    assert encoded_positions[2] == pytest.approx(0, abs=1)


def test_linker_set_position_profile_converts_only_explicit_metric_velocity():
    driver = PositionRecordingDriver(
        (
            {0: 0, 1: 0, 4: 0},
            {0: 0, 1: 0, 4: 0},
            {0: 0, 1: 0, 4: 0},
        )
    )
    linker = make_linker_for_encoding(ids=(0, 1, 4), gripper_id=4)
    linker._driver = driver

    linker.set_position([0.1, 0.0, 0.5], velocities=26 * PROFILE_VELOCITY_UNIT_RAD_PER_SEC)

    assert driver.position_calls == []
    sent_ids, positions, kwargs = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([0, 1, 4]))
    assert positions[0] == pytest.approx(65, abs=1)
    assert positions[2] == pytest.approx(256, abs=1)
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.array([26, 26, 26]))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.array([0, 0, 0]))
    np.testing.assert_array_equal(kwargs["currents_raw"], np.array([500, 500, 500]))
    assert "profile" not in kwargs


def test_linker_set_position_torque_overrides_profile_current():
    driver = PositionRecordingDriver(
        (
            {0: 0},
            {0: 0},
            {0: 0},
        )
    )
    linker = make_linker_for_encoding(ids=(0,), gripper_id=-1)
    linker._driver = driver

    linker.set_position([0.1], velocities=26 * PROFILE_VELOCITY_UNIT_RAD_PER_SEC, torque=[0.1])

    _, _, kwargs = driver.profile_calls[0]
    expected_current = round(((1000.0 / 9.3) * (0.1 / 0.0981) + 260) / 6.5)
    assert kwargs["currents_raw"][0] == pytest.approx(expected_current)


def test_linker_set_position_torque_without_velocity_uses_profile_defaults():
    driver = PositionRecordingDriver(
        (
            {0: 0},
            {0: 0},
            {0: 0},
        )
    )
    linker = make_linker_for_encoding(ids=(0,), gripper_id=-1)
    linker._driver = driver

    linker.set_position([0.1], torque=[0.1])

    assert driver.position_calls == []
    sent_ids, positions, kwargs = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([0]))
    assert positions[0] == pytest.approx(65, abs=1)
    np.testing.assert_array_equal(kwargs["velocities_raw"], np.array([250]))
    np.testing.assert_array_equal(kwargs["accelerations_raw"], np.array([0]))
    expected_current = round(((1000.0 / 9.3) * (0.1 / 0.0981) + 260) / 6.5)
    assert kwargs["currents_raw"][0] == pytest.approx(expected_current)


def test_linker_set_position_broadcasts_metric_profile_scalars_to_raw_arrays():
    driver = PositionRecordingDriver(
        (
            {0: 0, 1: 0},
            {0: 0, 1: 0},
            {0: 0, 1: 0},
        )
    )
    linker = make_linker_for_encoding(ids=(0, 1), gripper_id=-1)
    linker._driver = driver

    linker.set_position(
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
