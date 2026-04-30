from threading import Lock

import numpy as np
import pytest

from acetele.equipment.feetech.feetech_driver import (
    FeeTechDriver,
    FeeTechStateTimeoutError,
    Mode,
    TorqueEnable,
)
from acetele.equipment.feetech.linker import Linker


class FakeLinker(Linker):
    def __init__(self, current_pos, ids=(0, 1, 2, 3, 4), gripper_id=4):
        self._ids = np.array(ids)
        self._dof = len(self._ids)
        self._gripper_id = gripper_id
        self.current_pos = np.array(current_pos, dtype=float)
        self.position_commands = []
        self.torque_commands = []

    def act(self, encode_gripper=True, cal_torque_sign=False):
        return self.current_pos.copy(), np.zeros(self._dof), np.zeros(self._dof)

    def set_position(self, positions, ids=None, encode_gripper=True):
        self.position_commands.append((np.array(ids), np.array(positions, dtype=float)))

    def set_position_and_torque(self, positions, torques, ids=None, encode_gripper=True):
        self.torque_commands.append(
            (np.array(ids), np.array(positions, dtype=float), np.array(torques, dtype=float))
        )


def test_move_position_uses_shortest_signed_joint_error(monkeypatch):
    linker = FakeLinker(current_pos=[0.0, 0.0, 0.0, 0.0, 0.5])
    monkeypatch.setattr("acetele.equipment.feetech.linker.time.sleep", lambda _duration: None)

    linker.move_position(
        positions=[-2.0, 0.0, 0.0, 0.0, 0.5],
        max_velocity=10.0,
        control_period=0.1,
        step_size=1.0,
    )

    first_joint_path = [command[1][0] for command in linker.position_commands]
    assert first_joint_path[-1] == pytest.approx(-2.0)
    assert np.all(np.diff(first_joint_path) <= 0.0)
    assert np.min(first_joint_path) == pytest.approx(-2.0)


def test_move_position_wraps_revolute_joints_but_not_gripper(monkeypatch):
    current = [np.pi - 0.1, 0.0, 0.0, 0.0, 0.9]
    target = [-np.pi + 0.1, 0.0, 0.0, 0.0, 0.1]
    linker = FakeLinker(current_pos=current)
    monkeypatch.setattr("acetele.equipment.feetech.linker.time.sleep", lambda _duration: None)

    linker.move_position(
        positions=target,
        max_velocity=10.0,
        control_period=0.1,
        step_size=1.0,
    )

    final_position = linker.position_commands[-1][1]
    assert final_position[0] == pytest.approx(np.pi + 0.1)
    assert final_position[-1] == pytest.approx(0.1)


def test_move_position_limits_step_by_velocity_and_sleeps_each_command(monkeypatch):
    sleep_calls = []
    linker = FakeLinker(current_pos=[0.0, 0.0, 0.0, 0.0, 0.5])
    monkeypatch.setattr("acetele.equipment.feetech.linker.time.sleep", sleep_calls.append)

    steps = linker.move_position(
        positions=[0.012, 0.0, 0.0, 0.0, 0.5],
        max_velocity=1.0,
        control_period=0.004,
        step_size=0.01,
        min_steps=1,
    )

    first_joint_path = [command[1][0] for command in linker.position_commands]
    assert steps == 3
    assert np.max(np.abs(np.diff(first_joint_path))) <= 0.004 + 1e-12
    assert sleep_calls == [0.004] * len(linker.position_commands)


def test_move_position_rejects_max_steps_that_would_exceed_velocity_limit(monkeypatch):
    linker = FakeLinker(current_pos=[0.0, 0.0, 0.0, 0.0, 0.5])
    monkeypatch.setattr("acetele.equipment.feetech.linker.time.sleep", lambda _duration: None)

    with pytest.raises(ValueError, match="max_steps"):
        linker.move_position(
            positions=[1.0, 0.0, 0.0, 0.0, 0.5],
            max_velocity=1.0,
            control_period=0.004,
            step_size=0.01,
            min_steps=1,
            max_steps=100,
        )


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


class PositionOnlyDriver(FeeTechDriver):
    def __init__(self):
        self._ids = [1]
        self._mode = {1: Mode.Position}
        self._torque_enable = {1: TorqueEnable.Enable}
        self._packetHandler = FakePacketHandler()
        self._groupSyncWriteGoalCurrentHandler = FakeSyncWriteHandler()
        self._groupSyncWriteGoalPositionHandler = FakeSyncWriteHandler()
        self._groupSyncWriteGoalPositionAndCurrentHandler = FakeSyncWriteHandler()
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


class ModeTrackingDriver(PositionOnlyDriver):
    def __init__(self):
        super().__init__()
        self._mode = {1: Mode.Torque}
        self.mode_calls = []

    def set_mode(self, ids, modes):
        self.mode_calls.append((list(ids), list(modes)))
        for ft_id, mode in zip(ids, modes):
            self._mode[ft_id] = mode


def test_set_position_sends_multiturn_adjusted_position():
    driver = PositionOnlyDriver()

    driver.set_position([1], [100])

    assert driver._groupSyncWriteGoalPositionHandler.params == [(1, [100, 16])]


def test_set_position_and_current_sends_multiturn_adjusted_position():
    driver = PositionOnlyDriver()

    driver.set_position_and_current([1], [100], [33])

    assert driver._groupSyncWriteGoalPositionAndCurrentHandler.params == [(1, [100, 16, 33, 0])]


def test_set_position_switches_from_torque_to_position_mode():
    driver = ModeTrackingDriver()

    driver.set_position([1], [100])

    assert driver.mode_calls == [([1], [Mode.Position])]


def test_set_position_and_current_switches_from_torque_to_position_mode():
    driver = ModeTrackingDriver()

    driver.set_position_and_current([1], [100], [33])

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


def make_linker_for_encoding(ids=(0, 1, 2, 3, 4), gripper_id=4):
    linker = Linker.__new__(Linker)
    linker._ids = np.array(ids)
    linker._dof = len(ids)
    linker._signs = np.ones(len(ids))
    linker._home_poses = np.zeros(len(ids))
    linker._gripper_id = gripper_id
    linker._gripper_type = "ace_leader"
    linker._gripper_encoding_scale = 4.0 / np.pi
    linker._gripper_decoding_scale = np.pi / 4.0
    linker._servo_types = np.array(["HL3915"] * len(ids))
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

    def get_state(self):
        return self.state

    def set_position(self, ids, positions):
        self.position_calls.append((np.array(ids), np.array(positions)))


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

    _, encoded_positions = driver.position_calls[0]
    assert encoded_positions[-1] == pytest.approx(encoded_gripper_position, abs=1)


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

    sent_ids, encoded_positions = driver.position_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([4]))
    assert encoded_positions[0] == pytest.approx(256, abs=1)
