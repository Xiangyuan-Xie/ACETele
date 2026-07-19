from queue import Queue
from threading import Event, Lock, RLock, Thread
from types import SimpleNamespace

import pytest

from acetele.equipment.feetech.feetech_driver import (
    FeeTechCalibrationError,
    FeeTechCommandDispatchError,
    FeeTechDriver,
    Mode,
    TorqueEnable,
)
from acetele.equipment.feetech.feetech_sdk.scservo_def import COMM_SUCCESS


class FakePacketHandler:
    def __init__(self, failures=None):
        self.failures = failures or {}
        self.calls = []

    def reOfsCal(self, ft_id, pose):
        self.calls.append((ft_id, pose))
        if hasattr(self, "events"):
            self.events.append(("calibrate", ft_id, pose))
        return self.failures.get(ft_id, (COMM_SUCCESS, 0))

    def getTxRxResult(self, comm_result):
        return f"comm result {comm_result}"

    def getRxPacketError(self, error):
        return f"packet error {error}"


class CalibrationOnlyDriver(FeeTechDriver):
    def __init__(self, packet_handler, ids=(1, 2, 3)):
        self._ids = list(ids)
        self._packetHandler = packet_handler
        self._mode = {ft_id: Mode.Position for ft_id in self._ids}
        self._torque_enable = {ft_id: TorqueEnable.Enable for ft_id in self._ids}
        self._lock = Lock()
        self._lifecycle_lock = RLock()
        self._stop_flag = Event()
        self._comm_fault = None
        self._comm_thread = SimpleNamespace(is_alive=lambda: True)
        self._comm_task_queue = Queue()
        self.pause_calls = 0
        self.open_calls = 0
        self.events = []
        self.disable_error = None
        self.pause_error = None
        self.open_error = None
        packet_handler.events = self.events

    def set_torque_enable(
        self,
        ids,
        enables,
        force=False,
        *,
        wait=False,
        timeout=1.0,
    ):
        del timeout
        self.events.append(("disable", tuple(ids), tuple(enables), force, wait))
        if self.disable_error is not None:
            raise self.disable_error

    def _pause_comm_worker(self, timeout=1.0):
        del timeout
        self.pause_calls += 1
        self.events.append("pause")
        if self.pause_error is not None:
            raise self.pause_error

    def open(self):
        self.open_calls += 1
        self.events.append("open")
        if self.open_error is not None:
            raise self.open_error


def test_calibrate_raises_error_with_failed_id_pose_and_partial_success():
    packet_handler = FakePacketHandler(failures={2: (COMM_SUCCESS, 7)})
    driver = CalibrationOnlyDriver(packet_handler)

    with pytest.raises(FeeTechCalibrationError) as exc_info:
        driver.calibrate([1, 2, 3], [100, 200, 300])

    message = str(exc_info.value)
    assert "ID 2" in message
    assert "pose 200" in message
    assert "packet error 7" in message
    assert "already calibrated IDs: [1]" in message
    assert packet_handler.calls == [(1, 100), (2, 200)]
    assert driver.pause_calls == 1
    assert driver.open_calls == 1
    assert driver.events[:3] == [
        (
            "disable",
            (1, 2, 3),
            (TorqueEnable.Disable,) * 3,
            True,
            True,
        ),
        "pause",
        ("calibrate", 1, 100),
    ]


def test_calibrate_reopens_driver_after_success():
    packet_handler = FakePacketHandler()
    driver = CalibrationOnlyDriver(packet_handler)

    driver.calibrate([1, 2], [100, 200])

    assert packet_handler.calls == [(1, 100), (2, 200)]
    assert driver.pause_calls == 1
    assert driver.open_calls == 1
    assert driver.events == [
        (
            "disable",
            (1, 2),
            (TorqueEnable.Disable,) * 2,
            True,
            True,
        ),
        "pause",
        ("calibrate", 1, 100),
        ("calibrate", 2, 200),
        "open",
    ]


def test_calibrate_blocks_new_command_submission_until_worker_resumes():
    calibration_started = Event()
    release_calibration = Event()

    class BlockingPacketHandler(FakePacketHandler):
        def reOfsCal(self, ft_id, pose):
            calibration_started.set()
            assert release_calibration.wait(timeout=1.0)
            return super().reOfsCal(ft_id, pose)

    driver = CalibrationOnlyDriver(BlockingPacketHandler(), ids=(1,))
    calibration_thread = Thread(target=driver.calibrate, args=([1], [100]))
    calibration_thread.start()
    assert calibration_started.wait(timeout=1.0)

    command_submitted = Event()

    def submit_command() -> None:
        driver._submit_comm_task(lambda: None, label="test command")
        command_submitted.set()

    command_thread = Thread(target=submit_command)
    command_thread.start()
    assert not command_submitted.wait(timeout=0.05)

    release_calibration.set()
    calibration_thread.join(timeout=1.0)
    command_thread.join(timeout=1.0)

    assert not calibration_thread.is_alive()
    assert not command_thread.is_alive()
    assert command_submitted.is_set()
    assert driver._comm_task_queue.qsize() == 1


def test_calibrate_does_not_write_eeprom_when_torque_disable_fails():
    packet_handler = FakePacketHandler()
    driver = CalibrationOnlyDriver(packet_handler)
    driver.disable_error = FeeTechCommandDispatchError("disable failed")

    with pytest.raises(FeeTechCommandDispatchError, match="disable failed"):
        driver.calibrate([1], [100])

    assert packet_handler.calls == []
    assert driver.pause_calls == 0
    assert driver.open_calls == 0
    assert driver._mode == {1: None, 2: None, 3: None}
    assert driver._torque_enable == {1: None, 2: None, 3: None}


def test_calibrate_does_not_write_eeprom_when_worker_pause_fails():
    packet_handler = FakePacketHandler()
    driver = CalibrationOnlyDriver(packet_handler)
    driver.pause_error = FeeTechCommandDispatchError("pause failed")

    with pytest.raises(FeeTechCommandDispatchError, match="pause failed"):
        driver.calibrate([1], [100])

    assert packet_handler.calls == []
    assert driver.pause_calls == 1
    assert driver.open_calls == 0
    assert driver._mode == {1: None, 2: None, 3: None}
    assert driver._torque_enable == {1: None, 2: None, 3: None}


def test_calibration_error_is_preserved_when_resuming_worker_also_fails():
    packet_handler = FakePacketHandler(failures={2: (COMM_SUCCESS, 7)})
    driver = CalibrationOnlyDriver(packet_handler)
    driver.open_error = RuntimeError("resume failed")

    with pytest.raises(FeeTechCalibrationError) as exc_info:
        driver.calibrate([1, 2], [100, 200])

    assert "packet error 7" in str(exc_info.value)
    assert "additionally failed to resume" in str(exc_info.value)
    assert "resume failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, FeeTechCalibrationError)
    assert driver._mode == {1: None, 2: None, 3: None}
    assert driver._torque_enable == {1: None, 2: None, 3: None}


def test_successful_calibration_reports_worker_resume_failure():
    packet_handler = FakePacketHandler()
    driver = CalibrationOnlyDriver(packet_handler)
    driver.open_error = RuntimeError("resume failed")

    with pytest.raises(FeeTechCommandDispatchError, match="Calibration completed.*resume failed") as exc_info:
        driver.calibrate([1], [100])

    assert isinstance(exc_info.value.__cause__, RuntimeError)
