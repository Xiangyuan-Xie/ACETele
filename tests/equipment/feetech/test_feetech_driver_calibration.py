import pytest

from acetele.equipment.feetech.feetech_driver import FeeTechCalibrationError, FeeTechDriver
from acetele.equipment.feetech.feetech_sdk.scservo_def import COMM_SUCCESS


class FakePacketHandler:
    def __init__(self, failures=None):
        self.failures = failures or {}
        self.calls = []

    def reOfsCal(self, ft_id, pose):
        self.calls.append((ft_id, pose))
        return self.failures.get(ft_id, (COMM_SUCCESS, 0))

    def getTxRxResult(self, comm_result):
        return f"comm result {comm_result}"

    def getRxPacketError(self, error):
        return f"packet error {error}"


class CalibrationOnlyDriver(FeeTechDriver):
    def __init__(self, packet_handler, ids=(1, 2, 3)):
        self._ids = list(ids)
        self._packetHandler = packet_handler
        self.close_calls = 0
        self.open_calls = 0

    def close(self):
        self.close_calls += 1

    def open(self):
        self.open_calls += 1


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
    assert driver.close_calls == 1
    assert driver.open_calls == 1


def test_calibrate_reopens_driver_after_success():
    packet_handler = FakePacketHandler()
    driver = CalibrationOnlyDriver(packet_handler)

    driver.calibrate([1, 2], [100, 200])

    assert packet_handler.calls == [(1, 100), (2, 200)]
    assert driver.close_calls == 1
    assert driver.open_calls == 1
