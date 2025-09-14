import time
from collections import deque
from threading import Event, Lock, Thread
from typing import Deque, Dict, Sequence, Tuple

import numpy as np
from equipment.feetech.feetech_sdk.group_sync_read import GroupSyncRead
from equipment.feetech.feetech_sdk.hls import hls
from equipment.feetech.feetech_sdk.port_handler import PortHandler
from equipment.feetech.feetech_sdk.scservo_def import COMM_SUCCESS
from equipment.feetech.feetech_sdk.sms_sts import (
    SMS_STS_PRESENT_POSITION_L,
    SMS_STS_PRESENT_SPEED_L,
)
from loguru import logger


class FeeTechDriver:
    def __init__(self, ids: Sequence[int], port: str, baudrate: int = 1000000):
        self._ids = ids

        self._portHandler = PortHandler(port)
        assert self._portHandler.openPort(), f"Failed to open the port {port}"
        assert self._portHandler.setBaudRate(baudrate), f"Failed to set the baudrate {baudrate}"
        self._packetHandler = hls(self._portHandler)
        self._groupSyncReadHandler = GroupSyncRead(self._packetHandler, SMS_STS_PRESENT_POSITION_L, 4)

        self._position: Dict[int, int] = {}
        self._velocity: Dict[int, int] = {}

        self._time_windows: Deque[float] = deque(maxlen=100)

        self._lock = Lock()
        self._stop_flag = Event()
        self._comm_thread = Thread(target=self._comm_worker, daemon=True)
        self._comm_thread.start()

    def _comm_worker(self):
        while not self._stop_flag.is_set():
            start_time = time.perf_counter()
            self._read_pos_and_vel()
            end_time = time.perf_counter()
            self._time_windows.append(end_time - start_time)

    def _read_pos_and_vel(self):
        position = {}
        velocity = {}

        for ft_id in self._ids:
            if not self._groupSyncReadHandler.addParam(ft_id):
                logger.error(f"[ID:{ft_id}] groupSyncRead addparam failed")

        comm_result = self._groupSyncReadHandler.txRxPacket()
        if comm_result != COMM_SUCCESS:
            logger.error(self._packetHandler.getTxRxResult(comm_result))

        for ft_id in self._ids:
            data_result, error = self._groupSyncReadHandler.isAvailable(ft_id, SMS_STS_PRESENT_POSITION_L, 4)
            if not data_result:
                logger.error(f"[ID:{ft_id}] groupSyncRead getdata failed")
                continue
            if error != 0:
                logger.error(self._packetHandler.getRxPacketError(error))
                continue
            position[ft_id] = self._groupSyncReadHandler.getData(ft_id, SMS_STS_PRESENT_POSITION_L, 2)
            velocity[ft_id] = self._packetHandler.scs_tohost(
                self._groupSyncReadHandler.getData(ft_id, SMS_STS_PRESENT_SPEED_L, 2), 15
            )

        with self._lock:
            self._position = position
            self._velocity = velocity

        self._groupSyncReadHandler.clearParam()

    def calibrate(self, ids: Sequence[int], home_poses: Sequence[int]):
        assert len(ids) == len(home_poses), "ids and home_poses must have the same length."
        assert all(ft_id in self._ids for ft_id in ids), "some ids are not registered in FeeTechDriver."
        for ft_id, pose in zip(ids, home_poses):
            comm_result, error = self._packetHandler.reOfsCal(ft_id, pose)
            if comm_result != COMM_SUCCESS:
                logger.error(self._packetHandler.getTxRxResult(comm_result))
                continue
            if error != 0:
                logger.error(self._packetHandler.getRxPacketError(error))
                continue

    def get_pos_and_vel(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        with self._lock:
            return self._position, self._velocity

    def get_frequency(self) -> float:
        if len(self._time_windows) == 0:
            return 0.0
        return 1.0 / np.mean(self._time_windows)

    def close(self):
        self._stop_flag.set()
        self._comm_thread.join()


if __name__ == "__main__":
    driver = FeeTechDriver([1], "COM5")
    while True:
        print(driver.get_pos_and_vel())
        # print(driver.get_frequency())
        time.sleep(0.05)
