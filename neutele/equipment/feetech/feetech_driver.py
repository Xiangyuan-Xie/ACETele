import time
from collections import deque
from enum import Enum
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable, Deque, Dict, Sequence, Tuple

import numpy as np
from loguru import logger

from neutele.equipment.feetech.feetech_sdk.group_sync_read import GroupSyncRead
from neutele.equipment.feetech.feetech_sdk.group_sync_write import GroupSyncWrite
from neutele.equipment.feetech.feetech_sdk.hls import (
    HLS_GOAL_TORQUE_L,
    HLS_MODE,
    HLS_PRESENT_POSITION_L,
    HLS_PRESENT_SPEED_L,
    HLS_TORQUE_ENABLE,
    hls,
)
from neutele.equipment.feetech.feetech_sdk.port_handler import PortHandler
from neutele.equipment.feetech.feetech_sdk.scservo_def import COMM_SUCCESS


class Mode(Enum):
    Position = 0
    Velocity = 1
    Torque = 2


class TorqueEnable(Enum):
    Disable = 0
    Enable = 1


class FeeTechDriver:
    def __init__(self, ids: Sequence[int], port: str, baudrate: int = 1000000):
        self._ids = ids

        self._portHandler = PortHandler(port)
        assert self._portHandler.openPort(), f"Failed to open the port {port}"
        assert self._portHandler.setBaudRate(baudrate), f"Failed to set the baudrate {baudrate}"
        self._packetHandler = hls(self._portHandler)
        self._groupSyncReadHandler = GroupSyncRead(self._packetHandler, HLS_PRESENT_POSITION_L, 4)
        self._groupSyncWriteModeHandler = GroupSyncWrite(self._packetHandler, HLS_MODE, 1)
        self._groupSyncWriteTorqueEnableHandler = GroupSyncWrite(self._packetHandler, HLS_TORQUE_ENABLE, 1)
        self._groupSyncWriteGoalCurrentHandler = GroupSyncWrite(self._packetHandler, HLS_GOAL_TORQUE_L, 2)

        self._position: Dict[int, int] = {}
        self._velocity: Dict[int, int] = {}

        self._time_windows: Deque[float] = deque(maxlen=100)

        self._mode = {ft_id: Mode.Position for ft_id in self._ids}
        self._torque_enable = {ft_id: TorqueEnable.Enable for ft_id in self._ids}

        self._lock = Lock()
        self._stop_flag = Event()
        self._comm_task_queue: Queue[Callable] = Queue(maxsize=32)
        self._comm_thread = Thread(target=self._comm_worker, daemon=True)
        self._comm_thread.start()

        self.set_torque_enable(self._ids, [TorqueEnable.Disable] * len(self._ids))
        self.set_mode(self._ids, [Mode.Position for _ in self._ids])

    def _comm_worker(self):
        while not self._stop_flag.is_set():
            start_time = time.perf_counter()
            self._read_pos_and_vel()
            if not self._comm_task_queue.empty():
                self._comm_task_queue.get()()
            end_time = time.perf_counter()
            self._time_windows.append(end_time - start_time)

    def _read_pos_and_vel(self):
        position = {}
        velocity = {}

        for ft_id in self._ids:
            if not self._groupSyncReadHandler.addParam(ft_id):
                logger.error(f"[ID:{ft_id}] groupSyncRead addparam failed")

        self._groupSyncReadHandler.txRxPacket()
        # if comm_result != COMM_SUCCESS:
        #     logger.error(self._packetHandler.getTxRxResult(comm_result))

        for ft_id in self._ids:
            data_result, error = self._groupSyncReadHandler.isAvailable(ft_id, HLS_PRESENT_POSITION_L, 4)
            if not data_result:
                logger.error(f"[ID:{ft_id}] groupSyncRead getdata failed")
                continue
            if error != 0:
                logger.error(self._packetHandler.getRxPacketError(error))
                continue
            position[ft_id] = self._groupSyncReadHandler.getData(ft_id, HLS_PRESENT_POSITION_L, 2)
            velocity[ft_id] = self._packetHandler.scs_tohost(
                self._groupSyncReadHandler.getData(ft_id, HLS_PRESENT_SPEED_L, 2), 15
            )

        with self._lock:
            self._position = position
            self._velocity = velocity

        time.sleep(0.0005)

        self._groupSyncReadHandler.clearParam()

    def calibrate(self, ids: Sequence[int], home_poses: Sequence[int]) -> bool:
        assert len(ids) == len(home_poses), "ids and home_poses must have the same length."
        assert all(ft_id in self._ids for ft_id in ids), "some ids are not registered in FeeTechDriver."
        self.close()
        for ft_id, pose in zip(ids, home_poses):
            comm_result, error = self._packetHandler.reOfsCal(ft_id, pose)
            if comm_result != COMM_SUCCESS:
                logger.error(self._packetHandler.getTxRxResult(comm_result))
                return False
            if error != 0:
                logger.error(self._packetHandler.getRxPacketError(error))
                return False
        self.open()
        time.sleep(0.1)
        return True

    def set_torque_enable(self, ids: Sequence[int], enables: Sequence[TorqueEnable]):
        assert len(ids) == len(enables), "ids and enables must have the same length."

        index = [i for i, (ft_id, enable) in enumerate(zip(ids, enables)) if self._torque_enable[ft_id] != enable]
        target_ids = np.array([ids[i] for i in index])
        target_enables = np.array([enables[i] for i in index])

        if not self._groupSyncWriteTorqueEnableHandler.avail_flag.is_set():
            with self._groupSyncWriteTorqueEnableHandler.avail_condition:
                self._groupSyncWriteTorqueEnableHandler.avail_condition.wait()

        self._groupSyncWriteTorqueEnableHandler.avail_flag.clear()
        for ft_id, enable in zip(target_ids, target_enables):
            if not self._groupSyncWriteTorqueEnableHandler.addParam(ft_id, [enable.value]):
                logger.error(f"[ID:{ft_id}] groupSyncWriteTorqueEnable addparam failed")

        def task():
            self._groupSyncWriteTorqueEnableHandler.txPacket()
            # if comm_result != COMM_SUCCESS:
            #     logger.error(self._packetHandler.getTxRxResult(comm_result))
            self._groupSyncWriteTorqueEnableHandler.clearParam()

        self._comm_task_queue.put(task)

        for ft_id, enable in zip(target_ids, target_enables):
            self._torque_enable[ft_id] = enable

    def set_mode(self, ids: Sequence[int], modes: Sequence[Mode]):
        assert len(ids) == len(modes), "ids and modes must have the same length."

        index = [i for i, (ft_id, mode) in enumerate(zip(ids, modes)) if self._mode[ft_id] != mode]
        target_ids = np.array([ids[i] for i in index])
        target_modes = np.array([modes[i] for i in index])

        set_torque_enable_waiting_list = []
        for ft_id in target_ids:
            if self._torque_enable[ft_id] == TorqueEnable.Enable:
                set_torque_enable_waiting_list.append(ft_id)
        self.set_torque_enable(
            set_torque_enable_waiting_list, [TorqueEnable.Disable] * len(set_torque_enable_waiting_list)
        )

        if not self._groupSyncWriteModeHandler.avail_flag.is_set():
            with self._groupSyncWriteModeHandler.avail_condition:
                self._groupSyncWriteModeHandler.avail_condition.wait()

        self._groupSyncWriteModeHandler.avail_flag.clear()
        for ft_id, mode in zip(target_ids, target_modes):
            if not self._groupSyncWriteModeHandler.addParam(ft_id, [mode.value]):
                logger.error(f"[ID:{ft_id}] groupSyncWriteMode addparam failed")

        def task():
            self._groupSyncWriteModeHandler.txPacket()
            # if comm_result != COMM_SUCCESS:
            #     logger.error(self._packetHandler.getTxRxResult(comm_result))
            self._groupSyncWriteModeHandler.clearParam()

        self._comm_task_queue.put(task)

        for ft_id, mode in zip(target_ids, target_modes):
            self._mode[ft_id] = mode

    def set_current(self, ids: Sequence[int], goal_currents: Sequence[int]):
        assert len(ids) == len(goal_currents), "ids and currents must have the same length."

        set_mode_waiting_list = []
        for ft_id in ids:
            if self._mode[ft_id] != Mode.Torque:
                set_mode_waiting_list.append(ft_id)
        self.set_mode(set_mode_waiting_list, [Mode.Torque] * len(set_mode_waiting_list))

        set_torque_enable_waiting_list = []
        for ft_id in ids:
            if not self._torque_enable[ft_id] == TorqueEnable.Enable:
                set_torque_enable_waiting_list.append(ft_id)
        self.set_torque_enable(
            set_torque_enable_waiting_list, [TorqueEnable.Enable] * len(set_torque_enable_waiting_list)
        )

        if not self._groupSyncWriteGoalCurrentHandler.avail_flag.is_set():
            with self._groupSyncWriteGoalCurrentHandler.avail_condition:
                self._groupSyncWriteGoalCurrentHandler.avail_condition.wait()

        self._groupSyncWriteGoalCurrentHandler.avail_flag.clear()
        for ft_id, current in zip(ids, goal_currents):
            current = self._packetHandler.scs_toscs(current, 15)
            if not self._groupSyncWriteGoalCurrentHandler.addParam(
                ft_id, [self._packetHandler.scs_lobyte(current), self._packetHandler.scs_hibyte(current)]
            ):
                logger.error(f"[ID:{ft_id}] groupSyncWriteGoalCurrent addparam failed")

        def task():
            self._groupSyncWriteGoalCurrentHandler.txPacket()
            # if comm_result != COMM_SUCCESS:
            #     logger.error(self._packetHandler.getTxRxResult(comm_result))
            self._groupSyncWriteGoalCurrentHandler.clearParam()

        self._comm_task_queue.put(task)

    def get_pos_and_vel(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        while True:
            with self._lock:
                if len(self._position) == len(self._ids) and len(self._velocity) == len(self._ids):
                    return self._position, self._velocity
            time.sleep(0.01)

    def get_frequency(self) -> float:
        if len(self._time_windows) == 0:
            return 0.0
        return 1.0 / np.mean(self._time_windows)

    def open(self):
        self._stop_flag.clear()
        self._comm_thread = Thread(target=self._comm_worker, daemon=True)
        self._comm_thread.start()

    def close(self):
        self._stop_flag.set()
        self._comm_thread.join()


if __name__ == "__main__":
    driver = FeeTechDriver([0, 1, 2, 3], "COM5")
    while True:
        print(driver.get_pos_and_vel())
        # print(driver.get_frequency())
        time.sleep(0.05)
