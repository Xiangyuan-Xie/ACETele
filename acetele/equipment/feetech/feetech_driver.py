from __future__ import annotations

import time
from collections import deque
from enum import Enum
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Callable, Deque, Dict, Sequence, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

from acetele.equipment.feetech.feetech_sdk.group_sync_read import GroupSyncRead
from acetele.equipment.feetech.feetech_sdk.group_sync_write import GroupSyncWrite
from acetele.equipment.feetech.feetech_sdk.hls import (
    HLS_ACC,
    HLS_GOAL_TORQUE_L,
    HLS_MODE,
    HLS_PRESENT_CURRENT_L,
    HLS_PRESENT_POSITION_L,
    HLS_PRESENT_SPEED_L,
    HLS_TORQUE_ENABLE,
    hls,
)
from acetele.equipment.feetech.feetech_sdk.port_handler import PortHandler
from acetele.equipment.feetech.feetech_sdk.scservo_def import COMM_SUCCESS


class Mode(Enum):
    Position = 0
    Velocity = 1
    Torque = 2


class TorqueEnable(Enum):
    Disable = 0
    Enable = 1


class FeeTechCalibrationError(RuntimeError):
    pass


class FeeTechStateTimeoutError(RuntimeError):
    pass


class FeeTechDriver:
    def __init__(
        self,
        ids: Sequence[int],
        port: str,
        baudrate: int = 1000000,
    ):
        self._ids = ids
        self._id_to_index = {int(ft_id): index for index, ft_id in enumerate(self._ids)}

        logger.info(f"FeeTechDriver initializing with port: {port}, baudrate: {baudrate}")

        self._portHandler = PortHandler(port)
        assert self._portHandler.openPort(), f"Failed to open the port {port}"
        assert self._portHandler.setBaudRate(baudrate), f"Failed to set the baudrate {baudrate}"
        self._packetHandler = hls(self._portHandler)
        self._groupSyncReadHandler = GroupSyncRead(self._packetHandler, HLS_PRESENT_POSITION_L, 15)
        self._groupSyncWriteModeHandler = GroupSyncWrite(self._packetHandler, HLS_MODE, 1)
        self._groupSyncWriteTorqueEnableHandler = GroupSyncWrite(self._packetHandler, HLS_TORQUE_ENABLE, 1)
        self._groupSyncWriteGoalCurrentHandler = GroupSyncWrite(self._packetHandler, HLS_GOAL_TORQUE_L, 2)
        self._groupSyncWriteGoalPositionProfileHandler = GroupSyncWrite(self._packetHandler, HLS_ACC, 7)

        self._position: Dict[int, int] = {}
        self._velocity: Dict[int, int] = {}
        self._current: Dict[int, int] = {}

        self._time_windows: Deque[float] = deque(maxlen=100)

        self._mode = {ft_id: Mode.Position for ft_id in self._ids}
        self._torque_enable = {ft_id: TorqueEnable.Enable for ft_id in self._ids}

        self._lock = Lock()
        self._stop_flag = Event()
        self._comm_task_queue: Queue[Callable] = Queue(maxsize=32)

        self._comm_thread = Thread(target=self._comm_worker, daemon=True)
        self._comm_thread.start()

        print("FeeTechDriver warmup...")
        for _ in tqdm(range(100)):
            self.get_state()
        self.set_torque_enable(self._ids, [TorqueEnable.Disable] * len(self._ids), force=True)
        self.set_mode(self._ids, [Mode.Position for _ in self._ids], force=True)

    def _comm_worker(self):
        while not self._stop_flag.is_set():
            start_time = time.perf_counter()
            self._read_state()
            try:
                task = self._comm_task_queue.get_nowait()
            except Empty:
                pass
            else:
                task()
            end_time = time.perf_counter()
            self._time_windows.append(end_time - start_time)
            time.sleep(0.001)

    def _write_sync_params(self, handler: GroupSyncWrite, params: Sequence[Tuple[int, Sequence[int]]], label: str):
        if not handler.avail_flag.is_set():
            with handler.avail_condition:
                handler.avail_condition.wait()

        handler.avail_flag.clear()
        for ft_id, data in params:
            if not handler.addParam(ft_id, list(data)):
                logger.error(f"[ID:{ft_id}] {label} addparam failed")
        handler.txPacket()
        handler.clearParam()

    def _read_state(self):
        position = {}
        velocity = {}
        current = {}

        for ft_id in self._ids:
            if not self._groupSyncReadHandler.addParam(ft_id):
                logger.error(f"[ID:{ft_id}] groupSyncRead addparam failed")

        self._groupSyncReadHandler.txRxPacket()
        # if comm_result != COMM_SUCCESS:
        #     logger.error(self._packetHandler.getTxRxResult(comm_result))

        for ft_id in self._ids:
            data_result, error = self._groupSyncReadHandler.isAvailable(ft_id, HLS_PRESENT_POSITION_L, 15)
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
            current[ft_id] = self._groupSyncReadHandler.getData(ft_id, HLS_PRESENT_CURRENT_L, 2)

        with self._lock:
            self._position = position
            self._velocity = velocity
            self._current = current

        self._groupSyncReadHandler.clearParam()

    def calibrate(self, ids: Sequence[int], home_poses: Sequence[int]) -> None:
        assert len(ids) == len(home_poses), "ids and home_poses must have the same length."
        assert all(ft_id in self._ids for ft_id in ids), "some ids are not registered in FeeTechDriver."
        calibrated_ids: list[int] = []
        self.close()
        try:
            for ft_id, pose in zip(ids, home_poses):
                comm_result, error = self._packetHandler.reOfsCal(ft_id, pose)
                if comm_result != COMM_SUCCESS:
                    error_message = self._packetHandler.getTxRxResult(comm_result)
                    logger.error(error_message)
                    raise FeeTechCalibrationError(
                        f"Failed to calibrate ID {ft_id} at pose {pose}: {error_message}. "
                        f"Calibration may be partially written; already calibrated IDs: {calibrated_ids}"
                    )
                if error != 0:
                    error_message = self._packetHandler.getRxPacketError(error)
                    logger.error(error_message)
                    raise FeeTechCalibrationError(
                        f"Failed to calibrate ID {ft_id} at pose {pose}: {error_message}. "
                        f"Calibration may be partially written; already calibrated IDs: {calibrated_ids}"
                    )
                calibrated_ids.append(ft_id)
        finally:
            self.open()
            time.sleep(0.1)

    def set_torque_enable(self, ids: Sequence[int], enables: Sequence[TorqueEnable], force: bool = False):
        assert len(ids) == len(enables), "ids and enables must have the same length."

        index = [
            i
            for i, (ft_id, enable) in enumerate(zip(ids, enables))
            if force or self._torque_enable.get(ft_id) != enable
        ]
        target_ids = np.array([ids[i] for i in index])
        target_enables = np.array([enables[i] for i in index])
        if len(target_ids) == 0:
            return

        params = [(int(ft_id), [enable.value]) for ft_id, enable in zip(target_ids, target_enables)]

        def task():
            self._write_sync_params(self._groupSyncWriteTorqueEnableHandler, params, "groupSyncWriteTorqueEnable")

        self._comm_task_queue.put(task)

        for ft_id, enable in zip(target_ids, target_enables):
            self._torque_enable[ft_id] = enable

    def set_mode(self, ids: Sequence[int], modes: Sequence[Mode], force: bool = False):
        assert len(ids) == len(modes), "ids and modes must have the same length."

        index = [i for i, (ft_id, mode) in enumerate(zip(ids, modes)) if force or self._mode[ft_id] != mode]
        target_ids = np.array([ids[i] for i in index])
        target_modes = np.array([modes[i] for i in index])
        if len(target_ids) == 0:
            return

        set_torque_enable_waiting_list = []
        for ft_id in target_ids:
            if self._torque_enable.get(ft_id) != TorqueEnable.Disable:
                set_torque_enable_waiting_list.append(ft_id)
        self.set_torque_enable(
            set_torque_enable_waiting_list, [TorqueEnable.Disable] * len(set_torque_enable_waiting_list)
        )

        params = [(int(ft_id), [mode.value]) for ft_id, mode in zip(target_ids, target_modes)]

        def task():
            self._write_sync_params(self._groupSyncWriteModeHandler, params, "groupSyncWriteMode")

        self._comm_task_queue.put(task)

        for ft_id, mode in zip(target_ids, target_modes):
            self._mode[ft_id] = mode

    def set_current(self, ids: Sequence[int], goal_currents_raw: Sequence[int]):
        assert len(ids) == len(goal_currents_raw), "ids and currents must have the same length."
        if len(ids) == 0:
            return

        set_mode_waiting_list = []
        for ft_id in ids:
            if self._mode.get(ft_id) != Mode.Torque:
                set_mode_waiting_list.append(ft_id)
        self.set_mode(set_mode_waiting_list, [Mode.Torque] * len(set_mode_waiting_list))

        set_torque_enable_waiting_list = []
        for ft_id in ids:
            if not self._torque_enable.get(ft_id) == TorqueEnable.Enable:
                set_torque_enable_waiting_list.append(ft_id)
        self.set_torque_enable(
            set_torque_enable_waiting_list, [TorqueEnable.Enable] * len(set_torque_enable_waiting_list)
        )

        params = []
        for ft_id, current in zip(ids, goal_currents_raw):
            current = self._packetHandler.scs_toscs(current, 15)
            params.append((ft_id, [self._packetHandler.scs_lobyte(current), self._packetHandler.scs_hibyte(current)]))

        def task():
            self._write_sync_params(self._groupSyncWriteGoalCurrentHandler, params, "groupSyncWriteGoalCurrent")

        self._comm_task_queue.put(task)

    def set_position(
        self,
        ids: Sequence[int],
        goal_positions_raw: Sequence[int],
        currents_raw: Sequence[int] | None = None,
        velocities_raw: Sequence[int] | None = None,
        accelerations_raw: Sequence[int] | None = None,
    ):
        assert len(ids) == len(goal_positions_raw), "ids and positions must have the same length."
        if len(ids) == 0:
            return
        ids = np.asarray(ids)
        assert (
            velocities_raw is not None and accelerations_raw is not None and currents_raw is not None
        ), "velocities_raw, accelerations_raw, and currents_raw are required."
        assert len(ids) == len(velocities_raw), "ids and velocities must have the same length."
        velocities_array = np.asarray(velocities_raw, dtype=int)
        assert len(ids) == len(accelerations_raw), "ids and accelerations must have the same length."
        accelerations_array = np.asarray(accelerations_raw, dtype=int)
        assert len(ids) == len(currents_raw), "ids and currents must have the same length."
        currents_array = np.asarray(currents_raw, dtype=int)
        assert np.all(velocities_array >= 0), "velocities must be non-negative."
        assert np.all(accelerations_array >= 0), "accelerations must be non-negative."
        assert np.all(currents_array >= 0), "currents must be non-negative."
        assert np.all(velocities_array <= 32767), "velocities must fit in 15 bits."
        assert np.all(accelerations_array <= 255), "accelerations must fit in one byte."
        assert np.all(currents_array <= 32767), "currents must fit in 15 bits."
        current_positions_dict, _, _ = self.get_state()
        current_positions_array = np.array([current_positions_dict[int(ft_id)] for ft_id in ids])
        goal_positions_array = np.asarray(goal_positions_raw).copy()
        goal_positions_array += np.round((current_positions_array - goal_positions_array) / 4096).astype(int) * 4096

        set_mode_waiting_list = []
        for ft_id in ids:
            if self._mode.get(ft_id) != Mode.Position:
                set_mode_waiting_list.append(ft_id)
        self.set_mode(set_mode_waiting_list, [Mode.Position] * len(set_mode_waiting_list))

        set_torque_enable_waiting_list = []
        for ft_id in ids:
            if not self._torque_enable.get(ft_id) == TorqueEnable.Enable:
                set_torque_enable_waiting_list.append(ft_id)
        self.set_torque_enable(
            set_torque_enable_waiting_list, [TorqueEnable.Enable] * len(set_torque_enable_waiting_list)
        )

        profile_params = []
        for ft_id, position, velocity, acceleration, current in zip(
            ids,
            goal_positions_array,
            velocities_array,
            accelerations_array,
            currents_array,
        ):
            position = self._packetHandler.scs_toscs(position, 15)
            velocity = self._packetHandler.scs_toscs(int(velocity), 15)
            profile_params.append(
                (
                    ft_id,
                    [
                        int(acceleration),
                        self._packetHandler.scs_lobyte(position),
                        self._packetHandler.scs_hibyte(position),
                        self._packetHandler.scs_lobyte(int(current)),
                        self._packetHandler.scs_hibyte(int(current)),
                        self._packetHandler.scs_lobyte(velocity),
                        self._packetHandler.scs_hibyte(velocity),
                    ],
                )
            )

        def task():
            self._write_sync_params(
                self._groupSyncWriteGoalPositionProfileHandler,
                profile_params,
                "groupSyncWriteGoalPositionProfile",
            )

        self._comm_task_queue.put(task)

    def get_state(self, timeout: float = 1.0) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                if len(self._ids) == len(self._position) == len(self._velocity) == len(self._current):
                    return self._position, self._velocity, self._current
                missing_ids = [
                    ft_id
                    for ft_id in self._ids
                    if ft_id not in self._position or ft_id not in self._velocity or ft_id not in self._current
                ]
            if deadline is not None and time.monotonic() >= deadline:
                raise FeeTechStateTimeoutError(
                    f"Timed out waiting for complete FeeTech state; missing IDs: {missing_ids}"
                )
            time.sleep(0.001)

    def get_frequency(self) -> float:
        if len(self._time_windows) == 0:
            return 0.0
        return 1.0 / np.mean(self._time_windows)

    def open(self):
        self._stop_flag.clear()
        self._mode = {ft_id: None for ft_id in self._ids}
        self._torque_enable = {ft_id: None for ft_id in self._ids}
        self._comm_thread = Thread(target=self._comm_worker, daemon=True)
        self._comm_thread.start()

    def close(self):
        self._stop_flag.set()
        if self._comm_thread.is_alive():
            self._comm_thread.join()


if __name__ == "__main__":
    driver = FeeTechDriver([0, 1, 2, 3, 4], "/dev/ttyUSB0")
    while True:
        print(driver.get_state())
        # print(driver.get_frequency())
        time.sleep(0.05)
