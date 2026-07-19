from __future__ import annotations

import operator
import time
from collections import deque
from enum import Enum
from queue import Empty, Full, Queue
from threading import Event, Lock, RLock, Thread
from typing import Callable, Deque, Dict, Optional, Sequence, Tuple, Type, TypeVar

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
from acetele.equipment.feetech.feetech_sdk.scservo_def import COMM_SUCCESS, MAX_ID
from acetele.equipment.joint_device import TorqueEnable
from acetele.utils.joint import normalize_joint_ids


class Mode(Enum):
    Position = 0
    Velocity = 1
    Torque = 2


class FeeTechCalibrationError(RuntimeError):
    pass


class FeeTechStateTimeoutError(RuntimeError):
    pass


class FeeTechCommandDispatchError(RuntimeError):
    pass


class FeeTechCommandTimeoutError(FeeTechCommandDispatchError):
    pass


class _FeeTechTaskExecutionError(FeeTechCommandDispatchError):
    """A queued task ran and failed after worker-side cache handling."""


HLS_STATE_READ_LENGTH = HLS_PRESENT_CURRENT_L + 2 - HLS_PRESENT_POSITION_L
FEETECH_COMM_TIMEOUT = 1.0
FEETECH_COMM_QUEUE_SIZE = 32
FEETECH_MAX_SERVO_ID = MAX_ID
FEETECH_SIGNED_15_BIT_MAX = 32767
EnumT = TypeVar("EnumT", bound=Enum)


def normalize_feetech_servo_ids(
    ids: Sequence[int],
    *,
    field_name: str,
) -> Tuple[int, ...]:
    normalized = normalize_joint_ids(ids, field_name=field_name)
    if any(not 0 <= ft_id <= FEETECH_MAX_SERVO_ID for ft_id in normalized):
        raise ValueError(
            f"{field_name} must be between 0 and {FEETECH_MAX_SERVO_ID}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must be unique")
    return normalized


class _CacheCommitGuard:
    """Prevent a timed-out worker task from committing stale cache state."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._valid = True

    def commit(self, callback: Callable[[], None]) -> bool:
        with self._lock:
            if not self._valid:
                return False
            callback()
            return True

    def invalidate(self) -> None:
        with self._lock:
            self._valid = False


class FeeTechDriver:
    def __init__(
        self,
        ids: Sequence[int],
        port: str,
        baudrate: int = 1000000,
    ):
        self._ids = self._normalize_servo_ids(ids, field_name="driver ids")
        self._lifecycle_lock = RLock()

        logger.info(f"FeeTechDriver initializing with port: {port}, baudrate: {baudrate}")

        self._portHandler = PortHandler(port)
        try:
            if not self._portHandler.openPort():
                raise FeeTechCommandDispatchError(f"Failed to open the port {port}")
            if not self._portHandler.setBaudRate(baudrate):
                raise FeeTechCommandDispatchError(
                    f"Failed to set baudrate {baudrate} on port {port}"
                )
            self._packetHandler = hls(self._portHandler)
            self._groupSyncReadHandler = GroupSyncRead(
                self._packetHandler,
                HLS_PRESENT_POSITION_L,
                HLS_STATE_READ_LENGTH,
            )
            self._groupSyncWriteModeHandler = GroupSyncWrite(
                self._packetHandler,
                HLS_MODE,
                1,
            )
            self._groupSyncWriteTorqueEnableHandler = GroupSyncWrite(
                self._packetHandler,
                HLS_TORQUE_ENABLE,
                1,
            )
            self._groupSyncWriteGoalCurrentHandler = GroupSyncWrite(
                self._packetHandler,
                HLS_GOAL_TORQUE_L,
                2,
            )
            self._groupSyncWriteGoalPositionProfileHandler = GroupSyncWrite(
                self._packetHandler,
                HLS_ACC,
                7,
            )

            self._position: Dict[int, int] = {}
            self._velocity: Dict[int, int] = {}
            self._current: Dict[int, int] = {}

            self._time_windows: Deque[float] = deque(maxlen=100)

            # These caches describe writes confirmed by the SDK, not queued intent.
            self._mode: Dict[int, Optional[Mode]] = {
                ft_id: None for ft_id in self._ids
            }
            self._torque_enable: Dict[int, Optional[TorqueEnable]] = {
                ft_id: None for ft_id in self._ids
            }

            self._lock = Lock()
            self._stop_flag = Event()
            self._comm_fault: Optional[BaseException] = None
            self._comm_task_queue: Queue[Callable] = Queue(
                maxsize=FEETECH_COMM_QUEUE_SIZE
            )

            self._comm_thread = Thread(target=self._comm_worker, daemon=True)
            self._comm_thread.start()

            print("FeeTechDriver warmup...")
            for _ in tqdm(range(100)):
                self.get_state()
            self._submit_mode_transaction(
                self._ids,
                [Mode.Position for _ in self._ids],
                force=True,
                wait=True,
            )
        except BaseException as initialization_error:
            cleanup_error = self._cleanup_failed_initialization()
            if cleanup_error is not None:
                raise initialization_error from cleanup_error
            raise

    def _cleanup_failed_initialization(
        self,
        timeout: float = FEETECH_COMM_TIMEOUT,
    ) -> Optional[BaseException]:
        """Best-effort cleanup for objects that never completed construction."""
        first_error: Optional[BaseException] = None

        stop_flag = getattr(self, "_stop_flag", None)
        if stop_flag is not None:
            stop_flag.set()

        port_handler = getattr(self, "_portHandler", None)
        serial_port = getattr(port_handler, "ser", None)
        if serial_port is not None:
            for method_name in ("cancel_read", "cancel_write"):
                method = getattr(serial_port, method_name, None)
                if not callable(method):
                    continue
                try:
                    method()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc

        if port_handler is not None and getattr(port_handler, "is_open", False):
            try:
                port_handler.closePort()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

        comm_thread = getattr(self, "_comm_thread", None)
        if comm_thread is not None:
            try:
                if comm_thread.is_alive():
                    comm_thread.join(timeout=timeout)
                if comm_thread.is_alive() and first_error is None:
                    first_error = FeeTechCommandTimeoutError(
                        "Timed out waiting for FeeTech communication thread to close "
                        "after initialization failed"
                    )
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

        lock = getattr(self, "_lock", None)
        if lock is not None:
            try:
                with lock:
                    if hasattr(self, "_position"):
                        self._position = {}
                        self._velocity = {}
                        self._current = {}
                    if hasattr(self, "_mode"):
                        self._mode = {ft_id: None for ft_id in self._ids}
                    if hasattr(self, "_torque_enable"):
                        self._torque_enable = {
                            ft_id: None for ft_id in self._ids
                        }
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        return first_error

    def _comm_worker(self):
        while not self._stop_flag.is_set():
            start_time = time.perf_counter()
            try:
                self._read_state()
            except Exception as exc:
                if self._stop_flag.is_set():
                    break
                self._record_comm_fault(exc)
                logger.exception("FeeTech communication worker stopped after a state read error")
                break

            if self._stop_flag.is_set():
                break
            try:
                task = self._comm_task_queue.get_nowait()
            except Empty:
                pass
            else:
                try:
                    task()
                except FeeTechCommandDispatchError:
                    self._invalidate_control_caches()
                    logger.exception("FeeTech asynchronous write failed; continuing communication")
                except Exception as exc:
                    if self._stop_flag.is_set():
                        break
                    self._record_comm_fault(exc)
                    logger.exception("FeeTech communication worker stopped after an unexpected task error")
                    break
            end_time = time.perf_counter()
            self._time_windows.append(end_time - start_time)
            time.sleep(0.001)

    def _write_sync_params(
        self,
        handler: GroupSyncWrite,
        params: Sequence[Tuple[int, Sequence[int]]],
        label: str,
        timeout: float = FEETECH_COMM_TIMEOUT,
    ) -> None:
        timeout = self._validate_timeout(timeout)
        if not handler.avail_flag.wait(timeout):
            raise FeeTechCommandTimeoutError(
                f"Timed out waiting for {label} handler to become available"
            )

        handler.avail_flag.clear()
        try:
            for ft_id, data in params:
                if not handler.addParam(ft_id, list(data)):
                    raise FeeTechCommandDispatchError(
                        f"[ID:{ft_id}] {label} addParam failed; sync write was not sent"
                    )
            comm_result = handler.txPacket()
            if comm_result != COMM_SUCCESS:
                error_message = self._packetHandler.getTxRxResult(comm_result)
                raise FeeTechCommandDispatchError(
                    f"{label} failed with communication result {comm_result}: {error_message}"
                )
        finally:
            handler.clearParam()

    def _read_state(self):
        position = {}
        velocity = {}
        current = {}

        try:
            for ft_id in self._ids:
                if not self._groupSyncReadHandler.addParam(ft_id):
                    raise FeeTechCommandDispatchError(
                        f"[ID:{ft_id}] groupSyncRead addParam failed"
                    )

            comm_result = self._groupSyncReadHandler.txRxPacket()
            if comm_result != COMM_SUCCESS:
                error_message = self._packetHandler.getTxRxResult(comm_result)
                raise FeeTechCommandDispatchError(
                    f"groupSyncRead failed with communication result {comm_result}: "
                    f"{error_message}"
                )

            for ft_id in self._ids:
                data_result, error = self._groupSyncReadHandler.isAvailable(
                    ft_id, HLS_PRESENT_POSITION_L, HLS_STATE_READ_LENGTH
                )
                if not data_result:
                    logger.error(f"[ID:{ft_id}] groupSyncRead getdata failed")
                    continue
                if error != 0:
                    logger.error(self._packetHandler.getRxPacketError(error))
                    continue

                position[ft_id] = self._groupSyncReadHandler.getData(
                    ft_id, HLS_PRESENT_POSITION_L, 2
                )
                velocity[ft_id] = self._packetHandler.scs_tohost(
                    self._groupSyncReadHandler.getData(
                        ft_id, HLS_PRESENT_SPEED_L, 2
                    ),
                    15,
                )
                current[ft_id] = self._groupSyncReadHandler.getData(
                    ft_id, HLS_PRESENT_CURRENT_L, 2
                )

            with self._lock:
                self._position = position
                self._velocity = velocity
                self._current = current
        finally:
            self._groupSyncReadHandler.clearParam()

    def calibrate(self, ids: Sequence[int], home_poses: Sequence[int]) -> None:
        target_ids = self._normalize_target_ids(ids, field_name="calibration ids")
        target_home_poses = self._normalize_integer_sequence(
            home_poses,
            field_name="calibration home poses",
            minimum=-FEETECH_SIGNED_15_BIT_MAX,
            maximum=FEETECH_SIGNED_15_BIT_MAX,
        )
        if len(target_ids) != len(target_home_poses):
            raise ValueError("ids and home_poses must have the same length")
        if not target_ids:
            return
        with self._lifecycle_lock:
            self._calibrate(target_ids, target_home_poses)

    def _calibrate(
        self,
        target_ids: Tuple[int, ...],
        target_home_poses: Tuple[int, ...],
    ) -> None:
        try:
            self.set_torque_enable(
                target_ids,
                [TorqueEnable.Disable] * len(target_ids),
                force=True,
                wait=True,
            )
            self._pause_comm_worker()
        except Exception:
            self._invalidate_control_caches()
            raise

        calibrated_ids: list[int] = []
        calibration_error: Optional[Exception] = None
        resume_error: Optional[Exception] = None
        try:
            for ft_id, pose in zip(target_ids, target_home_poses):
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
        except Exception as exc:
            calibration_error = exc
        finally:
            try:
                self.open()
                time.sleep(0.1)
            except Exception as exc:
                self._invalidate_control_caches()
                resume_error = exc

        if calibration_error is not None:
            if resume_error is not None:
                raise FeeTechCalibrationError(
                    f"{calibration_error}; additionally failed to resume FeeTech communication: "
                    f"{resume_error}"
                ) from calibration_error
            raise calibration_error
        if resume_error is not None:
            raise FeeTechCommandDispatchError(
                f"Calibration completed but failed to resume FeeTech communication: {resume_error}"
            ) from resume_error

    @staticmethod
    def _validate_timeout(timeout: float) -> float:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not np.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number")
        return float(timeout)

    @staticmethod
    def _normalize_integer_sequence(
        values: Sequence[int],
        *,
        field_name: str,
        minimum: int,
        maximum: int,
    ) -> Tuple[int, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError(
                f"{field_name} must be a one-dimensional sequence of integers"
            )
        if isinstance(values, np.ndarray) and values.ndim != 1:
            raise ValueError(
                f"{field_name} must be a one-dimensional sequence of integers"
            )
        try:
            items = tuple(values)
        except TypeError as exc:
            raise ValueError(
                f"{field_name} must be a one-dimensional sequence of integers"
            ) from exc

        normalized = []
        for index, value in enumerate(items):
            if isinstance(value, (bool, np.bool_, np.ndarray)):
                raise ValueError(f"{field_name}[{index}] must be an integer")
            try:
                normalized_value = int(operator.index(value))
            except TypeError as exc:
                raise ValueError(f"{field_name}[{index}] must be an integer") from exc
            if not minimum <= normalized_value <= maximum:
                raise ValueError(
                    f"{field_name} must be between {minimum} and {maximum}"
                )
            normalized.append(normalized_value)
        return tuple(normalized)

    @staticmethod
    def _normalize_servo_ids(
        ids: Sequence[int],
        *,
        field_name: str,
    ) -> Tuple[int, ...]:
        return normalize_feetech_servo_ids(ids, field_name=field_name)

    def _normalize_target_ids(
        self,
        ids: Sequence[int],
        *,
        field_name: str = "ids",
    ) -> Tuple[int, ...]:
        normalized = self._normalize_servo_ids(ids, field_name=field_name)
        unknown = tuple(ft_id for ft_id in normalized if ft_id not in self._ids)
        if unknown:
            raise ValueError(f"{field_name} contains unregistered servo IDs: {unknown}")
        return normalized

    @staticmethod
    def _normalize_enum_sequence(
        values: Sequence[EnumT],
        *,
        enum_type: Type[EnumT],
        field_name: str,
    ) -> Tuple[EnumT, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{field_name} must be a one-dimensional sequence")
        if isinstance(values, np.ndarray) and values.ndim != 1:
            raise ValueError(f"{field_name} must be a one-dimensional sequence")
        try:
            items = tuple(values)
        except TypeError as exc:
            raise ValueError(f"{field_name} must be a one-dimensional sequence") from exc
        if any(not isinstance(value, enum_type) for value in items):
            raise ValueError(
                f"{field_name} must contain only {enum_type.__name__} values"
            )
        return items

    def _submit_comm_task(
        self,
        task: Callable[[], None],
        *,
        wait: bool = False,
        timeout: float = 1.0,
        label: str = "communication task",
        on_abandon: Optional[Callable[[], None]] = None,
    ) -> None:
        timeout = self._validate_timeout(timeout)

        if not wait:
            self._enqueue_comm_task(task, timeout=timeout, label=label)
            return

        completed = Event()
        task_state_lock = Lock()
        task_started = False
        task_completed = False
        task_cancelled = False
        task_abandoned = False
        task_error: Optional[BaseException] = None

        def abandon_locked() -> None:
            nonlocal task_abandoned, task_cancelled
            if task_completed or task_abandoned:
                return
            task_abandoned = True
            if not task_started:
                task_cancelled = True
            if on_abandon is not None:
                on_abandon()

        def tracked_task() -> None:
            nonlocal task_completed, task_error, task_started
            with task_state_lock:
                if task_cancelled:
                    task_completed = True
                    completed.set()
                    return
                task_started = True
            try:
                task()
            except BaseException as exc:
                with task_state_lock:
                    task_error = exc
                raise
            finally:
                with task_state_lock:
                    task_completed = True
                    completed.set()

        deadline = time.monotonic() + timeout
        try:
            self._enqueue_comm_task(tracked_task, timeout=timeout, label=label)
        except BaseException:
            with task_state_lock:
                abandon_locked()
            raise

        while True:
            with task_state_lock:
                if task_completed:
                    completed_error = task_error
                    break
            if not self._comm_thread.is_alive():
                with task_state_lock:
                    if task_completed:
                        completed_error = task_error
                        break
                    abandon_locked()
                raise FeeTechCommandDispatchError(
                    f"Cannot complete {label}: FeeTech communication thread stopped"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with task_state_lock:
                    if task_completed:
                        completed_error = task_error
                        break
                    abandon_locked()
                raise FeeTechCommandTimeoutError(
                    f"Timed out waiting for {label} to execute"
                )
            completed.wait(min(remaining, 0.01))

        if completed_error is not None:
            raise _FeeTechTaskExecutionError(
                f"Failed to execute {label}"
            ) from completed_error

    def _enqueue_comm_task(
        self,
        task: Callable[[], None],
        *,
        timeout: float,
        label: str,
    ) -> None:
        if not self._lifecycle_lock.acquire(timeout=timeout):
            raise FeeTechCommandTimeoutError(
                f"Timed out waiting to dispatch {label} during a driver lifecycle operation"
            )
        try:
            self._ensure_comm_available(label)
            try:
                self._comm_task_queue.put(task, timeout=timeout)
            except Full as exc:
                raise FeeTechCommandDispatchError(
                    f"Cannot dispatch {label}: FeeTech communication queue is full"
                ) from exc
        finally:
            self._lifecycle_lock.release()

    def _ensure_comm_available(self, label: str) -> None:
        with self._lock:
            comm_fault = self._comm_fault
        if comm_fault is not None:
            raise FeeTechCommandDispatchError(
                f"Cannot dispatch {label}: FeeTech communication worker is faulted"
            ) from comm_fault
        if self._stop_flag.is_set() or not self._comm_thread.is_alive():
            raise FeeTechCommandDispatchError(
                f"Cannot dispatch {label}: FeeTech communication thread is not running"
            )

    def _record_comm_fault(self, error: BaseException) -> None:
        self._stop_flag.set()
        with self._lock:
            if self._comm_fault is None:
                self._comm_fault = error
            self._position = {}
            self._velocity = {}
            self._current = {}
        self._invalidate_control_caches()

    def _update_mode_cache(
        self,
        ids: Sequence[int],
        modes: Sequence[Mode],
    ) -> None:
        with self._lock:
            for ft_id, mode in zip(ids, modes):
                self._mode[int(ft_id)] = mode

    def _update_torque_enable_cache(
        self,
        ids: Sequence[int],
        enables: Sequence[TorqueEnable],
    ) -> None:
        with self._lock:
            for ft_id, enable in zip(ids, enables):
                self._torque_enable[int(ft_id)] = enable

    def _invalidate_torque_enable_cache(self, ids: Sequence[int]) -> None:
        with self._lock:
            for ft_id in ids:
                self._torque_enable[int(ft_id)] = None

    def _submit_control_transaction(
        self,
        transaction: Callable[[], None],
        *,
        wait: bool = False,
        timeout: float = FEETECH_COMM_TIMEOUT,
        label: str,
        on_abandon: Optional[Callable[[], None]] = None,
    ) -> None:
        def guarded_transaction() -> None:
            try:
                transaction()
            except FeeTechCommandDispatchError:
                self._invalidate_control_caches()
                raise
            except BaseException as exc:
                self._record_comm_fault(exc)
                raise

        self._submit_comm_task(
            guarded_transaction,
            wait=wait,
            timeout=timeout,
            label=label,
            on_abandon=on_abandon,
        )

    def _execute_torque_enable_transaction(
        self,
        ids: Sequence[int],
        enables: Sequence[TorqueEnable],
        *,
        force: bool = False,
        update_cache: bool = True,
    ) -> None:
        with self._lock:
            current_modes = {
                int(ft_id): self._mode.get(int(ft_id)) for ft_id in ids
            }
            current_enables = {
                int(ft_id): self._torque_enable.get(int(ft_id)) for ft_id in ids
            }

        unknown_mode_ids = [
            int(ft_id)
            for ft_id, enable in zip(ids, enables)
            if enable == TorqueEnable.Enable
            and current_modes[int(ft_id)] is None
        ]
        if unknown_mode_ids:
            raise FeeTechCommandDispatchError(
                "Cannot enable torque while the confirmed mode is unknown for IDs: "
                f"{unknown_mode_ids}"
            )

        targets = [
            (int(ft_id), enable)
            for ft_id, enable in zip(ids, enables)
            if force or current_enables[int(ft_id)] != enable
        ]
        if not targets:
            return

        params = [(ft_id, [enable.value]) for ft_id, enable in targets]
        self._write_sync_params(
            self._groupSyncWriteTorqueEnableHandler,
            params,
            "groupSyncWriteTorqueEnable",
        )
        if update_cache:
            self._update_torque_enable_cache(
                [ft_id for ft_id, _ in targets],
                [enable for _, enable in targets],
            )

    def _execute_mode_transaction(
        self,
        ids: Sequence[int],
        modes: Sequence[Mode],
        *,
        force: bool = False,
    ) -> None:
        with self._lock:
            current_modes = {
                int(ft_id): self._mode.get(int(ft_id)) for ft_id in ids
            }
        targets = [
            (int(ft_id), mode)
            for ft_id, mode in zip(ids, modes)
            if force or current_modes[int(ft_id)] != mode
        ]
        if not targets:
            return

        target_ids = [ft_id for ft_id, _ in targets]
        self._execute_torque_enable_transaction(
            target_ids,
            [TorqueEnable.Disable] * len(target_ids),
        )
        self._write_sync_params(
            self._groupSyncWriteModeHandler,
            [(ft_id, [mode.value]) for ft_id, mode in targets],
            "groupSyncWriteMode",
        )
        self._update_mode_cache(
            target_ids,
            [mode for _, mode in targets],
        )

    def set_torque_enable(
        self,
        ids: Sequence[int],
        enables: Sequence[TorqueEnable],
        force: bool = False,
        *,
        wait: bool = False,
        timeout: float = 1.0,
    ) -> None:
        """Queue a torque command, optionally waiting for the local sync-write task.

        Waiting confirms that GroupSyncWrite returned COMM_SUCCESS; the protocol does
        not provide per-servo acknowledgement for this broadcast write.
        """
        if type(force) is not bool or type(wait) is not bool:
            raise ValueError("force and wait must be booleans")
        target_ids = self._normalize_target_ids(ids)
        target_enables = self._normalize_enum_sequence(
            enables,
            enum_type=TorqueEnable,
            field_name="enables",
        )
        if len(target_ids) != len(target_enables):
            raise ValueError("ids and enables must have the same length")
        if not target_ids:
            return

        cache_commit_guard = _CacheCommitGuard() if wait else None

        def abandon_transaction() -> None:
            if cache_commit_guard is not None:
                cache_commit_guard.invalidate()
            self._invalidate_torque_enable_cache(target_ids)

        def transaction() -> None:
            self._execute_torque_enable_transaction(
                target_ids,
                target_enables,
                force=force,
                update_cache=not wait,
            )
            if cache_commit_guard is not None:
                cache_commit_guard.commit(
                    lambda: self._update_torque_enable_cache(
                        target_ids,
                        target_enables,
                    )
                )

        self._submit_control_transaction(
            transaction,
            wait=wait,
            timeout=timeout,
            label="torque-enable sync write",
            on_abandon=abandon_transaction if wait else None,
        )

    def set_mode(self, ids: Sequence[int], modes: Sequence[Mode], force: bool = False):
        target_ids = self._normalize_target_ids(ids)
        target_modes = self._normalize_enum_sequence(
            modes,
            enum_type=Mode,
            field_name="modes",
        )
        if len(target_ids) != len(target_modes):
            raise ValueError("ids and modes must have the same length")
        if not target_ids:
            return
        if type(force) is not bool:
            raise ValueError("force must be a boolean")
        self._submit_mode_transaction(target_ids, target_modes, force=force)

    def _submit_mode_transaction(
        self,
        ids: Sequence[int],
        modes: Sequence[Mode],
        *,
        force: bool,
        wait: bool = False,
        timeout: float = FEETECH_COMM_TIMEOUT,
    ) -> None:
        def transaction() -> None:
            self._execute_mode_transaction(
                ids,
                modes,
                force=force,
            )

        self._submit_control_transaction(
            transaction,
            wait=wait,
            timeout=timeout,
            label="mode transaction",
        )

    def set_current(self, ids: Sequence[int], goal_currents_raw: Sequence[int]):
        target_ids = self._normalize_target_ids(ids)
        normalized_currents = self._normalize_integer_sequence(
            goal_currents_raw,
            field_name="goal currents",
            minimum=-FEETECH_SIGNED_15_BIT_MAX,
            maximum=FEETECH_SIGNED_15_BIT_MAX,
        )
        if len(target_ids) != len(normalized_currents):
            raise ValueError("ids and currents must have the same length")
        if not target_ids:
            return

        params = []
        for ft_id, current in zip(target_ids, normalized_currents):
            current = self._packetHandler.scs_toscs(current, 15)
            params.append((ft_id, [self._packetHandler.scs_lobyte(current), self._packetHandler.scs_hibyte(current)]))

        def transaction() -> None:
            self._execute_mode_transaction(
                target_ids,
                [Mode.Torque] * len(target_ids),
            )
            self._execute_torque_enable_transaction(
                target_ids,
                [TorqueEnable.Enable] * len(target_ids),
            )
            self._write_sync_params(
                self._groupSyncWriteGoalCurrentHandler,
                params,
                "groupSyncWriteGoalCurrent",
            )

        self._submit_control_transaction(
            transaction,
            label="goal-current transaction",
        )

    def set_position(
        self,
        ids: Sequence[int],
        goal_positions_raw: Sequence[int],
        currents_raw: Sequence[int] | None = None,
        velocities_raw: Sequence[int] | None = None,
        accelerations_raw: Sequence[int] | None = None,
    ):
        target_ids = self._normalize_target_ids(ids)
        goal_positions = self._normalize_integer_sequence(
            goal_positions_raw,
            field_name="goal positions",
            minimum=-FEETECH_SIGNED_15_BIT_MAX,
            maximum=FEETECH_SIGNED_15_BIT_MAX,
        )
        if len(target_ids) != len(goal_positions):
            raise ValueError("ids and positions must have the same length")
        if not target_ids:
            return
        if velocities_raw is None or accelerations_raw is None or currents_raw is None:
            raise ValueError("velocities_raw, accelerations_raw, and currents_raw are required")
        velocities = self._normalize_integer_sequence(
            velocities_raw,
            field_name="velocities",
            minimum=0,
            maximum=FEETECH_SIGNED_15_BIT_MAX,
        )
        if len(target_ids) != len(velocities):
            raise ValueError("ids and velocities must have the same length")
        accelerations = self._normalize_integer_sequence(
            accelerations_raw,
            field_name="accelerations",
            minimum=0,
            maximum=255,
        )
        if len(target_ids) != len(accelerations):
            raise ValueError("ids and accelerations must have the same length")
        currents = self._normalize_integer_sequence(
            currents_raw,
            field_name="currents",
            minimum=0,
            maximum=FEETECH_SIGNED_15_BIT_MAX,
        )
        if len(target_ids) != len(currents):
            raise ValueError("ids and currents must have the same length")
        current_positions_dict, _, _ = self.get_state()
        current_positions_array = np.array(
            [current_positions_dict[ft_id] for ft_id in target_ids],
            dtype=int,
        )
        goal_positions_array = np.asarray(goal_positions, dtype=int)
        goal_positions_array += np.round((current_positions_array - goal_positions_array) / 4096).astype(int) * 4096
        adjusted_positions = self._normalize_integer_sequence(
            goal_positions_array,
            field_name="adjusted goal positions",
            minimum=-FEETECH_SIGNED_15_BIT_MAX,
            maximum=FEETECH_SIGNED_15_BIT_MAX,
        )

        profile_params = []
        for ft_id, position, velocity, acceleration, current in zip(
            target_ids,
            adjusted_positions,
            velocities,
            accelerations,
            currents,
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

        def transaction() -> None:
            self._execute_mode_transaction(
                target_ids,
                [Mode.Position] * len(target_ids),
            )
            self._execute_torque_enable_transaction(
                target_ids,
                [TorqueEnable.Enable] * len(target_ids),
            )
            self._write_sync_params(
                self._groupSyncWriteGoalPositionProfileHandler,
                profile_params,
                "groupSyncWriteGoalPositionProfile",
            )

        self._submit_control_transaction(
            transaction,
            label="goal-position transaction",
        )

    def get_state(self, timeout: float = 1.0) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                comm_fault = self._comm_fault
                missing_ids = [
                    ft_id
                    for ft_id in self._ids
                    if ft_id not in self._position or ft_id not in self._velocity or ft_id not in self._current
                ]
                state_snapshot = None
                if not missing_ids:
                    state_snapshot = (
                        dict(self._position),
                        dict(self._velocity),
                        dict(self._current),
                    )
            if comm_fault is not None:
                raise FeeTechCommandDispatchError(
                    "Cannot read FeeTech state: communication worker is faulted"
                ) from comm_fault
            if self._stop_flag.is_set() or not self._comm_thread.is_alive():
                raise FeeTechCommandDispatchError(
                    "Cannot read FeeTech state: communication thread is not running"
                )
            if state_snapshot is not None:
                return state_snapshot
            if deadline is not None and time.monotonic() >= deadline:
                raise FeeTechStateTimeoutError(
                    f"Timed out waiting for complete FeeTech state; missing IDs: {missing_ids}"
                )
            time.sleep(0.001)

    def get_frequency(self) -> float:
        if len(self._time_windows) == 0:
            return 0.0
        return 1.0 / np.mean(self._time_windows)

    def _invalidate_control_caches(self) -> None:
        with self._lock:
            self._mode = {ft_id: None for ft_id in self._ids}
            self._torque_enable = {ft_id: None for ft_id in self._ids}

    def _pause_comm_worker(self, timeout: float = FEETECH_COMM_TIMEOUT) -> None:
        timeout = self._validate_timeout(timeout)
        self._stop_flag.set()
        if self._comm_thread.is_alive():
            self._comm_thread.join(timeout=timeout)
        if self._comm_thread.is_alive():
            self._invalidate_control_caches()
            raise FeeTechCommandTimeoutError(
                "Timed out waiting for FeeTech communication thread to pause"
            )

    def open(self) -> None:
        with self._lifecycle_lock:
            self._open()

    def _open(self) -> None:
        if self._comm_thread.is_alive():
            with self._lock:
                comm_fault = self._comm_fault
            if comm_fault is None and not self._stop_flag.is_set():
                return
            self._stop_flag.set()
            self._comm_thread.join(timeout=FEETECH_COMM_TIMEOUT)
            if self._comm_thread.is_alive():
                self._invalidate_control_caches()
                raise FeeTechCommandTimeoutError(
                    "Timed out waiting for the previous FeeTech communication thread "
                    "to stop before reopening"
                ) from comm_fault
        if not self._portHandler.is_open:
            if not self._portHandler.openPort():
                raise FeeTechCommandDispatchError(
                    f"Failed to reopen FeeTech port {self._portHandler.getPortName()}"
                )
        self._stop_flag.clear()
        with self._lock:
            self._comm_fault = None
            self._position = {}
            self._velocity = {}
            self._current = {}
        self._invalidate_control_caches()
        self._comm_task_queue = Queue(maxsize=FEETECH_COMM_QUEUE_SIZE)
        self._comm_thread = Thread(target=self._comm_worker, daemon=True)
        self._comm_thread.start()

    def close(self, timeout: float = FEETECH_COMM_TIMEOUT) -> None:
        timeout = self._validate_timeout(timeout)
        with self._lifecycle_lock:
            self._close(timeout)

    def _close(self, timeout: float) -> None:
        self._stop_flag.set()
        port_error: Optional[Exception] = None
        serial_port = getattr(self._portHandler, "ser", None)
        if serial_port is not None:
            for method_name in ("cancel_read", "cancel_write"):
                method = getattr(serial_port, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        logger.exception(f"Failed to {method_name} on FeeTech serial port")
        if self._portHandler.is_open:
            try:
                self._portHandler.closePort()
            except Exception as exc:
                port_error = exc
        if self._comm_thread.is_alive():
            self._comm_thread.join(timeout=timeout)
        self._invalidate_control_caches()
        if self._comm_thread.is_alive():
            raise FeeTechCommandTimeoutError(
                "Timed out waiting for FeeTech communication thread to close"
            ) from port_error
        if port_error is not None:
            raise FeeTechCommandDispatchError("Failed to close FeeTech serial port") from port_error


if __name__ == "__main__":
    driver = FeeTechDriver([0, 1, 2, 3, 4], "/dev/ttyUSB0")
    while True:
        print(driver.get_state())
        # print(driver.get_frequency())
        time.sleep(0.05)
