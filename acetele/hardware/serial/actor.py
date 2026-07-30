"""Single-owner serial bus actor with safety FIFO and latest-value motion mailbox.

Exactly one worker calls a protocol instance. Safety operations preserve FIFO ordering;
streaming motion is bounded by replacing unconsumed targets with the same key. This is
the central concurrency boundary that keeps vendor SDKs and serial ports single-threaded.
"""

from __future__ import annotations

import copy
import math
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Condition, Event, Lock, RLock, Thread
from typing import Any, Hashable, Optional, Protocol, Sequence


class BusError(RuntimeError):
    """Base error exposed by the serial actor boundary."""


class RecoverableBusError(BusError):
    """One frame failed but the actor and subsequent frames may continue."""


class FatalBusError(BusError):
    """The actor stopped because bus state can no longer be trusted."""


class MotionCommitGate:
    """Shared two-phase gate that keeps a motion batch invisible until committed."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._state: Optional[bool] = None

    @property
    def state(self) -> Optional[bool]:
        """Return ``None`` while staged, ``True`` if committed, else ``False``."""

        with self._lock:
            return self._state

    def commit(self) -> None:
        """Publish every envelope sharing this gate."""

        with self._lock:
            if self._state is False:
                raise RuntimeError("cannot commit an aborted motion batch")
            self._state = True

    def abort(self) -> None:
        """Keep every staged envelope permanently invisible to its worker."""

        with self._lock:
            if self._state is None:
                self._state = False


@dataclass(frozen=True)
class MotionEnvelope:
    """One replaceable device target with lifetime and safety generation."""

    key: Hashable
    device_id: int
    payload: Any
    submitted_at_ns: int
    deadline_ns: int
    generation: int
    commit_gate: Optional[MotionCommitGate] = None

    def __post_init__(self) -> None:
        if type(self.device_id) is not int or self.device_id < 0:
            raise ValueError("motion device_id must be a non-negative integer")
        if type(self.submitted_at_ns) is not int or self.submitted_at_ns < 0:
            raise ValueError("motion submitted_at_ns must be a non-negative integer")
        if type(self.deadline_ns) is not int or self.deadline_ns < self.submitted_at_ns:
            raise ValueError("motion deadline_ns must not precede submitted_at_ns")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("motion generation must be a non-negative integer")
        if self.commit_gate is not None and not isinstance(
            self.commit_gate,
            MotionCommitGate,
        ):
            raise ValueError("motion commit_gate must be a MotionCommitGate or None")


class SerialBusProtocol(Protocol):
    """Vendor protocol executed exclusively by a :class:`SerialBusActor`."""

    def connect(self) -> None:
        """Acquire the protocol transport and verify configured identities."""

        ...

    def disconnect(self) -> None:
        """Release the protocol transport."""

        ...

    def cancel(self) -> None:
        """Interrupt pending transport I/O during shutdown."""

        ...

    def execute_safety(self, label: str, payload: Any) -> Any:
        """Execute one ordered, non-replaceable safety operation."""

        ...

    def write_motion(self, targets: Sequence[MotionEnvelope]) -> None:
        """Write the latest committed target set for this cycle."""

        ...

    def read_fast_state(self, *, deadline_ns: Optional[int] = None) -> Any:
        """Read motion-critical state within the cycle deadline."""

        ...

    def read_slow_state(self, *, deadline_ns: Optional[int] = None) -> Any:
        """Read optional telemetry only when cycle budget permits."""

        ...


@dataclass(frozen=True)
class BusActorDiagnostics:
    """Independent timing and queue snapshot for latency analysis."""

    connected: bool
    fault: Optional[str]
    generation: int
    strict_fifo_depth: int
    pending_motion_count: int
    replaced_motion_count: int
    expired_motion_count: int
    recoverable_error_count: int
    last_motion_wait_s: Optional[float]
    max_motion_wait_s: Optional[float]
    last_motion_write_s: Optional[float]
    last_fast_read_s: Optional[float]
    last_slow_read_s: Optional[float]
    last_cycle_s: Optional[float]
    maximum_cycle_s: Optional[float]
    last_state_ns: Optional[int]
    state_age_s: Optional[float]
    state_frequency_hz: Optional[float]


@dataclass
class _SafetyTask:
    label: str
    payload: Any
    completed: Event
    result: Any = None
    error: Optional[BaseException] = None
    started: bool = False
    cancelled: bool = False
    finished: bool = False

    def __post_init__(self) -> None:
        self._lock = Lock()

    def try_start(self) -> bool:
        """Claim this task unless a waiting submitter cancelled it."""

        with self._lock:
            if self.cancelled:
                return False
            self.started = True
            return True

    def finish(
        self,
        *,
        result: Any = None,
        error: Optional[BaseException] = None,
    ) -> None:
        """Publish one result and wake a possible blocking submitter."""

        with self._lock:
            if self.finished:
                return
            self.result = result
            self.error = error
            self.finished = True
            self.completed.set()

    def wait_result(
        self,
        timeout: float,
    ) -> tuple[bool, Any, Optional[BaseException]]:
        """Wait boundedly and cancel a task that has not started."""

        self.completed.wait(timeout)
        with self._lock:
            if self.finished:
                return True, self.result, self.error
            if not self.started:
                self.cancelled = True
            return False, None, None


class SerialBusActor:
    """One worker owns one serial protocol and schedules bounded latest-value I/O."""

    def __init__(
        self,
        protocol: SerialBusProtocol,
        *,
        cycle_hz: float,
        slow_state_hz: float = 1.0,
        safety_queue_size: int = 32,
        clock_ns=time.monotonic_ns,
    ) -> None:
        if cycle_hz <= 0.0 or slow_state_hz <= 0.0:
            raise ValueError("serial actor rates must be positive")
        if type(safety_queue_size) is not int or safety_queue_size <= 0:
            raise ValueError("safety_queue_size must be a positive integer")
        self._protocol = protocol
        self._period_ns = max(1, round(1e9 / cycle_hz))
        self._slow_period_ns = max(1, round(1e9 / slow_state_hz))
        self._clock_ns = clock_ns
        # Safety work is ordered and lossless; motion is latest-value data. Separate
        # storage is the central scheduling invariant of the actor.
        self._condition = Condition(Lock())
        self._state_lock = RLock()
        self._safety_queue: Queue[_SafetyTask] = Queue(maxsize=safety_queue_size)
        self._pending_motion: dict[Hashable, MotionEnvelope] = {}
        self._generation = 0
        self._stop = Event()
        self._snapshot_ready = Event()
        self._worker: Optional[Thread] = None
        self._fault: Optional[BaseException] = None
        # Fast and slow snapshots are replaced atomically under _state_lock so readers
        # never observe a dictionary while the worker is filling it.
        self._snapshot: Any = None
        self._slow_snapshot: Any = None
        self._last_state_ns: Optional[int] = None
        self._state_intervals_ns: deque[int] = deque(maxlen=100)
        self._replaced_motion_count = 0
        self._expired_motion_count = 0
        self._recoverable_error_count = 0
        self._last_motion_wait_s: Optional[float] = None
        self._max_motion_wait_s: Optional[float] = None
        self._last_motion_write_s: Optional[float] = None
        self._last_fast_read_s: Optional[float] = None
        self._last_slow_read_s: Optional[float] = None
        self._last_cycle_s: Optional[float] = None
        self._maximum_cycle_s: Optional[float] = None

    @property
    def connected(self) -> bool:
        """Return whether the worker is alive and has no latched fault."""

        with self._state_lock:
            return self._worker is not None and self._worker.is_alive() and self._fault is None

    @property
    def generation(self) -> int:
        """Return the generation required by newly submitted motion."""

        with self._condition:
            return self._generation

    def connect(self) -> None:
        """Connect the protocol, then start its sole owning worker."""

        with self._state_lock:
            if self._worker is not None:
                raise RuntimeError("serial bus actor is already connected")
            self._fault = None
            self._snapshot = None
            self._slow_snapshot = None
            self._last_state_ns = None
            self._state_intervals_ns.clear()
            self._stop.clear()
            self._snapshot_ready.clear()
        # Identity checks and initial safe-mode writes happen synchronously here. A
        # worker is published only after the protocol has reached a known state.
        self._protocol.connect()
        worker = Thread(target=self._run, name="acetele-serial-bus", daemon=True)
        with self._state_lock:
            self._worker = worker
        try:
            worker.start()
        except BaseException:
            with self._state_lock:
                self._worker = None
            self._protocol.disconnect()
            raise

    def submit_motion(self, targets: Sequence[MotionEnvelope]) -> None:
        """Atomically replace pending motion targets by key without blocking on I/O."""

        targets = tuple(targets)
        if not targets:
            raise ValueError("motion submission requires at least one target")
        keys = tuple(target.key for target in targets)
        if len(set(keys)) != len(keys):
            raise ValueError("motion target keys must be unique within one submission")
        self._require_healthy()
        with self._condition:
            # One condition protects generation validation and the complete mailbox
            # replacement, so the worker can never observe half of a submitted batch.
            if any(target.generation != self._generation for target in targets):
                raise ValueError("motion target generation is stale")
            for target in targets:
                if target.key in self._pending_motion:
                    self._replaced_motion_count += 1
                self._pending_motion[target.key] = target
            self._condition.notify_all()

    def submit_safety(
        self,
        label: str,
        payload: Any,
        *,
        wait: bool = True,
        timeout: float = 1.0,
        clear_motion: bool = True,
    ) -> Any:
        """Queue an ordered safety transaction, optionally waiting for completion."""

        if not isinstance(label, str) or not label.strip():
            raise ValueError("safety task label must be a non-empty string")
        if timeout <= 0.0:
            raise ValueError("safety task timeout must be positive")
        self._require_healthy()
        if clear_motion:
            # Safety work defines a new command epoch. Invalidating motion before FIFO
            # insertion prevents a stale target from running after the safety action.
            self.discard_motion()
        task = _SafetyTask(label, payload, Event())
        try:
            self._safety_queue.put(task, timeout=timeout)
        except Full as exc:
            raise BusError(f"safety queue is full while submitting {label}") from exc
        with self._condition:
            self._condition.notify_all()
        if not wait:
            return None
        completed, result, error = task.wait_result(timeout)
        if not completed:
            raise BusError(f"timed out waiting for safety task {label}")
        if error is not None:
            raise BusError(f"safety task {label} failed") from error
        return result

    def discard_motion(self) -> int:
        """Invalidate all old envelopes by clearing the mailbox and bumping generation."""

        with self._condition:
            self._generation += 1
            self._pending_motion.clear()
            self._condition.notify_all()
            return self._generation

    def get_snapshot(self) -> Any:
        """Return a copy of the latest complete fast-state sample."""

        self._require_healthy()
        with self._state_lock:
            if self._snapshot is None:
                raise BusError("serial bus state is not available yet")
            return copy.deepcopy(self._snapshot)

    def wait_for_snapshot(self, *, timeout: float = 1.0) -> Any:
        """Wait for the first complete sample without exposing actor-owned state."""

        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("serial actor snapshot timeout must be finite and positive")
        self._require_healthy()
        if not self._snapshot_ready.wait(timeout):
            raise BusError("timed out waiting for initial serial bus state")
        self._require_healthy()
        with self._state_lock:
            if self._snapshot is None:
                raise BusError("serial bus state is not available yet")
            return copy.deepcopy(self._snapshot)

    def get_slow_snapshot(self) -> Any:
        """Return the latest optional slow-telemetry snapshot."""

        self._require_healthy()
        with self._state_lock:
            return copy.deepcopy(self._slow_snapshot)

    def diagnostics(self) -> BusActorDiagnostics:
        """Return queue depth, timing, replacement, and fault diagnostics."""

        with self._state_lock:
            intervals = tuple(self._state_intervals_ns)
            frequency = None
            if intervals and sum(intervals) > 0:
                frequency = len(intervals) * 1e9 / sum(intervals)
            worker = self._worker
            now_ns = self._clock_ns()
            state_age_s = (
                None
                if self._last_state_ns is None
                else max(0.0, (now_ns - self._last_state_ns) / 1e9)
            )
            state_values = (
                worker is not None and worker.is_alive() and self._fault is None,
                None if self._fault is None else str(self._fault),
                self._recoverable_error_count,
                self._last_motion_wait_s,
                self._max_motion_wait_s,
                self._last_motion_write_s,
                self._last_fast_read_s,
                self._last_slow_read_s,
                self._last_cycle_s,
                self._maximum_cycle_s,
                self._last_state_ns,
                state_age_s,
                frequency,
            )
        with self._condition:
            scheduling_values = (
                self._generation,
                len(self._pending_motion),
                self._replaced_motion_count,
                self._expired_motion_count,
            )
        return BusActorDiagnostics(
            connected=state_values[0],
            fault=state_values[1],
            generation=scheduling_values[0],
            strict_fifo_depth=self._safety_queue.qsize(),
            pending_motion_count=scheduling_values[1],
            replaced_motion_count=scheduling_values[2],
            expired_motion_count=scheduling_values[3],
            recoverable_error_count=state_values[2],
            last_motion_wait_s=state_values[3],
            max_motion_wait_s=state_values[4],
            last_motion_write_s=state_values[5],
            last_fast_read_s=state_values[6],
            last_slow_read_s=state_values[7],
            last_cycle_s=state_values[8],
            maximum_cycle_s=state_values[9],
            last_state_ns=state_values[10],
            state_age_s=state_values[11],
            state_frequency_hz=state_values[12],
        )

    def disconnect(self, *, timeout: float = 1.0) -> None:
        """Cancel I/O and stop within ``timeout``; never perform an unbounded join."""

        if timeout <= 0.0:
            raise ValueError("serial actor close timeout must be positive")
        with self._state_lock:
            worker = self._worker
        if worker is None:
            return
        # Wake every local waiter before interrupting transport I/O. This keeps
        # shutdown bounded even when no state sample has ever arrived.
        self._stop.set()
        self._snapshot_ready.set()
        self.discard_motion()
        with self._condition:
            self._condition.notify_all()
        cancel_error: Optional[BaseException] = None
        cancel = getattr(self._protocol, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except BaseException as exc:
                cancel_error = exc
        worker.join(timeout)
        # Complete queued callers regardless of worker outcome, then release the
        # physical transport. Neither cleanup step is allowed to become unbounded.
        self._fail_pending_safety(
            BusError("serial bus actor disconnected before safety task execution")
        )
        protocol_error: Optional[BaseException] = None
        try:
            self._protocol.disconnect()
        except BaseException as exc:
            protocol_error = exc
        with self._state_lock:
            if worker.is_alive():
                timeout_error = BusError("serial bus worker did not stop within timeout")
                self._fault = timeout_error
                raise timeout_error from (protocol_error or cancel_error)
            self._worker = None
        if protocol_error is not None:
            raise BusError("serial protocol disconnect failed") from protocol_error
        if cancel_error is not None:
            raise BusError("serial protocol cancellation failed") from cancel_error

    def _run(self) -> None:
        """Schedule safety, latest motion, and telemetry on the owning thread."""

        # Every cycle gives safety one slot, then writes the newest motion before the
        # mandatory fast read. Slow telemetry only uses remaining budget and cannot
        # delay a waiting safety transaction.
        next_fast_ns = self._clock_ns()
        next_slow_ns = next_fast_ns + self._slow_period_ns
        try:
            while not self._stop.is_set():
                cycle_start = self._clock_ns()
                self._execute_one_safety()
                now_ns = self._clock_ns()
                if now_ns >= next_fast_ns:
                    self._execute_latest_motion(now_ns)
                    fast_deadline_ns = max(
                        next_fast_ns + self._period_ns,
                        now_ns + self._period_ns,
                    )
                    self._read_fast(fast_deadline_ns)
                    next_fast_ns = fast_deadline_ns
                now_ns = self._clock_ns()
                if (
                    now_ns >= next_slow_ns
                    and now_ns < next_fast_ns
                    and self._safety_idle()
                ):
                    self._read_slow(next_fast_ns)
                    next_slow_ns = max(next_slow_ns + self._slow_period_ns, now_ns + 1)
                cycle_s = (self._clock_ns() - cycle_start) / 1e9
                with self._state_lock:
                    self._last_cycle_s = cycle_s
                    self._maximum_cycle_s = (
                        cycle_s
                        if self._maximum_cycle_s is None
                        else max(self._maximum_cycle_s, cycle_s)
                    )
                self._wait_for_work(next_fast_ns, next_slow_ns)
        except BaseException as exc:
            with self._state_lock:
                self._fault = exc
            self._stop.set()
            self._snapshot_ready.set()
            self._fail_pending_safety(exc)

    def _execute_one_safety(self) -> None:
        """Execute at most one strict-FIFO task and preserve its error for waiters."""

        try:
            task = self._safety_queue.get_nowait()
        except Empty:
            return
        result = None
        error: Optional[BaseException] = None
        try:
            if not task.try_start():
                return
            result = self._protocol.execute_safety(task.label, task.payload)
        except BaseException as exc:
            # A failed safety operation leaves hardware state unknown. Unlike a dropped
            # streaming frame, it is fatal and must stop this actor.
            error = exc
            raise
        finally:
            task.finish(result=result, error=error)
            self._safety_queue.task_done()

    def _execute_latest_motion(self, now_ns: int) -> bool:
        """Consume committed, current-generation targets that have not expired."""

        with self._condition:
            current_generation = self._generation
            targets = []
            expired = 0
            for key, target in tuple(self._pending_motion.items()):
                if target.generation != current_generation:
                    del self._pending_motion[key]
                    continue
                if target.deadline_ns < now_ns:
                    del self._pending_motion[key]
                    expired += 1
                    continue
                # A multi-bus batch remains in each mailbox until its shared gate is
                # committed; an aborted batch is discarded without touching hardware.
                gate_state = (
                    True if target.commit_gate is None else target.commit_gate.state
                )
                if gate_state is None:
                    continue
                del self._pending_motion[key]
                if gate_state:
                    targets.append(target)
        if expired:
            with self._state_lock:
                self._expired_motion_count += expired
        if not targets:
            return False
        active = tuple(targets)
        waits = tuple(max(0.0, (now_ns - target.submitted_at_ns) / 1e9) for target in active)
        write_started_ns = self._clock_ns()
        try:
            self._protocol.write_motion(active)
        except RecoverableBusError:
            with self._state_lock:
                self._recoverable_error_count += 1
                self._last_motion_write_s = (
                    self._clock_ns() - write_started_ns
                ) / 1e9
            return True
        with self._state_lock:
            self._last_motion_write_s = (
                self._clock_ns() - write_started_ns
            ) / 1e9
            self._last_motion_wait_s = max(waits)
            self._max_motion_wait_s = (
                self._last_motion_wait_s
                if self._max_motion_wait_s is None
                else max(self._max_motion_wait_s, self._last_motion_wait_s)
            )
        return True

    def _read_fast(self, deadline_ns: int) -> None:
        """Acquire one atomic fast snapshot with a budget-aware single retry."""

        read_started_ns = self._clock_ns()
        snapshot = None
        # One retry is allowed only while no newer motion or safety work is waiting.
        for attempt in range(2):
            try:
                snapshot = self._protocol.read_fast_state(deadline_ns=deadline_ns)
                break
            except RecoverableBusError:
                with self._state_lock:
                    self._recoverable_error_count += 1
                if (
                    attempt > 0
                    or self._clock_ns() >= deadline_ns
                    or not self._motion_and_safety_idle()
                ):
                    with self._state_lock:
                        self._last_fast_read_s = (
                            self._clock_ns() - read_started_ns
                        ) / 1e9
                    return
        if snapshot is None:
            return
        completed_ns = self._clock_ns()
        with self._state_lock:
            self._last_fast_read_s = (completed_ns - read_started_ns) / 1e9
            if self._last_state_ns is not None:
                self._state_intervals_ns.append(completed_ns - self._last_state_ns)
            self._snapshot = snapshot
            self._last_state_ns = completed_ns
            self._snapshot_ready.set()

    def _read_slow(self, deadline_ns: int) -> None:
        """Acquire optional telemetry without promoting failures to actor faults."""

        read_started_ns = self._clock_ns()
        try:
            snapshot = self._protocol.read_slow_state(deadline_ns=deadline_ns)
        except RecoverableBusError:
            with self._state_lock:
                self._recoverable_error_count += 1
                self._last_slow_read_s = (
                    self._clock_ns() - read_started_ns
                ) / 1e9
            return
        with self._state_lock:
            self._last_slow_read_s = (
                self._clock_ns() - read_started_ns
            ) / 1e9
            self._slow_snapshot = snapshot

    def _motion_and_safety_idle(self) -> bool:
        with self._condition:
            return self._safety_queue.empty() and not self._pending_motion

    def _safety_idle(self) -> bool:
        return self._safety_queue.empty()

    def _wait_for_work(self, next_fast_ns: int, next_slow_ns: int) -> None:
        """Sleep on the condition until work arrives or a read deadline expires."""

        with self._condition:
            while not self._stop.is_set() and self._safety_queue.empty():
                deadline_ns = next_fast_ns
                if not self._pending_motion:
                    deadline_ns = min(next_fast_ns, next_slow_ns)
                timeout = (deadline_ns - self._clock_ns()) / 1e9
                if timeout <= 0.0:
                    return
                self._condition.wait(timeout)

    def _require_healthy(self) -> None:
        """Reject public operations after a fault or outside a connected lifetime."""

        with self._state_lock:
            if self._fault is not None:
                raise FatalBusError("serial bus actor is faulted") from self._fault
            if self._worker is None or not self._worker.is_alive():
                raise BusError("serial bus actor is not connected")

    def _fail_pending_safety(self, error: BaseException) -> None:
        """Wake every queued safety caller with the actor's terminal error."""

        while True:
            try:
                task = self._safety_queue.get_nowait()
            except Empty:
                return
            task.finish(error=error)
            self._safety_queue.task_done()


__all__ = [
    "BusActorDiagnostics",
    "BusError",
    "FatalBusError",
    "MotionCommitGate",
    "MotionEnvelope",
    "RecoverableBusError",
    "SerialBusActor",
    "SerialBusProtocol",
]
