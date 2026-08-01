from __future__ import annotations

import time
from threading import Event, Thread

import pytest

from acetele.hardware.buses import (
    BusActor,
    BusError,
    MotionCommitGate,
    MotionEnvelope,
    RecoverableBusError,
)
from acetele.hardware.buses import actor as actor_module


class FakeProtocol:
    def __init__(self) -> None:
        self.read_gate = Event()
        self.read_started = Event()
        self.connected = False
        self.cancelled = False
        self.motion_batches: list[tuple[MotionEnvelope, ...]] = []
        self.safety: list[tuple[str, object]] = []
        self.sequence = 0
        self.fast_deadlines: list[int | None] = []
        self.slow_deadlines: list[int | None] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def cancel(self) -> None:
        self.cancelled = True
        self.read_gate.set()

    def execute_safety(self, label, payload):
        self.safety.append((label, payload))
        return payload

    def write_motion(self, targets):
        self.motion_batches.append(tuple(targets))

    def read_fast_state(self, *, deadline_ns=None):
        self.fast_deadlines.append(deadline_ns)
        self.read_started.set()
        self.read_gate.wait(1.0)
        self.sequence += 1
        return {"sequence": self.sequence}

    def read_slow_state(self, *, deadline_ns=None):
        self.slow_deadlines.append(deadline_ns)
        return {"temperature": 30.0}


def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("condition was not reached before timeout")


def _target(
    value: float,
    generation: int,
    *,
    submitted_at_ns: int | None = None,
    deadline_ns: int | None = None,
    commit_gate: MotionCommitGate | None = None,
):
    submitted_at_ns = time.monotonic_ns() if submitted_at_ns is None else submitted_at_ns
    return MotionEnvelope(
        key=("position", 1),
        device_id=1,
        payload=value,
        submitted_at_ns=submitted_at_ns,
        deadline_ns=(
            submitted_at_ns + 1_000_000_000 if deadline_ns is None else deadline_ns
        ),
        generation=generation,
        commit_gate=commit_gate,
    )


def test_latest_motion_mailbox_replaces_unstarted_target():
    protocol = FakeProtocol()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        generation = actor.generation
        actor.submit_motion((_target(1.0, generation),))
        actor.submit_motion((_target(2.0, generation),))
        actor.submit_motion((_target(3.0, generation),))
        assert actor.diagnostics().pending_motion_count == 1

        protocol.read_gate.set()
        _wait_until(lambda: bool(protocol.motion_batches))

        assert [target.payload for target in protocol.motion_batches[0]] == [3.0]
        assert actor.diagnostics().replaced_motion_count == 2
    finally:
        actor.disconnect()


def test_motion_commit_gate_hides_batch_until_runtime_accepts_it():
    protocol = FakeProtocol()
    protocol.read_gate.set()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        gate = MotionCommitGate()
        actor.submit_motion((_target(1.0, actor.generation, commit_gate=gate),))

        time.sleep(0.03)
        assert protocol.motion_batches == []
        assert actor.diagnostics().pending_motion_count == 1

        gate.commit()
        _wait_until(lambda: bool(protocol.motion_batches))
        assert protocol.motion_batches[0][0].payload == 1.0
    finally:
        actor.disconnect()


def test_aborted_motion_batch_is_removed_without_a_hardware_write():
    protocol = FakeProtocol()
    protocol.read_gate.set()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        gate = MotionCommitGate()
        actor.submit_motion((_target(1.0, actor.generation, commit_gate=gate),))
        gate.abort()

        _wait_until(lambda: actor.diagnostics().pending_motion_count == 0)
        assert protocol.motion_batches == []
    finally:
        actor.disconnect()


def test_safety_task_discards_old_generation_motion():
    protocol = FakeProtocol()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        actor.submit_motion((_target(1.0, actor.generation),))
        actor.submit_safety("disable", {"ids": (1,)}, wait=False)
        protocol.read_gate.set()
        _wait_until(lambda: bool(protocol.safety))
        time.sleep(0.02)

        assert protocol.motion_batches == []
        assert actor.generation == 1
    finally:
        actor.disconnect()


def test_timed_out_pending_safety_task_is_cancelled_before_execution():
    protocol = FakeProtocol()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        assert protocol.read_started.wait(0.2)
        with pytest.raises(RuntimeError, match="timed out"):
            actor.submit_safety("disable", None, timeout=0.01)

        protocol.read_gate.set()
        time.sleep(0.02)

        assert protocol.safety == []
    finally:
        actor.disconnect()


def test_safety_task_rechecks_completion_after_wait_deadline():
    class BoundaryEvent:
        def __init__(self) -> None:
            self.task = None

        def wait(self, _timeout):
            self.task.finish(result="completed")
            return False

        def set(self):
            pass

    completed = BoundaryEvent()
    task = actor_module._SafetyTask("disable", None, completed)  # noqa: SLF001
    completed.task = task

    assert task.wait_result(0.01) == (True, "completed", None)


def test_expired_motion_is_dropped():
    protocol = FakeProtocol()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        actor.submit_motion(
            (_target(1.0, actor.generation, submitted_at_ns=0, deadline_ns=1),)
        )
        protocol.read_gate.set()
        _wait_until(lambda: actor.diagnostics().expired_motion_count == 1)
        assert protocol.motion_batches == []
    finally:
        actor.disconnect()


def test_disconnect_cancels_blocking_protocol_read():
    protocol = FakeProtocol()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()

    actor.disconnect(timeout=0.2)

    assert protocol.cancelled
    assert not protocol.connected
    assert not actor.connected


def test_disconnect_discards_queued_safety_before_reconnect():
    protocol = FakeProtocol()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    assert protocol.read_started.wait(0.2)
    actor.submit_safety("old-disable", None, wait=False)

    actor.disconnect(timeout=0.2)
    actor.connect()
    try:
        time.sleep(0.02)
        assert protocol.safety == []
        assert actor.diagnostics().strict_fifo_depth == 0
    finally:
        actor.disconnect()


def test_snapshot_is_an_independent_copy():
    protocol = FakeProtocol()
    protocol.read_gate.set()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        _wait_until(lambda: actor.diagnostics().last_state_ns is not None)
        snapshot = actor.get_snapshot()
        snapshot["sequence"] = -1
        assert actor.get_snapshot()["sequence"] >= 1
    finally:
        actor.disconnect()


def test_wait_for_snapshot_blocks_until_first_complete_read():
    protocol = FakeProtocol()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        with pytest.raises(BusError, match="initial serial bus state"):
            actor.wait_for_snapshot(timeout=0.01)

        protocol.read_gate.set()
        snapshot = actor.wait_for_snapshot(timeout=0.2)

        assert snapshot["sequence"] >= 1
        snapshot["sequence"] = -1
        assert actor.get_snapshot()["sequence"] >= 1
    finally:
        actor.disconnect()


def test_stale_generation_is_rejected_before_mailbox_update():
    protocol = FakeProtocol()
    protocol.read_gate.set()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        actor.discard_motion()
        with pytest.raises(ValueError, match="generation"):
            actor.submit_motion((_target(1.0, 0),))
        assert actor.diagnostics().pending_motion_count == 0
    finally:
        actor.disconnect()


def test_fast_state_read_retries_once_with_the_same_cycle_deadline():
    class RetryProtocol(FakeProtocol):
        def read_fast_state(self, *, deadline_ns=None):
            self.fast_deadlines.append(deadline_ns)
            if len(self.fast_deadlines) == 1:
                raise RecoverableBusError("transient checksum failure")
            self.sequence += 1
            return {"sequence": self.sequence}

    protocol = RetryProtocol()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        _wait_until(lambda: actor.diagnostics().last_state_ns is not None)
        assert len(protocol.fast_deadlines) >= 2
        assert protocol.fast_deadlines[0] == protocol.fast_deadlines[1]
        diagnostics = actor.diagnostics()
        assert diagnostics.recoverable_error_count == 1
        assert diagnostics.last_fast_read_s is not None
        assert diagnostics.state_age_s is not None
    finally:
        actor.disconnect()


def test_slow_state_read_is_not_starved_by_continuous_motion():
    class ReplenishingProtocol(FakeProtocol):
        actor = None

        def read_fast_state(self, *, deadline_ns=None):
            snapshot = super().read_fast_state(deadline_ns=deadline_ns)
            if self.actor is not None:
                self.actor.submit_motion(
                    (_target(float(self.sequence), self.actor.generation),)
                )
            return snapshot

    protocol = ReplenishingProtocol()
    actor = BusActor(protocol, cycle_hz=100.0, slow_state_hz=100.0)
    actor.connect()
    protocol.actor = actor
    protocol.read_gate.set()
    try:
        _wait_until(lambda: len(protocol.slow_deadlines) >= 2)
        assert actor.diagnostics().pending_motion_count == 1
    finally:
        protocol.actor = None
        actor.disconnect()


def test_motion_write_and_state_read_respect_period_deadline():
    class ImmediateProtocol(FakeProtocol):
        def read_fast_state(self, *, deadline_ns=None):
            self.fast_deadlines.append(deadline_ns)
            self.sequence += 1
            return {"sequence": self.sequence}

    protocol = ImmediateProtocol()
    actor = BusActor(protocol, cycle_hz=20.0)
    actor.connect()
    try:
        _wait_until(lambda: protocol.sequence >= 1)
        previous_sequence = protocol.sequence
        submitted_at = time.monotonic()
        actor.submit_motion((_target(1.0, actor.generation),))

        time.sleep(0.015)
        assert protocol.motion_batches == []

        _wait_until(
            lambda: bool(protocol.motion_batches)
            and protocol.sequence > previous_sequence,
            timeout=0.15,
        )
        assert time.monotonic() - submitted_at >= 0.03
    finally:
        actor.disconnect()


def test_actor_watchdog_holds_without_an_adapter_poll_loop():
    protocol = FakeProtocol()
    protocol.read_gate.set()
    actor = BusActor(
        protocol,
        cycle_hz=100.0,
        motion_watchdog_ns=20_000_000,
    )
    actor.connect()
    try:
        actor.refresh_motion_watchdog(time.monotonic_ns())

        _wait_until(lambda: ("hold", None) in protocol.safety)

        diagnostics = actor.diagnostics()
        assert diagnostics.motion_watchdog_tripped
        assert actor.generation == 1
        with pytest.raises(BusError, match="watchdog"):
            actor.submit_motion((_target(1.0, actor.generation),))

        actor.submit_safety("set_enabled", True, clear_motion=False)
        assert not actor.motion_watchdog_tripped
        actor.submit_motion((_target(1.0, actor.generation),))
    finally:
        actor.disconnect()


def test_consecutive_motion_write_failures_emergency_stop_and_fault_actor():
    class FailingMotionProtocol(FakeProtocol):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def write_motion(self, targets):
            self.attempts += 1
            raise RecoverableBusError("write failed")

        def read_fast_state(self, *, deadline_ns=None):
            self.sequence += 1
            return {"sequence": self.sequence}

    protocol = FailingMotionProtocol()
    actor = BusActor(protocol, cycle_hz=200.0, motion_failure_limit=3)
    actor.connect()
    try:
        for value in (1.0, 2.0, 3.0):
            actor.submit_motion((_target(value, actor.generation),))
            _wait_until(lambda: protocol.attempts >= int(value))

        _wait_until(lambda: actor.diagnostics().fault is not None)
        diagnostics = actor.diagnostics()
        assert not diagnostics.connected
        assert diagnostics.consecutive_motion_error_count == 3
        assert ("emergency_stop", None) in protocol.safety
    finally:
        actor.disconnect()


def test_started_safety_timeout_is_fenced_before_later_work():
    class BlockingSafetyProtocol(FakeProtocol):
        def __init__(self):
            super().__init__()
            self.safety_started = Event()
            self.release_safety = Event()

        def execute_safety(self, label, payload):
            self.safety.append((label, payload))
            if label == "set_enabled":
                self.safety_started.set()
                assert self.release_safety.wait(1.0)
            return payload

        def read_fast_state(self, *, deadline_ns=None):
            self.sequence += 1
            return {"sequence": self.sequence}

    protocol = BlockingSafetyProtocol()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        errors = []

        def submit_enable():
            try:
                actor.submit_safety(
                    "set_enabled",
                    True,
                    timeout=0.02,
                    clear_motion=False,
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        worker = Thread(target=submit_enable)
        worker.start()
        assert protocol.safety_started.wait(0.2)
        worker.join(0.2)
        assert errors and "timed out" in str(errors[0])
        assert actor.diagnostics().safety_task_fenced
        with pytest.raises(BusError, match="timed-out safety task"):
            actor.submit_motion((_target(1.0, actor.generation),))

        protocol.release_safety.set()
        _wait_until(lambda: actor.diagnostics().fault is not None)
        assert protocol.safety[:2] == [
            ("set_enabled", True),
            ("emergency_stop", None),
        ]
    finally:
        protocol.release_safety.set()
        actor.disconnect()


def test_motion_diagnostics_measure_through_protocol_completion():
    class ImmediateProtocol(FakeProtocol):
        def read_fast_state(self, *, deadline_ns=None):
            self.sequence += 1
            return {"sequence": self.sequence}

    protocol = ImmediateProtocol()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        submitted_ns = time.monotonic_ns() - 5_000_000
        actor.submit_motion(
            (_target(1.0, actor.generation, submitted_at_ns=submitted_ns),)
        )
        _wait_until(lambda: actor.diagnostics().last_motion_success_ns is not None)

        diagnostics = actor.diagnostics()
        assert diagnostics.last_motion_end_to_end_s is not None
        assert diagnostics.last_motion_end_to_end_s >= 0.005
        assert diagnostics.max_motion_end_to_end_s == diagnostics.last_motion_end_to_end_s
        assert diagnostics.p95_motion_end_to_end_s == diagnostics.last_motion_end_to_end_s
        assert diagnostics.p99_motion_end_to_end_s == diagnostics.last_motion_end_to_end_s
        assert diagnostics.motion_end_to_end_sample_count == 1
        assert diagnostics.last_motion_generation == actor.generation
    finally:
        actor.disconnect()


def test_persistent_fast_state_loss_faults_inside_actor():
    class StateLossProtocol(FakeProtocol):
        def read_fast_state(self, *, deadline_ns=None):
            self.sequence += 1
            if self.sequence == 1:
                return {"sequence": self.sequence}
            raise RecoverableBusError("state unavailable")

    protocol = StateLossProtocol()
    actor = BusActor(
        protocol,
        cycle_hz=200.0,
        state_timeout_ns=20_000_000,
    )
    actor.connect()
    try:
        actor.wait_for_snapshot(timeout=0.2)
        _wait_until(lambda: actor.diagnostics().fault is not None)

        assert ("emergency_stop", None) in protocol.safety
        with pytest.raises(BusError, match="faulted"):
            actor.get_snapshot()
    finally:
        actor.disconnect()


def test_unexpected_worker_failure_attempts_emergency_stop_before_exit():
    class BrokenProtocol(FakeProtocol):
        def read_fast_state(self, *, deadline_ns=None):
            raise AssertionError("unexpected decoder failure")

    protocol = BrokenProtocol()
    actor = BusActor(protocol, cycle_hz=100.0)
    actor.connect()
    try:
        _wait_until(lambda: actor.diagnostics().fault is not None)

        assert protocol.safety == [("emergency_stop", None)]
        assert not actor.connected
    finally:
        actor.disconnect()
