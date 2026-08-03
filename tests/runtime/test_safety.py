from __future__ import annotations

import pytest

from acetele.runtime import RuntimeSafetyController, RuntimeSafetyState, SafetyTransition


def test_command_timeout_enters_hold_and_advances_generation():
    safety = RuntimeSafetyController()
    safety.connected()
    safety.ready()
    assert safety.accept_command(1, generation=0, deadline_ns=2)

    transition = safety.update(100_000_002, latest_state_ns=100_000_002)

    assert transition == SafetyTransition.HOLD
    assert safety.snapshot().state == RuntimeSafetyState.HOLD
    assert safety.snapshot().generation == 1


def test_stale_state_faults_before_accepting_more_motion():
    safety = RuntimeSafetyController()
    safety.connected()
    safety.ready()
    assert safety.accept_command(1, generation=0, deadline_ns=2)

    transition = safety.update(60_000_001, latest_state_ns=1)

    assert transition == SafetyTransition.FAULT
    assert safety.snapshot().state == RuntimeSafetyState.FAULT
    assert not safety.accept_command(60_000_002, generation=1, deadline_ns=70_000_000)


def test_command_atomically_activates_ready_state_and_enforces_generation_and_deadline():
    safety = RuntimeSafetyController()
    safety.connected()
    safety.ready()
    assert safety.accept_command(1, generation=0, deadline_ns=2)
    assert safety.snapshot().state == RuntimeSafetyState.ACTIVE

    assert not safety.accept_command(3, generation=1, deadline_ns=4)
    assert not safety.accept_command(5, generation=0, deadline_ns=4)
    assert safety.accept_command(5, generation=0, deadline_ns=6)


def test_manual_hold_and_disable_advance_generation():
    safety = RuntimeSafetyController()
    safety.connected()
    safety.ready()
    assert safety.accept_command(1, generation=0, deadline_ns=2)

    safety.hold()
    assert safety.snapshot().state == RuntimeSafetyState.HOLD
    assert safety.snapshot().generation == 1
    safety.disabled()
    assert safety.snapshot().state == RuntimeSafetyState.SAFE_DISABLED
    assert safety.snapshot().generation == 2


def test_latched_fault_requires_explicit_reset():
    safety = RuntimeSafetyController()
    safety.connected()
    safety.emergency_stop()

    with pytest.raises(RuntimeError, match="latched fault"):
        safety.disabled()

    safety.reset_fault()
    assert safety.snapshot().state == RuntimeSafetyState.SAFE_DISABLED
