"""Central runtime safety state machine and generation/deadline policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Optional


class RuntimeSafetyState(str, Enum):
    """Mutually exclusive lifecycle states for one robot runtime."""

    DISCONNECTED = "disconnected"
    SAFE_DISABLED = "safe_disabled"
    READY = "ready"
    ACTIVE = "active"
    HOLD = "hold"
    FAULT = "fault"


@dataclass(frozen=True)
class SafetySnapshot:
    """Immutable state-machine snapshot safe to share with adapters."""

    state: RuntimeSafetyState
    generation: int
    last_command_ns: Optional[int]
    fault_reason: Optional[str]


class RuntimeSafetyController:
    """Thread-safe authority for lifecycle, command generations, and fault latching.

    Timing belongs to the component that can enforce it: a bus actor owns actuator and
    feedback deadlines, while a transport session owns peer loss. Keeping timers out of
    this state machine prevents one scheduler delay from being interpreted three times.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._state = RuntimeSafetyState.DISCONNECTED
        self._generation = 0
        self._last_command_ns: Optional[int] = None
        self._fault_reason: Optional[str] = None

    def snapshot(self) -> SafetySnapshot:
        """Read state, generation, heartbeat, and fault atomically."""

        with self._lock:
            return SafetySnapshot(
                self._state,
                self._generation,
                self._last_command_ns,
                self._fault_reason,
            )

    def connected(self) -> None:
        """Enter SAFE_DISABLED after resources are connected."""

        with self._lock:
            if self._state != RuntimeSafetyState.DISCONNECTED:
                raise RuntimeError("runtime is already connected")
            self._state = RuntimeSafetyState.SAFE_DISABLED

    def ready(self) -> None:
        """Arm command admission without accepting motion yet."""

        with self._lock:
            if self._state not in (RuntimeSafetyState.SAFE_DISABLED, RuntimeSafetyState.HOLD):
                raise RuntimeError(f"cannot enter READY from {self._state.value}")
            self._state = RuntimeSafetyState.READY
            self._last_command_ns = None

    def disabled(self) -> None:
        """Invalidate old commands and enter SAFE_DISABLED."""

        with self._lock:
            if self._state == RuntimeSafetyState.DISCONNECTED:
                raise RuntimeError("cannot disable a disconnected runtime")
            if self._state == RuntimeSafetyState.FAULT:
                raise RuntimeError("cannot clear a latched fault without an explicit reset")
            self._generation += 1
            self._state = RuntimeSafetyState.SAFE_DISABLED
            self._last_command_ns = None
            self._fault_reason = None

    def reset_fault(self) -> None:
        """Explicitly clear a latched fault without enabling motion."""

        with self._lock:
            if self._state != RuntimeSafetyState.FAULT:
                raise RuntimeError(f"cannot reset fault from {self._state.value}")
            self._generation += 1
            self._state = RuntimeSafetyState.SAFE_DISABLED
            self._last_command_ns = None
            self._fault_reason = None

    def hold(self) -> None:
        """Invalidate old commands while retaining actuator holding force."""

        with self._lock:
            if self._state not in (
                RuntimeSafetyState.READY,
                RuntimeSafetyState.ACTIVE,
                RuntimeSafetyState.HOLD,
            ):
                raise RuntimeError(f"cannot enter HOLD from {self._state.value}")
            self._generation += 1
            self._state = RuntimeSafetyState.HOLD
            self._last_command_ns = None

    def accept_command(self, now_ns: int, *, generation: int, deadline_ns: int) -> bool:
        """Atomically admit the first READY command or refresh an ACTIVE heartbeat."""

        self._validate_time(now_ns)
        with self._lock:
            if (
                self._state not in (RuntimeSafetyState.READY, RuntimeSafetyState.ACTIVE)
                or generation != self._generation
                or now_ns > deadline_ns
            ):
                return False
            self._state = RuntimeSafetyState.ACTIVE
            self._last_command_ns = now_ns
            return True

    def fault(self, reason: str) -> None:
        """Latch a fault and invalidate all queued motion generations."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("fault reason must be a non-empty string")
        with self._lock:
            self._enter_fault_locked(reason)

    def emergency_stop(self, reason: str = "emergency stop requested") -> None:
        """Latch emergency FAULT until an explicit reset."""

        with self._lock:
            self._generation += 1
            self._state = RuntimeSafetyState.FAULT
            self._last_command_ns = None
            self._fault_reason = reason

    def disconnected(self) -> None:
        """Invalidate old commands and return to DISCONNECTED."""

        with self._lock:
            self._generation += 1
            self._state = RuntimeSafetyState.DISCONNECTED
            self._last_command_ns = None
            self._fault_reason = None

    def _enter_fault_locked(self, reason: str) -> None:
        """Latch FAULT while the caller already owns the controller lock."""

        self._generation += 1
        self._state = RuntimeSafetyState.FAULT
        self._last_command_ns = None
        self._fault_reason = reason

    @staticmethod
    def _validate_time(now_ns: int) -> None:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("monotonic time must be a non-negative integer")


__all__ = [
    "RuntimeSafetyController",
    "RuntimeSafetyState",
    "SafetySnapshot",
]
