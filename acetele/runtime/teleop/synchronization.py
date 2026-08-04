from __future__ import annotations

from enum import Enum


class TeleopMode(str, Enum):
    """Select the arm command representation admitted by one session."""

    JOINT = "joint"
    EE_POSE = "ee_pose"


class LeaderSyncMode(str, Enum):
    IDLE = "idle"
    SYNC_REQUEST = "sync_request"
    READY = "ready"
    TRACKING = "tracking"
    HOLD = "hold"
    STOP = "stop"


class FollowerSyncStatus(str, Enum):
    IDLE = "idle"
    READY = "ready"
    TRACKING = "tracking"
    HOLD = "hold"
    LOST = "lost"
    FAULT = "fault"


class FollowerSyncController:
    """Transport-independent peer synchronization and session heartbeat."""

    def __init__(self, session_timeout_ns: int):
        if type(session_timeout_ns) is not int or session_timeout_ns <= 0:
            raise ValueError("session_timeout_ns must be a positive integer")
        # The bus actor independently owns the short actuator watchdog. This controller
        # only decides when the remote peer has been absent long enough to require sync.
        self._session_timeout_ns = session_timeout_ns
        self.mode = LeaderSyncMode.IDLE
        self.status = FollowerSyncStatus.IDLE
        self.last_command_ns: int | None = None

    def set_mode(self, mode: LeaderSyncMode) -> None:
        self.mode = mode
        if mode == LeaderSyncMode.IDLE:
            self.status = FollowerSyncStatus.IDLE
            self.last_command_ns = None
        elif mode in (LeaderSyncMode.SYNC_REQUEST, LeaderSyncMode.READY):
            self.status = FollowerSyncStatus.READY
            self.last_command_ns = None
        elif mode == LeaderSyncMode.HOLD:
            self.status = FollowerSyncStatus.HOLD
            self.last_command_ns = None
        elif mode == LeaderSyncMode.STOP:
            self.status = FollowerSyncStatus.LOST
            self.last_command_ns = None

    def reset_peer(self) -> None:
        """Forget all state owned by a previous transport session."""

        self.mode = LeaderSyncMode.IDLE
        self.status = FollowerSyncStatus.IDLE
        self.last_command_ns = None

    @property
    def command_allowed(self) -> bool:
        return self.mode == LeaderSyncMode.TRACKING and self.status in (
            FollowerSyncStatus.READY,
            FollowerSyncStatus.TRACKING,
        )

    def accept_command(self, now_ns: int) -> bool:
        if not self.command_allowed:
            return False
        self.last_command_ns = int(now_ns)
        self.status = FollowerSyncStatus.TRACKING
        return True

    def update(self, now_ns: int) -> FollowerSyncStatus:
        if (
            self.status == FollowerSyncStatus.TRACKING
            and self.last_command_ns is not None
            and int(now_ns) - self.last_command_ns > self._session_timeout_ns
        ):
            self.status = FollowerSyncStatus.LOST
        return self.status


__all__ = [
    "FollowerSyncController",
    "FollowerSyncStatus",
    "LeaderSyncMode",
    "TeleopMode",
]
