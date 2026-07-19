from __future__ import annotations

from enum import Enum


class LeaderSyncMode(str, Enum):
    IDLE = "idle"
    SYNC_REQUEST = "sync_request"
    READY = "ready"
    TRACKING = "tracking"
    STOP = "stop"


class FollowerSyncStatus(str, Enum):
    IDLE = "idle"
    ALIGNING = "aligning"
    READY = "ready"
    TRACKING = "tracking"
    LOST = "lost"
    FAULT = "fault"


class FollowerSyncController:
    """Transport-independent follower synchronization and command heartbeat."""

    def __init__(self, heartbeat_timeout_ns: int):
        if heartbeat_timeout_ns <= 0:
            raise ValueError("heartbeat timeout must be positive")
        self._heartbeat_timeout_ns = int(heartbeat_timeout_ns)
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
        elif mode == LeaderSyncMode.STOP:
            self.status = FollowerSyncStatus.LOST
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
            and int(now_ns) - self.last_command_ns > self._heartbeat_timeout_ns
        ):
            self.status = FollowerSyncStatus.LOST
        return self.status


__all__ = ["FollowerSyncController", "FollowerSyncStatus", "LeaderSyncMode"]
