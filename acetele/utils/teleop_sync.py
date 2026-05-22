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
