"""Validated deployment options kept outside the hardware RobotSpec."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class PeerRole(str, Enum):
    """Select which side owns each direct TCP endpoint."""

    LEADER = "leader"
    FOLLOWER = "follower"


@dataclass(frozen=True)
class CurveCredentials:
    """Local private certificate and the only accepted peer certificate."""

    secret_key: Path
    peer_key: Path

    def __post_init__(self) -> None:
        for name in ("secret_key", "peer_key"):
            path = Path(getattr(self, name)).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"CURVE {name.replace('_', ' ')} does not exist: {path}")
            object.__setattr__(self, name, path)
        if self.secret_key.stat().st_mode & 0o077:
            raise ValueError("CURVE secret key permissions must not allow group or other access")


@dataclass(frozen=True)
class ZmqTeleopOptions:
    """Network timing, endpoints, and optional authentication for one peer."""

    role: PeerRole
    bind_host: str
    peer_host: str
    command_port: int = 5555
    state_port: int = 5556
    cycle_hz: float = 100.0
    heartbeat_timeout_ns: int = 500_000_000
    maximum_frame_bytes: int = 65_536
    curve: Optional[CurveCredentials] = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, PeerRole):
            raise ValueError("ZMQ role must be a PeerRole")
        for name in ("bind_host", "peer_host"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "://" in value:
                raise ValueError(f"{name} must be a non-empty host without a URI scheme")
        for name in ("command_port", "state_port"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 65_535:
                raise ValueError(f"{name} must be an integer in [1, 65535]")
        if self.command_port == self.state_port:
            raise ValueError("command_port and state_port must be different")
        if not math.isfinite(self.cycle_hz) or not 1.0 <= self.cycle_hz <= 250.0:
            raise ValueError("cycle_hz must be finite and in [1, 250]")
        if type(self.heartbeat_timeout_ns) is not int or self.heartbeat_timeout_ns <= 0:
            raise ValueError("heartbeat_timeout_ns must be a positive integer")
        if type(self.maximum_frame_bytes) is not int or not 1024 <= self.maximum_frame_bytes <= 65_536:
            raise ValueError("maximum_frame_bytes must be an integer in [1024, 65536]")
        if self.curve is not None and not isinstance(self.curve, CurveCredentials):
            raise ValueError("curve must be CurveCredentials or None")

    @property
    def command_endpoint(self) -> str:
        """Return the command endpoint from this peer's perspective."""

        host = self.bind_host if self.role == PeerRole.LEADER else self.peer_host
        return f"tcp://{host}:{self.command_port}"

    @property
    def state_endpoint(self) -> str:
        """Return the state endpoint from this peer's perspective."""

        host = self.bind_host if self.role == PeerRole.FOLLOWER else self.peer_host
        return f"tcp://{host}:{self.state_port}"


__all__ = ["CurveCredentials", "PeerRole", "ZmqTeleopOptions"]
