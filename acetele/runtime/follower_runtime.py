"""Structural runtime boundary consumed by follower teleoperation sessions."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from acetele.core import RobotCommand, RobotState
from acetele.runtime.robot import JointGroupInfo, RuntimeDiagnostics
from acetele.specification import RobotSpec


@runtime_checkable
class FollowerRuntime(Protocol):
    """Runtime operations required by a follower, independent of its backend."""

    spec: RobotSpec

    @property
    def generation(self) -> int: ...

    @property
    def command_timeout_ns(self) -> int: ...

    @property
    def joint_groups(self) -> Mapping[str, JointGroupInfo]: ...

    def command_lifetime_ns(self, group_name: str, *, minimum_ns: int) -> int: ...

    def connect(self) -> None: ...

    def read(self) -> RobotState: ...

    def write(self, command: RobotCommand) -> None: ...

    def write_auxiliary(self, command: RobotCommand) -> None: ...

    def hold(self) -> None: ...

    def set_enabled(self, enabled: bool) -> None: ...

    def emergency_stop(self) -> None: ...

    def disconnect(self, *, preserve_hold: bool = False) -> None: ...

    def diagnostics(self) -> RuntimeDiagnostics: ...


__all__ = ["FollowerRuntime"]
