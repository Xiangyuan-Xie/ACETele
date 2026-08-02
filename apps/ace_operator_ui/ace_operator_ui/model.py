"""Transport-neutral snapshots consumed by the operator window."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class JointView:
    """Named SI-unit joint sample detached from ROS and ZMQ message classes."""

    timestamp_ns: int = 0
    names: tuple[str, ...] = ()
    positions: tuple[float, ...] = ()
    velocities: tuple[float, ...] = ()
    efforts: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(self.names)
        positions = tuple(float(value) for value in self.positions)
        velocities = tuple(float(value) for value in self.velocities)
        efforts = tuple(float(value) for value in self.efforts)
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("joint timestamp must be a non-negative integer")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("joint names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("joint names must be unique")
        if len(positions) != len(names):
            raise ValueError("joint positions must match names")
        if velocities and len(velocities) != len(names):
            raise ValueError("joint velocities must match names")
        if efforts and len(efforts) != len(names):
            raise ValueError("joint efforts must match names")
        if not all(np.isfinite(value) for values in (positions, velocities, efforts) for value in values):
            raise ValueError("joint values must be finite")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "efforts", efforts)


@dataclass(frozen=True)
class OperatorSnapshot:
    """One coherent UI refresh detached from producer-owned buffers."""

    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)
    health: Mapping[str, str] = field(default_factory=dict)
    joints: JointView = JointView()
    metrics: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        images = {}
        for name, value in self.images.items():
            image = np.asarray(value).copy()
            image.setflags(write=False)
            images[str(name)] = image
        object.__setattr__(self, "images", MappingProxyType(images))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "health", MappingProxyType(dict(self.health)))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


class OperatorDataSource(Protocol):
    """Minimum adapter implemented by ROS 2 and ZeroMQ monitor sources."""

    def snapshot(self) -> OperatorSnapshot: ...


__all__ = ["JointView", "OperatorDataSource", "OperatorSnapshot"]
