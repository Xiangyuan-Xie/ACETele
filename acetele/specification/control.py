"""Immutable specifications for optional model-based effort control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ControlSpec:
    """Optional local gravity and redundant-posture assistance for one arm."""

    gravity_compensation: bool = False
    redundancy_posture: bool = False
    rest_posture_rad: Optional[tuple[float, ...]] = None

    def __post_init__(self) -> None:
        for name in ("gravity_compensation", "redundancy_posture"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        posture = self.rest_posture_rad
        if posture is not None:
            if isinstance(posture, (str, bytes)):
                raise ValueError("rest_posture_rad must be a numeric sequence")
            posture = tuple(float(value) for value in posture)
            if not posture or any(not math.isfinite(value) for value in posture):
                raise ValueError("rest_posture_rad values must be finite")
            object.__setattr__(self, "rest_posture_rad", posture)
        if self.redundancy_posture and posture is None:
            raise ValueError("redundancy_posture requires rest_posture_rad")


__all__ = ["ControlSpec"]
