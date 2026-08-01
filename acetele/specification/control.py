"""Immutable specifications for optional model-based position control."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PositionControlTuning:
    """Numerical policy for quasi-static adaptive position compensation."""

    adaptive_deadband_rad: float = 0.02
    adaptation_rate_per_s: float = 8.0
    offset_filter_bandwidth_per_s: float = 4.0
    maximum_adaptive_offset_rad: float = 0.10
    stable_time_s: float = 0.20
    target_stable_threshold_rad: float = 0.008
    target_reset_threshold_rad: float = 0.05
    target_direction_threshold_rad: float = 0.002
    velocity_threshold_rad_s: float = 0.05
    minimum_dt_s: float = 0.001
    maximum_dt_s: float = 0.05

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"position control tuning {name} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(
                    f"position control tuning {name} must be finite and positive"
                )
            object.__setattr__(self, name, normalized)
        if self.minimum_dt_s > self.maximum_dt_s:
            raise ValueError("position control minimum_dt_s cannot exceed maximum_dt_s")
        if self.target_stable_threshold_rad >= self.target_reset_threshold_rad:
            raise ValueError(
                "position control target_stable_threshold_rad must be smaller than "
                "target_reset_threshold_rad"
            )
        if self.target_direction_threshold_rad >= self.target_reset_threshold_rad:
            raise ValueError(
                "position control target_direction_threshold_rad must be smaller than "
                "target_reset_threshold_rad"
            )


@dataclass(frozen=True)
class ControlSpec:
    """Opt-in model-based position conditioning for one arm."""

    adaptive_position: bool = False
    gravity_position: bool = False
    gravity_compliance_rad_per_nm: Optional[tuple[float, ...]] = None
    position_tuning: PositionControlTuning = field(
        default_factory=PositionControlTuning
    )

    def __post_init__(self) -> None:
        if type(self.adaptive_position) is not bool or type(self.gravity_position) is not bool:
            raise ValueError("control feature switches must be booleans")
        if not isinstance(self.position_tuning, PositionControlTuning):
            raise ValueError("position_tuning must be a PositionControlTuning")
        compliance = self.gravity_compliance_rad_per_nm
        if compliance is not None:
            if isinstance(compliance, (str, bytes)):
                raise ValueError("gravity compliance must be a numeric sequence")
            compliance = tuple(float(value) for value in compliance)
            if not compliance or any(
                not math.isfinite(value) or value <= 0.0 for value in compliance
            ):
                raise ValueError("gravity compliance values must be finite and positive")
            object.__setattr__(self, "gravity_compliance_rad_per_nm", compliance)
        if self.gravity_position and compliance is None:
            raise ValueError(
                "gravity position compensation requires per-joint compliance calibration"
            )


__all__ = ["ControlSpec", "PositionControlTuning"]
