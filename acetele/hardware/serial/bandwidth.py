"""Conservative serial line budget calculated before hardware connection."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BusBudget:
    """Estimated wire utilization for one complete control cycle."""

    baudrate: int
    cycle_hz: float
    wire_bytes_per_cycle: int
    turnaround_s_per_cycle: float
    utilization: float
    max_utilization: float

    @property
    def feasible(self) -> bool:
        """Return whether estimated utilization stays within the safety margin."""

        return self.utilization <= self.max_utilization

    @property
    def maximum_cycle_hz(self) -> float:
        """Return the highest rate that satisfies ``max_utilization``."""

        cycle_time = self.wire_bytes_per_cycle * 10.0 / self.baudrate + self.turnaround_s_per_cycle
        return self.max_utilization / cycle_time

    def require_feasible(self, *, context: str) -> None:
        """Reject a configuration that would necessarily accumulate bus work."""

        if self.feasible:
            return
        raise ValueError(
            f"{context} requires {self.utilization:.1%} serial utilization at "
            f"{self.cycle_hz:g} Hz, exceeding the {self.max_utilization:.1%} limit; "
            f"maximum feasible rate is {self.maximum_cycle_hz:.1f} Hz"
        )


def calculate_bus_budget(
    *,
    baudrate: int,
    cycle_hz: float,
    wire_bytes_per_cycle: int,
    turnaround_s_per_cycle: float = 0.0,
    max_utilization: float = 0.70,
) -> BusBudget:
    """Estimate 8N1 wire time plus protocol-specific turnaround overhead."""

    if type(baudrate) is not int or baudrate <= 0:
        raise ValueError("baudrate must be a positive integer")
    if type(wire_bytes_per_cycle) is not int or wire_bytes_per_cycle <= 0:
        raise ValueError("wire_bytes_per_cycle must be a positive integer")
    for name, value in (
        ("cycle_hz", cycle_hz),
        ("turnaround_s_per_cycle", turnaround_s_per_cycle),
        ("max_utilization", max_utilization),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if cycle_hz <= 0.0 or turnaround_s_per_cycle < 0.0 or not 0.0 < max_utilization <= 1.0:
        raise ValueError("invalid serial bus budget parameters")
    occupied_s = wire_bytes_per_cycle * 10.0 / baudrate + turnaround_s_per_cycle
    return BusBudget(
        baudrate=baudrate,
        cycle_hz=float(cycle_hz),
        wire_bytes_per_cycle=wire_bytes_per_cycle,
        turnaround_s_per_cycle=float(turnaround_s_per_cycle),
        utilization=occupied_s * cycle_hz,
        max_utilization=float(max_utilization),
    )
