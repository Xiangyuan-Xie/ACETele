"""Hardware-free robot topology, profile, URDF, and bus-budget checker."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from acetele.config.spec_loader import load_robot_spec
from acetele.runtime import RobotRuntime


def check_robot_spec(path: str | Path) -> tuple[str, ...]:
    """Return concise preflight results without opening any configured port."""

    spec = load_robot_spec(path)
    runtime = RobotRuntime(spec)
    lines = [
        f"robot={spec.model} backend={spec.backend.value} urdf={runtime.preflight.urdf_path}"
    ]
    for name, bus in runtime.preflight.buses.items():
        lines.append(
            f"bus={name} type={bus.spec.type.value} rate={bus.spec.cycle_hz:g}Hz "
            f"utilization={bus.budget.utilization:.1%} "
            f"verified_disable={str(bus.supports_verified_disable).lower()} "
            f"verified_identity={str(bus.supports_verified_identity).lower()}"
        )
    return tuple(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Validate one TOML specification and print its preflight summary."""

    parser = argparse.ArgumentParser(
        description="Validate an ACETele robot spec without opening hardware",
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    for line in check_robot_spec(args.config):
        print(line)


if __name__ == "__main__":
    main()


__all__ = ["check_robot_spec", "main"]
