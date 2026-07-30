"""Explicit FEETECH home-offset calibration command-line tool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Sequence

from acetele.config.spec_loader import load_robot_spec
from acetele.core import Backend
from acetele.runtime import RobotRuntime

RuntimeFactory = Callable[..., RobotRuntime]


def calibrate_feetech_home(
    path: str | Path,
    *,
    runtime_factory: RuntimeFactory = RobotRuntime,
) -> None:
    """Calibrate all configured FEETECH packet joints at their current pose."""
    spec = load_robot_spec(path)
    if spec.backend != Backend.PHYSICAL:
        raise RuntimeError("FEETECH home calibration requires backend='physical'")
    runtime = runtime_factory(spec)
    runtime.home_calibration_targets()
    runtime.connect()
    calibration_error: Optional[BaseException] = None
    try:
        runtime.calibrate_home()
    except BaseException as exc:
        calibration_error = exc

    close_error: Optional[BaseException] = None
    try:
        runtime.disconnect()
    except BaseException as exc:
        close_error = exc

    if calibration_error is not None:
        if close_error is not None:
            raise calibration_error from close_error
        raise calibration_error
    if close_error is not None:
        raise close_error


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Require physical confirmation before writing nonvolatile servo offsets."""

    parser = argparse.ArgumentParser(
        description=(
            "Calibrate FEETECH packet servos at the configured home positions. "
            "Place every joint at its declared home pose before running."
        ),
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm that the robot is physically at its configured home pose",
    )
    args = parser.parse_args(argv)
    if not args.yes:
        parser.error("--yes is required because calibration writes nonvolatile servo state")
    calibrate_feetech_home(args.config)


if __name__ == "__main__":
    main()


__all__ = ["calibrate_feetech_home", "main"]
