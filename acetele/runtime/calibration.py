"""Safety-bounded runtime orchestration for FEETECH home calibration."""

from __future__ import annotations

from typing import Callable, Optional

from acetele.runtime.robot import RobotRuntime
from acetele.specification import Backend, RobotSpec

RuntimeFactory = Callable[..., RobotRuntime]
ProgressCallback = Callable[[str], None]


def _report_progress(
    callback: Optional[ProgressCallback],
    stage: str,
) -> None:
    """Keep presentation failures from interrupting hardware cleanup."""

    if callback is None:
        return
    try:
        callback(stage)
    except Exception:
        pass


def calibrate_feetech_home(
    spec: RobotSpec,
    *,
    runtime_factory: RuntimeFactory = RobotRuntime,
    progress: Optional[ProgressCallback] = None,
) -> None:
    """Calibrate every FEETECH packet joint from one pre-reviewed specification."""

    if spec.backend != Backend.PHYSICAL:
        raise RuntimeError("FEETECH home calibration requires backend='physical'")
    runtime = runtime_factory(spec)
    runtime.home_calibration_targets()
    _report_progress(progress, "preflight")
    runtime.connect()
    _report_progress(progress, "connect")

    calibration_error: Optional[BaseException] = None
    try:
        runtime.calibrate_home()
        _report_progress(progress, "write")
    except BaseException as exc:
        calibration_error = exc

    disconnect_error: Optional[BaseException] = None
    try:
        runtime.disconnect()
        _report_progress(progress, "disconnect")
    except BaseException as exc:
        disconnect_error = exc

    if calibration_error is not None:
        if disconnect_error is not None:
            raise calibration_error from disconnect_error
        raise calibration_error
    if disconnect_error is not None:
        raise disconnect_error
    _report_progress(progress, "complete")


__all__ = ["ProgressCallback", "calibrate_feetech_home"]
