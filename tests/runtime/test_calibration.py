from dataclasses import replace
from pathlib import Path

import pytest

from acetele.config import load_robot_spec
from acetele.runtime import calibrate_feetech_home
from acetele.specification import Backend

project_root = Path(__file__).resolve().parents[2]
leader_path = (
    project_root / "acetele/config/presets/ace_leader/feetech_hls_ttl.toml"
)


class RecordingRuntime:
    instances: list["RecordingRuntime"] = []

    def __init__(self, spec):
        self.spec = spec
        self.events = []
        self.instances.append(self)

    def home_calibration_targets(self):
        self.events.append("preflight")
        return {"arm": {0: 0}}

    def connect(self):
        self.events.append("connect")

    def calibrate_home(self):
        self.events.append("calibrate")

    def disconnect(self):
        self.events.append("disconnect")


def test_calibration_preflights_before_connecting_hardware():
    RecordingRuntime.instances.clear()
    spec = load_robot_spec(leader_path)

    calibrate_feetech_home(
        spec,
        runtime_factory=RecordingRuntime,
    )

    assert RecordingRuntime.instances[0].spec is spec
    assert RecordingRuntime.instances[0].events == [
        "preflight",
        "connect",
        "calibrate",
        "disconnect",
    ]


def test_calibration_reports_completed_stages_in_safety_order():
    RecordingRuntime.instances.clear()
    stages = []

    calibrate_feetech_home(
        load_robot_spec(leader_path),
        runtime_factory=RecordingRuntime,
        progress=stages.append,
    )

    assert stages == ["preflight", "connect", "write", "disconnect", "complete"]


def test_progress_callback_failure_does_not_interrupt_calibration_cleanup():
    RecordingRuntime.instances.clear()

    def failing_progress(_stage):
        raise RuntimeError("display failed")

    calibrate_feetech_home(
        load_robot_spec(leader_path),
        runtime_factory=RecordingRuntime,
        progress=failing_progress,
    )

    assert RecordingRuntime.instances[0].events[-1] == "disconnect"


def test_calibration_rejects_mock_before_runtime_construction():
    spec = replace(load_robot_spec(leader_path), backend=Backend.MOCK)
    RecordingRuntime.instances.clear()

    with pytest.raises(RuntimeError, match="backend='physical'"):
        calibrate_feetech_home(spec, runtime_factory=RecordingRuntime)

    assert not RecordingRuntime.instances
