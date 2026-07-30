from pathlib import Path

import pytest

from acetele.tools.calibrate_feetech_home import calibrate_feetech_home

project_root = Path(__file__).resolve().parents[2]


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

    calibrate_feetech_home(
        project_root / "acetele/config/ace_leader/feetech_hls_ttl.toml",
        runtime_factory=RecordingRuntime,
    )

    assert RecordingRuntime.instances[0].events == [
        "preflight",
        "connect",
        "calibrate",
        "disconnect",
    ]


def test_calibration_rejects_mock_before_runtime_construction(tmp_path):
    source = (
        project_root / "acetele/config/ace_leader/feetech_hls_ttl.toml"
    ).read_text(encoding="utf-8")
    config = tmp_path / "mock.toml"
    config.write_text(source.replace('backend = "physical"', 'backend = "mock"'))
    RecordingRuntime.instances.clear()

    with pytest.raises(RuntimeError, match="backend='physical'"):
        calibrate_feetech_home(config, runtime_factory=RecordingRuntime)

    assert not RecordingRuntime.instances
