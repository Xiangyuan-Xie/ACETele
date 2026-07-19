from pathlib import Path

import numpy as np
import pytest

import acetele.core.calibrate as calibrate_module
from acetele.core.calibrate import (
    ArmCalibrationResult,
    Calibration,
    CalibrationError,
)


class FakeDriver:
    instances: list["FakeDriver"] = []
    fail_ports: set[str] = set()
    close_fail_ports: set[str] = set()

    def __init__(self, ids, port):
        self.ids = list(ids)
        self.port = port
        self.calibrate_calls = []
        self.closed = False
        FakeDriver.instances.append(self)

    def calibrate(self, ids, home_poses):
        self.calibrate_calls.append((list(ids), np.array(home_poses).copy()))
        if self.port in FakeDriver.fail_ports:
            raise RuntimeError("driver calibration failed")

    def get_state(self):
        return ({ft_id: ft_id + 100 for ft_id in self.ids}, {}, {})

    def close(self):
        self.closed = True
        if self.port in FakeDriver.close_fail_ports:
            raise RuntimeError("driver close failed")


@pytest.fixture(autouse=True)
def reset_fake_driver():
    FakeDriver.instances = []
    FakeDriver.fail_ports = set()
    FakeDriver.close_fail_ports = set()


def write_config(path, body, config_file="robot.toml"):
    Path(path, "default.toml").write_text(f'[basic]\nconfig_file = "{config_file}"\n')
    Path(path, config_file).write_text(body)
    return Path(path, "default.toml")


def write_single_arm_config(path, port="/dev/test"):
    return write_config(
        path,
        f"""
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "standalone"

[arms.single]
port = "{port}"
joint_ids = [1]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]
""",
    )


def test_calibrate_encodes_home_poses_with_joint_signs_and_rounding(tmp_path):
    config_path = write_config(
        tmp_path,
        """
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "standalone"

[arms.single]
port = "/dev/test"
joint_ids = [1, 2, 3]
joint_names = ["joint_1", "joint_2", "joint_3"]
joint_signs = [1, -1, 1]
home_poses = [1.5707963267948966, 1.5707963267948966, 0.001]
servo_models = ["HL3915", "HL3915", "HL3915"]
""",
    )

    results = Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    driver = FakeDriver.instances[0]
    assert driver.calibrate_calls[0][0] == [1, 2, 3]
    np.testing.assert_array_equal(driver.calibrate_calls[0][1], np.array([1024, -1024, 1]))
    assert results == (
        ArmCalibrationResult(
            arm_index=0,
            port="/dev/test",
            ids=(1, 2, 3),
            encoded_home_poses=(1024, -1024, 1),
            positions_after_calibration={1: 101, 2: 102, 3: 103},
        ),
    )
    assert driver.closed


def test_calibrate_includes_gripper_on_shared_port(tmp_path):
    config_path = write_config(
        tmp_path,
        """
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "standalone"

[arms.single]
port = "/dev/test"
joint_ids = [0, 1]
joint_names = ["joint_1", "joint_2"]
joint_signs = [1, -1]
home_poses = [1.5707963267948966, 0.5]
servo_models = ["HL3915", "HL3915"]

[arms.single.end_effector]
kind = "gripper"
port = "/dev/test"
joint_id = 4
joint_name = "joint_5"
joint_sign = -1
home_pose = 1.0
servo_model = "HL3915"
travel_range_rad = 1.3744467859455345
""",
    )

    results = Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    driver = FakeDriver.instances[0]
    assert driver.calibrate_calls[0][0] == [0, 1, 4]
    np.testing.assert_array_equal(driver.calibrate_calls[0][1], np.array([1024, -326, -896]))
    assert results[0].ids == (0, 1, 4)


def test_calibrate_uses_separate_gripper_port(tmp_path):
    config_path = write_config(
        tmp_path,
        """
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "standalone"

[arms.single]
port = "/dev/arm"
joint_ids = [0, 1]
joint_names = ["joint_1", "joint_2"]
joint_signs = [1, -1]
home_poses = [1.5707963267948966, 0.5]
servo_models = ["HL3915", "HL3915"]

[arms.single.end_effector]
kind = "gripper"
port = "/dev/gripper"
joint_id = 4
joint_name = "joint_5"
joint_sign = -1
home_pose = 1.0
servo_model = "HL3915"
travel_range_rad = 1.3744467859455345
""",
    )

    results = Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    assert [(driver.ids, driver.port) for driver in FakeDriver.instances] == [
        ([0, 1], "/dev/arm"),
        ([4], "/dev/gripper"),
    ]
    assert results[0].encoded_home_poses == (1024, -326)
    assert results[1].encoded_home_poses == (-896,)
    assert all(driver.closed for driver in FakeDriver.instances)


def test_calibration_rejects_invalid_normalized_gripper_home_pose(tmp_path):
    config_path = write_config(
        tmp_path,
        """
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "standalone"

[arms.single]
port = "/dev/test"
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]

[arms.single.end_effector]
kind = "gripper"
port = "/dev/test"
joint_id = 4
joint_name = "joint_5"
joint_sign = 1
home_pose = 1.2
servo_model = "HL3915"
travel_range_rad = 0.7853981633974483
""",
    )

    with pytest.raises(ValueError, match="home pose must be between"):
        Calibration(config_path=config_path, driver_factory=FakeDriver)


def test_calibrate_returns_all_arm_results_for_dual_config(tmp_path):
    config_path = write_config(
        tmp_path,
        """
[basic]
robot_type = "ace_follower_dual"
backend = "physical"
runtime = "standalone"

[arms.left]
port = "/dev/left"
joint_ids = [1]
joint_names = ["left_joint"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]

[arms.right]
port = "/dev/right"
joint_ids = [2]
joint_names = ["right_joint"]
joint_signs = [-1]
home_poses = [3.141592653589793]
servo_models = ["HL3915"]
""",
    )

    results = Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    assert len(results) == 2
    assert results[0].port == "/dev/left"
    assert results[0].encoded_home_poses == (0,)
    assert results[1].port == "/dev/right"
    assert results[1].encoded_home_poses == (-2048,)


def test_calibration_accepts_direct_robot_config_path(tmp_path):
    robot_config = Path(tmp_path, "ace_follower.toml")
    robot_config.write_text(
        """
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "standalone"

[arms.single]
port = "/dev/direct"
joint_ids = [1]
joint_names = ["joint_1"]
joint_signs = [-1]
home_poses = [3.141592653589793]
servo_models = ["HL3915"]
"""
    )

    results = Calibration(config_path=robot_config, driver_factory=FakeDriver).calibrate()

    assert results[0].port == "/dev/direct"
    assert results[0].encoded_home_poses == (-2048,)


def test_calibrate_raises_calibration_error_with_arm_and_port(tmp_path):
    config_path = write_config(
        tmp_path,
        """
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "standalone"

[arms.single]
port = "/dev/fail"
joint_ids = [1]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]
""",
    )
    FakeDriver.fail_ports = {"/dev/fail"}

    with pytest.raises(CalibrationError, match=r"arm 0.*\/dev\/fail"):
        Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    assert FakeDriver.instances[0].closed


def test_calibration_failure_remains_primary_when_driver_close_also_fails(tmp_path):
    config_path = write_single_arm_config(tmp_path, port="/dev/fail")
    FakeDriver.fail_ports = {"/dev/fail"}
    FakeDriver.close_fail_ports = {"/dev/fail"}

    with pytest.raises(CalibrationError) as exc_info:
        Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    message = str(exc_info.value)
    assert "Failed to calibrate arm 0" in message
    assert "driver calibration failed" in message
    assert "additionally failed to close driver" in message
    assert "driver close failed" in message
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "driver calibration failed"
    assert FakeDriver.instances[0].closed


def test_successful_calibration_reports_driver_close_failure(tmp_path):
    config_path = write_single_arm_config(tmp_path, port="/dev/close-fail")
    FakeDriver.close_fail_ports = {"/dev/close-fail"}

    with pytest.raises(CalibrationError) as exc_info:
        Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    message = str(exc_info.value)
    assert "Calibration completed for arm 0" in message
    assert "/dev/close-fail" in message
    assert "driver close failed" in message
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "driver close failed"
    assert FakeDriver.instances[0].closed


def test_calibration_rejects_mock_backend_before_creating_driver(tmp_path):
    config_path = write_config(
        tmp_path,
        """
[basic]
robot_type = "ace_follower"
backend = "mock"
runtime = "standalone"

[arms.single]
port = "/dev/must-not-open"
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]
""",
    )

    with pytest.raises(CalibrationError, match="backend='physical'"):
        Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    assert FakeDriver.instances == []


def test_calibration_preflights_all_assemblies_before_creating_driver(tmp_path):
    config_path = write_config(
        tmp_path,
        """
[basic]
robot_type = "ace_follower_dual"
backend = "physical"
runtime = "standalone"

[arms.left]
port = "/dev/left"
joint_ids = [0]
joint_names = ["left_joint"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]

[arms.left.end_effector]
kind = "gripper"
port = "/dev/left"
joint_id = 4
joint_name = "left_gripper_joint"
joint_sign = 1
home_pose = 0.0
servo_model = "HL3915"
travel_range_rad = 0.7853981633974483

[arms.right]
port = "/dev/right"
joint_ids = [1]
joint_names = ["right_joint"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]

[arms.right.end_effector]
kind = "dexterous_hand"
model = "o6"
side = "right"
joint_ids = [10, 11, 12, 13, 14, 15]
""",
    )

    with pytest.raises(CalibrationError, match=r"O6DexterousHandConfig.*arm 1"):
        Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    assert FakeDriver.instances == []


def test_calibration_preflights_encoded_home_pose_range_before_creating_driver(tmp_path):
    config_path = write_config(
        tmp_path,
        """
[basic]
robot_type = "ace_follower_dual"
backend = "physical"
runtime = "standalone"

[arms.left]
port = "/dev/left"
joint_ids = [0]
joint_names = ["left_joint"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]

[arms.right]
port = "/dev/right"
joint_ids = [1]
joint_names = ["right_joint"]
joint_signs = [1]
home_poses = [100.0]
servo_models = ["HL3915"]
""",
    )

    with pytest.raises(CalibrationError, match=r"arm 1.*signed 15-bit range"):
        Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    assert FakeDriver.instances == []


def test_calibration_preflights_all_servo_ids_before_creating_driver(tmp_path):
    config_path = write_config(
        tmp_path,
        """
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "standalone"

[arms.single]
port = "/dev/arm"
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["HL3915"]

[arms.single.end_effector]
kind = "gripper"
port = "/dev/gripper"
joint_id = 253
joint_name = "joint_5"
joint_sign = 1
home_pose = 0.0
servo_model = "HL3915"
travel_range_rad = 0.7853981633974483
""",
    )

    with pytest.raises(CalibrationError, match=r"gripper.*between 0 and 252"):
        Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    assert FakeDriver.instances == []


@pytest.mark.parametrize(
    ("arm_model", "gripper_model", "match"),
    [
        ("BAD", "HL3915", r"arm 0.*unsupported servo models.*BAD"),
        ("HL3915", "BAD", r"gripper on arm 0.*unsupported servo models.*BAD"),
    ],
)
def test_calibration_preflights_servo_models_before_creating_driver(
    tmp_path,
    arm_model,
    gripper_model,
    match,
):
    config_path = write_config(
        tmp_path,
        f"""
[basic]
robot_type = "ace_follower"
backend = "physical"
runtime = "standalone"

[arms.single]
port = "/dev/arm"
joint_ids = [0]
joint_names = ["joint_1"]
joint_signs = [1]
home_poses = [0.0]
servo_models = ["{arm_model}"]

[arms.single.end_effector]
kind = "gripper"
port = "/dev/gripper"
joint_id = 4
joint_name = "joint_5"
joint_sign = 1
home_pose = 0.0
servo_model = "{gripper_model}"
travel_range_rad = 0.7853981633974483
""",
    )

    with pytest.raises(CalibrationError, match=match):
        Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    assert FakeDriver.instances == []


def test_calibration_main_passes_explicit_config_path(monkeypatch, tmp_path):
    config_path = tmp_path / "physical.toml"
    captured = {}

    class FakeCalibration:
        def __init__(self, config_path=None):
            captured["config_path"] = config_path

        def calibrate(self):
            captured["calibrated"] = True

    monkeypatch.setattr(calibrate_module, "Calibration", FakeCalibration)

    calibrate_module.main(["--config", str(config_path)])

    assert captured == {"config_path": config_path, "calibrated": True}
