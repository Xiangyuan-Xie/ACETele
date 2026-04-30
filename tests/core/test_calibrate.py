from pathlib import Path

import numpy as np
import pytest

from acetele.core.calibrate import (
    ArmCalibrationResult,
    Calibration,
    CalibrationError,
)


class FakeDriver:
    instances: list["FakeDriver"] = []
    fail_ports: set[str] = set()

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


@pytest.fixture(autouse=True)
def reset_fake_driver():
    FakeDriver.instances = []
    FakeDriver.fail_ports = set()


def write_config(path, config_file, body):
    Path(path, "default.toml").write_text(f'[basic]\nconfig_file = "{config_file}"\n')
    Path(path, config_file).write_text(body)
    return Path(path, "default.toml")


def test_calibrate_encodes_home_poses_with_joint_signs_and_rounding(tmp_path):
    config_path = write_config(
        tmp_path,
        "robot.toml",
        """
[basic]
robot_type = "ace_follower"
backend = "default"

[linker.single]
port = "/dev/test"
joint_ids = [1, 2, 3]
joint_signs = [1, -1, 1]
home_poses = [1.5707963267948966, 1.5707963267948966, 0.001]
gripper_id = -1
gripper_type = "ace_leader"
enable_gravity_compensation = false
enable_estimate_external_torque = false
servo_types = ["HL3915", "HL3915", "HL3915"]
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


def test_calibrate_returns_all_arm_results_for_dual_config(tmp_path):
    config_path = write_config(
        tmp_path,
        "robot.toml",
        """
[basic]
robot_type = "ace_follower"
backend = "default"

[linker.dual.left]
port = "/dev/left"
joint_ids = [1]
joint_signs = [1]
home_poses = [0.0]
gripper_id = -1
gripper_type = "ace_leader"
enable_gravity_compensation = false
enable_estimate_external_torque = false
servo_types = ["HL3915"]

[linker.dual.right]
port = "/dev/right"
joint_ids = [2]
joint_signs = [-1]
home_poses = [3.141592653589793]
gripper_id = -1
gripper_type = "ace_leader"
enable_gravity_compensation = false
enable_estimate_external_torque = false
servo_types = ["HL3915"]
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
backend = "default"

[linker.single]
port = "/dev/direct"
joint_ids = [1]
joint_signs = [-1]
home_poses = [3.141592653589793]
gripper_id = -1
gripper_type = "ace_leader"
enable_gravity_compensation = false
enable_estimate_external_torque = false
servo_types = ["HL3915"]
"""
    )

    results = Calibration(config_path=robot_config, driver_factory=FakeDriver).calibrate()

    assert results[0].port == "/dev/direct"
    assert results[0].encoded_home_poses == (-2048,)


def test_calibrate_raises_calibration_error_with_arm_and_port(tmp_path):
    config_path = write_config(
        tmp_path,
        "robot.toml",
        """
[basic]
robot_type = "ace_follower"
backend = "default"

[linker.single]
port = "/dev/fail"
joint_ids = [1]
joint_signs = [1]
home_poses = [0.0]
gripper_id = -1
gripper_type = "ace_leader"
enable_gravity_compensation = false
enable_estimate_external_torque = false
servo_types = ["HL3915"]
""",
    )
    FakeDriver.fail_ports = {"/dev/fail"}

    with pytest.raises(CalibrationError, match=r"arm 0.*\/dev\/fail"):
        Calibration(config_path=config_path, driver_factory=FakeDriver).calibrate()

    assert FakeDriver.instances[0].closed
