from pathlib import Path

from acetele.config.config_loader import ConfigLoader
from acetele.robot.base_robot import BaseRobot


def test_config_loader_accepts_direct_robot_config(tmp_path):
    config_path = Path(tmp_path, "ace_follower.toml")
    config_path.write_text(
        """
[basic]
robot_type = "ace_follower"
backend = "ros2"

[linker.single]
port = "/dev/test"
joint_ids = [1]
joint_signs = [1]
home_poses = [0.0]
gripper_id = -1
gripper_type = "ace_leader"
enable_gravity_compensation = false
enable_estimate_external_torque = false
servo_types = ["HL3915"]
"""
    )

    loader = ConfigLoader(config_path)

    assert loader.get_robot_type() == "ace_follower"
    assert loader.get_backend() == "ros2"
    assert loader.get_linker_config()[0]["port"] == "/dev/test"


def test_config_loader_can_override_backend_without_separate_robot_config(tmp_path):
    config_path = Path(tmp_path, "default.toml")
    robot_config_path = Path(tmp_path, "ace_leader.toml")
    config_path.write_text('[basic]\nconfig_file = "ace_leader.toml"\n')
    robot_config_path.write_text(
        """
[basic]
robot_type = "ace_leader"
backend = "mock"

[linker.single]
port = "/dev/test"
joint_ids = [1]
joint_signs = [1]
home_poses = [0.0]
gripper_id = -1
gripper_type = "ace_leader"
enable_gravity_compensation = false
enable_estimate_external_torque = false
servo_types = ["HL3915"]
"""
    )

    loader = ConfigLoader(config_path, backend_override="ros2")

    assert loader.get_robot_type() == "ace_leader"
    assert loader.get_backend() == "ros2"
    assert loader.get_linker_config()[0]["port"] == "/dev/test"


def test_ace_follower_pin_model_path_exists_and_loads():
    loader = ConfigLoader(Path("acetele/config/ace_follower.toml"))

    class TestRobot(BaseRobot):
        def act(self):
            raise NotImplementedError

    robot = TestRobot(loader)

    model = robot.get_pin_model()

    assert robot._urdf_model_path.endswith("ace_follower/description/ace_follower.urdf")
    assert model.nv == 5
    assert model.getFrameId("link_5") < len(model.frames)
