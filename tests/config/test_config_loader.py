import xml.etree.ElementTree as ET
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
enable_gravity_compensation = false
servo_types = ["HL3915"]

[gripper.single]
port = "/dev/test"
joint_id = 2
joint_sign = 1
home_pose = 0.0
servo_type = "HL3915"
gripper_type = "ace_leader"
"""
    )

    loader = ConfigLoader(config_path)

    assert loader.get_robot_type() == "ace_follower"
    assert loader.get_backend() == "ros2"
    assert loader.get_linker_config()[0]["port"] == "/dev/test"
    assert loader.get_gripper_config()[0]["joint_id"] == 2


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
enable_gravity_compensation = false
servo_types = ["HL3915"]

[gripper.single]
port = "/dev/test"
joint_id = 2
joint_sign = 1
home_pose = 0.0
servo_type = "HL3915"
gripper_type = "ace_leader"
"""
    )

    loader = ConfigLoader(config_path, backend_override="ros2")

    assert loader.get_robot_type() == "ace_leader"
    assert loader.get_backend() == "ros2"
    assert loader.get_linker_config()[0]["port"] == "/dev/test"
    assert loader.get_gripper_config()[0]["joint_id"] == 2
    assert not hasattr(loader, "get_public_joint_ids")


def test_ace_follower_pin_model_path_exists_and_loads():
    import pinocchio as pin

    loader = ConfigLoader(Path("acetele/config/ace_follower.toml"))

    class TestRobot(BaseRobot):
        def act(self):
            raise NotImplementedError

    robot = TestRobot(loader)

    model = robot.get_pin_model()

    assert robot._urdf_model_path.endswith("ace_follower/description/ace_follower.urdf")
    assert model.nv == 5
    assert model.getFrameId("link_5") < len(model.frames)
    reduced_model = pin.buildReducedModel(model, [model.getJointId("joint_5")], pin.neutral(model))
    assert reduced_model.nv == 4


def test_ace_follower_urdf_is_arm_only_x500_model():
    urdf_path = Path("acetele/robot/ace_follower/description/ace_follower.urdf")
    root = ET.parse(urdf_path).getroot()

    assert root.attrib["name"] == "ace_follower"

    link_names = {link.attrib["name"] for link in root.findall("link")}
    joint_names = {joint.attrib["name"] for joint in root.findall("joint")}
    assert link_names == {"base_link", "link_1", "link_2", "link_3", "link_4", "link_5"}
    assert joint_names == {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5"}

    base_link = root.find("./link[@name='base_link']")
    assert base_link is not None
    assert base_link.find("inertial") is None
    assert base_link.find("visual") is None
    assert base_link.find("collision") is None

    mesh_filenames = {mesh.attrib["filename"] for mesh in root.findall(".//mesh")}
    assert mesh_filenames == {f"meshes/link_{index}.STL" for index in range(1, 6)}
