import inspect
import types

import numpy as np
import pytest

from acetele.robot.ace_follower.ace_follower import AceFollowerRobot
from acetele.robot.ace_leader.ace_leader import AceLeaderRobot


class FakeArm:
    ids = np.array([0, 1])

    def __init__(self):
        self.position_calls = []

    def get_linker_state(self):
        return types.SimpleNamespace(
            public_positions=np.array([0.1, 0.2]),
            raw_positions=np.array([1.1, 1.2]),
            velocities=np.array([2.1, 2.2]),
            motor_torque_magnitude=np.array([3.1, 3.2]),
            motor_torque_signed=np.array([4.1, 4.2]),
        )

    def act(self):
        state = self.get_linker_state()
        return state.public_positions, state.velocities, state.motor_torque_magnitude

    def set_position(self, positions, **kwargs):
        self.position_calls.append((np.asarray(positions, dtype=float), kwargs))


class FakeGripper:
    joint_id = 4

    def __init__(self):
        self.position_calls = []

    def get_state(self):
        return types.SimpleNamespace(
            public_position=0.9,
            raw_position=1.9,
            velocity=2.9,
            motor_torque_magnitude=3.9,
            motor_torque_signed=4.9,
        )

    def act(self):
        state = self.get_state()
        return np.array([state.public_position]), np.array([state.velocity]), np.array([state.motor_torque_magnitude])

    def set_position(self, position, **kwargs):
        self.position_calls.append((float(position), kwargs))


def test_follower_robot_combines_arm_and_gripper_state_in_public_order():
    robot = AceFollowerRobot.__new__(AceFollowerRobot)
    robot.ids = np.array([0, 1, 4])
    robot.gripper_id = 4
    robot.gripper_index = 2
    robot._equipments = types.SimpleNamespace(single_arm=FakeArm(), gripper=FakeGripper())

    state = AceFollowerRobot.get_robot_state(robot)

    np.testing.assert_allclose(state.public_positions, np.array([0.1, 0.2, 0.9]))
    np.testing.assert_allclose(state.raw_positions, np.array([1.1, 1.2, 1.9]))
    np.testing.assert_allclose(state.velocities, np.array([2.1, 2.2, 2.9]))
    np.testing.assert_allclose(state.motor_torque_magnitude, np.array([3.1, 3.2, 3.9]))
    np.testing.assert_allclose(state.motor_torque_signed, np.array([4.1, 4.2, 4.9]))


def test_follower_robot_routes_full_command_to_arm_and_plain_gripper():
    arm = FakeArm()
    gripper = FakeGripper()
    robot = AceFollowerRobot.__new__(AceFollowerRobot)
    robot.ids = np.array([0, 1, 4])
    robot.gripper_id = 4
    robot.gripper_index = 2
    robot._equipments = types.SimpleNamespace(single_arm=arm, gripper=gripper)

    AceFollowerRobot.set_position(robot, [0.1, 0.2, 0.8])

    np.testing.assert_allclose(arm.position_calls[0][0], np.array([0.1, 0.2]))
    np.testing.assert_array_equal(arm.position_calls[0][1]["ids"], np.array([0, 1]))
    assert gripper.position_calls == [(0.8, {})]


def test_leader_robot_routes_full_command_to_arm_and_plain_gripper():
    arm = FakeArm()
    gripper = FakeGripper()
    robot = AceLeaderRobot.__new__(AceLeaderRobot)
    robot.ids = np.array([0, 1, 4])
    robot.gripper_id = 4
    robot.gripper_index = 2
    robot._equipments = types.SimpleNamespace(single_arm=arm, gripper=gripper)

    AceLeaderRobot.set_position(robot, [0.1, 0.2, 0.8])

    np.testing.assert_allclose(arm.position_calls[0][0], np.array([0.1, 0.2]))
    np.testing.assert_array_equal(arm.position_calls[0][1]["ids"], np.array([0, 1]))
    assert gripper.position_calls == [(0.8, {})]


def test_leader_robot_splits_full_profile_arrays_with_command():
    arm = FakeArm()
    gripper = FakeGripper()
    robot = AceLeaderRobot.__new__(AceLeaderRobot)
    robot.ids = np.array([0, 1, 4])
    robot.gripper_id = 4
    robot.gripper_index = 2
    robot._equipments = types.SimpleNamespace(single_arm=arm, gripper=gripper)

    AceLeaderRobot.set_position(
        robot,
        [0.1, 0.2, 0.8],
        velocities=[1.1, 1.2, 0.9],
        accelerations=[2.1, 2.2, 1.9],
        torque=[3.1, 3.2, 2.9],
    )

    np.testing.assert_allclose(arm.position_calls[0][1]["velocities"], np.array([1.1, 1.2]))
    np.testing.assert_allclose(arm.position_calls[0][1]["accelerations"], np.array([2.1, 2.2]))
    np.testing.assert_allclose(arm.position_calls[0][1]["torque"], np.array([3.1, 3.2]))
    assert gripper.position_calls == [(0.8, {"velocity": 0.9, "acceleration": 1.9, "torque": 2.9})]


def test_leader_robot_splits_explicit_id_command_by_id_not_numeric_sort():
    arm = FakeArm()
    gripper = FakeGripper()
    robot = AceLeaderRobot.__new__(AceLeaderRobot)
    robot.ids = np.array([1, 0, 4])
    robot.gripper_id = 4
    robot.gripper_index = 2
    robot._equipments = types.SimpleNamespace(single_arm=arm, gripper=gripper)

    AceLeaderRobot.set_position(
        robot,
        [0.8, 0.2, 0.1],
        ids=[4, 0, 1],
        velocities=[0.9, 1.2, 1.1],
    )

    np.testing.assert_allclose(arm.position_calls[0][0], np.array([0.2, 0.1]))
    np.testing.assert_array_equal(arm.position_calls[0][1]["ids"], np.array([0, 1]))
    np.testing.assert_allclose(arm.position_calls[0][1]["velocities"], np.array([1.2, 1.1]))
    assert gripper.position_calls == [(0.8, {"velocity": 0.9})]


@pytest.mark.parametrize("robot_cls", [AceLeaderRobot, AceFollowerRobot])
def test_robot_rejects_full_public_profile_arrays_for_explicit_subset(robot_cls):
    arm = FakeArm()
    gripper = FakeGripper()
    robot = robot_cls.__new__(robot_cls)
    robot.ids = np.array([0, 1, 4])
    robot.gripper_id = 4
    robot.gripper_index = 2
    robot._equipments = types.SimpleNamespace(single_arm=arm, gripper=gripper)

    with pytest.raises(ValueError, match="velocities.*scalar or match ids length"):
        robot_cls.set_position(
            robot,
            [0.1, 0.2],
            ids=[0, 1],
            velocities=[1.1, 1.2, 0.9],
        )


def test_robot_close_closes_unique_drivers_once():
    shared_driver = types.SimpleNamespace(close_calls=0)
    shared_driver.close = lambda: setattr(shared_driver, "close_calls", shared_driver.close_calls + 1)
    arm = types.SimpleNamespace(close=lambda: None)
    gripper = types.SimpleNamespace(close=lambda: None)
    robot = AceLeaderRobot.__new__(AceLeaderRobot)
    robot._equipments = types.SimpleNamespace(single_arm=arm, gripper=gripper)
    robot._drivers = (shared_driver,)

    AceLeaderRobot.close(robot)

    assert shared_driver.close_calls == 1


def test_follower_robot_creates_separate_gripper_driver_for_different_port(monkeypatch):
    import acetele.robot.ace_follower.ace_follower as follower_module

    created = []

    class FakeDriver:
        def __init__(self, ids, port):
            self.ids = list(ids)
            self.port = port
            created.append(self)

        def close(self):
            pass

    class FakeLinker:
        def __init__(self, config, driver, pin_model=None):
            self.ids = np.array(config["joint_ids"])
            self.driver = driver

    class FakeConfigLoader:
        def get_robot_type(self):
            return "ace_follower"

        def get_backend(self):
            return "default"

        def get_linker_config(self):
            return (
                {
                    "joint_ids": [0, 1],
                    "port": "/dev/arm",
                    "enable_gravity_compensation": False,
                },
            )

        def get_gripper_config(self):
            return (
                {
                    "joint_id": 4,
                    "port": "/dev/gripper",
                    "joint_sign": 1,
                    "home_pose": 0.0,
                    "servo_type": "HL3915",
                    "gripper_type": "ace_follower",
                },
            )

    monkeypatch.setattr(follower_module, "FeeTechDriver", FakeDriver)
    monkeypatch.setattr(follower_module, "Linker", FakeLinker)
    monkeypatch.setattr(
        follower_module,
        "Gripper",
        lambda config, driver: types.SimpleNamespace(id=config["joint_id"], driver=driver),
    )

    robot = AceFollowerRobot(FakeConfigLoader())

    assert robot.ids.tolist() == [0, 1, 4]
    assert robot.gripper_index == 2
    assert [(driver.ids, driver.port) for driver in created] == [
        ([0, 1], "/dev/arm"),
        ([4], "/dev/gripper"),
    ]
    assert robot._equipments.single_arm.driver is created[0]
    assert robot._equipments.gripper.driver is created[1]


def test_leader_robot_shares_driver_when_gripper_port_matches_arm(monkeypatch):
    import acetele.robot.ace_leader.ace_leader as leader_module

    created = []

    class FakeDriver:
        def __init__(self, ids, port):
            self.ids = list(ids)
            self.port = port
            created.append(self)

        def close(self):
            pass

    class FakeLinker:
        def __init__(self, config, driver, pin_model=None):
            self.ids = np.array(config["joint_ids"])
            self.driver = driver

    class FakeConfigLoader:
        def get_robot_type(self):
            return "ace_leader"

        def get_backend(self):
            return "default"

        def get_linker_config(self):
            return (
                {
                    "joint_ids": [0, 1],
                    "port": "/dev/shared",
                    "enable_gravity_compensation": False,
                },
            )

        def get_gripper_config(self):
            return (
                {
                    "joint_id": 4,
                    "port": "/dev/shared",
                    "joint_sign": 1,
                    "home_pose": 0.0,
                    "servo_type": "HL3915",
                    "gripper_type": "ace_leader",
                },
            )

    monkeypatch.setattr(leader_module, "FeeTechDriver", FakeDriver)
    monkeypatch.setattr(leader_module, "Linker", FakeLinker)
    monkeypatch.setattr(
        leader_module,
        "Gripper",
        lambda config, driver: types.SimpleNamespace(id=config["joint_id"], driver=driver),
    )
    monkeypatch.setattr(
        leader_module.AceLeaderRobot,
        "get_pin_model",
        lambda self: pytest.fail("pin model should not load when gravity compensation is disabled"),
    )

    robot = AceLeaderRobot(FakeConfigLoader())

    assert robot.ids.tolist() == [0, 1, 4]
    assert robot.gripper_index == 2
    assert [(driver.ids, driver.port) for driver in created] == [([0, 1, 4], "/dev/shared")]
    assert robot._equipments.single_arm.driver is created[0]
    assert robot._equipments.gripper.driver is created[0]


def test_leader_arm_pin_model_fixes_gripper_joint_when_gravity_compensation_enabled(monkeypatch):
    import acetele.robot.ace_leader.ace_leader as leader_module

    class FakeModel:
        nv = 2

        def __init__(self, name):
            self.name = name

    fixed_joint_calls = []

    class FakeConfigLoader:
        def get_robot_type(self):
            return "ace_leader"

        def get_backend(self):
            return "default"

        def get_linker_config(self):
            return (
                {
                    "joint_ids": [0, 1],
                    "port": "/dev/shared",
                    "enable_gravity_compensation": True,
                },
            )

        def get_gripper_config(self):
            return (
                {
                    "joint_id": 4,
                    "port": "/dev/shared",
                    "joint_sign": 1,
                    "home_pose": 0.0,
                    "servo_type": "HL3915",
                    "gripper_type": "ace_leader",
                },
            )

    class FakeLinker:
        def __init__(self, config, driver, pin_model=None):
            self.ids = np.array(config["joint_ids"])
            self.pin_model = pin_model

    monkeypatch.setattr(leader_module, "Linker", FakeLinker)
    monkeypatch.setattr(
        leader_module,
        "Gripper",
        lambda config, driver: types.SimpleNamespace(id=config["joint_id"]),
    )
    monkeypatch.setattr(
        leader_module,
        "FeeTechDriver",
        lambda ids, port: types.SimpleNamespace(ids=list(ids), port=port, close=lambda: None),
    )
    monkeypatch.setattr(
        leader_module.AceLeaderRobot,
        "_get_pin_model_with_fixed_joints",
        lambda self, fixed_joint_names: fixed_joint_calls.append(tuple(fixed_joint_names)) or FakeModel("arm"),
    )

    robot = AceLeaderRobot(FakeConfigLoader())

    assert fixed_joint_calls == [("joint_5",)]
    assert robot._equipments.single_arm.pin_model.name == "arm"


def test_robot_requires_gripper_port(monkeypatch):
    import acetele.robot.ace_leader.ace_leader as leader_module

    class FakeConfigLoader:
        def get_robot_type(self):
            return "ace_leader"

        def get_backend(self):
            return "default"

        def get_linker_config(self):
            return (
                {
                    "joint_ids": [0, 1],
                    "port": "/dev/shared",
                    "enable_gravity_compensation": False,
                },
            )

        def get_gripper_config(self):
            return (
                {
                    "joint_id": 4,
                    "joint_sign": 1,
                    "home_pose": 0.0,
                    "servo_type": "HL3915",
                    "gripper_type": "ace_leader",
                },
            )

    monkeypatch.setattr(leader_module.AceLeaderRobot, "get_pin_model", lambda self: None)

    with pytest.raises(ValueError, match="gripper.single.port"):
        AceLeaderRobot(FakeConfigLoader())


def test_robot_requires_gripper_joint_id(monkeypatch):
    import acetele.robot.ace_leader.ace_leader as leader_module

    class FakeConfigLoader:
        def get_robot_type(self):
            return "ace_leader"

        def get_backend(self):
            return "default"

        def get_linker_config(self):
            return (
                {
                    "joint_ids": [0, 1],
                    "port": "/dev/shared",
                    "enable_gravity_compensation": False,
                },
            )

        def get_gripper_config(self):
            return (
                {
                    "port": "/dev/shared",
                    "joint_sign": 1,
                    "home_pose": 0.0,
                    "servo_type": "HL3915",
                    "gripper_type": "ace_leader",
                },
            )

    monkeypatch.setattr(leader_module.AceLeaderRobot, "get_pin_model", lambda self: None)

    with pytest.raises(ValueError, match="gripper.single.joint_id"):
        AceLeaderRobot(FakeConfigLoader())


def test_robot_core_does_not_expose_force_estimate_or_non_gripper_wrappers():
    for robot_cls in (AceLeaderRobot, AceFollowerRobot):
        assert "encode_gripper" not in inspect.signature(robot_cls.act).parameters
        assert not hasattr(robot_cls, "non_gripper_ids_and_indices")
        assert not hasattr(robot_cls, "non_gripper_values")

    assert not hasattr(AceLeaderRobot, "apply_torque_" + "feedback")
    assert not hasattr(AceFollowerRobot, "update_external_estimate")
    assert not hasattr(AceFollowerRobot, "external_" + "joint_torque")
    assert not hasattr(AceFollowerRobot, "external_" + "wrench")
    assert not hasattr(AceFollowerRobot, "external_" + "wrench_frame_id")
