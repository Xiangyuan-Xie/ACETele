from __future__ import annotations

import builtins
from dataclasses import dataclass, replace

import numpy as np
import pytest

from acetele.config.robot_config import (
    ArmAssemblyConfig,
    ArmConfig,
    FeeTechGripperConfig,
    MockJointConfig,
    O6DexterousHandConfig,
    RobotConfig,
)
from acetele.core.make_robot import make_robot
from acetele.equipment.end_effector_factory import create_end_effector
from acetele.equipment.joint_device import MockJointDevice
from acetele.robot.ace_follower.ace_follower import AceDualFollowerRobot
from acetele.utils.teleop_sync import FollowerSyncController, FollowerSyncStatus, LeaderSyncMode


def _joint(name: str, joint_id: int, position: float = 0.0) -> MockJointConfig:
    return MockJointConfig(name, joint_id, position, -3.2, 3.2, 2.0)


def _arm(joints: tuple[MockJointConfig, ...]) -> ArmConfig:
    return ArmConfig(
        port=None,
        joint_ids=tuple(joint.joint_id for joint in joints),
        joint_names=tuple(joint.name for joint in joints),
        joint_signs=(1,) * len(joints),
        home_poses=tuple(joint.initial_position for joint in joints),
        servo_models=(),
        mock_joints=joints,
    )


def _o6_config(backend: str = "mock", runtime: str = "standalone") -> RobotConfig:
    left_arm_joints = tuple(_joint(f"joint_{index}", index) for index in range(5))
    right_arm_joints = tuple(_joint(f"joint_{index}", index) for index in range(5, 10))
    left_hand = O6DexterousHandConfig("left", tuple(range(10, 16)))
    right_hand = O6DexterousHandConfig("right", tuple(range(16, 22)))
    return RobotConfig(
        robot_type="ace_follower_dual",
        backend=backend,
        runtime=runtime,
        arm_assemblies=(
            ArmAssemblyConfig("left", _arm(left_arm_joints), left_hand),
            ArmAssemblyConfig("right", _arm(right_arm_joints), right_hand),
        ),
    )


@dataclass(frozen=True)
class _TwoJointEndEffectorConfig:
    joints: tuple[MockJointConfig, ...]

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def joint_ids(self) -> tuple[int, ...]:
        return tuple(joint.joint_id for joint in self.joints)


@create_end_effector.register
def _create_two_joint_end_effector(
    config: _TwoJointEndEffectorConfig,
    *,
    backend: str,
    driver=None,
):
    del driver
    if backend != "mock":
        raise RuntimeError("test end effector only supports the mock backend")
    return MockJointDevice(config.joints)


def test_programmatic_o6_robot_does_not_read_configuration_files(monkeypatch) -> None:
    monkeypatch.setattr(builtins, "open", lambda *_args, **_kwargs: pytest.fail("unexpected file read"))

    robot = make_robot(_o6_config())

    assert isinstance(robot, AceDualFollowerRobot)
    assert robot.dof == 22
    assert robot.joint_names[:10] == tuple(f"joint_{index}" for index in range(10))
    left_hand = _o6_config().arm_assemblies[0].end_effector
    assert isinstance(left_hand, O6DexterousHandConfig)
    assert robot.joint_names[10:16] == left_hand.joint_names
    assert len(robot.arm_assemblies) == 2


def test_o6_robot_routes_and_clips_full_named_joint_target() -> None:
    robot = make_robot(_o6_config())
    assert isinstance(robot, AceDualFollowerRobot)
    target = np.linspace(-4.0, 4.0, 22)

    robot.set_position(target)
    positions, velocities, efforts = robot.act()

    expected = target.copy()
    expected[:10] = (expected[:10] + np.pi) % (2.0 * np.pi) - np.pi
    expected[10:] = np.clip(expected[10:], -np.pi, np.pi)
    np.testing.assert_allclose(positions, expected)
    np.testing.assert_allclose(velocities, np.zeros(22))
    np.testing.assert_allclose(efforts, np.zeros(22))


def test_programmatic_mock_specs_override_builtin_arm_and_hand_limits() -> None:
    config = _o6_config()
    left_assembly = config.arm_assemblies[0]
    assert isinstance(left_assembly.end_effector, O6DexterousHandConfig)
    arm_specs = tuple(
        replace(joint, lower_limit=-0.25, upper_limit=0.25)
        for joint in left_assembly.arm.mock_joints
    )
    hand_specs = tuple(
        MockJointConfig(name, joint_id, 0.0, 0.0, 1.0, 0.5)
        for name, joint_id in zip(
            left_assembly.end_effector.joint_names,
            left_assembly.end_effector.joint_ids,
        )
    )
    left_assembly = replace(
        left_assembly,
        arm=replace(left_assembly.arm, mock_joints=arm_specs),
        end_effector=replace(left_assembly.end_effector, mock_joints=hand_specs),
    )
    robot = make_robot(replace(config, arm_assemblies=(left_assembly, config.arm_assemblies[1])))
    assert isinstance(robot, AceDualFollowerRobot)
    target = np.zeros(22)
    target[0] = -1.0
    target[10] = -1.0

    robot.set_position(target)
    positions, _, _ = robot.act()

    assert positions[0] == -0.25
    assert positions[10] == 0.0


def test_dual_robot_composes_different_end_effector_models_and_dof() -> None:
    config = _o6_config()
    right_hand = _TwoJointEndEffectorConfig(
        (
            _joint("rh_test_flex", 16),
            _joint("rh_test_spread", 17),
        )
    )
    assemblies = (
        config.arm_assemblies[0],
        replace(config.arm_assemblies[1], end_effector=right_hand),
    )
    mixed_config = replace(config, arm_assemblies=assemblies)

    robot = make_robot(mixed_config)
    assert isinstance(robot, AceDualFollowerRobot)
    target = np.linspace(-1.0, 1.0, 18)
    robot.set_position(target)

    positions, _, _ = robot.act()
    assert robot.dof == 18
    assert robot.joint_names[-2:] == right_hand.joint_names
    np.testing.assert_allclose(positions, target)


def test_o6_physical_backend_fails_explicitly() -> None:
    hand = _o6_config().arm_assemblies[0].end_effector
    assert isinstance(hand, O6DexterousHandConfig)

    with pytest.raises(RuntimeError, match="physical backend is not implemented"):
        create_end_effector(hand, backend="physical")


def test_end_effector_factory_rejects_unknown_backend_before_construction() -> None:
    gripper = FeeTechGripperConfig(
        port=None,
        joint_id=4,
        joint_name="joint_5",
        joint_sign=1,
        home_pose=0.0,
        servo_model="HL3915",
        travel_range_rad=1.0,
    )

    with pytest.raises(ValueError, match="backend must be 'mock' or 'physical'"):
        create_end_effector(gripper, backend="typo", driver=object())


def test_dual_physical_backend_fails_explicitly() -> None:
    config = _o6_config()
    physical_assemblies = tuple(
        replace(
            assembly,
            arm=replace(
                assembly.arm,
                port=f"/dev/{assembly.name}",
                servo_models=("HL3915",) * len(assembly.arm.joint_ids),
            ),
        )
        for assembly in config.arm_assemblies
    )

    with pytest.raises(RuntimeError, match="physical dual follower backend is not implemented"):
        make_robot(replace(config, backend="physical", arm_assemblies=physical_assemblies))


def test_dual_ros2_runtime_fails_at_single_entrypoint_map() -> None:
    with pytest.raises(ValueError, match="runtime 'ros2' is not supported"):
        make_robot(_o6_config(runtime="ros2"))


def test_o6_config_rejects_wrong_joint_count() -> None:
    with pytest.raises(ValueError, match="requires 6 joint IDs"):
        O6DexterousHandConfig("left", tuple(range(10, 15)))


def test_follower_sync_controller_requires_ready_and_times_out() -> None:
    sync = FollowerSyncController(heartbeat_timeout_ns=100)
    sync.set_mode(LeaderSyncMode.TRACKING)
    assert not sync.accept_command(0)

    sync.set_mode(LeaderSyncMode.SYNC_REQUEST)
    assert sync.status == FollowerSyncStatus.READY
    sync.set_mode(LeaderSyncMode.TRACKING)
    assert sync.accept_command(10)
    assert sync.status == FollowerSyncStatus.TRACKING
    assert sync.update(111) == FollowerSyncStatus.LOST
