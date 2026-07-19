from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from acetele.config.config_loader import ConfigLoader
from acetele.config.robot_config import (
    ArmAssemblyConfig,
    ArmConfig,
    FeeTechGripperConfig,
    MockJointConfig,
    O6DexterousHandConfig,
    RobotConfig,
)
from acetele.core.make_robot import make_robot
from acetele.equipment.feetech.feetech_driver import Mode, TorqueEnable
from acetele.equipment.joint_device import (
    CompositeJointDevice,
    JointDeviceState,
    MockJointDevice,
)
from acetele.equipment.joint_device import TorqueEnable as JointDeviceTorqueEnable
from acetele.robot.ace_follower.ace_follower import AceFollowerRobot
from acetele.robot.ace_leader.ace_leader import AceLeaderRobot
from acetele.robot.joint_robot import JointRobot


def _mock_joint(name: str, joint_id: int, position: float = 0.0) -> MockJointConfig:
    return MockJointConfig(name, joint_id, position, -1.0, 1.0, 2.0)


def _arm_config(
    *,
    port: str | None = "/dev/arm",
    adaptive: bool = False,
    gravity: bool = False,
    mock: bool = False,
) -> ArmConfig:
    return ArmConfig(
        port=port,
        joint_ids=(0, 1),
        joint_names=("joint_1", "joint_2"),
        joint_signs=(1, -1),
        home_poses=(0.0, 0.0),
        servo_models=() if mock else ("HL3915", "HL3915"),
        enable_gravity_compensation=gravity,
        enable_adaptive_compensation=adaptive,
    )


def _gripper_config(
    *,
    port: str | None = "/dev/arm",
) -> FeeTechGripperConfig:
    return FeeTechGripperConfig(
        port=port,
        joint_id=4,
        joint_name="joint_5",
        joint_sign=-1,
        home_pose=1.0,
        servo_model="HL3915",
        travel_range_rad=896.0 * np.pi / 2048.0,
    )


def _robot_config(
    *,
    robot_type: str = "ace_follower",
    backend: str = "mock",
    arm: ArmConfig | None = None,
    gripper: FeeTechGripperConfig | None = None,
) -> RobotConfig:
    is_mock = backend == "mock"
    return RobotConfig(
        robot_type=robot_type,
        backend=backend,
        runtime="standalone",
        arm_assemblies=(
            ArmAssemblyConfig(
                "single",
                arm or _arm_config(port=None if is_mock else "/dev/arm", mock=is_mock),
                gripper if gripper is not None else _gripper_config(port=None if is_mock else "/dev/arm"),
            ),
        ),
    )


def test_composite_joint_device_aggregates_arm_before_end_effector():
    arm = MockJointDevice((_mock_joint("joint_1", 0, 0.1), _mock_joint("joint_2", 1, 0.2)))
    gripper = MockJointDevice((_mock_joint("joint_5", 4, 0.8),))
    composite = CompositeJointDevice((arm, gripper))

    state = composite.get_state()

    assert composite.joint_names == ("joint_1", "joint_2", "joint_5")
    np.testing.assert_array_equal(composite.ids, np.array([0, 1, 4]))
    np.testing.assert_allclose(state.public_positions, np.array([0.1, 0.2, 0.8]))


def test_composite_routes_subset_commands_and_profiles_without_mutating_input():
    arm = RecordingMockDevice((_mock_joint("joint_1", 0), _mock_joint("joint_2", 1)))
    gripper = RecordingMockDevice((_mock_joint("joint_5", 4),))
    composite = CompositeJointDevice((arm, gripper))
    positions = np.array([0.4, 0.7])
    velocities = np.array([1.0, 2.0])

    composite.set_position(
        positions,
        ids=np.array([np.int64(1), np.int64(4)]),
        velocities=velocities,
    )

    np.testing.assert_allclose(positions, np.array([0.4, 0.7]))
    np.testing.assert_allclose(velocities, np.array([1.0, 2.0]))
    assert arm.commands[0][1].tolist() == [1]
    assert gripper.commands[0][1].tolist() == [4]
    np.testing.assert_allclose(arm.commands[0][2]["velocities"], np.array([1.0]))
    np.testing.assert_allclose(gripper.commands[0][2]["velocities"], np.array([2.0]))


@pytest.mark.parametrize(
    "invalid_ids",
    ([0.9], ["0"], [True], np.array([[0]], dtype=int)),
)
def test_composite_rejects_invalid_joint_ids_before_routing(invalid_ids):
    arm = RecordingMockDevice((_mock_joint("joint_1", 0),))
    composite = CompositeJointDevice((arm,))

    with pytest.raises(ValueError, match="joint ids"):
        composite.set_position([0.5], ids=invalid_ids)

    assert arm.commands == []
    np.testing.assert_allclose(composite.act()[0], np.array([0.0]))


def test_composite_rejects_duplicate_ids_and_names():
    first = MockJointDevice((_mock_joint("joint", 0),))

    with pytest.raises(ValueError, match="names must be globally unique"):
        CompositeJointDevice((first, MockJointDevice((_mock_joint("joint", 1),))))
    with pytest.raises(ValueError, match="ids must be globally unique"):
        CompositeJointDevice((first, MockJointDevice((_mock_joint("other", 0),))))


def test_mock_joint_device_rejects_duplicate_ids_and_names():
    with pytest.raises(ValueError, match="mock joint ids must be unique"):
        MockJointDevice((_mock_joint("joint_1", 0), _mock_joint("joint_2", 0)))
    with pytest.raises(ValueError, match="mock joint names must be unique"):
        MockJointDevice((_mock_joint("joint", 0), _mock_joint("joint", 1)))


class RecordingMockDevice(MockJointDevice):
    def __init__(self, joints):
        super().__init__(joints)
        self.commands = []

    def set_position(
        self,
        positions,
        ids=None,
        velocities=None,
        accelerations=None,
        torque=None,
    ):
        self.commands.append(
            (
                np.asarray(positions).copy(),
                np.asarray(ids).copy(),
                {
                    "velocities": velocities,
                    "accelerations": accelerations,
                    "torque": torque,
                },
            )
        )
        super().set_position(positions, ids=ids)


class RejectingPositionDevice(RecordingMockDevice):
    def validate_position_command(self, positions, ids=None, **kwargs):
        super().validate_position_command(positions, ids=ids, **kwargs)
        raise ValueError("device command rejected")


def test_composite_validates_all_devices_before_dispatching_any_command():
    arm = RecordingMockDevice((_mock_joint("joint_1", 0),))
    gripper = RejectingPositionDevice((_mock_joint("joint_5", 4),))
    composite = CompositeJointDevice((arm, gripper))

    with pytest.raises(ValueError, match="device command rejected"):
        composite.set_position([0.2, 0.8])

    assert arm.commands == []
    assert gripper.commands == []


def test_single_robot_composes_mock_arm_and_gripper_with_public_api():
    gripper_config = _gripper_config(port=None)
    robot = make_robot(_robot_config(gripper=gripper_config))

    assert isinstance(robot, AceFollowerRobot)
    assert robot.dof == 3
    assert robot.joint_names == ("joint_1", "joint_2", "joint_5")

    command = np.array([0.2, 0.3, 0.6])
    robot.set_position(command)
    positions, velocities, efforts = robot.act()

    np.testing.assert_allclose(positions, command)
    np.testing.assert_allclose(velocities, np.zeros(3))
    np.testing.assert_allclose(efforts, np.zeros(3))
    expected_gripper_raw_position = (
        command[-1] * gripper_config.travel_range_counts * np.pi / 2048.0
    )
    assert robot.get_robot_state().raw_positions[-1] == pytest.approx(
        expected_gripper_raw_position
    )


def test_mock_arm_initial_state_comes_from_home_poses():
    arm = replace(_arm_config(port=None, mock=True), home_poses=(0.25, 0.5))
    robot = make_robot(_robot_config(arm=arm))

    positions, _, _ = robot.act()

    np.testing.assert_allclose(positions[:2], arm.home_poses)
    assert isinstance(robot.get_robot_state(), JointDeviceState)


def test_mock_arm_unwraps_public_targets_before_applying_urdf_limits():
    loader = ConfigLoader(
        Path("acetele/config/ace_follower.toml"),
        backend_override="mock",
        runtime_override="standalone",
    )
    robot = AceFollowerRobot(loader)

    robot.set_position([-2.9], ids=[1])
    state = robot.get_robot_state()

    assert state.public_positions[1] == pytest.approx(-2.9)
    assert state.raw_positions[1] == pytest.approx(2.0 * np.pi - 2.9)


def test_make_robot_uses_topology_and_runtime_not_end_effector_type():
    leader = make_robot(_robot_config(robot_type="ace_leader"))
    follower_without_gripper = make_robot(
        replace(
            _robot_config(),
            arm_assemblies=(
                replace(_robot_config().arm_assemblies[0], end_effector=None),
            ),
        )
    )

    assert isinstance(leader, AceLeaderRobot)
    assert isinstance(follower_without_gripper, AceFollowerRobot)
    assert follower_without_gripper.dof == 2


def test_make_robot_accepts_config_loader_keyword():
    loader = ConfigLoader(
        Path("acetele/config/ace_leader.toml"),
        backend_override="mock",
        runtime_override="standalone",
    )

    robot = make_robot(config_loader=loader)

    assert isinstance(robot, AceLeaderRobot)


def test_follower_constructor_accepts_config_loader_keyword():
    robot = AceFollowerRobot(config_loader=_robot_config())

    assert isinstance(robot, AceFollowerRobot)


def test_leader_constructor_accepts_config_loader_keyword():
    loader = ConfigLoader(
        Path("acetele/config/ace_leader.toml"),
        backend_override="mock",
    )

    robot = AceLeaderRobot(config_loader=loader)

    assert isinstance(robot, AceLeaderRobot)


class FakeDriver:
    created: list["FakeDriver"] = []

    def __init__(self, ids, port):
        self.ids = tuple(ids)
        self.port = port
        self.close_calls = 0
        FakeDriver.created.append(self)

    def close(self):
        self.close_calls += 1


class FakePhysicalDevice:
    def __init__(self, config, driver, **kwargs):
        self.config = config
        self.driver = driver
        self.kwargs = kwargs
        self._ids = np.asarray(config.joint_ids, dtype=int)
        self._joint_names = config.joint_names
        self.close_calls = 0

    @property
    def ids(self):
        return self._ids

    @property
    def joint_names(self):
        return self._joint_names

    def get_state(self):
        zeros = np.zeros(len(self._ids))
        return JointDeviceState(zeros, zeros, zeros, zeros, zeros)

    def set_position(self, positions, ids=None, **kwargs):
        del positions, ids, kwargs

    def validate_position_command(self, positions, ids=None, **kwargs):
        del positions, ids, kwargs

    def set_torque_enable(self, enable, ids=None):
        del enable, ids

    def close(self):
        self.close_calls += 1


@pytest.fixture
def physical_device_fakes(monkeypatch):
    FakeDriver.created = []
    monkeypatch.setattr("acetele.robot.joint_robot.FeeTechDriver", FakeDriver)
    monkeypatch.setattr("acetele.robot.joint_robot.FeeTechArm", FakePhysicalDevice)
    monkeypatch.setattr("acetele.equipment.end_effector_factory.FeeTechGripper", FakePhysicalDevice)
    return FakeDriver


def test_physical_arm_and_gripper_share_driver_on_same_port(physical_device_fakes):
    robot = make_robot(_robot_config(backend="physical"))

    assert len(FakeDriver.created) == 1
    assert FakeDriver.created[0].ids == (0, 1, 4)
    assert robot.arm.driver is FakeDriver.created[0]
    assert robot.end_effector.driver is FakeDriver.created[0]


def test_physical_gripper_uses_separate_driver_on_different_port(physical_device_fakes):
    config = _robot_config(
        backend="physical",
        gripper=_gripper_config(port="/dev/gripper"),
    )

    robot = make_robot(config)

    assert [(driver.ids, driver.port) for driver in FakeDriver.created] == [
        ((0, 1), "/dev/arm"),
        ((4,), "/dev/gripper"),
    ]
    assert robot.arm.driver is FakeDriver.created[0]
    assert robot.end_effector.driver is FakeDriver.created[1]


def test_driver_construction_interrupt_closes_previously_created_driver(
    physical_device_fakes,
    monkeypatch,
):
    del physical_device_fakes
    created = []

    class InterruptingDriver:
        def __init__(self, ids, port):
            if created:
                raise KeyboardInterrupt("driver construction interrupted")
            self.ids = tuple(ids)
            self.port = port
            self.close_calls = 0
            created.append(self)

        def close(self):
            self.close_calls += 1

    monkeypatch.setattr(
        "acetele.robot.joint_robot.FeeTechDriver",
        InterruptingDriver,
    )
    config = _robot_config(
        backend="physical",
        gripper=_gripper_config(port="/dev/gripper"),
    )

    with pytest.raises(KeyboardInterrupt, match="construction interrupted"):
        make_robot(config)

    assert len(created) == 1
    assert created[0].close_calls == 1


@pytest.mark.parametrize("backend", ["physical", "mock"])
def test_single_arm_robot_requires_urdf_before_device_creation(
    physical_device_fakes,
    monkeypatch,
    backend,
):
    def initialize_without_urdf(robot, config_loader):
        robot.robot_config = (
            config_loader
            if isinstance(config_loader, RobotConfig)
            else config_loader.get_robot_config()
        )
        robot._robot_name = "test"
        robot._urdf_model_path = None

    monkeypatch.setattr(
        "acetele.robot.joint_robot.BaseRobot.__init__",
        initialize_without_urdf,
    )

    with pytest.raises(RuntimeError, match="URDF model is required"):
        make_robot(_robot_config(backend=backend))

    assert FakeDriver.created == []


@pytest.mark.parametrize(
    ("joint_names", "match"),
    [
        (("joint_1", "joint_missing"), "missing"),
        (("joint_2", "joint_1"), "kinematic order"),
    ],
)
def test_invalid_arm_urdf_mapping_fails_before_driver_creation(
    physical_device_fakes,
    joint_names,
    match,
):
    arm = replace(_arm_config(), joint_names=joint_names)

    with pytest.raises(ValueError, match=match):
        make_robot(_robot_config(backend="physical", arm=arm))

    assert FakeDriver.created == []


def test_invalid_gripper_urdf_mapping_fails_before_driver_creation(physical_device_fakes):
    gripper = replace(_gripper_config(), joint_name="joint_missing")

    with pytest.raises(ValueError, match="missing"):
        make_robot(_robot_config(backend="physical", gripper=gripper))

    assert FakeDriver.created == []


@pytest.mark.parametrize("device", ["arm", "gripper"])
def test_unknown_servo_model_fails_before_driver_creation(
    physical_device_fakes,
    device,
):
    arm = _arm_config()
    gripper = _gripper_config()
    if device == "arm":
        arm = replace(arm, servo_models=("BAD", "HL3915"))
    else:
        gripper = replace(gripper, servo_model="BAD")

    with pytest.raises(ValueError, match="unsupported servo models"):
        make_robot(
            _robot_config(
                backend="physical",
                arm=arm,
                gripper=gripper,
            )
        )

    assert FakeDriver.created == []


@pytest.mark.parametrize("device", ["arm", "gripper"])
def test_out_of_range_servo_id_fails_before_driver_creation(
    physical_device_fakes,
    device,
):
    arm = _arm_config()
    gripper = _gripper_config()
    if device == "arm":
        arm = replace(arm, joint_ids=(0, 253))
    else:
        gripper = replace(gripper, joint_id=253)

    with pytest.raises(ValueError, match="between 0 and 252"):
        make_robot(
            _robot_config(
                backend="physical",
                arm=arm,
                gripper=gripper,
            )
        )

    assert FakeDriver.created == []


def test_invalid_adaptive_limits_fail_before_driver_creation(
    physical_device_fakes,
    monkeypatch,
):
    config = _robot_config(
        backend="physical",
        arm=_arm_config(adaptive=True),
    )
    monkeypatch.setattr(
        JointRobot,
        "_get_joint_position_limits",
        lambda *_args: (_ for _ in ()).throw(ValueError("invalid limits")),
    )

    with pytest.raises(ValueError, match="invalid limits"):
        make_robot(config)

    assert FakeDriver.created == []


def test_pinocchio_build_failure_happens_before_driver_creation(
    physical_device_fakes,
    monkeypatch,
):
    config = _robot_config(
        backend="physical",
        arm=_arm_config(gravity=True),
    )
    monkeypatch.setattr(
        JointRobot,
        "_get_pin_model_for_joint_names",
        lambda *_args: (_ for _ in ()).throw(ValueError("pin model failed")),
    )

    with pytest.raises(ValueError, match="pin model failed"):
        make_robot(config)

    assert FakeDriver.created == []


def test_pinocchio_dof_mismatch_happens_before_driver_creation(
    physical_device_fakes,
    monkeypatch,
):
    config = _robot_config(
        backend="physical",
        arm=_arm_config(gravity=True),
    )
    monkeypatch.setattr(
        JointRobot,
        "_get_pin_model_for_joint_names",
        lambda *_args: SimpleNamespace(nv=1),
    )

    with pytest.raises(ValueError, match="must match arm joint count"):
        make_robot(config)

    assert FakeDriver.created == []


def test_physical_o6_is_rejected_before_driver_creation(physical_device_fakes):
    hand = O6DexterousHandConfig(side="left", joint_ids=tuple(range(10, 16)))
    config = replace(
        _robot_config(backend="physical"),
        arm_assemblies=(
            ArmAssemblyConfig("single", _arm_config(), hand),
        ),
    )

    with pytest.raises(RuntimeError, match="physical backend is not implemented"):
        make_robot(config)

    assert FakeDriver.created == []


def test_servo_id_remapping_does_not_change_urdf_joint_mapping(physical_device_fakes):
    arm = replace(_arm_config(), joint_ids=(10, 11))

    robot = make_robot(_robot_config(backend="physical", arm=arm))

    assert FakeDriver.created[0].ids == (10, 11, 4)
    assert robot.arm.joint_names == ("joint_1", "joint_2")


def test_adaptive_compensation_receives_urdf_limits_without_pin_model(
    physical_device_fakes,
    monkeypatch,
):
    config = _robot_config(
        backend="physical",
        arm=_arm_config(adaptive=True),
    )
    monkeypatch.setattr(
        JointRobot,
        "_get_joint_position_limits",
        lambda _self, names: ([-1.0, -0.5], [1.0, 0.5]),
    )
    monkeypatch.setattr(
        JointRobot,
        "_get_pin_model_for_joint_names",
        lambda *_args: pytest.fail("adaptive compensation must not load Pinocchio"),
    )

    robot = make_robot(config)

    assert robot.arm.kwargs["position_limits"] == ((-1.0, -0.5), (1.0, 0.5))
    assert robot.arm.kwargs["pin_model"] is None


def test_physical_arm_receives_urdf_limits_when_adaptive_compensation_is_disabled(
    physical_device_fakes,
    monkeypatch,
):
    config = _robot_config(
        backend="physical",
        arm=_arm_config(adaptive=False),
    )
    monkeypatch.setattr(
        JointRobot,
        "_get_joint_position_limits",
        lambda _self, names: ([-1.0, -0.5], [1.0, 0.5]),
    )

    robot = make_robot(config)

    assert robot.arm.kwargs["position_limits"] == ((-1.0, -0.5), (1.0, 0.5))


def test_gravity_compensation_builds_model_from_arm_joint_names(
    physical_device_fakes,
    monkeypatch,
):
    config = _robot_config(
        backend="physical",
        arm=_arm_config(gravity=True),
    )
    calls = []
    model = SimpleNamespace(nv=2)
    monkeypatch.setattr(
        JointRobot,
        "_get_pin_model_for_joint_names",
        lambda _self, names: calls.append(names) or model,
    )

    robot = make_robot(config)

    assert calls == [("joint_1", "joint_2")]
    assert robot.arm.kwargs["pin_model"] is model


def test_gravity_compensation_supports_arm_without_end_effector(
    physical_device_fakes,
    monkeypatch,
):
    config = _robot_config(
        backend="physical",
        arm=_arm_config(gravity=True),
    )
    config = replace(
        config,
        arm_assemblies=(replace(config.arm_assemblies[0], end_effector=None),),
    )
    calls = []
    model = SimpleNamespace(nv=2)
    monkeypatch.setattr(
        JointRobot,
        "_get_pin_model_for_joint_names",
        lambda _self, names: calls.append(names) or model,
    )

    robot = make_robot(config)

    assert calls == [("joint_1", "joint_2")]
    assert robot.end_effector is None
    assert robot.arm.kwargs["pin_model"] is model


def test_robot_close_closes_devices_then_each_shared_driver_once(physical_device_fakes):
    robot = make_robot(_robot_config(backend="physical"))

    robot.close()
    robot.close()

    assert robot.arm.close_calls == 1
    assert robot.end_effector.close_calls == 1
    assert FakeDriver.created[0].close_calls == 1


def test_robot_close_continues_after_device_close_error(physical_device_fakes):
    robot = make_robot(_robot_config(backend="physical"))

    def fail_arm_close():
        robot.arm.close_calls += 1
        raise RuntimeError("arm close failed")

    robot.arm.close = fail_arm_close

    with pytest.raises(RuntimeError, match="arm close failed"):
        robot.close()

    assert robot.arm.close_calls == 1
    assert robot.end_effector.close_calls == 1
    assert FakeDriver.created[0].close_calls == 1

    robot.close()

    assert robot.arm.close_calls == 1
    assert robot.end_effector.close_calls == 1
    assert FakeDriver.created[0].close_calls == 1


def test_device_construction_error_closes_driver_even_when_device_close_fails(
    physical_device_fakes,
    monkeypatch,
):
    def fail_end_effector_build(*_args, **_kwargs):
        raise ValueError("end effector build failed")

    def fail_device_close(self):
        self.close_calls += 1
        raise RuntimeError("device close failed")

    monkeypatch.setattr(
        JointRobot,
        "_build_physical_end_effector",
        staticmethod(fail_end_effector_build),
    )
    monkeypatch.setattr(FakePhysicalDevice, "close", fail_device_close)

    with pytest.raises(ValueError, match="end effector build failed") as exc_info:
        make_robot(_robot_config(backend="physical"))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "device close failed"
    assert FakeDriver.created[0].close_calls == 1


def test_composite_construction_error_closes_all_devices_and_driver(
    physical_device_fakes,
    monkeypatch,
):
    closed_devices = []

    def record_device_close(device):
        device.close_calls += 1
        closed_devices.append(device)

    class FailingCompositeJointDevice:
        def __init__(self, _devices):
            raise ValueError("composite build failed")

    monkeypatch.setattr(FakePhysicalDevice, "close", record_device_close)
    monkeypatch.setattr(
        "acetele.robot.joint_robot.CompositeJointDevice",
        FailingCompositeJointDevice,
    )

    with pytest.raises(ValueError, match="composite build failed"):
        make_robot(_robot_config(backend="physical"))

    assert len(FakeDriver.created) == 1
    assert FakeDriver.created[0].close_calls == 1
    assert len(closed_devices) == 2
    assert {device.config.joint_names for device in closed_devices} == {
        ("joint_1", "joint_2"),
        ("joint_5",),
    }


def test_composite_torque_enable_routes_selected_ids():
    arm = TorqueRecordingDevice((_mock_joint("joint_1", 0), _mock_joint("joint_2", 1)))
    gripper = TorqueRecordingDevice((_mock_joint("joint_5", 4),))
    composite = CompositeJointDevice((arm, gripper))

    composite.set_torque_enable(TorqueEnable.Disable, ids=[1, 4])

    assert arm.torque_calls == [(TorqueEnable.Disable, [1])]
    assert gripper.torque_calls == [(TorqueEnable.Disable, [4])]


def test_feetech_driver_reexports_joint_device_torque_enable():
    assert TorqueEnable is JointDeviceTorqueEnable


@pytest.mark.parametrize("invalid_enable", ("disable", 0, False, Mode.Position))
def test_mock_rejects_invalid_torque_enable(invalid_enable):
    device = MockJointDevice((_mock_joint("joint_1", 0),))

    with pytest.raises(ValueError, match="TorqueEnable"):
        device.set_torque_enable(invalid_enable)


@pytest.mark.parametrize("invalid_enable", ("disable", 0, False, Mode.Position))
def test_composite_rejects_invalid_torque_enable_before_routing(invalid_enable):
    device = TorqueRecordingDevice((_mock_joint("joint_1", 0),))
    composite = CompositeJointDevice((device,))

    with pytest.raises(ValueError, match="TorqueEnable"):
        composite.set_torque_enable(invalid_enable)

    assert device.torque_calls == []


@pytest.mark.parametrize("invalid_ids", ([0.9], ["0"], [True]))
def test_composite_rejects_invalid_torque_enable_ids_before_routing(invalid_ids):
    arm = TorqueRecordingDevice((_mock_joint("joint_1", 0),))
    composite = CompositeJointDevice((arm,))

    with pytest.raises(ValueError, match="joint ids"):
        composite.set_torque_enable(TorqueEnable.Disable, ids=invalid_ids)

    assert arm.torque_calls == []


class TorqueRecordingDevice(MockJointDevice):
    def __init__(self, joints):
        super().__init__(joints)
        self.torque_calls = []

    def set_torque_enable(self, enable, ids=None):
        self.torque_calls.append((enable, np.asarray(ids, dtype=int).tolist()))
