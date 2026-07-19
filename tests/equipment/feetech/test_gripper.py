import numpy as np
import pytest

from acetele.config.robot_config import FeeTechGripperConfig
from acetele.equipment.feetech.feetech_driver import FeeTechCommandDispatchError, TorqueEnable
from acetele.equipment.feetech.gripper import FeeTechGripper
from acetele.equipment.feetech.servo_specs import HLS_PROFILE_DEFAULTS_BY_SERVO
from acetele.equipment.joint_device import JointDevice

HL3915_TORQUE_CURRENT_MAPPING = 1000.0 / 9.3
HL3915_NO_LOAD_CURRENT = 260
HL3915_DEFAULT_PROFILE_VELOCITY = HLS_PROFILE_DEFAULTS_BY_SERVO["HL3915"]["velocity"]
LEADER_TRAVEL_RAD = np.pi / 4.0
FOLLOWER_TRAVEL_RAD = 896.0 * np.pi / 2048.0


def raw_current_for_torque(torque_nm):
    return round(((HL3915_TORQUE_CURRENT_MAPPING * (torque_nm / 0.0981)) + HL3915_NO_LOAD_CURRENT) / 6.5)


class PositionRecordingDriver:
    def __init__(self, state):
        self.state = state
        self.profile_calls = []
        self.torque_enable_calls = []
        self.close_calls = 0
        self.events = []

    def get_state(self):
        return self.state

    def set_position(self, ids, positions, **kwargs):
        self.profile_calls.append((np.array(ids), np.array(positions), kwargs))

    def set_torque_enable(self, ids, enables, **kwargs):
        self.events.append("torque_disable")
        self.torque_enable_calls.append((list(ids), list(enables), kwargs))

    def close(self):
        self.events.append("close")
        self.close_calls += 1


def make_gripper(driver=None, travel_range_rad=LEADER_TRAVEL_RAD, joint_sign=1):
    config = FeeTechGripperConfig(
        port="/dev/test",
        joint_id=4,
        joint_name="joint_5",
        joint_sign=joint_sign,
        home_pose=0.0,
        servo_model="HL3915",
        travel_range_rad=travel_range_rad,
    )
    if driver is None:
        driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    return FeeTechGripper(config, driver=driver)


def test_gripper_implements_joint_device_contract():
    gripper = make_gripper()

    assert isinstance(gripper, JointDevice)
    assert gripper.joint_names == ("joint_5",)
    np.testing.assert_array_equal(gripper.ids, np.array([4]))


def test_gripper_decoding_inverts_encoded_act_value():
    encoded_gripper_position = 512
    driver = PositionRecordingDriver(({4: encoded_gripper_position}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver)

    positions, _, _ = gripper.act()
    gripper.set_position(positions)

    _, encoded_positions, _ = driver.profile_calls[0]
    assert encoded_positions[0] == pytest.approx(encoded_gripper_position, abs=1)


def test_gripper_returns_common_array_state():
    driver = PositionRecordingDriver(({4: 256}, {4: 10}, {4: -100}))
    state = make_gripper(driver=driver).get_state()

    assert state.public_positions.shape == (1,)
    assert state.raw_positions.shape == (1,)
    assert state.velocities.shape == (1,)
    assert state.motor_torque_magnitude.shape == (1,)
    assert state.motor_torque_signed.shape == (1,)


def test_set_position_uses_follower_gripper_travel():
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver, travel_range_rad=FOLLOWER_TRAVEL_RAD)

    gripper.set_position([0.5])

    sent_ids, encoded_positions, _ = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([4]))
    assert encoded_positions[0] == pytest.approx(448, abs=1)


def test_set_position_applies_gripper_sign_once():
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    gripper = make_gripper(
        driver=driver,
        travel_range_rad=FOLLOWER_TRAVEL_RAD,
        joint_sign=-1,
    )

    gripper.set_position([1.0])

    _, encoded_positions, _ = driver.profile_calls[0]
    assert encoded_positions[0] == pytest.approx(-896, abs=1)


@pytest.mark.parametrize(
    ("joint_sign", "encoded_position"),
    [(1, 4096 + 896), (-1, 4096 - 896)],
)
def test_gripper_normalized_feedback_ignores_full_encoder_turns(
    joint_sign,
    encoded_position,
):
    driver = PositionRecordingDriver(
        ({4: encoded_position}, {4: 0}, {4: 0})
    )
    gripper = make_gripper(
        driver=driver,
        travel_range_rad=FOLLOWER_TRAVEL_RAD,
        joint_sign=joint_sign,
    )

    state = gripper.get_state()

    assert state.public_positions[0] == pytest.approx(1.0)
    assert abs(state.raw_positions[0]) > np.pi


@pytest.mark.parametrize("travel_range_counts", [1, 2047])
def test_gripper_count_range_boundaries_round_trip(travel_range_counts):
    travel_range_rad = travel_range_counts * np.pi / 2048.0
    driver = PositionRecordingDriver(
        ({4: travel_range_counts}, {4: 0}, {4: 0})
    )
    gripper = make_gripper(driver=driver, travel_range_rad=travel_range_rad)

    assert gripper.config.travel_range_counts == travel_range_counts
    assert gripper.get_state().public_positions[0] == pytest.approx(1.0)

    gripper.set_position([1.0])

    assert driver.profile_calls[0][1][0] == travel_range_counts


@pytest.mark.parametrize(
    "travel_range_rad",
    [0.49 * np.pi / 2048.0, np.pi],
)
def test_gripper_rejects_unrepresentable_count_ranges(travel_range_rad):
    with pytest.raises(ValueError, match="between 1 and 2047"):
        FeeTechGripperConfig(
            port="/dev/test",
            joint_id=4,
            joint_name="joint_5",
            joint_sign=1,
            home_pose=0.0,
            servo_model="HL3915",
            travel_range_rad=travel_range_rad,
        )


def test_set_position_accepts_joint_device_profiles():
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver, travel_range_rad=FOLLOWER_TRAVEL_RAD)

    gripper.set_position([0.25], torque=[0.09])

    sent_ids, encoded_positions, kwargs = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([4]))
    assert encoded_positions[0] == pytest.approx(224, abs=1)
    assert kwargs["velocities_raw"] == [HL3915_DEFAULT_PROFILE_VELOCITY]
    assert kwargs["currents_raw"] == [pytest.approx(raw_current_for_torque(0.09))]


@pytest.mark.parametrize(
    ("positions", "profiles", "match"),
    [
        ([np.nan], {}, "finite"),
        ([0.5], {"velocities": -0.1}, "non-negative"),
        ([0.5], {"accelerations": -0.1}, "non-negative"),
        ([0.5], {"torque": np.inf}, "finite"),
        ([0.5], {"velocities": 1e9}, "encoded velocity"),
        ([0.5], {"accelerations": 1e9}, "encoded acceleration"),
        ([0.5], {"torque": 1e9}, "encoded current"),
        ([0.5], {"velocities": [1.0, 2.0]}, "match ids length"),
    ],
)
def test_gripper_validates_complete_position_profile_without_dispatch(
    positions,
    profiles,
    match,
):
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver)

    with pytest.raises(ValueError, match=match):
        gripper.validate_position_command(positions, **profiles)

    assert driver.profile_calls == []


def test_set_position_rejects_unknown_gripper_joint():
    gripper = make_gripper()

    with pytest.raises(ValueError, match="unknown joint"):
        gripper.set_position([0.5], ids=[3])


@pytest.mark.parametrize("invalid_ids", ([4.0], ["4"], [True]))
def test_gripper_rejects_invalid_joint_ids_before_writing(invalid_ids):
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver)

    with pytest.raises(ValueError, match="gripper joint ids"):
        gripper.set_position([0.5], ids=invalid_ids)

    assert driver.profile_calls == []


def test_gripper_close_waits_for_disable_without_closing_shared_driver():
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver)

    gripper.close()

    assert driver.torque_enable_calls == [
        ([4], [TorqueEnable.Disable], {"force": True, "wait": True})
    ]
    assert driver.events == ["torque_disable"]
    assert driver.close_calls == 0


def test_gripper_close_disables_torque_before_closing_self_created_driver(monkeypatch):
    created = []

    class CapturingFeeTechDriver(PositionRecordingDriver):
        def __init__(self, ids, port):
            super().__init__(({int(ids[0]): 0}, {int(ids[0]): 0}, {int(ids[0]): 0}))
            self.port = port
            created.append(self)

    monkeypatch.setattr("acetele.equipment.feetech.gripper.FeeTechDriver", CapturingFeeTechDriver)
    config = FeeTechGripperConfig(
        port="/dev/test",
        joint_id=4,
        joint_name="joint_5",
        joint_sign=1,
        home_pose=0.0,
        servo_model="HL3915",
        travel_range_rad=LEADER_TRAVEL_RAD,
    )
    gripper = FeeTechGripper(config)

    gripper.close()

    assert created[0].events == ["torque_disable", "close"]
    assert created[0].close_calls == 1


def test_gripper_close_releases_self_created_driver_when_disable_fails(monkeypatch):
    created = []

    class FailingFeeTechDriver(PositionRecordingDriver):
        def __init__(self, ids, port):
            super().__init__(({int(ids[0]): 0}, {int(ids[0]): 0}, {int(ids[0]): 0}))
            del port
            created.append(self)

        def set_torque_enable(self, ids, enables, **kwargs):
            del ids, enables, kwargs
            self.events.append("torque_disable")
            raise FeeTechCommandDispatchError("write failed")

    monkeypatch.setattr("acetele.equipment.feetech.gripper.FeeTechDriver", FailingFeeTechDriver)
    config = FeeTechGripperConfig(
        port="/dev/test",
        joint_id=4,
        joint_name="joint_5",
        joint_sign=1,
        home_pose=0.0,
        servo_model="HL3915",
        travel_range_rad=LEADER_TRAVEL_RAD,
    )
    gripper = FeeTechGripper(config)

    with pytest.raises(FeeTechCommandDispatchError, match="write failed"):
        gripper.close()

    assert created[0].events == ["torque_disable", "close"]
    assert created[0].close_calls == 1


def test_gripper_close_preserves_disable_error_when_driver_close_also_fails(
    monkeypatch,
):
    disable_error = FeeTechCommandDispatchError("disable failed")
    close_error = RuntimeError("close failed")

    class FailingFeeTechDriver(PositionRecordingDriver):
        def __init__(self, ids, port):
            del port
            super().__init__(
                ({int(ids[0]): 0}, {int(ids[0]): 0}, {int(ids[0]): 0})
            )

        def set_torque_enable(self, ids, enables, **kwargs):
            del ids, enables, kwargs
            raise disable_error

        def close(self):
            raise close_error

    monkeypatch.setattr(
        "acetele.equipment.feetech.gripper.FeeTechDriver",
        FailingFeeTechDriver,
    )
    gripper = FeeTechGripper(
        FeeTechGripperConfig(
            port="/dev/test",
            joint_id=4,
            joint_name="joint_5",
            joint_sign=1,
            home_pose=0.0,
            servo_model="HL3915",
            travel_range_rad=LEADER_TRAVEL_RAD,
        )
    )

    with pytest.raises(FeeTechCommandDispatchError) as exc_info:
        gripper.close()

    assert exc_info.value is disable_error
    assert exc_info.value.__cause__ is close_error
