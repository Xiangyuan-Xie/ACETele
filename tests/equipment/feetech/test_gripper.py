import numpy as np
import pytest

from acetele.equipment.feetech.gripper import Gripper
from acetele.equipment.feetech.servo_specs import HLS_PROFILE_DEFAULTS_BY_SERVO

HL3915_TORQUE_CURRENT_MAPPING = 1000.0 / 9.3
HL3915_NO_LOAD_CURRENT = 260
HL3915_DEFAULT_PROFILE_VELOCITY = HLS_PROFILE_DEFAULTS_BY_SERVO["HL3915"]["velocity"]


def raw_current_for_torque(torque_nm):
    return round(((HL3915_TORQUE_CURRENT_MAPPING * (torque_nm / 0.0981)) + HL3915_NO_LOAD_CURRENT) / 6.5)


class PositionRecordingDriver:
    def __init__(self, state):
        self.state = state
        self.profile_calls = []
        self.torque_enable_calls = []

    def get_state(self):
        return self.state

    def set_position(self, ids, positions, **kwargs):
        self.profile_calls.append((np.array(ids), np.array(positions), kwargs))

    def set_torque_enable(self, ids, enables):
        self.torque_enable_calls.append((list(ids), list(enables)))


def make_gripper(driver=None, gripper_type="ace_leader", joint_sign=1):
    config = {
        "joint_id": 4,
        "joint_sign": joint_sign,
        "home_pose": 0.0,
        "servo_type": "HL3915",
        "gripper_type": gripper_type,
    }
    if driver is None:
        driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    return Gripper(config, driver=driver)


def test_gripper_requires_joint_id():
    config = {
        "joint_sign": 1,
        "home_pose": 0.0,
        "servo_type": "HL3915",
        "gripper_type": "ace_leader",
    }
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))

    with pytest.raises(ValueError, match="gripper.single.joint_id"):
        Gripper(config, driver=driver)


def test_gripper_decoding_inverts_encoded_act_value():
    encoded_gripper_position = 512
    driver = PositionRecordingDriver(({4: encoded_gripper_position}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver)

    positions, _, _ = gripper.act()
    gripper.set_position(float(positions[0]))

    _, encoded_positions, _ = driver.profile_calls[0]
    assert encoded_positions[0] == pytest.approx(encoded_gripper_position, abs=1)


def test_set_position_uses_follower_gripper_travel():
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver, gripper_type="ace_follower")

    gripper.set_position(0.5)

    sent_ids, encoded_positions, _ = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([4]))
    assert encoded_positions[0] == pytest.approx(448, abs=1)


def test_set_position_applies_gripper_sign_once():
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver, gripper_type="ace_follower", joint_sign=-1)

    gripper.set_position(1.0)

    _, encoded_positions, _ = driver.profile_calls[0]
    assert encoded_positions[0] == pytest.approx(-896, abs=1)


def test_set_position_accepts_plain_torque_profile():
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver, gripper_type="ace_follower")

    gripper.set_position(0.25, torque=0.09)

    sent_ids, encoded_positions, kwargs = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([4]))
    assert encoded_positions[0] == pytest.approx(224, abs=1)
    assert kwargs["velocities_raw"] == [HL3915_DEFAULT_PROFILE_VELOCITY]
    assert kwargs["currents_raw"] == [pytest.approx(raw_current_for_torque(0.09))]


def test_gripper_exposes_only_plain_position_api():
    for name in (
        "set_" + "fragi" + "le_position",
        "get_" + "force_" + "control_state",
        "reset_" + "force_" + "control",
    ):
        assert not hasattr(Gripper, name)
