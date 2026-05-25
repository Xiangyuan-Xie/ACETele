import numpy as np
import pytest

from acetele.equipment.feetech.gripper import Gripper, GripperForceStatus
from acetele.equipment.feetech.servo_specs import PROFILE_VELOCITY_UNIT_RAD_PER_SEC


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


def make_gripper(driver=None, gripper_type="ace_leader", joint_sign=1, fragile=True):
    config = {
        "joint_id": 4,
        "joint_sign": joint_sign,
        "home_pose": 0.0,
        "servo_type": "HL3915",
        "gripper_type": gripper_type,
        "enable_fragile_force_control": fragile,
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


def test_fragile_tuning_config_keys_are_ignored():
    driver = PositionRecordingDriver(({4: 700}, {4: 0}, {4: 0}))
    config = {
        "joint_id": 4,
        "joint_sign": 1,
        "home_pose": 0.0,
        "servo_type": "HL3915",
        "gripper_type": "ace_follower",
        "enable_fragile_force_control": True,
        "fragile_close_velocity": 99.0,
        "fragile_max_hold_torque_nm": 99.0,
    }
    gripper = Gripper(config, driver=driver)

    gripper.set_fragile_position(0.2)

    _, _, kwargs = driver.profile_calls[0]
    assert kwargs["velocities_raw"][0] == pytest.approx(round(0.12 / PROFILE_VELOCITY_UNIT_RAD_PER_SEC))
    expected_current = round(((1000.0 / 9.3) * (0.12 / 0.0981) + 260) / 6.5)
    assert kwargs["currents_raw"][0] == pytest.approx(expected_current)


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


def test_fragile_gripper_disabled_uses_plain_position_command():
    driver = PositionRecordingDriver(({4: 0}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver, fragile=False)

    assert not gripper.set_fragile_position(0.25)
    assert driver.profile_calls == []


def test_fragile_gripper_closes_slowly_with_safe_current_limit():
    driver = PositionRecordingDriver(({4: 700}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver, gripper_type="ace_follower")

    assert gripper.set_fragile_position(0.2)

    sent_ids, encoded_positions, kwargs = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([4]))
    assert encoded_positions[0] == pytest.approx(179, abs=1)
    assert kwargs["velocities_raw"][0] == pytest.approx(round(0.12 / PROFILE_VELOCITY_UNIT_RAD_PER_SEC))
    expected_current = round(((1000.0 / 9.3) * (0.12 / 0.0981) + 260) / 6.5)
    assert kwargs["currents_raw"][0] == pytest.approx(expected_current)
    assert gripper.get_force_control_state().status == GripperForceStatus.CLOSING.value


def test_fragile_gripper_stays_idle_near_command_position():
    driver = PositionRecordingDriver(({4: 448}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver, gripper_type="ace_follower")

    assert gripper.set_fragile_position(0.51)

    sent_ids, encoded_positions, kwargs = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([4]))
    assert encoded_positions[0] == pytest.approx(457, abs=1)
    assert kwargs["currents_raw"][0] == 500
    assert gripper.get_force_control_state().status == GripperForceStatus.IDLE.value


def test_fragile_gripper_enters_holding_after_confirmed_contact():
    contact_current = round(((1000.0 / 9.3) * (0.08 / 0.0981) + 260) / 6.5)
    driver = PositionRecordingDriver(({4: 448}, {4: 0}, {4: contact_current}))
    gripper = make_gripper(driver=driver, gripper_type="ace_follower")
    state = gripper.get_state()

    for _ in range(3):
        assert gripper.set_fragile_position(0.1, state=state)

    gripper_state = gripper.get_force_control_state()
    assert gripper_state.status == GripperForceStatus.HOLDING.value
    assert gripper_state.hold_position == pytest.approx(0.5)
    assert gripper_state.contact_torque_nm == pytest.approx(state.motor_torque_magnitude)
    assert gripper_state.hold_torque_nm == pytest.approx(state.motor_torque_magnitude + 0.02)
    _, encoded_positions, kwargs = driver.profile_calls[-1]
    assert encoded_positions[0] == pytest.approx(448, abs=1)
    expected_current = round(((1000.0 / 9.3) * (gripper_state.hold_torque_nm / 0.0981) + 260) / 6.5)
    assert kwargs["currents_raw"][0] == pytest.approx(expected_current)


def test_fragile_gripper_holding_ignores_further_close_command():
    driver = PositionRecordingDriver(({4: 448}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver, gripper_type="ace_follower")
    gripper._force_state.status = GripperForceStatus.HOLDING.value
    gripper._force_state.hold_position = 0.5
    gripper._force_state.hold_torque_nm = 0.09

    gripper.set_fragile_position(0.1)

    _, encoded_positions, kwargs = driver.profile_calls[0]
    assert encoded_positions[0] == pytest.approx(448, abs=1)
    expected_current = round(((1000.0 / 9.3) * (0.09 / 0.0981) + 260) / 6.5)
    assert kwargs["currents_raw"][0] == pytest.approx(expected_current)
    assert gripper.get_force_control_state().status == GripperForceStatus.HOLDING.value


def test_fragile_gripper_release_command_exits_holding_and_opens():
    driver = PositionRecordingDriver(({4: 448}, {4: 0}, {4: 0}))
    gripper = make_gripper(driver=driver, gripper_type="ace_follower")
    gripper._force_state.status = GripperForceStatus.HOLDING.value
    gripper._force_state.hold_position = 0.5
    gripper._force_state.hold_torque_nm = 0.09

    gripper.set_fragile_position(0.7)

    sent_ids, encoded_positions, kwargs = driver.profile_calls[0]
    np.testing.assert_array_equal(sent_ids, np.array([4]))
    assert encoded_positions[0] == pytest.approx(627, abs=1)
    assert kwargs["velocities_raw"][0] == pytest.approx(round(0.8 / PROFILE_VELOCITY_UNIT_RAD_PER_SEC))
    assert kwargs["currents_raw"][0] == 500
    assert gripper.get_force_control_state().status == GripperForceStatus.OPENING.value


def test_fragile_gripper_hold_torque_is_clamped():
    contact_current = round(((1000.0 / 9.3) * (0.20 / 0.0981) + 260) / 6.5)
    driver = PositionRecordingDriver(({4: 448}, {4: 0}, {4: contact_current}))
    gripper = make_gripper(driver=driver, gripper_type="ace_follower")
    state = gripper.get_state()

    for _ in range(3):
        gripper.set_fragile_position(0.1, state=state)

    assert gripper.get_force_control_state().hold_torque_nm == pytest.approx(0.12)
