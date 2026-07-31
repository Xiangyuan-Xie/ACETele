from __future__ import annotations

import pytest

from acetele.config.specs import (
    ArmSpec,
    BusSpec,
    BusType,
    ControlSpec,
    DexterousHandSpec,
    JointSpec,
    ParallelGripperSpec,
    PositionControlTuning,
    RobotSpec,
)


def _joint(name: str = "joint_1", servo_id: int = 1) -> JointSpec:
    return JointSpec(name, servo_id, "HL3915", 1, 0.0)


def test_feetech_packet_physical_layer_matches_family():
    with pytest.raises(ValueError, match="HLS requires"):
        BusSpec(
            "arm",
            BusType.FEETECH_PACKET,
            "/dev/ttyUSB0",
            1_000_000,
            100.0,
            physical_layer="rs485",
            family="hls",
        )


@pytest.mark.parametrize("value", (0, 1, None, "true"))
def test_bus_identity_acknowledgement_requires_a_real_boolean(value):
    with pytest.raises(ValueError, match="allow_unverified_identity"):
        BusSpec(
            "arm",
            BusType.FEETECH_PACKET,
            "/dev/ttyUSB0",
            1_000_000,
            100.0,
            physical_layer="ttl",
            family="hls",
            allow_unverified_identity=value,
        )


def test_robot_spec_freezes_sequences_and_validates_bus_references():
    bus = BusSpec(
        "arm",
        BusType.FEETECH_PACKET,
        "/dev/ttyUSB0",
        1_000_000,
        100.0,
        physical_layer="ttl",
        family="hls",
    )
    joints = [_joint()]
    buses = [bus]
    arms = [ArmSpec("single", "arm", joints)]
    spec = RobotSpec("ace_follower", buses, arms)

    joints.append(_joint("joint_2", 2))
    buses.clear()
    arms.clear()

    assert len(spec.buses) == 1
    assert len(spec.arms) == 1
    assert len(spec.arms[0].joints) == 1

    with pytest.raises(ValueError, match="unknown bus"):
        RobotSpec("robot", (bus,), (ArmSpec("single", "other", (_joint(),)),))


def test_gravity_position_requires_per_joint_compliance():
    with pytest.raises(ValueError, match="compliance calibration"):
        ControlSpec(gravity_position=True)

    with pytest.raises(ValueError, match="match its joint count"):
        ArmSpec(
            "single",
            "arm",
            (_joint(),),
            ControlSpec(
                gravity_position=True,
                gravity_compliance_rad_per_nm=(0.01, 0.02),
            ),
        )


def test_position_control_tuning_is_validated_as_one_specification():
    with pytest.raises(ValueError, match="minimum_dt_s"):
        PositionControlTuning(minimum_dt_s=0.1, maximum_dt_s=0.05)
    with pytest.raises(ValueError, match="target_stable_threshold_rad"):
        PositionControlTuning(
            target_stable_threshold_rad=0.05,
            target_reset_threshold_rad=0.05,
        )
    with pytest.raises(ValueError, match="position_tuning"):
        ControlSpec(position_tuning={})

    tuning = PositionControlTuning(adaptive_deadband_rad=0.01)
    assert tuning.adaptive_deadband_rad == 0.01


def test_robot_spec_rejects_multiple_actors_for_one_physical_port():
    first_bus = BusSpec(
        "first",
        BusType.FEETECH_PACKET,
        "/dev/ttyUSB0",
        1_000_000,
        100.0,
        physical_layer="ttl",
        family="hls",
    )
    second_bus = BusSpec(
        "second",
        BusType.FEETECH_PACKET,
        "/dev/ttyUSB0",
        1_000_000,
        100.0,
        physical_layer="ttl",
        family="hls",
    )

    with pytest.raises(ValueError, match="ports must be unique"):
        RobotSpec(
            "robot",
            (first_bus, second_bus),
            (
                ArmSpec("left", "first", (_joint("joint_1", 1),)),
                ArmSpec("right", "second", (_joint("joint_2", 1),)),
            ),
        )


def test_end_effector_specs_reject_noninvertible_or_reserved_addresses():
    with pytest.raises(ValueError, match=r"\(0, pi\)"):
        ParallelGripperSpec("arm", _joint(), 3.141592653589793)
    with pytest.raises(ValueError, match=r"\[1, 247\]"):
        DexterousHandSpec("hand", "linker", "O6", "right", 0)


def test_arm_tool_frame_is_explicit_and_strict_when_configured():
    arm = ArmSpec("single", "arm", (_joint(),), tool_frame="link_5")
    assert arm.tool_frame == "link_5"

    with pytest.raises(ValueError, match="tool_frame"):
        ArmSpec("single", "arm", (_joint(),), tool_frame="")
