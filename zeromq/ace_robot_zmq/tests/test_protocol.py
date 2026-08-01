from __future__ import annotations

import msgpack
import numpy as np
import pytest
from ace_robot_zmq.protocol import (
    FollowerFrame,
    JointTarget,
    LeaderFrame,
    MessagePackCodec,
    ProtocolError,
)

from acetele.core import EndEffectorPose, JointState, JointUnit
from acetele.runtime.teleop import FollowerSyncStatus, LeaderSyncMode, TeleopMode


def _pose() -> EndEffectorPose:
    return EndEffectorPose(10, "base_link", [0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])


def _state(count: int) -> JointState:
    names = tuple(f"joint_{index}" for index in range(1, count + 1))
    return JointState(
        names,
        np.linspace(0.0, 0.1, count),
        np.zeros(count),
        np.zeros(count),
        20,
        3,
        JointUnit.RADIAN,
    )


@pytest.mark.parametrize("count", (4, 7, 14))
def test_follower_frame_round_trips_supported_arm_sizes(count):
    codec = MessagePackCodec()
    frame = FollowerFrame(
        b"f" * 16,
        7,
        100,
        FollowerSyncStatus.TRACKING,
        {"single": _state(count)},
        _pose(),
    )

    decoded = codec.decode_follower(codec.encode_follower(frame))

    assert decoded.session_id == frame.session_id
    assert decoded.status == FollowerSyncStatus.TRACKING
    assert decoded.joint_states["single"].names == frame.joint_states["single"].names
    assert decoded.joint_states["single"].positions.tolist() == pytest.approx(
        frame.joint_states["single"].positions
    )
    assert decoded.ee_pose_state is not None
    assert decoded.ee_pose_state.position_m.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_leader_frame_round_trips_joint_pose_and_end_effector_variants():
    codec = MessagePackCodec()
    joint = LeaderFrame(
        b"l" * 16,
        1,
        200,
        LeaderSyncMode.TRACKING,
        TeleopMode.JOINT,
        JointTarget(("joint_1", "joint_2"), (0.1, 0.2)),
        end_effector_commands={
            "single.end_effector": JointTarget(("joint_5",), (0.5,))
        },
    )
    pose = LeaderFrame(
        b"l" * 16,
        2,
        201,
        LeaderSyncMode.TRACKING,
        TeleopMode.EE_POSE,
        ee_pose_command=_pose(),
    )

    decoded_joint = codec.decode_leader(codec.encode_leader(joint))
    decoded_pose = codec.decode_leader(codec.encode_leader(pose))

    assert decoded_joint.arm_command == joint.arm_command
    assert decoded_joint.end_effector_commands["single.end_effector"].positions == (0.5,)
    assert decoded_pose.ee_pose_command is not None
    assert decoded_pose.ee_pose_command.quaternion_xyzw.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_codec_rejects_unknown_version_nonfinite_values_and_oversized_frames():
    codec = MessagePackCodec(maximum_frame_bytes=1024)
    valid = codec.encode_leader(
        LeaderFrame(
            b"x" * 16,
            0,
            1,
            LeaderSyncMode.IDLE,
            TeleopMode.JOINT,
        )
    )
    document = msgpack.unpackb(valid, raw=False)
    document["version"] = 99

    with pytest.raises(ProtocolError, match="unsupported protocol version"):
        codec.decode_leader(msgpack.packb(document, use_bin_type=True))
    with pytest.raises(ProtocolError, match="finite"):
        JointTarget(("joint_1",), (float("nan"),))
    with pytest.raises(ProtocolError, match="exceeds"):
        codec.decode_leader(b"x" * 1025)


@pytest.mark.parametrize("value", (True, "1.0", None, {}))
def test_codec_rejects_values_that_only_look_numeric(value):
    with pytest.raises(ProtocolError, match="real numbers"):
        JointTarget(("joint_1",), (value,))


def test_decoded_arrays_and_mappings_do_not_mutate_internal_contracts():
    codec = MessagePackCodec()
    decoded = codec.decode_follower(
        codec.encode_follower(
            FollowerFrame(
                b"s" * 16,
                1,
                2,
                FollowerSyncStatus.READY,
                {"single": _state(4)},
            )
        )
    )

    with pytest.raises(ValueError):
        decoded.joint_states["single"].positions[0] = 5.0
    with pytest.raises(TypeError):
        decoded.joint_states["other"] = _state(4)
