from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ace_robot_zmq.px4_xrce import (
    ArmJointStateEncoder,
    ArmJointStateSchema,
    Px4XrceBridge,
    Px4XrceError,
    Px4XrceOptions,
)

from acetele.config import load_robot_spec
from acetele.core import JointState, JointUnit, RobotState
from acetele.specification import ArmSpec, JointSpec

repository_root = Path(__file__).resolve().parents[3]


def _follower_spec():
    return load_robot_spec(
        repository_root
        / "acetele"
        / "config"
        / "presets"
        / "ace_follower"
        / "feetech_hls_ttl.toml"
    )


def _joint(name: str, servo_id: int) -> JointSpec:
    return JointSpec(name, servo_id, "HL3915", 1, 0.0)


def _spec_with_arm_counts(*counts: int):
    base = _follower_spec()
    buses = []
    arms = []
    for arm_index, count in enumerate(counts):
        bus = replace(
            base.buses[0],
            name=f"arm{arm_index}",
            port=f"mock://arm{arm_index}",
        )
        joints = tuple(
            _joint(f"arm{arm_index}_joint_{joint_index + 1}", joint_index)
            for joint_index in range(count)
        )
        buses.append(bus)
        arms.append(
            ArmSpec(
                f"arm{arm_index}",
                bus.name,
                joints,
                end_effector=None,
            )
        )
    return replace(base, buses=tuple(buses), arms=tuple(arms))


def _state(spec, *, timestamp_ns: int = 900_000_000) -> RobotState:
    groups = {}
    offset = 0
    for arm in spec.arms:
        count = len(arm.joints)
        values = np.arange(offset, offset + count, dtype=float) / 10.0
        groups[arm.name] = JointState(
            tuple(joint.name for joint in arm.joints),
            values,
            values + 1.0,
            np.zeros(count),
            timestamp_ns,
            1,
            JointUnit.RADIAN,
        )
        offset += count
    return RobotState(groups, {})


def test_packaged_schema_is_identical_to_px4_msgs():
    packaged = ArmJointStateSchema().message_path.read_bytes()
    px4_message = (
        repository_root / "third_party" / "px4_msgs" / "msg" / "ArmJointState.msg"
    ).read_bytes()

    assert packaged == px4_message
    assert hashlib.sha256(packaged).hexdigest() == ArmJointStateSchema().sha256


@pytest.mark.parametrize("arm_counts", ((4,), (7,), (7, 7)))
def test_encoder_packs_arm_only_state_in_assembly_order(arm_counts):
    schema = ArmJointStateSchema()
    spec = _spec_with_arm_counts(*arm_counts)
    payload = ArmJointStateEncoder(spec).encode(
        _state(spec),
        sequence=42,
        monotonic_now_ns=1_000_000_000,
        wall_now_ns=5_000_000_000,
    )

    unpacked = schema.payload_struct.unpack(payload)
    count = sum(arm_counts)
    assert len(payload) == 136
    assert unpacked[:5] == (5_000_000, 4_900_000, 42, count, True)
    positions = unpacked[5 : 5 + schema.maximum_joints]
    velocities = unpacked[5 + schema.maximum_joints :]
    assert positions[:count] == pytest.approx(np.arange(count) / 10.0)
    assert velocities[:count] == pytest.approx(np.arange(count) / 10.0 + 1.0)
    assert positions[count:] == (0.0,) * (schema.maximum_joints - count)
    assert velocities[count:] == (0.0,) * (schema.maximum_joints - count)


def test_encoder_accepts_explicit_px4_clock_domain_timestamps():
    spec = _spec_with_arm_counts(4)
    payload = ArmJointStateEncoder(spec).encode(
        _state(spec),
        sequence=7,
        monotonic_now_ns=1_000_000_000,
        wall_now_ns=9_000_000_000,
        timestamp_us=5000,
        timestamp_sample_us=4000,
    )

    assert ArmJointStateSchema().payload_struct.unpack(payload)[:3] == (5000, 4000, 7)


@pytest.mark.parametrize(
    ("timestamp_us", "timestamp_sample_us"),
    ((None, 1), (1, None), (0, 0), (1, 2)),
)
def test_encoder_rejects_invalid_explicit_px4_timestamps(
    timestamp_us,
    timestamp_sample_us,
):
    spec = _spec_with_arm_counts(4)
    with pytest.raises(ValueError, match="timestamps"):
        ArmJointStateEncoder(spec).encode(
            _state(spec),
            sequence=0,
            monotonic_now_ns=1_000_000_000,
            wall_now_ns=9_000_000_000,
            timestamp_us=timestamp_us,
            timestamp_sample_us=timestamp_sample_us,
        )


def test_encoder_excludes_end_effector_and_rejects_wrong_arm_names():
    spec = _follower_spec()
    arm = spec.arms[0]
    arm_state = JointState(
        tuple(joint.name for joint in arm.joints),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        10,
        0,
    )
    gripper = arm.end_effector
    assert gripper is not None
    state = RobotState(
        {
            arm.name: arm_state,
            f"{arm.name}.end_effector": JointState(
                (gripper.joint.name,),
                np.ones(1),
                np.zeros(1),
                np.zeros(1),
                10,
                0,
                JointUnit.NORMALIZED,
            ),
        },
        {},
    )

    payload = ArmJointStateEncoder(spec).encode(
        state,
        sequence=0,
        monotonic_now_ns=10,
        wall_now_ns=10_000,
    )
    assert ArmJointStateSchema().payload_struct.unpack(payload)[3] == 4

    wrong_state = RobotState(
        {arm.name: replace(arm_state, names=("wrong", "j2", "j3", "j4"))},
        {},
    )
    with pytest.raises(ValueError, match="names do not match"):
        ArmJointStateEncoder(spec).encode(
            wrong_state,
            sequence=0,
            monotonic_now_ns=10,
            wall_now_ns=10_000,
        )


def test_encoder_rejects_too_many_joints_and_future_samples():
    with pytest.raises(ValueError, match="between 4 and 14"):
        ArmJointStateEncoder(_spec_with_arm_counts(8, 7))

    spec = _spec_with_arm_counts(4)
    with pytest.raises(ValueError, match="timestamp is in the future"):
        ArmJointStateEncoder(spec).encode(
            _state(spec, timestamp_ns=101),
            sequence=0,
            monotonic_now_ns=100,
            wall_now_ns=100,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("agent_port", 0),
        ("domain_id", -1),
        ("client_key", 0),
        ("client_key", 1),
        ("startup_timeout_s", float("nan")),
    ),
)
def test_xrce_options_reject_invalid_network_values(tmp_path, field_name, value):
    values = {"prefix": tmp_path, field_name: value}
    with pytest.raises(ValueError):
        Px4XrceOptions(**values)


def test_schema_matches_px4_generated_cdr_layout():
    schema = ArmJointStateSchema()
    payload = schema.payload_struct.pack(
        1,
        2,
        3,
        4,
        True,
        *range(14),
        *range(14),
    )

    assert len(payload) == 136
    assert payload[:22] == struct.pack("<QQIB?", 1, 2, 3, 4, True)
    assert payload[22:24] == b"\0\0"


def test_bridge_rejects_an_unpinned_agent_before_starting_processes(tmp_path):
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "share" / "ace-px4-xrce").mkdir(parents=True)
    for name in ("MicroXRCEAgent", "ace-px4-xrce-publisher"):
        executable = prefix / "bin" / name
        executable.write_text("", encoding="utf-8")
        executable.chmod(0o700)
    schema = ArmJointStateSchema()
    (prefix / "share" / "ace-px4-xrce" / "manifest.json").write_text(
        json.dumps(
            {
                "agent_version": "3.0.1",
                "client_version": "2.4.0",
                "schema_sha256": schema.sha256,
            }
        ),
        encoding="utf-8",
    )
    started = []
    bridge = Px4XrceBridge(
        _spec_with_arm_counts(4),
        Px4XrceOptions(prefix=prefix),
        popen_factory=lambda *_args, **_kwargs: started.append(True),
    )

    with pytest.raises(Px4XrceError, match="incompatible"):
        bridge.start()

    assert not started


def test_bridge_rejects_an_occupied_agent_port_before_starting_processes(tmp_path):
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "share" / "ace-px4-xrce").mkdir(parents=True)
    for name in ("MicroXRCEAgent", "ace-px4-xrce-publisher"):
        executable = prefix / "bin" / name
        executable.write_text("", encoding="utf-8")
        executable.chmod(0o700)
    schema = ArmJointStateSchema()
    (prefix / "share" / "ace-px4-xrce" / "manifest.json").write_text(
        json.dumps(
            {
                "agent_version": "2.4.2",
                "client_version": "2.4.0",
                "schema_sha256": schema.sha256,
            }
        ),
        encoding="utf-8",
    )
    occupied = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    occupied.bind(("0.0.0.0", 0))
    port = occupied.getsockname()[1]
    started = []
    bridge = Px4XrceBridge(
        _spec_with_arm_counts(4),
        Px4XrceOptions(prefix=prefix, agent_port=port),
        popen_factory=lambda *_args, **_kwargs: started.append(True),
    )
    try:
        with pytest.raises(Px4XrceError, match="already in use"):
            bridge.start()
    finally:
        occupied.close()

    assert not started


@pytest.mark.skipif(
    "ACETELE_TEST_XRCE_PREFIX" not in os.environ,
    reason="native XRCE installation was not requested",
)
def test_native_agent_sidecar_session_and_latest_sample_round_trip():
    prefix = Path(os.environ["ACETELE_TEST_XRCE_PREFIX"])
    port_probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port_probe.bind(("127.0.0.1", 0))
    port = port_probe.getsockname()[1]
    port_probe.close()

    spec = _follower_spec()
    bridge = Px4XrceBridge(
        spec,
        Px4XrceOptions(prefix=prefix, agent_port=port, startup_timeout_s=10.0),
    )
    try:
        bridge.start()
        now_ns = time.monotonic_ns()
        assert bridge.publish(_state(spec, timestamp_ns=now_ns), sequence=17)
        deadline = time.monotonic() + 1.0
        diagnostics = bridge.diagnostics()
        while diagnostics.last_acknowledged_sequence != 17 and time.monotonic() < deadline:
            time.sleep(0.01)
            diagnostics = bridge.diagnostics()
        assert diagnostics.last_acknowledged_sequence == 17
        assert diagnostics.sent_samples == 1
        assert diagnostics.acknowledged_samples == 1
    finally:
        bridge.close()


def test_sustained_state_ipc_backpressure_becomes_fatal(tmp_path):
    class RunningProcess:
        @staticmethod
        def poll():
            return None

    class BlockedSocket:
        @staticmethod
        def send(_payload):
            raise BlockingIOError

    clock_values = iter((1_000_000_000, 1_100_000_001))
    spec = _spec_with_arm_counts(4)
    bridge = Px4XrceBridge(
        spec,
        Px4XrceOptions(prefix=tmp_path),
        clock_ns=lambda: next(clock_values),
        wall_clock_ns=lambda: 2_000_000_000,
    )
    bridge._started = True
    bridge._agent = RunningProcess()
    bridge._publisher = RunningProcess()
    bridge._state_socket = BlockedSocket()
    state = _state(spec, timestamp_ns=900_000_000)

    assert not bridge.publish(state, sequence=0)
    with pytest.raises(Px4XrceError, match="remained blocked"):
        bridge.publish(state, sequence=1)


def test_bridge_converts_invalid_measured_state_to_a_link_failure(tmp_path):
    class RunningProcess:
        @staticmethod
        def poll():
            return None

    spec = _spec_with_arm_counts(4)
    invalid_state = RobotState({}, {})
    bridge = Px4XrceBridge(spec, Px4XrceOptions(prefix=tmp_path))
    bridge._started = True
    bridge._agent = RunningProcess()
    bridge._publisher = RunningProcess()

    with pytest.raises(Px4XrceError, match="arm state is invalid"):
        bridge.publish(invalid_state, sequence=0)


def test_bridge_rejects_non_monotonic_explicit_px4_timestamps(tmp_path):
    class RunningProcess:
        @staticmethod
        def poll():
            return None

    class RecordingSocket:
        @staticmethod
        def send(payload):
            return len(payload)

    spec = _spec_with_arm_counts(4)
    bridge = Px4XrceBridge(spec, Px4XrceOptions(prefix=tmp_path))
    bridge._started = True
    bridge._agent = RunningProcess()
    bridge._publisher = RunningProcess()
    bridge._state_socket = RecordingSocket()
    state = _state(spec)

    assert bridge.publish(
        state,
        sequence=0,
        timestamp_us=5000,
        timestamp_sample_us=5000,
    )
    with pytest.raises(Px4XrceError, match="increase monotonically"):
        bridge.publish(
            state,
            sequence=1,
            timestamp_us=5000,
            timestamp_sample_us=5000,
        )


def test_acknowledgement_age_uses_the_sidecar_monotonic_timestamp(tmp_path):
    receiver, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    receiver.setblocking(False)
    bridge = Px4XrceBridge(_spec_with_arm_counts(4), Px4XrceOptions(prefix=tmp_path))
    bridge._ack_socket = receiver
    bridge._last_sent_ns = 800_000_000
    bridge._first_unacknowledged_sent_ns = 800_000_000
    sender.send(b"ACK 3 800")
    try:
        bridge._drain_acknowledgements(1_000_000_000)
        assert bridge.diagnostics().last_acknowledged_sequence == 3
        with pytest.raises(Px4XrceError, match="acknowledgement is stale"):
            bridge._require_fresh_ack(1_000_000_000)
    finally:
        receiver.close()
        sender.close()
        bridge._ack_socket = None
