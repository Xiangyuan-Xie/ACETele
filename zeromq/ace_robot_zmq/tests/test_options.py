from __future__ import annotations

import pytest
from ace_robot_zmq.options import PeerRole, ZmqTeleopOptions


def test_role_owns_the_expected_bind_endpoint():
    leader = ZmqTeleopOptions(PeerRole.LEADER, "0.0.0.0", "follower.local")
    follower = ZmqTeleopOptions(PeerRole.FOLLOWER, "0.0.0.0", "leader.local")

    assert leader.command_endpoint == "tcp://0.0.0.0:5555"
    assert leader.state_endpoint == "tcp://follower.local:5556"
    assert follower.command_endpoint == "tcp://leader.local:5555"
    assert follower.state_endpoint == "tcp://0.0.0.0:5556"


@pytest.mark.parametrize(
    "kwargs",
    (
        {"peer_host": ""},
        {"peer_host": "tcp://host"},
        {"command_port": 0},
        {"command_port": 5556},
        {"cycle_hz": 0.0},
        {"maximum_frame_bytes": 65_537},
    ),
)
def test_options_reject_ambiguous_or_unsafe_endpoints(kwargs):
    values = {"role": PeerRole.LEADER, "bind_host": "0.0.0.0", "peer_host": "host"}
    values.update(kwargs)
    with pytest.raises(ValueError):
        ZmqTeleopOptions(**values)
