from __future__ import annotations

from ace_robot_zmq.options import PeerRole, ZmqTeleopOptions
from ace_robot_zmq.protocol import FollowerFrame, MessagePackCodec
from ace_robot_zmq.sdk import PoseLeaderClient
from ace_robot_zmq.transport import TransportDiagnostics

from acetele.core import EndEffectorPose, JointState
from acetele.runtime.teleop import FollowerSyncStatus, LeaderSyncMode


class LoopbackPeer:
    def __init__(self, incoming: bytes) -> None:
        self.incoming = incoming
        self.sent: list[bytes] = []

    def open(self) -> None:
        pass

    def receive(self, *, timeout_ms: int = 0):
        payload, self.incoming = self.incoming, b""
        return payload or None

    def send(self, payload: bytes) -> bool:
        self.sent.append(payload)
        return True

    def record_admission(self, _admission) -> None:
        pass

    def record_rejection(self, _reason: str) -> None:
        pass

    def record_encode_duration(self, _duration_ns: int) -> None:
        pass

    def diagnostics(self) -> TransportDiagnostics:
        return TransportDiagnostics()

    def close(self) -> None:
        pass


def test_pose_client_performs_sync_handshake_before_publishing_pose():
    codec = MessagePackCodec()
    feedback = FollowerFrame(
        b"f" * 16,
        0,
        1,
        FollowerSyncStatus.READY,
        {
            "single": JointState(
                ("joint_1",),
                (0.0,),
                (0.0,),
                (0.0,),
                1,
                0,
            )
        },
    )
    peer = LoopbackPeer(codec.encode_follower(feedback))
    client = PoseLeaderClient(
        ZmqTeleopOptions(PeerRole.LEADER, "127.0.0.1", "127.0.0.1"),
        peer=peer,
        codec=codec,
        session_id=b"v" * 16,
    )
    client.set_pose(
        EndEffectorPose(2, "base_link", (0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    )

    client.step()

    command = codec.decode_leader(peer.sent[-1])
    assert client.mode == LeaderSyncMode.TRACKING
    assert command.mode == LeaderSyncMode.TRACKING
    assert command.ee_pose_command is not None
