"""Safe external-pose producer for VR and other non-robot leader devices."""

from __future__ import annotations

import time
import uuid
from types import MappingProxyType
from typing import Mapping, Optional

from ace_robot_zmq.options import PeerRole, ZmqTeleopOptions
from ace_robot_zmq.protocol import (
    FollowerFrame,
    JointTarget,
    LeaderFrame,
    MessagePackCodec,
    ProtocolError,
)
from ace_robot_zmq.transport import PeerSequenceGate, TransportDiagnostics, ZmqPeer

from acetele.core import EndEffectorPose
from acetele.runtime.teleop import FollowerSyncStatus, LeaderSyncMode, TeleopMode


class PoseLeaderClient:
    """Publish the latest external TCP pose through the standard synchronization flow.

    The caller owns sampling cadence and calls :meth:`step` repeatedly. No hidden
    thread is created, which keeps VR event-loop integration explicit and predictable.
    """

    def __init__(
        self,
        options: ZmqTeleopOptions,
        *,
        peer: Optional[ZmqPeer] = None,
        codec: Optional[MessagePackCodec] = None,
        session_id: Optional[bytes] = None,
    ) -> None:
        if options.role != PeerRole.LEADER:
            raise ValueError("PoseLeaderClient requires leader ZMQ options")
        self.options = options
        self.peer = peer or ZmqPeer(options)
        self.codec = codec or MessagePackCodec(
            maximum_frame_bytes=options.maximum_frame_bytes
        )
        self._gate = PeerSequenceGate(
            heartbeat_timeout_ns=options.heartbeat_timeout_ns
        )
        self._session_id = session_id or uuid.uuid4().bytes
        if len(self._session_id) != 16:
            raise ValueError("session_id must contain 16 bytes")
        self._sequence = 0
        self._mode = LeaderSyncMode.SYNC_REQUEST
        self._pose: Optional[EndEffectorPose] = None
        self._end_effectors: Mapping[str, JointTarget] = MappingProxyType({})
        self._latest_follower: Optional[FollowerFrame] = None
        self._closed = False

    @property
    def mode(self) -> LeaderSyncMode:
        return self._mode

    @property
    def latest_follower(self) -> Optional[FollowerFrame]:
        return self._latest_follower

    def connect(self) -> None:
        self.peer.open()

    def set_pose(self, pose: EndEffectorPose) -> None:
        if not isinstance(pose, EndEffectorPose):
            raise ValueError("pose must be an EndEffectorPose")
        self._pose = pose

    def set_end_effectors(self, commands: Mapping[str, JointTarget]) -> None:
        values = dict(commands)
        if any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(command, JointTarget)
            for name, command in values.items()
        ):
            raise ValueError("end-effector commands must map names to JointTarget")
        self._end_effectors = MappingProxyType(values)

    def step(self, *, timeout_ms: int = 0) -> Optional[FollowerFrame]:
        """Receive feedback, advance handshake, and publish the latest pose once."""

        payload = self.peer.receive(timeout_ms=timeout_ms)
        if payload is not None:
            self._process_follower(payload)
        if self._mode == LeaderSyncMode.TRACKING and self._pose is None:
            # Keep requesting synchronization until a real pose is available. This
            # prevents an empty tracking frame from refreshing follower state.
            self._mode = LeaderSyncMode.SYNC_REQUEST
        frame = LeaderFrame(
            self._session_id,
            self._sequence,
            time.time_ns(),
            self._mode,
            TeleopMode.EE_POSE,
            None,
            self._pose if self._mode == LeaderSyncMode.TRACKING else None,
            self._end_effectors if self._mode == LeaderSyncMode.TRACKING else {},
        )
        self._sequence += 1
        started_ns = time.monotonic_ns()
        payload = self.codec.encode_leader(frame)
        self.peer.record_encode_duration(time.monotonic_ns() - started_ns)
        self.peer.send(payload)
        return self._latest_follower

    def stop(self) -> None:
        self._mode = LeaderSyncMode.STOP

    def diagnostics(self) -> TransportDiagnostics:
        return self.peer.diagnostics()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.peer.close()

    def _process_follower(self, payload: bytes) -> None:
        now_ns = time.monotonic_ns()
        try:
            frame = self.codec.decode_follower(payload)
        except (ProtocolError, TypeError, ValueError) as exc:
            self.peer.record_rejection(str(exc))
            return
        admission = self._gate.admit(
            frame.session_id,
            frame.sequence,
            now_ns=now_ns,
        )
        self.peer.record_admission(admission)
        if not admission.accepted:
            return
        if admission.new_session:
            self._mode = LeaderSyncMode.SYNC_REQUEST
        self._latest_follower = frame
        if frame.status == FollowerSyncStatus.READY and self._pose is not None:
            self._mode = LeaderSyncMode.TRACKING
        elif frame.status in (
            FollowerSyncStatus.IDLE,
            FollowerSyncStatus.LOST,
        ):
            self._mode = LeaderSyncMode.SYNC_REQUEST
        elif frame.status == FollowerSyncStatus.FAULT:
            self._mode = LeaderSyncMode.STOP


__all__ = ["PoseLeaderClient"]
