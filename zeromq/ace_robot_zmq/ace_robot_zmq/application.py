"""Thin ZeroMQ loops composed around ACETele's transport-independent sessions."""

from __future__ import annotations

import math
import time
import uuid
from types import MappingProxyType
from typing import Callable, Mapping, Optional

import numpy as np
from ace_robot_zmq.options import PeerRole, ZmqTeleopOptions
from ace_robot_zmq.protocol import (
    FollowerFrame,
    JointTarget,
    LeaderFrame,
    MessagePackCodec,
    ProtocolError,
)
from ace_robot_zmq.px4_xrce import (
    Px4XrceBridge,
    Px4XrceDiagnostics,
    Px4XrceError,
    Px4XrceOptions,
)
from ace_robot_zmq.transport import PeerSequenceGate, TransportDiagnostics, ZmqPeer

from acetele.core import JointState, RobotState
from acetele.runtime import (
    FollowerRuntime,
    FollowerTeleopSession,
    LeaderTeleopSession,
    RobotRuntime,
)
from acetele.runtime.teleop import LeaderSyncMode, TeleopMode
from acetele.specification import ParallelGripperSpec, RobotSpec


def _close_preserving_primary(
    primary: Optional[BaseException],
    callbacks: tuple[Callable[[], None], ...],
) -> None:
    """Attempt every cleanup and preserve the operational error as primary."""

    cleanup_error: Optional[BaseException] = None
    for callback in callbacks:
        try:
            callback()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
    if primary is not None:
        if cleanup_error is not None:
            raise primary from cleanup_error
        raise primary
    if cleanup_error is not None:
        raise cleanup_error


class LeaderApplication:
    """Sample a physical leader and publish coherent commands to one follower."""

    def __init__(
        self,
        spec: RobotSpec,
        options: ZmqTeleopOptions,
        *,
        teleop_mode: TeleopMode = TeleopMode.JOINT,
        runtime_factory: Callable[..., RobotRuntime] = RobotRuntime,
        peer: Optional[ZmqPeer] = None,
        codec: Optional[MessagePackCodec] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        session_id: Optional[bytes] = None,
        end_effector_threshold: float = 0.001,
        end_effector_keepalive_ns: int = 100_000_000,
    ) -> None:
        if options.role != PeerRole.LEADER:
            raise ValueError("LeaderApplication requires leader ZMQ options")
        if spec.model != "ace_leader":
            raise ValueError("ZMQ leader requires an ace_leader RobotSpec")
        if not math.isfinite(end_effector_threshold) or end_effector_threshold <= 0.0:
            raise ValueError("end_effector_threshold must be finite and positive")
        if type(end_effector_keepalive_ns) is not int or end_effector_keepalive_ns <= 0:
            raise ValueError("end_effector_keepalive_ns must be a positive integer")
        runtime = runtime_factory(spec)
        self.session = LeaderTeleopSession(
            runtime,
            follower_timeout_ns=options.heartbeat_timeout_ns,
            teleop_mode=teleop_mode,
        )
        self.options = options
        self.peer = peer or ZmqPeer(options)
        self.codec = codec or MessagePackCodec(
            maximum_frame_bytes=options.maximum_frame_bytes
        )
        self._gate = PeerSequenceGate(
            heartbeat_timeout_ns=options.heartbeat_timeout_ns
        )
        self._clock_ns = clock_ns
        self._wall_clock_ns = wall_clock_ns
        self._session_id = session_id or uuid.uuid4().bytes
        if len(self._session_id) != 16:
            raise ValueError("session_id must contain 16 bytes")
        self._sequence = 0
        self._closed = False
        self._connected = False
        self._last_end_effector_positions: dict[str, np.ndarray] = {}
        self._last_end_effector_send_ns: dict[str, int] = {}
        self._end_effector_threshold = float(end_effector_threshold)
        self._end_effector_keepalive_ns = end_effector_keepalive_ns

    def connect(self) -> None:
        """Open network resources before connecting hardware, with full cleanup."""

        if self._connected:
            return
        self.peer.open()
        try:
            self.session.connect()
        except BaseException as exc:
            _close_preserving_primary(exc, (self.peer.close,))
        self._connected = True

    def process_incoming(self, payload: bytes, *, now_ns: Optional[int] = None) -> bool:
        """Validate one follower frame before it can refresh synchronization state."""

        received_ns = self._clock_ns() if now_ns is None else now_ns
        try:
            frame = self.codec.decode_follower(payload)
            arm_states = self._validate_follower_frame(frame)
        except (KeyError, ProtocolError, TypeError, ValueError) as exc:
            self.peer.record_rejection(str(exc))
            return False
        admission = self._gate.admit(
            frame.session_id,
            frame.sequence,
            now_ns=received_ns,
        )
        self.peer.record_admission(admission)
        if not admission.accepted:
            return False
        if admission.new_session:
            self.session.reset_peer()
        names = tuple(name for state in arm_states for name in state.names)
        positions = tuple(value for state in arm_states for value in state.positions)
        self.session.observe_follower_state(names, positions, now_ns=received_ns)
        self.session.observe_follower_status(frame.status, now_ns=received_ns)
        return True

    def publish_once(self, *, now_ns: Optional[int] = None) -> LeaderFrame:
        """Advance the leader session and publish one latest command snapshot."""

        current_ns = self._clock_ns() if now_ns is None else now_ns
        state = self.session.step(now_ns=current_ns)
        self._start_tracking_if_ready(state)
        arm_command = None
        pose_command = None
        if self.session.mode == LeaderSyncMode.TRACKING:
            if self.session.teleop_mode == TeleopMode.JOINT:
                arm_states = tuple(
                    state.joints[arm.name] for arm in self.session.runtime.spec.arms
                )
                arm_command = JointTarget(
                    tuple(name for sample in arm_states for name in sample.names),
                    tuple(value for sample in arm_states for value in sample.positions),
                )
            else:
                pose_command = self.session.end_effector_pose(
                    state,
                    timestamp_ns=current_ns,
                )
        frame = LeaderFrame(
            self._session_id,
            self._sequence,
            self._wall_clock_ns(),
            self.session.mode,
            self.session.teleop_mode,
            arm_command,
            pose_command,
            self._end_effector_commands(state, current_ns),
        )
        self._sequence += 1
        encode_started_ns = self._clock_ns()
        payload = self.codec.encode_leader(frame)
        self.peer.record_encode_duration(self._clock_ns() - encode_started_ns)
        self.peer.send(payload)
        return frame

    def run(self, should_stop: Callable[[], bool]) -> None:
        """Poll feedback immediately while publishing at the configured cycle rate."""

        self.connect()
        period_ns = round(1e9 / self.options.cycle_hz)
        next_publish_ns = self._clock_ns()
        while not should_stop():
            now_ns = self._clock_ns()
            timeout_ms = max(0, math.ceil((next_publish_ns - now_ns) / 1e6))
            payload = self.peer.receive(timeout_ms=timeout_ms)
            if payload is not None:
                self.process_incoming(payload, now_ns=self._clock_ns())
            now_ns = self._clock_ns()
            if now_ns >= next_publish_ns:
                self.publish_once(now_ns=now_ns)
                next_publish_ns = max(next_publish_ns + period_ns, now_ns + period_ns)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        callbacks = (self.session.close, self.peer.close) if self._connected else (self.peer.close,)
        self._connected = False
        _close_preserving_primary(None, callbacks)

    def diagnostics(self) -> TransportDiagnostics:
        """Return the latest immutable network counters and timing sample."""

        return self.peer.diagnostics()

    def _validate_follower_frame(self, frame: FollowerFrame) -> tuple[JointState, ...]:
        expected_groups = tuple(self.session.runtime.joint_groups)
        if tuple(frame.joint_states) != expected_groups:
            raise ProtocolError(
                f"follower groups must be {expected_groups}, got {tuple(frame.joint_states)}"
            )
        for group_name, group in self.session.runtime.joint_groups.items():
            state = frame.joint_states[group_name]
            if state.names != group.names or state.unit != group.unit:
                raise ProtocolError(f"follower state group '{group_name}' does not match RobotSpec")
        return tuple(frame.joint_states[arm.name] for arm in self.session.runtime.spec.arms)

    def _start_tracking_if_ready(self, state: RobotState) -> None:
        if self.session.mode != LeaderSyncMode.READY:
            return
        gripper_groups = tuple(
            f"{arm.name}.end_effector"
            for arm in self.session.runtime.spec.arms
            if isinstance(arm.end_effector, ParallelGripperSpec)
        )
        if not gripper_groups or all(
            float(state.joints[group].positions[0]) >= 1.0 for group in gripper_groups
        ):
            self.session.start_tracking()

    def _end_effector_commands(
        self,
        state: RobotState,
        now_ns: int,
    ) -> Mapping[str, JointTarget]:
        if self.session.mode != LeaderSyncMode.TRACKING:
            return MappingProxyType({})
        result: dict[str, JointTarget] = {}
        for group_name, sample in state.joints.items():
            if not group_name.endswith(".end_effector"):
                continue
            positions = np.asarray(sample.positions, dtype=float)
            previous = self._last_end_effector_positions.get(group_name)
            previous_ns = self._last_end_effector_send_ns.get(group_name)
            changed = previous is None or np.max(np.abs(positions - previous)) > self._end_effector_threshold
            keepalive = previous_ns is None or now_ns - previous_ns >= self._end_effector_keepalive_ns
            if not changed and not keepalive:
                continue
            result[group_name] = JointTarget(sample.names, tuple(positions))
            self._last_end_effector_positions[group_name] = positions.copy()
            self._last_end_effector_send_ns[group_name] = now_ns
        return MappingProxyType(result)


class FollowerApplication:
    """Apply validated latest commands and publish measured follower state."""

    def __init__(
        self,
        spec: RobotSpec,
        options: ZmqTeleopOptions,
        *,
        teleop_mode: TeleopMode = TeleopMode.JOINT,
        translation_scale: float = 2.0,
        rotation_scale: float = 1.0,
        runtime_factory: Callable[..., FollowerRuntime] = RobotRuntime,
        peer: Optional[ZmqPeer] = None,
        codec: Optional[MessagePackCodec] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        session_id: Optional[bytes] = None,
        xrce_options: Optional[Px4XrceOptions] = None,
        xrce_bridge: Optional[Px4XrceBridge] = None,
        xrce_timestamp_provider: Optional[
            Callable[[RobotState], tuple[int, int]]
        ] = None,
    ) -> None:
        if options.role != PeerRole.FOLLOWER:
            raise ValueError("FollowerApplication requires follower ZMQ options")
        if spec.model != "ace_follower":
            raise ValueError("ZMQ follower requires an ace_follower RobotSpec")
        runtime = runtime_factory(spec, command_timeout_ns=options.heartbeat_timeout_ns)
        self.session = FollowerTeleopSession(
            runtime,
            heartbeat_timeout_ns=options.heartbeat_timeout_ns,
            teleop_mode=teleop_mode,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
        )
        self.options = options
        self.peer = peer or ZmqPeer(options)
        self.xrce = xrce_bridge or Px4XrceBridge(
            spec,
            xrce_options or Px4XrceOptions(),
            clock_ns=clock_ns,
            wall_clock_ns=wall_clock_ns,
        )
        self.codec = codec or MessagePackCodec(
            maximum_frame_bytes=options.maximum_frame_bytes
        )
        self._gate = PeerSequenceGate(
            heartbeat_timeout_ns=options.heartbeat_timeout_ns
        )
        self._clock_ns = clock_ns
        self._wall_clock_ns = wall_clock_ns
        self._session_id = session_id or uuid.uuid4().bytes
        if len(self._session_id) != 16:
            raise ValueError("session_id must contain 16 bytes")
        self._sequence = 0
        self._px4_sequence = 0
        self._xrce_timestamp_provider = xrce_timestamp_provider
        self._closed = False
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return
        self.xrce.start()
        try:
            self.peer.open()
            self.session.connect()
        except BaseException as exc:
            _close_preserving_primary(exc, (self.peer.close, self.xrce.close))
        self._connected = True

    def process_incoming(self, payload: bytes, *, now_ns: Optional[int] = None) -> bool:
        """Submit only a semantically valid command from the active leader session."""

        received_ns = self._clock_ns() if now_ns is None else now_ns
        processing_started_ns = self._clock_ns()
        try:
            frame = self.codec.decode_leader(payload)
            self._validate_leader_frame(frame)
        except (KeyError, ProtocolError, TypeError, ValueError) as exc:
            self.peer.record_rejection(str(exc))
            return False
        admission = self._gate.admit(
            frame.session_id,
            frame.sequence,
            now_ns=received_ns,
        )
        self.peer.record_admission(admission)
        if not admission.accepted:
            return False
        if admission.new_session:
            self.session.reset_peer()
        try:
            self.session.set_mode(frame.mode)
            end_effectors = {
                name: (command.names, command.positions)
                for name, command in frame.end_effector_commands.items()
            }
            arm_accepted = (
                self.session.write_tracking_frame(
                    arm_names=(
                        None if frame.arm_command is None else frame.arm_command.names
                    ),
                    arm_positions=(
                        None if frame.arm_command is None else frame.arm_command.positions
                    ),
                    arm_pose=frame.ee_pose_command,
                    end_effectors=end_effectors,
                    now_ns=received_ns,
                )
                if frame.mode == LeaderSyncMode.TRACKING
                else False
            )
            if frame.mode == LeaderSyncMode.TRACKING and not arm_accepted:
                raise ProtocolError("tracking frame was not admitted by the follower session")
        except (RuntimeError, ValueError, ProtocolError) as exc:
            self.peer.record_rejection(str(exc))
            return False
        self.peer.record_runtime_stage_duration(
            self._clock_ns() - processing_started_ns
        )
        return True

    def publish_once(self, *, now_ns: Optional[int] = None) -> FollowerFrame:
        current_ns = self._clock_ns() if now_ns is None else now_ns
        state = self.session.read(now_ns=current_ns)
        try:
            timestamps = (
                None
                if self._xrce_timestamp_provider is None
                else self._xrce_timestamp_provider(state)
            )
            if timestamps is None:
                self.xrce.publish(state, sequence=self._px4_sequence)
            else:
                self.xrce.publish(
                    state,
                    sequence=self._px4_sequence,
                    timestamp_us=timestamps[0],
                    timestamp_sample_us=timestamps[1],
                )
        except Px4XrceError:
            self.session.reset_peer()
            raise
        self._px4_sequence = (self._px4_sequence + 1) & 0xFFFFFFFF
        pose = None
        if self.session.teleop_mode == TeleopMode.EE_POSE:
            pose = self.session.end_effector_pose(state, timestamp_ns=current_ns)
        frame = FollowerFrame(
            self._session_id,
            self._sequence,
            self._wall_clock_ns(),
            self.session.status,
            state.joints,
            pose,
        )
        self._sequence += 1
        encode_started_ns = self._clock_ns()
        payload = self.codec.encode_follower(frame)
        self.peer.record_encode_duration(self._clock_ns() - encode_started_ns)
        self.peer.send(payload)
        return frame

    def run(self, should_stop: Callable[[], bool]) -> None:
        self.connect()
        period_ns = round(1e9 / self.options.cycle_hz)
        next_publish_ns = self._clock_ns()
        while not should_stop():
            now_ns = self._clock_ns()
            timeout_ms = max(0, math.ceil((next_publish_ns - now_ns) / 1e6))
            payload = self.peer.receive(timeout_ms=timeout_ms)
            if payload is not None:
                self.process_incoming(payload, now_ns=self._clock_ns())
            now_ns = self._clock_ns()
            if now_ns >= next_publish_ns:
                self.publish_once(now_ns=now_ns)
                next_publish_ns = max(next_publish_ns + period_ns, now_ns + period_ns)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        callbacks = (
            (self.session.close, self.peer.close, self.xrce.close)
            if self._connected
            else (self.peer.close, self.xrce.close)
        )
        self._connected = False
        _close_preserving_primary(None, callbacks)

    def diagnostics(self) -> TransportDiagnostics:
        """Return the latest immutable network counters and timing sample."""

        return self.peer.diagnostics()

    def xrce_diagnostics(self) -> Px4XrceDiagnostics:
        """Return native PX4 publication health independently of ZMQ diagnostics."""

        return self.xrce.diagnostics()

    def _validate_leader_frame(self, frame: LeaderFrame) -> None:
        if frame.teleop_mode != self.session.teleop_mode:
            raise ProtocolError("leader and follower teleop modes do not match")
        if frame.mode == LeaderSyncMode.TRACKING:
            if self.session.teleop_mode == TeleopMode.JOINT and frame.arm_command is None:
                raise ProtocolError("tracking joint frame requires arm_command")
            if self.session.teleop_mode == TeleopMode.EE_POSE and frame.ee_pose_command is None:
                raise ProtocolError("tracking ee_pose frame requires ee_pose_command")
        if frame.arm_command is not None and frame.arm_command.names != self.session.arm_names:
            raise ProtocolError("arm command names do not match RobotSpec")
        for group_name, command in frame.end_effector_commands.items():
            expected = self.session.end_effector_names.get(group_name)
            if expected is None or command.names != expected:
                raise ProtocolError(
                    f"end-effector command '{group_name}' does not match RobotSpec"
                )


__all__ = ["FollowerApplication", "LeaderApplication"]
