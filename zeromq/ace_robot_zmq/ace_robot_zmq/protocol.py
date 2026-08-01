"""Versioned, bounded MessagePack protocol for direct teleoperation peers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Optional

import msgpack
import numpy as np

from acetele.core import EndEffectorPose, JointState, JointUnit
from acetele.runtime.teleop import FollowerSyncStatus, LeaderSyncMode, TeleopMode


class ProtocolError(ValueError):
    """Raised when an untrusted wire frame violates the declared schema."""


def _reject_extension(_code: int, _data: bytes) -> Any:
    """Reject MessagePack extension values instead of constructing custom objects."""

    raise ProtocolError("MessagePack extension values are not supported")


def _integer(value: Any, *, name: str, maximum: int = (1 << 64) - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ProtocolError(f"{name} must be an unsigned integer")
    return value


def _session_id(value: Any) -> bytes:
    if not isinstance(value, bytes) or len(value) != 16:
        raise ProtocolError("session_id must contain exactly 16 bytes")
    return value


def _names(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ProtocolError(f"{name} must be a non-empty array")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ProtocolError(f"{name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ProtocolError(f"{name} must be unique")
    return result


def _vector(value: Any, *, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ProtocolError(f"{name} must contain {length} values")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise ProtocolError(f"{name} must contain real numbers")
    result = tuple(float(item) for item in value)
    if not np.all(np.isfinite(result)):
        raise ProtocolError(f"{name} must contain only finite values")
    return result


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a map")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ProtocolError(
            f"{name} fields differ from schema: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class JointTarget:
    """Finite ordered positions for one arm or end-effector group."""

    names: tuple[str, ...]
    positions: tuple[float, ...]

    def __post_init__(self) -> None:
        names = _names(self.names, name="joint target names")
        positions = _vector(
            self.positions,
            name="joint target positions",
            length=len(names),
        )
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True)
class LeaderFrame:
    """One coherent leader synchronization and latest-command snapshot."""

    session_id: bytes
    sequence: int
    sent_at_ns: int
    mode: LeaderSyncMode
    teleop_mode: TeleopMode
    arm_command: Optional[JointTarget] = None
    ee_pose_command: Optional[EndEffectorPose] = None
    end_effector_commands: Mapping[str, JointTarget] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _session_id(self.session_id))
        _integer(self.sequence, name="leader sequence")
        _integer(self.sent_at_ns, name="leader sent_at_ns")
        if not isinstance(self.mode, LeaderSyncMode):
            raise ProtocolError("leader mode must be a LeaderSyncMode")
        if not isinstance(self.teleop_mode, TeleopMode):
            raise ProtocolError("leader teleop_mode must be a TeleopMode")
        if self.arm_command is not None and not isinstance(self.arm_command, JointTarget):
            raise ProtocolError("arm_command must be a JointTarget or None")
        if self.ee_pose_command is not None and not isinstance(
            self.ee_pose_command, EndEffectorPose
        ):
            raise ProtocolError("ee_pose_command must be an EndEffectorPose or None")
        if self.arm_command is not None and self.ee_pose_command is not None:
            raise ProtocolError("leader frame cannot contain joint and pose commands together")
        if self.teleop_mode == TeleopMode.JOINT and self.ee_pose_command is not None:
            raise ProtocolError("joint teleop frame cannot contain an ee pose command")
        if self.teleop_mode == TeleopMode.EE_POSE and self.arm_command is not None:
            raise ProtocolError("ee_pose frame cannot contain an arm joint command")
        commands = dict(self.end_effector_commands)
        if any(not isinstance(name, str) or not name.strip() for name in commands):
            raise ProtocolError("end-effector group names must be non-empty strings")
        if any(not isinstance(command, JointTarget) for command in commands.values()):
            raise ProtocolError("end-effector commands must contain JointTarget values")
        object.__setattr__(self, "end_effector_commands", MappingProxyType(commands))


@dataclass(frozen=True)
class FollowerFrame:
    """One coherent follower status and measured joint-state snapshot."""

    session_id: bytes
    sequence: int
    sent_at_ns: int
    status: FollowerSyncStatus
    joint_states: Mapping[str, JointState]
    ee_pose_state: Optional[EndEffectorPose] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _session_id(self.session_id))
        _integer(self.sequence, name="follower sequence")
        _integer(self.sent_at_ns, name="follower sent_at_ns")
        if not isinstance(self.status, FollowerSyncStatus):
            raise ProtocolError("follower status must be a FollowerSyncStatus")
        states = dict(self.joint_states)
        if not states or any(
            not isinstance(name, str) or not name.strip() for name in states
        ):
            raise ProtocolError("follower joint-state groups must be non-empty names")
        if any(not isinstance(state, JointState) for state in states.values()):
            raise ProtocolError("follower joint_states must contain JointState values")
        object.__setattr__(self, "joint_states", MappingProxyType(states))
        if self.ee_pose_state is not None and not isinstance(
            self.ee_pose_state, EndEffectorPose
        ):
            raise ProtocolError("ee_pose_state must be an EndEffectorPose or None")


class MessagePackCodec:
    """Encode and strictly validate the single-frame protocol without object hooks."""

    def __init__(self, *, maximum_frame_bytes: int = 65_536, version: int = 1) -> None:
        if type(maximum_frame_bytes) is not int or not 1024 <= maximum_frame_bytes <= 65_536:
            raise ValueError("maximum_frame_bytes must be an integer in [1024, 65536]")
        if type(version) is not int or version <= 0:
            raise ValueError("protocol version must be a positive integer")
        self.maximum_frame_bytes = maximum_frame_bytes
        self.version = version

    def encode_leader(self, frame: LeaderFrame) -> bytes:
        if not isinstance(frame, LeaderFrame):
            raise ValueError("encode_leader requires a LeaderFrame")
        return self._pack(
            {
                "version": self.version,
                "session_id": frame.session_id,
                "sequence": frame.sequence,
                "sent_at_ns": frame.sent_at_ns,
                "mode": frame.mode.value,
                "teleop_mode": frame.teleop_mode.value,
                "arm_command": self._target_document(frame.arm_command),
                "ee_pose_command": self._pose_document(frame.ee_pose_command),
                "end_effector_commands": {
                    name: self._target_document(command)
                    for name, command in frame.end_effector_commands.items()
                },
            }
        )

    def decode_leader(self, payload: bytes) -> LeaderFrame:
        document = self._unpack(payload)
        expected = {
            "version",
            "session_id",
            "sequence",
            "sent_at_ns",
            "mode",
            "teleop_mode",
            "arm_command",
            "ee_pose_command",
            "end_effector_commands",
        }
        _exact_keys(document, expected, name="leader frame")
        self._version(document["version"])
        commands_document = _mapping(
            document["end_effector_commands"],
            name="end_effector_commands",
        )
        commands: dict[str, JointTarget] = {}
        for name, value in commands_document.items():
            command = self._target_from_document(
                value,
                name=f"end_effector_commands.{name}",
            )
            if command is None:
                raise ProtocolError("end-effector command values cannot be null")
            commands[name] = command
        try:
            mode = LeaderSyncMode(document["mode"])
            teleop_mode = TeleopMode(document["teleop_mode"])
        except (TypeError, ValueError) as exc:
            raise ProtocolError("leader frame contains an unknown mode") from exc
        return LeaderFrame(
            _session_id(document["session_id"]),
            _integer(document["sequence"], name="leader sequence"),
            _integer(document["sent_at_ns"], name="leader sent_at_ns"),
            mode,
            teleop_mode,
            self._target_from_document(document["arm_command"], name="arm_command"),
            self._pose_from_document(document["ee_pose_command"], name="ee_pose_command"),
            commands,
        )

    def encode_follower(self, frame: FollowerFrame) -> bytes:
        if not isinstance(frame, FollowerFrame):
            raise ValueError("encode_follower requires a FollowerFrame")
        return self._pack(
            {
                "version": self.version,
                "session_id": frame.session_id,
                "sequence": frame.sequence,
                "sent_at_ns": frame.sent_at_ns,
                "status": frame.status.value,
                "joint_states": {
                    name: self._state_document(state)
                    for name, state in frame.joint_states.items()
                },
                "ee_pose_state": self._pose_document(frame.ee_pose_state),
            }
        )

    def decode_follower(self, payload: bytes) -> FollowerFrame:
        document = self._unpack(payload)
        expected = {
            "version",
            "session_id",
            "sequence",
            "sent_at_ns",
            "status",
            "joint_states",
            "ee_pose_state",
        }
        _exact_keys(document, expected, name="follower frame")
        self._version(document["version"])
        states_document = _mapping(document["joint_states"], name="joint_states")
        states = {
            name: self._state_from_document(value, name=f"joint_states.{name}")
            for name, value in states_document.items()
        }
        try:
            status = FollowerSyncStatus(document["status"])
        except (TypeError, ValueError) as exc:
            raise ProtocolError("follower frame contains an unknown status") from exc
        return FollowerFrame(
            _session_id(document["session_id"]),
            _integer(document["sequence"], name="follower sequence"),
            _integer(document["sent_at_ns"], name="follower sent_at_ns"),
            status,
            states,
            self._pose_from_document(document["ee_pose_state"], name="ee_pose_state"),
        )

    def _pack(self, document: Mapping[str, Any]) -> bytes:
        payload = msgpack.packb(document, use_bin_type=True)
        if len(payload) > self.maximum_frame_bytes:
            raise ProtocolError("encoded frame exceeds maximum_frame_bytes")
        return payload

    def _unpack(self, payload: bytes) -> Mapping[str, Any]:
        if not isinstance(payload, bytes) or not payload:
            raise ProtocolError("wire frame must be non-empty bytes")
        if len(payload) > self.maximum_frame_bytes:
            raise ProtocolError("wire frame exceeds maximum_frame_bytes")
        try:
            document = msgpack.unpackb(
                payload,
                raw=False,
                strict_map_key=True,
                ext_hook=_reject_extension,
                max_str_len=self.maximum_frame_bytes,
                max_bin_len=self.maximum_frame_bytes,
                max_array_len=256,
                max_map_len=128,
                max_ext_len=0,
            )
        except (ValueError, TypeError, msgpack.ExtraData, msgpack.FormatError, msgpack.StackError) as exc:
            raise ProtocolError("invalid MessagePack frame") from exc
        return _mapping(document, name="wire frame")

    def _version(self, value: Any) -> None:
        if _integer(value, name="protocol version") != self.version:
            raise ProtocolError(f"unsupported protocol version {value}")

    @staticmethod
    def _target_document(target: Optional[JointTarget]) -> Optional[dict[str, Any]]:
        if target is None:
            return None
        return {"names": list(target.names), "positions": list(target.positions)}

    @staticmethod
    def _target_from_document(value: Any, *, name: str) -> Optional[JointTarget]:
        if value is None:
            return None
        document = _mapping(value, name=name)
        _exact_keys(document, {"names", "positions"}, name=name)
        names = _names(document["names"], name=f"{name}.names")
        return JointTarget(
            names,
            _vector(document["positions"], name=f"{name}.positions", length=len(names)),
        )

    @staticmethod
    def _pose_document(pose: Optional[EndEffectorPose]) -> Optional[dict[str, Any]]:
        if pose is None:
            return None
        return {
            "timestamp_ns": pose.timestamp_ns,
            "frame_id": pose.frame_id,
            "position_m": pose.position_m.tolist(),
            "quaternion_xyzw": pose.quaternion_xyzw.tolist(),
        }

    @staticmethod
    def _pose_from_document(value: Any, *, name: str) -> Optional[EndEffectorPose]:
        if value is None:
            return None
        document = _mapping(value, name=name)
        expected = {"timestamp_ns", "frame_id", "position_m", "quaternion_xyzw"}
        _exact_keys(document, expected, name=name)
        if not isinstance(document["frame_id"], str):
            raise ProtocolError(f"{name}.frame_id must be a string")
        try:
            return EndEffectorPose(
                _integer(document["timestamp_ns"], name=f"{name}.timestamp_ns"),
                document["frame_id"],
                _vector(document["position_m"], name=f"{name}.position_m", length=3),
                _vector(
                    document["quaternion_xyzw"],
                    name=f"{name}.quaternion_xyzw",
                    length=4,
                ),
            )
        except ValueError as exc:
            raise ProtocolError(str(exc)) from exc

    @staticmethod
    def _state_document(state: JointState) -> dict[str, Any]:
        return {
            "names": list(state.names),
            "positions": state.positions.tolist(),
            "velocities": state.velocities.tolist(),
            "efforts": state.efforts.tolist(),
            "timestamp_ns": state.timestamp_ns,
            "sequence": state.sequence,
            "unit": state.unit.value,
        }

    @staticmethod
    def _state_from_document(value: Any, *, name: str) -> JointState:
        document = _mapping(value, name=name)
        expected = {
            "names",
            "positions",
            "velocities",
            "efforts",
            "timestamp_ns",
            "sequence",
            "unit",
        }
        _exact_keys(document, expected, name=name)
        names = _names(document["names"], name=f"{name}.names")
        try:
            unit = JointUnit(document["unit"])
            return JointState(
                names,
                _vector(document["positions"], name=f"{name}.positions", length=len(names)),
                _vector(document["velocities"], name=f"{name}.velocities", length=len(names)),
                _vector(document["efforts"], name=f"{name}.efforts", length=len(names)),
                _integer(document["timestamp_ns"], name=f"{name}.timestamp_ns"),
                _integer(document["sequence"], name=f"{name}.sequence"),
                unit,
            )
        except ValueError as exc:
            raise ProtocolError(str(exc)) from exc


__all__ = [
    "FollowerFrame",
    "JointTarget",
    "LeaderFrame",
    "MessagePackCodec",
    "ProtocolError",
]
