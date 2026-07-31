"""Source-neutral Cartesian mapping and position-priority inverse kinematics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from acetele.core import EndEffectorPose
from acetele.model import ArmKinematics


class TeleopMode(str, Enum):
    """Select the one arm command representation admitted by a teleoperation session."""

    JOINT = "joint"
    EE_POSE = "ee_pose"


@dataclass(frozen=True)
class CartesianTeleopTuning:
    """Conservative numerical policy for online Cartesian teleoperation."""

    maximum_iterations: int = 10
    damping: float = 1e-3
    maximum_joint_step_rad: float = 0.08
    position_tolerance_m: float = 0.002
    orientation_tolerance_rad: float = 0.03
    maximum_translation_m: float = 0.35
    maximum_rotation_rad: float = math.pi

    def __post_init__(self) -> None:
        if type(self.maximum_iterations) is not int or self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be a positive integer")
        for field_name in (
            "damping",
            "maximum_joint_step_rad",
            "position_tolerance_m",
            "orientation_tolerance_rad",
            "maximum_translation_m",
            "maximum_rotation_rad",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Cartesian teleop {field_name} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(
                    f"Cartesian teleop {field_name} must be finite and positive"
                )
            object.__setattr__(self, field_name, normalized)


@dataclass(frozen=True)
class CartesianTeleopDiagnostics:
    """Immutable diagnostics for the most recent mapped IK command."""

    target_pose: EndEffectorPose
    achieved_pose: EndEffectorPose
    position_residual_m: float
    orientation_residual_rad: float
    iterations: int
    position_rank: int
    full_rank: int
    singular: bool
    joint_limited: bool
    command_limited: bool


@dataclass(frozen=True)
class CartesianIKResult:
    """One detached joint solution and its diagnostic evidence."""

    positions: np.ndarray
    diagnostics: CartesianTeleopDiagnostics

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float).copy()
        if positions.ndim != 1 or not np.all(np.isfinite(positions)):
            raise ValueError("Cartesian IK positions must be a finite vector")
        positions.setflags(write=False)
        object.__setattr__(self, "positions", positions)


class CartesianTeleopController:
    """Map relative source-tool motion onto a follower and solve its reachable pose.

    Translation is the primary task. Orientation is optimized only in the remaining
    nullspace, allowing a four-DOF arm to follow the reachable projection of a full
    ``SE(3)`` command without sacrificing position to an impossible orientation target.
    """

    def __init__(
        self,
        kinematics: ArmKinematics,
        *,
        translation_scale: float = 2.0,
        rotation_scale: float = 1.0,
        tuning: CartesianTeleopTuning = CartesianTeleopTuning(),
    ) -> None:
        if not isinstance(kinematics, ArmKinematics):
            raise ValueError("Cartesian teleop requires ArmKinematics")
        for field_name, value in (
            ("translation_scale", translation_scale),
            ("rotation_scale", rotation_scale),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if not isinstance(tuning, CartesianTeleopTuning):
            raise ValueError("tuning must be CartesianTeleopTuning")
        self.kinematics = kinematics
        self.translation_scale = float(translation_scale)
        self.rotation_scale = float(rotation_scale)
        self.tuning = tuning
        self._source_anchor: Optional[np.ndarray] = None
        self._follower_anchor: Optional[np.ndarray] = None
        self._source_frame: Optional[str] = None
        self._diagnostics: Optional[CartesianTeleopDiagnostics] = None

    @property
    def anchored(self) -> bool:
        """Return whether this synchronization cycle has captured both initial poses."""

        return self._source_anchor is not None

    def reset(self) -> None:
        """Discard cycle-local anchors so stale relative motion cannot be replayed."""

        self._source_anchor = None
        self._follower_anchor = None
        self._source_frame = None
        self._diagnostics = None

    def diagnostics(self) -> Optional[CartesianTeleopDiagnostics]:
        """Return the immutable most recent diagnostic snapshot, if any."""

        return self._diagnostics

    def solve(
        self,
        source_pose: EndEffectorPose,
        current_positions: Sequence[float] | np.ndarray,
        *,
        timestamp_ns: int,
    ) -> CartesianIKResult:
        """Map one absolute source sample and solve a bounded follower joint target."""

        if not isinstance(source_pose, EndEffectorPose):
            raise ValueError("source_pose must be an EndEffectorPose")
        if type(timestamp_ns) is not int or timestamp_ns < 0:
            raise ValueError("Cartesian solve timestamp_ns must be non-negative")
        current = np.asarray(current_positions, dtype=float)
        expected = len(self.kinematics.joint_names)
        if current.shape != (expected,) or not np.all(np.isfinite(current)):
            raise ValueError(f"current_positions must contain {expected} finite values")
        source = self.kinematics.pose_matrix(source_pose)
        follower = self.kinematics.forward_matrix(current)
        if self._source_anchor is None:
            # The first command defines a relative-motion origin and intentionally maps
            # to the follower's current pose, preventing a mode-transition jump.
            self._source_anchor = source
            self._follower_anchor = follower
            self._source_frame = source_pose.frame_id
        elif source_pose.frame_id != self._source_frame:
            raise ValueError(
                "Cartesian source frame changed during tracking; resynchronize first"
            )
        target, command_limited = self._map_target(source)
        result = self._solve_ik(
            target,
            current.copy(),
            timestamp_ns=timestamp_ns,
            command_limited=command_limited,
        )
        self._diagnostics = result.diagnostics
        return result

    def _map_target(self, source: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply scaled source motion in anchor coordinates to the follower anchor."""

        if self._source_anchor is None or self._follower_anchor is None:
            raise RuntimeError("Cartesian teleop anchors are unavailable")
        source_anchor = self._source_anchor
        follower_anchor = self._follower_anchor
        relative_rotation = source_anchor[:3, :3].T @ source[:3, :3]
        relative_translation = source_anchor[:3, :3].T @ (
            source[:3, 3] - source_anchor[:3, 3]
        )
        scaled_translation = relative_translation * self.translation_scale
        scaled_rotation_vector = (
            self.kinematics.rotation_log(relative_rotation) * self.rotation_scale
        )
        limited = False
        translation_norm = float(np.linalg.norm(scaled_translation))
        if translation_norm > self.tuning.maximum_translation_m:
            scaled_translation *= self.tuning.maximum_translation_m / translation_norm
            limited = True
        rotation_norm = float(np.linalg.norm(scaled_rotation_vector))
        if rotation_norm > self.tuning.maximum_rotation_rad:
            scaled_rotation_vector *= self.tuning.maximum_rotation_rad / rotation_norm
            limited = True
        delta = np.eye(4, dtype=float)
        delta[:3, :3] = self.kinematics.rotation_exp(scaled_rotation_vector)
        delta[:3, 3] = scaled_translation
        return follower_anchor @ delta, limited

    def _solve_ik(
        self,
        target: np.ndarray,
        positions: np.ndarray,
        *,
        timestamp_ns: int,
        command_limited: bool,
    ) -> CartesianIKResult:
        """Solve primary translation and secondary orientation with bounded DLS steps."""

        lower = self.kinematics.lower_limits
        upper = self.kinematics.upper_limits
        best = positions.copy()
        best_position_error = math.inf
        best_orientation_error = math.inf
        iterations = 0
        joint_limited = False

        for iteration in range(self.tuning.maximum_iterations + 1):
            current = self.kinematics.forward_matrix(positions)
            position_error = target[:3, 3] - current[:3, 3]
            orientation_error = self.kinematics.rotation_log(
                target[:3, :3] @ current[:3, :3].T
            )
            position_norm = float(np.linalg.norm(position_error))
            orientation_norm = float(np.linalg.norm(orientation_error))
            if self._better_solution(
                position_norm,
                orientation_norm,
                best_position_error,
                best_orientation_error,
            ):
                best = positions.copy()
                best_position_error = position_norm
                best_orientation_error = orientation_norm
            iterations = iteration
            if (
                position_norm <= self.tuning.position_tolerance_m
                and orientation_norm <= self.tuning.orientation_tolerance_rad
            ) or iteration == self.tuning.maximum_iterations:
                break

            linear, angular = self.kinematics.jacobian(positions)
            linear_inverse = self._damped_inverse(linear)
            primary_step = linear_inverse @ position_error
            nullspace = np.eye(len(positions)) - linear_inverse @ linear
            secondary = angular @ nullspace
            secondary_step = nullspace @ self._damped_inverse(secondary) @ (
                orientation_error - angular @ primary_step
            )
            step = primary_step + secondary_step
            if not np.all(np.isfinite(step)):
                raise RuntimeError("Cartesian IK produced a non-finite joint step")
            maximum = float(np.max(np.abs(step)))
            if maximum > self.tuning.maximum_joint_step_rad:
                step *= self.tuning.maximum_joint_step_rad / maximum
            candidate = positions + step
            clipped = np.clip(candidate, lower, upper)
            joint_limited = joint_limited or not np.array_equal(candidate, clipped)
            if np.max(np.abs(clipped - positions)) <= np.finfo(float).eps:
                break
            positions = clipped

        achieved_matrix = self.kinematics.forward_matrix(best)
        linear, angular = self.kinematics.jacobian(best)
        combined = np.vstack((linear, angular))
        nonzero_singular_values = np.linalg.svd(combined, compute_uv=False)
        nonzero_singular_values = nonzero_singular_values[
            nonzero_singular_values > self.tuning.damping
        ]
        singular = (
            len(nonzero_singular_values) < min(combined.shape)
            or (
                len(nonzero_singular_values) > 1
                and nonzero_singular_values[0] / nonzero_singular_values[-1] > 1e4
            )
        )
        target_pose = self.kinematics.pose_from_matrix(
            target,
            timestamp_ns=timestamp_ns,
        )
        achieved_pose = self.kinematics.pose_from_matrix(
            achieved_matrix,
            timestamp_ns=timestamp_ns,
        )
        diagnostics = CartesianTeleopDiagnostics(
            target_pose=target_pose,
            achieved_pose=achieved_pose,
            position_residual_m=best_position_error,
            orientation_residual_rad=best_orientation_error,
            iterations=iterations,
            position_rank=int(np.linalg.matrix_rank(linear)),
            full_rank=int(np.linalg.matrix_rank(combined)),
            singular=bool(singular),
            joint_limited=joint_limited,
            command_limited=command_limited,
        )
        return CartesianIKResult(best, diagnostics)

    def _damped_inverse(self, jacobian: np.ndarray) -> np.ndarray:
        """Compute a stable right pseudoinverse without branching on task rank."""

        task_matrix = jacobian @ jacobian.T
        regularized = task_matrix + np.eye(task_matrix.shape[0]) * self.tuning.damping**2
        return jacobian.T @ np.linalg.solve(regularized, np.eye(task_matrix.shape[0]))

    def _better_solution(
        self,
        position_error: float,
        orientation_error: float,
        best_position_error: float,
        best_orientation_error: float,
    ) -> bool:
        """Compare candidates lexicographically so orientation never trades away position."""

        if position_error < best_position_error - 1e-12:
            return True
        return (
            abs(position_error - best_position_error) <= self.tuning.position_tolerance_m
            and orientation_error < best_orientation_error
        )


__all__ = [
    "CartesianIKResult",
    "CartesianTeleopController",
    "CartesianTeleopDiagnostics",
    "CartesianTeleopTuning",
    "TeleopMode",
]
