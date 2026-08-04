"""FACTR-style gravity and redundant-posture joint-effort assistance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from acetele.core import JointState, JointUnit
from acetele.model.dynamics import ArmDynamics


def _readonly(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class EffortControlTuning:
    """Conservative internal gains shared by every effort-assisted Leader arm."""

    # FACTR's published hardware configuration scales model-based inverse dynamics
    # to 85% before combining it with the remaining assistance terms.
    gravity_gain: float = 0.85
    rest_stiffness_nm_per_rad: float = 0.15
    rest_damping_nm_s_per_rad: float = 0.03
    svd_relative_threshold: float = 1e-4
    joint_limit_margin_rad: float = 0.15
    joint_limit_stiffness_nm_per_rad: float = 0.4
    maximum_auxiliary_fraction: float = 0.35
    effort_rate_limit_nm_s: float = 5.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_auxiliary_fraction > 1.0:
            raise ValueError("maximum_auxiliary_fraction must not exceed one")


@dataclass(frozen=True)
class EffortControlDiagnostics:
    """Detached terms and constraints from one local effort-control update."""

    gravity_nm: np.ndarray
    rest_nm: np.ndarray
    limit_nm: np.ndarray
    total_nm: np.ndarray
    jacobian_rank: int
    nullity: int
    effort_limited: np.ndarray
    rate_limited: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "gravity_nm",
            "rest_nm",
            "limit_nm",
            "total_nm",
            "effort_limited",
            "rate_limited",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True)
class EffortControlResult:
    """One finite effort target and its matching diagnostic snapshot."""

    efforts_nm: np.ndarray
    diagnostics: EffortControlDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "efforts_nm", _readonly(self.efforts_nm))


class LeaderEffortController:
    """Compute gravity torque plus a full-pose null-space posture objective."""

    def __init__(
        self,
        dynamics: ArmDynamics,
        *,
        gravity_compensation: bool,
        redundancy_posture: bool,
        rest_posture_rad: Optional[Sequence[float]],
        effort_limits_nm: Optional[Sequence[float]] = None,
        tuning: EffortControlTuning = EffortControlTuning(),
    ) -> None:
        if not isinstance(dynamics, ArmDynamics):
            raise ValueError("effort controller requires ArmDynamics")
        if type(gravity_compensation) is not bool or type(redundancy_posture) is not bool:
            raise ValueError("effort control switches must be booleans")
        count = len(dynamics.joint_names)
        rest = np.zeros(count) if rest_posture_rad is None else np.asarray(
            rest_posture_rad, dtype=float
        )
        if rest.shape != (count,) or not np.all(np.isfinite(rest)):
            raise ValueError("rest posture must match the finite arm joint vector")
        if np.any(rest < dynamics.lower_limits) or np.any(rest > dynamics.upper_limits):
            raise ValueError("rest posture exceeds URDF joint limits")
        self._dynamics = dynamics
        self._gravity_enabled = gravity_compensation
        self._redundancy_enabled = redundancy_posture
        self._rest = rest.copy()
        limits = (
            dynamics.effort_limits
            if effort_limits_nm is None
            else np.asarray(effort_limits_nm, dtype=float)
        )
        if (
            limits.shape != (count,)
            or not np.all(np.isfinite(limits))
            or np.any(limits <= 0.0)
            or np.any(limits > dynamics.effort_limits)
        ):
            raise ValueError(
                "effort limits must be finite, positive, and no greater than URDF limits"
            )
        self._effort_limits = limits.copy()
        self._tuning = tuning
        self._last_update_ns: Optional[int] = None
        self._last_effort: Optional[np.ndarray] = None
        self._diagnostics = self._empty_diagnostics(count)

    def reset(self) -> None:
        """Forget slew history after a mode or safety discontinuity."""

        self._last_update_ns = None
        self._last_effort = None

    def compute(self, state: JointState, *, now_ns: int) -> EffortControlResult:
        """Evaluate one bounded effort command from measured local arm state."""

        if state.names != self._dynamics.joint_names or state.unit != JointUnit.RADIAN:
            raise ValueError("effort-control state does not match its arm model")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("effort-control time must be a non-negative integer")

        q = np.asarray(state.positions, dtype=float)
        velocity = np.asarray(state.velocities, dtype=float)
        gravity = (
            self._tuning.gravity_gain * self._dynamics.inverse_dynamics(q, velocity)
            if self._gravity_enabled
            else np.zeros_like(q)
        )
        jacobian = self._dynamics.full_jacobian(q)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        threshold = (
            0.0
            if singular_values.size == 0
            else self._tuning.svd_relative_threshold * singular_values[0]
        )
        rank = int(np.count_nonzero(singular_values > threshold))
        nullity = len(q) - rank

        # A column-full-rank Jacobian has no physical null space. Return exact zeros
        # instead of allowing SVD roundoff to make a low-DOF Leader feel artificially
        # stiff. Redundant arms use the Moore-Penrose projector from the full 6D task.
        rest = np.zeros_like(q)
        if self._redundancy_enabled and nullity > 0:
            raw_rest = (
                -self._tuning.rest_stiffness_nm_per_rad * (q - self._rest)
                - self._tuning.rest_damping_nm_s_per_rad * velocity
            )
            projector = np.eye(len(q)) - np.linalg.pinv(
                jacobian,
                rcond=self._tuning.svd_relative_threshold,
            ) @ jacobian
            rest = projector @ raw_rest

        limit = self._joint_limit_torque(q)
        effort_limits = self._effort_limits
        auxiliary_limit = effort_limits * self._tuning.maximum_auxiliary_fraction
        auxiliary = rest + limit
        bounded_auxiliary = np.clip(auxiliary, -auxiliary_limit, auxiliary_limit)
        effort_limited = ~np.isclose(auxiliary, bounded_auxiliary)
        desired = np.clip(gravity + bounded_auxiliary, -effort_limits, effort_limits)
        effort_limited |= ~np.isclose(gravity + bounded_auxiliary, desired)

        rate_limited = np.zeros(len(q), dtype=bool)
        if self._last_effort is not None and self._last_update_ns is not None:
            dt = float(np.clip((now_ns - self._last_update_ns) / 1e9, 0.001, 0.05))
            maximum_change = self._tuning.effort_rate_limit_nm_s * dt
            output = np.clip(
                desired,
                self._last_effort - maximum_change,
                self._last_effort + maximum_change,
            )
            rate_limited = ~np.isclose(output, desired)
        else:
            # The first command must support gravity immediately; ramp only subsequent
            # changes, which are predominantly operator motion and auxiliary posture.
            output = desired
        if not np.all(np.isfinite(output)):
            raise RuntimeError("effort controller produced a non-finite command")

        self._last_effort = output.copy()
        self._last_update_ns = now_ns
        self._diagnostics = EffortControlDiagnostics(
            gravity,
            rest,
            limit,
            output,
            rank,
            nullity,
            effort_limited,
            rate_limited,
        )
        return EffortControlResult(output, self._diagnostics)

    def diagnostics(self) -> EffortControlDiagnostics:
        """Return an independent snapshot of the last control update."""

        value = self._diagnostics
        return EffortControlDiagnostics(
            value.gravity_nm,
            value.rest_nm,
            value.limit_nm,
            value.total_nm,
            value.jacobian_rank,
            value.nullity,
            value.effort_limited,
            value.rate_limited,
        )

    def _joint_limit_torque(self, positions: np.ndarray) -> np.ndarray:
        """Apply a continuous inward spring only inside each URDF limit margin."""

        lower_distance = positions - self._dynamics.lower_limits
        upper_distance = self._dynamics.upper_limits - positions
        margin = self._tuning.joint_limit_margin_rad
        torque = np.where(
            lower_distance < margin,
            self._tuning.joint_limit_stiffness_nm_per_rad * (margin - lower_distance),
            0.0,
        )
        torque -= np.where(
            upper_distance < margin,
            self._tuning.joint_limit_stiffness_nm_per_rad * (margin - upper_distance),
            0.0,
        )
        return torque

    @staticmethod
    def _empty_diagnostics(count: int) -> EffortControlDiagnostics:
        zeros = np.zeros(count)
        return EffortControlDiagnostics(
            zeros,
            zeros,
            zeros,
            zeros,
            0,
            count,
            np.zeros(count, dtype=bool),
            np.zeros(count, dtype=bool),
        )


__all__ = [
    "EffortControlDiagnostics",
    "EffortControlResult",
    "EffortControlTuning",
    "LeaderEffortController",
]
