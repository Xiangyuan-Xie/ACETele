"""Pinocchio dynamics and full spatial Jacobian for one configured arm."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from acetele.model.urdf import build_reduced_pinocchio_model, load_urdf_model


class ArmDynamics:
    """Validated reduced model used by local joint-effort assistance.

    The model preserves the configured joint order and evaluates the TCP Jacobian in
    ``LOCAL_WORLD_ALIGNED`` coordinates. Keeping all six spatial rows is important:
    posture torque may use only the true null space of the complete TCP pose task.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        joint_names: Sequence[str],
        tool_frame: str,
    ) -> None:
        import pinocchio as pin

        names = tuple(joint_names)
        urdf = load_urdf_model(urdf_path)
        metadata = urdf.arm_metadata(names, require_limits=True)
        urdf.require_frame(tool_frame)
        model = build_reduced_pinocchio_model(urdf.path, names)
        if not model.existFrame(tool_frame):
            raise ValueError(
                f"reduced Pinocchio model is missing configured tool frame '{tool_frame}'"
            )
        model_names = tuple(str(name) for name in model.names[1:])
        if model_names != names:
            raise ValueError(
                "reduced Pinocchio joint order does not match configured arm order; "
                f"expected {names}, got {model_names}"
            )
        if model.nq != len(names) or model.nv != len(names):
            raise ValueError("reduced Pinocchio dynamics model has an invalid DOF count")
        for name in names:
            inertia = model.inertias[model.getJointId(name)]
            if (
                not math.isfinite(float(inertia.mass))
                or inertia.mass <= 0.0
                or not np.all(np.isfinite(np.asarray(inertia.lever)))
                or not np.all(np.isfinite(np.asarray(inertia.inertia)))
            ):
                raise ValueError(
                    f"Pinocchio joint '{name}' requires finite positive inertial data"
                )

        self._pin = pin
        self._model = model
        self._data = model.createData()
        self._frame_id = model.getFrameId(tool_frame)
        self._joint_names = names
        self._lower = self._readonly(metadata.lower_limits)
        self._upper = self._readonly(metadata.upper_limits)
        self._effort = self._readonly(metadata.effort_limits)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return the exact vector order expected by all dynamics operations."""

        return self._joint_names

    @property
    def lower_limits(self) -> np.ndarray:
        """Return an independent read-only lower-limit vector."""

        return self._readonly(self._lower)

    @property
    def upper_limits(self) -> np.ndarray:
        """Return an independent read-only upper-limit vector."""

        return self._readonly(self._upper)

    @property
    def effort_limits(self) -> np.ndarray:
        """Return an independent read-only URDF effort-limit vector."""

        return self._readonly(self._effort)

    def inverse_dynamics(
        self,
        positions: Sequence[float] | np.ndarray,
        velocities: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Return RNEA torque for zero acceleration in newton-metres."""

        q = self._vector(positions, "positions")
        velocity = self._vector(velocities, "velocities")
        torque = np.asarray(
            self._pin.rnea(
                self._model,
                self._data,
                q,
                velocity,
                np.zeros(len(self._joint_names)),
            ),
            dtype=float,
        ).copy()
        if torque.shape != q.shape or not np.all(np.isfinite(torque)):
            raise RuntimeError("Pinocchio returned non-finite inverse dynamics")
        return torque

    def full_jacobian(
        self,
        positions: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Return the complete world-aligned 6-by-N TCP Jacobian."""

        q = self._vector(positions, "positions")
        jacobian = np.asarray(
            self._pin.computeFrameJacobian(
                self._model,
                self._data,
                q,
                self._frame_id,
                self._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            ),
            dtype=float,
        ).copy()
        expected = (6, len(self._joint_names))
        if jacobian.shape != expected or not np.all(np.isfinite(jacobian)):
            raise RuntimeError("Pinocchio returned an invalid TCP Jacobian")
        return jacobian

    def _vector(
        self,
        values: Sequence[float] | np.ndarray,
        field_name: str,
    ) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (len(self._joint_names),) or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"dynamics {field_name} must contain "
                f"{len(self._joint_names)} finite values"
            )
        return vector.copy()

    @staticmethod
    def _readonly(values: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float).copy()
        array.setflags(write=False)
        return array


__all__ = ["ArmDynamics"]
