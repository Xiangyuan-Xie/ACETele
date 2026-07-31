"""Validated forward kinematics and Jacobians for one configured arm."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from acetele.core import EndEffectorPose
from acetele.model.urdf import build_reduced_pinocchio_model, load_urdf_model


class ArmKinematics:
    """Pinocchio-backed kinematics with vendor-neutral NumPy inputs and outputs."""

    def __init__(
        self,
        urdf_path: str | Path,
        joint_names: Sequence[str],
        tool_frame: str,
    ) -> None:
        import pinocchio as pin

        urdf = load_urdf_model(urdf_path)
        names = tuple(joint_names)
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
        self._pin = pin
        self._model = model
        self._data = model.createData()
        self._frame_id = model.getFrameId(tool_frame)
        self._joint_names = names
        self._tool_frame = tool_frame
        self._root_frame = urdf.root_link
        self._lower_limits = self._readonly(metadata.lower_limits)
        self._upper_limits = self._readonly(metadata.upper_limits)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return the exact vector order expected by every kinematics operation."""

        return self._joint_names

    @property
    def tool_frame(self) -> str:
        """Return the controlled URDF link frame."""

        return self._tool_frame

    @property
    def root_frame(self) -> str:
        """Return the reference frame in which poses and Jacobians are expressed."""

        return self._root_frame

    @property
    def lower_limits(self) -> np.ndarray:
        """Return an independent read-only lower-limit vector."""

        return self._readonly(self._lower_limits)

    @property
    def upper_limits(self) -> np.ndarray:
        """Return an independent read-only upper-limit vector."""

        return self._readonly(self._upper_limits)

    def forward(
        self,
        positions: Sequence[float] | np.ndarray,
        *,
        timestamp_ns: int = 0,
    ) -> EndEffectorPose:
        """Return the tool pose for one finite joint vector."""

        transform = self.forward_matrix(positions)
        quaternion = self._pin.Quaternion(transform[:3, :3]).coeffs()
        return EndEffectorPose(
            timestamp_ns,
            self._root_frame,
            transform[:3, 3],
            quaternion,
        )

    def forward_matrix(
        self,
        positions: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Return a detached homogeneous root-to-tool transform."""

        q = self._positions(positions)
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacements(self._model, self._data)
        placement = self._data.oMf[self._frame_id]
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = placement.rotation
        transform[:3, 3] = placement.translation
        return transform

    def jacobian(
        self,
        positions: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return world-aligned linear and angular tool Jacobians."""

        q = self._positions(positions)
        spatial = self._pin.computeFrameJacobian(
            self._model,
            self._data,
            q,
            self._frame_id,
            self._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        # Pinocchio stores linear rows first and angular rows second. Returning the two
        # tasks separately makes the position-priority controller's hierarchy explicit.
        return np.asarray(spatial[:3], dtype=float).copy(), np.asarray(
            spatial[3:], dtype=float
        ).copy()

    def rotation_log(self, rotation: np.ndarray) -> np.ndarray:
        """Map a finite rotation matrix to its three-dimensional tangent vector."""

        matrix = np.asarray(rotation, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("rotation must be a finite 3x3 matrix")
        return np.asarray(self._pin.log3(matrix), dtype=float).copy()

    def rotation_exp(self, tangent: Sequence[float] | np.ndarray) -> np.ndarray:
        """Map a finite three-dimensional tangent vector to a rotation matrix."""

        vector = np.asarray(tangent, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("rotation tangent must be a finite vector of length three")
        return np.asarray(self._pin.exp3(vector), dtype=float).copy()

    def pose_matrix(self, pose: EndEffectorPose) -> np.ndarray:
        """Convert a core pose into a homogeneous transform without retaining aliases."""

        if not isinstance(pose, EndEffectorPose):
            raise ValueError("pose must be an EndEffectorPose")
        quaternion = self._pin.Quaternion(np.asarray(pose.quaternion_xyzw))
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = quaternion.matrix()
        transform[:3, 3] = pose.position_m
        return transform

    def pose_from_matrix(
        self,
        transform: np.ndarray,
        *,
        timestamp_ns: int,
    ) -> EndEffectorPose:
        """Convert a finite homogeneous transform into the core pose contract."""

        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("pose transform must be a finite 4x4 matrix")
        quaternion = self._pin.Quaternion(matrix[:3, :3]).coeffs()
        return EndEffectorPose(
            timestamp_ns,
            self._root_frame,
            matrix[:3, 3],
            quaternion,
        )

    def _positions(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        positions = np.asarray(values, dtype=float)
        if positions.shape != (len(self._joint_names),) or not np.all(
            np.isfinite(positions)
        ):
            raise ValueError(
                f"joint positions must contain {len(self._joint_names)} finite values"
            )
        return positions.copy()

    @staticmethod
    def _readonly(values: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float).copy()
        array.setflags(write=False)
        return array


__all__ = ["ArmKinematics"]
