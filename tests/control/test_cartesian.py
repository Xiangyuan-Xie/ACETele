from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acetele.control import CartesianTeleopController
from acetele.model import ArmKinematics


def _kinematics() -> ArmKinematics:
    path = (
        Path(__file__).resolve().parents[2]
        / "acetele/model/robots/ace_follower/description/ace_follower.urdf"
    )
    return ArmKinematics(
        path,
        ("joint_1", "joint_2", "joint_3", "joint_4"),
        "link_5",
    )


def test_first_cartesian_sample_anchors_without_a_joint_jump():
    kinematics = _kinematics()
    controller = CartesianTeleopController(kinematics)
    positions = np.array([0.0, 1.0, 0.0, 0.0])

    result = controller.solve(
        kinematics.forward(positions, timestamp_ns=1),
        positions,
        timestamp_ns=1,
    )

    assert result.positions == pytest.approx(positions)
    assert result.diagnostics.position_residual_m == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        result.positions[0] = 1.0


def test_relative_mapping_scales_translation_but_not_rotation():
    kinematics = _kinematics()
    controller = CartesianTeleopController(
        kinematics,
        translation_scale=2.0,
        rotation_scale=1.0,
    )
    positions = np.array([0.0, 1.0, 0.0, 0.0])
    anchor = kinematics.forward_matrix(positions)
    controller.solve(
        kinematics.pose_from_matrix(anchor, timestamp_ns=1),
        positions,
        timestamp_ns=1,
    )
    reachable = kinematics.forward_matrix(positions + [0.02, 0.02, 0.0, 0.0])
    expected_delta = np.linalg.inv(anchor) @ reachable
    source_delta = expected_delta.copy()
    source_delta[:3, 3] *= 0.5

    result = controller.solve(
        kinematics.pose_from_matrix(anchor @ source_delta, timestamp_ns=2),
        positions,
        timestamp_ns=2,
    )
    target = kinematics.pose_matrix(result.diagnostics.target_pose)
    mapped = np.linalg.inv(anchor) @ target

    assert mapped[:3, 3] == pytest.approx(expected_delta[:3, 3], abs=1e-9)
    assert kinematics.rotation_log(mapped[:3, :3]) == pytest.approx(
        kinematics.rotation_log(expected_delta[:3, :3]), abs=1e-9
    )


def test_cartesian_solver_rejects_an_unreachable_primary_task_without_progress():
    kinematics = _kinematics()
    controller = CartesianTeleopController(kinematics)
    positions = np.array([0.0, 1.0, 0.0, 0.0])
    anchor = kinematics.forward_matrix(positions)
    controller.solve(
        kinematics.pose_from_matrix(anchor, timestamp_ns=1),
        positions,
        timestamp_ns=1,
    )
    source = anchor.copy()
    source[:3, 3] += anchor[:3, :3] @ np.array([0.05, 0.0, 0.0])

    with pytest.raises(RuntimeError, match="no meaningful progress"):
        controller.solve(
            kinematics.pose_from_matrix(source, timestamp_ns=2),
            positions,
            timestamp_ns=2,
        )

    assert controller.diagnostics() is None


def test_real_four_dof_model_returns_reachable_projection_within_limits():
    kinematics = _kinematics()
    controller = CartesianTeleopController(kinematics)
    positions = np.array([0.0, 1.0, 0.0, 0.0])
    anchor = kinematics.forward_matrix(positions)
    controller.solve(
        kinematics.pose_from_matrix(anchor, timestamp_ns=1),
        positions,
        timestamp_ns=1,
    )
    target = anchor.copy()
    target[:3, 3] += [0.04, 0.03, 0.02]
    target[:3, :3] = kinematics.rotation_exp([0.3, -0.2, 0.4]) @ target[:3, :3]

    result = controller.solve(
        kinematics.pose_from_matrix(target, timestamp_ns=2),
        positions,
        timestamp_ns=2,
    )

    assert np.all(np.isfinite(result.positions))
    assert np.all(result.positions >= kinematics.lower_limits)
    assert np.all(result.positions <= kinematics.upper_limits)
    assert result.diagnostics.position_rank == 2
    assert result.diagnostics.full_rank == 4


def test_reset_requires_a_new_no_jump_anchor_and_frame_changes_are_rejected():
    kinematics = _kinematics()
    controller = CartesianTeleopController(kinematics)
    positions = np.array([0.0, 1.0, 0.0, 0.0])
    pose = kinematics.forward(positions, timestamp_ns=1)
    controller.solve(pose, positions, timestamp_ns=1)
    changed_frame = type(pose)(
        2,
        "vr_origin",
        pose.position_m,
        pose.quaternion_xyzw,
    )

    with pytest.raises(ValueError, match="frame changed"):
        controller.solve(changed_frame, positions, timestamp_ns=2)

    controller.reset()
    result = controller.solve(changed_frame, positions, timestamp_ns=3)
    assert result.positions == pytest.approx(positions)
