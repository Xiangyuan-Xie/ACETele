from __future__ import annotations

import numpy as np
import pytest

from acetele.hardware.state_estimator import (
    RobustJointStateEstimator,
    StateEstimatorTuning,
)


def _estimator() -> RobustJointStateEstimator:
    count = np.pi / 2048.0
    return RobustJointStateEstimator(
        1,
        StateEstimatorTuning(
            acceleration_std_rad_s2=6.0,
            position_std_rad=4.0 * count,
            velocity_std_rad_s=0.9,
            position_gate_rad=8.0 * count,
            reanchor_gate_rad=4.0 * count,
            velocity_consistency_rad_s=0.5,
        ),
    )


def test_position_and_velocity_spikes_are_rejected_without_state_pulse():
    estimator = _estimator()
    estimator.update([0.0], [0.0], timestamp_s=0.0, sample_id=0)
    result = estimator.update([0.05], [1.5], timestamp_s=0.01, sample_id=1)

    assert not result.position_accepted[0]
    assert not result.velocity_accepted[0]
    assert abs(result.positions[0]) < 0.01
    assert abs(result.velocities[0]) <= 0.4


def test_duplicate_sample_returns_same_readonly_snapshot():
    estimator = _estimator()
    first = estimator.update([0.1], [0.0], timestamp_s=1.0, sample_id=1)
    duplicate = estimator.update([0.2], [2.0], timestamp_s=1.1, sample_id=1)

    np.testing.assert_array_equal(duplicate.positions, first.positions)
    with pytest.raises(ValueError):
        duplicate.positions[0] = 9.0


def test_three_consistent_rejections_reanchor_real_state_change():
    estimator = _estimator()
    estimator.update([0.0], [0.0], timestamp_s=0.0, sample_id=0)
    estimator.update([0.5], [0.0], timestamp_s=0.01, sample_id=1)
    estimator.update([0.5], [0.0], timestamp_s=0.02, sample_id=2)
    result = estimator.update([0.5], [0.0], timestamp_s=0.03, sample_id=3)

    assert result.position_accepted[0]
    assert result.positions[0] == pytest.approx(0.5)
    assert result.velocities[0] == pytest.approx(0.0)


def test_covariance_remains_symmetric_positive_semidefinite():
    estimator = _estimator()
    for sample in range(1000):
        timestamp = sample * 0.01
        estimator.update(
            [0.2 * timestamp],
            [0.2],
            timestamp_s=timestamp,
            sample_id=sample,
        )

    covariance_diagonal = estimator.diagnostics()["covariance_diagonal"]
    assert np.all(np.isfinite(covariance_diagonal))
    assert np.all(covariance_diagonal >= 0.0)
