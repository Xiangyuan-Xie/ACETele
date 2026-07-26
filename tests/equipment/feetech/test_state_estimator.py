import numpy as np
import pytest

from acetele.equipment.feetech.state_estimator import (
    ENCODER_POSITION_RESOLUTION_RAD,
    FeeTechStateEstimator,
)


def _settle(estimator, *, samples=100, position=0.0, start_sample_id=0):
    estimate = None
    for offset in range(samples):
        sample_id = start_sample_id + offset
        estimate = estimator.update(
            np.array([position]),
            np.array([0.0]),
            timestamp=sample_id * 0.01,
            sample_id=sample_id,
        )
    return estimate


def test_velocity_spike_is_rejected_without_polluting_the_estimate():
    estimator = FeeTechStateEstimator(1)
    estimator.update(
        np.array([0.0]),
        np.array([0.0]),
        timestamp=0.0,
        sample_id=1,
    )
    estimator.update(
        np.array([0.01]),
        np.array([1.0]),
        timestamp=0.01,
        sample_id=2,
    )

    estimate = estimator.update(
        np.array([0.02]),
        np.array([2500.0]),
        timestamp=0.02,
        sample_id=3,
    )

    diagnostics = estimator.get_diagnostics()
    assert estimate.velocity_accepted.tolist() == [False]
    assert abs(estimate.velocities[0]) < 5.0
    assert diagnostics["measured_velocities"].tolist() == [2500.0]
    assert diagnostics["velocity_rejection_count"].tolist() == [1]


def test_position_spike_uses_prediction_until_measurement_recovers():
    estimator = FeeTechStateEstimator(1)
    estimator.update(
        np.array([0.0]),
        np.array([0.0]),
        timestamp=0.0,
        sample_id=1,
    )
    estimator.update(
        np.array([0.0]),
        np.array([0.0]),
        timestamp=0.01,
        sample_id=2,
    )

    spike = estimator.update(
        np.array([2.0]),
        np.array([0.0]),
        timestamp=0.02,
        sample_id=3,
    )
    recovered = estimator.update(
        np.array([0.0]),
        np.array([0.0]),
        timestamp=0.03,
        sample_id=4,
    )

    assert spike.position_accepted.tolist() == [False]
    assert abs(spike.positions[0]) < 0.1
    assert recovered.position_accepted.tolist() == [True]
    assert recovered.positions[0] == 0.0


def test_moderate_position_spike_does_not_reach_public_state_or_velocity():
    estimator = FeeTechStateEstimator(1)
    _settle(estimator)

    spike = estimator.update(
        np.array([0.05]),
        np.array([0.0]),
        timestamp=1.0,
        sample_id=100,
    )
    recovered = estimator.update(
        np.array([0.0]),
        np.array([0.0]),
        timestamp=1.01,
        sample_id=101,
    )

    assert spike.position_accepted.tolist() == [False]
    assert abs(spike.positions[0]) < 0.01
    assert abs(spike.velocities[0]) <= 0.5
    assert recovered.position_accepted.tolist() == [True]
    assert abs(recovered.positions[0]) < 0.01


def test_stationary_velocity_pulse_is_rejected():
    estimator = FeeTechStateEstimator(1)
    _settle(estimator)

    estimate = estimator.update(
        np.array([0.0]),
        np.array([1.5]),
        timestamp=1.0,
        sample_id=100,
    )

    assert estimate.velocity_accepted.tolist() == [False]
    assert abs(estimate.velocities[0]) < 0.2


def test_velocity_estimate_smooths_quantized_measurements():
    estimator = FeeTechStateEstimator(1)
    raw_velocities = []
    filtered_velocities = []

    for sample_id in range(1, 81):
        timestamp = (sample_id - 1) * 0.01
        raw_velocity = 0.5 if sample_id % 2 else 1.5
        estimate = estimator.update(
            np.array([timestamp]),
            np.array([raw_velocity]),
            timestamp=timestamp,
            sample_id=sample_id,
        )
        if sample_id > 40:
            raw_velocities.append(raw_velocity)
            filtered_velocities.append(float(estimate.velocities[0]))

    assert np.var(filtered_velocities) < 0.2 * np.var(raw_velocities)
    assert np.mean(filtered_velocities) == pytest.approx(1.0, abs=0.15)


def test_position_estimate_smooths_encoder_quantization():
    estimator = FeeTechStateEstimator(1)
    measured_positions = []
    estimated_positions = []

    for sample_id in range(300):
        position = (
            2.0 if sample_id % 2 else -2.0
        ) * ENCODER_POSITION_RESOLUTION_RAD
        estimate = estimator.update(
            np.array([position]),
            np.array([0.0]),
            timestamp=sample_id * 0.01,
            sample_id=sample_id,
        )
        if sample_id >= 100:
            measured_positions.append(position)
            estimated_positions.append(float(estimate.positions[0]))

    assert np.std(estimated_positions) < 0.3 * np.std(measured_positions)


def test_physical_acceleration_tracks_with_low_position_delay():
    estimator = FeeTechStateEstimator(1)
    position = 0.0
    velocity = 0.0
    measured_positions = []
    estimated_positions = []
    velocity_errors = []
    position_errors = []

    for sample_id in range(200):
        timestamp = sample_id * 0.01
        if 0.5 <= timestamp < 0.525:
            velocity = min(1.0, velocity + 40.0 * 0.01)
        position += velocity * 0.01
        estimate = estimator.update(
            np.array([position]),
            np.array([velocity]),
            timestamp=timestamp,
            sample_id=sample_id,
        )
        measured_positions.append(position)
        estimated_positions.append(float(estimate.positions[0]))
        if 0.7 <= timestamp <= 1.3:
            position_errors.append(abs(estimate.positions[0] - position))
            velocity_errors.append(abs(estimate.velocities[0] - velocity))

    measured_crossing = int(np.flatnonzero(np.asarray(measured_positions) >= 0.1)[0])
    estimated_crossing = int(np.flatnonzero(np.asarray(estimated_positions) >= 0.1)[0])
    assert max(position_errors) < 0.02
    assert np.mean(velocity_errors) < 0.15
    assert (estimated_crossing - measured_crossing) * 0.01 <= 0.02


def test_duplicate_sample_id_does_not_apply_measurement_twice():
    estimator = FeeTechStateEstimator(1)
    first = estimator.update(
        np.array([0.0]),
        np.array([0.0]),
        timestamp=0.0,
        sample_id=7,
    )

    duplicate = estimator.update(
        np.array([1.0]),
        np.array([100.0]),
        timestamp=0.1,
        sample_id=7,
    )

    np.testing.assert_allclose(duplicate.positions, first.positions)
    np.testing.assert_allclose(duplicate.velocities, first.velocities)
    assert estimator.get_diagnostics()["velocity_rejection_count"].tolist() == [0]


def test_out_of_order_timestamp_freezes_estimate_and_records_rejection():
    estimator = FeeTechStateEstimator(1)
    previous = estimator.update(
        np.array([0.2]),
        np.array([0.0]),
        timestamp=1.0,
        sample_id=1,
    )

    stale = estimator.update(
        np.array([1.0]),
        np.array([10.0]),
        timestamp=0.5,
        sample_id=2,
    )

    np.testing.assert_allclose(stale.positions, previous.positions)
    np.testing.assert_allclose(stale.velocities, previous.velocities)
    assert stale.position_accepted.tolist() == [False]
    assert stale.velocity_accepted.tolist() == [False]
    diagnostics = estimator.get_diagnostics()
    assert diagnostics["position_rejection_count"].tolist() == [1]
    assert diagnostics["velocity_rejection_count"].tolist() == [1]


def test_nonfinite_observations_do_not_pollute_estimate():
    estimator = FeeTechStateEstimator(1)
    _settle(estimator)

    estimate = estimator.update(
        np.array([np.nan]),
        np.array([np.inf]),
        timestamp=1.0,
        sample_id=100,
    )

    assert np.all(np.isfinite(estimate.positions))
    assert np.all(np.isfinite(estimate.velocities))
    assert estimate.position_accepted.tolist() == [False]
    assert estimate.velocity_accepted.tolist() == [False]


def test_nonfinite_observation_after_long_gap_preserves_last_position():
    estimator = FeeTechStateEstimator(1)
    previous = estimator.update(
        np.array([0.4]),
        np.array([0.0]),
        timestamp=0.0,
        sample_id=1,
    )

    estimate = estimator.update(
        np.array([np.nan]),
        np.array([np.inf]),
        timestamp=1.0,
        sample_id=2,
    )

    np.testing.assert_allclose(estimate.positions, previous.positions)
    assert estimate.velocities.tolist() == [0.0]
    assert estimate.position_accepted.tolist() == [False]


def test_three_stable_rejected_positions_reanchor_the_filter():
    estimator = FeeTechStateEstimator(1)
    _settle(estimator, samples=50)

    first = estimator.update(
        np.array([0.2]),
        np.array([0.0]),
        timestamp=0.50,
        sample_id=50,
    )
    second = estimator.update(
        np.array([0.2]),
        np.array([0.0]),
        timestamp=0.51,
        sample_id=51,
    )
    third = estimator.update(
        np.array([0.2]),
        np.array([0.0]),
        timestamp=0.52,
        sample_id=52,
    )

    assert first.position_accepted.tolist() == [False]
    assert second.position_accepted.tolist() == [False]
    assert third.position_accepted.tolist() == [True]
    assert third.positions[0] == pytest.approx(0.2)
    assert third.velocities[0] == 0.0
    assert estimator.get_diagnostics()["consecutive_position_rejections"].tolist() == [0]


def test_position_unwraps_across_pi_boundary():
    estimator = FeeTechStateEstimator(1)
    initial = np.pi - 0.005
    estimator.update(
        np.array([initial]),
        np.array([0.0]),
        timestamp=0.0,
        sample_id=1,
    )

    estimate = estimator.update(
        np.array([-np.pi + 0.005]),
        np.array([1.0]),
        timestamp=0.01,
        sample_id=2,
    )

    assert estimate.position_accepted.tolist() == [True]
    assert abs(estimate.positions[0] - initial) < 0.02


@pytest.mark.parametrize(
    ("positions", "velocities", "spike_index"),
    [
        (
            [3.138525, 3.138525, 3.136991, 3.138525, 3.138525],
            [0.0, 0.0, -0.076655, 0.0, 0.0],
            None,
        ),
        (
            [-1.56466043, -1.56312644, -1.56619442, -1.56466043, -1.56312644],
            [0.0, 0.0, -0.15330972, 0.0, 0.07665486],
            None,
        ),
        (
            [0.0, 0.0, -0.09970875, -0.00306796, -0.00153398],
            [0.0, 0.0, -3.60277843, 0.15330972, 0.0],
            2,
        ),
        (
            [-0.00306796, -0.00306796, 0.00766990, 0.0, -0.00153398],
            [0.0, -0.07665486, 0.45992917, -0.15330972, 0.0],
            None,
        ),
    ],
)
def test_real_log_snippets_do_not_amplify_velocity(positions, velocities, spike_index):
    estimator = FeeTechStateEstimator(1)
    estimates = []

    for sample_id, (position, velocity) in enumerate(zip(positions, velocities)):
        estimates.append(
            estimator.update(
                np.array([position]),
                np.array([velocity]),
                timestamp=sample_id * 0.1,
                sample_id=sample_id,
            )
        )

    filtered_velocities = np.asarray(
        [estimate.velocities[0] for estimate in estimates]
    )
    assert np.max(np.abs(filtered_velocities)) <= np.max(np.abs(velocities))
    assert np.all(np.isfinite(filtered_velocities))
    assert abs(estimates[-1].positions[0] - positions[-1]) < 0.02
    if spike_index is not None:
        assert estimates[spike_index].position_accepted.tolist() == [False]


def test_covariance_remains_finite_symmetric_and_positive_semidefinite():
    estimator = FeeTechStateEstimator(2)
    rng = np.random.default_rng(42)

    for sample_id in range(5000):
        timestamp = sample_id * 0.004
        position = np.array(
            [
                0.3 * np.sin(timestamp),
                -0.2 * np.cos(0.5 * timestamp),
            ]
        )
        velocity = np.array(
            [
                0.3 * np.cos(timestamp),
                0.1 * np.sin(0.5 * timestamp),
            ]
        )
        estimator.update(
            position + rng.normal(0.0, ENCODER_POSITION_RESOLUTION_RAD, 2),
            velocity + rng.normal(0.0, 0.05, 2),
            timestamp=timestamp,
            sample_id=sample_id,
        )

    covariance = estimator._covariance
    assert np.all(np.isfinite(covariance))
    np.testing.assert_allclose(covariance, np.swapaxes(covariance, 1, 2), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(covariance) >= 0.0)


def test_diagnostics_are_independent_copies():
    estimator = FeeTechStateEstimator(1)
    estimator.update(
        np.array([0.0]),
        np.array([0.0]),
        timestamp=0.0,
        sample_id=1,
    )

    diagnostics = estimator.get_diagnostics()
    diagnostics["estimated_velocities"][0] = 10.0
    diagnostics["covariance_diagonal"][0, 0] = 10.0

    assert estimator.get_diagnostics()["estimated_velocities"][0] == 0.0
    assert estimator.get_diagnostics()["covariance_diagonal"][0, 0] != 10.0
