from __future__ import annotations

import numpy as np
import pytest

from acetele.control import PositionControlPipeline, StreamingPositionTuning
from acetele.core import JointCommand, JointState
from acetele.model import ArmModelMetadata


def _metadata() -> ArmModelMetadata:
    return ArmModelMetadata(("joint",), (-1.0,), (1.0,), (2.0,), (3.0,))


def _state(position: float, velocity: float = 0.0) -> JointState:
    return JointState(("joint",), [position], [velocity], [0.0], 0, 0)


def _command(position: float, now_ns: int) -> JointCommand:
    return JointCommand(("joint",), [position], now_ns, now_ns + 50_000_000, 0)


def test_streaming_position_tuning_rejects_nonpositive_or_nonfinite_limits():
    with pytest.raises(ValueError, match="finite and positive"):
        StreamingPositionTuning(velocity_limit_rad_s=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        StreamingPositionTuning(acceleration_limit_rad_s2=float("nan"))


def test_pipeline_rate_limits_against_urdf_velocity():
    pipeline = PositionControlPipeline(_metadata())
    pipeline.update_feedback(_state(0.0))

    result = pipeline.apply(_command(1.0, 10_000_000), now_ns=10_000_000)

    assert result.positions[0] == pytest.approx(0.002)
    assert pipeline.diagnostics().command_limited.tolist() == [True]


def test_pipeline_enforces_acceleration_in_software():
    pipeline = PositionControlPipeline(_metadata())
    pipeline.update_feedback(_state(0.0))
    initial = JointCommand(
        ("joint",),
        [0.0],
        0,
        50_000_000,
        0,
        velocity_limits=[2.0],
        acceleration_limits=[1.5],
    )
    pipeline.apply(initial, now_ns=0)
    positions = [0.0]
    for index in range(1, 31):
        now_ns = index * 10_000_000
        command = JointCommand(
            ("joint",),
            [1.0],
            now_ns,
            now_ns + 50_000_000,
            0,
            velocity_limits=[2.0],
            acceleration_limits=[1.5],
        )
        positions.append(float(pipeline.apply(command, now_ns=now_ns).positions[0]))

    interval_velocities = np.diff(positions) / 0.01
    interval_accelerations = np.diff(
        np.concatenate(([0.0], interval_velocities))
    ) / 0.01

    assert np.max(np.abs(interval_velocities)) <= 2.0 + 1e-9
    assert np.max(np.abs(interval_accelerations)) <= 1.5 + 1e-9


def test_pipeline_decelerates_before_reversing_direction():
    pipeline = PositionControlPipeline(_metadata())
    pipeline.update_feedback(_state(0.0))
    positions = []
    for index in range(50):
        now_ns = index * 10_000_000
        command = JointCommand(
            ("joint",),
            [1.0 if index < 20 else -1.0],
            now_ns,
            now_ns + 50_000_000,
            0,
            velocity_limits=[2.0],
            acceleration_limits=[2.0],
        )
        positions.append(float(pipeline.apply(command, now_ns=now_ns).positions[0]))

    velocities = np.diff(positions) / 0.01
    accelerations = np.diff(velocities) / 0.01

    assert velocities[19] > 0.0
    assert velocities[20] > 0.0
    assert np.max(np.abs(accelerations)) <= 2.0 + 1e-9


def test_pipeline_diagnostics_are_independent_snapshots():
    pipeline = PositionControlPipeline(_metadata())
    diagnostics = pipeline.diagnostics()

    with pytest.raises(ValueError):
        diagnostics.command_limited[0] = True
    np.testing.assert_array_equal(
        pipeline.diagnostics().command_limited,
        np.array([False]),
    )


def test_prepared_command_does_not_change_history_until_committed():
    pipeline = PositionControlPipeline(_metadata())
    pipeline.update_feedback(_state(0.0))

    prepared = pipeline.prepare(_command(1.0, 10_000_000), now_ns=10_000_000)

    assert prepared.command.positions[0] == pytest.approx(0.002)
    assert pipeline.diagnostics().command_limited.tolist() == [False]

    pipeline.commit(prepared)

    assert pipeline.diagnostics().command_limited.tolist() == [True]


def test_rebase_uses_latest_feedback_as_rate_limit_origin():
    pipeline = PositionControlPipeline(_metadata())
    pipeline.update_feedback(_state(0.0))
    pipeline.apply(_command(0.05, 0), now_ns=0)

    pipeline.update_feedback(_state(0.6))
    pipeline.rebase_to_feedback()
    result = pipeline.apply(_command(1.0, 220_000_000), now_ns=220_000_000)

    assert result.positions[0] == pytest.approx(0.602)
