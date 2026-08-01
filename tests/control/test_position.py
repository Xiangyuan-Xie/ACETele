from __future__ import annotations

import numpy as np
import pytest

from acetele.control import PositionControlPipeline
from acetele.core import JointCommand, JointState
from acetele.model import ArmModelMetadata
from acetele.specification import ControlSpec, PositionControlTuning


def _metadata() -> ArmModelMetadata:
    return ArmModelMetadata(("joint",), (-1.0,), (1.0,), (2.0,), (3.0,))


def _state(position: float, velocity: float = 0.0) -> JointState:
    return JointState(("joint",), [position], [velocity], [0.0], 0, 0)


def _command(position: float, now_ns: int) -> JointCommand:
    return JointCommand(("joint",), [position], now_ns, now_ns + 50_000_000, 0)


def test_pipeline_rate_limits_against_urdf_velocity():
    pipeline = PositionControlPipeline(_metadata(), ControlSpec())
    pipeline.update_feedback(_state(0.0))

    result = pipeline.apply(_command(1.0, 10_000_000), now_ns=10_000_000)

    assert result.positions[0] == pytest.approx(0.002)
    assert pipeline.diagnostics().command_limited.tolist() == [True]


def test_adaptive_residual_learns_only_after_static_gate():
    pipeline = PositionControlPipeline(
        _metadata(),
        ControlSpec(adaptive_position=True),
    )
    pipeline.update_feedback(_state(0.0))
    pipeline.apply(_command(0.05, 0), now_ns=0)
    pipeline.apply(_command(0.05, 210_000_000), now_ns=210_000_000)

    diagnostics = pipeline.diagnostics()
    assert diagnostics.adaptive_active.tolist() == [True]
    assert diagnostics.adaptive_estimate_rad[0] > 0.0
    assert diagnostics.adaptive_offset_rad[0] > 0.0


def test_pipeline_uses_the_tuning_owned_by_control_spec():
    pipeline = PositionControlPipeline(
        _metadata(),
        ControlSpec(
            adaptive_position=True,
            position_tuning=PositionControlTuning(stable_time_s=0.01),
        ),
    )
    pipeline.update_feedback(_state(0.0))
    pipeline.apply(_command(0.05, 0), now_ns=0)
    pipeline.apply(_command(0.05, 20_000_000), now_ns=20_000_000)

    assert pipeline.diagnostics().adaptive_active.tolist() == [True]


def test_pipeline_diagnostics_are_independent_snapshots():
    pipeline = PositionControlPipeline(_metadata(), ControlSpec())
    diagnostics = pipeline.diagnostics()

    with pytest.raises(ValueError):
        diagnostics.command_limited[0] = True
    np.testing.assert_array_equal(
        pipeline.diagnostics().command_limited,
        np.array([False]),
    )


def test_prepared_command_does_not_change_history_until_committed():
    pipeline = PositionControlPipeline(_metadata(), ControlSpec())
    pipeline.update_feedback(_state(0.0))

    prepared = pipeline.prepare(_command(1.0, 10_000_000), now_ns=10_000_000)

    assert prepared.command.positions[0] == pytest.approx(0.002)
    assert pipeline.diagnostics().command_limited.tolist() == [False]

    pipeline.commit(prepared)

    assert pipeline.diagnostics().command_limited.tolist() == [True]


def test_rebase_uses_latest_feedback_and_clears_adaptive_history():
    pipeline = PositionControlPipeline(
        _metadata(),
        ControlSpec(adaptive_position=True),
    )
    pipeline.update_feedback(_state(0.0))
    pipeline.apply(_command(0.05, 0), now_ns=0)
    pipeline.apply(_command(0.05, 210_000_000), now_ns=210_000_000)
    assert pipeline.diagnostics().adaptive_estimate_rad[0] > 0.0

    pipeline.update_feedback(_state(0.6))
    pipeline.rebase_to_feedback()
    result = pipeline.apply(_command(1.0, 220_000_000), now_ns=220_000_000)

    assert result.positions[0] == pytest.approx(0.602)
    diagnostics = pipeline.diagnostics()
    assert diagnostics.adaptive_estimate_rad.tolist() == [0.0]
    assert diagnostics.adaptive_offset_rad.tolist() == [0.0]
