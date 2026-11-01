"""Verifies the physical example using mock buses and blocked serial access.

Private example helpers are exercised to verify hardware gates without a
browser or any real actuator connection.
"""

# pylint: disable=protected-access

from __future__ import annotations

from collections.abc import Callable, Iterator
import dataclasses
import importlib.util
import pathlib
import sys
import time
import types
import typing
from unittest import mock

import numpy as np
import pytest

from acetele import config
from acetele import runtime
from acetele import specification

pytest.importorskip("viser")
pytest.importorskip("yourdfpy")

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "examples/eaa_follower_physical.toml"
_RuntimeClass = runtime.RobotRuntime
_INITIAL = np.array([0.2, -0.2, 0.1, -0.1])


@pytest.fixture(name="example")
def example_fixture() -> types.ModuleType:
    """Loads the physical example without executing its CLI."""
    path = _ROOT / "examples/eaa_physical_follower.py"
    spec = importlib.util.spec_from_file_location("eaa_physical_follower", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the physical follower example")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="make_bridge")
def make_bridge_fixture(
    example: types.ModuleType,
) -> Iterator[Callable[[np.ndarray], typing.Any]]:
    """Replaces physical runtime construction with an actual core mock runtime.

    Args:
        example: The imported physical example.

    Yields:
        A factory accepting initial measured mock positions.
    """
    bridges = []
    with mock.patch("serial.Serial") as serial_constructor:

        def create(initial: np.ndarray) -> typing.Any:
            """Creates a mock-backed bridge without physical serial access."""

            def mock_runtime(
                spec: specification.RobotSpec,
                **kwargs: typing.Any,
            ) -> typing.Any:
                """Substitutes mock devices at specified measured positions."""
                arm = spec.arms[0]
                joints = tuple(
                    dataclasses.replace(joint, home_position_rad=float(value))
                    for joint, value in zip(arm.joints, initial, strict=True)
                )
                mock_spec = dataclasses.replace(
                    spec,
                    backend=specification.Backend.MOCK,
                    arms=(dataclasses.replace(arm, joints=joints),),
                )
                return _RuntimeClass(mock_spec, **kwargs)

            with mock.patch.object(
                example.runtime, "RobotRuntime", side_effect=mock_runtime
            ):
                bridge = example._PhysicalFollowerBridge(_CONFIG)
            bridges.append(bridge)
            return bridge

        try:
            yield create
        finally:
            for bridge in bridges:
                bridge.close()
            serial_constructor.assert_not_called()


@pytest.fixture(name="bridge")
def bridge_fixture(
    make_bridge: Callable[[np.ndarray], typing.Any],
) -> typing.Any:
    """Connects a mock-backed bridge inside its legal URDF range.

    Args:
        make_bridge: The guarded bridge factory.

    Returns:
        A bridge holding a known initial pose.
    """
    bridge = make_bridge(_INITIAL)
    bridge.connect()
    return bridge


def _drive(
    bridge: typing.Any,
    target: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    """Publishes targets at the viewer cadence using only mock hardware.

    Args:
        bridge: The mock-backed example bridge.
        target: Four canonical target positions in radians.
        duration_s: Duration of the simulated operator connection.

    Returns:
        The latest measured mock positions.
    """
    finish = time.monotonic() + duration_s
    positions = bridge.read_positions()
    while time.monotonic() < finish:
        positions = bridge.step(target, client_connected=True)
        time.sleep(0.04)
    return positions


def test_default_cli_only_preflights(example: types.ModuleType) -> None:
    """The default CLI cannot create a serial connection or a server."""
    with (
        mock.patch("serial.Serial") as serial_constructor,
        mock.patch.object(example.viser, "ViserServer") as server_constructor,
        mock.patch.object(_RuntimeClass, "connect") as connect,
    ):
        assert example.main([]) == 0
        connect.assert_not_called()
        serial_constructor.assert_not_called()
        server_constructor.assert_not_called()


def test_cancelled_confirmation_never_connects(
    example: types.ModuleType,
) -> None:
    """Adding --connect still requires the exact terminal confirmation."""
    with (
        mock.patch("builtins.input", return_value="cancel"),
        mock.patch.object(_RuntimeClass, "connect") as connect,
        mock.patch.object(example.viser, "ViserServer") as server,
    ):
        assert example.main(["--connect"]) == 130
        connect.assert_not_called()
        server.assert_not_called()


def test_mock_config_is_rejected_before_runtime_creation(
    example: types.ModuleType,
) -> None:
    """The physical entry cannot silently present mock feedback as real."""
    spec = dataclasses.replace(
        config.load_robot_spec(_CONFIG), backend=specification.Backend.MOCK
    )
    with (
        mock.patch.object(example.config, "load_robot_spec", return_value=spec),
        mock.patch.object(example.runtime, "RobotRuntime") as constructor,
    ):
        with pytest.raises(ValueError, match="backend = 'physical'"):
            example._PhysicalFollowerBridge(_CONFIG)
        constructor.assert_not_called()


def test_constructor_does_not_connect(
    make_bridge: Callable[[np.ndarray], typing.Any],
) -> None:
    """Constructing a bridge leaves the owned runtime disconnected."""
    bridge = make_bridge(_INITIAL)
    assert not bridge.session.runtime.connected
    assert not bridge.tracking


def test_startup_holds_without_moving(bridge: typing.Any) -> None:
    """An unrelated slider target cannot move the startup holding pose."""
    target = _INITIAL + np.array([0.1, -0.1, 0.1, 0.1])
    positions = _drive(bridge, target, 0.16)
    np.testing.assert_allclose(positions, _INITIAL)
    assert not bridge.tracking
    assert bridge.session.runtime.diagnostics().safety.state.value == "hold"


def test_gravity_sag_is_held_and_never_clipped_into_tracking(
    make_bridge: Callable[[np.ndarray], typing.Any],
) -> None:
    """Out-of-range feedback may be held but cannot start streaming motion."""
    sag = np.radians([-6.1, 4.0, -1.8, -0.1])
    bridge = make_bridge(sag)
    positions = bridge.connect()
    np.testing.assert_allclose(positions, sag)
    with mock.patch.object(bridge.session, "write_arm") as write:
        with pytest.raises(ValueError, match="URDF limits"):
            bridge.start_tracking()
        positions = bridge.step(np.zeros(4), client_connected=True)
        write.assert_not_called()
    np.testing.assert_allclose(positions, sag)
    assert not bridge.tracking
    assert bridge.session.runtime.diagnostics().safety.state.value == "hold"


def test_start_seeds_feedback_and_keeps_stationary_target_alive(
    bridge: typing.Any,
) -> None:
    """Explicit Start seeds feedback and publication sustains the lease."""
    np.testing.assert_allclose(bridge.start_tracking(), _INITIAL)
    target = _INITIAL + np.array([0.005, -0.005, 0.005, 0.005])
    positions = _drive(bridge, target, 0.9)
    assert bridge.tracking
    assert bridge.session.status.value == "tracking"
    np.testing.assert_allclose(positions, target, atol=0.002, rtol=0.0)


def test_disconnect_holds_and_reconnection_does_not_resume(
    bridge: typing.Any,
) -> None:
    """A returning browser cannot resume a stale slider target."""
    bridge.start_tracking()
    target = _INITIAL + np.array([0.1, -0.1, 0.1, 0.1])
    _drive(bridge, target, 0.2)
    bridge.step(target, client_connected=False)
    held = _drive(bridge, target, 0.12)
    positions = _drive(bridge, target, 0.16)
    assert not bridge.tracking
    np.testing.assert_allclose(positions, held)
    np.testing.assert_allclose(bridge.start_tracking(), held)


@pytest.mark.parametrize(
    "target",
    [np.zeros(3), np.full(4, np.nan), np.array([-0.01, 0.0, 0.0, 0.0])],
    ids=["incomplete", "non-finite", "out-of-range"],
)
def test_bad_target_holds_before_admission(
    bridge: typing.Any,
    target: np.ndarray,
) -> None:
    """Invalid targets never reach the motion admission method."""
    bridge.start_tracking()
    with mock.patch.object(bridge.session, "write_arm") as write:
        with pytest.raises(ValueError, match="target"):
            bridge.step(target, client_connected=True)
        write.assert_not_called()
    assert not bridge.tracking
    assert bridge.session.runtime.diagnostics().safety.state.value == "hold"


def test_stale_feedback_holds_before_admission(
    bridge: typing.Any,
) -> None:
    """A cached stale state cannot admit another streaming target."""
    bridge.start_tracking()
    sample = types.SimpleNamespace(
        positions=_INITIAL,
        timestamp_ns=time.monotonic_ns() - 1_000_000_000,
    )
    state = types.SimpleNamespace(joints={bridge.group_name: sample})
    with (
        mock.patch.object(bridge.session, "read", return_value=state),
        mock.patch.object(bridge.session, "write_arm") as write,
    ):
        with pytest.raises(RuntimeError, match="stale"):
            bridge.step(_INITIAL, client_connected=True)
        write.assert_not_called()
    assert not bridge.tracking


def test_stop_latches_and_cannot_be_restarted(bridge: typing.Any) -> None:
    """Explicit Stop closes tracking and requires a fresh process session."""
    bridge.start_tracking()
    bridge.stop()
    assert not bridge.tracking
    with pytest.raises(RuntimeError, match="faulted"):
        bridge.start_tracking()


def test_request_priority_prevents_queued_start(
    example: types.ModuleType,
) -> None:
    """STOP overrides HOLD and START; HOLD overrides START."""
    requests = example._Requests()
    requests.submit(example._Request.START)
    requests.submit(example._Request.HOLD)
    requests.submit(example._Request.START)
    assert requests.take() == example._Request.HOLD
    requests.submit(example._Request.HOLD)
    requests.submit(example._Request.STOP)
    requests.submit(example._Request.START)
    assert requests.take() == example._Request.STOP
    assert requests.take() == example._Request.NONE


def test_ui_sag_blocks_start_and_preserves_feedback_pose(
    example: types.ModuleType,
    make_bridge: Callable[[np.ndarray], typing.Any],
) -> None:
    """The display never replaces sagging feedback with clipped targets."""
    sag = np.radians([-6.1, 4.0, -1.8, -0.1])
    bridge = make_bridge(sag)
    bridge.connect()
    viewer = object.__new__(example._Viewer)
    viewer._robot = mock.Mock()
    viewer._start = types.SimpleNamespace(disabled=False)
    viewer._feedback = types.SimpleNamespace(content="")
    viewer._sliders = [
        types.SimpleNamespace(value=0.0, disabled=False) for _ in range(4)
    ]
    with mock.patch.object(viewer, "seed") as seed:
        viewer.refresh(sag, bridge)
        seed.assert_not_called()
    viewer._robot.update_cfg.assert_called_once()
    np.testing.assert_allclose(viewer._robot.update_cfg.call_args.args[0], sag)
    assert viewer._start.disabled
    assert all(slider.disabled for slider in viewer._sliders)
    assert "Tracking remains blocked" in viewer._feedback.content


def test_feedback_leaving_limits_holds_before_admission(
    bridge: typing.Any,
) -> None:
    """A valid startup cannot keep streaming after feedback leaves its range."""
    bridge.start_tracking()
    sample = types.SimpleNamespace(
        positions=np.array([-0.01, -0.2, 0.1, -0.1]),
        timestamp_ns=time.monotonic_ns(),
    )
    state = types.SimpleNamespace(joints={bridge.group_name: sample})
    with (
        mock.patch.object(bridge.session, "read", return_value=state),
        mock.patch.object(bridge.session, "write_arm") as write,
    ):
        with pytest.raises(ValueError, match="Feedback left URDF limits"):
            bridge.step(_INITIAL, client_connected=True)
        write.assert_not_called()
    assert not bridge.tracking
    assert bridge.session.runtime.diagnostics().safety.state.value == "hold"


def test_tracking_log_reports_serial_write_evidence(
    example: types.ModuleType,
    bridge: typing.Any,
) -> None:
    """The terminal distinguishes requested targets from measured feedback."""
    with mock.patch.object(example.logging, "info") as log:
        example._log_tracking(bridge, _INITIAL, _INITIAL)
    log.assert_called_once()
    assert "serial_tx_samples" in log.call_args.args[0]
    assert "Target_deg" in log.call_args.args[0]
    assert "Feedback_deg" in log.call_args.args[0]


def test_viewer_loop_reports_hold_after_feedback_leaves_limits(
    example: types.ModuleType,
    bridge: typing.Any,
) -> None:
    """The operator loop displays the rejected pose after revoking tracking."""
    bridge.start_tracking()
    sample = types.SimpleNamespace(
        positions=np.array([-0.01, -0.2, 0.1, -0.1]),
        timestamp_ns=time.monotonic_ns(),
    )
    state = types.SimpleNamespace(joints={bridge.group_name: sample})
    server = mock.Mock()
    server.get_clients.return_value = {1: object()}
    viewer = mock.Mock()
    viewer.targets.return_value = _INITIAL
    with (
        mock.patch.object(example, "_Viewer", return_value=viewer),
        mock.patch.object(bridge.session, "read", return_value=state),
        mock.patch.object(bridge.session, "write_arm") as write,
        mock.patch.object(
            example.time, "sleep", side_effect=KeyboardInterrupt
        ),
    ):
        with pytest.raises(KeyboardInterrupt):
            example._run_viewer(server, bridge)
        write.assert_not_called()
    assert not bridge.tracking
    assert "Feedback left URDF limits" in viewer.refresh.call_args.args[2]
    assert bridge.session.runtime.diagnostics().safety.state.value == "hold"


def test_close_preserves_hold_and_is_repeatable(bridge: typing.Any) -> None:
    """Normal cleanup requests preserved HOLD rather than torque release."""
    with mock.patch.object(
        bridge.session, "close", wraps=bridge.session.close
    ) as close:
        bridge.close()
        bridge.close()
        close.assert_called_once_with(preserve_hold=True)
    assert not bridge.session.runtime.connected


def test_connection_failure_closes_without_enabling(
    make_bridge: Callable[[np.ndarray], typing.Any],
) -> None:
    """A failed startup cannot enable torque during cleanup."""
    bridge = make_bridge(_INITIAL)
    with (
        mock.patch.object(
            bridge.session, "connect", side_effect=RuntimeError("offline")
        ),
        mock.patch.object(bridge.session, "hold_position") as enable,
        mock.patch.object(
            bridge.session, "close", wraps=bridge.session.close
        ) as close,
    ):
        with pytest.raises(RuntimeError, match="offline"):
            bridge.connect()
        enable.assert_not_called()
        close.assert_called_once_with(preserve_hold=True)


def test_feedback_recovery_needs_distinct_samples_and_manual_start(bridge: typing.Any) -> None:
    """Cached or recovered feedback never replays the retired tracking target."""
    bridge.start_tracking()
    stale = types.SimpleNamespace(
        joints={bridge.group_name: types.SimpleNamespace(
            positions=_INITIAL, timestamp_ns=time.monotonic_ns() - 1_000_000_000
        )}
    )
    with mock.patch.object(bridge.session, "read", return_value=stale):
        with pytest.raises(RuntimeError, match="stale"):
            bridge.step(_INITIAL + 0.1, client_connected=True)
    assert not bridge.tracking
    assert not bridge.feedback_ready
    assert bridge.session.runtime.diagnostics().safety.state.value == "hold"

    fresh_position = _INITIAL + np.array([0.001, -0.001, 0.001, 0.001])
    first_timestamp = time.monotonic_ns()
    first = types.SimpleNamespace(
        joints={bridge.group_name: types.SimpleNamespace(
            positions=fresh_position, timestamp_ns=first_timestamp
        )}
    )
    with (
        mock.patch.object(bridge.session, "read", return_value=first),
        mock.patch.object(bridge.session, "write_arm") as write,
    ):
        for _ in range(5):
            bridge.step(_INITIAL + 0.1, client_connected=True)
        write.assert_not_called()
        assert not bridge.feedback_ready
        with pytest.raises(RuntimeError, match="recovery pending"):
            bridge.start_tracking()

    with mock.patch.object(bridge.session, "write_arm") as write:
        for _ in range(2):
            fresh = types.SimpleNamespace(
                joints={bridge.group_name: types.SimpleNamespace(
                    positions=fresh_position, timestamp_ns=time.monotonic_ns()
                )}
            )
            with mock.patch.object(bridge.session, "read", return_value=fresh):
                bridge.step(_INITIAL + 0.1, client_connected=True)
        assert bridge.feedback_ready
        assert not bridge.tracking
        write.assert_not_called()
        with mock.patch.object(bridge.session, "read", return_value=fresh):
            np.testing.assert_allclose(bridge.start_tracking(), fresh_position)
        assert bridge.tracking


def test_viewer_survives_repeated_stale_feedback(
    example: types.ModuleType, bridge: typing.Any
) -> None:
    """Feedback loss keeps the viewer alive without admitting another target."""
    bridge.start_tracking()
    stale = types.SimpleNamespace(
        joints={bridge.group_name: types.SimpleNamespace(
            positions=_INITIAL, timestamp_ns=time.monotonic_ns() - 1_000_000_000
        )}
    )
    server = mock.Mock()
    server.get_clients.return_value = {1: object()}
    viewer = mock.Mock()
    viewer.targets.return_value = _INITIAL + 0.1
    with (
        mock.patch.object(example, "_Viewer", return_value=viewer),
        mock.patch.object(bridge.session, "read", return_value=stale),
        mock.patch.object(bridge.session, "write_arm") as write,
        mock.patch.object(example.time, "sleep", side_effect=[None, None, KeyboardInterrupt]),
    ):
        with pytest.raises(KeyboardInterrupt):
            example._run_viewer(server, bridge)
        write.assert_not_called()
    server.stop.assert_not_called()
    assert not bridge.tracking
    assert "stale" in bridge.feedback_error


def test_viewer_keeps_fault_visible_and_does_not_reconnect(
    example: types.ModuleType, bridge: typing.Any
) -> None:
    """A failed worker cannot be cleared or reconnected by the viewer loop."""
    server = mock.Mock()
    server.get_clients.return_value = {1: object()}
    viewer = mock.Mock()
    viewer.targets.return_value = _INITIAL
    with (
        mock.patch.object(example, "_Viewer", return_value=viewer),
        mock.patch.object(bridge.session, "read", side_effect=RuntimeError("worker faulted")),
        mock.patch.object(bridge.session, "connect") as connect,
        mock.patch.object(bridge.session.runtime, "reset_emergency_stop") as reset,
        mock.patch.object(example.time, "sleep", side_effect=KeyboardInterrupt),
    ):
        with pytest.raises(KeyboardInterrupt):
            example._run_viewer(server, bridge)
        connect.assert_not_called()
        reset.assert_not_called()
    assert not bridge.feedback_ready
    assert "worker faulted" in bridge.feedback_error
    server.stop.assert_not_called()


def test_paused_viewer_labels_cached_feedback_and_blocks_start(
    example: types.ModuleType, bridge: typing.Any
) -> None:
    """The last good pose is display-only while recovery is incomplete."""
    bridge.pause_tracking("feedback timeout")
    viewer = object.__new__(example._Viewer)
    viewer._robot = mock.Mock()
    viewer._start = types.SimpleNamespace(disabled=False)
    viewer._feedback = types.SimpleNamespace(content="")
    viewer._sliders = [types.SimpleNamespace(value=0.0, disabled=False) for _ in range(4)]
    with mock.patch.object(viewer, "seed") as seed:
        viewer.refresh(bridge.display_positions, bridge)
        seed.assert_not_called()
    assert viewer._start.disabled
    assert all(slider.disabled for slider in viewer._sliders)
    assert "last valid sample" in viewer._feedback.content
    assert "PAUSED" in viewer._feedback.content


def test_hold_failure_does_not_mask_original_feedback_failure(
    example: types.ModuleType, bridge: typing.Any
) -> None:
    """A failed HOLD stays visible without restarting or closing the viewer."""
    bridge.start_tracking()
    server = mock.Mock()
    server.get_clients.return_value = {1: object()}
    viewer = mock.Mock()
    viewer.targets.return_value = _INITIAL
    with (
        mock.patch.object(example, "_Viewer", return_value=viewer),
        mock.patch.object(bridge.session, "read", side_effect=RuntimeError("feedback unavailable")),
        mock.patch.object(bridge.session.runtime, "hold", side_effect=RuntimeError("HOLD unconfirmed")),
        mock.patch.object(example.time, "sleep", side_effect=KeyboardInterrupt),
    ):
        with pytest.raises(KeyboardInterrupt):
            example._run_viewer(server, bridge)
    assert not bridge.tracking
    assert bridge.feedback_error == "feedback unavailable"
    assert not bridge.feedback_ready
    server.stop.assert_not_called()
