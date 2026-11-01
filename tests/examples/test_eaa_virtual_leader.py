"""Tests the virtual leader's mock bridge and operator safety transitions.

Private example helpers are exercised directly to avoid requiring a browser
for runtime, synchronization, disconnect, and serial-access regression tests.
"""

# pylint: disable=protected-access

from __future__ import annotations

from collections.abc import Iterator
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
from acetele import specification

pytest.importorskip("viser")
pytest.importorskip("yourdfpy")

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIG = (
    _ROOT
    / "acetele/config/presets/eaa_follower/fashionstar_rs485.toml"
)


@pytest.fixture(name="example")
def example_fixture() -> types.ModuleType:
    """Loads the example without running its CLI or starting its server."""
    path = _ROOT / "examples/eaa_virtual_leader.py"
    spec = importlib.util.spec_from_file_location("eaa_virtual_leader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the virtual leader example")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="bridge")
def bridge_fixture(example: types.ModuleType) -> Iterator[typing.Any]:
    """Connects only mock devices and asserts serial creation never occurs.

    Args:
        example: The imported example module.

    Yields:
        A connected mock bridge initially holding its measured position.
    """
    with mock.patch.object(
        example,
        "_reject_serial",
        side_effect=AssertionError("A mock bridge attempted serial access"),
    ) as serial_guard:
        bridge = example._MockFollowerBridge(_CONFIG)
        try:
            bridge.connect()
            yield bridge
        finally:
            bridge.close()
            serial_guard.assert_not_called()


def _target(bridge: typing.Any, delta: float = 0.03) -> np.ndarray:
    """Returns a small target directed into each joint's legal range.

    Args:
        bridge: The connected mock bridge.
        delta: Desired displacement magnitude in radians.

    Returns:
        A bounded target in canonical joint order.
    """
    metadata = bridge.session.runtime.preflight.arms[bridge.group_name]
    lower = np.asarray(metadata.lower_limits)
    upper = np.asarray(metadata.upper_limits)
    initial = bridge.read_positions()
    midpoint = 0.5 * (lower + upper)
    direction = np.where(midpoint >= initial, 1.0, -1.0)
    return np.clip(initial + delta * direction, lower, upper)


def _drive(
    bridge: typing.Any,
    target: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Publishes an unchanged target at the same cadence as the viewer.

    Args:
        bridge: The connected mock bridge.
        target: Positions to publish on each cycle.
        duration: Duration of the simulated operator connection in seconds.

    Returns:
        The final measured mock positions.
    """
    deadline = time.monotonic() + duration
    positions = bridge.read_positions()
    while time.monotonic() < deadline:
        positions = bridge.step(target, client_connected=True)
        time.sleep(0.04)
    return positions


def test_physical_backend_is_rejected_before_runtime_creation(
    example: types.ModuleType,
) -> None:
    """A physical configuration cannot reach runtime construction."""
    physical = dataclasses.replace(
        config.load_robot_spec(_CONFIG),
        backend=specification.Backend.PHYSICAL,
    )
    with (
        mock.patch.object(
            example.config, "load_robot_spec", return_value=physical
        ),
        mock.patch.object(example.runtime, "RobotRuntime") as constructor,
    ):
        with pytest.raises(ValueError, match="backend = 'mock'"):
            example._MockFollowerBridge(_CONFIG)
        constructor.assert_not_called()


def test_serial_transport_is_unconditionally_rejected(
    example: types.ModuleType,
) -> None:
    """The independent transport guard cannot create a real serial port."""
    with pytest.raises(RuntimeError, match="serial access is disabled"):
        example._reject_serial("/dev/ttyUSB0", 115200, object())


def test_slider_preview_cannot_move_follower_before_start(
    bridge: typing.Any,
) -> None:
    """Moving preview targets while holding never admits motion."""
    initial = bridge.read_positions()
    positions = _drive(bridge, _target(bridge), 0.16)
    assert not bridge.tracking
    assert bridge.session.runtime.diagnostics().safety.state.value == "hold"
    np.testing.assert_allclose(positions, initial)


def test_unchanged_slider_target_tracks_beyond_session_timeout(
    bridge: typing.Any,
) -> None:
    """Periodic publication keeps a stationary operator target admitted."""
    bridge.start_tracking()
    target = _target(bridge)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        positions = bridge.step(target, client_connected=True)
        if np.allclose(positions, target, atol=0.002, rtol=0.0):
            break
        time.sleep(0.04)
    else:
        pytest.fail("Mock feedback did not converge to the slider target")
    positions = _drive(bridge, target, 0.65)
    assert bridge.tracking
    assert bridge.session.status.value == "tracking"
    np.testing.assert_allclose(positions, target, atol=0.002, rtol=0.0)


def test_hold_freezes_motion_and_restart_uses_measured_pose(
    bridge: typing.Any,
) -> None:
    """Hold revokes old targets and each restart returns a fresh pose seed."""
    bridge.start_tracking()
    target = _target(bridge, delta=0.1)
    _drive(bridge, target, 0.3)
    bridge.hold()
    held = _drive(bridge, target, 0.12)
    positions = _drive(bridge, target, 0.16)
    assert not bridge.tracking
    np.testing.assert_allclose(positions, held)
    seed = bridge.start_tracking()
    np.testing.assert_allclose(seed, held)
    assert not np.allclose(seed, target, atol=0.005, rtol=0.0)
    positions = _drive(bridge, seed, 0.16)
    np.testing.assert_allclose(positions, seed, atol=0.002, rtol=0.0)


def test_browser_reconnection_cannot_resume_old_motion(
    bridge: typing.Any,
) -> None:
    """A missing client forces HOLD and reconnecting does not start motion."""
    bridge.start_tracking()
    target = _target(bridge)
    _drive(bridge, target, 0.12)
    bridge.step(target, client_connected=False)
    assert not bridge.tracking
    assert bridge.session.runtime.diagnostics().safety.state.value == "hold"
    held = _drive(bridge, target, 0.12)
    positions = _drive(bridge, target, 0.16)
    assert not bridge.tracking
    np.testing.assert_allclose(positions, held)


@pytest.mark.parametrize(
    "target",
    [np.array([float("nan"), 0.0, 0.0, 0.0]), np.zeros(3)],
    ids=["non-finite", "wrong-joint-count"],
)
def test_invalid_targets_revoke_tracking(
    bridge: typing.Any,
    target: np.ndarray,
) -> None:
    """An invalid UI target leaves the bridge holding instead of active."""
    bridge.start_tracking()
    with pytest.raises(ValueError):
        bridge.step(target, client_connected=True)
    assert not bridge.tracking
    assert bridge.session.runtime.diagnostics().safety.state.value == "hold"


def test_hold_request_has_priority_over_pending_start(
    example: types.ModuleType,
) -> None:
    """Disconnect/HOLD cannot be overwritten by a queued start callback."""
    requests = example._ControlRequests()
    requests.request_start()
    requests.request_hold()
    requests.request_start()
    assert requests.take() == (False, True)
    assert requests.take() == (False, False)
    requests.request_start()
    assert requests.take() == (True, False)


def test_close_disconnects_and_is_repeatable(bridge: typing.Any) -> None:
    """Cleanup disconnects every mock actor and can safely run again."""
    bridge.close()
    bridge.close()
    assert not bridge.tracking
    assert not bridge.session.runtime.connected
