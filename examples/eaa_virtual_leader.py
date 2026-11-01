"""Controls a mock EAA follower from a browser-based virtual leader.

Run this example in the acetele environment:
    python /home/lei/ACETele/examples/eaa_virtual_leader.py

Click Start tracking before moving the joint sliders. Starting synchronizes
the sliders to current feedback. Hold and browser disconnection stop tracking;
reconnection requires another explicit start. This example rejects physical
backends and serial transports. No real robot is connected.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import logging
import pathlib
import threading
import time
import typing

import numpy as np
import viser
from viser import extras
import yourdfpy

from acetele import config
from acetele import control
from acetele import runtime
from acetele import specification
from acetele.runtime.teleop import synchronization

_JOINT_NAMES = ("Joint_1", "Joint_2", "Joint_3", "Joint_4")
_CONFIG_RELATIVE_PATH = (
    pathlib.Path("acetele")
    / "config"
    / "presets"
    / "eaa_follower"
    / "fashionstar_rs485.toml"
)
_COMMAND_PERIOD_S = 0.04
_DISPLAY_PERIOD_S = 0.2


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parses the mock viewer's command-line options.

    Args:
        argv: Arguments without the program name, or None to use sys.argv.

    Returns:
        The parsed viewer options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1]
        / _CONFIG_RELATIVE_PATH,
        help="EAA RobotSpec TOML; its backend must be mock.",
    )
    parser.add_argument(
        "--urdf",
        type=pathlib.Path,
        help="Optional EAA URDF override for both display and mock control.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8081, help="HTTP port.")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def _reject_serial(
    port: str,
    baudrate: int,
    direction_control: object,
) -> typing.NoReturn:
    """Rejects any attempt to construct a physical serial transport.

    Args:
        port: The rejected serial port.
        baudrate: The unused baud rate.
        direction_control: The unused physical direction configuration.

    Raises:
        RuntimeError: Always, because this example is mock-only.
    """
    del baudrate, direction_control  # Unused; all serial access is rejected.
    raise RuntimeError(f"Physical serial access is disabled: {port}")


class _MockFollowerBridge:
    """Owns a mock follower session operated only by the main control loop.

    Attributes:
        spec: The validated mock EAA specification.
        group_name: The configured arm group receiving joint targets.
        session: The existing ACETele follower session.
        tracking: Whether the operator has explicitly started tracking.
    """

    def __init__(
        self,
        config_path: pathlib.Path,
        urdf_path: pathlib.Path | None = None,
    ) -> None:
        """Creates a mock-only bridge without connecting any device.

        Args:
            config_path: The EAA mock RobotSpec TOML file.
            urdf_path: Optional model override shared by control and display.

        Raises:
            ValueError: The specification is physical or incompatible.
        """
        spec = config.load_robot_spec(config_path.expanduser().resolve())
        if spec.backend != specification.Backend.MOCK:
            raise ValueError("This example requires backend = 'mock'")
        if spec.model != "eaa_follower" or len(spec.arms) != 1:
            raise ValueError("This example requires one EAA follower arm")
        if tuple(joint.name for joint in spec.arms[0].joints) != _JOINT_NAMES:
            raise ValueError(f"Expected joint order {_JOINT_NAMES}")
        if urdf_path is not None:
            spec = dataclasses.replace(
                spec,
                urdf_path=str(urdf_path.expanduser().resolve()),
            )
        self.spec = spec
        self.group_name = spec.arms[0].name
        robot = runtime.RobotRuntime(
            spec,
            transport_factory=_reject_serial,
            command_timeout_ns=500_000_000,
        )
        velocities = robot.preflight.arms[self.group_name].velocity_limits
        if not np.all(np.isfinite(velocities)) or np.any(
            np.asarray(velocities) <= 0.0
        ):
            raise ValueError("URDF velocity limits must be finite and positive")
        self.session = runtime.FollowerTeleopSession(
            robot,
            session_timeout_ns=500_000_000,
            motion_cycle_hz=100.0,
            motion_tuning=control.StreamingPositionTuning(
                velocity_limit_rad_s=0.2,
                acceleration_limit_rad_s2=0.5,
            ),
        )
        self.tracking = False

    def connect(self) -> np.ndarray:
        """Connects the in-memory follower and holds its first measured pose.

        Returns:
            Initial joint feedback in canonical EAA order, in radians.
        """
        self.session.connect()
        deadline = time.monotonic() + 2.0
        while True:
            try:
                positions = self.read_positions()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        self.session.hold_position()
        return positions

    def read_positions(self) -> np.ndarray:
        """Returns an independent copy of the latest mock joint feedback."""
        state = self.session.read(now_ns=time.monotonic_ns())
        return state.joints[self.group_name].positions.copy()

    def start_tracking(self) -> np.ndarray:
        """Synchronizes a new tracking cycle to the current feedback pose.

        Returns:
            The pose that must seed the virtual leader's sliders.
        """
        self.hold()
        positions = self.read_positions()
        self.session.set_mode(synchronization.LeaderSyncMode.SYNC_REQUEST)
        self.session.set_mode(synchronization.LeaderSyncMode.TRACKING)
        self.tracking = True
        return positions

    def hold(self) -> None:
        """Revokes command publication and puts the follower into HOLD."""
        self.tracking = False
        self.session.set_mode(synchronization.LeaderSyncMode.HOLD)

    def step(self, target: np.ndarray, client_connected: bool) -> np.ndarray:
        """Publishes the latest target only while an operator is tracking.

        Args:
            target: Latest requested joint positions, in canonical EAA order.
            client_connected: Whether a browser client remains connected.

        Returns:
            The latest mock follower joint positions in radians.
        """
        if self.tracking and not client_connected:
            self.hold()
        if self.tracking:
            try:
                accepted = self.session.write_arm(
                    _JOINT_NAMES, target, now_ns=time.monotonic_ns()
                )
            except (RuntimeError, ValueError):
                self.hold()
                raise
            if not accepted:
                self.hold()
                logging.warning("Tracking stopped; press Start to synchronize")
        return self.read_positions()

    def close(self) -> None:
        """Holds if connected, then disconnects the mock follower session."""
        self.tracking = False
        try:
            if self.session.runtime.connected:
                self.session.set_mode(synchronization.LeaderSyncMode.HOLD)
        finally:
            self.session.close()


class _ControlRequests:
    """Stores bounded UI requests with HOLD taking priority over START."""

    def __init__(self) -> None:
        """Initializes an empty, thread-safe request mailbox."""
        self._lock = threading.Lock()
        self._start = False
        self._hold = False

    def request_start(self, event: viser.GuiEvent | None = None) -> None:
        """Requests tracking without doing runtime work in a GUI callback.

        Args:
            event: Optional, unused Viser button event.
        """
        del event  # Unused; the main loop owns all session transitions.
        with self._lock:
            if not self._hold:
                self._start = True

    def request_hold(
        self,
        event: viser.GuiEvent | viser.ClientHandle | None = None,
    ) -> None:
        """Requests HOLD and invalidates pending starts after disconnection.

        Args:
            event: Optional, unused button event or disconnected client.
        """
        del event  # Unused; any client disconnect must revoke tracking.
        with self._lock:
            self._hold = True
            self._start = False

    def take(self) -> tuple[bool, bool]:
        """Returns and clears the pending start and hold request flags."""
        with self._lock:
            requests = (self._start, self._hold)
            self._start = False
            self._hold = False
        return requests


class _Viewer:
    """Displays virtual joint targets, operator controls, and mock feedback."""

    def __init__(
        self,
        server: viser.ViserServer,
        model: yourdfpy.URDF,
        requests: _ControlRequests,
    ) -> None:
        """Builds the viewer without owning any runtime or physical device.

        Args:
            server: The Viser server hosting the interface.
            model: The same EAA model used by the mock control runtime.
            requests: Thread-safe operator requests consumed by the main loop.
        """
        server.initial_camera.position = (0.9, 0.9, 0.7)
        server.initial_camera.look_at = (-0.1, 0.0, 0.15)
        self._robot = extras.ViserUrdf(server, model)
        self._sliders: list[viser.GuiSliderHandle] = []
        server.gui.add_markdown(
            "**Mock EAA follower — no hardware connection.**\n\n"
            "Start tracking synchronizes the sliders to current feedback. "
            "Then move the sliders. Hold or closing the page stops tracking."
        )
        self._start_button = server.gui.add_button("Start tracking")
        self._start_button.on_click(requests.request_start)
        server.gui.add_button("Hold").on_click(requests.request_hold)
        self._feedback = server.gui.add_markdown("Connecting mock follower...")
        limits = self._robot.get_actuated_joint_limits()
        for name, (lower, upper) in limits.items():
            self._sliders.append(
                server.gui.add_slider(
                    f"{name} (rad)",
                    min=float(lower),
                    max=float(upper),
                    step=0.01,
                    initial_value=float(np.clip(0.0, lower, upper)),
                )
            )
        for slider in self._sliders:
            slider.on_update(self._update_pose)
        server.on_client_disconnect(requests.request_hold)

    def target_positions(self) -> np.ndarray:
        """Returns a new array containing the latest slider targets."""
        return np.asarray([slider.value for slider in self._sliders])

    def set_target(self, positions: np.ndarray) -> None:
        """Synchronizes all virtual targets to one measured follower pose.

        Args:
            positions: Feedback positions in canonical EAA order, in radians.
        """
        for slider, position in zip(self._sliders, positions, strict=True):
            slider.value = float(position)
        self._update_pose()

    def _update_pose(self, event: viser.GuiEvent | None = None) -> None:
        """Updates the virtual pose when a slider changes.

        Args:
            event: Optional, unused slider event.
        """
        del event  # Unused; always read the complete latest target.
        self._robot.update_cfg(self.target_positions())

    def show_feedback(
        self,
        positions: np.ndarray,
        bridge: _MockFollowerBridge,
    ) -> None:
        """Displays targets, feedback, synchronization, and safety state.

        Args:
            positions: The latest mock follower feedback in radians.
            bridge: The bridge whose current tracking state is displayed.
        """
        targets = self.target_positions()
        safety = bridge.session.runtime.diagnostics().safety.state.value
        rows = [
            f"**State: {safety.upper()}** | "
            f"Session: {bridge.session.status.value.upper()}",
            "",
            "| Joint | Target (rad) | Feedback (rad) |",
            "| --- | ---: | ---: |",
        ]
        for name, target, position in zip(
            _JOINT_NAMES, targets, positions, strict=True
        ):
            rows.append(f"| {name} | {target:.3f} | {position:.3f} |")
        self._feedback.content = "\n".join(rows)
        self._start_button.disabled = bridge.tracking


def _run_viewer(
    server: viser.ViserServer,
    bridge: _MockFollowerBridge,
    positions: np.ndarray,
) -> None:
    """Runs target publication at approximately 25 Hz until interrupted.

    Args:
        server: The running Viser server.
        bridge: A connected mock follower initially holding its pose.
        positions: Initial feedback used to seed the virtual leader.
    """
    model = yourdfpy.URDF.load(str(bridge.spec.urdf_path))
    requests = _ControlRequests()
    viewer = _Viewer(server, model, requests)
    viewer.set_target(positions)
    next_display = 0.0
    while True:
        was_tracking = bridge.tracking
        start_requested, hold_requested = requests.take()
        connected = bool(server.get_clients())
        if hold_requested:
            bridge.hold()
        elif start_requested and connected:
            viewer.set_target(bridge.start_tracking())
        positions = bridge.step(viewer.target_positions(), connected)
        if was_tracking and not bridge.tracking:
            viewer.set_target(positions)
        if time.monotonic() >= next_display:
            viewer.show_feedback(positions, bridge)
            if bridge.tracking:
                logging.info(
                    "Target=%s feedback=%s",
                    np.round(viewer.target_positions(), 3).tolist(),
                    np.round(positions, 3).tolist(),
                )
            next_display = time.monotonic() + _DISPLAY_PERIOD_S
        time.sleep(_COMMAND_PERIOD_S)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the mock bridge and cleans up the session on every exit path.

    Args:
        argv: Arguments without the program name, or None to use sys.argv.
    """
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    bridge = _MockFollowerBridge(args.config, args.urdf)
    server = None
    try:
        positions = bridge.connect()
        server = viser.ViserServer(host=args.host, port=args.port)
        logging.info("Viewer URL: http://%s:%s/", args.host, server.get_port())
        logging.info("Mock mode: velocity=0.2 rad/s; acceleration=0.5 rad/s^2")
        logging.info("Click Start tracking in the browser; Ctrl+C stops.")
        _run_viewer(server, bridge, positions)
    except KeyboardInterrupt:
        logging.info("Stopping the virtual leader and mock follower.")
    finally:
        try:
            bridge.close()
        finally:
            if server is not None:
                server.stop()


if __name__ == "__main__":
    main()
