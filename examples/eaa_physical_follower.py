"""Controls one physical EAA follower from local Viser joint sliders.

Without --connect, this example performs static preflight only. Connecting
requires terminal confirmation, briefly releases torque during the normal
FashionStar handshake, then holds the measured pose. Tracking requires an
explicit browser Start and an initial pose inside the unchanged URDF limits.
Gravity sag outside those limits can be held, but never silently clipped into
a motion command. Support and position the arm inside its permitted range
before connecting when tracking is required.

Hold and browser disconnection revoke tracking. Normal exit preserves HOLD;
Stop explicitly requests torque release. An independent physical emergency
stop remains necessary. The existing mock example is separate and unchanged.
Feedback failures pause tracking while keeping the viewer available. Three
distinct fresh samples permit a new manual Start; hardware faults stay latched.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import enum
import logging
import pathlib
import threading
import time

import numpy as np
import viser
from viser import extras
import yourdfpy

from acetele import config
from acetele import control
from acetele import runtime
from acetele import specification
from acetele.hardware.devices.adapter import AdapterRegistry, default_adapter_registry
from acetele.hardware.devices.servos.fashionstar.adapter import FashionStarAdapter
from acetele.runtime import safety
from acetele.runtime.teleop import synchronization

_JOINT_NAMES = ("Joint_1", "Joint_2", "Joint_3", "Joint_4")
_COMMAND_PERIOD_S = 0.04
_DISPLAY_PERIOD_S = 0.2
_LOG_PERIOD_S = 1.0
_MAX_STATE_AGE_NS = 250_000_000
_IO_TIMEOUT_S = 0.15
_RECOVERY_SAMPLES = 3


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parses explicit physical connection and local viewer options.

    Args:
        argv: Arguments without the program name, or None for sys.argv.

    Returns:
        The validated CLI options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("eaa_follower_physical.toml"),
        help="Physical EAA RobotSpec TOML file.",
    )
    parser.add_argument(
        "--connect",
        action="store_true",
        help="Prompt before hardware connection; otherwise preflight only.",
    )
    parser.add_argument(
        "--port", type=int, default=8082, help="Local HTTP port."
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


class _PhysicalFollowerBridge:
    """Owns one physical session without opening hardware in its constructor.

    Attributes:
        spec: The validated physical EAA configuration.
        group_name: The single controlled arm group.
        session: The existing ACETele follower session.
        lower: Canonical URDF lower position limits in radians.
        upper: Canonical URDF upper position limits in radians.
        tracking: Whether this operator has explicitly started tracking.
    """

    def __init__(self, config_path: pathlib.Path) -> None:
        """Checks the physical plan and constructs an unconnected session.

        Args:
            config_path: Physical EAA RobotSpec TOML file.

        Raises:
            ValueError: The plan is incompatible or fails static preflight.
        """
        self.spec = config.load_robot_spec(config_path.expanduser().resolve())
        if self.spec.backend != specification.Backend.PHYSICAL:
            raise ValueError("This example requires backend = 'physical'")
        if self.spec.model != "eaa_follower" or len(self.spec.arms) != 1:
            raise ValueError("This example requires one EAA follower arm")
        arm = self.spec.arms[0]
        if tuple(joint.name for joint in arm.joints) != _JOINT_NAMES:
            raise ValueError(f"Expected joint order {_JOINT_NAMES}")
        if arm.end_effector is not None or len(self.spec.buses) != 1:
            raise ValueError("Only the four-joint EAA arm bus is supported")
        bus = self.spec.buses[0]
        if bus.type != specification.BusType.FASHIONSTAR_RS485:
            raise ValueError("The EAA bus must use fashionstar_rs485")
        defaults = default_adapter_registry()
        adapters = [FashionStarAdapter(operation_timeout_s=_IO_TIMEOUT_S)]
        for kind in specification.BusType:
            if kind != specification.BusType.FASHIONSTAR_RS485:
                adapter = defaults.require(kind)
                if all(existing is not adapter for existing in adapters):
                    adapters.append(adapter)
        robot = runtime.RobotRuntime(
            self.spec,
            command_timeout_ns=500_000_000,
            adapter_registry=AdapterRegistry(adapters),
        )
        self.group_name = arm.name
        metadata = robot.preflight.arms[self.group_name]
        self.lower = np.asarray(metadata.lower_limits)
        self.upper = np.asarray(metadata.upper_limits)
        velocities = np.asarray(metadata.velocity_limits)
        if not np.all(np.isfinite(velocities)) or np.any(velocities <= 0.0):
            raise ValueError("URDF velocity limits must be finite and positive")
        self.session = runtime.FollowerTeleopSession(
            robot,
            session_timeout_ns=500_000_000,
            motion_cycle_hz=100.0,
            motion_tuning=control.StreamingPositionTuning(
                velocity_limit_rad_s=1.0,
                acceleration_limit_rad_s2=0.5,
            ),
        )
        self.tracking = False
        self._closed = True
        self._last_positions: np.ndarray | None = None
        self._last_feedback_ns: int | None = None
        self._feedback_error: str | None = None
        self._recovery_samples = 0
        self._recovery_timestamp_ns: int | None = None

    @property
    def faulted(self) -> bool:
        """Keep runtime and actor faults latched until explicit hardware recovery."""
        diagnostics = self.session.runtime.diagnostics()
        return diagnostics.safety.state == safety.RuntimeSafetyState.FAULT or any(
            bus.fault is not None for bus in diagnostics.buses.values()
        )

    @property
    def feedback_ready(self) -> bool:
        """Allow Start only after fresh feedback has recovered without a fault."""
        return (
            self._last_positions is not None
            and self._feedback_error is None
            and not self.faulted
        )

    @property
    def feedback_error(self) -> str | None:
        return self._feedback_error

    @property
    def display_positions(self) -> np.ndarray:
        """Return the last valid sample for display, never for motion admission."""
        if self._last_positions is None:
            raise RuntimeError("No valid follower sample is available for display")
        return self._last_positions.copy()

    def pause_tracking(self, reason: str) -> None:
        """Invalidate targets and request HOLD without reconnecting hardware."""
        changed = self._feedback_error != reason
        self._feedback_error = reason
        self._recovery_samples = 0
        self._recovery_timestamp_ns = self._last_feedback_ns
        self.tracking = False
        try:
            self.hold()
        except (RuntimeError, ValueError, OSError) as exc:
            if changed:
                logging.warning("Tracking paused; HOLD could not be confirmed: %s", exc)

    def connect(self) -> np.ndarray:
        """Connects and holds the measured pose without admitting tracking.

        Returns:
            Initial measured positions, including any gravity sag.
        """
        if not self._closed:
            raise RuntimeError("The follower is already connected")
        self._closed = False
        try:
            self.session.connect()
            deadline = time.monotonic() + 2.0
            while True:
                try:
                    self.read_positions()
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
            self.session.hold_position()
            return self.read_positions()
        except BaseException:
            self.close()
            raise

    def read_positions(self) -> np.ndarray:
        """Returns finite, fresh measured feedback in canonical joint order.

        Returns:
            An independent array of four joint positions in radians.

        Raises:
            RuntimeError: Feedback is stale, invalid, or the worker failed.
        """
        state = self.session.read(now_ns=time.monotonic_ns())
        joints = state.joints[self.group_name]
        age_ns = time.monotonic_ns() - joints.timestamp_ns
        if not 0 <= age_ns <= _MAX_STATE_AGE_NS:
            raise RuntimeError("Follower feedback is stale; tracking stopped")
        positions = joints.positions.copy()
        if positions.shape != (4,) or not np.all(np.isfinite(positions)):
            raise RuntimeError("Follower feedback is invalid; tracking stopped")
        self._last_positions = positions.copy()
        self._last_feedback_ns = joints.timestamp_ns
        if self._feedback_error is not None and not self.faulted:
            if (
                self._recovery_timestamp_ns is None
                or joints.timestamp_ns > self._recovery_timestamp_ns
            ):
                self._recovery_samples += 1
                self._recovery_timestamp_ns = joints.timestamp_ns
            if self._recovery_samples >= _RECOVERY_SAMPLES:
                self._feedback_error = None
                logging.info("Fresh feedback recovered; click Start to resume tracking.")
        return positions

    def limit_violations(self, positions: np.ndarray) -> str:
        """Describes measured joints outside the unchanged model limits.

        Args:
            positions: Canonical measured positions in radians.

        Returns:
            A readable list, or an empty string when all joints are inside.
        """
        outside = (positions < self.lower) | (positions > self.upper)
        return "; ".join(
            f"{name}: {np.degrees(position):.1f} deg"
            for name, position, invalid in zip(
                _JOINT_NAMES, positions, outside, strict=True
            )
            if invalid
        )

    def start_tracking(self) -> np.ndarray:
        """Starts a fresh cycle only from a legal measured holding pose.

        Returns:
            The measured pose that must seed every virtual leader slider.

        Raises:
            ValueError: The measured pose is outside the URDF limits.
        """
        self.hold()
        if self.faulted:
            raise RuntimeError("Follower is faulted; inspect hardware before restarting")
        positions = self.read_positions()
        if not self.feedback_ready:
            raise RuntimeError("Feedback recovery pending; wait for fresh samples before Start")
        violations = self.limit_violations(positions)
        if violations:
            raise ValueError(f"Tracking blocked by URDF limits: {violations}")
        self.session.set_mode(synchronization.LeaderSyncMode.SYNC_REQUEST)
        self.session.set_mode(synchronization.LeaderSyncMode.TRACKING)
        self.tracking = True
        return positions

    def hold(self) -> None:
        """Invalidates streaming targets while maintaining actuator support."""
        self.tracking = False
        self.session.set_mode(synchronization.LeaderSyncMode.HOLD)

    def stop(self) -> None:
        """Latches STOP and requests torque release; the arm may fall."""
        self.tracking = False
        self.session.set_mode(synchronization.LeaderSyncMode.STOP)

    def step(self, target: np.ndarray, client_connected: bool) -> np.ndarray:
        """Publishes complete targets only during a healthy tracking cycle.

        Args:
            target: Four canonical virtual leader positions in radians.
            client_connected: Whether a browser client is connected.

        Returns:
            Current measured joint feedback.

        Raises:
            ValueError: A tracking target is incomplete or outside its limits.
            RuntimeError: Feedback or the local motion worker has failed.
        """
        try:
            positions = self.read_positions()
            if self.tracking and not client_connected:
                self.hold()
            if self.tracking:
                violations = self.limit_violations(positions)
                if violations:
                    raise ValueError(
                        "Feedback left URDF limits; tracking stopped: "
                        f"{violations}"
                    )
                if self.session.status not in (
                    synchronization.FollowerSyncStatus.READY,
                    synchronization.FollowerSyncStatus.TRACKING,
                ):
                    self.hold()
                    return positions
                target = np.asarray(target, dtype=float)
                if (
                    target.shape != (4,)
                    or not np.all(np.isfinite(target))
                    or np.any(target < self.lower)
                    or np.any(target > self.upper)
                ):
                    raise ValueError("Invalid or out-of-limit joint target")
                accepted = self.session.write_arm(
                    _JOINT_NAMES, target, now_ns=time.monotonic_ns()
                )
                if not accepted:
                    self.hold()
                    raise RuntimeError("Target rejected; synchronize again")
            return positions
        except (RuntimeError, OSError) as exc:
            self.pause_tracking(str(exc))
            raise
        except ValueError:
            self.hold()
            raise

    def close(self) -> None:
        """Stops workers and closes serial while preserving the holding goal."""
        if self._closed:
            return
        self._closed = True
        self.tracking = False
        try:
            state = self.session.runtime.diagnostics().safety.state
            if state in (
                safety.RuntimeSafetyState.READY,
                safety.RuntimeSafetyState.ACTIVE,
                safety.RuntimeSafetyState.HOLD,
            ):
                self.hold()
        finally:
            self.session.close(preserve_hold=True)


class _Request(enum.IntEnum):
    """Defines a bounded priority order for operator requests."""

    NONE = 0
    START = 1
    HOLD = 2
    STOP = 3


class _Requests:
    """Queues one operator request without performing I/O in GUI callbacks."""

    def __init__(self) -> None:
        """Creates an empty mailbox."""
        self._lock = threading.Lock()
        self._pending = _Request.NONE

    def submit(self, request: _Request) -> None:
        """Keeps the highest-priority request until the main loop consumes it.

        Args:
            request: An explicit operator or disconnect request.
        """
        with self._lock:
            self._pending = max(self._pending, request)

    def take(self) -> _Request:
        """Returns and clears the pending operator request."""
        with self._lock:
            request = self._pending
            self._pending = _Request.NONE
            return request


class _Viewer:
    """Displays measured feedback and explicitly armed virtual targets."""

    def __init__(
        self,
        server: viser.ViserServer,
        bridge: _PhysicalFollowerBridge,
        requests: _Requests,
    ) -> None:
        """Builds local controls without performing hardware transactions.

        Args:
            server: The local Viser server.
            bridge: The connected physical follower bridge.
            requests: Operator mailbox consumed by the control loop.
        """
        server.initial_camera.position = (0.9, 0.9, 0.7)
        server.initial_camera.look_at = (-0.1, 0.0, 0.15)
        model = yourdfpy.URDF.load(str(bridge.spec.urdf_path))
        self._robot = extras.ViserUrdf(server, model)
        if tuple(self._robot.get_actuated_joint_limits()) != _JOINT_NAMES:
            raise ValueError("Viewer joint order differs from the EAA runtime")
        server.gui.add_markdown(
            "**Physical EAA follower**\n\n"
            "Starts in HOLD. Start tracking seeds targets from feedback. "
            "Target rate limit: 0.05 rad/s. Closing this page requests HOLD. "
            "ACTIVE indicates software command admission; verify feedback. "
            "Stop requests torque release and may let the arm fall. "
            "Use the independent hardware emergency stop when needed."
        )
        self._start = server.gui.add_button("Start tracking")
        self._start.on_click(lambda _: requests.submit(_Request.START))
        server.gui.add_button("Hold").on_click(
            lambda _: requests.submit(_Request.HOLD)
        )
        server.gui.add_button("Stop / request torque release").on_click(
            lambda _: requests.submit(_Request.STOP)
        )
        self._feedback = server.gui.add_markdown("Reading follower state...")
        self._sliders = [
            server.gui.add_slider(
                f"{name} (rad)",
                min=float(lower),
                max=float(upper),
                step=0.001,
                initial_value=float(np.clip(0.0, lower, upper)),
                disabled=True,
            )
            for name, lower, upper in zip(
                _JOINT_NAMES, bridge.lower, bridge.upper, strict=True
            )
        ]
        for slider in self._sliders:
            slider.on_update(self._update_pose)
        server.on_client_disconnect(
            lambda _: requests.submit(_Request.HOLD)
        )

    def targets(self) -> np.ndarray:
        """Returns the complete virtual leader target in canonical order."""
        return np.asarray([slider.value for slider in self._sliders])

    def seed(self, positions: np.ndarray) -> None:
        """Sets sliders from a measured pose already verified within limits.

        Args:
            positions: Canonical joint feedback in radians.
        """
        for slider, position in zip(self._sliders, positions, strict=True):
            slider.value = float(position)
        self._update_pose()

    def _update_pose(self, event: viser.GuiEvent | None = None) -> None:
        """Draws the current virtual leader target.

        Args:
            event: Optional, unused slider update event.
        """
        del event
        self._robot.update_cfg(self.targets())

    def refresh(
        self,
        positions: np.ndarray,
        bridge: _PhysicalFollowerBridge,
        message: str = "",
    ) -> None:
        """Displays feedback while keeping sagging startup poses unmodified.

        Args:
            positions: Fresh measured follower positions in radians.
            bridge: The bridge providing tracking and safety state.
            message: Optional explanation of a rejected operator request.
        """
        violations = bridge.limit_violations(positions)
        if not bridge.tracking:
            if not violations and bridge.feedback_ready:
                self.seed(positions)
            else:
                self._robot.update_cfg(positions)
        self._start.disabled = (
            bridge.tracking or bool(violations) or not bridge.feedback_ready
        )
        for slider in self._sliders:
            slider.disabled = not bridge.tracking
        state = bridge.session.runtime.diagnostics().safety.state.value
        rows = [f"**State: {state.upper()}**", ""]
        if bridge.feedback_error:
            rows += [
                "**Tracking paused: feedback unavailable or recovering.**",
                bridge.feedback_error,
                "Displayed feedback is the last valid sample. "
                "Start becomes available after fresh feedback recovers. "
                "A latched hardware fault requires explicit inspection and recovery.",
                "",
            ]
        if violations:
            rows += [
                f"Measured pose outside URDF limits: {violations}.",
                "Tracking remains blocked; the controller requests HOLD. "
                "Support and position the arm within its permitted range "
                "before connecting for tracking.",
                "",
            ]
        if message:
            rows += [message, ""]
        rows += [
            "| Joint | Target (deg) | Feedback (deg) |",
            "| --- | ---: | ---: |",
        ]
        for name, target, position in zip(
            _JOINT_NAMES, self.targets(), positions, strict=True
        ):
            goal = (
                f"{np.degrees(target):.2f}"
                if bridge.tracking
                else "PAUSED" if bridge.feedback_error else "HOLD"
            )
            rows.append(f"| {name} | {goal} | {np.degrees(position):.2f} |")
        self._feedback.content = "\n".join(rows)


def _log_tracking(
    bridge: _PhysicalFollowerBridge,
    positions: np.ndarray,
    targets: np.ndarray,
) -> None:
    """Logs software targets and serial write evidence without new bus I/O.

    Args:
        bridge: The connected physical follower bridge.
        positions: The latest measured joint feedback in radians.
        targets: The current requested joint positions in radians.
    """
    diagnostics = bridge.session.runtime.diagnostics()
    now_ns = time.monotonic_ns()
    for name, bus in diagnostics.buses.items():
        tx_age_ms = (
            None
            if bus.last_motion_success_ns is None
            else (now_ns - bus.last_motion_success_ns) / 1e6
        )
        logging.info(
            "Target_deg=%s Feedback_deg=%s bus=%s "
            "serial_tx_samples=%s last_tx_age_ms=%s "
            "state_age_ms=%s last_read_ms=%s state_hz=%s "
            "io_errors=%s expired=%s fault=%s",
            np.round(np.degrees(targets), 2).tolist(),
            np.round(np.degrees(positions), 2).tolist(),
            name,
            bus.motion_end_to_end_sample_count,
            tx_age_ms,
            None if bus.state_age_s is None else round(bus.state_age_s * 1000, 2),
            (
                None
                if bus.last_fast_read_s is None
                else round(bus.last_fast_read_s * 1000, 2)
            ),
            None if bus.state_frequency_hz is None else round(bus.state_frequency_hz, 2),
            bus.recoverable_error_count,
            bus.expired_motion_count,
            bus.fault,
        )


def _confirm_connection(bridge: _PhysicalFollowerBridge) -> bool:
    """Requires a deliberate terminal confirmation before hardware access.

    Args:
        bridge: The statically validated physical plan.

    Returns:
        Whether the exact connection phrase was entered.
    """
    bus = bridge.spec.buses[0]
    print(
        "Confirm EAA follower, power, clearance and independent emergency stop."
    )
    print(
        "Support the arm: connection briefly releases torque, then enters HOLD."
    )
    print("Product model identity is not verified by the generic protocol.")
    print("Normal program exit preserves HOLD; Stop requests torque release.")
    phrase = f"CONNECT {bus.port}"
    try:
        return input(f"Type {phrase} to connect: ").strip() == phrase
    except (EOFError, KeyboardInterrupt):
        return False


def _run_viewer(
    server: viser.ViserServer,
    bridge: _PhysicalFollowerBridge,
) -> None:
    """Runs the operator loop until interruption or an explicit STOP.

    Args:
        server: The already listening local viewer server.
        bridge: The connected physical follower holding its measured pose.
    """
    requests = _Requests()
    viewer = _Viewer(server, bridge, requests)
    viewer.refresh(bridge.display_positions, bridge)
    next_display = 0.0
    next_log = 0.0
    message = ""
    last_error = ""
    recovery_pending = False
    while True:
        request = requests.take()
        connected = bool(server.get_clients())
        try:
            if request == _Request.STOP:
                bridge.stop()
                return
            if request == _Request.HOLD:
                bridge.hold()
                message = "Holding. Reconnection requires another explicit Start."
            elif request == _Request.START and connected:
                viewer.seed(bridge.start_tracking())
                message = ""
            positions = bridge.step(viewer.targets(), connected)
        except (RuntimeError, ValueError, OSError) as exc:
            message = str(exc)
            if isinstance(exc, (RuntimeError, OSError)):
                bridge.pause_tracking(message)
                recovery_pending = True
            if message != last_error:
                logging.warning("Tracking paused: %s", message)
                last_error = message
            positions = bridge.display_positions
        else:
            if recovery_pending and bridge.feedback_ready:
                message = "Feedback recovered. Click Start to resume tracking."
                last_error = ""
                recovery_pending = False
        if time.monotonic() >= next_display:
            viewer.refresh(positions, bridge, message)
            next_display = time.monotonic() + _DISPLAY_PERIOD_S
        if bridge.tracking and time.monotonic() >= next_log:
            _log_tracking(bridge, positions, viewer.targets())
            next_log = time.monotonic() + _LOG_PERIOD_S
        time.sleep(_COMMAND_PERIOD_S)


def main(argv: Sequence[str] | None = None) -> int:
    """Runs static preflight, optionally connecting after local confirmation.

    Args:
        argv: Arguments without the program name, or None for sys.argv.

    Returns:
        Zero on success, one on runtime failure, or 130 on cancellation.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    bridge = _PhysicalFollowerBridge(args.config)
    bus = bridge.spec.buses[0]
    print(f"Config: {args.config.resolve()}")
    print(f"EAA follower: {bus.port}, {bus.baudrate} baud")
    print(
        f"Read budget: {_IO_TIMEOUT_S * 1000:.0f} ms; "
        f"feedback age limit: {_MAX_STATE_AGE_NS / 1e6:.0f} ms"
    )
    for joint in bridge.spec.arms[0].joints:
        print(
            f"  {joint.name}: ID {joint.servo_id}, {joint.servo_model}, "
            f"firmware {joint.firmware_version}, direction {joint.direction:+d}"
        )
    budget = bridge.session.runtime.preflight.buses[bus.name].budget
    print(f"Static bus utilization: {budget.utilization:.1%}")
    if not args.connect:
        print("Static preflight passed. No serial port was opened.")
        print("Use --connect only after reviewing the physical setup.")
        return 0
    if not _confirm_connection(bridge):
        print("Cancelled. No serial port was opened.")
        return 130
    server = None
    exit_code = 0
    try:
        # Reserve the HTTP port before any torque-affecting handshake occurs.
        server = viser.ViserServer(host="127.0.0.1", port=args.port)
        bridge.connect()
        logging.info("Viewer URL: http://127.0.0.1:%s/", server.get_port())
        logging.info("Holding measured pose. Click Start for tracking.")
        _run_viewer(server, bridge)
    except KeyboardInterrupt:
        logging.info("Stopping target publication; preserving HOLD.")
    except (RuntimeError, ValueError, OSError):
        logging.exception("Physical follower stopped; inspect hardware state.")
        exit_code = 1
    finally:
        try:
            bridge.close()
        finally:
            if server is not None:
                server.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
