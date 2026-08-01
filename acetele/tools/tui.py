"""Curses front end for launch generation and FEETECH calibration.

The terminal renderer is intentionally thin. Configuration discovery, command
generation, calibration planning, and state persistence remain ordinary Python so
they can be tested without a terminal or hardware.
"""

from __future__ import annotations

import curses
import json
import math
import os
import shlex
import sys
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from acetele.config import load_robot_spec, packaged_robot_specs
from acetele.runtime import RobotRuntime, calibrate_feetech_home
from acetele.specification import Backend, BusType, ParallelGripperSpec, RobotSpec


class TuiWorkflow(str, Enum):
    """Operations exposed by the unified terminal front end."""

    ROS2_LAUNCH = "ros2_launch"
    ZMQ_LAUNCH = "zmq_launch"
    CALIBRATE = "calibrate"


def _default_xrce_prefix() -> Path:
    configured = os.environ.get("ACETELE_XRCE_PREFIX")
    if configured:
        return Path(configured)
    return Path.home() / ".local" / "lib" / "acetele" / "xrce-2.4.2"


@dataclass(frozen=True)
class RobotSpecChoice:
    """One validated RobotSpec together with display-only preflight metadata."""

    path: Path
    spec: RobotSpec
    summary: tuple[str, ...]
    resource_id: Optional[str] = None

    @property
    def label(self) -> str:
        robot = self.spec.model.replace("_", " ").upper()
        if self.resource_id is None:
            return f"{robot} | Custom: {self.path.name}"
        profile = Path(self.resource_id).stem.replace("_", " ").replace("-", " ").upper()
        return f"{robot} | {profile}"


@dataclass(frozen=True)
class LaunchSelection:
    """Validated runtime-only parameters used to generate a ROS 2 command."""

    choice: RobotSpecChoice
    teleop_mode: str = "joint"
    translation_scale: float = 2.0
    rotation_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.teleop_mode not in ("joint", "ee_pose"):
            raise ValueError("teleop mode must be 'joint' or 'ee_pose'")
        for name, value in (
            ("translation scale", self.translation_scale),
            ("rotation scale", self.rotation_scale),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name.replace(" ", "_"), normalized)


@dataclass(frozen=True)
class ZmqLaunchSelection:
    """Validated direct-peer parameters used to generate a ZMQ command."""

    choice: RobotSpecChoice
    role: str
    peer_host: str
    bind_host: str = "0.0.0.0"
    command_port: int = 5555
    state_port: int = 5556
    teleop_mode: str = "joint"
    translation_scale: float = 2.0
    rotation_scale: float = 1.0
    curve_secret_key: Optional[Path] = None
    curve_peer_key: Optional[Path] = None
    xrce_prefix: Path = field(default_factory=_default_xrce_prefix)
    xrce_agent_port: int = 8888
    xrce_domain_id: int = 0
    xrce_namespace: str = ""
    xrce_client_key: int = 0xACED0001
    xrce_startup_timeout: float = 3.0

    def __post_init__(self) -> None:
        if self.role not in ("leader", "follower"):
            raise ValueError("ZMQ role must be 'leader' or 'follower'")
        expected_model = f"ace_{self.role}"
        if self.choice.spec.model != expected_model:
            raise ValueError(
                f"ZMQ {self.role} requires a {expected_model} RobotSpec"
            )
        if self.teleop_mode not in ("joint", "ee_pose"):
            raise ValueError("teleop mode must be 'joint' or 'ee_pose'")
        for name in ("peer_host", "bind_host"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "://" in value:
                raise ValueError(f"{name} must be a non-empty host without URI scheme")
        for name in ("command_port", "state_port"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 65_535:
                raise ValueError(f"{name} must be an integer in [1, 65535]")
        if self.command_port == self.state_port:
            raise ValueError("ZMQ command and state ports must differ")
        for name in ("translation_scale", "rotation_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, normalized)
        key_paths = (self.curve_secret_key, self.curve_peer_key)
        if any(path is None for path in key_paths) and not all(
            path is None for path in key_paths
        ):
            raise ValueError("both CURVE key paths must be supplied together")
        for name in ("curve_secret_key", "curve_peer_key"):
            path = getattr(self, name)
            if path is not None:
                object.__setattr__(self, name, Path(path).expanduser().resolve())
        object.__setattr__(
            self,
            "xrce_prefix",
            Path(self.xrce_prefix).expanduser().resolve(),
        )
        if type(self.xrce_agent_port) is not int or not 1 <= self.xrce_agent_port <= 65_535:
            raise ValueError("XRCE agent port must be an integer in [1, 65535]")
        if type(self.xrce_domain_id) is not int or not 0 <= self.xrce_domain_id <= 232:
            raise ValueError("XRCE domain ID must be an integer in [0, 232]")
        if not isinstance(self.xrce_namespace, str):
            raise ValueError("XRCE namespace must be a string")
        namespace = self.xrce_namespace.strip("/")
        if "//" in namespace or len(namespace) > 64:
            raise ValueError("XRCE namespace must be at most 64 characters")
        object.__setattr__(self, "xrce_namespace", namespace)
        if (
            type(self.xrce_client_key) is not int
            or not 1 <= self.xrce_client_key <= 0xFFFFFFFF
            or self.xrce_client_key == 1
        ):
            raise ValueError(
                "XRCE client key must be non-zero and differ from PX4's default key 1"
            )
        if (
            isinstance(self.xrce_startup_timeout, bool)
            or not isinstance(self.xrce_startup_timeout, (int, float))
            or not math.isfinite(float(self.xrce_startup_timeout))
            or self.xrce_startup_timeout <= 0.0
        ):
            raise ValueError("XRCE startup timeout must be finite and positive")
        object.__setattr__(
            self,
            "xrce_startup_timeout",
            float(self.xrce_startup_timeout),
        )


@dataclass(frozen=True)
class CalibrationJointPlan:
    """One desired home coordinate passed to FEETECH offset calibration."""

    bus_name: str
    port: str
    joint_name: str
    servo_id: int
    servo_model: str
    direction: int
    home_position_rad: float
    target_raw_position: int


@dataclass(frozen=True)
class CalibrationPlan:
    """Complete all-or-nothing static plan for one RobotSpec calibration."""

    choice: RobotSpecChoice
    joints: tuple[CalibrationJointPlan, ...]


@dataclass(frozen=True)
class TuiResult:
    """Action returned after curses has restored the caller's terminal."""

    workflow: TuiWorkflow
    launch: Optional[LaunchSelection] = None
    zmq_launch: Optional[ZmqLaunchSelection] = None
    calibration: Optional[CalibrationPlan] = None


class TuiStateStore:
    """Versioned best-effort persistence for recent TUI selections."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or self.default_path()

    @staticmethod
    def default_path() -> Path:
        root = os.environ.get("XDG_STATE_HOME")
        state_root = Path(root).expanduser() if root else Path.home() / ".local" / "state"
        return state_root / "acetele" / "tui.json"

    def load(self) -> dict[str, Any]:
        """Return validated top-level state or an empty state on any corruption."""

        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        if not isinstance(document, dict) or document.get("version") != 1:
            return {}
        return document

    def save(self, document: Mapping[str, Any]) -> Optional[str]:
        """Atomically persist state and return a warning instead of blocking work."""

        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(document)
            payload["version"] = 1
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return f"could not save TUI state: {exc}"
        return None


def inspect_robot_spec(
    path: str | Path,
    *,
    resource_id: Optional[str] = None,
) -> RobotSpecChoice:
    """Load one spec and complete hardware-free runtime preflight."""

    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".toml":
        raise ValueError("RobotSpec path must end in .toml")
    if not resolved.is_file():
        raise FileNotFoundError(f"RobotSpec file does not exist: {resolved}")
    spec = load_robot_spec(resolved)
    runtime = RobotRuntime(spec)
    summary = [
        f"robot={spec.model} backend={spec.backend.value}",
        f"urdf={runtime.preflight.urdf_path}",
    ]
    for name, bus in runtime.preflight.buses.items():
        summary.append(
            f"bus={name} type={bus.spec.type.value} port={bus.spec.port} "
            f"rate={bus.spec.cycle_hz:g}Hz utilization={bus.budget.utilization:.1%}"
        )
        summary.append(
            f"  verified_disable={str(bus.supports_verified_disable).lower()} "
            f"verified_identity={str(bus.supports_verified_identity).lower()} "
            f"external_estop={str(bus.spec.external_estop).lower()}"
        )
    return RobotSpecChoice(resolved, spec, tuple(summary), resource_id)


def discover_packaged_robot_specs() -> tuple[RobotSpecChoice, ...]:
    """Preflight every RobotSpec exposed by the shared package catalog."""

    choices: list[RobotSpecChoice] = []
    for resource in packaged_robot_specs():
        try:
            choices.append(
                inspect_robot_spec(
                    resource.path,
                    resource_id=resource.resource_id,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"packaged RobotSpec '{resource.resource_id}' failed preflight"
            ) from exc
    return tuple(choices)


def build_ros_launch_command(selection: LaunchSelection) -> str:
    """Return one shell-safe command without executing ROS or opening hardware."""

    arguments = [
        "ros2",
        "launch",
        "ace_robot_ros2",
        "ace_robot.launch.py",
        f"config_path:={selection.choice.path}",
        f"teleop_mode:={selection.teleop_mode}",
    ]
    if selection.teleop_mode == "ee_pose":
        arguments.extend(
            (
                f"translation_scale:={selection.translation_scale}",
                f"rotation_scale:={selection.rotation_scale}",
            )
        )
    return shlex.join(arguments)


def build_zmq_launch_command(selection: ZmqLaunchSelection) -> str:
    """Return a shell-safe direct-peer command without importing the ZMQ adapter."""

    arguments = [
        sys.executable,
        "-m",
        "ace_robot_zmq",
        selection.role,
        "--config",
        str(selection.choice.path),
        "--peer-host",
        selection.peer_host,
        "--bind-host",
        selection.bind_host,
        "--command-port",
        str(selection.command_port),
        "--state-port",
        str(selection.state_port),
        "--teleop-mode",
        selection.teleop_mode,
    ]
    if selection.teleop_mode == "ee_pose" and selection.role == "follower":
        arguments.extend(
            (
                "--translation-scale",
                str(selection.translation_scale),
                "--rotation-scale",
                str(selection.rotation_scale),
            )
        )
    if selection.role == "follower":
        arguments.extend(
            (
                "--xrce-prefix",
                str(selection.xrce_prefix),
                "--xrce-agent-port",
                str(selection.xrce_agent_port),
                "--xrce-domain-id",
                str(selection.xrce_domain_id),
                "--xrce-namespace",
                selection.xrce_namespace,
                "--xrce-client-key",
                hex(selection.xrce_client_key),
                "--xrce-startup-timeout",
                str(selection.xrce_startup_timeout),
            )
        )
    if selection.curve_secret_key is not None:
        arguments.extend(
            (
                "--curve-secret-key",
                str(selection.curve_secret_key),
                "--curve-peer-key",
                str(selection.curve_peer_key),
            )
        )
    return shlex.join(arguments)


def build_calibration_plan(choice: RobotSpecChoice) -> CalibrationPlan:
    """Build every EEPROM write before allowing hardware connection."""

    if choice.spec.backend != Backend.PHYSICAL:
        raise RuntimeError("FEETECH home calibration requires backend='physical'")
    incompatible = [
        f"{bus.name} ({bus.type.value})"
        for bus in choice.spec.buses
        if bus.type != BusType.FEETECH_PACKET
    ]
    if incompatible:
        raise ValueError(
            "FEETECH home calibration cannot include non-packet buses: "
            + ", ".join(incompatible)
        )

    runtime = RobotRuntime(choice.spec)
    targets = runtime.home_calibration_targets()
    buses = {bus.name: bus for bus in choice.spec.buses}
    rows: list[CalibrationJointPlan] = []
    # Preserve RobotSpec order so the confirmation page matches URDF/ROS joint order.
    for arm in choice.spec.arms:
        ordered_joints = [(arm.bus, joint) for joint in arm.joints]
        if isinstance(arm.end_effector, ParallelGripperSpec):
            ordered_joints.append((arm.end_effector.bus, arm.end_effector.joint))
        for bus_name, joint in ordered_joints:
            rows.append(
                CalibrationJointPlan(
                    bus_name=bus_name,
                    port=buses[bus_name].port,
                    joint_name=joint.name,
                    servo_id=joint.servo_id,
                    servo_model=joint.servo_model,
                    direction=joint.direction,
                    home_position_rad=joint.home_position_rad,
                    target_raw_position=targets[bus_name][joint.servo_id],
                )
            )
    if not rows:
        raise ValueError("RobotSpec contains no FEETECH joints to calibrate")
    return CalibrationPlan(choice, tuple(rows))


def config_reference(choice: RobotSpecChoice) -> dict[str, str]:
    """Serialize packaged configs portably and custom configs absolutely."""

    if choice.resource_id is not None:
        return {"kind": "packaged", "value": choice.resource_id}
    return {"kind": "custom", "value": str(choice.path)}


def resolve_config_reference(
    value: Any,
    packaged: Sequence[RobotSpecChoice],
) -> Optional[RobotSpecChoice]:
    """Restore a saved reference, ignoring stale custom paths safely."""

    if not isinstance(value, dict):
        return None
    kind = value.get("kind")
    reference = value.get("value")
    if not isinstance(reference, str):
        return None
    if kind == "packaged":
        return next(
            (choice for choice in packaged if choice.resource_id == reference),
            None,
        )
    if kind == "custom":
        try:
            return inspect_robot_spec(reference)
        except (OSError, RuntimeError, ValueError):
            return None
    return None


class _BiosTui:
    """Small ACETele-specific menu modeled after ACELab's curses workflow."""

    def __init__(
        self,
        packaged: Sequence[RobotSpecChoice],
        *,
        launch_selection: LaunchSelection,
        calibration_choice: RobotSpecChoice,
        zmq_selection: Optional[ZmqLaunchSelection] = None,
    ) -> None:
        self.packaged = tuple(packaged)
        self.launch_choice = launch_selection.choice
        self.teleop_mode = launch_selection.teleop_mode
        self.translation_scale = str(launch_selection.translation_scale)
        self.rotation_scale = str(launch_selection.rotation_scale)
        if zmq_selection is None:
            role = (
                "follower"
                if launch_selection.choice.spec.model == "ace_follower"
                else "leader"
            )
            zmq_selection = ZmqLaunchSelection(
                launch_selection.choice,
                role,
                "127.0.0.1",
            )
        self.zmq_choice = zmq_selection.choice
        self.zmq_role = zmq_selection.role
        self.zmq_peer_host = zmq_selection.peer_host
        self.zmq_bind_host = zmq_selection.bind_host
        self.zmq_command_port = str(zmq_selection.command_port)
        self.zmq_state_port = str(zmq_selection.state_port)
        self.zmq_teleop_mode = zmq_selection.teleop_mode
        self.zmq_translation_scale = str(zmq_selection.translation_scale)
        self.zmq_rotation_scale = str(zmq_selection.rotation_scale)
        self.zmq_curve_enabled = zmq_selection.curve_secret_key is not None
        self.zmq_curve_secret_key = (
            "" if zmq_selection.curve_secret_key is None else str(zmq_selection.curve_secret_key)
        )
        self.zmq_curve_peer_key = (
            "" if zmq_selection.curve_peer_key is None else str(zmq_selection.curve_peer_key)
        )
        self.zmq_xrce_prefix = str(zmq_selection.xrce_prefix)
        self.zmq_xrce_agent_port = str(zmq_selection.xrce_agent_port)
        self.zmq_xrce_domain_id = str(zmq_selection.xrce_domain_id)
        self.zmq_xrce_namespace = zmq_selection.xrce_namespace
        self.zmq_xrce_client_key = hex(zmq_selection.xrce_client_key)
        self.zmq_xrce_startup_timeout = str(zmq_selection.xrce_startup_timeout)
        self.calibration_choice = calibration_choice
        self.status = "Use Up/Down to move. Enter selects. Esc goes back."
        self._danger_attribute = curses.A_REVERSE | curses.A_BOLD
        self._warning_attribute = curses.A_BOLD

    def run(self, stdscr) -> Optional[TuiResult]:
        """Configure curses once, then dispatch the selected workflow."""

        set_escape_delay = getattr(curses, "set_escdelay", None)
        if callable(set_escape_delay):
            set_escape_delay(25)
        curses.curs_set(0)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_RED)
            self._danger_attribute = curses.color_pair(4) | curses.A_BOLD
            self._warning_attribute = curses.color_pair(3) | curses.A_BOLD

        while True:
            workflow = self._select_menu(
                stdscr,
                "ACETELE CONTROL",
                (
                    "Launch ROS 2 Robot",
                    "Launch ZMQ Robot",
                    "Calibrate FEETECH Home",
                ),
                (
                    "Generate a validated ros2 launch command without starting it.",
                    "Generate one direct Leader/Follower ZeroMQ command.",
                    "Write FEETECH packet home offsets after explicit physical confirmation.",
                ),
            )
            if workflow is None:
                return None
            if workflow == 0:
                result = self._launch_workflow(stdscr)
            elif workflow == 1:
                result = self._zmq_workflow(stdscr)
            else:
                result = self._calibration_workflow(stdscr)
            if result is not None:
                return result

    def _launch_workflow(self, stdscr) -> Optional[TuiResult]:
        selected = 0
        while True:
            fields = ["RobotSpec", "Teleop mode"]
            values = [self.launch_choice.label, self.teleop_mode]
            if self.teleop_mode == "ee_pose":
                fields.extend(("Translation scale", "Rotation scale"))
                values.extend((self.translation_scale, self.rotation_scale))
            review_index = len(fields)
            fields.append("Review and generate")
            values.append("")
            key = self._draw_form(
                stdscr,
                "ROS 2 LAUNCH SETUP",
                fields,
                values,
                selected,
                self.launch_choice.summary,
            )
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
            elif key == curses.KEY_DOWN:
                selected = min(len(fields) - 1, selected + 1)
            elif key in (10, 13, curses.KEY_ENTER):
                if selected == review_index:
                    result = self._review_launch(stdscr)
                    if result is not None:
                        return result
                elif selected == 0:
                    choice = self._choose_spec(stdscr, self.launch_choice, calibration=False)
                    if choice is not None:
                        self.launch_choice = choice
                elif selected == 1:
                    mode = self._select_menu(
                        stdscr,
                        "SELECT TELEOP MODE",
                        ("joint", "ee_pose"),
                        (
                            "Publish joint-space commands.",
                            "Publish relative TCP poses and solve follower IK.",
                        ),
                        initial=0 if self.teleop_mode == "joint" else 1,
                    )
                    if mode is not None:
                        self.teleop_mode = ("joint", "ee_pose")[mode]
                        selected = min(selected, 1)
                elif selected == 2:
                    value = self._edit_text(stdscr, "Translation scale", self.translation_scale)
                    if value is not None:
                        self.translation_scale = value
                elif selected == 3:
                    value = self._edit_text(stdscr, "Rotation scale", self.rotation_scale)
                    if value is not None:
                        self.rotation_scale = value
            elif key in (27, ord("q"), ord("Q")):
                return None

    def _zmq_workflow(self, stdscr) -> Optional[TuiResult]:
        """Edit deployment-only ZMQ fields without changing the RobotSpec."""

        selected = 0
        while True:
            fields = [
                ("spec", "RobotSpec", self.zmq_choice.label),
                ("role", "Role", self.zmq_role),
                ("mode", "Teleop mode", self.zmq_teleop_mode),
                ("peer", "Peer host", self.zmq_peer_host),
                ("bind", "Bind host", self.zmq_bind_host),
                ("command_port", "Command port", self.zmq_command_port),
                ("state_port", "State port", self.zmq_state_port),
                (
                    "security",
                    "Security",
                    "CURVE" if self.zmq_curve_enabled else "plaintext",
                ),
            ]
            if self.zmq_teleop_mode == "ee_pose" and self.zmq_role == "follower":
                fields.extend(
                    (
                        ("translation", "Translation scale", self.zmq_translation_scale),
                        ("rotation", "Rotation scale", self.zmq_rotation_scale),
                    )
                )
            if self.zmq_role == "follower":
                fields.extend(
                    (
                        ("xrce_prefix", "XRCE prefix", self.zmq_xrce_prefix),
                        (
                            "xrce_port",
                            "XRCE agent UDP port",
                            self.zmq_xrce_agent_port,
                        ),
                        ("xrce_domain", "XRCE domain ID", self.zmq_xrce_domain_id),
                        ("xrce_namespace", "XRCE namespace", self.zmq_xrce_namespace),
                        ("xrce_key", "XRCE client key", self.zmq_xrce_client_key),
                        (
                            "xrce_timeout",
                            "XRCE startup timeout",
                            self.zmq_xrce_startup_timeout,
                        ),
                    )
                )
            if self.zmq_curve_enabled:
                fields.extend(
                    (
                        ("secret", "Local secret key", self.zmq_curve_secret_key),
                        ("peer_key", "Peer public key", self.zmq_curve_peer_key),
                    )
                )
            fields.append(("review", "Review and generate", ""))
            key = self._draw_form(
                stdscr,
                "ZMQ TELEOP SETUP",
                tuple(field[1] for field in fields),
                tuple(field[2] for field in fields),
                selected,
                (*self.zmq_choice.summary, self._zmq_security_summary()),
            )
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
                continue
            if key == curses.KEY_DOWN:
                selected = min(len(fields) - 1, selected + 1)
                continue
            if key in (27, ord("q"), ord("Q")):
                return None
            if key not in (10, 13, curses.KEY_ENTER):
                continue
            field = fields[selected][0]
            if field == "review":
                result = self._review_zmq(stdscr)
                if result is not None:
                    return result
            elif field == "spec":
                selected_spec = self._choose_spec(
                    stdscr,
                    self.zmq_choice,
                    calibration=False,
                )
                if selected_spec is not None:
                    self.zmq_choice = selected_spec
                    if selected_spec.spec.model in ("ace_leader", "ace_follower"):
                        self.zmq_role = selected_spec.spec.model.removeprefix("ace_")
            elif field == "role":
                role_index = self._select_menu(
                    stdscr,
                    "SELECT ZMQ ROLE",
                    ("leader", "follower"),
                    ("Publish commands and consume state.", "Apply commands and publish state."),
                    initial=0 if self.zmq_role == "leader" else 1,
                )
                if role_index is not None:
                    self.zmq_role = ("leader", "follower")[role_index]
                    matching = next(
                        (
                            item
                            for item in self.packaged
                            if item.spec.model == f"ace_{self.zmq_role}"
                        ),
                        None,
                    )
                    if matching is not None:
                        self.zmq_choice = matching
            elif field == "mode":
                mode_index = self._select_menu(
                    stdscr,
                    "SELECT TELEOP MODE",
                    ("joint", "ee_pose"),
                    ("Publish joint-space commands.", "Publish TCP poses and solve follower IK."),
                    initial=0 if self.zmq_teleop_mode == "joint" else 1,
                )
                if mode_index is not None:
                    self.zmq_teleop_mode = ("joint", "ee_pose")[mode_index]
            elif field == "security":
                security_index = self._select_menu(
                    stdscr,
                    "SELECT ZMQ SECURITY",
                    ("plaintext", "CURVE"),
                    (
                        "Trusted wired networks only; no peer authentication.",
                        "Authenticate and encrypt using an exact peer public key.",
                    ),
                    initial=int(self.zmq_curve_enabled),
                )
                if security_index is not None:
                    self.zmq_curve_enabled = security_index == 1
            else:
                attributes = {
                    "peer": ("Peer host", "zmq_peer_host"),
                    "bind": ("Bind host", "zmq_bind_host"),
                    "command_port": ("Command port", "zmq_command_port"),
                    "state_port": ("State port", "zmq_state_port"),
                    "translation": ("Translation scale", "zmq_translation_scale"),
                    "rotation": ("Rotation scale", "zmq_rotation_scale"),
                    "secret": ("Local CURVE secret key", "zmq_curve_secret_key"),
                    "peer_key": ("Peer CURVE public key", "zmq_curve_peer_key"),
                    "xrce_prefix": ("XRCE install prefix", "zmq_xrce_prefix"),
                    "xrce_port": ("XRCE agent UDP port", "zmq_xrce_agent_port"),
                    "xrce_domain": ("XRCE domain ID", "zmq_xrce_domain_id"),
                    "xrce_namespace": ("XRCE namespace", "zmq_xrce_namespace"),
                    "xrce_key": ("XRCE client key", "zmq_xrce_client_key"),
                    "xrce_timeout": (
                        "XRCE startup timeout",
                        "zmq_xrce_startup_timeout",
                    ),
                }
                prompt, attribute = attributes[field]
                value = self._edit_text(stdscr, prompt, getattr(self, attribute))
                if value is not None:
                    setattr(self, attribute, value)

    def _calibration_workflow(self, stdscr) -> Optional[TuiResult]:
        selected = 0
        while True:
            try:
                plan = build_calibration_plan(self.calibration_choice)
                details = self._calibration_lines(plan, include_warning=False)
            except (OSError, RuntimeError, ValueError) as exc:
                details = (str(exc),)
            key = self._draw_form(
                stdscr,
                "FEETECH HOME CALIBRATION",
                ("RobotSpec", "Review and calibrate"),
                (self.calibration_choice.label, ""),
                selected,
                details,
            )
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
            elif key == curses.KEY_DOWN:
                selected = min(1, selected + 1)
            elif key in (10, 13, curses.KEY_ENTER):
                if selected == 0:
                    choice = self._choose_spec(stdscr, self.calibration_choice, calibration=True)
                    if choice is not None:
                        self.calibration_choice = choice
                else:
                    result = self._review_calibration(stdscr)
                    if result is not None:
                        return result
            elif key in (27, ord("q"), ord("Q")):
                return None

    def _review_launch(self, stdscr) -> Optional[TuiResult]:
        """Validate the launch form and show the final command before accepting it."""

        try:
            selection = LaunchSelection(
                self.launch_choice,
                self.teleop_mode,
                float(self.translation_scale),
                float(self.rotation_scale),
            )
            command = build_ros_launch_command(selection)
        except (TypeError, ValueError) as exc:
            self.status = str(exc)
            return None
        lines = [*selection.choice.summary, "", "Command:", command]
        if self._confirm_document(stdscr, "CONFIRM ROS 2 COMMAND", lines):
            return TuiResult(TuiWorkflow.ROS2_LAUNCH, launch=selection)
        return None

    def _review_zmq(self, stdscr) -> Optional[TuiResult]:
        try:
            selection = ZmqLaunchSelection(
                self.zmq_choice,
                self.zmq_role,
                self.zmq_peer_host,
                self.zmq_bind_host,
                int(self.zmq_command_port),
                int(self.zmq_state_port),
                self.zmq_teleop_mode,
                float(self.zmq_translation_scale),
                float(self.zmq_rotation_scale),
                (
                    Path(self.zmq_curve_secret_key)
                    if self.zmq_curve_enabled and self.zmq_curve_secret_key
                    else None
                ),
                (
                    Path(self.zmq_curve_peer_key)
                    if self.zmq_curve_enabled and self.zmq_curve_peer_key
                    else None
                ),
                Path(self.zmq_xrce_prefix),
                int(self.zmq_xrce_agent_port),
                int(self.zmq_xrce_domain_id),
                self.zmq_xrce_namespace,
                int(self.zmq_xrce_client_key, 0),
                float(self.zmq_xrce_startup_timeout),
            )
            command = build_zmq_launch_command(selection)
        except (TypeError, ValueError) as exc:
            self.status = str(exc)
            return None
        lines = [
            *selection.choice.summary,
            "",
            self._zmq_security_summary(),
            "Command:",
            command,
        ]
        if self._confirm_document(stdscr, "CONFIRM ZMQ COMMAND", lines):
            return TuiResult(TuiWorkflow.ZMQ_LAUNCH, zmq_launch=selection)
        return None

    def _zmq_security_summary(self) -> str:
        if self.zmq_curve_enabled:
            return "CURVE authentication and encryption enabled."
        return "WARNING: plaintext TCP is for trusted wired networks only."

    def _review_calibration(self, stdscr) -> Optional[TuiResult]:
        """Build the complete write plan before requesting the safety phrase."""

        try:
            plan = build_calibration_plan(self.calibration_choice)
        except (OSError, RuntimeError, ValueError) as exc:
            self.status = str(exc)
            return None
        if self._confirm_calibration(stdscr, plan):
            return TuiResult(TuiWorkflow.CALIBRATE, calibration=plan)
        return None

    def _choose_spec(
        self,
        stdscr,
        current: RobotSpecChoice,
        *,
        calibration: bool,
    ) -> Optional[RobotSpecChoice]:
        choices: list[RobotSpecChoice] = []
        for choice in self.packaged:
            if calibration:
                try:
                    build_calibration_plan(choice)
                except (OSError, RuntimeError, ValueError):
                    continue
            choices.append(choice)
        if current.resource_id is None and all(current.path != choice.path for choice in choices):
            try:
                if calibration:
                    build_calibration_plan(current)
                choices.insert(0, current)
            except (OSError, RuntimeError, ValueError):
                pass

        while True:
            labels = tuple(choice.label for choice in choices) + ("Custom RobotSpec...",)
            details = tuple("\n".join(choice.summary) for choice in choices) + (
                "Enter an absolute, relative, or ~/ path to a RobotSpec TOML.",
            )
            initial = next(
                (index for index, choice in enumerate(choices) if choice.path == current.path),
                0,
            )
            selected = self._select_menu(
                stdscr,
                "SELECT ROBOT SPEC",
                labels,
                details,
                initial=initial,
            )
            if selected is None:
                return None
            if selected < len(choices):
                return choices[selected]
            path = self._edit_text(stdscr, "RobotSpec path", str(current.path))
            if path is None:
                continue
            try:
                choice = inspect_robot_spec(path)
                if calibration:
                    build_calibration_plan(choice)
                return choice
            except (OSError, RuntimeError, ValueError) as exc:
                self.status = str(exc)

    def _confirm_calibration(self, stdscr, plan: CalibrationPlan) -> bool:
        lines = self._calibration_lines(plan, include_warning=True)
        line_attributes = self._calibration_line_attributes(lines)
        scroll = 0
        while True:
            key, scroll = self._draw_document(
                stdscr,
                "CONFIRM FEETECH EEPROM WRITE",
                lines,
                scroll,
                "Enter=Calibrate  Esc=Back  Up/Down=Scroll",
                line_attributes=line_attributes,
            )
            if key in (10, 13, curses.KEY_ENTER):
                return True
            elif key in (27, ord("q"), ord("Q")):
                return False
            elif key == curses.KEY_UP:
                scroll = max(0, scroll - 1)
            elif key == curses.KEY_DOWN:
                scroll += 1

    def _confirm_document(self, stdscr, title: str, lines: Sequence[str]) -> bool:
        scroll = 0
        while True:
            key, scroll = self._draw_document(
                stdscr,
                title,
                lines,
                scroll,
                "Enter=Generate  Esc=Back  Up/Down=Scroll",
            )
            if key in (10, 13, curses.KEY_ENTER):
                return True
            if key in (27, ord("q"), ord("Q")):
                return False
            if key == curses.KEY_UP:
                scroll = max(0, scroll - 1)
            elif key == curses.KEY_DOWN:
                scroll += 1

    @staticmethod
    def _calibration_lines(
        plan: CalibrationPlan,
        *,
        include_warning: bool,
    ) -> tuple[str, ...]:
        lines = [
            *plan.choice.summary,
            "",
            "Nonvolatile home calibration targets:",
        ]
        for joint in plan.joints:
            lines.append(
                f"{joint.bus_name} {joint.port} | {joint.joint_name} id={joint.servo_id} "
                f"model={joint.servo_model} direction={joint.direction:+d} "
                f"home={joint.home_position_rad:.6g}rad "
                f"target_raw={joint.target_raw_position}"
            )
        if include_warning:
            lines.extend(
                (
                    "",
                    "WARNING: This writes nonvolatile servo calibration data.",
                    "target_raw is the desired reading at mechanical Home, not the "
                    "offset currently stored in EEPROM.",
                    "Place every joint at its declared mechanical Home pose.",
                    "Keep the robot supported and the independent emergency stop reachable.",
                    "Packet success does not verify torque disable at every individual servo.",
                )
            )
        return tuple(lines)

    def _calibration_line_attributes(self, lines: Sequence[str]) -> dict[int, int]:
        """Assign explicit visual severity to calibration capability and risk lines."""

        attributes: dict[int, int] = {}
        warning_started = False
        for index, line in enumerate(lines):
            if line.startswith("WARNING:"):
                warning_started = True
                attributes[index] = self._danger_attribute
            elif warning_started or "verified_disable=false" in line or "verified_identity=false" in line:
                attributes[index] = self._warning_attribute
            elif line == "Nonvolatile home calibration targets:":
                attributes[index] = curses.A_BOLD
        return attributes

    def _select_menu(
        self,
        stdscr,
        title: str,
        labels: Sequence[str],
        details: Sequence[str],
        *,
        initial: int = 0,
    ) -> Optional[int]:
        selected = min(max(initial, 0), len(labels) - 1)
        while True:
            height, width = stdscr.getmaxyx()
            stdscr.erase()
            if not self._terminal_is_large_enough(stdscr, height, width):
                key = stdscr.getch()
                if key in (27, ord("q"), ord("Q")):
                    return None
                continue
            self._draw_header(stdscr, title, width)
            visible_rows = max(height - 7, 1)
            offset = min(max(selected - visible_rows + 1, 0), max(len(labels) - visible_rows, 0))
            for row, index in enumerate(range(offset, min(offset + visible_rows, len(labels))), start=2):
                attribute = curses.A_REVERSE if index == selected else curses.A_NORMAL
                self._draw_line(stdscr, row, f" {labels[index]}", width, attribute)
            detail = details[selected].replace("\n", " | ")
            self._draw_line(stdscr, height - 3, detail, width, curses.color_pair(3))
            self._draw_line(stdscr, height - 2, self.status, width)
            self._draw_line(stdscr, height - 1, "Enter=Select  Esc/Q=Back  Up/Down=Move", width, curses.A_BOLD)
            stdscr.refresh()
            key = stdscr.getch()
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
            elif key == curses.KEY_DOWN:
                selected = min(len(labels) - 1, selected + 1)
            elif key in (10, 13, curses.KEY_ENTER):
                return selected
            elif key in (27, ord("q"), ord("Q")):
                return None

    def _draw_form(
        self,
        stdscr,
        title: str,
        fields: Sequence[str],
        values: Sequence[str],
        selected: int,
        detail_lines: Sequence[str],
    ) -> int:
        while True:
            height, width = stdscr.getmaxyx()
            stdscr.erase()
            if not self._terminal_is_large_enough(stdscr, height, width):
                key = stdscr.getch()
                if key in (27, ord("q"), ord("Q")):
                    return key
                continue
            self._draw_header(stdscr, title, width)
            for index, (menu_field, value) in enumerate(zip(fields, values), start=0):
                attribute = curses.A_REVERSE if index == selected else curses.A_NORMAL
                self._draw_line(
                    stdscr,
                    index + 2,
                    f" {menu_field:<20} {value}",
                    width,
                    attribute,
                )
            detail = " | ".join(detail_lines)
            self._draw_line(stdscr, height - 3, detail, width, curses.color_pair(3))
            self._draw_line(stdscr, height - 2, self.status, width)
            self._draw_line(
                stdscr,
                height - 1,
                "Enter=Open/Review  Esc/Q=Back  Up/Down=Move",
                width,
                curses.A_BOLD,
            )
            stdscr.refresh()
            return stdscr.getch()

    def _draw_document(
        self,
        stdscr,
        title: str,
        lines: Sequence[str],
        scroll: int,
        footer: str,
        *,
        line_attributes: Optional[Mapping[int, int]] = None,
    ) -> tuple[int, int]:
        # Confirmation content must be visible before Enter can authorize an action.
        # A resize loop therefore consumes every key except explicit cancel.
        while True:
            height, width = stdscr.getmaxyx()
            stdscr.erase()
            if self._terminal_is_large_enough(stdscr, height, width):
                break
            key = stdscr.getch()
            if key in (27, ord("q"), ord("Q")):
                return key, 0
        self._draw_header(stdscr, title, width)
        wrapped: list[tuple[str, int]] = []
        attributes = line_attributes or {}
        for index, line in enumerate(lines):
            attribute = attributes.get(index, curses.A_NORMAL)
            segments = textwrap.wrap(
                line,
                max(width - 2, 1),
                replace_whitespace=False,
            ) or [""]
            wrapped.extend((segment, attribute) for segment in segments)
        visible = max(height - 4, 1)
        maximum_scroll = max(len(wrapped) - visible, 0)
        scroll = min(max(scroll, 0), maximum_scroll)
        for row, (line, attribute) in enumerate(
            wrapped[scroll : scroll + visible],
            start=2,
        ):
            self._draw_line(stdscr, row, line, width, attribute)
        self._draw_line(stdscr, height - 2, self.status, width, curses.color_pair(3))
        self._draw_line(stdscr, height - 1, footer, width, curses.A_BOLD)
        stdscr.refresh()
        return stdscr.getch(), scroll

    def _edit_text(self, stdscr, prompt: str, initial: str) -> Optional[str]:
        value = list(initial)
        cursor = len(value)
        curses.curs_set(1)
        try:
            while True:
                height, width = stdscr.getmaxyx()
                stdscr.erase()
                self._draw_header(stdscr, "EDIT VALUE", width)
                self._draw_line(stdscr, 3, prompt, width, curses.color_pair(3))
                available = max(width - 3, 1)
                start = max(cursor - available + 1, 0)
                visible = "".join(value[start : start + available])
                self._draw_line(stdscr, 5, "> " + visible, width)
                self._draw_line(stdscr, height - 1, "Enter=Accept  Esc=Cancel", width, curses.A_BOLD)
                try:
                    stdscr.move(5, min(2 + cursor - start, width - 1))
                except curses.error:
                    pass
                stdscr.refresh()
                key = stdscr.get_wch()
                if key in ("\n", "\r", curses.KEY_ENTER):
                    return "".join(value).strip()
                if key == "\x1b":
                    return None
                if key in (curses.KEY_BACKSPACE, 127, "\b", "\x7f"):
                    if cursor:
                        cursor -= 1
                        value.pop(cursor)
                elif key == curses.KEY_LEFT:
                    cursor = max(0, cursor - 1)
                elif key == curses.KEY_RIGHT:
                    cursor = min(len(value), cursor + 1)
                elif isinstance(key, str) and key.isprintable():
                    value.insert(cursor, key)
                    cursor += 1
        finally:
            curses.curs_set(0)

    @staticmethod
    def _terminal_is_large_enough(stdscr, height: int, width: int) -> bool:
        if height >= 14 and width >= 64:
            return True
        message = "Terminal too small. Resize to at least 64x14, or press Q to exit."
        try:
            stdscr.addnstr(0, 0, message, max(width - 1, 0), curses.A_BOLD)
            stdscr.refresh()
        except curses.error:
            pass
        return False

    def _draw_header(self, stdscr, title: str, width: int) -> None:
        self._draw_line(stdscr, 0, f" {title} ", width, curses.color_pair(1) | curses.A_BOLD)
        self._draw_line(stdscr, 1, "=" * max(width - 1, 0), width, curses.A_DIM)

    @staticmethod
    def _draw_line(stdscr, row: int, text: str, width: int, attribute: int = 0) -> None:
        try:
            stdscr.addnstr(row, 0, text, max(width - 1, 0), attribute)
        except curses.error:
            pass


def _default_choice(
    packaged: Sequence[RobotSpecChoice],
    resource_id: str,
) -> RobotSpecChoice:
    return next(
        (choice for choice in packaged if choice.resource_id == resource_id),
        packaged[0],
    )


def _restore_launch_selection(
    packaged: Sequence[RobotSpecChoice],
    state: Mapping[str, Any],
) -> LaunchSelection:
    section = state.get("launch")
    values = section if isinstance(section, dict) else {}
    choice = resolve_config_reference(values.get("config"), packaged) or _default_choice(
        packaged,
        "ace_leader/feetech_hls_ttl.toml",
    )
    try:
        return LaunchSelection(
            choice,
            str(values.get("teleop_mode", "joint")),
            float(values.get("translation_scale", 2.0)),
            float(values.get("rotation_scale", 1.0)),
        )
    except (TypeError, ValueError):
        return LaunchSelection(choice)


def _restore_zmq_selection(
    packaged: Sequence[RobotSpecChoice],
    state: Mapping[str, Any],
) -> ZmqLaunchSelection:
    """Restore a complete ZMQ form without trusting stale persisted values."""

    section = state.get("zmq_launch")
    values = section if isinstance(section, dict) else {}
    role = values.get("role", "leader")
    if role not in ("leader", "follower"):
        role = "leader"
    default_resource = f"ace_{role}/feetech_hls_ttl.toml"
    choice = resolve_config_reference(values.get("config"), packaged) or _default_choice(
        packaged,
        default_resource,
    )
    if choice.spec.model not in ("ace_leader", "ace_follower"):
        choice = _default_choice(packaged, default_resource)
    role = choice.spec.model.removeprefix("ace_")

    secret_value = values.get("curve_secret_key")
    peer_value = values.get("curve_peer_key")
    secret_key = Path(secret_value) if isinstance(secret_value, str) else None
    peer_key = Path(peer_value) if isinstance(peer_value, str) else None
    try:
        return ZmqLaunchSelection(
            choice,
            role,
            str(values.get("peer_host", "127.0.0.1")),
            str(values.get("bind_host", "0.0.0.0")),
            values.get("command_port", 5555),
            values.get("state_port", 5556),
            str(values.get("teleop_mode", "joint")),
            values.get("translation_scale", 2.0),
            values.get("rotation_scale", 1.0),
            secret_key,
            peer_key,
            Path(str(values.get("xrce_prefix", _default_xrce_prefix()))),
            values.get("xrce_agent_port", 8888),
            values.get("xrce_domain_id", 0),
            str(values.get("xrce_namespace", "")),
            values.get("xrce_client_key", 0xACED0001),
            values.get("xrce_startup_timeout", 3.0),
        )
    except (TypeError, ValueError):
        return ZmqLaunchSelection(choice, role, "127.0.0.1")


def _restore_calibration_choice(
    packaged: Sequence[RobotSpecChoice],
    state: Mapping[str, Any],
) -> RobotSpecChoice:
    section = state.get("calibration")
    values = section if isinstance(section, dict) else {}
    restored = resolve_config_reference(values.get("config"), packaged)
    candidates = [restored] if restored is not None else []
    candidates.extend(packaged)
    for choice in candidates:
        try:
            build_calibration_plan(choice)
            return choice
        except (OSError, RuntimeError, ValueError):
            continue
    raise RuntimeError("no FEETECH packet RobotSpec is available for calibration")


def _updated_state(previous: Mapping[str, Any], result: TuiResult) -> dict[str, Any]:
    document = dict(previous)
    if result.workflow == TuiWorkflow.ROS2_LAUNCH and result.launch is not None:
        document["launch"] = {
            "config": config_reference(result.launch.choice),
            "teleop_mode": result.launch.teleop_mode,
            "translation_scale": result.launch.translation_scale,
            "rotation_scale": result.launch.rotation_scale,
        }
    elif result.workflow == TuiWorkflow.ZMQ_LAUNCH and result.zmq_launch is not None:
        selection = result.zmq_launch
        document["zmq_launch"] = {
            "config": config_reference(selection.choice),
            "role": selection.role,
            "peer_host": selection.peer_host,
            "bind_host": selection.bind_host,
            "command_port": selection.command_port,
            "state_port": selection.state_port,
            "teleop_mode": selection.teleop_mode,
            "translation_scale": selection.translation_scale,
            "rotation_scale": selection.rotation_scale,
            "curve_secret_key": (
                None if selection.curve_secret_key is None else str(selection.curve_secret_key)
            ),
            "curve_peer_key": (
                None if selection.curve_peer_key is None else str(selection.curve_peer_key)
            ),
            "xrce_prefix": str(selection.xrce_prefix),
            "xrce_agent_port": selection.xrce_agent_port,
            "xrce_domain_id": selection.xrce_domain_id,
            "xrce_namespace": selection.xrce_namespace,
            "xrce_client_key": selection.xrce_client_key,
            "xrce_startup_timeout": selection.xrce_startup_timeout,
        }
    elif result.workflow == TuiWorkflow.CALIBRATE and result.calibration is not None:
        document["calibration"] = {
            "config": config_reference(result.calibration.choice),
        }
    return document


def _error_chain(error: BaseException) -> str:
    messages = []
    current: Optional[BaseException] = error
    while current is not None:
        text = str(current) or type(current).__name__
        messages.append(f"{type(current).__name__}: {text}")
        current = current.__cause__
    return " <- ".join(messages)


def main() -> int:
    """Run the TUI and perform only the explicitly confirmed action."""

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("ACETele TUI requires an interactive terminal.", file=sys.stderr)
        return 2
    try:
        packaged = discover_packaged_robot_specs()
        store = TuiStateStore()
        state = store.load()
        launch = _restore_launch_selection(packaged, state)
        zmq_launch = _restore_zmq_selection(packaged, state)
        calibration = _restore_calibration_choice(packaged, state)
        result = curses.wrapper(
            _BiosTui(
                packaged,
                launch_selection=launch,
                zmq_selection=zmq_launch,
                calibration_choice=calibration,
            ).run
        )
    except KeyboardInterrupt:
        return 130
    except (curses.error, OSError, RuntimeError, ValueError) as exc:
        print(_error_chain(exc), file=sys.stderr)
        return 2

    if result is None:
        return 0
    warning = store.save(_updated_state(state, result))
    if warning is not None:
        print(f"warning: {warning}", file=sys.stderr)

    if result.workflow == TuiWorkflow.ROS2_LAUNCH and result.launch is not None:
        print(build_ros_launch_command(result.launch))
        return 0
    if result.workflow == TuiWorkflow.ZMQ_LAUNCH and result.zmq_launch is not None:
        print(build_zmq_launch_command(result.zmq_launch))
        return 0
    if result.workflow == TuiWorkflow.CALIBRATE and result.calibration is not None:
        try:
            calibrate_feetech_home(
                result.calibration.choice.spec,
                progress=lambda stage: print(f"[{stage}]"),
            )
        except KeyboardInterrupt:
            return 130
        except BaseException as exc:
            print(_error_chain(exc), file=sys.stderr)
            return 1
        return 0
    print("TUI returned an incomplete action.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CalibrationJointPlan",
    "CalibrationPlan",
    "LaunchSelection",
    "RobotSpecChoice",
    "TuiResult",
    "TuiStateStore",
    "TuiWorkflow",
    "ZmqLaunchSelection",
    "build_calibration_plan",
    "build_ros_launch_command",
    "build_zmq_launch_command",
    "config_reference",
    "discover_packaged_robot_specs",
    "inspect_robot_spec",
    "main",
    "resolve_config_reference",
]
