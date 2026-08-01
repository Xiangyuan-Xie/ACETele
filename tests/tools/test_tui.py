from __future__ import annotations

import curses
import shlex
from dataclasses import replace
from pathlib import Path

import pytest

from acetele.specification import BusSpec, BusType, DirectionControl
from acetele.tools.tui import (
    LaunchSelection,
    TuiResult,
    TuiStateStore,
    TuiWorkflow,
    ZmqLaunchSelection,
    _BiosTui,
    _restore_zmq_selection,
    _updated_state,
    build_calibration_plan,
    build_ros_launch_command,
    build_zmq_launch_command,
    config_reference,
    discover_packaged_robot_specs,
    inspect_robot_spec,
    resolve_config_reference,
)

project_root = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def packaged():
    return discover_packaged_robot_specs()


def _choice(choices, resource_id):
    return next(choice for choice in choices if choice.resource_id == resource_id)


def _custom_config(tmp_path: Path, *, directory_name="custom configs") -> Path:
    source_path = project_root / "acetele/config/presets/ace_leader/feetech_hls_ttl.toml"
    urdf_path = project_root / "acetele/model/robots/ace_leader/description/ace_leader.urdf"
    destination = tmp_path / directory_name / "leader robot.toml"
    destination.parent.mkdir()
    source = source_path.read_text(encoding="utf-8")
    source = source.replace(
        'urdf_path = "../../../model/robots/ace_leader/description/ace_leader.urdf"',
        f'urdf_path = "{urdf_path}"',
    )
    destination.write_text(source, encoding="utf-8")
    return destination


def test_discovers_all_packaged_robot_specs_without_hardware(packaged):
    assert tuple(choice.resource_id for choice in packaged) == (
        "ace_leader/feetech_hls_ttl.toml",
        "ace_follower/fashionstar_rs485.toml",
        "ace_follower/feetech_hls_ttl.toml",
        "ace_follower/feetech_sms_rs485.toml",
    )
    assert all(choice.path.is_absolute() for choice in packaged)
    assert all(choice.summary for choice in packaged)
    assert tuple(choice.label for choice in packaged) == (
        "ACE LEADER | FEETECH HLS TTL",
        "ACE FOLLOWER | FASHIONSTAR RS485",
        "ACE FOLLOWER | FEETECH HLS TTL",
        "ACE FOLLOWER | FEETECH SMS RS485",
    )


def test_inspects_custom_paths_with_spaces_and_expands_home(tmp_path, monkeypatch):
    config = _custom_config(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    choice = inspect_robot_spec(Path("~") / config.relative_to(tmp_path))

    assert choice.path == config.resolve()
    assert choice.resource_id is None
    assert choice.spec.model == "ace_leader"
    assert choice.label == "ACE LEADER | Custom: leader robot.toml"


@pytest.mark.parametrize("name", ("robot.yaml", "missing.toml"))
def test_custom_spec_rejects_wrong_suffix_and_missing_file(tmp_path, name):
    with pytest.raises((FileNotFoundError, ValueError)):
        inspect_robot_spec(tmp_path / name)


def test_joint_launch_command_omits_cartesian_scales(packaged):
    selection = LaunchSelection(
        _choice(packaged, "ace_follower/feetech_hls_ttl.toml"),
        "joint",
    )

    arguments = shlex.split(build_ros_launch_command(selection))

    assert arguments[:4] == ["ros2", "launch", "ace_robot_ros2", "ace_robot.launch.py"]
    assert f"config_path:={selection.choice.path}" in arguments
    assert "teleop_mode:=joint" in arguments
    assert not any(argument.startswith("translation_scale:=") for argument in arguments)
    assert not any(argument.startswith("rotation_scale:=") for argument in arguments)


def test_ee_pose_launch_command_round_trips_a_path_with_spaces(tmp_path, packaged):
    custom = replace(
        _choice(packaged, "ace_leader/feetech_hls_ttl.toml"),
        path=tmp_path / "robot configs" / "leader.toml",
        resource_id=None,
    )
    selection = LaunchSelection(custom, "ee_pose", 2.5, 0.75)

    arguments = shlex.split(build_ros_launch_command(selection))

    assert f"config_path:={custom.path}" in arguments
    assert "teleop_mode:=ee_pose" in arguments
    assert "translation_scale:=2.5" in arguments
    assert "rotation_scale:=0.75" in arguments


def test_default_ee_pose_scales_remain_explicit_floats(packaged):
    arguments = shlex.split(build_ros_launch_command(LaunchSelection(packaged[0], "ee_pose")))

    assert "translation_scale:=2.0" in arguments
    assert "rotation_scale:=1.0" in arguments


def test_zmq_joint_command_round_trips_network_and_config_arguments(packaged):
    selection = ZmqLaunchSelection(
        _choice(packaged, "ace_leader/feetech_hls_ttl.toml"),
        "leader",
        "192.0.2.20",
        "0.0.0.0",
        6001,
        6002,
    )

    arguments = shlex.split(build_zmq_launch_command(selection))

    assert arguments[1:4] == ["-m", "ace_robot_zmq", "leader"]
    assert arguments[arguments.index("--config") + 1] == str(selection.choice.path)
    assert arguments[arguments.index("--peer-host") + 1] == "192.0.2.20"
    assert arguments[arguments.index("--command-port") + 1] == "6001"
    assert arguments[arguments.index("--state-port") + 1] == "6002"
    assert "--translation-scale" not in arguments
    assert "--curve-secret-key" not in arguments


def test_zmq_follower_ee_pose_command_quotes_paths_and_curve_keys(tmp_path, packaged):
    custom = replace(
        _choice(packaged, "ace_follower/feetech_hls_ttl.toml"),
        path=tmp_path / "robot configs" / "follower.toml",
        resource_id=None,
    )
    selection = ZmqLaunchSelection(
        custom,
        "follower",
        "leader.local",
        teleop_mode="ee_pose",
        translation_scale=2.5,
        rotation_scale=0.75,
        curve_secret_key=tmp_path / "curve keys" / "follower.key_secret",
        curve_peer_key=tmp_path / "curve keys" / "leader.key",
    )

    arguments = shlex.split(build_zmq_launch_command(selection))

    assert arguments[arguments.index("--config") + 1] == str(custom.path)
    assert arguments[arguments.index("--translation-scale") + 1] == "2.5"
    assert arguments[arguments.index("--rotation-scale") + 1] == "0.75"
    assert arguments[arguments.index("--curve-secret-key") + 1] == str(
        selection.curve_secret_key
    )
    assert arguments[arguments.index("--curve-peer-key") + 1] == str(
        selection.curve_peer_key
    )
    assert arguments[arguments.index("--xrce-prefix") + 1] == str(
        selection.xrce_prefix
    )
    assert arguments[arguments.index("--xrce-agent-port") + 1] == "8888"
    assert arguments[arguments.index("--xrce-domain-id") + 1] == "0"
    assert arguments[arguments.index("--xrce-namespace") + 1] == ""
    assert arguments[arguments.index("--xrce-client-key") + 1] == "0xaced0001"
    assert arguments[arguments.index("--xrce-startup-timeout") + 1] == "3.0"


@pytest.mark.parametrize(
    "changes",
    (
        {"role": "follower"},
        {"peer_host": "tcp://host"},
        {"command_port": 0},
        {"state_port": 5555},
        {"translation_scale": float("inf")},
        {"curve_secret_key": Path("secret")},
        {"xrce_client_key": 1},
        {"xrce_domain_id": 233},
    ),
)
def test_zmq_selection_rejects_invalid_deployment_parameters(packaged, changes):
    values = {
        "choice": _choice(packaged, "ace_leader/feetech_hls_ttl.toml"),
        "role": "leader",
        "peer_host": "127.0.0.1",
    }
    values.update(changes)

    with pytest.raises(ValueError):
        ZmqLaunchSelection(**values)


@pytest.mark.parametrize(
    ("mode", "translation", "rotation"),
    (("invalid", 2.0, 1.0), ("ee_pose", 0.0, 1.0), ("ee_pose", 2.0, float("nan"))),
)
def test_launch_selection_rejects_invalid_runtime_parameters(packaged, mode, translation, rotation):
    with pytest.raises(ValueError):
        LaunchSelection(packaged[0], mode, translation, rotation)


def test_calibration_plan_contains_arm_and_gripper_in_public_order(packaged):
    choice = _choice(packaged, "ace_follower/feetech_hls_ttl.toml")

    plan = build_calibration_plan(choice)

    assert tuple(joint.joint_name for joint in plan.joints) == (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
    )
    assert tuple(joint.servo_id for joint in plan.joints) == (0, 1, 2, 3, 4)
    assert tuple(joint.target_raw_position for joint in plan.joints) == (
        -1023,
        2047,
        0,
        0,
        -896,
    )


def test_sms_packet_config_is_calibratable_but_fashionstar_is_not(packaged):
    sms = _choice(packaged, "ace_follower/feetech_sms_rs485.toml")
    fashionstar = _choice(packaged, "ace_follower/fashionstar_rs485.toml")

    assert len(build_calibration_plan(sms).joints) == 4
    with pytest.raises(ValueError, match="non-packet buses"):
        build_calibration_plan(fashionstar)


@pytest.mark.parametrize(
    "bus_type",
    (BusType.FEETECH_MODBUS_RTU, BusType.FASHIONSTAR_RS485, BusType.LINKER_HAND_RS485),
)
def test_calibration_rejects_any_mixed_non_packet_bus(packaged, bus_type):
    original = _choice(packaged, "ace_leader/feetech_hls_ttl.toml")
    extra_bus = BusSpec(
        "extra",
        bus_type,
        "/dev/ttyUSB9",
        115200,
        10.0,
        DirectionControl.AUTO,
        physical_layer="rs485",
        external_estop=True,
        allow_unverified_identity=True,
    )
    spec = replace(original.spec, buses=(*original.spec.buses, extra_bus))
    mixed = replace(original, spec=spec)

    with pytest.raises(ValueError, match="extra"):
        build_calibration_plan(mixed)


def test_mock_calibration_is_rejected_before_hardware(tmp_path):
    config = _custom_config(tmp_path, directory_name="mock")
    source = config.read_text(encoding="utf-8")
    config.write_text(source.replace('backend = "physical"', 'backend = "mock"'), encoding="utf-8")
    choice = inspect_robot_spec(config)

    with pytest.raises(RuntimeError, match="backend='physical'"):
        build_calibration_plan(choice)


def test_state_store_round_trips_and_atomically_replaces(tmp_path):
    path = tmp_path / "state" / "tui.json"
    store = TuiStateStore(path)

    assert store.load() == {}
    assert store.save({"launch": {"teleop_mode": "ee_pose"}}) is None
    assert store.load() == {
        "version": 1,
        "launch": {"teleop_mode": "ee_pose"},
    }
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("payload", ("not-json", "[]", '{"version": 99}'))
def test_state_store_ignores_corrupt_or_unknown_state(tmp_path, payload):
    path = tmp_path / "tui.json"
    path.write_text(payload, encoding="utf-8")

    assert TuiStateStore(path).load() == {}


def test_config_references_restore_packaged_and_custom_specs(tmp_path, packaged):
    built_in = _choice(packaged, "ace_follower/feetech_hls_ttl.toml")
    custom = inspect_robot_spec(_custom_config(tmp_path))

    assert resolve_config_reference(config_reference(built_in), packaged) is built_in
    restored = resolve_config_reference(config_reference(custom), packaged)
    assert restored is not None
    assert restored.path == custom.path
    assert resolve_config_reference({"kind": "custom", "value": str(tmp_path / "gone.toml")}, packaged) is None


def test_zmq_selection_persistence_is_separate_and_recovers_from_stale_values(packaged):
    choice = _choice(packaged, "ace_follower/feetech_hls_ttl.toml")
    selection = ZmqLaunchSelection(
        choice,
        "follower",
        "leader.local",
        command_port=7001,
        state_port=7002,
        teleop_mode="ee_pose",
        translation_scale=2.25,
        rotation_scale=0.8,
    )

    state = _updated_state({}, TuiResult(TuiWorkflow.ZMQ_LAUNCH, zmq_launch=selection))
    restored = _restore_zmq_selection(packaged, state)

    assert restored == selection
    assert "launch" not in state
    invalid = {
        "zmq_launch": {
            "config": config_reference(choice),
            "role": "follower",
            "peer_host": "tcp://invalid",
        }
    }
    fallback = _restore_zmq_selection(packaged, invalid)
    assert fallback.role == "follower"
    assert fallback.peer_host == "127.0.0.1"


@pytest.mark.parametrize("key", (10, 13, curses.KEY_ENTER))
def test_calibration_review_accepts_confirmation_keys(monkeypatch, packaged, key):
    choice = _choice(packaged, "ace_leader/feetech_hls_ttl.toml")
    app = _BiosTui(
        packaged,
        launch_selection=LaunchSelection(choice),
        calibration_choice=choice,
    )
    monkeypatch.setattr(app, "_draw_document", lambda *_args, **_kwargs: (key, 0))

    assert app._confirm_calibration(object(), build_calibration_plan(choice))


def test_calibration_review_can_be_cancelled(monkeypatch, packaged):
    choice = _choice(packaged, "ace_leader/feetech_hls_ttl.toml")
    app = _BiosTui(
        packaged,
        launch_selection=LaunchSelection(choice),
        calibration_choice=choice,
    )
    monkeypatch.setattr(app, "_draw_document", lambda *_args, **_kwargs: (27, 0))

    assert not app._confirm_calibration(object(), build_calibration_plan(choice))


@pytest.mark.parametrize(
    ("mode", "down_count"),
    (("joint", 2), ("ee_pose", 4)),
)
def test_launch_form_can_open_review_and_confirm_with_enter(
    monkeypatch,
    packaged,
    mode,
    down_count,
):
    choice = _choice(packaged, "ace_leader/feetech_hls_ttl.toml")
    app = _BiosTui(
        packaged,
        launch_selection=LaunchSelection(choice, mode),
        calibration_choice=choice,
    )
    keys = iter((*((curses.KEY_DOWN,) * down_count), 10))
    monkeypatch.setattr(app, "_draw_form", lambda *_args, **_kwargs: next(keys))
    monkeypatch.setattr(app, "_confirm_document", lambda *_args, **_kwargs: True)

    result = app._launch_workflow(object())

    assert result is not None
    assert result.launch == LaunchSelection(choice, mode)


def test_calibration_form_can_open_review_and_confirm_with_enter(monkeypatch, packaged):
    choice = _choice(packaged, "ace_leader/feetech_hls_ttl.toml")
    app = _BiosTui(
        packaged,
        launch_selection=LaunchSelection(choice),
        calibration_choice=choice,
    )
    keys = iter((curses.KEY_DOWN, 10))
    monkeypatch.setattr(app, "_draw_form", lambda *_args, **_kwargs: next(keys))
    monkeypatch.setattr(app, "_confirm_calibration", lambda *_args, **_kwargs: True)

    result = app._calibration_workflow(object())

    assert result is not None
    assert result.calibration == build_calibration_plan(choice)


class _FakeScreen:
    def __init__(self, keys, *, height=24, width=100):
        self.keys = iter(keys)
        self.height = height
        self.width = width
        self.rows = []

    def getmaxyx(self):
        return self.height, self.width

    def erase(self):
        self.rows.clear()

    def addnstr(self, row, column, text, count, attribute=0):
        self.rows.append((row, column, text[:count], attribute))

    def refresh(self):
        return None

    def getch(self):
        return next(self.keys)


def test_bios_menu_moves_and_selects_with_arrow_keys(monkeypatch, packaged):
    choice = _choice(packaged, "ace_leader/feetech_hls_ttl.toml")
    app = _BiosTui(
        packaged,
        launch_selection=LaunchSelection(choice),
        calibration_choice=choice,
    )
    screen = _FakeScreen((curses.KEY_DOWN, 10))
    monkeypatch.setattr(curses, "color_pair", lambda _index: 0)

    selected = app._select_menu(
        screen,
        "MENU",
        ("first", "second"),
        ("first detail", "second detail"),
    )

    assert selected == 1


def test_small_terminal_cannot_confirm_an_unseen_document(monkeypatch, packaged):
    choice = _choice(packaged, "ace_leader/feetech_hls_ttl.toml")
    app = _BiosTui(
        packaged,
        launch_selection=LaunchSelection(choice),
        calibration_choice=choice,
    )
    screen = _FakeScreen((10, ord("q")), height=8, width=40)
    monkeypatch.setattr(curses, "color_pair", lambda _index: 0)

    assert not app._confirm_document(screen, "CONFIRM", ("dangerous action",))


def test_calibration_document_highlights_capability_risks_and_warning_block(packaged):
    choice = _choice(packaged, "ace_leader/feetech_hls_ttl.toml")
    app = _BiosTui(
        packaged,
        launch_selection=LaunchSelection(choice),
        calibration_choice=choice,
    )
    app._danger_attribute = 101
    app._warning_attribute = 202
    lines = app._calibration_lines(build_calibration_plan(choice), include_warning=True)

    attributes = app._calibration_line_attributes(lines)

    capability_index = next(index for index, line in enumerate(lines) if "verified_disable=false" in line)
    warning_index = next(index for index, line in enumerate(lines) if line.startswith("WARNING:"))
    assert attributes[capability_index] == 202
    assert attributes[warning_index] == 101
    assert all(attributes[index] == 202 for index in range(warning_index + 1, len(lines)))


def test_document_keeps_highlight_when_a_source_line_wraps(monkeypatch, packaged):
    choice = _choice(packaged, "ace_leader/feetech_hls_ttl.toml")
    app = _BiosTui(
        packaged,
        launch_selection=LaunchSelection(choice),
        calibration_choice=choice,
    )
    screen = _FakeScreen((ord("q"),), height=14, width=64)
    monkeypatch.setattr(curses, "color_pair", lambda _index: 0)

    app._draw_document(
        screen,
        "WARNING",
        ("This warning is long enough to wrap while retaining its visual severity." * 2,),
        0,
        "Esc=Back",
        line_attributes={0: 303},
    )

    body_rows = [row for row in screen.rows if 2 <= row[0] < screen.height - 2]
    assert len(body_rows) > 1
    assert all(row[3] == 303 for row in body_rows)
