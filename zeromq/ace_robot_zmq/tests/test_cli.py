from __future__ import annotations

import signal
from typing import Callable

from ace_robot_zmq import cli


class _Application:
    interrupt: Callable[[int, object], None]

    def __init__(self, *_args, **_kwargs) -> None:
        self.closed = False

    def run(self, should_stop) -> None:
        self.interrupt(signal.SIGINT, None)
        assert should_stop()

    def close(self) -> None:
        self.closed = True


class _NoopApplication(_Application):
    def run(self, _should_stop) -> None:
        pass


def test_cli_rejects_nonfinite_heartbeat_before_creating_an_application(
    tmp_path,
    monkeypatch,
):
    created = []
    monkeypatch.setattr(cli, "load_robot_spec", lambda _path: object())
    monkeypatch.setattr(cli, "LeaderApplication", lambda *_args, **_kwargs: created.append(True))

    result = cli.main(
        (
            "leader",
            "--config",
            str(tmp_path / "leader.toml"),
            "--peer-host",
            "follower.local",
            "--heartbeat-timeout",
            "nan",
        )
    )

    assert result == 2
    assert not created


def test_sigint_returns_130_after_closing_the_application(tmp_path, monkeypatch):
    handlers = {}
    application = _Application()
    monkeypatch.setattr(cli, "load_robot_spec", lambda _path: object())
    monkeypatch.setattr(cli, "LeaderApplication", lambda *_args, **_kwargs: application)
    monkeypatch.setattr(
        cli.signal,
        "signal",
        lambda signum, callback: handlers.__setitem__(signum, callback),
    )
    application.interrupt = lambda signum, frame: handlers[signum](signum, frame)

    result = cli.main(
        (
            "leader",
            "--config",
            str(tmp_path / "leader.toml"),
            "--peer-host",
            "follower.local",
        )
    )

    assert result == 130
    assert application.closed


def test_follower_cli_passes_validated_xrce_options(tmp_path, monkeypatch):
    captured = {}
    application = _NoopApplication()
    monkeypatch.setattr(cli, "load_robot_spec", lambda _path: object())

    def make_follower(*_args, **kwargs):
        captured.update(kwargs)
        return application

    monkeypatch.setattr(cli, "FollowerApplication", make_follower)

    result = cli.main(
        (
            "follower",
            "--config",
            str(tmp_path / "follower.toml"),
            "--peer-host",
            "leader.local",
            "--xrce-prefix",
            str(tmp_path / "xrce stack"),
            "--xrce-agent-port",
            "9999",
            "--xrce-domain-id",
            "7",
            "--xrce-namespace",
            "vehicle_1",
            "--xrce-client-key",
            "0xACED0042",
            "--xrce-startup-timeout",
            "4.5",
        )
    )

    assert result == 0
    options = captured["xrce_options"]
    assert options.prefix == (tmp_path / "xrce stack").resolve()
    assert options.agent_port == 9999
    assert options.domain_id == 7
    assert options.namespace == "vehicle_1"
    assert options.client_key == 0xACED0042
    assert options.startup_timeout_s == 4.5
    assert "hold_on_start" not in captured
