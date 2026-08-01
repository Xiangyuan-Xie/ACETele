import importlib
import sys
import types

import pytest


def test_joystick_connection_timeout_stops_thread_and_releases_pygame(monkeypatch):
    events = []
    fake_pygame = types.ModuleType("pygame")
    fake_pygame.joystick = types.SimpleNamespace(
        quit=lambda: events.append("joystick_quit")
    )
    fake_pygame.quit = lambda: events.append("pygame_quit")
    monkeypatch.setitem(sys.modules, "pygame", fake_pygame)
    monkeypatch.delitem(
        sys.modules,
        "acetele.hardware.inputs.joystick",
        raising=False,
    )
    module = importlib.import_module("acetele.hardware.inputs.joystick")

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon
            self.alive = False

        def start(self):
            self.alive = True
            events.append("thread_start")

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            events.append(("thread_join", timeout))
            self.alive = False

    monkeypatch.setattr(
        module.threading,
        "Thread",
        FakeThread,
    )
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda _duration: None,
    )

    with pytest.raises(RuntimeError, match="No joystick detected"):
        module.JoystickDriver(connect_timeout=0.001)

    assert events == [
        "thread_start",
        ("thread_join", 1.0),
        "joystick_quit",
        "pygame_quit",
    ]


def test_joystick_connection_timeout_does_not_release_pygame_with_live_worker(monkeypatch):
    events = []
    fake_pygame = types.ModuleType("pygame")
    fake_pygame.joystick = types.SimpleNamespace(
        quit=lambda: events.append("joystick_quit")
    )
    fake_pygame.quit = lambda: events.append("pygame_quit")
    monkeypatch.setitem(sys.modules, "pygame", fake_pygame)
    monkeypatch.delitem(
        sys.modules,
        "acetele.hardware.inputs.joystick",
        raising=False,
    )
    module = importlib.import_module("acetele.hardware.inputs.joystick")

    class StuckThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon
            self.alive = False

        def start(self):
            self.alive = True
            events.append("thread_start")

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            events.append(("thread_join", timeout))

    monkeypatch.setattr(module.threading, "Thread", StuckThread)
    monkeypatch.setattr(module.time, "sleep", lambda _duration: None)

    with pytest.raises(RuntimeError, match="worker thread did not stop") as exc_info:
        module.JoystickDriver(connect_timeout=0.001)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert events == ["thread_start", ("thread_join", 1.0)]
