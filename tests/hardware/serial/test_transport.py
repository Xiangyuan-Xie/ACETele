from __future__ import annotations

import time

from acetele.hardware.serial import SerialTransport


class _SerialHandle:
    is_open = True

    def __init__(self) -> None:
        self.write_timeout = None
        self.payloads: list[bytes] = []

    def write(self, payload):
        self.payloads.append(payload)
        return len(payload)

    def flush(self):
        raise AssertionError("write must not call the unbounded serial flush")


def test_serial_write_does_not_use_unbounded_flush():
    transport = SerialTransport("mock", 1_000_000)
    handle = _SerialHandle()
    transport._serial = handle  # noqa: SLF001

    transport.write(b"request", deadline_ns=time.monotonic_ns() + 50_000_000)

    assert handle.payloads == [b"request"]
    assert 0.0 < handle.write_timeout <= 0.05
