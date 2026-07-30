"""Deadline-bound serial transport with optional kernel RS485 direction control."""

from __future__ import annotations

import time
from enum import Enum
from threading import RLock
from typing import Optional

import serial
from serial.rs485 import RS485Settings


class SerialDirectionControl(str, Enum):
    """Direction switching strategy for the serial adapter."""

    AUTO = "auto"
    RTS = "rts"


class SerialTransport:
    """Deadline-based pyserial transport with explicit connection ownership."""

    def __init__(
        self,
        port: str,
        baudrate: int,
        *,
        direction_control: SerialDirectionControl = SerialDirectionControl.AUTO,
    ) -> None:
        if not isinstance(port, str) or not port.strip():
            raise ValueError("serial port must be a non-empty string")
        if type(baudrate) is not int or baudrate <= 0:
            raise ValueError("serial baudrate must be a positive integer")
        if not isinstance(direction_control, SerialDirectionControl):
            raise ValueError("direction_control must be a SerialDirectionControl")
        self.port = port
        self.baudrate = baudrate
        self.direction_control = direction_control
        self._lock = RLock()
        self._serial: Optional[serial.Serial] = None

    @property
    def connected(self) -> bool:
        """Return whether this transport owns an open serial handle."""

        with self._lock:
            return self._serial is not None and self._serial.is_open

    def connect(self) -> None:
        """Open and configure the serial handle exactly once."""

        with self._lock:
            if self.connected:
                return
            handle = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05,
                write_timeout=0.05,
            )
            try:
                if self.direction_control == SerialDirectionControl.RTS:
                    handle.rs485_mode = RS485Settings(
                        rts_level_for_tx=True,
                        rts_level_for_rx=False,
                    )
                handle.reset_input_buffer()
                handle.reset_output_buffer()
            except BaseException:
                handle.close()
                raise
            self._serial = handle

    def write(self, payload: bytes, *, deadline_ns: int) -> None:
        """Write a complete frame before an absolute monotonic deadline."""

        if not isinstance(payload, bytes) or not payload:
            raise ValueError("serial payload must be non-empty bytes")
        handle = self._require_connected()
        remaining = self._remaining_seconds(deadline_ns)
        handle.write_timeout = remaining
        written = handle.write(payload)
        if written != len(payload):
            raise TimeoutError(f"serial write sent {written} of {len(payload)} bytes")
        # Reads are deadline-bound and begin only after the request is queued. Kernel
        # RS485 direction control drains the transmitter before switching direction;
        # pyserial.flush() is intentionally avoided because its drain is unbounded.

    def read_exact(self, count: int, *, deadline_ns: int) -> bytes:
        """Read exactly ``count`` bytes or fail at the absolute deadline."""

        if type(count) is not int or count <= 0:
            raise ValueError("serial read count must be a positive integer")
        handle = self._require_connected()
        result = bytearray()
        while len(result) < count:
            handle.timeout = self._remaining_seconds(deadline_ns)
            chunk = handle.read(count - len(result))
            if not chunk:
                raise TimeoutError(f"serial read timed out after {len(result)} of {count} bytes")
            result.extend(chunk)
        return bytes(result)

    def read_available(self, *, maximum: int = 4096) -> bytes:
        """Read only bytes already buffered by the OS, without waiting."""

        if type(maximum) is not int or maximum <= 0:
            raise ValueError("maximum must be a positive integer")
        handle = self._require_connected()
        count = min(maximum, max(0, handle.in_waiting))
        return b"" if count == 0 else bytes(handle.read(count))

    def discard_input(self) -> None:
        """Drop stale bytes before starting a request/response exchange."""

        self._require_connected().reset_input_buffer()

    def cancel(self) -> None:
        """Best-effort cancellation used to unblock bounded actor shutdown."""

        with self._lock:
            handle = self._serial
            if handle is None:
                return
            for method_name in ("cancel_read", "cancel_write"):
                method = getattr(handle, method_name, None)
                if callable(method):
                    try:
                        method()
                    except (OSError, serial.SerialException):
                        pass

    def disconnect(self) -> None:
        """Release the owned serial handle; repeated calls are harmless."""

        with self._lock:
            handle = self._serial
            self._serial = None
        if handle is None:
            return
        try:
            for method_name in ("cancel_read", "cancel_write"):
                method = getattr(handle, method_name, None)
                if callable(method):
                    try:
                        method()
                    except (OSError, serial.SerialException):
                        pass
        finally:
            handle.close()

    def _require_connected(self) -> serial.Serial:
        """Return the owned handle without exposing a disconnected stale object."""

        with self._lock:
            if self._serial is None or not self._serial.is_open:
                raise RuntimeError("serial transport is not connected")
            return self._serial

    @staticmethod
    def _remaining_seconds(deadline_ns: int) -> float:
        """Convert an absolute monotonic deadline into a pyserial timeout."""

        if type(deadline_ns) is not int or deadline_ns < 0:
            raise ValueError("deadline_ns must be a non-negative integer")
        remaining = (deadline_ns - time.monotonic_ns()) / 1e9
        if remaining <= 0.0:
            raise TimeoutError("serial operation deadline expired")
        return remaining


__all__ = ["SerialDirectionControl", "SerialTransport"]
