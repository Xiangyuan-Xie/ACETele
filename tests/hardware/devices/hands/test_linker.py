from __future__ import annotations

import struct
import time
from dataclasses import replace

import pytest

from acetele.hardware.buses import MotionEnvelope, append_crc, calculate_bus_budget
from acetele.hardware.buses.modbus_rtu import ModbusProtocolError
from acetele.hardware.devices.hands.linker import (
    LinkerHandModbusProtocol,
    LinkerHandMotion,
    linker_hand_profiles,
)


def test_registry_covers_every_rs485_profile_in_official_sdk():
    assert linker_hand_profiles.names == ("O6", "L6", "L7", "L10")
    assert linker_hand_profiles.require("L10", context="hand").joint_count == 10
    with pytest.raises(ValueError, match="unsupported model"):
        linker_hand_profiles.require("L20", context="hand")


def test_profile_bandwidth_rejects_fake_100_hz_on_slow_o6_bus():
    profile = linker_hand_profiles.require("O6", context="hand")
    budget = calculate_bus_budget(
        baudrate=115200,
        cycle_hz=100.0,
        wire_bytes_per_cycle=profile.wire_bytes_per_cycle,
        turnaround_s_per_cycle=profile.turnaround_s_per_cycle,
    )

    assert not budget.feasible
    assert budget.maximum_cycle_hz < 20.0
    with pytest.raises(ValueError, match="maximum feasible rate"):
        budget.require_feasible(context="Linker Hand O6")


class ModbusTransport:
    def __init__(self, joint_count: int) -> None:
        self.joint_count = joint_count
        self.responses = bytearray()
        self.writes: list[bytes] = []
        self.connected = False

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def cancel(self):
        pass

    def write(self, frame, *, deadline_ns):
        self.writes.append(frame)
        slave_id, function = frame[:2]
        if function == 0x04:
            address, count = struct.unpack(">HH", frame[2:6])
            if address >= 100:
                values = (self.joint_count,) + (0,) * (count - 1)
            else:
                values = tuple((index * 17) % 256 for index in range(count))
            payload = bytes((slave_id, function, count * 2)) + struct.pack(
                f">{count}H", *values
            )
            self.responses.extend(append_crc(payload))
        elif function == 0x10:
            self.responses.extend(append_crc(frame[:6]))

    def read_exact(self, count, *, deadline_ns):
        if len(self.responses) < count:
            raise TimeoutError("no Modbus response")
        result = bytes(self.responses[:count])
        del self.responses[:count]
        return result


class ModbusExceptionTransport(ModbusTransport):
    def write(self, frame, *, deadline_ns):
        self.writes.append(frame)
        self.responses.extend(append_crc(bytes((frame[0], frame[1] | 0x80, 0x02))))


def test_generic_protocol_uses_profile_layout_and_normalized_units():
    profile = replace(
        linker_hand_profiles.require("L6", context="test"),
        frame_gap_s=0.0,
    )
    transport = ModbusTransport(profile.joint_count)
    protocol = LinkerHandModbusProtocol(transport, {39: profile})
    protocol.connect()
    now = time.monotonic_ns()
    target = MotionEnvelope(
        ("position", 39),
        39,
        LinkerHandMotion((0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
        now,
        now + 50_000_000,
        0,
    )

    protocol.write_motion((target,))
    state = protocol.read_fast_state()[39]

    write_frame = next(frame for frame in transport.writes if frame[1] == 0x10)
    assert struct.unpack(">6H", write_frame[7:-2]) == (0, 51, 102, 153, 204, 255)
    assert state.positions[0] == 0.0
    assert state.positions[1] == pytest.approx(17.0 / 255.0)


def test_modbus_write_exception_is_decoded_without_waiting_for_eight_bytes():
    profile = replace(
        linker_hand_profiles.require("L6", context="test"),
        frame_gap_s=0.0,
    )
    transport = ModbusExceptionTransport(profile.joint_count)
    protocol = LinkerHandModbusProtocol(transport, {39: profile})
    now = time.monotonic_ns()
    target = MotionEnvelope(
        ("position", 39),
        39,
        LinkerHandMotion((0.0,) * profile.joint_count),
        now,
        now + 50_000_000,
        0,
    )

    with pytest.raises(RuntimeError) as exc_info:
        protocol.write_motion((target,))

    assert isinstance(exc_info.value.__cause__, ModbusProtocolError)
