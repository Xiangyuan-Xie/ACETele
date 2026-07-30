from __future__ import annotations

import math
import struct
import time

import pytest

from acetele.hardware.serial import MotionEnvelope, RecoverableBusError, append_crc
from acetele.hardware.smart_servos.feetech import (
    FeetechInstruction,
    FeetechModbusBusProtocol,
    FeetechModbusMotion,
    FeetechPacketBusProtocol,
    FeetechPacketCodec,
    FeetechPacketMotion,
    FeetechPhysicalLayer,
    feetech_modbus_profiles,
    feetech_packet_profiles,
)


def _status(servo_id: int, parameters: bytes = b"", error: int = 0) -> bytes:
    length = len(parameters) + 2
    body = bytes((servo_id, length, error)) + parameters
    return b"\xff\xff" + body + bytes((FeetechPacketCodec.checksum(body),))


def test_packet_codec_matches_official_sdk_frames():
    assert FeetechPacketCodec.encode_ping(1) == bytes.fromhex("ff ff 01 02 01 fb")
    assert FeetechPacketCodec.encode_read(1, 56, 2) == bytes.fromhex(
        "ff ff 01 04 02 38 02 be"
    )
    assert FeetechPacketCodec.encode_write(1, 42, b"\x00\x08") == bytes.fromhex(
        "ff ff 01 05 03 2a 00 08 c4"
    )


def test_packet_codec_signed_magnitude_round_trip_and_sync_limits():
    for value in (-32767, -1, 0, 1, 32767):
        encoded = FeetechPacketCodec.encode_signed_magnitude(value)
        assert FeetechPacketCodec.decode_signed_magnitude(encoded) == value

    with pytest.raises(ValueError, match="equal-length"):
        FeetechPacketCodec.encode_sync_write(41, {1: b"\x00", 2: b"\x00\x01"})


def test_packet_profiles_keep_ttl_and_rs485_identity_separate():
    assert (
        feetech_packet_profiles.require("HL3960", context="test").physical_layer
        == FeetechPhysicalLayer.TTL
    )
    sms = feetech_packet_profiles.require("SM8512BL", context="test")
    assert sms.physical_layer == FeetechPhysicalLayer.RS485
    assert sms.model_number == 11272


class PacketTransport:
    def __init__(
        self,
        models: dict[int, int],
        *,
        position: int = 1024,
        reserved_register_value: int = 0,
    ) -> None:
        self.models = models
        self.position = position
        self.reserved_register_value = reserved_register_value
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
        servo_id = frame[2]
        instruction = FeetechInstruction(frame[4])
        parameters = frame[5:-1]
        if instruction == FeetechInstruction.READ:
            address, count = parameters
            assert address == 3 and count == 2
            self.responses.extend(
                _status(servo_id, FeetechPacketCodec.word(self.models[servo_id]))
            )
        elif instruction == FeetechInstruction.SYNC_READ:
            address, count, *servo_ids = parameters
            assert (address, count) == (56, 15)
            for state_id in servo_ids:
                state = bytearray(15)
                state[0:2] = FeetechPacketCodec.word(
                    FeetechPacketCodec.encode_signed_magnitude(self.position)
                )
                state[2:4] = FeetechPacketCodec.word(10)
                state[4:6] = FeetechPacketCodec.word(100)
                state[6] = 120
                state[7] = 35
                for offset in (8, 9, 11, 12):
                    state[offset] = self.reserved_register_value
                state[13:15] = FeetechPacketCodec.word(20)
                self.responses.extend(_status(state_id, bytes(state)))

    def read_exact(self, count, *, deadline_ns):
        if len(self.responses) < count:
            raise TimeoutError("no FEETECH packet response")
        result = bytes(self.responses[:count])
        del self.responses[:count]
        return result


def _packet_motion(servo_id: int, position_rad: float) -> MotionEnvelope:
    now = time.monotonic_ns()
    return MotionEnvelope(
        ("position", servo_id),
        servo_id,
        FeetechPacketMotion(position_rad),
        now,
        now + 50_000_000,
        0,
    )


def test_packet_protocol_checks_identity_and_requires_explicit_enable():
    profile = feetech_packet_profiles.require("SM8512BL", context="test")
    transport = PacketTransport({1: 11272})
    protocol = FeetechPacketBusProtocol(transport, {1: profile})
    protocol.connect()

    with pytest.raises(RecoverableBusError, match="software-disabled"):
        protocol.write_motion((_packet_motion(1, math.pi / 2.0),))

    with pytest.raises(RuntimeError, match="complete state sample"):
        protocol.execute_safety("set_enabled", True)
    protocol.read_fast_state()
    protocol.execute_safety("set_enabled", True)
    protocol.write_motion((_packet_motion(1, math.pi / 2.0),))
    state = protocol.read_fast_state()[1]

    sync_motion = next(
        frame
        for frame in transport.writes
        if frame[4] == FeetechInstruction.SYNC_WRITE and frame[5] == 41
    )
    assert sync_motion[9:11] == FeetechPacketCodec.word(1024)
    assert state.position_rad == pytest.approx(math.pi / 2.0)
    assert state.current_a == pytest.approx(0.13)


def test_packet_protocol_ignores_nonzero_reserved_state_registers():
    profile = feetech_packet_profiles.require("HL3960", context="test")
    transport = PacketTransport({1: 1234}, reserved_register_value=0xA5)
    protocol = FeetechPacketBusProtocol(transport, {1: profile})

    protocol.connect()
    state = protocol.read_fast_state()[1]

    assert state.status == 0


def test_hls_profile_records_unverified_model_number_from_hardware():
    profile = feetech_packet_profiles.require("HL3960", context="test")
    transport = PacketTransport({1: 1234})
    protocol = FeetechPacketBusProtocol(transport, {1: profile})

    protocol.connect()
    protocol.read_fast_state()

    assert protocol.read_slow_state()[1].model_number == 1234
    assert transport.connected


def test_hls_profile_checks_explicit_model_number_when_configured():
    profile = feetech_packet_profiles.require("HL3960", context="test")
    transport = PacketTransport({1: 1234})
    protocol = FeetechPacketBusProtocol(
        transport,
        {1: profile},
        expected_model_numbers={1: 4321},
    )

    with pytest.raises(RuntimeError, match="reports model 1234"):
        protocol.connect()

    assert not transport.connected


def test_packet_protocol_uses_nearest_multiturn_position_target():
    profile = feetech_packet_profiles.require("HL3960", context="test")
    transport = PacketTransport({1: 1234}, position=5000)
    protocol = FeetechPacketBusProtocol(transport, {1: profile})
    protocol.connect()
    protocol.read_fast_state()
    protocol.execute_safety("set_enabled", True)

    protocol.write_motion((_packet_motion(1, 1000 * 2.0 * math.pi / 4096),))

    sync_motion = transport.writes[-1]
    encoded = FeetechPacketCodec.decode_word(sync_motion[9:11])
    assert FeetechPacketCodec.decode_signed_magnitude(encoded) == 5096


def test_packet_protocol_rejects_exhausted_multiturn_position_range():
    profile = feetech_packet_profiles.require("HL3960", context="test")
    transport = PacketTransport({1: 1234}, position=-32623)
    protocol = FeetechPacketBusProtocol(transport, {1: profile})
    protocol.connect()
    protocol.read_fast_state()
    protocol.execute_safety("set_enabled", True)

    with pytest.raises(RecoverableBusError, match="motion write failed") as caught:
        protocol.write_motion((_packet_motion(1, -1022 * 2.0 * math.pi / 4096),))

    assert "range is exhausted" in str(caught.value.__cause__)


class ModbusTransport:
    def __init__(self) -> None:
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
        if function == 0x03:
            address, count = struct.unpack(">HH", frame[2:6])
            if address == 0:
                values = (2008, 2005)
            else:
                assert (address, count) == (256, 8)
                values = (0, 1024, 2, 100, 120, 35, 0, 20)
            payload = bytes((slave_id, function, count * 2)) + struct.pack(
                f">{count}H", *values
            )
            self.responses.extend(append_crc(payload))
        elif function == 0x10:
            self.responses.extend(append_crc(frame[:6]))

    def read_exact(self, count, *, deadline_ns):
        if len(self.responses) < count:
            raise TimeoutError("no FEETECH Modbus response")
        result = bytes(self.responses[:count])
        del self.responses[:count]
        return result


def _modbus_motion(slave_id: int, position_rad: float) -> MotionEnvelope:
    now = time.monotonic_ns()
    return MotionEnvelope(
        ("position", slave_id),
        slave_id,
        FeetechModbusMotion(position_rad),
        now,
        now + 50_000_000,
        0,
    )


def test_modbus_protocol_uses_documented_control_and_feedback_registers():
    profile = feetech_modbus_profiles.require("SM29-24", context="test")
    transport = ModbusTransport()
    protocol = FeetechModbusBusProtocol(transport, {1: profile})
    protocol.connect()
    with pytest.raises(RuntimeError, match="complete state sample"):
        protocol.execute_safety("set_enabled", True)
    protocol.read_fast_state()
    protocol.execute_safety("set_enabled", True)
    protocol.write_motion((_modbus_motion(1, math.pi / 2.0),))
    state = protocol.read_fast_state()[1]

    goal_write = next(
        frame
        for frame in transport.writes
        if frame[1] == 0x10 and struct.unpack(">H", frame[2:4])[0] == 128
    )
    assert struct.unpack(">H", goal_write[7:9])[0] == 1024
    assert state.position_rad == pytest.approx(math.pi / 2.0)
    assert state.current_a == pytest.approx(0.13)
