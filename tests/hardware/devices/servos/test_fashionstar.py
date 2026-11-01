from __future__ import annotations

import math
import struct
import time
from dataclasses import replace

import pytest

from acetele.hardware.buses import MotionEnvelope, RecoverableBusError
from acetele.hardware.devices.servos.fashionstar import (
    FashionStarBusProtocol,
    FashionStarCommand,
    FashionStarMotion,
    FashionStarPacketCodec,
    FashionStarProtocolError,
    FashionStarStopMode,
    fashionstar_rs485_profiles,
)


def _response(command: FashionStarCommand, parameters: bytes) -> bytes:
    body = b"\x05\x1c" + bytes((command, len(parameters))) + parameters
    return body + bytes((sum(body) & 0xFF,))


def test_position_request_matches_official_protocol_example():
    frame = FashionStarPacketCodec.encode_single_turn_position(
        0,
        math.pi / 2.0,
        interval_ms=500,
    )

    assert frame == bytes.fromhex("12 4c 08 07 00 84 03 f4 01 00 00 e9")


def test_official_position_response_and_checksum_are_validated():
    packet = FashionStarPacketCodec.decode_response(
        bytes.fromhex("05 1c 08 02 00 01 2c")
    )
    assert packet.command == FashionStarCommand.SET_ANGLE
    assert packet.parameters == b"\x00\x01"

    invalid = bytearray(bytes.fromhex("05 1c 08 02 00 01 2c"))
    invalid[-1] ^= 1
    with pytest.raises(FashionStarProtocolError, match="checksum"):
        FashionStarPacketCodec.decode_response(bytes(invalid))


def test_monitor_response_converts_documented_units():
    parameters = struct.pack("<BHHHHBih", 3, 24000, 1250, 5000, 1000, 4, 900, 0)
    state = FashionStarPacketCodec.decode_monitor(
        FashionStarPacketCodec.decode_response(
            _response(FashionStarCommand.QUERY_MONITOR, parameters)
        )
    )

    assert state.servo_id == 3
    assert state.voltage_v == 24.0
    assert state.current_a == 1.25
    assert state.power_w == 5.0
    assert state.position_rad == pytest.approx(math.pi / 2.0)


def test_registry_contains_only_documented_rs485_models():
    assert len(fashionstar_rs485_profiles.names) == 12
    assert "HX8-R50W-M" in fashionstar_rs485_profiles.names
    assert all("-R" in model for model in fashionstar_rs485_profiles.names)
    with pytest.raises(ValueError, match="unsupported model"):
        fashionstar_rs485_profiles.require("RA8-R50", context="joint_1")


class FashionStarTransport:
    def __init__(
        self,
        *,
        supports_sync: bool = True,
        response_switch: int = 0,
        firmware_version: int = 316,
        firmware_versions: dict[int, int] | None = None,
    ) -> None:
        self.responses = bytearray()
        self.writes: list[bytes] = []
        self.write_deadlines: list[int] = []
        self.connected = False
        self.supports_sync = supports_sync
        self.response_switch = response_switch
        self.firmware_version = firmware_version
        self.firmware_versions = dict(firmware_versions or {})

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def cancel(self):
        pass

    def write(self, frame, *, deadline_ns):
        self.writes.append(frame)
        self.write_deadlines.append(deadline_ns)
        command = FashionStarCommand(frame[2])
        parameters = frame[4:-1]
        if command == FashionStarCommand.PING:
            self.responses.extend(_response(command, parameters))
        elif command == FashionStarCommand.READ_DATA:
            servo_id, data_id = parameters
            if data_id == FashionStarPacketCodec.response_switch_data_id:
                data = bytes((self.response_switch,))
            else:
                assert data_id == FashionStarPacketCodec.firmware_version_data_id
                data = self.firmware_versions.get(
                    servo_id,
                    self.firmware_version,
                ).to_bytes(2, "little")
            self.responses.extend(_response(command, bytes((servo_id, data_id)) + data))
        elif command == FashionStarCommand.QUERY_MONITOR:
            servo_id = parameters[0]
            monitor = struct.pack(
                "<BHHHHBih", servo_id, 24000, 100, 200, 1000, 0, 0, 0
            )
            self.responses.extend(_response(command, monitor))
        elif (
            self.supports_sync
            and command == FashionStarCommand.SYNC
            and parameters[0] == FashionStarCommand.QUERY_MONITOR
        ):
            for servo_id in parameters[3:]:
                monitor = struct.pack(
                    "<BHHHHBih", servo_id, 24000, 100, 200, 1000, 0, 0, 0
                )
                self.responses.extend(
                    _response(FashionStarCommand.QUERY_MONITOR, monitor)
                )

    def read_exact(self, count, *, deadline_ns):
        if len(self.responses) < count:
            raise TimeoutError("no response")
        value = bytes(self.responses[:count])
        del self.responses[:count]
        return value

    def discard_input(self):
        self.responses.clear()

    def read_available(self, *, maximum=4096):
        value = bytes(self.responses[:maximum])
        del self.responses[:maximum]
        return value


def _motion(servo_id: int, position: float, generation: int = 0) -> MotionEnvelope:
    now = time.monotonic_ns()
    return MotionEnvelope(
        ("position", servo_id),
        servo_id,
        FashionStarMotion(position),
        now,
        now + 50_000_000,
        generation,
    )


def _profile():
    return replace(
        fashionstar_rs485_profiles.require("HX8-R50W-M", context="test"),
        minimum_command_interval_s=1e-9,
    )


def test_read_data_codec_matches_the_official_response_switch_frame():
    request = FashionStarPacketCodec.encode_read_data(
        1,
        FashionStarPacketCodec.response_switch_data_id,
    )
    packet = FashionStarPacketCodec.decode_response(
        _response(FashionStarCommand.READ_DATA, bytes((1, 33, 0)))
    )

    assert request == b"\x12\x4c\x03\x02\x01\x21\x85"
    assert FashionStarPacketCodec.decode_read_data(
        packet,
        servo_id=1,
        data_id=33,
        size=1,
    ) == b"\x00"


def test_connect_rejects_non_interruptible_response_mode_before_motion_commands():
    transport = FashionStarTransport(response_switch=1)
    protocol = FashionStarBusProtocol(transport, {1: _profile()})

    with pytest.raises(RuntimeError, match="response_switch must be 0"):
        protocol.connect()

    assert not transport.connected
    assert [frame[2] for frame in transport.writes] == [
        FashionStarCommand.PING,
        FashionStarCommand.READ_DATA,
    ]


def test_successful_capability_probe_enables_sync_without_trusting_config():
    transport = FashionStarTransport()
    protocol = FashionStarBusProtocol(transport, {1: _profile(), 2: _profile()})
    protocol.connect()
    protocol.execute_safety("set_enabled", True)
    transport.writes.clear()

    first = _motion(1, 0.1)
    second = _motion(2, 0.2)
    protocol.write_motion((first, second))

    assert [frame[2] for frame in transport.writes] == [FashionStarCommand.SYNC]
    assert transport.write_deadlines[-1] == min(first.deadline_ns, second.deadline_ns)


def test_connect_reads_and_verifies_firmware_from_the_servo():
    transport = FashionStarTransport(firmware_version=317)
    protocol = FashionStarBusProtocol(
        transport,
        {1: _profile()},
        firmware_versions={1: 316},
    )

    with pytest.raises(RuntimeError, match="reports firmware 317, expected 316"):
        protocol.connect()

    assert not transport.connected


def test_slow_state_reports_observed_firmware_instead_of_configuration():
    transport = FashionStarTransport(firmware_version=317)
    protocol = FashionStarBusProtocol(transport, {1: _profile()})
    protocol.connect()
    protocol.read_fast_state()

    assert protocol.read_slow_state()[1]["firmware_version"] == 317


def test_failed_capability_probe_falls_back_to_individual_frames():
    transport = FashionStarTransport(supports_sync=False)
    protocol = FashionStarBusProtocol(transport, {1: _profile(), 2: _profile()})
    protocol.connect()
    protocol.execute_safety("set_enabled", True)
    transport.writes.clear()

    protocol.write_motion((_motion(1, 0.1), _motion(2, 0.2)))

    assert [frame[2] for frame in transport.writes] == [
        FashionStarCommand.SET_MULTI_TURN_ANGLE,
        FashionStarCommand.SET_MULTI_TURN_ANGLE,
    ]


def test_v316_devices_use_one_sync_frame_and_sync_monitor():
    transport = FashionStarTransport(firmware_versions={2: 400})
    protocol = FashionStarBusProtocol(
        transport,
        {1: _profile(), 2: _profile()},
        firmware_versions={1: 316, 2: 400},
    )
    protocol.connect()
    protocol.execute_safety("set_enabled", True)
    transport.writes.clear()

    protocol.write_motion((_motion(1, 0.1), _motion(2, 0.2)))
    states = protocol.read_fast_state()

    assert [frame[2] for frame in transport.writes] == [
        FashionStarCommand.SYNC,
        FashionStarCommand.SYNC,
    ]
    assert set(states) == {1, 2}
    assert states[1].timestamp_ns > 0
    assert states[1].timestamp_ns == states[2].timestamp_ns


@pytest.mark.parametrize(
    "synchronized, received_ids, missing_ids, awaiting_id",
    [
        (True, [1, 3], [2], None),
        (False, [1], [2, 3], 2),
    ],
)
def test_monitor_timeout_identifies_incomplete_devices(
    synchronized, received_ids, missing_ids, awaiting_id, caplog
):
    """A lost reply identifies incomplete samples without publishing a partial state."""

    class MissingReplyTransport(FashionStarTransport):
        drop_reply = False

        def write(self, frame, *, deadline_ns):
            super().write(frame, deadline_ns=deadline_ns)
            if not self.drop_reply:
                return
            pending = self.responses
            self.responses = bytearray()
            while pending:
                length = pending[3] + 5
                reply = bytes(pending[:length])
                del pending[:length]
                if not (
                    reply[2] == FashionStarCommand.QUERY_MONITOR and reply[4] == 2
                ):
                    self.responses.extend(reply)

    transport = MissingReplyTransport(supports_sync=synchronized)
    protocol = FashionStarBusProtocol(
        transport, {servo_id: _profile() for servo_id in (1, 2, 3)}
    )
    protocol.connect()
    # First establish a complete snapshot, then lose only device 2's response.
    assert set(protocol.read_fast_state()) == {1, 2, 3}
    transport.drop_reply = True
    with pytest.raises(RecoverableBusError) as failure:
        protocol.read_fast_state(deadline_ns=time.monotonic_ns() + 50_000_000)

    assert isinstance(failure.value.__cause__, TimeoutError)
    assert f"mode={'sync' if synchronized else 'unicast'}" in str(failure.value)
    assert f"received_ids={received_ids}" in str(failure.value)
    assert f"missing_ids={missing_ids}" in str(failure.value)
    assert f"awaiting_id={awaiting_id}" in str(failure.value)
    assert "elapsed_ms=" in caplog.text
    assert "budget_ms=" in caplog.text
    assert "cause=no response" in caplog.text
    assert set(protocol.read_slow_state()) == {1, 2, 3}


def test_emergency_stop_latches_software_disable_until_explicit_enable():
    transport = FashionStarTransport()
    protocol = FashionStarBusProtocol(transport, {1: _profile()})
    protocol.connect()

    protocol.execute_safety("emergency_stop", None)
    with pytest.raises(RecoverableBusError, match="software-disabled"):
        protocol.write_motion((_motion(1, 0.1),))

    protocol.execute_safety("set_enabled", True)
    protocol.write_motion((_motion(1, 0.1),))


@pytest.mark.parametrize("timeout_s, succeeds", [(0.05, False), (0.15, True)])
def test_delayed_monitor_uses_io_budget_and_conservative_sample_time(timeout_s, succeeds):
    """A 121 ms reply fits the EAA budget without being timestamped as brand-new."""
    clock = [1_000_000_000]

    class DelayedTransport(FashionStarTransport):
        delay_ns = 0
        available_ns = 0

        def write(self, frame, *, deadline_ns):
            super().write(frame, deadline_ns=deadline_ns)
            if frame[2] == FashionStarCommand.QUERY_MONITOR:
                self.available_ns = clock[0] + self.delay_ns

        def read_exact(self, count, *, deadline_ns):
            if self.available_ns > clock[0]:
                if self.available_ns >= deadline_ns:
                    clock[0] = deadline_ns
                    raise TimeoutError("delayed monitor")
                clock[0] = self.available_ns
            return super().read_exact(count, deadline_ns=deadline_ns)

    transport = DelayedTransport()
    protocol = FashionStarBusProtocol(
        transport, {1: _profile()}, operation_timeout_s=timeout_s,
        clock_ns=lambda: clock[0],
    )
    protocol.connect()
    transport.delay_ns = 121_000_000
    start = clock[0]
    if succeeds:
        states = protocol.read_fast_state(deadline_ns=start + round(timeout_s * 1e9))
        assert states[1].timestamp_ns == start
        assert clock[0] - states[1].timestamp_ns == 121_000_000
    else:
        with pytest.raises(RecoverableBusError, match="monitor read failed"):
            protocol.read_fast_state(deadline_ns=start + round(timeout_s * 1e9))


def test_monitor_retry_discards_late_frame_before_new_query():
    """A valid old packet arriving during recovery cannot replace a new sample."""

    class LateTransport(FashionStarTransport):
        omit_reply = False
        delayed_frame = b""

        def write(self, frame, *, deadline_ns):
            super().write(frame, deadline_ns=deadline_ns)
            if self.omit_reply and frame[2] == FashionStarCommand.QUERY_MONITOR:
                self.responses.clear()

        def read_available(self, *, maximum=4096):
            if self.delayed_frame:
                value, self.delayed_frame = self.delayed_frame, b""
                return value
            return super().read_available(maximum=maximum)

    transport = LateTransport()
    protocol = FashionStarBusProtocol(transport, {1: _profile()}, operation_timeout_s=0.15)
    protocol.connect()
    transport.omit_reply = True
    with pytest.raises(RecoverableBusError):
        protocol.read_fast_state()
    transport.omit_reply = False
    transport.delayed_frame = _response(
        FashionStarCommand.QUERY_MONITOR,
        struct.pack("<BHHHHBih", 1, 24000, 100, 200, 1000, 0, 900, 0),
    )
    transport.writes.clear()
    states = protocol.read_fast_state(deadline_ns=time.monotonic_ns() + 150_000_000)
    assert states[1].position_rad == 0.0
    assert transport.delayed_frame == b""
    assert [frame[2] for frame in transport.writes] == [FashionStarCommand.QUERY_MONITOR]


def test_adapter_applies_custom_io_timeout_only_to_its_protocol():
    """A scoped adapter can change EAA timing without changing default protocols."""
    from types import SimpleNamespace
    from acetele.hardware.devices.servos.fashionstar.adapter import FashionStarAdapter
    from acetele.specification import Backend, DirectionControl

    plan = SimpleNamespace(
        backend=Backend.PHYSICAL,
        spec=SimpleNamespace(port="fake", baudrate=115200, direction_control=DirectionControl.AUTO),
        profiles={1: _profile()}, firmware_versions={1: 316},
    )
    factory = lambda *args: FashionStarTransport()
    default = FashionStarAdapter().create_protocol(plan, factory)
    tuned = FashionStarAdapter(operation_timeout_s=0.15).create_protocol(plan, factory)
    assert default.operation_timeout_ns == 50_000_000
    assert tuned.operation_timeout_ns == 150_000_000


def test_continuously_noisy_input_cannot_start_a_new_monitor_query():
    """Recovery remains bounded and sends no query while old input is arriving."""

    class NoisyTransport(FashionStarTransport):
        noisy = False

        def read_available(self, *, maximum=4096):
            if self.noisy:
                return b"old bytes"
            return super().read_available(maximum=maximum)

    transport = NoisyTransport()
    protocol = FashionStarBusProtocol(transport, {1: _profile()}, operation_timeout_s=0.15)
    protocol.connect()
    # Fail one complete transaction and then model a stream of delayed bytes.
    original_read = transport.read_exact
    transport.read_exact = lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("lost reply"))
    with pytest.raises(RecoverableBusError):
        protocol.read_fast_state()
    transport.read_exact = original_read
    transport.noisy = True
    transport.writes.clear()
    started = time.monotonic()
    with pytest.raises(RecoverableBusError) as failure:
        protocol.read_fast_state(deadline_ns=time.monotonic_ns() + 150_000_000)
    assert "did not become quiet" in str(failure.value.__cause__)
    assert time.monotonic() - started < 0.5
    assert not transport.writes


def test_enable_establishes_hardware_hold_before_opening_motion_path():
    transport = FashionStarTransport()
    protocol = FashionStarBusProtocol(transport, {1: _profile()})
    protocol.connect()
    transport.writes.clear()

    protocol.execute_safety("set_enabled", True)

    assert len(transport.writes) == 1
    assert transport.writes[0][2] == FashionStarCommand.STOP_CONTROL
    assert transport.writes[0][5] == FashionStarStopMode.HOLD
