"""Shared bus scheduling, transport, and wire-protocol infrastructure."""

from acetele.hardware.buses.actor import (
    BusActor,
    BusActorDiagnostics,
    BusError,
    BusProtocol,
    DeviceEnableRequest,
    FatalBusError,
    MotionCommitGate,
    MotionEnvelope,
    RecoverableBusError,
    resolve_device_enable_request,
)
from acetele.hardware.buses.budget import BusBudget, calculate_bus_budget
from acetele.hardware.buses.modbus_rtu import (
    ModbusProtocolError,
    append_crc,
    crc16,
    decode_read_holding_registers,
    decode_read_input_registers,
    decode_write_registers_response,
    encode_read_holding_registers,
    encode_read_input_registers,
    encode_write_registers,
)
from acetele.hardware.buses.serial import SerialDirectionControl, SerialTransport

__all__ = [
    "BusActor",
    "BusActorDiagnostics",
    "BusBudget",
    "BusError",
    "BusProtocol",
    "DeviceEnableRequest",
    "FatalBusError",
    "ModbusProtocolError",
    "MotionCommitGate",
    "MotionEnvelope",
    "RecoverableBusError",
    "SerialDirectionControl",
    "SerialTransport",
    "append_crc",
    "calculate_bus_budget",
    "crc16",
    "decode_read_holding_registers",
    "decode_read_input_registers",
    "decode_write_registers_response",
    "encode_read_holding_registers",
    "encode_read_input_registers",
    "encode_write_registers",
    "resolve_device_enable_request",
]
