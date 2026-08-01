"""Immutable specifications for one physical control bus."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BusType(str, Enum):
    """Supported protocol families and physical transport semantics."""

    FEETECH_PACKET = "feetech_packet"
    FEETECH_MODBUS_RTU = "feetech_modbus_rtu"
    FASHIONSTAR_RS485 = "fashionstar_rs485"
    LINKER_HAND_RS485 = "linker_hand_rs485"


class DirectionControl(str, Enum):
    """How an RS485 adapter switches between transmit and receive."""

    AUTO = "auto"
    RTS = "rts"


@dataclass(frozen=True)
class BusSpec:
    """Static configuration for one physical port and its bus actor."""

    name: str
    type: BusType
    port: str
    baudrate: int
    cycle_hz: float
    direction_control: DirectionControl = DirectionControl.AUTO
    physical_layer: Optional[str] = None
    family: Optional[str] = None
    max_utilization: float = 0.70
    external_estop: bool = False
    allow_unverified_identity: bool = False

    def __post_init__(self) -> None:
        _name(self.name, field_name="bus name")
        if not isinstance(self.type, BusType):
            raise ValueError("bus type must be a BusType")
        _name(self.port, field_name=f"bus '{self.name}' port")
        if type(self.baudrate) is not int or self.baudrate <= 0:
            raise ValueError(f"bus '{self.name}' baudrate must be a positive integer")
        if not math.isfinite(self.cycle_hz) or self.cycle_hz <= 0.0:
            raise ValueError(f"bus '{self.name}' cycle_hz must be finite and positive")
        if not isinstance(self.direction_control, DirectionControl):
            raise ValueError(f"bus '{self.name}' direction_control must be a DirectionControl")
        if not math.isfinite(self.max_utilization) or not 0.0 < self.max_utilization <= 1.0:
            raise ValueError(f"bus '{self.name}' max_utilization must be in (0, 1]")
        if type(self.external_estop) is not bool:
            raise ValueError(f"bus '{self.name}' external_estop must be a boolean")
        if type(self.allow_unverified_identity) is not bool:
            raise ValueError(
                f"bus '{self.name}' allow_unverified_identity must be a boolean"
            )
        if self.type == BusType.FEETECH_PACKET:
            if self.physical_layer not in ("ttl", "rs485"):
                raise ValueError("feetech_packet buses require physical_layer='ttl' or 'rs485'")
            if self.family not in ("hls", "sms"):
                raise ValueError("feetech_packet buses require family='hls' or 'sms'")
            if self.family == "hls" and self.physical_layer != "ttl":
                raise ValueError("FEETECH HLS requires the TTL physical layer")
            if self.family == "sms" and self.physical_layer != "rs485":
                raise ValueError("FEETECH SMS requires the RS485 physical layer")
        elif self.physical_layer not in (None, "rs485"):
            raise ValueError(f"bus '{self.name}' uses the RS485 physical layer")


def _name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


__all__ = ["BusSpec", "BusType", "DirectionControl"]
