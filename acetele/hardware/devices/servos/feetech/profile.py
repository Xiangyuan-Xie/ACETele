"""Exact FEETECH model profiles and documented SI/register conversions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from acetele.hardware.devices.profile import ProfileRegistry, ProtocolSource


class FeetechPacketFamily(str, Enum):
    """Packet-compatible servo family with a distinct memory map."""

    HLS = "hls"
    SMS = "sms"


class FeetechPhysicalLayer(str, Enum):
    """Electrical layer required by a packet profile."""

    TTL = "ttl"
    RS485 = "rs485"


@dataclass(frozen=True)
class FeetechPacketServoProfile:
    """HLS/SMS packet profile; no nearest-model fallback is permitted."""

    model: str
    family: FeetechPacketFamily
    physical_layer: FeetechPhysicalLayer
    model_number: int | None
    counts_per_revolution: int
    velocity_unit_rad_s: float
    acceleration_unit_rad_s2: float
    current_unit_a: float
    default_velocity_raw: int
    default_acceleration_raw: int
    default_goal_torque_raw: int | None
    source: ProtocolSource
    torque_constant_kgcm_per_a: float | None = None
    no_load_current_a: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("FEETECH profile model must be a non-empty string")
        if not isinstance(self.family, FeetechPacketFamily):
            raise ValueError("FEETECH profile family is invalid")
        if not isinstance(self.physical_layer, FeetechPhysicalLayer):
            raise ValueError("FEETECH profile physical layer is invalid")
        if self.model_number is not None and (
            type(self.model_number) is not int or not 0 <= self.model_number <= 0xFFFF
        ):
            raise ValueError("FEETECH model number must be a uint16 or None")
        if type(self.counts_per_revolution) is not int or self.counts_per_revolution <= 0:
            raise ValueError("FEETECH counts per revolution must be positive")
        for name in (
            "velocity_unit_rad_s",
            "acceleration_unit_rad_s2",
            "current_unit_a",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"FEETECH {name} must be finite and positive")
        if not 0 <= self.default_velocity_raw <= 0x7FFF:
            raise ValueError("FEETECH default velocity is outside the register range")
        if not 0 <= self.default_acceleration_raw <= 0xFF:
            raise ValueError("FEETECH default acceleration is outside the register range")
        if self.family == FeetechPacketFamily.HLS:
            if (
                type(self.default_goal_torque_raw) is not int
                or not 1 <= self.default_goal_torque_raw <= 0x7FFF
            ):
                raise ValueError("FEETECH HLS default goal torque must be positive")
        elif self.default_goal_torque_raw is not None:
            raise ValueError("FEETECH SMS profiles cannot define HLS goal torque")
        self._validate_torque_estimator()

    def _validate_torque_estimator(self) -> None:
        torque_constant = self.torque_constant_kgcm_per_a
        no_load_current = self.no_load_current_a
        if (torque_constant is None) != (no_load_current is None):
            raise ValueError("FEETECH torque estimator parameters must be provided together")
        if torque_constant is not None and no_load_current is not None and (
            not math.isfinite(torque_constant)
            or torque_constant <= 0.0
            or not math.isfinite(no_load_current)
            or no_load_current < 0.0
        ):
            raise ValueError("FEETECH torque estimator parameters are invalid")


@dataclass(frozen=True)
class FeetechModbusServoProfile:
    """FEETECH Modbus profile pinned to firmware and servo versions."""

    model: str
    firmware_version: int
    servo_version: int
    counts_per_revolution: int
    current_unit_a: float
    velocity_unit_rad_s: float
    acceleration_unit_rad_s2: float
    default_velocity_raw: int
    default_acceleration_raw: int
    default_torque_limit_raw: int
    source: ProtocolSource
    torque_constant_kgcm_per_a: float | None = None
    no_load_current_a: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("FEETECH Modbus model must be a non-empty string")
        for name in (
            "firmware_version",
            "servo_version",
            "counts_per_revolution",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"FEETECH Modbus {name} must be positive")
        for name in (
            "current_unit_a",
            "velocity_unit_rad_s",
            "acceleration_unit_rad_s2",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"FEETECH Modbus {name} must be finite and positive")
        for name in (
            "default_velocity_raw",
            "default_acceleration_raw",
            "default_torque_limit_raw",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 0xFFFF:
                raise ValueError(f"FEETECH Modbus {name} must be a uint16")
        torque_constant = self.torque_constant_kgcm_per_a
        no_load_current = self.no_load_current_a
        if (torque_constant is None) != (no_load_current is None):
            raise ValueError("FEETECH Modbus torque estimator parameters must be provided together")
        if torque_constant is not None and no_load_current is not None and (
            not math.isfinite(torque_constant)
            or torque_constant <= 0.0
            or not math.isfinite(no_load_current)
            or no_load_current < 0.0
        ):
            raise ValueError("FEETECH Modbus torque estimator parameters are invalid")


def _build_profile_registries() -> tuple[
    ProfileRegistry[FeetechPacketServoProfile],
    ProfileRegistry[FeetechModbusServoProfile],
]:
    """Build the public catalogs while keeping construction details local."""

    hls_source = ProtocolSource(
        "https://gitee.com/ftservo/FTServo_Arduino/raw/main/src/HLSCL.h",
        "Official FTServo HLS application API 2024-11-21",
        "9d2659439ba418270de2613d99d720d254eb3bded4e28ec7a0c173825e9d2d4b",
    )
    sms_source = ProtocolSource(
        "https://gitee.com/ftservo/FTServo_Arduino/raw/main/src/SMS_STS.h",
        "Official FTServo SMS/STS application API 2024-11-21",
        "3267f608ce6123570fd2033526d1749f2783baf3e401757b83028a24b0b76899",
    )
    modbus_source = ProtocolSource(
        "https://www.feetechrc.com/Data/feetechrc/upload/file/20240702/"
        "MODBUS%20RTU%20%E5%86%85%E5%AD%98%E8%A1%A8.pdf",
        "SM2924BLMB register table 2022-07-15",
        "70083087980f7c1f889071cb63b1b14bf3a4c2746e6d6d2c6c0617f7e64bd5d4",
    )
    velocity_unit = 0.732 * math.pi / 30.0
    acceleration_unit = math.radians(8.7)
    current_unit = 0.0065

    def hls_profile(
        model: str,
        *,
        velocity: int,
        acceleration: int,
        goal_torque: int,
        torque_constant_kgcm_per_a: float,
        no_load_current_a: float,
    ) -> FeetechPacketServoProfile:
        """Create an HLS profile from shared register units and model tuning."""

        return FeetechPacketServoProfile(
            model=model,
            family=FeetechPacketFamily.HLS,
            physical_layer=FeetechPhysicalLayer.TTL,
            model_number=None,
            counts_per_revolution=4096,
            velocity_unit_rad_s=velocity_unit,
            acceleration_unit_rad_s2=acceleration_unit,
            current_unit_a=current_unit,
            # These raw values were captured from clean power-on hardware. They are
            # deliberately immutable profile data: reading SRAM during startup could
            # learn values written by the last run.
            default_velocity_raw=velocity,
            default_acceleration_raw=acceleration,
            default_goal_torque_raw=goal_torque,
            source=hls_source,
            torque_constant_kgcm_per_a=torque_constant_kgcm_per_a,
            no_load_current_a=no_load_current_a,
        )

    packet_profiles = ProfileRegistry(
        (
            (
                "HL3960",
                hls_profile(
                    "HL3960",
                    velocity=100,
                    acceleration=0,
                    goal_torque=400,
                    torque_constant_kgcm_per_a=14.84,
                    no_load_current_a=0.300,
                ),
            ),
            (
                "HL3950",
                hls_profile(
                    "HL3950",
                    velocity=110,
                    acceleration=0,
                    goal_torque=1000,
                    torque_constant_kgcm_per_a=20.8,
                    no_load_current_a=0.330,
                ),
            ),
            (
                "HL3930",
                hls_profile(
                    "HL3930",
                    velocity=100,
                    acceleration=250,
                    goal_torque=1000,
                    torque_constant_kgcm_per_a=12.5,
                    no_load_current_a=0.150,
                ),
            ),
            (
                "HL3915",
                hls_profile(
                    "HL3915",
                    velocity=250,
                    acceleration=0,
                    goal_torque=500,
                    torque_constant_kgcm_per_a=9.3,
                    no_load_current_a=0.260,
                ),
            ),
            (
                "SM8512BL",
                FeetechPacketServoProfile(
                    model="SM8512BL",
                    family=FeetechPacketFamily.SMS,
                    physical_layer=FeetechPhysicalLayer.RS485,
                    model_number=11272,
                    counts_per_revolution=4096,
                    velocity_unit_rad_s=velocity_unit,
                    acceleration_unit_rad_s2=acceleration_unit,
                    current_unit_a=current_unit,
                    default_velocity_raw=110,
                    default_acceleration_raw=0,
                    default_goal_torque_raw=None,
                    source=sms_source,
                ),
            ),
        )
    )
    modbus_profiles = ProfileRegistry(
        (
            (
                "SM29-24",
                FeetechModbusServoProfile(
                    model="SM29-24",
                    firmware_version=2008,
                    servo_version=2005,
                    counts_per_revolution=4096,
                    current_unit_a=current_unit,
                    velocity_unit_rad_s=50.0 * 2.0 * math.pi / 4096.0,
                    acceleration_unit_rad_s2=100.0 * 2.0 * math.pi / 4096.0,
                    default_velocity_raw=100,
                    default_acceleration_raw=0,
                    default_torque_limit_raw=1000,
                    source=modbus_source,
                ),
            ),
        )
    )
    return packet_profiles, modbus_profiles


feetech_packet_profiles, feetech_modbus_profiles = _build_profile_registries()


__all__ = [
    "feetech_modbus_profiles",
    "feetech_packet_profiles",
    "FeetechModbusServoProfile",
    "FeetechPacketFamily",
    "FeetechPacketServoProfile",
    "FeetechPhysicalLayer",
]
