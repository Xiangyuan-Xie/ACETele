"""Pinned FashionStar RS485 profiles and firmware capability thresholds."""

from __future__ import annotations

from dataclasses import dataclass

from acetele.hardware.profiles import ProfileRegistry, ProtocolSource


@dataclass(frozen=True)
class FashionStarServoProfile:
    """Protocol behavior for one exact RS485 servo model."""

    model: str
    multi_turn: bool
    minimum_command_interval_s: float
    sync_firmware_minimum: int
    source: ProtocolSource

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("FashionStar profile model must be a non-empty string")
        if type(self.multi_turn) is not bool:
            raise ValueError("FashionStar profile multi_turn must be a boolean")
        if self.minimum_command_interval_s <= 0.0:
            raise ValueError("FashionStar command interval must be positive")
        if type(self.sync_firmware_minimum) is not int or self.sync_firmware_minimum <= 0:
            raise ValueError("FashionStar sync firmware minimum must be positive")

    def supports_sync(self, firmware_version: int | None) -> bool:
        """Return whether documented synchronous commands are safe to use."""

        return (
            type(firmware_version) is int
            and firmware_version >= self.sync_firmware_minimum
        )


def _build_profiles() -> ProfileRegistry[FashionStarServoProfile]:
    """Build the documented multi-turn RS485 model catalog."""

    source = ProtocolSource(
        "https://pypi.org/project/fashionstar-uart-sdk/1.3.12/",
        "fashionstar-uart-sdk 1.3.12",
        "1c510512bc2a485a562084eb60570527b92231626b2ab36dd97292467241ebed",
    )
    models = (
        "HX8-R28H-M",
        "HX8-R28W-M",
        "HX8-R29H-M",
        "HX8-R29W-M",
        "HX8-R50H-M",
        "HX8-R50W-M",
        "HX8-R51H-M",
        "HX8-R51W-M",
        "RX18-R100H-M",
        "RX18-R100W-M",
        "RX18-R101H-M",
        "RX18-R101W-M",
    )
    return ProfileRegistry(
        (
            model,
            FashionStarServoProfile(
                model=model,
                multi_turn=True,
                minimum_command_interval_s=0.005,
                sync_firmware_minimum=316,
                source=source,
            ),
        )
        for model in models
    )


fashionstar_rs485_profiles = _build_profiles()


__all__ = ["fashionstar_rs485_profiles", "FashionStarServoProfile"]
