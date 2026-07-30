"""Document-pinned Linker Hand RS485 register profiles for supported models."""

from __future__ import annotations

from dataclasses import dataclass

from acetele.hardware.profiles import ProfileRegistry, ProtocolSource


@dataclass(frozen=True)
class LinkerHandProfile:
    """Model-specific joint order, register map, timing, and provenance."""

    model: str
    joint_names: tuple[str, ...]
    position_address: int
    torque_address: int
    speed_address: int
    temperature_address: int
    error_address: int
    version_address: int
    version_count: int
    frame_gap_s: float
    source: ProtocolSource

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        if not self.model or not names or len(set(names)) != len(names):
            raise ValueError("Linker Hand profile requires a model and unique joint names")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Linker Hand profile joint names must be non-empty strings")
        for field_name in (
            "position_address",
            "torque_address",
            "speed_address",
            "temperature_address",
            "error_address",
            "version_address",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Linker Hand {field_name} must be a non-negative integer")
        if type(self.version_count) is not int or self.version_count <= 0:
            raise ValueError("Linker Hand version_count must be positive")
        if self.frame_gap_s < 0.0:
            raise ValueError("Linker Hand frame_gap_s cannot be negative")
        object.__setattr__(self, "joint_names", names)

    @property
    def joint_count(self) -> int:
        """Return the number of normalized hand joints."""

        return len(self.joint_names)

    @property
    def fast_register_count(self) -> int:
        """Return contiguous position-through-speed register count."""

        return self.speed_address + self.joint_count - self.position_address

    @property
    def wire_bytes_per_cycle(self) -> int:
        """Estimate one position write plus one fast-state read."""

        write_request = 9 + self.joint_count * 2
        write_response = 8
        read_request = 8
        read_response = 5 + self.fast_register_count * 2
        return write_request + write_response + read_request + read_response

    @property
    def turnaround_s_per_cycle(self) -> float:
        """Return required request/response frame-gap overhead."""

        return self.frame_gap_s * 2.0


def _build_profiles() -> ProfileRegistry[LinkerHandProfile]:
    """Build the Linker catalog from one pinned SDK revision."""

    source_url = "https://github.com/linker-bot/linkerhand-python-sdk"
    source_version = "3.1.1+fbec1057"
    six_joint_names = (
        "thumb_pitch",
        "thumb_yaw",
        "index_pitch",
        "middle_pitch",
        "ring_pitch",
        "little_pitch",
    )
    return ProfileRegistry(
        (
            (
                "O6",
                LinkerHandProfile(
                    "O6",
                    six_joint_names,
                    0,
                    6,
                    12,
                    18,
                    24,
                    30,
                    15,
                    0.030,
                    ProtocolSource(
                        source_url,
                        source_version,
                        "320b59543b74ba77f96a2721a40b16468273af2458091a77f5c81d8f6aa01c16",
                    ),
                ),
            ),
            (
                "L6",
                LinkerHandProfile(
                    "L6",
                    six_joint_names,
                    0,
                    6,
                    12,
                    18,
                    24,
                    148,
                    8,
                    0.006,
                    ProtocolSource(
                        source_url,
                        source_version,
                        "2f2bb25f22fd1512e4a038f7cba498ff4df3fa4d0009a06ec5e9dec07507c517",
                    ),
                ),
            ),
            (
                "L7",
                LinkerHandProfile(
                    "L7",
                    six_joint_names + ("thumb_roll",),
                    0,
                    7,
                    14,
                    21,
                    28,
                    153,
                    6,
                    0.005,
                    ProtocolSource(
                        source_url,
                        source_version,
                        "dcb1e324666ed6eff01b42bbf80fed3ecf5285a8bbde13f220eec30fc96b0359",
                    ),
                ),
            ),
            (
                "L10",
                LinkerHandProfile(
                    "L10",
                    (
                        "thumb_cmc_pitch",
                        "thumb_cmc_roll",
                        "index_mcp_pitch",
                        "middle_mcp_pitch",
                        "ring_mcp_pitch",
                        "little_mcp_pitch",
                        "index_mcp_roll",
                        "ring_mcp_roll",
                        "little_mcp_roll",
                        "thumb_cmc_yaw",
                    ),
                    0,
                    10,
                    20,
                    40,
                    50,
                    158,
                    6,
                    0.005,
                    ProtocolSource(
                        source_url,
                        source_version,
                        "3150fef977e849d3519853c5dd6de1c0c8963eba9e498fba3f0ec8b38aeb6e94",
                    ),
                ),
            ),
        )
    )


linker_hand_profiles = _build_profiles()


__all__ = ["linker_hand_profiles", "LinkerHandProfile"]
