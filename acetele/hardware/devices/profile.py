"""Auditable vendor profile provenance and immutable profile registries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, Iterable, Mapping, TypeVar


@dataclass(frozen=True)
class ProtocolSource:
    """Pinned source document used to justify one protocol/profile implementation."""

    url: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ValueError("protocol source URL must use HTTPS")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("protocol source version must be a non-empty string")
        if not isinstance(self.sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.sha256
        ):
            raise ValueError("protocol source sha256 must contain 64 lowercase hex digits")


ProfileT = TypeVar("ProfileT")


class ProfileRegistry(Generic[ProfileT]):
    """Immutable exact-name profile lookup; unknown hardware is never guessed."""

    def __init__(self, profiles: Iterable[tuple[str, ProfileT]]) -> None:
        values = tuple(profiles)
        names = tuple(name for name, _ in values)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("profile names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("profile names must be unique")
        self._profiles: Mapping[str, ProfileT] = MappingProxyType(dict(values))

    @property
    def names(self) -> tuple[str, ...]:
        """Return supported profile names in registration order."""

        return tuple(self._profiles)

    def require(self, name: str, *, context: str) -> ProfileT:
        """Return an exact profile or raise a contextual configuration error."""

        try:
            return self._profiles[name]
        except KeyError as exc:
            supported = ", ".join(self.names) or "none"
            raise ValueError(
                f"{context} uses unsupported model '{name}'; supported profiles: {supported}"
            ) from exc


__all__ = ["ProfileRegistry", "ProtocolSource"]
