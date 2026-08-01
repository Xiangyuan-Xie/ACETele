"""Execution backends selected by an immutable robot specification."""

from enum import Enum


class Backend(str, Enum):
    """Select whether a specification creates physical or deterministic mock buses."""

    PHYSICAL = "physical"
    MOCK = "mock"


__all__ = ["Backend"]
