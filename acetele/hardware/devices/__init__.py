"""Vendor device adapters and immutable hardware profiles."""

from acetele.hardware.devices.adapter import (
    AdapterPlan,
    AdapterRegistry,
    AutomaticFaultAction,
    BusAdapter,
    DecodedJointSample,
    HardwareFault,
    default_adapter_registry,
)

__all__ = [
    "AutomaticFaultAction",
    "AdapterPlan",
    "AdapterRegistry",
    "BusAdapter",
    "DecodedJointSample",
    "HardwareFault",
    "default_adapter_registry",
]
