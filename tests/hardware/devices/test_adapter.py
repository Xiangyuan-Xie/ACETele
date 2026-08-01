import pytest

from acetele.hardware.devices import AdapterRegistry, default_adapter_registry
from acetele.hardware.devices.hands.linker.adapter import LinkerHandAdapter
from acetele.specification import BusType


@pytest.mark.parametrize("bus_type", tuple(BusType))
def test_default_registry_has_one_adapter_for_every_bus_type(bus_type):
    adapter = default_adapter_registry().require(bus_type)

    assert bus_type in adapter.bus_types


def test_adapter_registry_rejects_incomplete_registration():
    with pytest.raises(ValueError, match="missing bus types"):
        AdapterRegistry((LinkerHandAdapter(),))
