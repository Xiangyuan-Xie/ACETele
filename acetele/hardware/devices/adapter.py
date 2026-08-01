"""Vendor-neutral bus adapter contract and explicit protocol registry.

Adapters own every fact that depends on a device protocol: exact model profiles,
wire budgets, protocol construction, state decoding, command encoding, and hardware
fault interpretation. Runtime code consumes only the immutable :class:`AdapterPlan`
produced during side-effect-free preflight.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from acetele.core import JointCommand, JointUnit
from acetele.estimation import StateEstimatorTuning
from acetele.hardware.buses import BusBudget, BusProtocol, MotionEnvelope
from acetele.hardware.simulators import (
    MockBusProtocol,
    MockDeviceDefinition,
    MockDeviceState,
    MockMotion,
)
from acetele.specification import Backend, BusSpec, BusType, DexterousHandSpec, JointSpec


class GroupDescription(Protocol):
    """Minimum group metadata required at the hardware boundary."""

    @property
    def name(self) -> str:
        ...

    @property
    def bus(self) -> str:
        ...

    @property
    def unit(self) -> JointUnit:
        ...

    @property
    def joint_names(self) -> tuple[str, ...]:
        ...

    @property
    def joints(self) -> tuple[JointSpec, ...]:
        ...

    @property
    def travel_range_rad(self) -> Optional[float]:
        ...

    @property
    def hand(self) -> Optional[DexterousHandSpec]:
        ...

    @property
    def is_arm(self) -> bool:
        ...


TransportFactory = Callable[[str, int, Any], Any]


@dataclass(frozen=True)
class AdapterPlan:
    """Immutable, side-effect-free plan for one configured bus."""

    spec: BusSpec
    backend: Backend
    adapter: "BusAdapter"
    groups: tuple[GroupDescription, ...]
    profiles: Mapping[int, Any]
    expected_models: Mapping[int, int]
    firmware_versions: Mapping[int, int]
    budget: BusBudget
    device_models: Mapping[int, str]
    supports_software_disable: bool
    supports_verified_disable: bool
    supports_verified_identity: bool
    supports_acceleration_limits: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))
        for field_name in (
            "profiles",
            "expected_models",
            "firmware_versions",
            "device_models",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )


@dataclass(frozen=True)
class DecodedJointSample:
    """Canonical SI sample decoded from one actor snapshot."""

    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    efforts: tuple[float, ...]
    timestamp_ns: int


class BusAdapter(ABC):
    """Protocol-family port consumed by preflight and ``RobotRuntime``."""

    bus_types: tuple[BusType, ...]

    @abstractmethod
    def preflight(
        self,
        bus: BusSpec,
        groups: Sequence[GroupDescription],
        backend: Backend,
    ) -> AdapterPlan:
        """Resolve exact profiles, capabilities, and bandwidth without I/O."""

    @abstractmethod
    def create_protocol(
        self,
        plan: AdapterPlan,
        transport_factory: TransportFactory,
        *,
        clock_ns=time.monotonic_ns,
    ) -> BusProtocol:
        """Create an unconnected protocol instance for the planned bus."""

    @abstractmethod
    def decode_group(
        self,
        plan: AdapterPlan,
        group: GroupDescription,
        snapshot: Mapping[int, Any],
    ) -> DecodedJointSample:
        """Decode one group into canonical positions, velocities, and efforts."""

    @abstractmethod
    def validate_command(
        self,
        plan: AdapterPlan,
        group: GroupDescription,
        command: JointCommand,
    ) -> None:
        """Validate protocol capabilities before any payload is staged."""

    @abstractmethod
    def encode_group(
        self,
        plan: AdapterPlan,
        group: GroupDescription,
        command: JointCommand,
        position_references: Mapping[int, float],
    ) -> tuple[MotionEnvelope, ...]:
        """Encode a validated canonical command into actor envelopes."""

    def estimator_tuning(
        self,
        plan: AdapterPlan,
        group: GroupDescription,
    ) -> Optional[StateEstimatorTuning]:
        """Return measurement tuning, or ``None`` for already aggregated devices."""

        return None

    def hand_joint_names(self, hand: DexterousHandSpec) -> tuple[str, ...]:
        """Return profile-defined hand joints when this adapter supports hands."""

        raise ValueError(
            f"bus adapter for {self.bus_types[0].value} does not support dexterous hands"
        )

    def capture_position_references(
        self,
        plan: AdapterPlan,
        snapshot: Mapping[int, Any],
    ) -> Mapping[int, float]:
        """Return continuous raw positions required by command encoding."""

        return {}

    def fault_reason(
        self,
        plan: AdapterPlan,
        fast_snapshot: Any,
        slow_snapshot: Any,
    ) -> Optional[str]:
        """Translate protocol status into an actionable runtime fault."""

        if not isinstance(fast_snapshot, Mapping):
            return None
        for device_id, state in fast_snapshot.items():
            status = getattr(state, "status", None)
            if type(status) is int and status:
                return (
                    f"bus '{plan.spec.name}' device {device_id} reported "
                    f"hardware status 0x{status:02x}"
                )
        return None

    def calibration_targets(
        self,
        plan: AdapterPlan,
    ) -> Optional[Mapping[int, int]]:
        """Return raw home offsets, or ``None`` when calibration is unsupported."""

        return None

    def negotiated_budget(self, plan: AdapterPlan, protocol: BusProtocol) -> BusBudget:
        """Return the post-connect wire budget after capability negotiation."""

        return plan.budget

    @staticmethod
    def mock_protocol(
        plan: AdapterPlan,
        *,
        clock_ns=time.monotonic_ns,
    ) -> MockBusProtocol:
        """Build deterministic mock devices while retaining real profile preflight."""

        definitions: dict[int, MockDeviceDefinition] = {}
        for group in plan.groups:
            if group.hand is not None:
                definitions[group.hand.slave_id] = MockDeviceDefinition(
                    (0.0,) * len(group.joint_names)
                )
            else:
                for joint in group.joints:
                    definitions[joint.servo_id] = MockDeviceDefinition(
                        (joint.home_position_rad * joint.direction,)
                    )
        return MockBusProtocol(definitions, clock_ns=clock_ns)

    @staticmethod
    def decode_mock_group(
        group: GroupDescription,
        snapshot: Mapping[int, Any],
    ) -> Optional[DecodedJointSample]:
        """Decode a mock snapshot, returning ``None`` for a physical protocol."""

        if group.hand is not None:
            state = snapshot[group.hand.slave_id]
            if not isinstance(state, MockDeviceState):
                return None
            return DecodedJointSample(
                tuple(state.positions),
                tuple(state.velocities),
                tuple(state.efforts),
                state.timestamp_ns,
            )
        states = tuple(snapshot[joint.servo_id] for joint in group.joints)
        if not states or not all(isinstance(state, MockDeviceState) for state in states):
            return None
        return DecodedJointSample(
            tuple(state.positions[0] * joint.direction for state, joint in zip(states, group.joints)),
            tuple(state.velocities[0] * joint.direction for state, joint in zip(states, group.joints)),
            tuple(state.efforts[0] for state in states),
            min(state.timestamp_ns for state in states),
        )

    @staticmethod
    def encode_mock_group(
        group: GroupDescription,
        command: JointCommand,
    ) -> tuple[MotionEnvelope, ...]:
        """Encode canonical positions for the deterministic mock protocol."""

        if group.hand is not None:
            return (
                motion_envelope(
                    group.hand.slave_id,
                    MockMotion(tuple(command.positions)),
                    command,
                ),
            )
        targets = []
        for index, joint in enumerate(group.joints):
            position = float(command.positions[index])
            if group.travel_range_rad is not None:
                position *= group.travel_range_rad
            targets.append(
                motion_envelope(
                    joint.servo_id,
                    MockMotion((position * joint.direction,)),
                    command,
                )
            )
        return tuple(targets)

    @staticmethod
    def validate_mock_command(
        group: GroupDescription,
        command: JointCommand,
    ) -> None:
        """Apply the mock protocol's intentionally small capability surface."""

        has_velocity = command.velocity_limits is not None
        if command.acceleration_limits is not None or command.effort_limits is not None:
            raise ValueError(
                f"joint group '{group.name}' contains limits unsupported by its mock device"
            )
        if has_velocity and not group.is_arm:
            raise ValueError(
                f"joint group '{group.name}' contains limits unsupported by its mock device"
            )


class AdapterRegistry:
    """Explicit one-adapter-per-bus-type registry."""

    def __init__(self, adapters: Sequence[BusAdapter]) -> None:
        values: dict[BusType, BusAdapter] = {}
        for adapter in adapters:
            if not isinstance(adapter, BusAdapter):
                raise ValueError("adapter registry entries must implement BusAdapter")
            for bus_type in adapter.bus_types:
                if bus_type in values:
                    raise ValueError(f"duplicate adapter for bus type {bus_type.value}")
                values[bus_type] = adapter
        missing = tuple(bus_type.value for bus_type in BusType if bus_type not in values)
        if missing:
            raise ValueError("adapter registry is missing bus types: " + ", ".join(missing))
        self._adapters = MappingProxyType(values)

    def require(self, bus_type: BusType) -> BusAdapter:
        """Return the exact adapter registered for ``bus_type``."""

        if not isinstance(bus_type, BusType):
            raise ValueError("bus_type must be a BusType")
        return self._adapters[bus_type]


@lru_cache(maxsize=1)
def default_adapter_registry() -> AdapterRegistry:
    """Construct the built-in registry without importing vendors into runtime."""

    from acetele.hardware.devices.hands.linker.adapter import LinkerHandAdapter
    from acetele.hardware.devices.servos.fashionstar.adapter import FashionStarAdapter
    from acetele.hardware.devices.servos.feetech.adapter import FeetechAdapter

    return AdapterRegistry((FeetechAdapter(), FashionStarAdapter(), LinkerHandAdapter()))


def motion_envelope(
    device_id: int,
    payload: Any,
    command: JointCommand,
) -> MotionEnvelope:
    """Create an actor envelope before its current generation is attached."""

    return MotionEnvelope(
        ("position", device_id),
        device_id,
        payload,
        command.submitted_at_ns,
        command.deadline_ns,
        0,
    )


__all__ = [
    "AdapterPlan",
    "AdapterRegistry",
    "BusAdapter",
    "DecodedJointSample",
    "GroupDescription",
    "TransportFactory",
    "default_adapter_registry",
    "motion_envelope",
]
