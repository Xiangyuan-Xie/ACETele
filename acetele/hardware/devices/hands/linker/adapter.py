"""Linker Hand RS485 adapter for the vendor-neutral runtime boundary."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from acetele.core import JointCommand
from acetele.hardware.buses import (
    BusProtocol,
    MotionEnvelope,
    SerialDirectionControl,
    calculate_bus_budget,
)
from acetele.hardware.devices.adapter import (
    AdapterPlan,
    BusAdapter,
    DecodedJointSample,
    GroupDescription,
    TransportFactory,
    motion_envelope,
)
from acetele.hardware.devices.hands.linker.modbus import (
    LinkerHandModbusProtocol,
    LinkerHandMotion,
)
from acetele.hardware.devices.hands.linker.profile import linker_hand_profiles
from acetele.specification import Backend, BusSpec, BusType, DexterousHandSpec


class LinkerHandAdapter(BusAdapter):
    """Resolve model-defined hand vectors and their Modbus transport."""

    bus_types = (BusType.LINKER_HAND_RS485,)

    @staticmethod
    def hand_joint_names(hand: DexterousHandSpec) -> tuple[str, ...]:
        """Return the profile-defined public joint order for a hand."""

        return linker_hand_profiles.require(
            hand.model.upper(),
            context="dexterous hand",
        ).joint_names

    def preflight(
        self,
        bus: BusSpec,
        groups: Sequence[GroupDescription],
        backend: Backend,
    ) -> AdapterPlan:
        if not groups or any(group.hand is None or group.joints for group in groups):
            raise ValueError(f"Linker Hand bus '{bus.name}' must contain only hands")
        profiles = {
            group.hand.slave_id: linker_hand_profiles.require(
                group.hand.model.upper(),
                context=f"hand on bus '{bus.name}'",
            )
            for group in groups
            if group.hand is not None
        }
        wire_bytes = sum(profile.wire_bytes_per_cycle for profile in profiles.values())
        turnaround = sum(
            profile.turnaround_s_per_cycle for profile in profiles.values()
        )
        budget = calculate_bus_budget(
            baudrate=bus.baudrate,
            cycle_hz=bus.cycle_hz,
            wire_bytes_per_cycle=wire_bytes,
            turnaround_s_per_cycle=turnaround,
            max_utilization=bus.max_utilization,
        )
        verified = backend == Backend.MOCK
        return AdapterPlan(
            spec=bus,
            backend=backend,
            adapter=self,
            groups=tuple(groups),
            profiles=profiles,
            expected_models={},
            firmware_versions={},
            budget=budget,
            device_models={
                device_id: profile.model for device_id, profile in profiles.items()
            },
            supports_software_disable=verified,
            supports_verified_disable=verified,
            supports_verified_identity=verified,
            supports_acceleration_limits=False,
        )

    def create_protocol(
        self,
        plan: AdapterPlan,
        transport_factory: TransportFactory,
        *,
        clock_ns=time.monotonic_ns,
    ) -> BusProtocol:
        if plan.backend == Backend.MOCK:
            return self.mock_protocol(plan, clock_ns=clock_ns)
        bus = plan.spec
        transport = transport_factory(
            bus.port,
            bus.baudrate,
            SerialDirectionControl(bus.direction_control.value),
        )
        return LinkerHandModbusProtocol(
            transport,
            plan.profiles,
            clock_ns=clock_ns,
        )

    def decode_group(
        self,
        plan: AdapterPlan,
        group: GroupDescription,
        snapshot: Mapping[int, Any],
    ) -> DecodedJointSample:
        mock = self.decode_mock_group(group, snapshot)
        if mock is not None:
            return mock
        if group.hand is None:
            raise RuntimeError("Linker adapter received a scalar servo group")
        state = snapshot[group.hand.slave_id]
        return DecodedJointSample(
            tuple(state.positions),
            tuple(state.velocities),
            tuple(state.efforts),
            state.timestamp_ns,
        )

    def validate_command(
        self,
        plan: AdapterPlan,
        group: GroupDescription,
        command: JointCommand,
    ) -> None:
        if plan.backend == Backend.MOCK:
            self.validate_mock_command(group, command)
            return
        if any(
            value is not None
            for value in (
                command.velocity_limits,
                command.acceleration_limits,
                command.effort_limits,
            )
        ):
            raise ValueError(
                f"joint group '{group.name}' supports position commands only"
            )

    def encode_group(
        self,
        plan: AdapterPlan,
        group: GroupDescription,
        command: JointCommand,
        position_references: Mapping[int, float],
    ) -> tuple[MotionEnvelope, ...]:
        if plan.backend == Backend.MOCK:
            return self.encode_mock_group(group, command)
        if group.hand is None:
            raise RuntimeError("Linker adapter received a scalar servo group")
        return (
            motion_envelope(
                group.hand.slave_id,
                LinkerHandMotion(tuple(command.positions)),
                command,
            ),
        )

    def fault_reason(
        self,
        plan: AdapterPlan,
        fast_snapshot: Any,
        slow_snapshot: Any,
    ) -> str | None:
        if not isinstance(slow_snapshot, Mapping):
            return None
        for device_id, state in slow_snapshot.items():
            errors = tuple(getattr(state, "errors", ()))
            failed = tuple(index for index, value in enumerate(errors) if value)
            if failed:
                return (
                    f"bus '{plan.spec.name}' device {device_id} reported Linker Hand "
                    f"joint errors at indices {failed}"
                )
        return None


__all__ = ["LinkerHandAdapter"]
