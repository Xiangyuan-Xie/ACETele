"""FashionStar RS485 adapter for the vendor-neutral runtime boundary."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Optional, Sequence

from acetele.core import JointCommand
from acetele.estimation import StateEstimatorTuning
from acetele.hardware.buses import (
    BusBudget,
    BusProtocol,
    MotionEnvelope,
    SerialDirectionControl,
    calculate_bus_budget,
)
from acetele.hardware.devices.adapter import (
    AdapterPlan,
    AutomaticFaultAction,
    BusAdapter,
    DecodedJointSample,
    GroupDescription,
    HardwareFault,
    TransportFactory,
    motion_envelope,
)
from acetele.hardware.devices.servos.fashionstar.packet import (
    FashionStarBusProtocol,
    FashionStarMonitorState,
    FashionStarMotion,
)
from acetele.hardware.devices.servos.fashionstar.profile import (
    fashionstar_rs485_profiles,
)
from acetele.specification import Backend, BusSpec, BusType


class FashionStarAdapter(BusAdapter):
    """Resolve profiles and translate FashionStar packet state and motion."""

    bus_types = (BusType.FASHIONSTAR_RS485,)

    def preflight(
        self,
        bus: BusSpec,
        groups: Sequence[GroupDescription],
        backend: Backend,
    ) -> AdapterPlan:
        joints = tuple(joint for group in groups for joint in group.joints)
        if not joints or any(group.hand is not None for group in groups):
            raise ValueError(f"servo bus '{bus.name}' must contain only servo joints")
        profiles: dict[int, Any] = {}
        firmware: dict[int, int] = {}
        for joint in joints:
            profiles[joint.servo_id] = fashionstar_rs485_profiles.require(
                joint.servo_model,
                context=f"joint '{joint.name}'",
            )
            if joint.firmware_version is None and backend == Backend.PHYSICAL:
                raise ValueError(
                    f"joint '{joint.name}' requires firmware_version for physical "
                    "FashionStar connection verification"
                )
            if joint.firmware_version is not None:
                firmware[joint.servo_id] = joint.firmware_version
        budget = self._budget(bus, profiles, firmware)
        verified = backend == Backend.MOCK
        return AdapterPlan(
            spec=bus,
            backend=backend,
            adapter=self,
            groups=tuple(groups),
            profiles=profiles,
            expected_models={},
            firmware_versions=firmware,
            budget=budget,
            device_models={
                device_id: profile.model for device_id, profile in profiles.items()
            },
            supports_software_disable=True,
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
        return FashionStarBusProtocol(
            transport,
            plan.profiles,
            firmware_versions=plan.firmware_versions,
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
        states = tuple(snapshot[joint.servo_id] for joint in group.joints)
        if not all(isinstance(state, FashionStarMonitorState) for state in states):
            raise RuntimeError("FashionStar snapshot contains an unsupported state type")
        return DecodedJointSample(
            tuple(state.position_rad * joint.direction for state, joint in zip(states, group.joints)),
            (math.nan,) * len(states),
            (0.0,) * len(states),
            min(state.timestamp_ns for state in states),
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
        if (
            command.effort_limits is not None
            or (command.acceleration_limits is not None and not group.is_arm)
            or (command.velocity_limits is not None and not group.is_arm)
        ):
            raise ValueError(
                f"joint group '{group.name}' contains limits unsupported by FashionStar"
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
        targets: list[MotionEnvelope] = []
        for index, joint in enumerate(group.joints):
            position = float(command.positions[index])
            if group.travel_range_rad is not None:
                position *= group.travel_range_rad
            targets.append(
                motion_envelope(
                    joint.servo_id,
                    FashionStarMotion(position * joint.direction),
                    command,
                )
            )
        return tuple(targets)

    def estimator_tuning(
        self,
        plan: AdapterPlan,
        group: GroupDescription,
    ) -> Optional[StateEstimatorTuning]:
        position_step = math.radians(0.1)
        velocity_step = 0.25
        return StateEstimatorTuning(
            acceleration_std_rad_s2=6.0,
            position_std_rad=4.0 * position_step,
            velocity_std_rad_s=12.0 * velocity_step,
            position_gate_rad=8.0 * position_step,
            reanchor_gate_rad=4.0 * position_step,
            velocity_consistency_rad_s=6.0 * velocity_step,
        )

    def hardware_fault(
        self,
        plan: AdapterPlan,
        fast_snapshot: Any,
        slow_snapshot: Any,
    ) -> Optional[HardwareFault]:
        if not isinstance(fast_snapshot, Mapping):
            return None
        for device_id, state in fast_snapshot.items():
            status = getattr(state, "status", None)
            if type(status) is int and status & 0xFE:
                return HardwareFault(
                    (
                        f"bus '{plan.spec.name}' device {device_id} reported "
                        f"hardware status 0x{status & 0xFE:02x}"
                    ),
                    AutomaticFaultAction.DISABLE,
                )
        return None

    def negotiated_budget(self, plan: AdapterPlan, protocol: BusProtocol) -> BusBudget:
        synchronized = getattr(protocol, "synchronized", None)
        return self._budget(
            plan.spec,
            plan.profiles,
            plan.firmware_versions,
            synchronized=synchronized,
        )

    @staticmethod
    def _budget(
        bus: BusSpec,
        profiles: Mapping[int, Any],
        firmware: Mapping[int, int],
        *,
        synchronized: Optional[bool] = None,
    ) -> BusBudget:
        if synchronized is None:
            synchronized = all(
                profile.supports_sync(firmware.get(device_id))
                for device_id, profile in profiles.items()
            )
        if synchronized:
            wire_bytes = 16 + 39 * len(profiles)
            turnaround = 2.0 * max(
                profile.minimum_command_interval_s for profile in profiles.values()
            )
        else:
            wire_bytes = 45 * len(profiles)
            turnaround = 2.0 * sum(
                profile.minimum_command_interval_s for profile in profiles.values()
            )
        return calculate_bus_budget(
            baudrate=bus.baudrate,
            cycle_hz=bus.cycle_hz,
            wire_bytes_per_cycle=wire_bytes,
            turnaround_s_per_cycle=turnaround,
            max_utilization=bus.max_utilization,
        )


__all__ = ["FashionStarAdapter"]
