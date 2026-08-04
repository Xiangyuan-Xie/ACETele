"""FEETECH packet and Modbus adapters for the vendor-neutral runtime boundary."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from acetele.core import JointCommand, JointEffortCommand
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
    effort_envelope,
    motion_envelope,
)
from acetele.hardware.devices.servos.feetech.modbus import (
    FeetechModbusBusProtocol,
    FeetechModbusFastState,
    FeetechModbusMotion,
)
from acetele.hardware.devices.servos.feetech.packet import (
    FeetechPacketBusProtocol,
    FeetechPacketEffort,
    FeetechPacketFastState,
    FeetechPacketMotion,
    nearest_multiturn_position_target,
)
from acetele.hardware.devices.servos.feetech.profile import (
    feetech_modbus_profiles,
    feetech_packet_profiles,
)
from acetele.hardware.simulators import MockEffort
from acetele.specification import Backend, BusSpec, BusType


class FeetechAdapter(BusAdapter):
    """Resolve and operate FEETECH packet and Modbus servo families."""

    bus_types = (BusType.FEETECH_PACKET, BusType.FEETECH_MODBUS_RTU)

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
        expected: dict[int, int] = {}
        for joint in joints:
            profile: Any
            if bus.type == BusType.FEETECH_PACKET:
                profile = feetech_packet_profiles.require(
                    joint.servo_model,
                    context=f"joint '{joint.name}'",
                )
                if (
                    profile.family.value != bus.family
                    or profile.physical_layer.value != bus.physical_layer
                ):
                    raise ValueError(
                        f"joint '{joint.name}' profile {profile.model} uses "
                        f"{profile.family.value}/{profile.physical_layer.value}, not "
                        f"{bus.family}/{bus.physical_layer}"
                    )
                model_number = (
                    joint.expected_model_number
                    if joint.expected_model_number is not None
                    else profile.model_number
                )
                if model_number is not None:
                    expected[joint.servo_id] = model_number
            else:
                profile = feetech_modbus_profiles.require(
                    joint.servo_model,
                    context=f"joint '{joint.name}'",
                )
            profiles[joint.servo_id] = profile
        verified_disable = backend == Backend.MOCK or bus.type == BusType.FEETECH_MODBUS_RTU
        verified_identity = (
            backend == Backend.MOCK
            or bus.type == BusType.FEETECH_MODBUS_RTU
            or len(expected) == len(profiles)
        )
        effort_limits_nm = {
            device_id: (
                max(
                    profile.default_goal_torque_raw * profile.current_unit_a
                    - profile.no_load_current_a,
                    0.0,
                )
                * profile.torque_constant_kgcm_per_a
                * 0.0980665
            )
            for device_id, profile in profiles.items()
            if profile.family.value == "hls"
            and profile.default_goal_torque_raw is not None
            and profile.torque_constant_kgcm_per_a is not None
            and profile.no_load_current_a is not None
        }
        budget = self._budget(bus, len(profiles))
        return AdapterPlan(
            spec=bus,
            backend=backend,
            adapter=self,
            groups=tuple(groups),
            profiles=profiles,
            expected_models=expected,
            firmware_versions={},
            budget=budget,
            device_models={
                device_id: profile.model for device_id, profile in profiles.items()
            },
            supports_software_disable=True,
            supports_verified_disable=verified_disable,
            supports_verified_identity=verified_identity,
            supports_acceleration_limits=backend == Backend.PHYSICAL,
            supports_effort_control=(
                bus.type == BusType.FEETECH_PACKET
                and len(effort_limits_nm) == len(profiles)
            ),
            effort_limits_nm=effort_limits_nm,
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
        if bus.type == BusType.FEETECH_PACKET:
            return FeetechPacketBusProtocol(
                transport,
                plan.profiles,
                expected_model_numbers=plan.expected_models,
                clock_ns=clock_ns,
            )
        return FeetechModbusBusProtocol(transport, plan.profiles, clock_ns=clock_ns)

    def decode_group(
        self,
        plan: AdapterPlan,
        group: GroupDescription,
        snapshot: Mapping[int, Any],
    ) -> DecodedJointSample:
        mock = self.decode_mock_group(group, snapshot)
        if mock is not None:
            return mock
        positions: list[float] = []
        velocities: list[float] = []
        efforts: list[float] = []
        timestamps: list[int] = []
        for joint in group.joints:
            state = snapshot[joint.servo_id]
            if not isinstance(state, (FeetechPacketFastState, FeetechModbusFastState)):
                raise RuntimeError(f"unsupported FEETECH state {type(state).__name__}")
            positions.append(state.position_rad * joint.direction)
            velocities.append(state.velocity_rad_s * joint.direction)
            efforts.append(
                self._estimate_effort(
                    plan.profiles[joint.servo_id],
                    state.current_a,
                    joint.direction,
                )
            )
            timestamps.append(state.timestamp_ns)
        return DecodedJointSample(
            tuple(positions),
            tuple(velocities),
            tuple(efforts),
            min(timestamps),
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
        if command.effort_limits is not None:
            register = (
                "HLS goal-torque field"
                if plan.spec.type == BusType.FEETECH_PACKET
                else "Modbus torque-ratio field"
            )
            raise ValueError(
                f"joint group '{group.name}' cannot map SI effort limits to the "
                f"FEETECH {register} without a calibrated actuator model"
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
            position, velocity, acceleration = _physical_target(group, command, index)
            profile = plan.profiles[joint.servo_id]
            position *= joint.direction
            if plan.spec.type == BusType.FEETECH_PACKET:
                self._validate_packet_registers(
                    profile,
                    position,
                    position_references.get(joint.servo_id),
                    velocity,
                    acceleration,
                )
                payload: Any = FeetechPacketMotion(
                    position,
                    velocity,
                    acceleration,
                )
            else:
                self._validate_modbus_registers(
                    profile,
                    position,
                    velocity,
                    acceleration,
                )
                payload = FeetechModbusMotion(position, velocity, acceleration)
            targets.append(motion_envelope(joint.servo_id, payload, command))
        return tuple(targets)

    def validate_effort_command(
        self,
        plan: AdapterPlan,
        group: GroupDescription,
        command: JointEffortCommand,
    ) -> None:
        """Validate a calibrated HLS Torque-mode command without touching hardware."""

        if not plan.supports_effort_control or not group.is_arm:
            raise ValueError(
                f"joint group '{group.name}' does not support calibrated effort control"
            )
        if plan.backend == Backend.MOCK:
            return
        for index, joint in enumerate(group.joints):
            self._effort_to_raw_current(
                plan.profiles[joint.servo_id],
                float(command.efforts_nm[index]),
                joint.direction,
            )

    def encode_effort_group(
        self,
        plan: AdapterPlan,
        group: GroupDescription,
        command: JointEffortCommand,
    ) -> tuple[MotionEnvelope, ...]:
        """Convert canonical joint Nm into signed HLS current register targets."""

        self.validate_effort_command(plan, group, command)
        targets = []
        for index, joint in enumerate(group.joints):
            effort_nm = float(command.efforts_nm[index])
            payload = (
                MockEffort(effort_nm)
                if plan.backend == Backend.MOCK
                else FeetechPacketEffort(
                    self._effort_to_raw_current(
                        plan.profiles[joint.servo_id],
                        effort_nm,
                        joint.direction,
                    )
                )
            )
            targets.append(
                effort_envelope(
                    joint.servo_id,
                    payload,
                    command,
                )
            )
        return tuple(targets)

    def estimator_tuning(
        self,
        plan: AdapterPlan,
        group: GroupDescription,
    ) -> Optional[StateEstimatorTuning]:
        profiles = tuple(plan.profiles[joint.servo_id] for joint in group.joints)
        position_step = max(
            2.0 * math.pi / profile.counts_per_revolution for profile in profiles
        )
        velocity_step = max(profile.velocity_unit_rad_s for profile in profiles)
        return StateEstimatorTuning(
            acceleration_std_rad_s2=6.0,
            position_std_rad=4.0 * position_step,
            velocity_std_rad_s=12.0 * velocity_step,
            position_gate_rad=8.0 * position_step,
            reanchor_gate_rad=4.0 * position_step,
            velocity_consistency_rad_s=6.0 * velocity_step,
        )

    def capture_position_references(
        self,
        plan: AdapterPlan,
        snapshot: Mapping[int, Any],
    ) -> Mapping[int, float]:
        if plan.spec.type != BusType.FEETECH_PACKET:
            return {}
        return {
            device_id: state.position_rad
            for device_id, state in snapshot.items()
            if isinstance(state, FeetechPacketFastState)
        }

    def hardware_fault(
        self,
        plan: AdapterPlan,
        fast_snapshot: Any,
        slow_snapshot: Any,
    ) -> Optional[HardwareFault]:
        """Decode FEETECH packet status bits before runtime selects containment."""

        if isinstance(fast_snapshot, Mapping):
            for device_id, state in fast_snapshot.items():
                if not isinstance(state, FeetechPacketFastState) or not state.status:
                    continue
                labels = tuple(
                    label
                    for mask, label in (
                        (0x01, "supply-voltage"),
                        (0x02, "angle-sensor"),
                        (0x04, "over-temperature"),
                        (0x08, "over-current"),
                        (0x10, "angle-limit"),
                        (0x20, "overload"),
                    )
                    if state.status & mask
                )
                unknown = state.status & ~0x3F
                decoded = ", ".join(labels)
                if unknown:
                    decoded = ", ".join(
                        value for value in (decoded, f"unknown-0x{unknown:02x}") if value
                    )
                return HardwareFault(
                    (
                        f"bus '{plan.spec.name}' device {device_id} reported FEETECH "
                        f"{decoded} (status 0x{state.status:02x}, "
                        f"current={state.current_a:.3f} A, voltage={state.voltage_v:.1f} V, "
                        f"temperature={state.temperature_c} C)"
                    ),
                    AutomaticFaultAction.DISABLE,
                )
        return super().hardware_fault(plan, fast_snapshot, slow_snapshot)

    def calibration_targets(
        self,
        plan: AdapterPlan,
    ) -> Optional[Mapping[int, int]]:
        if plan.spec.type != BusType.FEETECH_PACKET:
            return None
        targets: dict[int, int] = {}
        for group in plan.groups:
            for joint in group.joints:
                profile = plan.profiles[joint.servo_id]
                raw = round(
                    joint.home_position_rad
                    * joint.direction
                    * profile.counts_per_revolution
                    / (2.0 * math.pi)
                )
                if not -0x7FFF <= raw <= 0x7FFF:
                    raise ValueError(
                        f"joint '{joint.name}' home position exceeds the FEETECH "
                        "signed-15-bit calibration range"
                    )
                targets[joint.servo_id] = raw
        return targets

    @staticmethod
    def _effort_to_raw_current(profile, effort_nm: float, direction: int) -> int:
        """Invert the calibrated current-to-joint-torque estimate used in feedback."""

        if not math.isfinite(effort_nm):
            raise ValueError("FEETECH joint effort must be finite")
        torque_constant = profile.torque_constant_kgcm_per_a
        no_load_current = profile.no_load_current_a
        if torque_constant is None or no_load_current is None:
            raise ValueError(
                f"FEETECH profile {profile.model} has no calibrated torque model"
            )
        if effort_nm == 0.0:
            return 0
        current_magnitude = (
            abs(effort_nm) / (torque_constant * 0.0980665) + no_load_current
        )
        motor_current_a = -direction * math.copysign(current_magnitude, effort_nm)
        raw = round(motor_current_a / profile.current_unit_a)
        if not -0x7FFF <= raw <= 0x7FFF:
            raise ValueError(
                f"FEETECH effort for {profile.model} exceeds signed-15-bit current range"
            )
        return raw

    @staticmethod
    def _budget(bus: BusSpec, device_count: int) -> BusBudget:
        return calculate_bus_budget(
            baudrate=bus.baudrate,
            cycle_hz=bus.cycle_hz,
            wire_bytes_per_cycle=(16 + 31 * device_count)
            if bus.type == BusType.FEETECH_PACKET
            else 56 * device_count,
            max_utilization=bus.max_utilization,
        )

    @staticmethod
    def _estimate_effort(profile: Any, current_a: float, direction: int) -> float:
        torque_constant = getattr(profile, "torque_constant_kgcm_per_a", None)
        no_load_current = getattr(profile, "no_load_current_a", None)
        if torque_constant is None or no_load_current is None:
            return 0.0
        magnitude_nm = (
            max(abs(current_a) - no_load_current, 0.0)
            * torque_constant
            * 0.0980665
        )
        return magnitude_nm * float(np.sign(-current_a * direction))

    @staticmethod
    def _validate_packet_registers(
        profile: Any,
        position_rad: float,
        reference_rad: Optional[float],
        velocity_rad_s: Optional[float],
        acceleration_rad_s2: Optional[float],
    ) -> None:
        if reference_rad is None:
            raise ValueError(
                f"FEETECH profile {profile.model} requires a current position sample"
            )
        scale = profile.counts_per_revolution / (2.0 * math.pi)
        nearest_multiturn_position_target(
            round(reference_rad * scale),
            round(position_rad * scale),
            profile.counts_per_revolution,
        )
        values = (
            (
                "velocity",
                profile.default_velocity_raw
                if velocity_rad_s is None
                else round(velocity_rad_s / profile.velocity_unit_rad_s),
                0x7FFF,
            ),
            (
                "acceleration",
                profile.default_acceleration_raw
                if acceleration_rad_s2 is None
                else round(acceleration_rad_s2 / profile.acceleration_unit_rad_s2),
                0xFF,
            ),
        )
        for label, raw, maximum in values:
            if not 0 <= raw <= maximum:
                raise ValueError(
                    f"FEETECH {label} cannot be encoded by profile {profile.model}"
                )

    @staticmethod
    def _validate_modbus_registers(
        profile: Any,
        position_rad: float,
        velocity_rad_s: Optional[float],
        acceleration_rad_s2: Optional[float],
    ) -> None:
        raw_position = round(
            position_rad * profile.counts_per_revolution / (2.0 * math.pi)
        )
        if not -0x8000 <= raw_position <= 0x7FFF:
            raise ValueError(
                f"FEETECH Modbus position cannot be encoded by profile {profile.model}"
            )
        for label, value, unit, default in (
            (
                "velocity",
                velocity_rad_s,
                profile.velocity_unit_rad_s,
                profile.default_velocity_raw,
            ),
            (
                "acceleration",
                acceleration_rad_s2,
                profile.acceleration_unit_rad_s2,
                profile.default_acceleration_raw,
            ),
        ):
            raw = default if value is None else round(value / unit)
            if not 0 <= raw <= 0xFFFF:
                raise ValueError(
                    f"FEETECH Modbus {label} cannot be encoded by profile {profile.model}"
                )


def _physical_target(
    group: GroupDescription,
    command: JointCommand,
    index: int,
) -> tuple[float, Optional[float], Optional[float]]:
    position = float(command.positions[index])
    velocity = (
        None
        if command.velocity_limits is None
        else float(command.velocity_limits[index])
    )
    acceleration = (
        None
        if command.acceleration_limits is None
        else float(command.acceleration_limits[index])
    )
    if group.travel_range_rad is not None:
        position *= group.travel_range_rad
        if velocity is not None:
            velocity *= group.travel_range_rad
        if acceleration is not None:
            acceleration *= group.travel_range_rad
    return position, velocity, acceleration


__all__ = ["FeetechAdapter"]
