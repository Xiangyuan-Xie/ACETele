"""Robot assembly, bus routing, control conditioning, and safety lifecycle.

``RobotRuntime`` is the pure-Python composition root. Construction performs static
preflight only; ``connect()`` owns hardware creation. A command is fully validated and
encoded before a shared commit gate makes its per-bus envelopes visible.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

import numpy as np

from acetele.config.specs import (
    BusSpec,
    BusType,
    DexterousHandSpec,
    JointSpec,
    ParallelGripperSpec,
    RobotSpec,
)
from acetele.control import PositionControlDiagnostics, PositionControlPipeline
from acetele.core import (
    Backend,
    JointCommand,
    JointState,
    JointUnit,
    RobotCommand,
    RobotState,
    SensorState,
)
from acetele.hardware.dexterous_hands.linker_hand import (
    LinkerHandModbusProtocol,
    LinkerHandMotion,
    linker_hand_profiles,
)
from acetele.hardware.mock import MockBusProtocol, MockDeviceDefinition, MockDeviceState, MockMotion
from acetele.hardware.serial import (
    BusActorDiagnostics,
    BusBudget,
    MotionCommitGate,
    MotionEnvelope,
    SerialBusActor,
    SerialDirectionControl,
    SerialTransport,
    calculate_bus_budget,
)
from acetele.hardware.smart_servos.fashionstar import (
    FashionStarBusProtocol,
    FashionStarMonitorState,
    FashionStarMotion,
    fashionstar_rs485_profiles,
)
from acetele.hardware.smart_servos.feetech import (
    FeetechModbusBusProtocol,
    FeetechModbusFastState,
    FeetechModbusMotion,
    FeetechPacketBusProtocol,
    FeetechPacketFamily,
    FeetechPacketFastState,
    FeetechPacketMotion,
    feetech_modbus_profiles,
    feetech_packet_profiles,
    nearest_multiturn_position_target,
)
from acetele.hardware.state_estimator import (
    RobustJointStateEstimator,
    StateEstimatorTuning,
)
from acetele.model import ArmModelMetadata, build_reduced_pinocchio_model, load_urdf_model
from acetele.runtime.safety import (
    RuntimeSafetyController,
    RuntimeSafetyState,
    SafetySnapshot,
    SafetyTransition,
)
from acetele.utils.angle import unwrap_near, wrap_to_pi


@dataclass(frozen=True)
class BusPreflight:
    """Validated profile, identity, safety, and bandwidth facts for one bus."""

    spec: BusSpec
    budget: BusBudget
    device_models: Mapping[int, str]
    supports_software_disable: bool
    supports_verified_disable: bool
    supports_verified_identity: bool


@dataclass(frozen=True)
class RuntimePreflight:
    """Static facts that must exist before any serial port may be opened."""

    urdf_path: Path
    arms: Mapping[str, ArmModelMetadata]
    buses: Mapping[str, BusPreflight]


@dataclass(frozen=True)
class RuntimeDiagnostics:
    """Aggregate safety, bus, controller, and estimator diagnostics."""

    safety: SafetySnapshot
    buses: Mapping[str, BusActorDiagnostics]
    controls: Mapping[str, PositionControlDiagnostics]
    estimators: Mapping[str, Mapping[str, np.ndarray]]


@dataclass(frozen=True)
class JointGroupInfo:
    """Canonical order and unit of one commandable joint group."""

    names: tuple[str, ...]
    unit: JointUnit


@dataclass(frozen=True)
class _GroupPlan:
    name: str
    bus: str
    unit: JointUnit
    joint_names: tuple[str, ...]
    joints: tuple[JointSpec, ...]
    metadata: Optional[ArmModelMetadata] = None
    travel_range_rad: Optional[float] = None
    hand: Optional[DexterousHandSpec] = None


TransportFactory = Callable[[str, int, SerialDirectionControl], SerialTransport]


def _serialized_operation(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def wrapped(self: "RobotRuntime", *args: Any, **kwargs: Any) -> Any:
        with self._operation_lock:
            return method(self, *args, **kwargs)

    return wrapped


def _default_transport_factory(
    port: str,
    baudrate: int,
    direction_control: SerialDirectionControl,
) -> SerialTransport:
    return SerialTransport(
        port,
        baudrate,
        direction_control=direction_control,
    )


class RobotRuntime:
    """Validated robot assembly with explicit hardware lifecycle and safety state."""

    def __init__(
        self,
        spec: RobotSpec,
        *,
        transport_factory: TransportFactory = _default_transport_factory,
        clock_ns=time.monotonic_ns,
        command_timeout_ns: int = 100_000_000,
    ) -> None:
        if not isinstance(spec, RobotSpec):
            raise ValueError("RobotRuntime requires a RobotSpec")
        self.spec = spec
        self._clock_ns = clock_ns
        self._transport_factory = transport_factory
        self._operation_lock = RLock()
        self._safety = RuntimeSafetyController(command_timeout_ns=command_timeout_ns)
        self._actors: dict[str, SerialBusActor] = {}
        self._sequence: dict[str, int] = {}
        self._last_state_ns: Optional[int] = None
        self._profiles: dict[str, dict[int, Any]] = {}
        self._expected_models: dict[str, dict[int, int]] = {}
        self._firmware_versions: dict[str, dict[int, int]] = {}
        self._groups: dict[str, _GroupPlan] = {}
        self._pin_models: dict[str, Any] = {}
        self._pipelines: dict[str, PositionControlPipeline] = {}
        self._estimators: dict[str, RobustJointStateEstimator] = {}
        self._raw_packet_positions: dict[tuple[str, int], float] = {}
        # Preflight is deliberately constructor-only and side-effect free: every model,
        # capability, register range, and bandwidth error is found before a port opens.
        self.preflight = self._preflight()
        self._state_timeout_ns = {
            name: max(50_000_000, math.ceil(2.5e9 / bus.spec.cycle_hz))
            for name, bus in self.preflight.buses.items()
        }

    @property
    def connected(self) -> bool:
        """Return whether every configured actor is alive and healthy."""

        return bool(self._actors) and all(actor.connected for actor in self._actors.values())

    @property
    def generation(self) -> int:
        """Return the generation required by newly built commands."""

        return self._safety.snapshot().generation

    @property
    def command_timeout_ns(self) -> int:
        """Return the active-motion heartbeat timeout."""

        return self._safety.command_timeout_ns

    @property
    def joint_groups(self) -> Mapping[str, JointGroupInfo]:
        """Return immutable public routing metadata for all joint groups."""

        return MappingProxyType(
            {
                name: JointGroupInfo(group.joint_names, group.unit)
                for name, group in self._groups.items()
            }
        )

    def command_lifetime_ns(
        self,
        group_name: str,
        *,
        minimum_ns: int = 50_000_000,
    ) -> int:
        """Return enough lifetime for one bus cycle plus its preflight I/O budget."""
        if type(minimum_ns) is not int or minimum_ns <= 0:
            raise ValueError("minimum command lifetime must be a positive integer")
        try:
            group = self._groups[group_name]
        except KeyError as exc:
            raise ValueError(f"unknown joint group '{group_name}'") from exc
        budget = self.preflight.buses[group.bus].budget
        period_s = 1.0 / budget.cycle_hz
        occupied_s = budget.utilization / budget.cycle_hz
        return max(minimum_ns, math.ceil((period_s + occupied_s) * 1e9))

    @_serialized_operation
    def connect(self) -> None:
        """Connect all buses, cleaning every partial resource on failure."""

        if self._actors:
            raise RuntimeError("robot runtime is already connected")
        self._raw_packet_positions.clear()
        actors: dict[str, SerialBusActor] = {}
        try:
            # Connect one actor at a time, retaining local ownership until every bus has
            # negotiated capabilities and produced a complete initial state snapshot.
            for bus in self.spec.buses:
                protocol = self._create_protocol(bus)
                actor = SerialBusActor(protocol, cycle_hz=bus.cycle_hz)
                actor.connect()
                actors[bus.name] = actor
                if isinstance(protocol, FashionStarBusProtocol):
                    actual_budget = self._calculate_budget(
                        bus,
                        self._profiles[bus.name],
                        self._firmware_versions[bus.name],
                        synchronized=protocol.synchronized,
                    )
                    actual_budget.require_feasible(
                        context=f"bus '{bus.name}' negotiated capabilities"
                    )
            for bus_name, actor in actors.items():
                self._cache_packet_positions(
                    bus_name,
                    actor.wait_for_snapshot(timeout=1.0),
                )
        except BaseException as exc:
            # Construction is transactional at the process-resource level: failure on
            # a later bus still closes every actor created earlier.
            cleanup_error = _disconnect_actors(actors)
            if cleanup_error is not None:
                raise exc from cleanup_error
            raise
        self._actors = actors
        self._safety.connected()

    def home_calibration_targets(self) -> Mapping[str, Mapping[int, int]]:
        """Return a fully validated FEETECH packet calibration plan without I/O."""
        targets: dict[str, Mapping[int, int]] = {}
        unsupported: list[str] = []
        # Build the entire EEPROM plan first. No actor is touched until all home poses
        # are known to fit the packet family's signed position representation.
        for bus in self.spec.buses:
            joints = self._joints_for_bus(bus.name)
            if not joints:
                continue
            if bus.type != BusType.FEETECH_PACKET:
                unsupported.append(f"{bus.name} ({bus.type.value})")
                continue
            bus_targets: dict[int, int] = {}
            for joint in joints:
                profile = self._profiles[bus.name][joint.servo_id]
                raw_position = round(
                    joint.home_position_rad
                    * joint.direction
                    * profile.counts_per_revolution
                    / (2.0 * math.pi)
                )
                if not -0x7FFF <= raw_position <= 0x7FFF:
                    raise ValueError(
                        f"joint '{joint.name}' home position exceeds the FEETECH "
                        "signed-15-bit calibration range"
                    )
                bus_targets[joint.servo_id] = raw_position
            targets[bus.name] = MappingProxyType(bus_targets)
        if unsupported:
            raise ValueError(
                "home calibration is not implemented for buses: " + ", ".join(unsupported)
            )
        if not targets:
            raise ValueError("robot spec contains no FEETECH packet joints to calibrate")
        return MappingProxyType(targets)

    @_serialized_operation
    def calibrate_home(self) -> None:
        """Write configured FEETECH home positions while the runtime is safely disabled."""
        if self.spec.backend != Backend.PHYSICAL:
            raise RuntimeError("home calibration requires backend='physical'")
        targets = self.home_calibration_targets()
        self._require_connected()
        if self._safety.snapshot().state != RuntimeSafetyState.SAFE_DISABLED:
            raise RuntimeError("home calibration requires the SAFE_DISABLED runtime state")
        for bus_name, bus_targets in targets.items():
            self._actors[bus_name].submit_safety(
                "calibrate_offset",
                dict(bus_targets),
                wait=True,
                clear_motion=True,
            )

    @_serialized_operation
    def read(self) -> RobotState:
        """Build robot state from coherent bus snapshots and update safety timeouts."""

        self._require_connected()
        # Copy one snapshot per bus before decoding groups. This prevents arm and
        # end-effector views from accidentally observing different actor cycles.
        snapshots = {name: actor.get_snapshot() for name, actor in self._actors.items()}
        for bus_name, snapshot in snapshots.items():
            self._cache_packet_positions(bus_name, snapshot)
        slow_snapshots = {
            name: actor.get_slow_snapshot() for name, actor in self._actors.items()
        }
        # Protocol health and freshness are evaluated before publishing any state to a
        # controller, so a bad sample cannot leak into a lower-level policy.
        fault_reason = self._hardware_fault_reason(snapshots, slow_snapshots)
        if fault_reason is not None:
            self._latch_hardware_fault(fault_reason)
            raise RuntimeError(fault_reason)
        joint_states: dict[str, JointState] = {}
        timestamps: list[int] = []
        safety_state = self._safety.snapshot().state
        for name, group in self._groups.items():
            state = self._read_group(group, snapshots[group.bus])
            joint_states[name] = state
            pipeline = self._pipelines.get(name)
            if pipeline is not None:
                pipeline.update_feedback(state)
                if safety_state != RuntimeSafetyState.ACTIVE:
                    pipeline.rebase_to_feedback()
            timestamps.append(state.timestamp_ns)
        if not timestamps:
            raise RuntimeError("robot runtime has no joint state groups")
        self._last_state_ns = min(timestamps)
        now_ns = self._clock_ns()
        transition = self._safety.update(
            now_ns,
            state_stale=self._hardware_state_is_stale(now_ns),
        )
        if transition == SafetyTransition.HOLD:
            self._rebase_position_pipelines()
            self._submit_all_safety("hold", None, wait=False)
        elif transition == SafetyTransition.FAULT:
            self._submit_all_safety("emergency_stop", None, wait=False)
            raise RuntimeError("robot runtime hardware state is stale")
        sensor_states = self._sensor_states(slow_snapshots)
        return RobotState(joint_states, sensor_states)

    @_serialized_operation
    def write(self, command: RobotCommand) -> None:
        """Validate, condition, stage, and publish one logical robot command."""

        self._require_connected()
        if not isinstance(command, RobotCommand) or not command.joints:
            raise ValueError("robot write requires a non-empty RobotCommand")
        now_ns = self._clock_ns()
        transition = self._safety.update(
            now_ns,
            state_stale=self._hardware_state_is_stale(now_ns),
        )
        if transition != SafetyTransition.NONE:
            if transition == SafetyTransition.HOLD:
                self._rebase_position_pipelines()
                self._submit_all_safety("hold", None, wait=False)
            else:
                self._submit_all_safety("emergency_stop", None, wait=False)
            raise RuntimeError(f"robot runtime rejected motion after {transition.value}")
        snapshot = self._safety.snapshot()
        if snapshot.state not in (
            RuntimeSafetyState.READY,
            RuntimeSafetyState.ACTIVE,
        ):
            raise RuntimeError(f"robot runtime cannot move while {snapshot.state.value}")

        # Phase 1 is side-effect free: validate every group, prepare controller state,
        # and encode every vendor payload before any actor can observe the command.
        envelopes: dict[str, list[MotionEnvelope]] = {}
        prepared_controls = []
        minimum_deadline: Optional[int] = None
        for group_name, group_command in command.joints.items():
            group = self._groups.get(group_name)
            if group is None:
                raise ValueError(f"robot command references unknown joint group '{group_name}'")
            self._validate_group_command(group, group_command, snapshot.generation, now_ns)
            pipeline = self._pipelines.get(group_name)
            if pipeline is not None:
                prepared = pipeline.prepare(group_command, now_ns=now_ns)
                group_command = prepared.command
                self._validate_group_command(
                    group,
                    group_command,
                    snapshot.generation,
                    now_ns,
                )
                prepared_controls.append((pipeline, prepared))
            minimum_deadline = (
                group_command.deadline_ns
                if minimum_deadline is None
                else min(minimum_deadline, group_command.deadline_ns)
            )
            envelopes.setdefault(group.bus, []).extend(
                self._encode_group(group, group_command)
            )
        validation_completed_ns = self._clock_ns()
        if minimum_deadline is None or minimum_deadline < validation_completed_ns:
            raise RuntimeError("robot command expired during validation")
        # Phase 2 stages envelopes on all buses behind one gate. Only after safety state
        # and controller memory commit does the gate expose the complete robot update.
        commit_gate = MotionCommitGate()
        try:
            for bus_name, targets in envelopes.items():
                actor = self._actors[bus_name]
                generation = actor.generation
                actor.submit_motion(
                    tuple(
                        MotionEnvelope(
                            target.key,
                            target.device_id,
                            target.payload,
                            target.submitted_at_ns,
                            target.deadline_ns,
                            generation,
                            commit_gate,
                        )
                        for target in targets
                    )
                )
            accepted_ns = self._clock_ns()
            if minimum_deadline < accepted_ns:
                raise RuntimeError("robot command expired during submission")
            if snapshot.state == RuntimeSafetyState.READY:
                self._safety.activate(accepted_ns)
            if not self._safety.accept_command(
                accepted_ns,
                generation=snapshot.generation,
                deadline_ns=minimum_deadline,
            ):
                raise RuntimeError("robot command is stale or belongs to another generation")
            for pipeline, prepared in prepared_controls:
                pipeline.commit(prepared)
            commit_gate.commit()
        except BaseException:
            commit_gate.abort()
            self._latch_hardware_fault("motion submission failed")
            raise

    @_serialized_operation
    def hold(self) -> None:
        """Invalidate pending motion and hold the latest trustworthy positions."""

        self._require_connected()
        state = self._safety.snapshot().state
        if state not in (
            RuntimeSafetyState.READY,
            RuntimeSafetyState.ACTIVE,
            RuntimeSafetyState.HOLD,
        ):
            raise RuntimeError(f"cannot enter HOLD from {state.value}")
        # Close the software command gate before waiting for the bus transaction.
        # A failed hardware hold remains conservatively latched in software.
        self._safety.hold()
        self._rebase_position_pipelines()
        self._submit_all_safety("hold", None, wait=True)

    @_serialized_operation
    def set_enabled(self, enabled: bool) -> None:
        """Enable READY motion or enter a generation-bumped disabled state."""

        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        self._require_connected()
        state = self._safety.snapshot().state
        if state == RuntimeSafetyState.FAULT:
            raise RuntimeError("robot runtime fault is latched; reset it explicitly")
        if enabled and state not in (
            RuntimeSafetyState.SAFE_DISABLED,
            RuntimeSafetyState.HOLD,
        ):
            raise RuntimeError(f"cannot enter READY from {state.value}")
        if enabled:
            self._refresh_position_pipeline_feedback()
        self._rebase_position_pipelines()
        self._submit_set_enabled(enabled)
        if enabled:
            self._safety.ready()
        else:
            self._safety.disabled()

    @_serialized_operation
    def emergency_stop(self) -> None:
        """Latch the strongest profile-supported stop on every connected bus."""

        self._require_connected()
        self._safety.emergency_stop()
        self._submit_all_safety("emergency_stop", None, wait=True)

    @_serialized_operation
    def reset_emergency_stop(self, *, external_estop_reset: bool = False) -> None:
        """Clear the latch into SAFE_DISABLED after any external E-stop reset."""

        if type(external_estop_reset) is not bool:
            raise ValueError("external_estop_reset must be a boolean")
        self._require_connected()
        if self._safety.snapshot().state != RuntimeSafetyState.FAULT:
            raise RuntimeError("robot runtime has no latched fault to reset")
        unverifiable = tuple(
            name
            for name, bus in self.preflight.buses.items()
            if not bus.supports_verified_disable
        )
        if unverifiable and not external_estop_reset:
            raise RuntimeError(
                "external emergency-stop reset confirmation is required for buses: "
                + ", ".join(unverifiable)
            )
        # Verified buses are explicitly driven to torque-off. Unverifiable buses can
        # only have pending software commands invalidated after external confirmation.
        for bus_name, actor in self._actors.items():
            if self.preflight.buses[bus_name].supports_verified_disable:
                actor.submit_safety("set_enabled", False, wait=True)
            else:
                actor.discard_motion()
        self._safety.reset_fault()

    @_serialized_operation
    def disconnect(self) -> None:
        """Best-effort stop and close all actors while preserving the first error."""

        actors = self._actors
        if not actors:
            return
        # Detach actors first so concurrent diagnostics cannot treat a partially closed
        # set as a connected robot. The operation lock excludes command/read callers.
        self._actors = {}
        self._raw_packet_positions.clear()
        first_error: Optional[BaseException] = None
        # Request the strongest stop everywhere before closing any transport. Continue
        # after errors so one broken bus cannot leak all remaining resources.
        for actor in actors.values():
            try:
                actor.submit_safety("emergency_stop", None, wait=True)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        cleanup_error = _disconnect_actors(actors)
        self._safety.disconnected()
        if first_error is not None:
            raise first_error from cleanup_error
        if cleanup_error is not None:
            raise cleanup_error

    def diagnostics(self) -> RuntimeDiagnostics:
        """Return independent snapshots of runtime diagnostic state."""

        return RuntimeDiagnostics(
            self._safety.snapshot(),
            MappingProxyType(
                {name: actor.diagnostics() for name, actor in self._actors.items()}
            ),
            MappingProxyType(
                {name: pipeline.diagnostics() for name, pipeline in self._pipelines.items()}
            ),
            MappingProxyType(
                {
                    name: MappingProxyType(estimator.diagnostics())
                    for name, estimator in self._estimators.items()
                }
            ),
        )

    def _preflight(self) -> RuntimePreflight:
        """Resolve all model, routing, profile, capability, and bandwidth metadata."""

        # URDF joint order is the kinematic authority; bus IDs never participate in
        # model lookup or public state ordering.
        urdf_path = self._resolve_urdf_path()
        urdf = load_urdf_model(urdf_path)
        arm_metadata: dict[str, ArmModelMetadata] = {}
        for arm in self.spec.arms:
            arm_names = tuple(joint.name for joint in arm.joints)
            metadata = urdf.arm_metadata(arm_names, require_limits=True)
            if arm.tool_frame is not None:
                urdf.require_frame(arm.tool_frame)
            arm_metadata[arm.name] = metadata
            combined_names = arm_names
            if isinstance(arm.end_effector, ParallelGripperSpec):
                combined_names += (arm.end_effector.joint.name,)
            urdf.arm_metadata(combined_names, require_limits=False)
            pin_model = None
            if arm.control.gravity_position:
                pin_model = build_reduced_pinocchio_model(urdf_path, arm_names)
                self._pin_models[arm.name] = pin_model
            self._groups[arm.name] = _GroupPlan(
                arm.name,
                arm.bus,
                JointUnit.RADIAN,
                arm_names,
                arm.joints,
                metadata=metadata,
            )
            self._pipelines[arm.name] = PositionControlPipeline(
                metadata,
                arm.control,
                pin_model=pin_model,
            )
            if isinstance(arm.end_effector, ParallelGripperSpec):
                gripper_end = arm.end_effector
                group_name = f"{arm.name}.end_effector"
                self._groups[group_name] = _GroupPlan(
                    group_name,
                    gripper_end.bus,
                    JointUnit.NORMALIZED,
                    (gripper_end.joint.name,),
                    (gripper_end.joint,),
                    travel_range_rad=gripper_end.travel_range_rad,
                )
            elif isinstance(arm.end_effector, DexterousHandSpec):
                hand_end = arm.end_effector
                profile = linker_hand_profiles.require(
                    hand_end.model.upper(),
                    context=f"arm '{arm.name}' end effector",
                )
                group_name = f"{arm.name}.end_effector"
                self._groups[group_name] = _GroupPlan(
                    group_name,
                    hand_end.bus,
                    JointUnit.NORMALIZED,
                    profile.joint_names,
                    (),
                    hand=hand_end,
                )

        bus_preflight: dict[str, BusPreflight] = {}
        # Vendor profiles convert each bus into exact protocol capabilities and a wire
        # budget. The resulting metadata is immutable for the connected lifetime.
        for bus in self.spec.buses:
            joints = self._joints_for_bus(bus.name)
            hands = self._hands_for_bus(bus.name)
            (
                profiles,
                expected,
                firmware,
                supports_software_disable,
                supports_verified_disable,
                supports_verified_identity,
            ) = self._resolve_profiles(
                bus,
                joints,
                hands,
            )
            if (
                self.spec.backend == Backend.PHYSICAL
                and not supports_verified_disable
                and not bus.external_estop
            ):
                raise ValueError(
                    f"bus '{bus.name}' lacks verified torque disable; "
                    "configure and provide an independent hardware emergency stop"
                )
            if (
                self.spec.backend == Backend.PHYSICAL
                and not supports_verified_identity
                and not bus.allow_unverified_identity
            ):
                raise ValueError(
                    f"bus '{bus.name}' protocol cannot verify every configured device model; "
                    "set allow_unverified_identity=true only after independently checking "
                    "the installed hardware"
                )
            self._profiles[bus.name] = profiles
            self._expected_models[bus.name] = expected
            self._firmware_versions[bus.name] = firmware
            budget = self._calculate_budget(bus, profiles, firmware)
            budget.require_feasible(context=f"bus '{bus.name}'")
            bus_preflight[bus.name] = BusPreflight(
                bus,
                budget,
                MappingProxyType(
                    {
                        device_id: getattr(profile, "model", type(profile).__name__)
                        for device_id, profile in profiles.items()
                    }
                ),
                supports_software_disable,
                supports_verified_disable,
                supports_verified_identity,
            )
        if self.spec.backend == Backend.PHYSICAL:
            # Estimators are sized and tuned from the worst encoder resolution in each
            # group, keeping one coherent noise model for the returned vector.
            for group_name, group in self._groups.items():
                if group.hand is None:
                    self._estimators[group_name] = self._create_estimator(
                        group,
                        bus_preflight[group.bus].spec,
                    )
        return RuntimePreflight(
            urdf_path,
            MappingProxyType(arm_metadata),
            MappingProxyType(bus_preflight),
        )

    def _resolve_urdf_path(self) -> Path:
        """Return an explicit URDF or the model's packaged default, requiring a file."""

        if self.spec.urdf_path is not None:
            return Path(self.spec.urdf_path)
        path = (
            Path(__file__).resolve().parents[1]
            / "model"
            / "robots"
            / self.spec.model
            / "description"
            / f"{self.spec.model}.urdf"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"robot '{self.spec.model}' has no packaged URDF; set basic.urdf_path"
            )
        return path

    def _joints_for_bus(self, bus_name: str) -> tuple[JointSpec, ...]:
        """Collect arm and parallel-gripper joints sharing one physical bus."""

        joints: list[JointSpec] = []
        for arm in self.spec.arms:
            if arm.bus == bus_name:
                joints.extend(arm.joints)
            if isinstance(arm.end_effector, ParallelGripperSpec):
                if arm.end_effector.bus == bus_name:
                    joints.append(arm.end_effector.joint)
        return tuple(joints)

    def _hands_for_bus(self, bus_name: str) -> tuple[DexterousHandSpec, ...]:
        """Collect dexterous hands assigned to one dedicated Linker bus."""

        return tuple(
            arm.end_effector
            for arm in self.spec.arms
            if isinstance(arm.end_effector, DexterousHandSpec)
            and arm.end_effector.bus == bus_name
        )

    def _resolve_profiles(
        self,
        bus: BusSpec,
        joints: tuple[JointSpec, ...],
        hands: tuple[DexterousHandSpec, ...],
    ) -> tuple[dict[int, Any], dict[int, int], dict[int, int], bool, bool, bool]:
        """Bind exact vendor profiles and derive identity/disable capabilities."""

        profiles: dict[int, Any] = {}
        expected: dict[int, int] = {}
        firmware: dict[int, int] = {}
        if bus.type == BusType.LINKER_HAND_RS485:
            # Hands use one slave ID for a model-defined joint vector and therefore
            # cannot share this protocol actor with scalar smart servos.
            if joints or not hands:
                raise ValueError(f"Linker Hand bus '{bus.name}' must contain only hands")
            for hand in hands:
                profiles[hand.slave_id] = linker_hand_profiles.require(
                    hand.model.upper(),
                    context=f"hand on bus '{bus.name}'",
                )
            is_mock = self.spec.backend == Backend.MOCK
            return profiles, expected, firmware, is_mock, is_mock, is_mock
        if hands or not joints:
            raise ValueError(f"servo bus '{bus.name}' must contain at least one servo joint")
        for joint in joints:
            # Exact-name lookup is intentional: using a nearby model can silently apply
            # a different register map, scale, or safe-stop behavior.
            profile: Any
            if bus.type == BusType.FEETECH_PACKET:
                profile = feetech_packet_profiles.require(
                    joint.servo_model,
                    context=f"joint '{joint.name}'",
                )
                if profile.family.value != bus.family or profile.physical_layer.value != bus.physical_layer:
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
            elif bus.type == BusType.FEETECH_MODBUS_RTU:
                profile = feetech_modbus_profiles.require(
                    joint.servo_model,
                    context=f"joint '{joint.name}'",
                )
            elif bus.type == BusType.FASHIONSTAR_RS485:
                profile = fashionstar_rs485_profiles.require(
                    joint.servo_model,
                    context=f"joint '{joint.name}'",
                )
                if joint.firmware_version is None and self.spec.backend == Backend.PHYSICAL:
                    raise ValueError(
                        f"joint '{joint.name}' requires firmware_version for physical "
                        "FashionStar connection verification"
                    )
                if joint.firmware_version is not None:
                    firmware[joint.servo_id] = joint.firmware_version
            else:
                raise ValueError(f"unsupported bus type {bus.type.value}")
            profiles[joint.servo_id] = profile
        supports_software_disable = True
        supports_verified_disable = (
            self.spec.backend == Backend.MOCK
            or bus.type == BusType.FEETECH_MODBUS_RTU
        )
        supports_verified_identity = (
            self.spec.backend == Backend.MOCK
            or bus.type == BusType.FEETECH_MODBUS_RTU
            or (
                bus.type == BusType.FEETECH_PACKET
                and len(expected) == len(profiles)
            )
        )
        return (
            profiles,
            expected,
            firmware,
            supports_software_disable,
            supports_verified_disable,
            supports_verified_identity,
        )

    @staticmethod
    def _calculate_budget(
        bus: BusSpec,
        profiles: Mapping[int, Any],
        firmware: Mapping[int, int],
        synchronized: Optional[bool] = None,
    ) -> BusBudget:
        """Estimate worst-cycle wire occupancy for one negotiated protocol path."""

        count = len(profiles)
        turnaround = 0.0
        if bus.type == BusType.FEETECH_PACKET:
            wire_bytes = 16 + 31 * count
        elif bus.type == BusType.FEETECH_MODBUS_RTU:
            wire_bytes = 56 * count
        elif bus.type == BusType.FASHIONSTAR_RS485:
            if synchronized is None:
                synchronized = all(
                    profile.supports_sync(firmware.get(device_id))
                    for device_id, profile in profiles.items()
                )
            if synchronized:
                wire_bytes = 16 + 39 * count
                turnaround = 2.0 * max(
                    profile.minimum_command_interval_s for profile in profiles.values()
                )
            else:
                wire_bytes = 45 * count
                turnaround = 2.0 * sum(
                    profile.minimum_command_interval_s for profile in profiles.values()
                )
        else:
            wire_bytes = sum(profile.wire_bytes_per_cycle for profile in profiles.values())
            turnaround = sum(
                profile.turnaround_s_per_cycle for profile in profiles.values()
            )
        return calculate_bus_budget(
            baudrate=bus.baudrate,
            cycle_hz=bus.cycle_hz,
            wire_bytes_per_cycle=wire_bytes,
            turnaround_s_per_cycle=turnaround,
            max_utilization=bus.max_utilization,
        )

    def _create_protocol(self, bus: BusSpec):
        """Instantiate, but do not connect, the protocol selected during preflight."""

        profiles = self._profiles[bus.name]
        if self.spec.backend == Backend.MOCK:
            # Mock definitions preserve the same bus grouping and public units while
            # replacing only transport/protocol I/O.
            definitions: dict[int, MockDeviceDefinition] = {}
            for group in self._groups.values():
                if group.bus != bus.name:
                    continue
                if group.hand is not None:
                    definitions[group.hand.slave_id] = MockDeviceDefinition(
                        (0.0,) * len(group.joint_names)
                    )
                else:
                    for joint in group.joints:
                        definitions[joint.servo_id] = MockDeviceDefinition(
                            (joint.home_position_rad * joint.direction,)
                        )
            return MockBusProtocol(definitions, clock_ns=self._clock_ns)
        transport = self._transport_factory(
            bus.port,
            bus.baudrate,
            SerialDirectionControl(bus.direction_control.value),
        )
        if bus.type == BusType.FEETECH_PACKET:
            return FeetechPacketBusProtocol(
                transport,
                profiles,
                expected_model_numbers=self._expected_models[bus.name],
                clock_ns=self._clock_ns,
            )
        if bus.type == BusType.FEETECH_MODBUS_RTU:
            return FeetechModbusBusProtocol(transport, profiles, clock_ns=self._clock_ns)
        if bus.type == BusType.FASHIONSTAR_RS485:
            return FashionStarBusProtocol(
                transport,
                profiles,
                firmware_versions=self._firmware_versions[bus.name],
                clock_ns=self._clock_ns,
            )
        return LinkerHandModbusProtocol(transport, profiles, clock_ns=self._clock_ns)

    def _create_estimator(
        self,
        group: _GroupPlan,
        bus: BusSpec,
    ) -> RobustJointStateEstimator:
        """Derive robust-filter measurement noise from vendor encoder resolutions."""

        profiles = tuple(
            self._profiles[group.bus][joint.servo_id] for joint in group.joints
        )
        if bus.type in (BusType.FEETECH_PACKET, BusType.FEETECH_MODBUS_RTU):
            position_steps = tuple(
                2.0 * math.pi / profile.counts_per_revolution for profile in profiles
            )
            velocity_steps = tuple(profile.velocity_unit_rad_s for profile in profiles)
        elif bus.type == BusType.FASHIONSTAR_RS485:
            position_steps = (math.radians(0.1),) * len(profiles)
            velocity_steps = (0.25,) * len(profiles)
        else:
            raise ValueError(f"cannot estimate state for bus {bus.type.value}")
        position_step = max(position_steps)
        velocity_step = max(velocity_steps)
        return RobustJointStateEstimator(
            len(group.joint_names),
            StateEstimatorTuning(
                acceleration_std_rad_s2=6.0,
                position_std_rad=4.0 * position_step,
                velocity_std_rad_s=12.0 * velocity_step,
                position_gate_rad=8.0 * position_step,
                reanchor_gate_rad=4.0 * position_step,
                velocity_consistency_rad_s=6.0 * velocity_step,
            ),
        )

    def _read_group(self, group: _GroupPlan, snapshot: Mapping[int, Any]) -> JointState:
        """Decode one routed group, filter it, and expose canonical order and units."""

        positions: list[float] = []
        velocities: list[float] = []
        efforts: list[float] = []
        timestamps: list[int] = []
        if group.hand is not None:
            # A hand already returns one model-ordered normalized vector.
            state = snapshot[group.hand.slave_id]
            positions.extend(state.positions)
            velocities.extend(state.velocities)
            efforts.extend(state.efforts)
            timestamps.append(state.timestamp_ns)
        else:
            # Scalar servos are decoded individually, then direction-corrected into the
            # canonical joint convention before filtering.
            for joint in group.joints:
                state = snapshot[joint.servo_id]
                if isinstance(state, MockDeviceState):
                    position = state.positions[0]
                    velocity = state.velocities[0]
                    effort = state.efforts[0]
                elif isinstance(state, FashionStarMonitorState):
                    position = state.position_rad
                    velocity = math.nan
                    effort = 0.0
                elif isinstance(state, (FeetechPacketFastState, FeetechModbusFastState)):
                    position = state.position_rad
                    velocity = state.velocity_rad_s
                    effort = self._estimate_feetech_effort(
                        self._profiles[group.bus][joint.servo_id],
                        state.current_a,
                        joint.direction,
                    )
                else:
                    raise RuntimeError(f"unsupported state type {type(state).__name__}")
                position *= joint.direction
                velocity *= joint.direction
                positions.append(position)
                velocities.append(velocity)
                efforts.append(effort)
                timestamps.append(state.timestamp_ns)
        estimator = self._estimators.get(group.name)
        if estimator is not None:
            # The oldest member timestamp identifies the coherent group sample and also
            # suppresses repeated filtering when an actor snapshot is read twice.
            sample_id = min(timestamps)
            estimate = estimator.update(
                positions,
                velocities,
                timestamp_s=sample_id / 1e9,
                sample_id=sample_id,
            )
            positions = estimate.positions.tolist()
            velocities = estimate.velocities.tolist()
        if group.metadata is not None:
            # Arm limits select the continuous angle branch. Non-arm rotary groups keep
            # their public representation wrapped to [-pi, pi].
            references = 0.5 * (
                np.asarray(group.metadata.lower_limits)
                + np.asarray(group.metadata.upper_limits)
            )
            positions = unwrap_near(positions, references).tolist()
        elif group.unit == JointUnit.RADIAN or group.travel_range_rad is not None:
            positions = wrap_to_pi(positions).tolist()
        if group.travel_range_rad is not None:
            positions = np.clip(
                np.asarray(positions) / group.travel_range_rad,
                0.0,
                1.0,
            ).tolist()
            velocities = (
                np.asarray(velocities) / group.travel_range_rad
            ).tolist()
        sequence = self._sequence.get(group.name, 0)
        self._sequence[group.name] = sequence + 1
        return JointState(
            group.joint_names,
            positions,
            velocities,
            efforts,
            min(timestamps),
            sequence,
            group.unit,
        )

    @staticmethod
    def _estimate_feetech_effort(profile: Any, current_a: float, direction: int) -> float:
        """Estimate signed output torque after removing documented no-load current."""

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

    def _sensor_states(self, snapshots: Mapping[str, Any]) -> dict[str, SensorState]:
        """Expose low-rate bus telemetry as immutable generic sensor channels."""

        result: dict[str, SensorState] = {}
        for bus_name, snapshot in snapshots.items():
            if not isinstance(snapshot, Mapping) or not snapshot:
                continue
            values = {str(device_id): state for device_id, state in snapshot.items()}
            timestamps = tuple(
                timestamp
                for state in snapshot.values()
                if type(timestamp := getattr(state, "timestamp_ns", None)) is int
            )
            timestamp_ns = min(timestamps) if timestamps else self._clock_ns()
            sequence_key = f"sensor:{bus_name}"
            sequence = self._sequence.get(sequence_key, 0)
            self._sequence[sequence_key] = sequence + 1
            result[bus_name] = SensorState(
                bus_name,
                values,
                timestamp_ns,
                sequence,
            )
        return result

    def _validate_group_command(
        self,
        group: _GroupPlan,
        command: JointCommand,
        generation: int,
        now_ns: int,
    ) -> None:
        """Validate public units, limits, generation, and protocol capabilities."""

        # Name order is part of the command contract; silently reordering would hide a
        # model/configuration mismatch and could command the wrong physical joint.
        if command.names != group.joint_names:
            raise ValueError(
                f"joint group '{group.name}' expects names {group.joint_names}, "
                f"got {command.names}"
            )
        if command.unit != group.unit:
            raise ValueError(
                f"joint group '{group.name}' expects {group.unit.value} commands"
            )
        if command.generation != generation or command.deadline_ns < now_ns:
            raise ValueError(f"joint group '{group.name}' command is stale")
        if group.unit == JointUnit.NORMALIZED:
            if np.any(command.positions < 0.0) or np.any(command.positions > 1.0):
                raise ValueError(f"joint group '{group.name}' positions must be in [0, 1]")
        elif group.metadata is not None:
            lower = np.asarray(group.metadata.lower_limits)
            upper = np.asarray(group.metadata.upper_limits)
            if np.any(command.positions < lower) or np.any(command.positions > upper):
                raise ValueError(f"joint group '{group.name}' exceeds URDF position limits")
        # Capability checks happen before any vendor payload is encoded or staged.
        bus = self.preflight.buses[group.bus].spec
        has_velocity = command.velocity_limits is not None
        has_acceleration = command.acceleration_limits is not None
        has_effort = command.effort_limits is not None
        if group.hand is not None and (has_velocity or has_acceleration or has_effort):
            raise ValueError(
                f"joint group '{group.name}' supports position commands only"
            )
        if self.spec.backend == Backend.MOCK:
            if has_acceleration or has_effort or (
                has_velocity and group.metadata is None
            ):
                raise ValueError(
                    f"joint group '{group.name}' contains limits unsupported by its mock device"
                )
            return
        if bus.type == BusType.FASHIONSTAR_RS485:
            if has_acceleration or has_effort or (
                has_velocity and group.metadata is None
            ):
                raise ValueError(
                    f"joint group '{group.name}' contains limits unsupported by FashionStar"
                )
        elif bus.type == BusType.FEETECH_MODBUS_RTU and has_effort:
            raise ValueError(
                f"joint group '{group.name}' cannot map SI effort limits to Modbus torque ratio"
            )
        elif bus.type == BusType.FEETECH_PACKET and has_effort:
            unsupported = tuple(
                joint.name
                for joint in group.joints
                if self._profiles[group.bus][joint.servo_id].family
                != FeetechPacketFamily.HLS
            )
            if unsupported:
                raise ValueError(
                    "FEETECH packet effort limits require HLS profiles; unsupported joints: "
                    + ", ".join(unsupported)
                )

    def _encode_group(
        self,
        group: _GroupPlan,
        command: JointCommand,
    ) -> tuple[MotionEnvelope, ...]:
        """Convert one validated canonical command into vendor motion envelopes."""

        if group.hand is not None:
            return (
                self._envelope(
                    group.hand.slave_id,
                    LinkerHandMotion(tuple(command.positions)),
                    command,
                ),
            )
        result: list[MotionEnvelope] = []
        bus = self.preflight.buses[group.bus].spec
        for index, joint in enumerate(group.joints):
            # Undo public normalization and apply mechanical direction only at the
            # hardware boundary; all control/model code remains in canonical SI units.
            position = float(command.positions[index])
            if group.travel_range_rad is not None:
                position *= group.travel_range_rad
            position *= joint.direction
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
                if velocity is not None:
                    velocity *= group.travel_range_rad
                if acceleration is not None:
                    acceleration *= group.travel_range_rad
            payload: Any
            if self.spec.backend == Backend.MOCK:
                payload = MockMotion((position,))
            elif bus.type == BusType.FEETECH_PACKET:
                profile = self._profiles[group.bus][joint.servo_id]
                current_limit = self._packet_current_limit(
                    profile,
                    None
                    if command.effort_limits is None
                    else float(command.effort_limits[index]),
                )
                self._validate_packet_register_limits(
                    profile,
                    position,
                    self._raw_packet_positions.get((group.bus, joint.servo_id)),
                    velocity,
                    acceleration,
                    current_limit,
                )
                payload = FeetechPacketMotion(
                    position,
                    velocity,
                    acceleration,
                    current_limit,
                )
            elif bus.type == BusType.FEETECH_MODBUS_RTU:
                profile = self._profiles[group.bus][joint.servo_id]
                self._validate_modbus_register_limits(
                    profile,
                    position,
                    velocity,
                    acceleration,
                )
                payload = FeetechModbusMotion(position, velocity, acceleration)
            elif bus.type == BusType.FASHIONSTAR_RS485:
                payload = FashionStarMotion(position)
            else:
                raise RuntimeError(f"unsupported motion bus {bus.type.value}")
            result.append(self._envelope(joint.servo_id, payload, command))
        return tuple(result)

    @staticmethod
    def _packet_current_limit(profile: Any, effort_limit_nm: Optional[float]) -> Optional[float]:
        """Convert an SI torque ceiling to the HLS current-limit convention."""

        if effort_limit_nm is None:
            return None
        torque_constant = profile.torque_constant_kgcm_per_a
        no_load_current = profile.no_load_current_a
        if torque_constant is None or no_load_current is None:
            raise ValueError(
                f"FEETECH profile {profile.model} cannot convert SI effort to current"
            )
        if effort_limit_nm == 0.0:
            return 0.0
        return no_load_current + effort_limit_nm / (torque_constant * 0.0980665)

    @staticmethod
    def _validate_packet_register_limits(
        profile: Any,
        position_rad: float,
        reference_position_rad: Optional[float],
        velocity_rad_s: Optional[float],
        acceleration_rad_s2: Optional[float],
        current_limit_a: Optional[float],
    ) -> None:
        """Prove a packet-family command fits every target register before staging."""

        if reference_position_rad is None:
            raise ValueError(
                f"FEETECH profile {profile.model} requires a current position sample"
            )
        goal_position = round(
            position_rad * profile.counts_per_revolution / (2.0 * math.pi)
        )
        current_position = round(
            reference_position_rad
            * profile.counts_per_revolution
            / (2.0 * math.pi)
        )
        # This call validates that the nearest equivalent angle still fits the servo's
        # signed-magnitude multi-turn range; the protocol repeats the same conversion.
        nearest_multiturn_position_target(
            current_position,
            goal_position,
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
        for name, raw, maximum in values:
            if not 0 <= raw <= maximum:
                raise ValueError(
                    f"FEETECH {name} cannot be encoded by profile {profile.model}"
                )
        if current_limit_a is not None:
            raw_current = round(current_limit_a / profile.current_unit_a)
            if not 0 <= raw_current <= 0x7FFF:
                raise ValueError(
                    f"FEETECH current limit cannot be encoded by profile {profile.model}"
                )

    def _cache_packet_positions(self, bus_name: str, snapshot: Any) -> None:
        """Cache raw continuous packet positions needed for multi-turn prevalidation."""

        bus = self.preflight.buses[bus_name].spec
        if bus.type != BusType.FEETECH_PACKET or not isinstance(snapshot, Mapping):
            return
        for device_id, state in snapshot.items():
            if isinstance(state, FeetechPacketFastState):
                self._raw_packet_positions[(bus_name, device_id)] = state.position_rad

    @staticmethod
    def _validate_modbus_register_limits(
        profile: Any,
        position_rad: float,
        velocity_rad_s: Optional[float],
        acceleration_rad_s2: Optional[float],
    ) -> None:
        """Prove a FEETECH Modbus command fits signed/unsigned register widths."""

        position = round(
            position_rad * profile.counts_per_revolution / (2.0 * math.pi)
        )
        if not -0x8000 <= position <= 0x7FFF:
            raise ValueError(
                f"FEETECH Modbus position cannot be encoded by profile {profile.model}"
            )
        for name, value, unit, default in (
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
                    f"FEETECH Modbus {name} cannot be encoded by profile {profile.model}"
                )

    @staticmethod
    def _envelope(device_id: int, payload: Any, command: JointCommand) -> MotionEnvelope:
        """Create an actor envelope; the actor generation is attached at submission."""

        return MotionEnvelope(
            ("position", device_id),
            device_id,
            payload,
            command.submitted_at_ns,
            command.deadline_ns,
            0,
        )

    def _hardware_state_is_stale(self, now_ns: int) -> bool:
        """Return whether any bus missed its cycle-derived freshness deadline."""

        for bus_name, actor in self._actors.items():
            last_state_ns = actor.diagnostics().last_state_ns
            if (
                last_state_ns is None
                or now_ns - last_state_ns > self._state_timeout_ns[bus_name]
            ):
                return True
        return False

    def _refresh_position_pipeline_feedback(self) -> None:
        """Seed controller memory from current hardware before enabling motion."""

        if not self._pipelines:
            return
        bus_names = {
            self._groups[name].bus
            for name in self._pipelines
            if self._groups[name].bus in self._actors
        }
        snapshots = {
            name: self._actors[name].get_snapshot()
            for name in bus_names
        }
        for bus_name, snapshot in snapshots.items():
            self._cache_packet_positions(bus_name, snapshot)
        for name, pipeline in self._pipelines.items():
            group = self._groups[name]
            snapshot = snapshots.get(group.bus)
            if snapshot is not None:
                pipeline.update_feedback(self._read_group(group, snapshot))

    def _rebase_position_pipelines(self) -> None:
        """Clear learned controller history after a safety-state discontinuity."""

        for pipeline in self._pipelines.values():
            pipeline.rebase_to_feedback()

    def _hardware_fault_reason(
        self,
        fast_snapshots: Mapping[str, Any],
        slow_snapshots: Mapping[str, Any],
    ) -> Optional[str]:
        """Translate vendor status fields into the first actionable runtime fault."""

        for bus in self.spec.buses:
            fast = fast_snapshots.get(bus.name)
            if isinstance(fast, Mapping):
                for device_id, state in fast.items():
                    status = getattr(state, "status", None)
                    if type(status) is not int:
                        continue
                    if bus.type == BusType.FASHIONSTAR_RS485:
                        # Bit zero is the documented torque-enabled state, not a fault.
                        status &= 0xFE
                    if status:
                        return (
                            f"bus '{bus.name}' device {device_id} reported "
                            f"hardware status 0x{status:02x}"
                        )
            if bus.type != BusType.LINKER_HAND_RS485:
                continue
            slow = slow_snapshots.get(bus.name)
            if not isinstance(slow, Mapping):
                continue
            for device_id, state in slow.items():
                errors = tuple(getattr(state, "errors", ()))
                failed = tuple(index for index, value in enumerate(errors) if value)
                if failed:
                    return (
                        f"bus '{bus.name}' device {device_id} reported Linker Hand "
                        f"joint errors at indices {failed}"
                    )
        return None

    def _latch_hardware_fault(self, reason: str) -> None:
        """Latch software FAULT and best-effort stop every bus without masking reason."""

        if self._safety.snapshot().state != RuntimeSafetyState.FAULT:
            self._safety.fault(reason)
        for actor in self._actors.values():
            try:
                actor.submit_safety(
                    "emergency_stop",
                    None,
                    wait=False,
                    clear_motion=True,
                )
            except BaseException:
                try:
                    actor.discard_motion()
                except BaseException:
                    pass

    def _submit_set_enabled(self, enabled: bool) -> None:
        """Apply enable/disable semantics across heterogeneous bus capabilities."""

        first_error: Optional[BaseException] = None
        for bus_name, actor in self._actors.items():
            preflight = self.preflight.buses.get(bus_name)
            supports_disable = (
                preflight is None or preflight.supports_software_disable
            )
            if enabled and not supports_disable:
                continue
            # A bus without documented disable can only hold. Such hardware is already
            # required by preflight to have an independent external emergency stop.
            label = "set_enabled" if supports_disable else "hold"
            payload = enabled if supports_disable else None
            try:
                actor.submit_safety(
                    label,
                    payload,
                    wait=True,
                    clear_motion=not enabled,
                )
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            self._latch_hardware_fault("safety task 'set_enabled' failed")
            raise first_error

    def _submit_all_safety(
        self,
        label: str,
        payload: Any,
        *,
        wait: bool,
        clear_motion: bool = True,
    ) -> None:
        """Submit one safety action to every bus and preserve the first failure."""

        first_error: Optional[BaseException] = None
        for actor in self._actors.values():
            try:
                actor.submit_safety(
                    label,
                    payload,
                    wait=wait,
                    clear_motion=clear_motion,
                )
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            # Failure of a non-stop safety action escalates to emergency stop on every
            # remaining actor; cleanup attempts never replace the originating error.
            self._safety.fault(f"safety task '{label}' failed")
            if label != "emergency_stop":
                for actor in self._actors.values():
                    try:
                        actor.submit_safety(
                            "emergency_stop",
                            None,
                            wait=False,
                            clear_motion=True,
                        )
                    except BaseException:
                        pass
            raise first_error

    def _require_connected(self) -> None:
        """Require all actors healthy, latching a robot-wide fault otherwise."""

        if not self._actors:
            raise RuntimeError("robot runtime is not connected")
        failed = tuple(name for name, actor in self._actors.items() if not actor.connected)
        if not failed:
            return
        reason = "robot runtime bus actor faulted: " + ", ".join(failed)
        if self._safety.snapshot().state != RuntimeSafetyState.FAULT:
            self._safety.fault(reason)
            for actor in self._actors.values():
                try:
                    if actor.connected:
                        actor.submit_safety(
                            "emergency_stop",
                            None,
                            wait=False,
                            clear_motion=True,
                        )
                    else:
                        actor.discard_motion()
                except BaseException:
                    pass
        raise RuntimeError(reason)


def _disconnect_actors(actors: Mapping[str, SerialBusActor]) -> Optional[BaseException]:
    """Disconnect all actors in reverse order and return the first cleanup error."""

    first_error: Optional[BaseException] = None
    for actor in reversed(tuple(actors.values())):
        try:
            actor.disconnect()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    return first_error


__all__ = [
    "BusPreflight",
    "JointGroupInfo",
    "RobotRuntime",
    "RuntimeDiagnostics",
    "RuntimePreflight",
]
