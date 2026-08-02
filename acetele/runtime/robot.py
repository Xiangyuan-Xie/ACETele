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
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

import numpy as np

from acetele.control import PositionControlDiagnostics, PositionControlPipeline
from acetele.core import (
    JointCommand,
    JointState,
    JointUnit,
    RobotCommand,
    RobotState,
    SensorState,
)
from acetele.estimation import RobustJointStateEstimator
from acetele.hardware.buses import (
    BusActor,
    BusActorDiagnostics,
    DeviceEnableRequest,
    MotionCommitGate,
    MotionEnvelope,
    SerialDirectionControl,
    SerialTransport,
)
from acetele.hardware.devices.adapter import AdapterRegistry
from acetele.model import unwrap_near, wrap_to_pi
from acetele.runtime.preflight import (
    BusPreflight,
    JointGroupPlan,
    RuntimePreflight,
    build_runtime_plan,
)
from acetele.runtime.safety import (
    RuntimeSafetyController,
    RuntimeSafetyState,
    SafetySnapshot,
    SafetyTransition,
)
from acetele.specification import Backend, RobotSpec


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
        adapter_registry: Optional[AdapterRegistry] = None,
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
        self._actors: dict[str, BusActor] = {}
        self._sequence: dict[str, int] = {}
        self._last_state_ns: Optional[int] = None
        plan = build_runtime_plan(spec, adapter_registry=adapter_registry)
        self.preflight = plan.preflight
        self._adapter_plans = dict(plan.adapters)
        self._groups = dict(plan.groups)
        self._pin_models = dict(plan.pin_models)
        self._pipelines: dict[str, PositionControlPipeline] = {}
        self._estimators: dict[str, RobustJointStateEstimator] = {}
        self._position_references: dict[str, dict[int, float]] = {
            bus.name: {} for bus in spec.buses
        }
        controls = {arm.name: arm.control for arm in spec.arms}
        for name, group in self._groups.items():
            if group.is_arm:
                if group.metadata is None:
                    raise RuntimeError(f"arm group '{name}' has no model metadata")
                self._pipelines[name] = PositionControlPipeline(
                    group.metadata,
                    controls[name],
                    pin_model=self._pin_models.get(name),
                )
            if spec.backend == Backend.PHYSICAL and group.hand is None:
                adapter_plan = self._adapter_plans[group.bus]
                tuning = adapter_plan.adapter.estimator_tuning(adapter_plan, group)
                if tuning is not None:
                    self._estimators[name] = RobustJointStateEstimator(
                        len(group.joint_names),
                        tuning,
                    )
        self._state_timeout_ns = {
            name: max(50_000_000, round(2.5e9 / bus.spec.cycle_hz))
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
        for references in self._position_references.values():
            references.clear()
        actors: dict[str, BusActor] = {}
        try:
            # Connect one actor at a time, retaining local ownership until every bus has
            # negotiated capabilities and produced a complete initial state snapshot.
            for bus in self.spec.buses:
                adapter_plan = self._adapter_plans[bus.name]
                protocol = adapter_plan.adapter.create_protocol(
                    adapter_plan,
                    self._transport_factory,
                    clock_ns=self._clock_ns,
                )
                actor = BusActor(
                    protocol,
                    cycle_hz=bus.cycle_hz,
                    motion_watchdog_ns=self._safety.command_timeout_ns,
                    state_timeout_ns=self._state_timeout_ns[bus.name],
                )
                actor.connect()
                actors[bus.name] = actor
                actual_budget = adapter_plan.adapter.negotiated_budget(
                    adapter_plan,
                    protocol,
                )
                actual_budget.require_feasible(
                    context=f"bus '{bus.name}' negotiated capabilities"
                )
            for bus_name, actor in actors.items():
                self._capture_position_references(
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
            plan = self._adapter_plans[bus.name]
            bus_targets = plan.adapter.calibration_targets(plan)
            if bus_targets is None:
                unsupported.append(f"{bus.name} ({bus.type.value})")
                continue
            targets[bus.name] = MappingProxyType(dict(bus_targets))
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
            self._capture_position_references(bus_name, snapshot)
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
            # Refresh every physical bus only after the complete robot command has passed
            # validation and safety admission. The actor thread can then enforce loss of
            # this heartbeat even if ROS/ZMQ callback scheduling stops temporarily.
            for actor in self._actors.values():
                if isinstance(actor, BusActor):
                    actor.refresh_motion_watchdog(accepted_ns)
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
    def enable_arm_groups(self, group_names: tuple[str, ...]) -> None:
        """Enter READY while enabling only selected arms on their shared buses.

        Leader synchronization uses this boundary to hold arm joints without also
        applying torque to a passive gripper used as the operator's start trigger.
        """

        group_names = tuple(group_names)
        if not group_names:
            raise ValueError("at least one arm group must be enabled")
        if any(not isinstance(group_name, str) for group_name in group_names):
            raise ValueError("arm group names must be strings")
        if len(set(group_names)) != len(group_names):
            raise ValueError("arm groups to enable must be unique")
        unknown = set(group_names) - set(self._groups)
        if unknown:
            raise ValueError(
                "unknown joint groups to enable: "
                + ", ".join(sorted(unknown))
            )
        non_arm = tuple(
            group_name
            for group_name in group_names
            if not self._groups[group_name].is_arm
        )
        if non_arm:
            raise ValueError(
                "targeted synchronization enable accepts arm groups only: "
                + ", ".join(non_arm)
            )
        self._require_connected()
        state = self._safety.snapshot().state
        if state == RuntimeSafetyState.FAULT:
            raise RuntimeError("robot runtime fault is latched; reset it explicitly")
        if state not in (RuntimeSafetyState.SAFE_DISABLED, RuntimeSafetyState.HOLD):
            raise RuntimeError(f"cannot enter READY from {state.value}")

        device_ids_by_bus: dict[str, list[int]] = {}
        for group_name in group_names:
            group = self._groups[group_name]
            device_ids = tuple(joint.servo_id for joint in group.joints)
            device_ids_by_bus.setdefault(group.bus, []).extend(device_ids)

        self._refresh_position_pipeline_feedback()
        self._rebase_position_pipelines()
        self._submit_arm_enable(device_ids_by_bus)
        self._safety.ready()

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
        for references in self._position_references.values():
            references.clear()
        first_error: Optional[BaseException] = None
        # Request the strongest stop everywhere before closing any transport. Continue
        # after errors so one broken bus cannot leak all remaining resources.
        for actor in actors.values():
            if not actor.connected:
                # A faulted worker already attempted its protocol-level emergency stop
                # before terminating. No worker remains to consume another FIFO task;
                # proceed directly to bounded transport cancellation and disconnect.
                actor.discard_motion()
                continue
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

    def _read_group(
        self,
        group: JointGroupPlan,
        snapshot: Mapping[int, Any],
    ) -> JointState:
        """Decode, estimate, and expose one group in canonical model order."""

        adapter_plan = self._adapter_plans[group.bus]
        sample = adapter_plan.adapter.decode_group(adapter_plan, group, snapshot)
        positions = list(sample.positions)
        velocities = list(sample.velocities)
        estimator = self._estimators.get(group.name)
        if estimator is not None:
            estimate = estimator.update(
                positions,
                velocities,
                timestamp_s=sample.timestamp_ns / 1e9,
                sample_id=sample.timestamp_ns,
            )
            positions = estimate.positions.tolist()
            velocities = estimate.velocities.tolist()
        if group.metadata is not None:
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
            sample.efforts,
            sample.timestamp_ns,
            sequence,
            group.unit,
        )

    def _capture_position_references(
        self,
        bus_name: str,
        snapshot: Mapping[int, Any],
    ) -> None:
        """Refresh adapter-specific continuous positions used during encoding."""

        plan = self._adapter_plans[bus_name]
        references = plan.adapter.capture_position_references(plan, snapshot)
        self._position_references[bus_name].update(references)

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
        group: JointGroupPlan,
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
        plan = self._adapter_plans[group.bus]
        plan.adapter.validate_command(plan, group, command)

    def _encode_group(
        self,
        group: JointGroupPlan,
        command: JointCommand,
    ) -> tuple[MotionEnvelope, ...]:
        """Delegate canonical command encoding to the planned bus adapter."""

        plan = self._adapter_plans[group.bus]
        return plan.adapter.encode_group(
            plan,
            group,
            command,
            self._position_references[group.bus],
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
            self._capture_position_references(bus_name, snapshot)
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
        """Ask each adapter to interpret its own status and telemetry fields."""

        for bus_name, plan in self._adapter_plans.items():
            reason = plan.adapter.fault_reason(
                plan,
                fast_snapshots.get(bus_name),
                slow_snapshots.get(bus_name),
            )
            if reason is not None:
                return reason
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

    def _submit_arm_enable(self, device_ids_by_bus: Mapping[str, list[int]]) -> None:
        """Enable selected arm devices and fault the runtime on partial failure."""

        first_error: Optional[BaseException] = None
        for bus_name, device_ids in device_ids_by_bus.items():
            preflight = self.preflight.buses[bus_name]
            if not preflight.supports_software_disable:
                continue
            request = DeviceEnableRequest(True, tuple(device_ids))
            try:
                self._actors[bus_name].submit_safety(
                    "set_enabled",
                    request,
                    wait=True,
                    clear_motion=False,
                )
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            self._latch_hardware_fault("targeted joint-group enable failed")
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
            self._synchronize_actor_watchdogs()
            return
        details: list[str] = []
        for name in failed:
            actor = self._actors[name]
            diagnostics = getattr(actor, "diagnostics", None)
            fault = None
            if callable(diagnostics):
                try:
                    fault = diagnostics().fault
                except BaseException:
                    fault = None
            details.append(name if not fault else f"{name} ({fault})")
        reason = "robot runtime bus actor faulted: " + ", ".join(details)
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

    def _synchronize_actor_watchdogs(self) -> None:
        """Mirror an autonomous bus HOLD into the robot-wide safety state."""

        tripped = tuple(
            name
            for name, actor in self._actors.items()
            if isinstance(actor, BusActor) and actor.motion_watchdog_tripped
        )
        if not tripped:
            return
        state = self._safety.snapshot().state
        if state not in (RuntimeSafetyState.READY, RuntimeSafetyState.ACTIVE):
            return
        # One timed-out bus invalidates the complete robot command generation. Hold every
        # assembly so a second bus cannot continue moving after the first one stopped.
        self._safety.hold()
        self._rebase_position_pipelines()
        self._submit_all_safety("hold", None, wait=False, clear_motion=True)


def _disconnect_actors(actors: Mapping[str, BusActor]) -> Optional[BaseException]:
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
