"""Side-effect-free robot model, adapter, capability, and bandwidth preflight."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional

from acetele.core import JointUnit
from acetele.hardware.buses import BusBudget
from acetele.hardware.devices.adapter import (
    AdapterPlan,
    AdapterRegistry,
    default_adapter_registry,
)
from acetele.model import ArmDynamics, ArmModelMetadata, load_urdf_model
from acetele.specification import (
    Backend,
    BusSpec,
    DexterousHandSpec,
    JointSpec,
    ParallelGripperSpec,
    RobotSpec,
)


@dataclass(frozen=True)
class JointGroupPlan:
    """Canonical routing facts shared by models, control, and one bus adapter."""

    name: str
    bus: str
    unit: JointUnit
    joint_names: tuple[str, ...]
    joints: tuple[JointSpec, ...]
    metadata: Optional[ArmModelMetadata] = None
    travel_range_rad: Optional[float] = None
    hand: Optional[DexterousHandSpec] = None
    is_arm: bool = False


@dataclass(frozen=True)
class BusPreflight:
    """Validated device identities, safety capabilities, and wire budget."""

    spec: BusSpec
    budget: BusBudget
    device_models: Mapping[int, str]
    supports_software_disable: bool
    supports_verified_disable: bool
    supports_verified_identity: bool
    supports_acceleration_limits: bool
    supports_effort_control: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "device_models",
            MappingProxyType(dict(self.device_models)),
        )


@dataclass(frozen=True)
class RuntimePreflight:
    """Public static facts proven before any transport may be created."""

    urdf_path: Path
    arms: Mapping[str, ArmModelMetadata]
    buses: Mapping[str, BusPreflight]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arms", MappingProxyType(dict(self.arms)))
        object.__setattr__(self, "buses", MappingProxyType(dict(self.buses)))


@dataclass(frozen=True)
class RuntimePlan:
    """Internal immutable composition plan consumed by ``RobotRuntime``."""

    preflight: RuntimePreflight
    groups: Mapping[str, JointGroupPlan]
    adapters: Mapping[str, AdapterPlan]
    dynamics: Mapping[str, ArmDynamics]

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))
        object.__setattr__(self, "adapters", MappingProxyType(dict(self.adapters)))
        object.__setattr__(self, "dynamics", MappingProxyType(dict(self.dynamics)))


def build_runtime_plan(
    spec: RobotSpec,
    *,
    adapter_registry: Optional[AdapterRegistry] = None,
) -> RuntimePlan:
    """Validate one complete robot specification without opening hardware."""

    registry = adapter_registry or default_adapter_registry()
    urdf_path = _resolve_urdf_path(spec)
    urdf = load_urdf_model(urdf_path)
    bus_specs = {bus.name: bus for bus in spec.buses}
    groups: dict[str, JointGroupPlan] = {}
    arm_metadata: dict[str, ArmModelMetadata] = {}
    dynamics: dict[str, ArmDynamics] = {}

    for arm in spec.arms:
        arm_names = tuple(joint.name for joint in arm.joints)
        metadata = urdf.arm_metadata(arm_names, require_limits=True)
        if arm.tool_frame is not None:
            urdf.require_frame(arm.tool_frame)
        arm_metadata[arm.name] = metadata
        combined_names = arm_names
        if isinstance(arm.end_effector, ParallelGripperSpec):
            combined_names += (
                arm.end_effector.kinematic_joint_names
                or (arm.end_effector.joint.name,)
            )
        urdf.arm_metadata(
            combined_names,
            require_limits=False,
            require_angular=False,
        )
        effort_control = (
            arm.control.gravity_compensation or arm.control.redundancy_posture
        )
        if effort_control:
            if arm.tool_frame is None:
                raise ValueError(
                    f"arm '{arm.name}' requires tool_frame for effort control"
                )
            rest = arm.control.rest_posture_rad
            if rest is not None and (
                any(value < lower for value, lower in zip(rest, metadata.lower_limits))
                or any(value > upper for value, upper in zip(rest, metadata.upper_limits))
            ):
                raise ValueError(f"arm '{arm.name}' rest posture exceeds URDF limits")
            if any(not math.isfinite(value) or value <= 0.0 for value in metadata.effort_limits):
                raise ValueError(
                    f"arm '{arm.name}' requires finite positive URDF effort limits"
                )
            dynamics[arm.name] = ArmDynamics(urdf_path, arm_names, arm.tool_frame)
        groups[arm.name] = JointGroupPlan(
            arm.name,
            arm.bus,
            JointUnit.RADIAN,
            arm_names,
            arm.joints,
            metadata=metadata,
            is_arm=True,
        )

        if isinstance(arm.end_effector, ParallelGripperSpec):
            gripper = arm.end_effector
            name = f"{arm.name}.end_effector"
            groups[name] = JointGroupPlan(
                name,
                gripper.bus,
                JointUnit.NORMALIZED,
                (gripper.joint.name,),
                (gripper.joint,),
                travel_range_rad=gripper.travel_range_rad,
            )
        elif isinstance(arm.end_effector, DexterousHandSpec):
            hand = arm.end_effector
            bus = bus_specs[hand.bus]
            joint_names = registry.require(bus.type).hand_joint_names(hand)
            name = f"{arm.name}.end_effector"
            groups[name] = JointGroupPlan(
                name,
                hand.bus,
                JointUnit.NORMALIZED,
                joint_names,
                (),
                hand=hand,
            )

    adapter_plans: dict[str, AdapterPlan] = {}
    public_buses: dict[str, BusPreflight] = {}
    for bus in spec.buses:
        bus_groups = tuple(group for group in groups.values() if group.bus == bus.name)
        adapter = registry.require(bus.type)
        plan = adapter.preflight(bus, bus_groups, spec.backend)
        if (
            spec.backend == Backend.PHYSICAL
            and not plan.supports_verified_disable
            and not bus.external_estop
        ):
            raise ValueError(
                f"bus '{bus.name}' lacks verified torque disable; configure and provide "
                "an independent hardware emergency stop"
            )
        if (
            spec.backend == Backend.PHYSICAL
            and not plan.supports_verified_identity
            and not bus.allow_unverified_identity
        ):
            raise ValueError(
                f"bus '{bus.name}' protocol cannot verify every configured device model; "
                "set allow_unverified_identity=true only after independently checking "
                "the installed hardware"
            )
        plan.budget.require_feasible(context=f"bus '{bus.name}'")
        adapter_plans[bus.name] = plan
        public_buses[bus.name] = BusPreflight(
            spec=bus,
            budget=plan.budget,
            device_models=plan.device_models,
            supports_software_disable=plan.supports_software_disable,
            supports_verified_disable=plan.supports_verified_disable,
            supports_verified_identity=plan.supports_verified_identity,
            supports_acceleration_limits=plan.supports_acceleration_limits,
            supports_effort_control=plan.supports_effort_control,
        )

    for arm in spec.arms:
        if not (
            arm.control.gravity_compensation or arm.control.redundancy_posture
        ):
            continue
        adapter_plan = adapter_plans[arm.bus]
        if not adapter_plan.supports_effort_control:
            raise ValueError(
                f"arm '{arm.name}' enables effort control, but bus "
                f"'{arm.bus}' does not support calibrated torque mode"
            )

    return RuntimePlan(
        RuntimePreflight(urdf_path, arm_metadata, public_buses),
        groups,
        adapter_plans,
        dynamics,
    )


def _resolve_urdf_path(spec: RobotSpec) -> Path:
    """Resolve an explicit URDF or require the model's packaged default."""

    if spec.urdf_path is not None:
        path = Path(spec.urdf_path)
    else:
        path = (
            Path(__file__).resolve().parents[1]
            / "model"
            / "robots"
            / spec.model
            / "description"
            / f"{spec.model}.urdf"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"robot '{spec.model}' has no packaged URDF; set basic.urdf_path"
        )
    return path


__all__ = [
    "BusPreflight",
    "JointGroupPlan",
    "RuntimePlan",
    "RuntimePreflight",
    "build_runtime_plan",
]
