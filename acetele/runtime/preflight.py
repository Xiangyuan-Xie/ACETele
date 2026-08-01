"""Side-effect-free robot model, adapter, capability, and bandwidth preflight."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

from acetele.core import JointUnit
from acetele.hardware.buses import BusBudget
from acetele.hardware.devices.adapter import (
    AdapterPlan,
    AdapterRegistry,
    default_adapter_registry,
)
from acetele.model import ArmModelMetadata, build_reduced_pinocchio_model, load_urdf_model
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
    pin_models: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))
        object.__setattr__(self, "adapters", MappingProxyType(dict(self.adapters)))
        object.__setattr__(self, "pin_models", MappingProxyType(dict(self.pin_models)))


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
    pin_models: dict[str, Any] = {}

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
        if arm.control.gravity_position:
            pin_models[arm.name] = build_reduced_pinocchio_model(urdf_path, arm_names)
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
        )

    return RuntimePlan(
        RuntimePreflight(urdf_path, arm_metadata, public_buses),
        groups,
        adapter_plans,
        pin_models,
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
