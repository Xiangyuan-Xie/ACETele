"""Dependency-light URDF metadata and optional Pinocchio reduction.

XML parsing remains independent of Pinocchio so mapping and limit errors fail during
static preflight, before a serial port is opened. Pinocchio is imported only when a
model-based controller is enabled.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class UrdfJoint:
    """URDF joint fields required by runtime validation and control."""

    name: str
    type: str
    parent: str
    child: str
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None

    @property
    def angular_dof(self) -> bool:
        """Return whether the joint contributes one angular degree of freedom."""

        return self.type in ("revolute", "continuous")


@dataclass(frozen=True)
class ArmModelMetadata:
    """Ordered joint limits for one configured arm."""

    joint_names: tuple[str, ...]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    velocity_limits: tuple[float, ...]
    effort_limits: tuple[float, ...]


@dataclass(frozen=True)
class UrdfModel:
    """Validated URDF tree with deterministic movable-joint order."""

    path: Path
    root_link: str
    links: tuple[str, ...]
    joints: Mapping[str, UrdfJoint]
    movable_joint_order: tuple[str, ...]

    def require_frame(self, frame_name: str) -> str:
        """Require a link frame that can serve as a Cartesian control frame."""

        if not isinstance(frame_name, str) or not frame_name.strip():
            raise ValueError("URDF frame name must be a non-empty string")
        if frame_name not in self.links:
            raise ValueError(f"URDF is missing configured tool frame '{frame_name}'")
        return frame_name

    def arm_metadata(
        self,
        joint_names: Sequence[str],
        *,
        require_limits: bool = True,
        require_angular: bool = True,
    ) -> ArmModelMetadata:
        """Select an arm after validating joint type, limits, and tree order."""

        names = _joint_names(joint_names)
        missing = tuple(name for name in names if name not in self.joints)
        if missing:
            raise ValueError(f"URDF is missing configured joints: {missing}")
        for name in names:
            joint = self.joints[name]
            if require_angular and not joint.angular_dof:
                raise ValueError(
                    f"configured joint '{name}' must be revolute or continuous; "
                    f"got '{joint.type}'"
                )
            if not require_angular and joint.type == "fixed":
                raise ValueError(f"configured joint '{name}' must be movable")
        selected = set(names)
        # Filtering canonical tree order by the selected names validates ordering
        # without requiring configured arm joints to be contiguous in the full model.
        expected = tuple(name for name in self.movable_joint_order if name in selected)
        if expected != names:
            raise ValueError(
                "configured joint names must follow URDF kinematic order; "
                f"expected {expected}, got {names}"
            )

        lower: list[float] = []
        upper: list[float] = []
        velocity: list[float] = []
        effort: list[float] = []
        for name in names:
            joint = self.joints[name]
            if require_limits and (joint.lower is None or joint.upper is None):
                raise ValueError(
                    f"URDF joint '{name}' requires finite lower and upper limits"
                )
            lower.append(-math.inf if joint.lower is None else joint.lower)
            upper.append(math.inf if joint.upper is None else joint.upper)
            velocity.append(math.inf if joint.velocity is None else joint.velocity)
            effort.append(math.inf if joint.effort is None else joint.effort)
        return ArmModelMetadata(
            names,
            tuple(lower),
            tuple(upper),
            tuple(velocity),
            tuple(effort),
        )


def load_urdf_model(path: str | Path) -> UrdfModel:
    """Parse the URDF structure needed by hardware-free runtime preflight."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid URDF XML in {path}") from exc
    if root.tag != "robot":
        raise ValueError(f"URDF root element must be <robot>: {path}")

    links = tuple(
        element.attrib.get("name", "") for element in root.findall("link")
    )
    if not links or any(not name for name in links) or len(set(links)) != len(links):
        raise ValueError("URDF links must have unique non-empty names")
    link_set = set(links)
    joints: dict[str, UrdfJoint] = {}
    children: set[str] = set()
    by_parent: dict[str, list[UrdfJoint]] = {}
    # Parse only metadata needed by hardware-free preflight. Geometry and inertials are
    # deferred to Pinocchio when model-based control is actually enabled.
    for element in root.findall("joint"):
        name = element.attrib.get("name", "")
        joint_type = element.attrib.get("type", "")
        parent_element = element.find("parent")
        child_element = element.find("child")
        parent = "" if parent_element is None else parent_element.attrib.get("link", "")
        child = "" if child_element is None else child_element.attrib.get("link", "")
        if not name or name in joints:
            raise ValueError("URDF joints must have unique non-empty names")
        if not joint_type or parent not in link_set or child not in link_set:
            raise ValueError(f"URDF joint '{name}' has invalid type, parent, or child")
        limit = element.find("limit")
        joint = UrdfJoint(
            name,
            joint_type,
            parent,
            child,
            _optional_finite_attribute(limit, "lower", name),
            _optional_finite_attribute(limit, "upper", name),
            _optional_nonnegative_attribute(limit, "effort", name),
            _optional_nonnegative_attribute(limit, "velocity", name),
        )
        if joint.lower is not None and joint.upper is not None and joint.lower >= joint.upper:
            raise ValueError(f"URDF joint '{name}' lower limit must be less than upper")
        joints[name] = joint
        children.add(child)
        by_parent.setdefault(parent, []).append(joint)

    roots = link_set - children
    if len(roots) != 1:
        raise ValueError(f"URDF must contain exactly one root link; got {sorted(roots)}")
    root_link = next(iter(roots))
    order: list[str] = []
    visited_joints: set[str] = set()
    active_links: set[str] = set()

    # Parent-to-child traversal is canonical because PX4 arm messages carry no names
    # and cannot repair a configuration ordering error at the transport boundary.
    def visit(link: str) -> None:
        if link in active_links:
            raise ValueError(f"URDF kinematic tree contains a cycle at link '{link}'")
        active_links.add(link)
        for joint in by_parent.get(link, ()):
            if joint.name in visited_joints:
                raise ValueError(f"URDF joint '{joint.name}' is reachable more than once")
            visited_joints.add(joint.name)
            if joint.type != "fixed":
                order.append(joint.name)
            visit(joint.child)
        active_links.remove(link)

    visit(root_link)
    if len(visited_joints) != len(joints):
        raise ValueError("URDF contains joints disconnected from its root link")
    return UrdfModel(path, root_link, links, MappingProxyType(joints), tuple(order))


def build_reduced_pinocchio_model(
    urdf_path: str | Path,
    joint_names: Sequence[str],
) -> Any:
    """Build a Pinocchio model containing only the configured one-DOF joints."""

    import pinocchio as pin

    names = _joint_names(joint_names)
    model = pin.buildModelFromUrdf(str(Path(urdf_path).expanduser().resolve()))
    model_names = tuple(str(name) for name in model.names[1:])
    missing = tuple(name for name in names if name not in model_names)
    if missing:
        raise ValueError(f"Pinocchio model is missing configured joints: {missing}")
    selected = set(names)
    expected = tuple(name for name in model_names if name in selected)
    if expected != names:
        raise ValueError(
            "configured joint names must follow Pinocchio model order; "
            f"expected {expected}, got {names}"
        )
    for name in names:
        joint = model.joints[model.getJointId(name)]
        if joint.nq != 1 or joint.nv != 1:
            raise ValueError(f"Pinocchio joint '{name}' must have exactly one DOF")
    # Lock every movable joint outside this arm at neutral configuration. The reduced
    # model must then match the controller vector exactly.
    fixed_ids = [model.getJointId(name) for name in model_names if name not in selected]
    reduced = (
        model
        if not fixed_ids
        else pin.buildReducedModel(model, fixed_ids, pin.neutral(model))
    )
    if reduced.nq != len(names) or reduced.nv != len(names):
        raise ValueError(
            f"reduced Pinocchio model has nq/nv={reduced.nq}/{reduced.nv}, "
            f"expected {len(names)}/{len(names)}"
        )
    return reduced


def _joint_names(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize a non-empty, unique sequence of canonical model names."""

    if isinstance(values, (str, bytes)):
        raise ValueError("joint names must be a sequence")
    names = tuple(values)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("joint names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("joint names must be unique")
    return names


def _optional_finite_attribute(
    element: ET.Element | None,
    attribute: str,
    joint_name: str,
) -> float | None:
    """Parse one optional finite URDF limit attribute with joint context."""

    if element is None or attribute not in element.attrib:
        return None
    try:
        value = float(element.attrib[attribute])
    except ValueError as exc:
        raise ValueError(
            f"URDF joint '{joint_name}' has invalid {attribute} limit"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"URDF joint '{joint_name}' {attribute} limit must be finite")
    return value


def _optional_nonnegative_attribute(
    element: ET.Element | None,
    attribute: str,
    joint_name: str,
) -> float | None:
    """Parse an optional effort/velocity limit that cannot be negative."""

    value = _optional_finite_attribute(element, attribute, joint_name)
    if value is not None and value < 0.0:
        raise ValueError(f"URDF joint '{joint_name}' {attribute} limit cannot be negative")
    return value


__all__ = [
    "ArmModelMetadata",
    "UrdfJoint",
    "UrdfModel",
    "build_reduced_pinocchio_model",
    "load_urdf_model",
]
