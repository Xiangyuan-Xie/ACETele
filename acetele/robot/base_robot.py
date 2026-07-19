from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

from acetele.config.config_loader import ConfigLoader
from acetele.config.robot_config import RobotConfig


class BaseRobot(ABC):
    def __init__(self, config_loader: ConfigLoader | RobotConfig):
        self.robot_config = (
            config_loader
            if isinstance(config_loader, RobotConfig)
            else config_loader.get_robot_config()
        )
        robot_type = self.robot_config.robot_type
        self._robot_name = (
            f"{robot_type}_{self.robot_config.backend}_{self.robot_config.runtime}"
        )

        self._urdf_model_path: Optional[str]
        urdf_model_path = Path(__file__).resolve().parent / robot_type / "description" / f"{robot_type}.urdf"
        if urdf_model_path.exists() and urdf_model_path.is_file():
            self._urdf_model_path = str(urdf_model_path)
        else:
            self._urdf_model_path = None

    @property
    def name(self) -> str:
        return self._robot_name

    @abstractmethod
    def act(self):
        raise NotImplementedError(
            f"Class '{self.__class__.__name__}' must implement abstract method '{self.act.__name__}()'."
        )

    def close(self):
        pass

    def get_pin_model(self):
        import pinocchio as pin

        if self._urdf_model_path is None:
            raise RuntimeError("URDF model path is not available.")
        pin_model, _, _ = pin.buildModelsFromUrdf(
            filename=self._urdf_model_path, package_dirs=str(Path(self._urdf_model_path).parent)
        )
        return pin_model

    def _get_joint_position_limits(
        self,
        joint_names: Sequence[str],
    ) -> tuple[list[float], list[float]]:
        if self._urdf_model_path is None:
            raise RuntimeError("URDF model path is not available for joint position limits.")

        requested_names = self._validate_joint_names(joint_names)
        root = ET.parse(self._urdf_model_path).getroot()
        joints = {joint.attrib.get("name"): joint for joint in root.findall("joint")}
        lower_limits = []
        upper_limits = []
        for joint_name in requested_names:
            joint = joints.get(joint_name)
            if joint is None:
                raise ValueError(f"URDF joint '{joint_name}' is missing.")
            limit = joint.find("limit")
            if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
                raise ValueError(f"URDF joint '{joint_name}' must define lower and upper position limits.")
            try:
                lower = float(limit.attrib["lower"])
                upper = float(limit.attrib["upper"])
            except ValueError as exc:
                raise ValueError(f"URDF joint '{joint_name}' has invalid position limits.") from exc
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(
                    f"URDF joint '{joint_name}' lower position limit must be finite and less than upper."
                )
            lower_limits.append(lower)
            upper_limits.append(upper)
        return lower_limits, upper_limits

    def _validate_urdf_joint_mapping(
        self,
        arm_joint_names: Sequence[str],
        end_effector_joint_names: Sequence[str] = (),
    ) -> None:
        if self._urdf_model_path is None:
            raise RuntimeError("URDF model path is not available for joint mapping validation.")

        arm_names = self._validate_joint_names(arm_joint_names)
        end_effector_names = tuple(end_effector_joint_names)
        configured_names = self._validate_joint_names(arm_names + end_effector_names)
        root = ET.parse(self._urdf_model_path).getroot()
        joints = list(root.findall("joint"))
        joints_by_name = {joint.attrib.get("name"): joint for joint in joints}

        for joint_name in configured_names:
            joint = joints_by_name.get(joint_name)
            if joint is None:
                raise ValueError(f"URDF joint '{joint_name}' is missing.")
            joint_type = joint.attrib.get("type")
            if joint_type not in ("revolute", "continuous"):
                raise ValueError(
                    f"configured joint '{joint_name}' must be a one-DOF angular URDF joint; "
                    f"got type '{joint_type}'"
                )

        model_order = self._get_urdf_movable_joint_order(root, joints)
        configured_name_set = set(configured_names)
        expected_order = tuple(name for name in model_order if name in configured_name_set)
        if expected_order != configured_names:
            raise ValueError(
                "configured joint names must follow URDF kinematic order; "
                f"expected {expected_order}, got {configured_names}"
            )

    @staticmethod
    def _get_urdf_movable_joint_order(
        root: ET.Element,
        joints: Sequence[ET.Element],
    ) -> tuple[str, ...]:
        link_names = {
            link.attrib["name"]
            for link in root.findall("link")
            if link.attrib.get("name")
        }
        children: set[str] = set()
        joints_by_parent: dict[str, list[tuple[ET.Element, str]]] = {}
        for joint in joints:
            joint_name = joint.attrib.get("name", "<unnamed>")
            parent = joint.find("parent")
            child = joint.find("child")
            if (
                parent is None
                or child is None
                or not parent.attrib.get("link")
                or not child.attrib.get("link")
            ):
                raise ValueError(
                    f"URDF joint '{joint_name}' must define parent and child links."
                )
            parent_link = parent.attrib["link"]
            child_link = child.attrib["link"]
            joints_by_parent.setdefault(parent_link, []).append((joint, child_link))
            children.add(child_link)

        root_links = link_names - children
        if len(root_links) != 1:
            raise ValueError(f"URDF must contain exactly one root link; got {sorted(root_links)}")

        ordered_names: list[str] = []
        visited_joints: set[str] = set()
        active_links: set[str] = set()

        def visit(link_name: str) -> None:
            if link_name in active_links:
                raise ValueError(f"URDF kinematic tree contains a cycle at link '{link_name}'")
            active_links.add(link_name)
            for joint, child_link in joints_by_parent.get(link_name, ()):
                joint_name = joint.attrib.get("name")
                if not joint_name or joint_name in visited_joints:
                    raise ValueError("URDF joint names must be non-empty and unique")
                visited_joints.add(joint_name)
                if joint.attrib.get("type") != "fixed":
                    ordered_names.append(joint_name)
                visit(child_link)
            active_links.remove(link_name)

        visit(next(iter(root_links)))
        if len(visited_joints) != len(joints):
            raise ValueError("URDF contains joints disconnected from its root link")
        return tuple(ordered_names)

    @staticmethod
    def _validate_joint_names(joint_names: Sequence[str]) -> tuple[str, ...]:
        requested_names = tuple(joint_names)
        if not requested_names:
            raise ValueError("at least one joint name is required")
        if any(not isinstance(name, str) or not name.strip() for name in requested_names):
            raise ValueError("joint names must be non-empty strings")
        if len(set(requested_names)) != len(requested_names):
            raise ValueError("joint names must be unique")
        return requested_names

    def _get_pin_model_for_joint_names(self, joint_names: Sequence[str]):
        import pinocchio as pin

        requested_names = self._validate_joint_names(joint_names)
        pin_model = self.get_pin_model()
        model_joint_names = tuple(str(name) for name in pin_model.names[1:])
        missing_names = tuple(name for name in requested_names if name not in model_joint_names)
        if missing_names:
            raise ValueError(f"Pinocchio model is missing configured arm joints: {missing_names}")

        requested_name_set = set(requested_names)
        ordered_requested_names = tuple(
            name for name in model_joint_names if name in requested_name_set
        )
        if ordered_requested_names != requested_names:
            raise ValueError(
                "configured arm joint_names must follow Pinocchio model order; "
                f"expected {ordered_requested_names}, got {requested_names}"
            )

        for joint_name in requested_names:
            joint_id = pin_model.getJointId(joint_name)
            joint = pin_model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(
                    f"configured arm joint '{joint_name}' must have exactly one position and velocity DOF"
                )

        fixed_joint_ids = [
            pin_model.getJointId(name)
            for name in model_joint_names
            if name not in requested_name_set
        ]
        if not fixed_joint_ids:
            return pin_model
        return pin.buildReducedModel(
            pin_model,
            fixed_joint_ids,
            pin.neutral(pin_model),
        )
