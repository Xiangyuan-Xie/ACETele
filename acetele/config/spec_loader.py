"""Strict TOML adapter for the typed robot specification.

Parsing reports precise TOML paths; frozen spec classes then enforce cross-table
topology invariants. Neither phase imports ROS nor opens hardware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import tomli

from acetele.config.specs import (
    ArmSpec,
    BusSpec,
    BusType,
    ControlSpec,
    DexterousHandSpec,
    DirectionControl,
    JointSpec,
    ParallelGripperSpec,
    RobotSpec,
)
from acetele.core import Backend


class RobotSpecLoader:
    """Load the hardware-oriented schema without opening devices or importing ROS."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def load(self) -> RobotSpec:
        """Parse and validate the configured robot without creating hardware."""

        with self.path.open("rb") as stream:
            document = tomli.load(stream)
        # Reject removed layouts before generic unknown-field handling so users receive
        # an actionable migration path instead of a vague schema error.
        if any(name in document for name in ("linker", "gripper", "dexterous_hand")):
            raise ValueError(
                "legacy device tables are unsupported; migrate to [buses.<name>] and "
                "[arms.<name>]"
            )
        _reject_unknown(document, {"basic", "buses", "arms"}, "configuration root")
        basic = _table(document, "basic")
        _reject_unknown(basic, {"model", "backend", "urdf_path"}, "basic")
        model = _string(basic, "model", "basic.model")
        backend = _enum(Backend, basic.get("backend", "physical"), "basic.backend")
        raw_urdf = basic.get("urdf_path")
        urdf_path = None
        if raw_urdf is not None:
            # Relative model paths belong to the configuration file, not the caller's
            # current working directory.
            configured = Path(_string_value(raw_urdf, "basic.urdf_path")).expanduser()
            urdf_path = str(
                configured.resolve()
                if configured.is_absolute()
                else (self.path.parent / configured).resolve()
            )

        buses_table = _table(document, "buses")
        buses = tuple(
            self._parse_bus(name, value)
            for name, value in buses_table.items()
        )
        arms_table = _table(document, "arms")
        arms = tuple(
            self._parse_arm(name, value)
            for name, value in arms_table.items()
        )
        return RobotSpec(model, buses, arms, backend=backend, urdf_path=urdf_path)

    @staticmethod
    def _parse_bus(name: str, raw: Any) -> BusSpec:
        """Parse one bus table while retaining its full TOML path in errors."""

        path = f"buses.{name}"
        values = _mapping(raw, path)
        _reject_unknown(
            values,
            {
                "type",
                "port",
                "baudrate",
                "cycle_hz",
                "direction_control",
                "physical_layer",
                "family",
                "max_utilization",
                "external_estop",
                "allow_unverified_identity",
            },
            path,
        )
        bus_type = _enum(BusType, values.get("type"), f"{path}.type")
        return BusSpec(
            name=_string_value(name, "bus name"),
            type=bus_type,
            port=_string(values, "port", f"{path}.port"),
            baudrate=_integer(values, "baudrate", f"{path}.baudrate"),
            cycle_hz=_real(values, "cycle_hz", f"{path}.cycle_hz"),
            direction_control=_enum(
                DirectionControl,
                values.get("direction_control", "auto"),
                f"{path}.direction_control",
            ),
            physical_layer=_optional_string(values.get("physical_layer"), f"{path}.physical_layer"),
            family=_optional_string(values.get("family"), f"{path}.family"),
            max_utilization=_real_value(
                values.get("max_utilization", 0.70),
                f"{path}.max_utilization",
            ),
            external_estop=_boolean_value(
                values.get("external_estop", False),
                f"{path}.external_estop",
            ),
            allow_unverified_identity=_boolean_value(
                values.get("allow_unverified_identity", False),
                f"{path}.allow_unverified_identity",
            ),
        )

    def _parse_arm(self, name: str, raw: Any) -> ArmSpec:
        """Parse one ordered arm, its control options, and optional end effector."""

        path = f"arms.{name}"
        values = _mapping(raw, path)
        if "joint_ids" in values:
            raise ValueError(
                f"legacy field '{path}.joint_ids' is unsupported; use "
                f"[[{path}.joints]] with explicit name and servo_id"
            )
        _reject_unknown(
            values,
            {"bus", "joints", "control", "end_effector"},
            path,
        )
        raw_joints = values.get("joints")
        if not isinstance(raw_joints, list) or not raw_joints:
            raise ValueError(f"'{path}.joints' must be a non-empty array of tables")
        # Array-of-table order is preserved as the public URDF/ROS joint order.
        joints = tuple(
            self._parse_joint(value, f"{path}.joints[{index}]")
            for index, value in enumerate(raw_joints)
        )
        control_values = values.get("control", {})
        control_table = _mapping(control_values, f"{path}.control")
        _reject_unknown(
            control_table,
            {
                "adaptive_position",
                "gravity_position",
                "gravity_compliance_rad_per_nm",
            },
            f"{path}.control",
        )
        control = ControlSpec(
            adaptive_position=_boolean_value(
                control_table.get("adaptive_position", False),
                f"{path}.control.adaptive_position",
            ),
            gravity_position=_boolean_value(
                control_table.get("gravity_position", False),
                f"{path}.control.gravity_position",
            ),
            gravity_compliance_rad_per_nm=_optional_real_sequence(
                control_table.get("gravity_compliance_rad_per_nm"),
                f"{path}.control.gravity_compliance_rad_per_nm",
            ),
        )
        end_effector = self._parse_end_effector(values.get("end_effector"), path)
        return ArmSpec(
            _string_value(name, "arm name"),
            _string(values, "bus", f"{path}.bus"),
            joints,
            control,
            end_effector,
        )

    def _parse_end_effector(self, raw: Any, arm_path: str):
        """Dispatch an end-effector table to its exact typed specification."""

        if raw is None:
            return None
        path = f"{arm_path}.end_effector"
        values = _mapping(raw, path)
        kind = _string(values, "kind", f"{path}.kind")
        if kind == "parallel_gripper":
            _reject_unknown(
                values,
                {"kind", "bus", "joint", "travel_range_rad"},
                path,
            )
            return ParallelGripperSpec(
                bus=_string(values, "bus", f"{path}.bus"),
                joint=self._parse_joint(values.get("joint"), f"{path}.joint"),
                travel_range_rad=_real(
                    values,
                    "travel_range_rad",
                    f"{path}.travel_range_rad",
                ),
            )
        if kind == "dexterous_hand":
            _reject_unknown(
                values,
                {"kind", "bus", "vendor", "model", "side", "slave_id"},
                path,
            )
            return DexterousHandSpec(
                bus=_string(values, "bus", f"{path}.bus"),
                vendor=_string(values, "vendor", f"{path}.vendor"),
                model=_string(values, "model", f"{path}.model"),
                side=_string(values, "side", f"{path}.side"),
                slave_id=_integer(values, "slave_id", f"{path}.slave_id"),
            )
        raise ValueError(
            f"configuration field '{path}.kind' must be 'parallel_gripper' or "
            "'dexterous_hand'"
        )

    @staticmethod
    def _parse_joint(raw: Any, path: str) -> JointSpec:
        """Parse one scalar servo binding without coercing IDs or names."""

        values = _mapping(raw, path)
        _reject_unknown(
            values,
            {
                "name",
                "servo_id",
                "servo_model",
                "direction",
                "home_position_rad",
                "expected_model_number",
                "firmware_version",
            },
            path,
        )
        expected_model_number = values.get("expected_model_number")
        firmware_version = values.get("firmware_version")
        return JointSpec(
            name=_string(values, "name", f"{path}.name"),
            servo_id=_integer(values, "servo_id", f"{path}.servo_id"),
            servo_model=_string(values, "servo_model", f"{path}.servo_model"),
            direction=_integer(values, "direction", f"{path}.direction"),
            home_position_rad=_real(
                values,
                "home_position_rad",
                f"{path}.home_position_rad",
            ),
            expected_model_number=(
                None
                if expected_model_number is None
                else _integer_value(expected_model_number, f"{path}.expected_model_number")
            ),
            firmware_version=(
                None
                if firmware_version is None
                else _integer_value(firmware_version, f"{path}.firmware_version")
            ),
        )


def load_robot_spec(path: str | Path) -> RobotSpec:
    """Load a validated :class:`RobotSpec` from ``path``."""

    return RobotSpecLoader(path).load()


def _table(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(document.get(name), name)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"configuration field '{path}' must be a table")
    return value


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"configuration table '{path}' contains unknown fields: {unknown}"
        )


def _string(values: Mapping[str, Any], key: str, path: str) -> str:
    if key not in values:
        raise ValueError(f"missing required configuration field '{path}'")
    return _string_value(values[key], path)


def _string_value(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"configuration field '{path}' must be a non-empty string")
    return value


def _optional_string(value: Any, path: str) -> Optional[str]:
    return None if value is None else _string_value(value, path)


def _integer(values: Mapping[str, Any], key: str, path: str) -> int:
    if key not in values:
        raise ValueError(f"missing required configuration field '{path}'")
    return _integer_value(values[key], path)


def _integer_value(value: Any, path: str) -> int:
    if type(value) is not int:
        raise ValueError(f"configuration field '{path}' must be an integer")
    return value


def _real(values: Mapping[str, Any], key: str, path: str) -> float:
    if key not in values:
        raise ValueError(f"missing required configuration field '{path}'")
    return _real_value(values[key], path)


def _real_value(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"configuration field '{path}' must be a real number")
    return float(value)


def _boolean_value(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"configuration field '{path}' must be a boolean")
    return value


def _optional_real_sequence(value: Any, path: str) -> Optional[tuple[float, ...]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ValueError(f"configuration field '{path}' must be an array of numbers")
    return tuple(
        _real_value(item, f"{path}[{index}]") for index, item in enumerate(value)
    )


def _enum(enum_type, value: Any, path: str):
    """Parse a string-backed enum and report all accepted wire values."""

    if not isinstance(value, str):
        raise ValueError(f"configuration field '{path}' must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(
            f"configuration field '{path}' must be one of: {choices}"
        ) from exc


__all__ = ["RobotSpecLoader", "load_robot_spec"]
