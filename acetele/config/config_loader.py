from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tomli

from acetele.config.robot_config import (
    ArmAssemblyConfig,
    ArmConfig,
    EndEffectorConfig,
    FeeTechGripperConfig,
    O6DexterousHandConfig,
    RobotConfig,
    _normalize_finite_real,
    _normalize_finite_real_sequence,
)
from acetele.utils.joint import (
    normalize_joint_id,
    normalize_joint_ids,
    normalize_joint_sign,
    normalize_joint_signs,
)

__all__ = ["ConfigLoader"]


class ConfigLoader:
    def __init__(
        self,
        config_path: Path = Path(__file__).parent / "default.toml",
        backend_override: Optional[str] = None,
        runtime_override: Optional[str] = None,
    ):
        config_path = Path(config_path).expanduser().resolve()

        with open(config_path, "rb") as file:
            entry_config = tomli.load(file)

        config_file = entry_config.get("basic", {}).get("config_file")
        if config_file:
            robot_config_path = config_path.parent / self._require_string(
                config_file,
                config_path="basic.config_file",
            )
            with open(robot_config_path, "rb") as file:
                self._robot_config = tomli.load(file)
        else:
            self._robot_config = entry_config

        basic = self._robot_config.setdefault("basic", {})
        if backend_override is not None:
            basic["backend"] = backend_override
        if runtime_override is not None:
            basic["runtime"] = runtime_override

        self._reject_legacy_schema()
        self._parsed_config = self._parse_robot_config()

    def _reject_legacy_schema(self) -> None:
        basic = self._robot_config.get("basic", {})
        legacy_sections = sorted({"linker", "gripper", "dexterous_hand"}.intersection(self._robot_config))
        if "variant" in basic or legacy_sections:
            details = ", ".join(legacy_sections) or "basic.variant"
            raise ValueError(
                f"legacy robot configuration detected ({details}); migrate devices under "
                "[arms.<name>] and [arms.<name>.end_effector]"
            )

    def get_robot_type(self) -> str:
        return self._parsed_config.robot_type

    def get_backend(self) -> str:
        return self._parsed_config.backend

    def get_runtime(self) -> str:
        return self._parsed_config.runtime

    def get_robot_config(self) -> RobotConfig:
        return self._parsed_config

    def _parse_robot_config(self) -> RobotConfig:
        basic = self._robot_config.get("basic", {})
        try:
            robot_type = self._require_string(
                basic["robot_type"],
                config_path="basic.robot_type",
            )
            backend = self._require_string(
                basic["backend"],
                config_path="basic.backend",
            )
            runtime = self._require_string(
                basic["runtime"],
                config_path="basic.runtime",
            )
        except KeyError as exc:
            raise ValueError(f"missing required basic configuration field '{exc.args[0]}'") from exc

        raw_arms = self._robot_config.get("arms")
        if not isinstance(raw_arms, dict) or not raw_arms:
            raise ValueError("robot configuration requires at least one [arms.<name>] table")

        assemblies = tuple(
            self._parse_arm_assembly(name, raw_config)
            for name, raw_config in raw_arms.items()
        )
        return RobotConfig(
            robot_type=robot_type,
            backend=backend,
            runtime=runtime,
            arm_assemblies=assemblies,
        )

    def _parse_arm_assembly(self, name: str, raw_config: Dict[str, Any]) -> ArmAssemblyConfig:
        self._reject_mock_parameters(raw_config, config_path=f"arms.{name}")
        joint_ids = normalize_joint_ids(raw_config["joint_ids"], field_name=f"arm '{name}' joint_ids")
        joint_names = self._string_sequence(
            self._require_explicit_joint_names(
                raw_config,
                field_name="joint_names",
                config_path=f"arms.{name}.joint_names",
            ),
            config_path=f"arms.{name}.joint_names",
        )
        arm = ArmConfig(
            port=self._optional_string(
                raw_config.get("port"),
                config_path=f"arms.{name}.port",
            ),
            joint_ids=joint_ids,
            joint_names=joint_names,
            joint_signs=normalize_joint_signs(
                raw_config["joint_signs"],
                field_name=f"arms.{name}.joint_signs",
            ),
            home_poses=_normalize_finite_real_sequence(
                raw_config["home_poses"],
                field_name=f"arms.{name}.home_poses",
            ),
            servo_models=self._string_sequence(
                raw_config.get("servo_models", ()),
                config_path=f"arms.{name}.servo_models",
            ),
            enable_gravity_compensation=self._optional_boolean(
                raw_config,
                field_name="enable_gravity_compensation",
                config_path=f"arms.{name}.enable_gravity_compensation",
            ),
            enable_adaptive_compensation=self._optional_boolean(
                raw_config,
                field_name="enable_adaptive_compensation",
                config_path=f"arms.{name}.enable_adaptive_compensation",
            ),
            control_period=_normalize_finite_real(
                raw_config.get("control_period", 0.004),
                field_name=f"arms.{name}.control_period",
            ),
        )
        end_effector = self._parse_end_effector(name, raw_config.get("end_effector"))
        return ArmAssemblyConfig(name=name, arm=arm, end_effector=end_effector)

    def _parse_end_effector(
        self,
        arm_name: str,
        raw_config: Optional[Dict[str, Any]],
    ) -> Optional[EndEffectorConfig]:
        if raw_config is None:
            return None
        config_path = f"arms.{arm_name}.end_effector"
        self._reject_mock_parameters(raw_config, config_path=config_path)
        kind = self._require_string(
            raw_config.get("kind", ""),
            config_path=f"{config_path}.kind",
        )
        if kind == "gripper":
            if "gripper_type" in raw_config:
                raise ValueError("gripper_type was removed; configure travel_range_rad instead")
            joint_id = normalize_joint_id(
                raw_config["joint_id"],
                field_name=f"gripper on arm '{arm_name}' joint_id",
            )
            return FeeTechGripperConfig(
                port=self._optional_string(
                    raw_config.get("port"),
                    config_path=f"arms.{arm_name}.end_effector.port",
                ),
                joint_id=joint_id,
                joint_name=self._require_string(
                    self._require_explicit_joint_names(
                        raw_config,
                        field_name="joint_name",
                        config_path=f"arms.{arm_name}.end_effector.joint_name",
                    ),
                    config_path=f"arms.{arm_name}.end_effector.joint_name",
                ),
                joint_sign=normalize_joint_sign(
                    raw_config["joint_sign"],
                    field_name=f"arms.{arm_name}.end_effector.joint_sign",
                ),
                home_pose=_normalize_finite_real(
                    raw_config["home_pose"],
                    field_name=f"{config_path}.home_pose",
                ),
                servo_model=self._require_string(
                    raw_config["servo_model"],
                    config_path=f"arms.{arm_name}.end_effector.servo_model",
                ),
                travel_range_rad=_normalize_finite_real(
                    raw_config["travel_range_rad"],
                    field_name=f"{config_path}.travel_range_rad",
                ),
            )
        if kind == "dexterous_hand":
            model = self._require_string(
                raw_config.get("model", ""),
                config_path=f"arms.{arm_name}.end_effector.model",
            )
            if model != "o6":
                raise ValueError(f"unsupported dexterous hand model '{model}'")
            return self._parse_o6_dexterous_hand(arm_name, raw_config)
        raise ValueError(f"unsupported end effector kind '{kind}' on arm '{arm_name}'")

    @classmethod
    def _parse_o6_dexterous_hand(
        cls,
        arm_name: str,
        raw_config: Dict[str, Any],
    ) -> O6DexterousHandConfig:
        config_path = f"arms.{arm_name}.end_effector"
        if "joint_names" in raw_config:
            raise ValueError(
                f"configuration field '{config_path}.joint_names' is model-defined for O6; "
                "remove it from TOML"
            )
        ids = normalize_joint_ids(
            raw_config["joint_ids"],
            field_name=f"O6 dexterous hand on arm '{arm_name}' joint_ids",
        )
        side = cls._require_string(
            raw_config.get("side", arm_name),
            config_path=f"{config_path}.side",
        )
        return O6DexterousHandConfig(side=side, joint_ids=ids)

    @staticmethod
    def _reject_mock_parameters(raw_config: Dict[str, Any], *, config_path: str) -> None:
        mock_fields = (
            "mock_joints",
            "wrap_public_positions",
            "initial_positions",
            "lower_limits",
            "upper_limits",
            "max_velocities",
        )
        configured = tuple(field for field in mock_fields if field in raw_config)
        if configured:
            fields = ", ".join(f"{config_path}.{field}" for field in configured)
            raise ValueError(
                f"mock-only configuration fields are not allowed in TOML: {fields}; "
                "mock state is derived from the robot model"
            )

    @staticmethod
    def _optional_string(value: Any, *, config_path: str) -> Optional[str]:
        if value is None:
            return None
        return ConfigLoader._require_string(value, config_path=config_path)

    @staticmethod
    def _require_string(value: Any, *, config_path: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"configuration field '{config_path}' must be a non-empty string"
            )
        return value

    @classmethod
    def _string_sequence(cls, values: Any, *, config_path: str) -> Tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError(
                f"configuration field '{config_path}' must be a sequence of strings"
            )
        try:
            items = tuple(values)
        except TypeError as exc:
            raise ValueError(
                f"configuration field '{config_path}' must be a sequence of strings"
            ) from exc
        return tuple(
            cls._require_string(value, config_path=f"{config_path}[{index}]")
            for index, value in enumerate(items)
        )

    @staticmethod
    def _optional_boolean(
        raw_config: Dict[str, Any],
        *,
        field_name: str,
        config_path: str,
    ) -> bool:
        if field_name not in raw_config:
            return False
        value = raw_config[field_name]
        if type(value) is not bool:
            raise ValueError(f"configuration field '{config_path}' must be a boolean")
        return value

    @staticmethod
    def _require_explicit_joint_names(
        raw_config: Dict[str, Any],
        *,
        field_name: str,
        config_path: str,
    ) -> Any:
        if field_name not in raw_config:
            raise ValueError(
                f"missing required configuration field '{config_path}'; configure kinematic "
                "joint names explicitly because joint IDs are hardware bus addresses"
            )
        return raw_config[field_name]
