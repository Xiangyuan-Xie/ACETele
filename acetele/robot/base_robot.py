from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from acetele.config.config_loader import ConfigLoader


@dataclass
class BaseEquipmentLibrary:
    pass


class BaseRobot(ABC):
    def __init__(self, config_loader: ConfigLoader):
        self._config_loader = config_loader
        self._name = f"{self._config_loader.get_robot_type()}_{self._config_loader.get_backend()}"
        self._equipments: BaseEquipmentLibrary = BaseEquipmentLibrary()

        self._urdf_model_path: Optional[str]
        urdf_model_path = (
            Path(__file__).resolve().parent
            / self._config_loader.get_robot_type()
            / "description"
            / f"{self._config_loader.get_robot_type()}.urdf"
        )
        if urdf_model_path.exists() and urdf_model_path.is_file():
            self._urdf_model_path = str(urdf_model_path)
        else:
            self._urdf_model_path = None

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def act(self):
        raise NotImplementedError(
            f"Class '{self.__class__.__name__}' must implement abstract method '{self.act.__name__}()'."
        )

    def close(self):
        for f in fields(self._equipments):
            value = getattr(self._equipments, f.name)
            method = getattr(value, "close", None)
            if callable(method):
                method()

    def get_pin_model(self):
        import pinocchio as pin

        if self._urdf_model_path is None:
            raise RuntimeError("URDF model path is not available.")
        pin_model, _, _ = pin.buildModelsFromUrdf(
            filename=self._urdf_model_path, package_dirs=str(Path(self._urdf_model_path).parent)
        )
        return pin_model

    def _get_pin_model_with_fixed_joints(self, fixed_joint_names):
        import pinocchio as pin

        pin_model = self.get_pin_model()
        fixed_joint_ids = []
        for joint_name in fixed_joint_names:
            joint_id = pin_model.getJointId(joint_name)
            if joint_id < len(pin_model.joints):
                fixed_joint_ids.append(joint_id)
        if not fixed_joint_ids:
            return pin_model
        return pin.buildReducedModel(
            pin_model,
            fixed_joint_ids,
            pin.neutral(pin_model),
        )
