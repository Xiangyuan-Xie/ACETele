import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from acetele.config.config_loader import ConfigLoader


@dataclass
class BaseEquipmentLibrary:
    pass


class BaseStation(ABC):
    def __init__(self, config_loader: ConfigLoader):
        self._config_loader = config_loader
        self._equipments: BaseEquipmentLibrary = BaseEquipmentLibrary()

        self._urdf_model_path: Optional[str]
        urdf_model_path = (
            Path(__file__).resolve().parent / "description" / f"{self._config_loader.get_station_type()}.urdf"
        )
        if urdf_model_path.exists() and urdf_model_path.is_file():
            self._urdf_model_path = str(urdf_model_path)
        else:
            self._urdf_model_path = None

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
        pin_model, _, _ = pin.buildModelsFromUrdf(filename=self._urdf_model_path)
        return pin_model


def make_station(config_dir: Optional[Path] = None, config_name: Optional[str] = None) -> BaseStation:
    if config_name:
        if config_dir:
            config_loader = ConfigLoader(config_dir=config_dir, entry_config_name=config_name)
        else:
            config_loader = ConfigLoader(entry_config_name=config_name)
    else:
        if config_dir:
            raise RuntimeError("Either config_dir or config_name must be provided.")
        else:
            config_loader = ConfigLoader()

    module_name, class_name = config_loader.get_station_info()
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(config_loader)
