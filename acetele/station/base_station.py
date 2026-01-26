from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Sequence

from acetele.config.config_loader import ConfigLoader


@dataclass
class BaseEquipmentLibrary:
    pass


class BaseStation(ABC):
    def __init__(self, config_loader: ConfigLoader):
        self._config_loader = config_loader
        self._equipments: BaseEquipmentLibrary = BaseEquipmentLibrary()

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

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        raise RuntimeError(f"Class '{self.__class__.__name__}' not support method '{self.act.__name__}()'.")
