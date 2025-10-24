from abc import ABC, abstractmethod
from typing import Dict, Sequence

from neutele.config.config_loader import ConfigLoader
from neutele.equipment.base_equipment import BaseEquipment


class BaseStation(ABC):
    def __init__(self, config_loader: ConfigLoader):
        self._config_loader = config_loader
        self._equipments: Dict[str, BaseEquipment] = {}

    @abstractmethod
    def act(self):
        raise NotImplementedError(
            f"Class '{self.__class__.__name__}' must implement abstract method '{self.act.__name__}()'."
        )

    def close(self):
        for equipment in self._equipments.values():
            equipment.close()

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        raise RuntimeError(f"Class '{self.__class__.__name__}' not support method '{self.act.__name__}()'.")
