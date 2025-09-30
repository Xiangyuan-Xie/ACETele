from abc import ABC, abstractmethod
from typing import Dict

from neutele.config.config_loader import ConfigLoader
from neutele.equipment.base_equipment import BaseEquipment


class BaseStation(ABC):
    def __init__(self, config_loader: ConfigLoader):
        self._config_loader = config_loader
        self._equipments: Dict[str, BaseEquipment] = {}

    @abstractmethod
    def act(self):
        pass

    def calibrate(self):
        pass

    def close(self):
        for equipment in self._equipments.values():
            equipment.close()
