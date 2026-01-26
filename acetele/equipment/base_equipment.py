from abc import ABC, abstractmethod
from typing import Sequence


class BaseEquipment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def act(self):
        raise NotImplementedError(
            f"Class '{self.__class__.__name__}' must implement abstract method '{self.act.__name__}()'."
        )

    def close(self):
        pass

    def apply_torque_feedback(self, external_torque: Sequence[float]):
        raise RuntimeError(f"Class '{self.__class__.__name__}' not support method '{self.act.__name__}()'.")
