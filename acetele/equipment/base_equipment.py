from abc import ABC, abstractmethod


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
