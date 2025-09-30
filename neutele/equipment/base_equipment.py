from abc import ABC, abstractmethod


class BaseEquipment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def act(self):
        pass

    def calibrate(self):
        pass

    def close(self):
        pass
