import time
from threading import Event, Lock, Thread
from typing import Dict, Tuple, Union

import pygame
from equipment.base_equipment import BaseEquipment

AXIS_MAP = {
    0: "MainX",
    1: "MainY",
    2: "SubX",
    3: "SubY",
    4: "Left",
    5: "Right",
}

BUTTON_MAP = {
    0: "A",
    1: "B",
    2: "X",
    3: "Y",
    4: "Left",
    5: "Right",
    6: "View",
    7: "Menu",
    10: "Home",
    11: "Share",
}


class Xbox360Driver(BaseEquipment):
    def __init__(self):
        super().__init__()
        self._channel = {
            "Button": {
                "A": False,
                "B": False,
                "X": False,
                "Y": False,
                "Left": False,
                "Right": False,
                "View": False,
                "Menu": False,
                "Home": False,
                "Share": False,
            },
            "Axis": {
                "MainX": 0.0,
                "MainY": 0.0,
                "SubX": 0.0,
                "SubY": 0.0,
                "Left": 0.0,
                "Right": 0.0,
            },
            "Direction": (0, 0),
        }

        self._lock = Lock()
        self._stop_flag = Event()
        self._comm_thread = Thread(target=self._comm_worker, daemon=True)
        self._comm_thread.start()

    def _comm_worker(self):
        pygame.init()
        pygame.joystick.init()
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        while not self._stop_flag.is_set():
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    with self._lock:
                        self._channel["Button"][BUTTON_MAP[event.button]] = True
                elif event.type == pygame.JOYBUTTONUP:
                    with self._lock:
                        self._channel["Button"][BUTTON_MAP[event.button]] = False
                elif event.type == pygame.JOYAXISMOTION:
                    with self._lock:
                        self._channel["Axis"][AXIS_MAP[event.axis]] = event.value
                elif event.type == pygame.JOYHATMOTION:
                    with self._lock:
                        self._channel["Axis"] = event.value

    def act(self) -> Dict[str, Union[Dict[str, bool], Dict[str, float], Tuple[float, float]]]:
        with self._lock:
            return self._channel

    def close(self):
        self._stop_flag.set()
        self._comm_thread.join()


if __name__ == "__main__":
    driver = Xbox360Driver()
    while True:
        print(driver.act())
        time.sleep(0.05)
