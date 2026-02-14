import os
import threading
import time
from typing import Dict, Optional, Union

import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa: E402

from acetele.equipment.base_equipment import BaseEquipment  # noqa: E402


class JoystickDriver(BaseEquipment):
    """
    Unified Joystick Driver using threading and Pygame.
    Can be configured for specific devices via name matching and axis/button mapping.
    """

    def __init__(
        self,
        device_name_pattern: Optional[str] = None,
        axis_map: Optional[Dict[int, str]] = None,
        button_map: Optional[Dict[int, str]] = None,
        hat_map: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the Joystick Driver.

        Args:
            device_name_pattern: Substring to match the device name (e.g. "Xbox", "JDK").
                                 If None, connects to the first available joystick.
            axis_map: Dictionary mapping axis index to name (e.g. {0: "Roll", 1: "Pitch"}).
            button_map: Dictionary mapping button index to name (e.g. {0: "A", 1: "B"}).
            hat_map: Dictionary mapping hat index to name.
        """
        super().__init__()
        self.device_name_pattern = device_name_pattern
        self.axis_map = axis_map or {}
        self.button_map = button_map or {}
        self.hat_map = hat_map or {}

        # Thread control
        self._stop_event = threading.Event()
        self._data_lock = threading.Lock()

        # Shared data
        self._latest_data: Optional[Dict] = None
        self._connected_device_name: Optional[str] = None
        self._is_connected = False

        # Start worker thread
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        pygame.init()

        pygame.joystick.init()
        joystick = self._connect_joystick()

        while not self._stop_event.is_set():
            if not joystick:
                pygame.joystick.quit()
                pygame.joystick.init()
                joystick = self._connect_joystick()
                if not joystick:
                    time.sleep(1.0)
                    continue
                else:
                    print(f"JoystickDriver connected to: {joystick.get_name()}")

            pygame.event.pump()

            if not pygame.joystick.get_init():
                joystick = None
                continue

            if not joystick.get_init():
                joystick.init()

            num_axes = joystick.get_numaxes()
            num_buttons = joystick.get_numbuttons()
            num_hats = joystick.get_numhats()

            raw_axes = np.array([joystick.get_axis(i) for i in range(num_axes)])
            raw_buttons = np.array([joystick.get_button(i) for i in range(num_buttons)])
            raw_hats = np.array([joystick.get_hat(i) for i in range(num_hats)])

            mapped_data = {
                "timestamp": time.time(),
                "connected": True,
                "name": self._connected_device_name,
                "raw": {"axes": raw_axes, "buttons": raw_buttons, "hats": raw_hats},
                "mapped": {},
            }

            if self.axis_map:
                for i, val in enumerate(raw_axes):
                    if i in self.axis_map:
                        name = self.axis_map[i]
                        mapped_data["mapped"][name] = val

            if self.button_map:
                for i, val in enumerate(raw_buttons):
                    if i in self.button_map:
                        name = self.button_map[i]
                        mapped_data["mapped"][name] = val

            if self.hat_map:
                for i, val in enumerate(raw_hats):
                    if i in self.hat_map:
                        name = self.hat_map[i]
                        mapped_data["mapped"][name] = val

            with self._data_lock:
                self._latest_data = mapped_data
                self._is_connected = True

            time.sleep(0.01)

        pygame.quit()

    def _connect_joystick(self):
        if pygame.joystick.get_count() == 0:
            return None

        target_joystick = None

        # Iterate over all joysticks
        for i in range(pygame.joystick.get_count()):
            try:
                js = pygame.joystick.Joystick(i)
                js.init()
                name = js.get_name()

                if self.device_name_pattern:
                    if self.device_name_pattern.lower() in name.lower():
                        target_joystick = js
                        break
                else:
                    target_joystick = js
                    break
            except pygame.error:
                continue

        if target_joystick:
            self._connected_device_name = target_joystick.get_name()
            return target_joystick

        return None

    def act(self) -> Union[Dict, None]:
        with self._data_lock:
            return self._latest_data

    def is_connected(self) -> bool:
        with self._data_lock:
            return self._is_connected

    def close(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        pygame.joystick.quit()
        pygame.quit()


# --------------------------------------------------------------------------------
# Specific Implementations
# --------------------------------------------------------------------------------


class XboxDriver(JoystickDriver):
    """
    Pre-configured driver for Xbox Controllers.
    """

    # Mapping based on standard Xbox controller layout
    AXIS_MAP = {
        0: "LeftStickX",
        1: "LeftStickY",
        2: "RightStickX",
        3: "RightStickY",
        4: "LeftTrigger",
        5: "RightTrigger",
    }

    BUTTON_MAP = {
        0: "A",
        1: "B",
        2: "X",
        3: "Y",
        4: "LeftBumper",
        5: "RightBumper",
        6: "Back",
        7: "Start",
        8: "Guide",
        9: "LeftStickButton",
        10: "RightStickButton",
    }

    def __init__(self, device_name_pattern="Xbox"):
        super().__init__(device_name_pattern=device_name_pattern, axis_map=self.AXIS_MAP, button_map=self.BUTTON_MAP)


class JDKFPVDriver(JoystickDriver):
    """
    Pre-configured driver for JDKFPV remote controller.
    """

    # Common FPV Dongle Mapping (Mode 2 default)
    AXIS_MAP = {
        0: "Yaw",  # Left Stick X
        1: "Throttle",  # Left Stick Y
        2: "Roll",  # Right Stick X
        3: "Pitch",  # Right Stick Y
        4: "Aux1",  # 3-pos Switch 1
        5: "Aux2",  # Button 1
        6: "Aux3",  # Button 2
        7: "Aux4",  # 3-pos Switch 2
    }

    def __init__(self, device_name_pattern="FPV"):
        super().__init__(device_name_pattern=device_name_pattern, axis_map=self.AXIS_MAP)


if __name__ == "__main__":
    driver = JDKFPVDriver()
    try:
        while True:
            data = driver.act()
            if data and data["connected"]:
                name = data["name"]
                print(f"\r[{name}] {' '.join([f'{k}:{v:.2f}' for k, v in data['mapped'].items()])}")
            else:
                print("\rWaiting for connection...")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        driver.close()
