import time
from multiprocessing import Event, Process, Queue
from typing import Dict, Tuple, Union

import numpy as np
import pygame

from neutele.equipment.base_equipment import BaseEquipment

# Mapping of axis indices to human-readable names
AXIS_MAP = {
    0: "MainX",  # Left stick horizontal
    1: "MainY",  # Left stick vertical
    2: "SubX",  # Right stick horizontal
    3: "SubY",  # Right stick vertical
    4: "Left",  # Left trigger
    5: "Right",  # Right trigger
}

# Mapping of button indices to human-readable names
BUTTON_MAP = {
    0: "A",  # A button
    1: "B",  # B button
    2: "X",  # X button
    3: "Y",  # Y button
    4: "Left",  # Left bumper
    5: "Right",  # Right bumper
    6: "View",  # View/Back button
    7: "Menu",  # Menu/Start button
    10: "Home",  # Xbox button
    11: "Share",  # Share button
}


def comm_worker(stop_flag, data_queue, control_queue):
    """
    Joystick communication worker running in a separate process.

    Args:
        stop_flag: Event to signal when to stop the worker
        data_queue: Queue for sending joystick data to main process
        control_queue: Queue for receiving control commands
    """
    try:
        # Initialize pygame for joystick handling
        pygame.init()
        pygame.joystick.init()

        # Check if any joystick is available
        if pygame.joystick.get_count() == 0:
            print("No joystick detected")
            data_queue.put({"error": "No joystick detected"})
            return

        # Initialize the first joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Joystick connected: {joystick.get_name()}")

        # Initialize channel state with default values
        channel_state = {
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
                "MainX": 0.0,  # Left stick X-axis
                "MainY": 0.0,  # Left stick Y-axis
                "SubX": 0.0,  # Right stick X-axis
                "SubY": 0.0,  # Right stick Y-axis
                "Left": 0.0,  # Left trigger
                "Right": 0.0,  # Right trigger
            },
            "Direction": (0, 0),  # D-pad direction
        }

        # Main worker loop
        while not stop_flag.is_set():
            # Check for control commands
            if not control_queue.empty():
                command = control_queue.get()
                if command == "stop":
                    break

            # Process all pygame events
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    # Handle button press events
                    if event.button in BUTTON_MAP:
                        channel_state["Button"][BUTTON_MAP[event.button]] = True
                elif event.type == pygame.JOYBUTTONUP:
                    # Handle button release events
                    if event.button in BUTTON_MAP:
                        channel_state["Button"][BUTTON_MAP[event.button]] = False
                elif event.type == pygame.JOYAXISMOTION:
                    # Handle axis motion events with deadzone
                    if abs(event.value) >= 0.05:
                        value = np.clip(event.value, -1.0, 1.0)
                    else:
                        value = 0.0

                    if event.axis in AXIS_MAP:
                        axis_name = AXIS_MAP[event.axis]
                        if axis_name in ["Left", "Right"]:
                            # Map trigger values from [-1, 1] to [0, 1]
                            channel_state["Axis"][axis_name] = (value + 1) / 2
                        else:
                            channel_state["Axis"][axis_name] = value
                elif event.type == pygame.JOYHATMOTION:
                    # Handle D-pad events
                    channel_state["Direction"] = event.value

            # Send updated state to main process
            data_queue.put(channel_state.copy())

            # Small sleep to reduce CPU usage
            time.sleep(0.01)

    except Exception as e:
        print(f"Joystick process error: {e}")
        data_queue.put({"error": str(e)})
    finally:
        # Clean up pygame resources
        pygame.quit()


class Xbox360Driver(BaseEquipment):
    """
    Xbox 360 controller driver using multiprocessing to avoid GIL limitations.

    This class provides a process-safe interface to read Xbox 360 controller
    inputs in real-time without being blocked by Python's GIL.
    """

    def __init__(self):
        """Initialize the Xbox 360 driver with multiprocessing."""
        super().__init__()
        # Create queues for inter-process communication
        self._data_queue = Queue()  # For joystick data
        self._control_queue = Queue()  # For control commands
        self._stop_flag = Event()  # Stop signal for worker process

        # Initialize current state with default values
        self._current_state = {
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

        # Start the communication process
        self._comm_process = Process(
            target=comm_worker, args=(self._stop_flag, self._data_queue, self._control_queue), daemon=True
        )
        self._comm_process.start()

        # Wait for process initialization
        time.sleep(1.0)

        # Check for initialization errors
        if not self._data_queue.empty():
            data = self._data_queue.get()
            if "error" in data:
                raise RuntimeError(f"Joystick initialization failed: {data['error']}")

    def act(self) -> Dict[str, Union[Dict[str, bool], Dict[str, float], Tuple[float, float]]]:
        """
        Get the current state of the Xbox 360 controller.

        Returns:
            Dictionary containing button states, axis values, and D-pad direction
        """
        # Update state with the latest data from the queue
        while not self._data_queue.empty():
            new_state = self._data_queue.get()
            if "error" not in new_state:  # Skip error messages
                self._current_state = new_state

        # Return a copy to avoid external modification
        return self._current_state.copy()

    def close(self):
        """Close the driver and clean up resources."""
        # Signal the worker process to stop
        self._stop_flag.set()
        self._control_queue.put("stop")

        # Wait for process to terminate
        if self._comm_process.is_alive():
            self._comm_process.join(timeout=2.0)
            if self._comm_process.is_alive():
                self._comm_process.terminate()


if __name__ == "__main__":
    # Test code
    driver = Xbox360Driver()
    try:
        while True:
            state = driver.act()
            print(f"\rButtons: {state['Button']} Axes: {state['Axis']} D-pad: {state['Direction']}", end="")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        driver.close()
