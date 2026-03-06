import importlib
import time
from typing import Optional

import numpy as np

from acetele.config.config_loader import ConfigLoader
from acetele.robot.base_robot import BaseRobot


def make_robot(config_loader: Optional[ConfigLoader] = None) -> BaseRobot:
    if not config_loader:
        config_loader = ConfigLoader()

    module_name, class_name = config_loader.get_robot_info()
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(config_loader)


if __name__ == "__main__":
    robot = make_robot()
    try:
        with np.printoptions(suppress=True):
            while True:
                print(robot.act())
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()
