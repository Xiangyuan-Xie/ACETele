import importlib
from pathlib import Path
from typing import Optional

from acetele.config.config_loader import ConfigLoader
from acetele.robot.base_robot import BaseRobot


def make_robot(config_dir: Optional[Path] = None, config_name: Optional[str] = None) -> BaseRobot:
    if config_name:
        if config_dir:
            config_loader = ConfigLoader(config_dir=config_dir, entry_config_name=config_name)
        else:
            config_loader = ConfigLoader(entry_config_name=config_name)
    else:
        if config_dir:
            raise RuntimeError("Either config_dir or config_name must be provided.")
        else:
            config_loader = ConfigLoader()

    module_name, class_name = config_loader.get_robot_info()
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(config_loader)
