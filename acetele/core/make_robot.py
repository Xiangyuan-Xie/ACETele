from __future__ import annotations

import importlib
import time
from typing import Optional

import numpy as np

from acetele.config.config_loader import ConfigLoader
from acetele.config.robot_config import RobotConfig
from acetele.robot.base_robot import BaseRobot

_ROBOT_ENTRYPOINTS = {
    ("ace_leader", "standalone"): (
        "acetele.robot.ace_leader.ace_leader",
        "AceLeaderRobot",
    ),
    ("ace_leader", "ros2"): (
        "acetele.robot.ace_leader.ace_leader_ros2",
        "AceLeaderROS2Robot",
    ),
    ("ace_follower", "standalone"): (
        "acetele.robot.ace_follower.ace_follower",
        "AceFollowerRobot",
    ),
    ("ace_follower", "ros2"): (
        "acetele.robot.ace_follower.ace_follower_ros2",
        "AceFollowerROS2Robot",
    ),
    ("ace_follower_dual", "standalone"): (
        "acetele.robot.ace_follower.ace_follower",
        "AceDualFollowerRobot",
    ),
}


def make_robot(config_loader: Optional[ConfigLoader | RobotConfig] = None) -> BaseRobot:
    source = ConfigLoader() if config_loader is None else config_loader
    robot_config = source if isinstance(source, RobotConfig) else source.get_robot_config()
    try:
        module_name, class_name = _ROBOT_ENTRYPOINTS[(robot_config.robot_type, robot_config.runtime)]
    except KeyError as exc:
        raise ValueError(
            f"runtime '{robot_config.runtime}' is not supported for robot type "
            f"'{robot_config.robot_type}'"
        ) from exc
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(source)


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
