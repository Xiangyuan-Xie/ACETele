"""Expose ROS package sources without installing a colcon workspace for unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "ros2" / "ace_robot_ros2"))
