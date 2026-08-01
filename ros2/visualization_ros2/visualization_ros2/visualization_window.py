"""ROS 2 adapter for the shared ACETele operator window."""

from __future__ import annotations

import sys
import threading

import rclpy
from ace_operator_ui import JointView, OperatorSnapshot
from ace_operator_ui.window import OperatorWindow
from PySide6.QtWidgets import QApplication
from rclpy.executors import MultiThreadedExecutor
from visualization_ros2.visualization_node import VisualizationNode


class RosOperatorSource:
    """Convert ROS-owned snapshots into transport-neutral UI values."""

    def __init__(self, node: VisualizationNode) -> None:
        self.node = node

    def snapshot(self) -> OperatorSnapshot:
        front_color, front_depth, wrist_color, wrist_depth = self.node.get_latest_images()
        images = {
            name: image
            for name, image in (
                ("front_color", front_color),
                ("front_depth", front_depth),
                ("wrist_color", wrist_color),
                ("wrist_depth", wrist_depth),
            )
            if image is not None
        }
        arm = self.node.get_latest_arm_state()
        stamp = arm.header.stamp
        front_metadata, wrist_metadata = self.node.get_latest_metadata()
        return OperatorSnapshot(
            images=images,
            metadata={"front": front_metadata, "wrist": wrist_metadata},
            health=self.node.get_status_info(),
            joints=JointView(
                stamp.sec * 1_000_000_000 + stamp.nanosec,
                tuple(arm.name),
                tuple(arm.position),
                tuple(arm.velocity),
                tuple(arm.effort),
            ),
            recording_state="unavailable",
        )

    def set_recording(self, active: bool) -> None:
        raise RuntimeError("recording control is not configured for this ROS 2 monitor")


def main(args=None) -> int:
    """Run ROS callbacks separately while Qt owns the main thread."""

    rclpy.init(args=args)
    node = VisualizationNode()
    app = QApplication(sys.argv)
    window = OperatorWindow(RosOperatorSource(node))
    window.show()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=False)
    ros_thread.start()
    try:
        return app.exec()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        if ros_thread.is_alive():
            ros_thread.join()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RosOperatorSource", "main"]
