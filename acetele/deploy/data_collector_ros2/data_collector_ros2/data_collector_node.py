import os
import re
import signal
import subprocess

import rclpy
from px4_msgs.msg import ManualControlSetpoint
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__("data_collector_node")
        self.declare_parameter("save_path", "~/data/")
        topics_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        self.declare_parameter("topics", None, topics_descriptor)
        self._topics = self.get_parameter("topics").value or []
        self._save_path = self.get_parameter("save_path").value

        # expand '~' and ensure absolute
        self._save_path = os.path.expanduser(self._save_path)
        if not os.path.isabs(self._save_path):
            self._save_path = os.path.abspath(self._save_path)
        os.makedirs(self._save_path, exist_ok=True)

        self._recording_process = None
        self._is_recording = False

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._subscription = self.create_subscription(
            ManualControlSetpoint,
            "/fmu/out/manual_control_setpoint",
            self._manual_control_callback,
            qos,
        )

        self.get_logger().info("DataCollector node started.")
        self.get_logger().info("Listening to: /fmu/out/manual_control_setpoint")
        self.get_logger().info(f"Saving bags to: {self._save_path}")

        self._combined_regex = self._generate_combined_regex(self._topics)
        self.get_logger().info(f"Combined Regex: {self._combined_regex}")

    def _generate_combined_regex(self, topics_list):
        if not topics_list:
            return ""

        patterns = []
        for item in topics_list:
            if any(char in item for char in "[]*+?|{}()^$"):
                patterns.append(item)
            else:
                patterns.append(f"^{re.escape(item)}$")

        return "|".join(patterns)

    def _manual_control_callback(self, msg: ManualControlSetpoint):
        if msg.aux1 >= 0.9 and not self._is_recording:
            self._start_recording()
        if msg.aux1 <= -0.9 and self._is_recording:
            self._stop_recording()

    def _start_recording(self):
        self.get_logger().info("Triggered: Starting recording...")

        cmd = [
            "ros2",
            "bag",
            "record",
            "--compression-mode",
            "file",
            "--compression-format",
            "zstd",
        ]

        if self._combined_regex:
            cmd.append("--regex")
            cmd.append(self._combined_regex)
        else:
            self.get_logger().warn("No topics configured. Recording nothing.")
            return

        self._recording_process = subprocess.Popen(cmd, cwd=self._save_path, preexec_fn=os.setsid)
        self._is_recording = True
        self.get_logger().info(f"Recording started. PID: {self._recording_process.pid}")

    def _stop_recording(self):
        self.get_logger().info("Triggered: Stopping recording...")
        if self._recording_process:
            try:
                os.killpg(os.getpgid(self._recording_process.pid), signal.SIGINT)
                self._recording_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.get_logger().warn("Recording process did not stop gracefully, killing it.")
                os.killpg(os.getpgid(self._recording_process.pid), signal.SIGKILL)

            self._recording_process = None
            self._is_recording = False
            self.get_logger().info("Recording stopped.")

    def close(self):
        if self._is_recording:
            self._stop_recording()


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()
