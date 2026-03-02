import os
import re
import signal
import subprocess
import time
from pathlib import Path
from shutil import which

import rclpy
from px4_msgs.msg import ManualControlSetpoint
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


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
        self._save_path = os.path.abspath(self._save_path)
        self._save_path_obj = Path(self._save_path)
        self._ensure_save_path()

        self._recording_process = None
        self._recording_log_file = None
        self._is_recording = False
        self._last_transition_time = 0.0
        self._transition_guard_sec = 0.8
        self._process_watchdog = self.create_timer(1.0, self._watch_recording_process)

        self._subscription = self.create_subscription(
            ManualControlSetpoint,
            "/fmu/out/manual_control_setpoint",
            self._manual_control_callback,
            qos_profile_sensor_data,
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
        if not msg.valid:
            return
        now = time.monotonic()
        if now - self._last_transition_time < self._transition_guard_sec:
            return
        if msg.aux1 >= 0.9 and not self._is_recording:
            self._last_transition_time = now
            self._start_recording()
        if msg.aux1 <= -0.9 and self._is_recording:
            self._last_transition_time = now
            self._stop_recording()

    def _start_recording(self):
        self.get_logger().info("Triggered: Starting recording...")
        if which("ros2") is None:
            self.get_logger().error("ros2 CLI not found in PATH. Recording aborted.")
            return

        if not self._ensure_save_path():
            self.get_logger().error(f"Save path not writable: {self._save_path}")
            return

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

        log_path = self._save_path_obj / f"rosbag_record_{int(time.time())}.log"
        self._recording_log_file = open(log_path, "w")
        self._recording_process = subprocess.Popen(
            cmd,
            cwd=self._save_path,
            preexec_fn=os.setsid,
            stdout=self._recording_log_file,
            stderr=subprocess.STDOUT,
        )
        self._is_recording = True
        self.get_logger().info(f"Recording started. PID: {self._recording_process.pid}")
        self.get_logger().info(f"Recording log: {log_path}")

    def _stop_recording(self):
        self.get_logger().info("Triggered: Stopping recording...")
        if self._recording_process:
            try:
                os.killpg(os.getpgid(self._recording_process.pid), signal.SIGINT)
                self._recording_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.get_logger().warn("Recording process did not stop gracefully, killing it.")
                os.killpg(os.getpgid(self._recording_process.pid), signal.SIGKILL)

            self._recording_process = None
            self._is_recording = False
            if self._recording_log_file:
                self._recording_log_file.close()
                self._recording_log_file = None
            self.get_logger().info("Recording stopped.")

    def close(self):
        if self._is_recording:
            self._stop_recording()
        if self._recording_log_file:
            self._recording_log_file.close()
            self._recording_log_file = None

    def _ensure_save_path(self):
        try:
            self._save_path_obj.mkdir(parents=True, exist_ok=True)
            test_file = self._save_path_obj / ".write_test"
            with open(test_file, "w") as f:
                f.write("ok")
            test_file.unlink(missing_ok=True)
            return True
        except Exception as exc:
            self.get_logger().error(f"Failed to prepare save_path: {exc}")
            return False

    def _watch_recording_process(self):
        if not self._is_recording or not self._recording_process:
            return
        ret = self._recording_process.poll()
        if ret is None:
            return
        self._is_recording = False
        self._recording_process = None
        if self._recording_log_file:
            self._recording_log_file.close()
            self._recording_log_file = None
        self.get_logger().error(f"Recording process exited unexpectedly with code {ret}")


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
