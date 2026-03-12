from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from realsense2_camera_msgs.msg import Metadata
from sensor_msgs.msg import CompressedImage, Image, JointState


class VisualizationNode(Node):
    def __init__(self) -> None:
        super().__init__("visualization")
        self.declare_parameter("front_color_topic", "/camera/front/color/image_raw")
        self.declare_parameter("front_depth_topic", "/camera/front/aligned_depth_to_color/image_raw")
        self.declare_parameter("wrist_color_topic", "/camera/wrist/color/image_raw")
        self.declare_parameter("wrist_depth_topic", "/camera/wrist/aligned_depth_to_color/image_raw")
        self.declare_parameter("front_color_metadata_topic", "/camera/front/color/metadata")
        self.declare_parameter("wrist_color_metadata_topic", "/camera/wrist/color/metadata")
        self.declare_parameter("arm_state_topic", "/arm/state")
        self.declare_parameter("color_transport", "compressed")
        self.declare_parameter("depth_transport", "compressedDepth")

        front_color_topic = self.get_parameter("front_color_topic").value
        front_depth_topic = self.get_parameter("front_depth_topic").value
        wrist_color_topic = self.get_parameter("wrist_color_topic").value
        wrist_depth_topic = self.get_parameter("wrist_depth_topic").value
        front_color_metadata_topic = self.get_parameter("front_color_metadata_topic").value
        wrist_color_metadata_topic = self.get_parameter("wrist_color_metadata_topic").value
        arm_state_topic = self.get_parameter("arm_state_topic").value
        color_transport = self.get_parameter("color_transport").value
        depth_transport = self.get_parameter("depth_transport").value

        effective_color_transport = color_transport
        effective_depth_transport = depth_transport
        if effective_color_transport == "compressedDepth":
            self.get_logger().warning("color_transport is compressedDepth, fallback to compressed")
            effective_color_transport = "compressed"
        if effective_depth_transport in ("compressed", "theora"):
            self.get_logger().warning(f"depth_transport is {effective_depth_transport}, fallback to compressedDepth")
            effective_depth_transport = "compressedDepth"

        self._bridge = CvBridge()
        self._data_lock = threading.Lock()
        self._latest_color: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_wrist_color: Optional[np.ndarray] = None
        self._latest_wrist_depth: Optional[np.ndarray] = None
        self._latest_arm_state = JointState()
        self._last_front_metadata_json = ""
        self._last_wrist_metadata_json = ""
        self._topic_status: Dict[str, Time] = {}
        self._topic_smoothed_latency: Dict[str, float] = {}

        self.get_logger().info(f"Using SensorDataQoS (Best Effort) for color transport: {effective_color_transport}")
        self._color_sub = self._create_image_subscription(
            front_color_topic,
            effective_color_transport,
            self._color_callback,
            is_depth=False,
        )
        self.get_logger().info(f"Using SensorDataQoS (Best Effort) for depth transport: {effective_depth_transport}")
        self._depth_sub = self._create_image_subscription(
            front_depth_topic,
            effective_depth_transport,
            self._depth_callback,
            is_depth=True,
        )
        self.get_logger().info(
            f"Using SensorDataQoS (Best Effort) for wrist color transport: {effective_color_transport}"
        )
        self._wrist_color_sub = self._create_image_subscription(
            wrist_color_topic,
            effective_color_transport,
            self._wrist_color_callback,
            is_depth=False,
        )
        self.get_logger().info(
            f"Using SensorDataQoS (Best Effort) for wrist depth transport: {effective_depth_transport}"
        )
        self._wrist_depth_sub = self._create_image_subscription(
            wrist_depth_topic,
            effective_depth_transport,
            self._wrist_depth_callback,
            is_depth=True,
        )

        self._front_metadata_sub = self.create_subscription(
            Metadata,
            front_color_metadata_topic,
            self._front_metadata_callback,
            qos_profile_sensor_data,
        )
        self._wrist_metadata_sub = self.create_subscription(
            Metadata,
            wrist_color_metadata_topic,
            self._wrist_metadata_callback,
            qos_profile_sensor_data,
        )
        self._arm_state_sub = self.create_subscription(
            JointState,
            arm_state_topic,
            self._arm_state_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info("Visualization Node Started.")
        self.get_logger().info(f"Transport - Color: {effective_color_transport}, Depth: {effective_depth_transport}")

    def _create_image_subscription(self, topic: str, transport: str, callback, is_depth: bool):
        if transport == "raw":
            return self.create_subscription(
                Image,
                topic,
                callback,
                qos_profile_sensor_data,
            )
        if transport == "compressed":
            compressed_topic = f"{topic}/compressed"
            return self.create_subscription(
                CompressedImage,
                compressed_topic,
                lambda msg: callback(msg, is_compressed=True, is_depth=is_depth),
                qos_profile_sensor_data,
            )
        if transport == "compressedDepth":
            compressed_topic = f"{topic}/compressedDepth"
            return self.create_subscription(
                CompressedImage,
                compressed_topic,
                lambda msg: callback(msg, is_compressed=True, is_depth=is_depth),
                qos_profile_sensor_data,
            )
        return self.create_subscription(
            Image,
            topic,
            callback,
            qos_profile_sensor_data,
        )

    def get_latest_images(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        with self._data_lock:
            front_color = None if self._latest_color is None else self._latest_color.copy()
            front_depth = None if self._latest_depth is None else self._latest_depth.copy()
            wrist_color = None if self._latest_wrist_color is None else self._latest_wrist_color.copy()
            wrist_depth = None if self._latest_wrist_depth is None else self._latest_wrist_depth.copy()
        return front_color, front_depth, wrist_color, wrist_depth

    def get_latest_arm_state(self) -> JointState:
        with self._data_lock:
            arm_state = JointState()
            arm_state.header = self._latest_arm_state.header
            arm_state.name = list(self._latest_arm_state.name)
            arm_state.position = list(self._latest_arm_state.position)
            arm_state.velocity = list(self._latest_arm_state.velocity)
            arm_state.effort = list(self._latest_arm_state.effort)
            return arm_state

    def get_latest_metadata(self) -> Tuple[str, str]:
        with self._data_lock:
            return self._last_front_metadata_json, self._last_wrist_metadata_json

    def get_status_info(self) -> Dict[str, str]:
        now = self.get_clock().now()
        with self._data_lock:
            info: Dict[str, str] = {}
            for key in sorted(self._topic_status.keys()):
                diff = (now - self._topic_status[key]).nanoseconds / 1e9
                if diff < 2.0:
                    status = "ONLINE"
                    if key in self._topic_smoothed_latency:
                        status = f"ONLINE ({self._topic_smoothed_latency[key]:.1f}ms)"
                    info[key] = status
                else:
                    info[key] = "OFFLINE"
            return info

    def _latency_ms(self, stamp) -> float:
        latency = (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds / 1e6
        if latency < 0.0:
            return 0.0
        return float(latency)

    def _update_status(self, key: str, latency_ms: float = 0.0) -> None:
        with self._data_lock:
            self._topic_status[key] = self.get_clock().now()
            if key not in self._topic_smoothed_latency:
                self._topic_smoothed_latency[key] = latency_ms
            else:
                alpha = 0.05
                self._topic_smoothed_latency[key] = (
                    alpha * latency_ms + (1.0 - alpha) * self._topic_smoothed_latency[key]
                )

    def _decode_color(self, msg) -> np.ndarray:
        if isinstance(msg, Image):
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _decode_depth(self, msg) -> np.ndarray:
        if isinstance(msg, Image):
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        try:
            depth_img = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if depth_img is not None and depth_img.size > 0:
                if depth_img.ndim == 3:
                    return cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
                return depth_img
        except Exception:
            pass
        data = np.frombuffer(msg.data, dtype=np.uint8)
        depth_img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise RuntimeError("Failed to decode compressed depth image")
        if depth_img.ndim == 3:
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        return depth_img

    def _color_callback(self, msg, is_compressed: bool = False, is_depth: bool = False) -> None:
        try:
            image = self._decode_color(msg)
            latency = self._latency_ms(msg.header.stamp)
            with self._data_lock:
                self._latest_color = image
            self._update_status("front_color", latency)
        except Exception as exc:
            self.get_logger().error(f"Error in color_callback: {exc}")

    def _depth_callback(self, msg, is_compressed: bool = False, is_depth: bool = True) -> None:
        try:
            image = self._decode_depth(msg)
            latency = self._latency_ms(msg.header.stamp)
            with self._data_lock:
                self._latest_depth = image
            self._update_status("front_depth", latency)
        except Exception as exc:
            self.get_logger().error(f"Error in depth_callback: {exc}")

    def _wrist_color_callback(self, msg, is_compressed: bool = False, is_depth: bool = False) -> None:
        try:
            image = self._decode_color(msg)
            latency = self._latency_ms(msg.header.stamp)
            with self._data_lock:
                self._latest_wrist_color = image
            self._update_status("wrist_color", latency)
        except Exception as exc:
            self.get_logger().error(f"Error in wrist_color_callback: {exc}")

    def _wrist_depth_callback(self, msg, is_compressed: bool = False, is_depth: bool = True) -> None:
        try:
            image = self._decode_depth(msg)
            latency = self._latency_ms(msg.header.stamp)
            with self._data_lock:
                self._latest_wrist_depth = image
            self._update_status("wrist_depth", latency)
        except Exception as exc:
            self.get_logger().error(f"Error in wrist_depth_callback: {exc}")

    def _front_metadata_callback(self, msg: Metadata) -> None:
        with self._data_lock:
            self._topic_status["front_metadata"] = self.get_clock().now()
            self._last_front_metadata_json = msg.json_data

    def _wrist_metadata_callback(self, msg: Metadata) -> None:
        with self._data_lock:
            self._topic_status["wrist_metadata"] = self.get_clock().now()
            self._last_wrist_metadata_json = msg.json_data

    def _arm_state_callback(self, msg: JointState) -> None:
        with self._data_lock:
            self._latest_arm_state = msg
            self._topic_status["arm_state"] = self.get_clock().now()
