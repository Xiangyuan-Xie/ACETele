import json
import os
import sys
import threading
import time

import numpy as np
import rclpy
from data_collector_ros2.data_collector_gui import DataCollectorWindow
from PIL import Image as PILImage
from PySide6.QtWidgets import QApplication
from rclpy.node import Node
from realsense2_camera_msgs.msg import Extrinsics, Metadata
from sensor_msgs.msg import CameraInfo, Image


def img_msg_to_numpy(msg):
    dtype_class = np.uint8
    channels = 1

    if msg.encoding == "bgr8":
        dtype_class = np.uint8
        channels = 3
    elif msg.encoding == "rgb8":
        dtype_class = np.uint8
        channels = 3
    elif msg.encoding == "mono8":
        dtype_class = np.uint8
        channels = 1
    elif msg.encoding == "16UC1" or msg.encoding == "mono16":
        dtype_class = np.uint16
        channels = 1
    else:
        if "8" in msg.encoding:
            dtype_class = np.uint8
        elif "16" in msg.encoding:
            dtype_class = np.uint16

    dtype = np.dtype(dtype_class)
    itemsize = dtype.itemsize

    buf = msg.data

    if channels == 3:
        shape = (msg.height, msg.width, 3)
        strides = (msg.step, 3 * itemsize, itemsize)
    else:
        shape = (msg.height, msg.width)
        strides = (msg.step, itemsize)

    # Use copy=True to own the data, safer for threading
    # Using np.ndarray constructor with buffer
    arr = np.ndarray(shape, dtype=dtype, buffer=buf, strides=strides)
    return arr.copy()


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__("data_collector")

        # Parameters
        self.declare_parameter("color_topic", "/camera/front/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/front/depth/image_rect_raw")
        self.declare_parameter("color_info_topic", "/camera/front/color/camera_info")
        self.declare_parameter("color_metadata_topic", "/camera/front/color/metadata")
        self.declare_parameter("depth_info_topic", "/camera/front/depth/camera_info")
        self.declare_parameter("depth_metadata_topic", "/camera/front/depth/metadata")
        self.declare_parameter("depth_to_color_ext_topic", "/camera/front/extrinsics/depth_to_color")
        self.declare_parameter("depth_to_depth_ext_topic", "/camera/front/extrinsics/depth_to_depth")

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.color_info_topic = self.get_parameter("color_info_topic").value
        self.color_metadata_topic = self.get_parameter("color_metadata_topic").value
        self.depth_info_topic = self.get_parameter("depth_info_topic").value
        self.depth_metadata_topic = self.get_parameter("depth_metadata_topic").value
        self.depth_to_color_ext_topic = self.get_parameter("depth_to_color_ext_topic").value
        self.depth_to_depth_ext_topic = self.get_parameter("depth_to_depth_ext_topic").value

        # QoS
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1
        )
        qos_reliable = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1
        )

        # Subscribers
        self.create_subscription(Image, self.color_topic, self.color_callback, qos)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, qos)
        self.create_subscription(CameraInfo, self.color_info_topic, self.color_info_callback, qos_reliable)
        self.create_subscription(Metadata, self.color_metadata_topic, self.color_metadata_callback, qos)
        self.create_subscription(CameraInfo, self.depth_info_topic, self.depth_info_callback, qos_reliable)
        self.create_subscription(Metadata, self.depth_metadata_topic, self.depth_metadata_callback, qos)
        self.create_subscription(Extrinsics, self.depth_to_color_ext_topic, self.ext_d2c_callback, qos_reliable)
        self.create_subscription(Extrinsics, self.depth_to_depth_ext_topic, self.ext_d2d_callback, qos_reliable)

        # Data storage
        self.latest_color = None
        self.latest_depth = None
        self.last_metadata = {}
        self.topic_status = {}
        self.data_lock = threading.Lock()

        # Recording
        self.is_recording_flag = False
        self.recording_dir = ""
        self.frame_count = 0
        self.recording_thread = threading.Thread(target=self.save_data_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        self.get_logger().info("Data Collector Node Started (Python - PySide6 - No OpenCV/CvBridge)")

    def color_callback(self, msg):
        try:
            cv_image = img_msg_to_numpy(msg)
            with self.data_lock:
                self.latest_color = cv_image
            self.update_status(self.color_topic)
        except Exception as e:
            self.get_logger().error(f"Error processing color image: {e}")

    def depth_callback(self, msg):
        try:
            cv_image = img_msg_to_numpy(msg)
            with self.data_lock:
                self.latest_depth = cv_image
            self.update_status(self.depth_topic)
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

    def color_info_callback(self, msg):
        self.update_status(self.color_info_topic)

    def color_metadata_callback(self, msg):
        self.update_metadata(msg)
        self.update_status(self.color_metadata_topic)

    def depth_info_callback(self, msg):
        self.update_status(self.depth_info_topic)

    def depth_metadata_callback(self, msg):
        self.update_status(self.depth_metadata_topic)

    def ext_d2c_callback(self, msg):
        self.update_status(self.depth_to_color_ext_topic)

    def ext_d2d_callback(self, msg):
        self.update_status(self.depth_to_depth_ext_topic)

    def update_status(self, topic):
        with self.data_lock:
            self.topic_status[topic] = time.time()

    def update_metadata(self, msg):
        try:
            json_str = msg.json_data
            data = json.loads(json_str)
            with self.data_lock:
                self.last_metadata = data
        except:
            pass

    def get_latest_images(self):
        with self.data_lock:
            c = self.latest_color.copy() if self.latest_color is not None else None
            d = self.latest_depth.copy() if self.latest_depth is not None else None
            return c, d

    def get_status_info(self):
        info = {}
        now = time.time()
        with self.data_lock:
            for topic, last_time in self.topic_status.items():
                diff = now - last_time
                if diff < 2.0:
                    info[topic] = "ONLINE"
                else:
                    info[topic] = f"OFFLINE ({int(diff)}s)"
        return info

    def get_metadata_json(self):
        with self.data_lock:
            return json.dumps(self.last_metadata, indent=2)

    def start_recording(self, output_dir):
        if self.is_recording_flag:
            return
        self.recording_dir = output_dir
        os.makedirs(self.recording_dir, exist_ok=True)
        self.frame_count = 0
        self.is_recording_flag = True
        self.get_logger().info(f"Started recording to: {self.recording_dir}")

    def stop_recording(self):
        if not self.is_recording_flag:
            return
        self.is_recording_flag = False
        self.get_logger().info(f"Stopped recording. Total frames: {self.frame_count}")

    def is_recording(self):
        return self.is_recording_flag

    def save_data_worker(self):
        while True:
            if not self.is_recording_flag:
                time.sleep(0.1)
                continue

            with self.data_lock:
                if self.latest_color is None or self.latest_depth is None:
                    continue
                color_snap = self.latest_color.copy()
                depth_snap = self.latest_depth.copy()
                meta_snap = self.last_metadata.copy()

            timestamp = str(int(time.time() * 1000))
            color_path = os.path.join(self.recording_dir, f"{timestamp}_color.jpg")
            depth_path = os.path.join(self.recording_dir, f"{timestamp}_depth.png")
            meta_path = os.path.join(self.recording_dir, f"{timestamp}_meta.json")

            try:
                # Save Color
                # Assuming BGR from ROS, convert to RGB for PIL
                if len(color_snap.shape) == 3 and color_snap.shape[2] == 3:
                    color_rgb = color_snap[..., ::-1]  # BGR to RGB
                    PILImage.fromarray(color_rgb).save(color_path)
                else:
                    PILImage.fromarray(color_snap).save(color_path)

                # Save Depth
                # PIL handles uint16
                PILImage.fromarray(depth_snap).save(depth_path)

                with open(meta_path, "w") as f:
                    json.dump(meta_snap, f, indent=2)
                self.frame_count += 1
            except Exception as e:
                self.get_logger().error(f"Failed to save data: {e}")

            time.sleep(0.033)


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()

    app = QApplication(sys.argv)
    window = DataCollectorWindow(node)
    window.show()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,))
    spin_thread.daemon = True
    spin_thread.start()

    try:
        sys.exit(app.exec())
    except Exception as e:
        print(e)
    finally:
        node.stop_recording()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
