import argparse
import bisect
import sqlite3
from pathlib import Path

import h5py
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# ==============================================================================
# 1. Extraction Helpers
# ==============================================================================


def extract_attr(attr_names, dtype=np.float32):
    """Extracts attributes from a ROS message, handling nested geometry types."""
    if isinstance(attr_names, str):
        attr_names = [attr_names]

    def _func(msg):
        vals = []
        for name in attr_names:
            if not hasattr(msg, name):
                return None
            val = getattr(msg, name)

            # Handle ROS geometry objects (Point, Vector3, Quaternion)
            if hasattr(val, "x") and hasattr(val, "y") and hasattr(val, "z"):
                # Quaternion (x, y, z, w) or Vector3/Point (x, y, z)
                comps = [val.x, val.y, val.z]
                if hasattr(val, "w"):
                    comps.append(val.w)
                vals.append(np.array(comps, dtype=dtype))
            else:
                # Standard scalar or array
                vals.append(np.atleast_1d(np.array(val, dtype=dtype)))

        return np.concatenate(vals) if vals else None

    return _func


def extract_image(msg):
    """Extracts image data as uint8 array."""
    # Handle CompressedImage
    if hasattr(msg, "format") and "compressed" in msg.format:
        return np.frombuffer(msg.data, dtype=np.uint8)
    # Handle Image
    if hasattr(msg, "data"):
        return np.frombuffer(msg.data, dtype=np.uint8)
    return None


# ==============================================================================
# 2. Configuration (User Provided)
# ==============================================================================

TOPIC_CONFIG = {
    "/fmu/out/manual_control_setpoint": {
        "outputs": [("action/base", extract_attr(["throttle", "yaw", "pitch", "roll"]))]
    },
    "/arm/command": {"outputs": [("action/arm", extract_attr("position"))]},
    "Xie_1/pose": {
        "outputs": [
            ("observation/base_position_mocap", extract_attr("position")),
            ("observation/base_orientation_mocap", extract_attr("orientation")),
        ]
    },
    "/fmu/out/vehicle_odometry": {
        "outputs": [
            ("observation/base_position", extract_attr("position")),
            ("observation/base_orientation", extract_attr("q")),
            ("observation/base_linear_velocity", extract_attr("velocity")),
            ("observation/base_angular_velocity", extract_attr("angular_velocity")),
        ]
    },
    "/arm/state": {
        "outputs": [
            ("observation/arm_position", extract_attr("position")),
            ("observation/arm_velocity", extract_attr("velocity")),
            ("observation/arm_effort", extract_attr("effort")),
        ]
    },
    "/camera/color/image_raw": {"outputs": [("observations/images/front/color", extract_image)]},
    "/camera/front/aligned_depth_to_color/image_raw": {"outputs": [("observations/images/front/depth", extract_image)]},
}

# ==============================================================================
# 3. Converter Logic
# ==============================================================================


class Bag2Hdf5Converter:
    def __init__(self, bag_path, output_path, sync_topic=None, tolerance_ms=50):
        self.bag_path = Path(bag_path)
        self.output_path = Path(output_path)
        self.sync_topic = sync_topic
        self.tolerance_ns = tolerance_ms * 1_000_000
        self.config = TOPIC_CONFIG

        # Resolve database file
        if self.bag_path.is_dir():
            files = list(self.bag_path.glob("*.db3")) + list(self.bag_path.glob("*.mcap"))
            if not files:
                raise ValueError(f"No .db3/.mcap found in {bag_path}")
            self.db_path = files[0]
        else:
            self.db_path = self.bag_path

    def _get_bag_topics(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, type FROM topics")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def read_messages(self):
        """Reads messages mapping configured topics to actual bag topics (handling slash mismatches)."""
        bag_topics = self._get_bag_topics()

        # Map configured topics to actual topics in bag
        topic_map = {}  # config_topic -> actual_topic_in_bag
        for cfg_topic in self.config:
            if cfg_topic in bag_topics:
                topic_map[cfg_topic] = cfg_topic
            else:
                # Try adding/removing leading slash
                alt = "/" + cfg_topic if not cfg_topic.startswith("/") else cfg_topic[1:]
                if alt in bag_topics:
                    print(f"Mapped '{cfg_topic}' -> '{alt}'")
                    topic_map[cfg_topic] = alt
                else:
                    print(f"Warning: Topic '{cfg_topic}' not found in bag.")

        if not topic_map:
            raise ValueError("No matching topics found in bag.")

        # Import message types
        msg_classes = {}
        for actual_topic in topic_map.values():
            try:
                msg_type = bag_topics[actual_topic]
                msg_classes[actual_topic] = get_message(msg_type)
            except Exception as e:
                print(f"Error loading message type for {actual_topic}: {e}")

        # Read data
        data = {cfg_topic: [] for cfg_topic in topic_map}
        placeholders = ",".join("?" for _ in topic_map)
        query = f"""
            SELECT topics.name, messages.timestamp, messages.data
            FROM messages JOIN topics ON messages.topic_id = topics.id
            WHERE topics.name IN ({placeholders}) ORDER BY messages.timestamp
        """

        print(f"Reading from {self.db_path.name}...")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, list(topic_map.values()))

            # Reverse map for lookup
            actual_to_config = {v: k for k, v in topic_map.items()}

            for topic, ts, raw_data in cursor:
                if topic in msg_classes:
                    try:
                        msg = deserialize_message(raw_data, msg_classes[topic])
                        cfg_topic = actual_to_config[topic]
                        data[cfg_topic].append((ts, msg))
                    except Exception:
                        pass

        return data

    def convert(self):
        data = self.read_messages()

        # 1. Determine Sync Topic
        if not self.sync_topic or self.sync_topic not in data:
            # Prioritize topics with 'action' or 'camera'
            candidates = [t for t in data if data[t]]
            if not candidates:
                return

            preferred = [t for t in candidates if "action" in t or "camera" in t]
            self.sync_topic = preferred[0] if preferred else max(candidates, key=lambda t: len(data[t]))
            print(f"Sync topic: {self.sync_topic}")

        # 2. Synchronize
        master_msgs = data[self.sync_topic]
        results = {path: [] for cfg in self.config.values() for path, _ in cfg["outputs"]}
        img_lens = {path: [] for path in results if "images" in path}

        for master_ts, master_msg in master_msgs:
            for cfg_topic, cfg in self.config.items():
                # Find message
                msg = None
                if cfg_topic == self.sync_topic:
                    msg = master_msg
                elif cfg_topic in data and data[cfg_topic]:
                    # Binary search for closest timestamp
                    topic_data = data[cfg_topic]
                    timestamps = [m[0] for m in topic_data]
                    idx = bisect.bisect_left(timestamps, master_ts)

                    # Check neighbors
                    candidates = []
                    if idx < len(timestamps):
                        candidates.append((timestamps[idx], topic_data[idx][1]))
                    if idx > 0:
                        candidates.append((timestamps[idx - 1], topic_data[idx - 1][1]))

                    if candidates:
                        # Select closest within tolerance
                        best_ts, best_msg = min(candidates, key=lambda x: abs(x[0] - master_ts))
                        if abs(best_ts - master_ts) <= self.tolerance_ns:
                            msg = best_msg
                        else:
                            # Use nearest neighbor even if out of tolerance
                            # User requested alignment, usually implies nearest neighbor
                            msg = best_msg

                # Extract data
                for out_path, func in cfg["outputs"]:
                    val = func(msg) if msg else None
                    results[out_path].append(val)
                    if out_path in img_lens:
                        img_lens[out_path].append(len(val) if val is not None else 0)

        # 3. Write HDF5
        print(f"Writing to {self.output_path}...")
        with h5py.File(self.output_path, "w") as f:
            f.attrs.update({"compress": True, "sim": False})

            for path, values in results.items():
                # Filter None
                valid_vals = [v for v in values if v is not None]
                if not valid_vals:
                    continue

                # Determine shape/type
                sample = valid_vals[0]
                if "images" in path:  # Variable length byte arrays
                    max_len = max(len(v) for v in values if v is not None)
                    dset = f.create_dataset(path, (len(values), max_len), dtype="uint8")
                    for i, v in enumerate(values):
                        if v is not None:
                            dset[i, : len(v)] = v
                else:  # Fixed length numerical arrays
                    # Fill missing with zeros
                    fill = np.zeros_like(sample)
                    arr = np.array([v if v is not None else fill for v in values])
                    f.create_dataset(path, data=arr)

            # Write compress_len (N_frames, N_cams)
            if img_lens:
                sorted_keys = sorted(img_lens.keys())
                lens = np.array([img_lens[k] for k in sorted_keys], dtype="int32").T
                f.create_dataset("compress_len", data=lens)

        self._print_stats()

    def _print_stats(self):
        print(f"\nSaved: {self.output_path} ({self.output_path.stat().st_size/1024**2:.2f} MB)")
        with h5py.File(self.output_path, "r") as f:

            def visit(name, node):
                if isinstance(node, h5py.Dataset):
                    print(f"  {name}: {node.shape} {node.dtype}")

            f.visititems(visit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path")
    parser.add_argument("output_path")
    parser.add_argument("--sync-topic", default=None)
    args = parser.parse_args()

    Bag2Hdf5Converter(args.bag_path, args.output_path, args.sync_topic).convert()
