from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def create_video_from_cameras(
    camera_names: List[str],
    output_video_path: str,
    fps: int = 30,
    frame_size: Optional[Tuple[int, int]] = None,  # Changed to Optional
    queue_size: int = 16,
) -> None:
    """
    Create a 2x2 grid video from two cameras (each with RGB and depth images).

    Args:
        camera_names: List of two camera names
        output_video_path: Path to save the output video
        fps: Original frames per second of the input images
        frame_size: Output video frame size (width, height)
        queue_size: Size of the queue for producer-consumer pattern
    """
    if len(camera_names) != 2:
        raise ValueError("Exactly 2 camera names are required")

    # Set target FPS to 30
    TARGET_FPS = 30

    # Collect image paths with proper type annotation
    image_paths: Dict[str, List[List[Path]]] = {"rgb": [], "depth": []}  # Fixed type annotation
    for camera in camera_names:
        folder = Path(f"../simulation/image/{camera}")
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        rgb_files = sorted(folder.glob("rgb_*.png"))
        depth_files = sorted(folder.glob("depth_*.png"))

        if not rgb_files or not depth_files:
            raise ValueError(f"Missing RGB or depth images in folder: {camera}")

        image_paths["rgb"].append(rgb_files)
        image_paths["depth"].append(depth_files)

    # Determine minimum number of frames available across all sequences
    min_frames = min(
        len(image_paths["rgb"][0]),
        len(image_paths["rgb"][1]),
        len(image_paths["depth"][0]),
        len(image_paths["depth"][1]),
    )

    # Calculate frame indices for uniform sampling if input FPS > 30
    if fps > TARGET_FPS:
        # Calculate the sampling ratio
        sampling_ratio = fps / TARGET_FPS
        # Generate evenly spaced frame indices
        frame_indices = []
        for i in range(min_frames):
            if int(i * TARGET_FPS / fps) != int((i - 1) * TARGET_FPS / fps):
                frame_indices.append(i)
        # Make sure to include the last frame
        if min_frames - 1 not in frame_indices:
            frame_indices.append(min_frames - 1)
        print(f"Input FPS: {fps}, Output FPS: {TARGET_FPS}, " f"Frame sampling ratio: {sampling_ratio:.2f}")
        print(f"Selected {len(frame_indices)} frames from {min_frames} total frames")
        total_frames = len(frame_indices)
    else:
        # Use all frames if input FPS <= 30
        frame_indices = list(range(min_frames))
        total_frames = min_frames
        print(f"Input FPS: {fps} <= 30, using all {min_frames} frames")

    # Determine frame dimensions
    sample = cv2.imread(str(image_paths["rgb"][0][0]))
    if frame_size is None:
        h, w = sample.shape[:2]
        sub_size = (w, h)  # width, height
    else:
        sub_size = (frame_size[0] // 2, frame_size[1] // 2)

    output_size = (sub_size[0] * 2, sub_size[1] * 2)

    # Initialize VideoWriter with target FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, TARGET_FPS, output_size)

    # Pre-allocate queue for producer-consumer pattern
    queue: Queue[Optional[np.ndarray]] = Queue(maxsize=queue_size)  # Fixed type annotation

    # ================= Producer Thread Function =================
    def producer() -> None:
        """Read images, arrange in 2x2 grid, and put frames in queue."""
        for idx, frame_idx in enumerate(frame_indices):
            frames = []

            # Camera 1 RGB
            img = cv2.imread(str(image_paths["rgb"][0][frame_idx]))
            frames.append(cv2.resize(img, sub_size))

            # Camera 2 RGB
            img = cv2.imread(str(image_paths["rgb"][1][frame_idx]))
            frames.append(cv2.resize(img, sub_size))

            # Camera 1 depth
            img = cv2.imread(str(image_paths["depth"][0][frame_idx]), cv2.IMREAD_UNCHANGED)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            frames.append(cv2.resize(img, sub_size))

            # Camera 2 depth
            img = cv2.imread(str(image_paths["depth"][1][frame_idx]), cv2.IMREAD_UNCHANGED)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            frames.append(cv2.resize(img, sub_size))

            # Arrange frames in 2x2 grid (pre-allocate buffer)
            frame_grid = np.empty((output_size[1], output_size[0], 3), dtype=np.uint8)

            h, w = sub_size[1], sub_size[0]
            frame_grid[0:h, 0:w] = frames[0]  # Top-left: Camera 1 RGB
            frame_grid[0:h, w : 2 * w] = frames[1]  # Top-right: Camera 2 RGB
            frame_grid[h : 2 * h, 0:w] = frames[2]  # Bottom-left: Camera 1 depth
            frame_grid[h : 2 * h, w : 2 * w] = frames[3]  # Bottom-right: Camera 2 depth

            queue.put(frame_grid)

            # Progress reporting
            if (idx + 1) % 50 == 0 or idx + 1 == total_frames:
                print(f"Produced: {idx + 1}/{total_frames} frames " f"(original frame {frame_idx + 1}/{min_frames})")

        queue.put(None)  # Termination signal

    # ================= Consumer Thread Function =================
    def consumer() -> None:
        """Get frames from queue and write to video file."""
        count = 0
        while True:
            frame = queue.get()
            if frame is None:
                break
            video_writer.write(frame)
            count += 1
        print(f"Written frames: {count}")

    # Start producer and consumer threads
    t_prod = Thread(target=producer, daemon=True)
    t_cons = Thread(target=consumer, daemon=True)

    t_prod.start()
    t_cons.start()

    # Wait for both threads to complete
    t_prod.join()
    t_cons.join()

    video_writer.release()
    print(f"Video saved to: {output_video_path}")
    print(f"Output video: {total_frames} frames at {TARGET_FPS} FPS")


# Simple usage example
if __name__ == "__main__":
    create_video_from_cameras(
        camera_names=["front_camera", "wrist_camera"],
        output_video_path="output.mp4",
        fps=100,  # Input FPS (will be downsampled to 30 FPS)
        frame_size=(1920, 1080),
    )
