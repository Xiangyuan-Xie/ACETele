"""
Video Composition Pipeline
Creates a multi-camera layout video from three image sequences.
"""

import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ============================================================================
# VISUAL CONSTANTS
# ============================================================================
BG_COLOR = (245, 245, 245)  # Background color (BGR)
BORDER_COLOR = (200, 200, 200)  # Border color
BORDER_THICKNESS = 2  # Border line thickness

# Layout constants
OUTER_MARGIN = 40  # Margin around entire frame
INNER_GAP = 20  # Gap between right panels
LEFT_RATIO = 0.64  # Left panel width ratio (0.0-1.0)

# Text rendering constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (60, 60, 60)
FONT_THICKNESS = 2


# ============================================================================
# PROGRESS TRACKING CLASS
# ============================================================================
class ProgressTracker:
    """Tracks and displays progress information for video generation."""

    def __init__(self, total_frames: int):
        """
        Initialize progress tracker.

        Args:
            total_frames: Total number of frames to process
        """
        self.total_frames = total_frames
        self.processed_frames = 0
        self.written_frames = 0
        self.start_time = time.time()
        self.lock = Lock()

    def update_processed(self, count: int = 1) -> None:
        """
        Update processed frame count.

        Args:
            count: Number of frames processed
        """
        with self.lock:
            self.processed_frames += count

    def update_written(self, count: int = 1) -> None:
        """
        Update written frame count.

        Args:
            count: Number of frames written
        """
        with self.lock:
            self.written_frames += count

    def get_elapsed_time(self) -> float:
        """
        Get elapsed time in seconds.

        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time

    def get_eta(self) -> float:
        """
        Calculate estimated time remaining.

        Returns:
            Estimated time remaining in seconds, or 0 if not enough data
        """
        if self.processed_frames == 0:
            return 0
        elapsed = self.get_elapsed_time()
        return (elapsed / self.processed_frames) * (self.total_frames - self.processed_frames)

    def format_time(self, seconds: float) -> str:
        """
        Format seconds to human-readable time.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (HH:MM:SS or MM:SS)
        """
        if seconds < 0:
            return "--:--:--"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def get_progress_bar(self, length: int = 20) -> str:
        """
        Generate a text-based progress bar.

        Args:
            length: Length of progress bar in characters

        Returns:
            Progress bar string
        """
        if self.total_frames == 0:
            return "[" + " " * length + "]"

        progress = min(self.processed_frames / self.total_frames, 1.0)
        filled_length = int(length * progress)
        bar = "█" * filled_length + "░" * (length - filled_length)
        return f"[{bar}]"

    def print_progress(self) -> None:
        """Print current progress status."""
        with self.lock:
            if self.total_frames == 0:
                return

            progress_percent = (self.processed_frames / self.total_frames) * 100
            elapsed = self.get_elapsed_time()
            eta = self.get_eta()

            # Calculate processing speed
            if elapsed > 0:
                fps = self.processed_frames / elapsed
            else:
                fps = 0

            progress_bar = self.get_progress_bar(30)

            print(
                f"\rProgress: {progress_bar} {progress_percent:6.2f}% | "
                f"Frames: {self.processed_frames:4d}/{self.total_frames} | "
                f"Speed: {fps:5.1f} fps | "
                f"Time: {self.format_time(elapsed)} | "
                f"ETA: {self.format_time(eta)}",
                end="",
                flush=True,
            )


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================
def resize_with_aspect_ratio(
    img: np.ndarray,
    target_size: Tuple[int, int],
    allow_upscale: bool = False,
) -> np.ndarray:
    """
    Resize image while preserving aspect ratio, padding to target size.

    Args:
        img: Input image (H, W, C)
        target_size: Target dimensions (width, height)
        allow_upscale: Whether to allow upscaling smaller images

    Returns:
        Resized and padded image
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size

    # Calculate scale factor
    scale = min(target_w / w, target_h / h)
    if not allow_upscale:
        scale = min(scale, 1.0)

    # Calculate new dimensions
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # Choose interpolation method
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # Center image on canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


def draw_border(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """
    Draw a border rectangle around a region.

    Args:
        img: Image to draw on
        x, y: Top-left corner coordinates
        w, h: Width and height of region
    """
    cv2.rectangle(img, (x, y), (x + w, y + h), BORDER_COLOR, BORDER_THICKNESS)


def draw_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    """
    Draw a text label at specified position.

    Args:
        img: Image to draw on
        text: Label text
        x, y: Top-left corner of text bounding box
    """
    cv2.putText(
        img,
        text,
        (x + 10, y + 28),  # Offset for visual balance
        FONT,
        FONT_SCALE,
        FONT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


# ============================================================================
# MAIN VIDEO GENERATION FUNCTION
# ============================================================================
def create_video_from_cameras(
    camera_names: List[str],
    output_video_path: str,
    fps: int,
    frame_size: Tuple[int, int],
    queue_size: int = 16,
    progress_interval: float = 0.5,
) -> None:
    """
    Create a composite video from three camera image sequences.

    Layout:
        +---------------------------------------+
        |                 LEFT                  |  TOP   |
        |                 PANEL                 +--------+
        |                 (64%)                 | BOTTOM |
        |                                       | PANEL  |
        +---------------------------------------+

    Args:
        camera_names: List of 3 camera directory names
        output_video_path: Path for output MP4 file
        fps: Source frame rate (used for time-based sampling)
        frame_size: Output video dimensions (width, height)
        queue_size: Max frames in producer-consumer queue
        progress_interval: Time interval for progress updates in seconds

    Raises:
        ValueError: If not exactly 3 cameras provided or no images found
    """
    if len(camera_names) != 3:
        raise ValueError("Exactly 3 camera names are required")

    TARGET_FPS = 30  # Output video frame rate

    # ------------------------------------------------------------------------
    # 1. PRINT STARTUP INFORMATION
    # ------------------------------------------------------------------------
    print("=" * 80)
    print("VIDEO COMPOSITION PIPELINE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {output_video_path}")
    print(f"Output resolution: {frame_size[0]}x{frame_size[1]}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Camera sources: {', '.join(camera_names)}")
    print("-" * 80)

    # ------------------------------------------------------------------------
    # 2. COLLECT IMAGE PATHS
    # ------------------------------------------------------------------------
    print("Collecting image paths...")
    image_paths: Dict[str, List[List[Path]]] = {"rgb": []}

    for cam in camera_names:
        folder = Path(f"../simulation/image/{cam}")
        files = sorted(folder.glob("rgb_*.png"))

        if not files:
            raise ValueError(f"No images found in {folder}")

        image_paths["rgb"].append(files)
        print(f"  {cam}: {len(files)} frames found")

    # Determine minimum sequence length
    min_frames = min(len(seq) for seq in image_paths["rgb"])
    print(f"Minimum sequence length: {min_frames} frames")

    # ------------------------------------------------------------------------
    # 3. FRAME SAMPLING (FPS ADJUSTMENT)
    # ------------------------------------------------------------------------
    print(f"Source FPS: {fps}")
    if fps > TARGET_FPS:
        # Downsample: keep frames where frame index changes
        frame_indices = [i for i in range(min_frames) if int(i * TARGET_FPS / fps) != int((i - 1) * TARGET_FPS / fps)]
        # Ensure last frame is included
        if min_frames - 1 not in frame_indices:
            frame_indices.append(min_frames - 1)
        print(f"Downsampling from {fps} FPS to {TARGET_FPS} FPS")
        print(f"Resulting frames: {len(frame_indices)} (reduction: {100 - (len(frame_indices)/min_frames*100):.1f}%)")
    else:
        # Use all frames
        frame_indices = list(range(min_frames))
        if fps < TARGET_FPS:
            print(f"Warning: Source FPS ({fps}) is lower than target FPS ({TARGET_FPS})")

    total_frames = len(frame_indices)
    print(f"Total frames to process: {total_frames}")

    # Initialize progress tracker
    progress = ProgressTracker(total_frames)
    print("-" * 80)

    # ------------------------------------------------------------------------
    # 4. LAYOUT GEOMETRY CALCULATION
    # ------------------------------------------------------------------------
    output_w, output_h = frame_size

    # Calculate usable area inside margins
    usable_h = output_h - 2 * OUTER_MARGIN
    usable_w = output_w - 2 * OUTER_MARGIN

    # Left panel width (64% of usable width)
    left_w = int(round(usable_w * LEFT_RATIO))
    right_w = usable_w - left_w  # Remaining width for right panels

    # Split right column vertically
    right_h = (usable_h - INNER_GAP) // 2
    bottom_h = usable_h - right_h - INNER_GAP

    # Panel dimensions
    left_size = (left_w, usable_h)
    right_size_top = (right_w, right_h)
    right_size_bottom = (right_w, bottom_h)

    # Panel positions
    left_x = OUTER_MARGIN
    left_y = OUTER_MARGIN
    right_x = left_x + left_w
    top_y = OUTER_MARGIN
    bottom_y = top_y + right_h + INNER_GAP

    # Print layout information
    print("Video Layout Configuration:")
    print(f"  Overall: {output_w}x{output_h} (WxH)")
    print(f"  Left panel: {left_size[0]}x{left_size[1]} at ({left_x}, {left_y})")
    print(f"  Top-right panel: {right_size_top[0]}x{right_size_top[1]} at ({right_x}, {top_y})")
    print(f"  Bottom-right panel: {right_size_bottom[0]}x{right_size_bottom[1]} at ({right_x}, {bottom_y})")
    print("-" * 80)

    # ------------------------------------------------------------------------
    # 5. VIDEO WRITER SETUP
    # ------------------------------------------------------------------------
    print("Initializing video writer...")
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), TARGET_FPS, (output_w, output_h))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize video writer for {output_video_path}")

    print("Video writer initialized successfully")

    # Producer-consumer queue
    queue: Queue[Optional[np.ndarray]] = Queue(maxsize=queue_size)

    # ------------------------------------------------------------------------
    # 6. PROGRESS MONITOR THREAD
    # ------------------------------------------------------------------------
    stop_progress_monitor = False
    progress_lock = Lock()

    def progress_monitor() -> None:
        """Monitor and display progress at regular intervals."""
        last_update = 0.0
        while True:
            with progress_lock:
                if stop_progress_monitor:
                    break

            current_time = time.time()
            if current_time - last_update >= progress_interval:
                progress.print_progress()
                last_update = current_time

            time.sleep(0.1)  # Short sleep to prevent CPU hogging

    # ------------------------------------------------------------------------
    # 7. PRODUCER THREAD (Frame Composition)
    # ------------------------------------------------------------------------
    def producer() -> None:
        """Read images, compose frames, and add to queue."""
        for idx, frame_idx in enumerate(frame_indices, 1):
            # Load images for all three cameras
            images = []
            for i in range(3):
                img_path = image_paths["rgb"][i][frame_idx]
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"\nWarning: Failed to load image {img_path}")
                    # Create a blank frame as fallback
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                images.append(img)

            # Create background canvas
            frame = np.full((output_h, output_w, 3), BG_COLOR, np.uint8)

            # Place and resize left panel (camera 0)
            frame[left_y : left_y + left_size[1], left_x : left_x + left_size[0]] = resize_with_aspect_ratio(
                images[0], left_size
            )

            # Place and resize top-right panel (camera 1)
            frame[top_y : top_y + right_size_top[1], right_x : right_x + right_size_top[0]] = resize_with_aspect_ratio(
                images[1], right_size_top
            )

            # Place and resize bottom-right panel (camera 2)
            frame[bottom_y : bottom_y + right_size_bottom[1], right_x : right_x + right_size_bottom[0]] = (
                resize_with_aspect_ratio(images[2], right_size_bottom)
            )

            # Draw borders
            draw_border(frame, left_x, left_y, *left_size)
            draw_border(frame, right_x, top_y, *right_size_top)
            draw_border(frame, right_x, bottom_y, *right_size_bottom)

            # Add labels
            draw_label(frame, "External View, 1x", left_x, left_y)
            draw_label(frame, "Front Camera, 1x", right_x, top_y)
            draw_label(frame, "Wrist Camera, 1x", right_x, bottom_y)

            # Add to processing queue
            queue.put(frame)

            # Update progress
            progress.update_processed(1)

        # Signal completion
        queue.put(None)

    # ------------------------------------------------------------------------
    # 8. CONSUMER THREAD (Video Writing)
    # ------------------------------------------------------------------------
    def consumer() -> None:
        """Read frames from queue and write to video file."""
        written = 0
        while True:
            frame = queue.get()
            if frame is None:  # Termination signal
                break
            writer.write(frame)
            written += 1
            progress.update_written(1)

        # Final progress update
        progress.print_progress()
        print()  # New line after progress bar

        print(f"\nTotal frames written: {written}")

    # ------------------------------------------------------------------------
    # 9. EXECUTION
    # ------------------------------------------------------------------------
    print("Starting video generation...")
    print("Press Ctrl+C to interrupt\n")

    # Start progress monitor
    progress_monitor_thread = Thread(target=progress_monitor, daemon=True)
    progress_monitor_thread.start()

    # Start producer and consumer threads
    producer_thread = Thread(target=producer)
    consumer_thread = Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    try:
        # Wait for threads to complete
        producer_thread.join()
        consumer_thread.join()

        # Stop progress monitor
        with progress_lock:
            stop_progress_monitor = True

        # Final status
        print("-" * 80)
        print("Video generation completed successfully!")

    except KeyboardInterrupt:
        print("\n\n" + "!" * 80)
        print("INTERRUPTED BY USER")
        print("Cleaning up resources...")

        # Signal threads to stop
        with progress_lock:
            stop_progress_monitor = True

        # Try to join threads
        producer_thread.join(timeout=1)
        consumer_thread.join(timeout=1)

        print("Cleanup complete.")
        print("!" * 80)
        return

    finally:
        # Always release the writer
        writer.release()

    # ------------------------------------------------------------------------
    # 10. FINAL SUMMARY
    # ------------------------------------------------------------------------
    elapsed = progress.get_elapsed_time()
    print(f"Processing time: {progress.format_time(elapsed)}")
    print(f"Average speed: {total_frames/elapsed:.1f} FPS" if elapsed > 0 else "Average speed: N/A")
    print(f"Output file: {output_video_path}")
    print(f"Output frame rate: {TARGET_FPS} FPS")
    print(f"Output resolution: {output_w}x{output_h}")
    print(f"Video duration: {progress.format_time(total_frames / TARGET_FPS)}")
    print("=" * 80)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Example usage
    create_video_from_cameras(
        camera_names=[
            "external_camera",
            "front_camera",
            "wrist_camera",
        ],
        output_video_path="output.mp4",
        fps=100,
        frame_size=(1920, 1080),
        progress_interval=0.5,  # Update progress every 0.5 seconds
    )
