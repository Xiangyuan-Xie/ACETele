import base64
import json
import time
from multiprocessing import Event, Process, Queue
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as EventType
from typing import Any, Dict, Optional, Union

import _queue
import cv2
import numpy as np
import zmq
from PySide6.QtCore import QObject, QTimer, Signal


class PublisherServer:
    """
    ZeroMQ-based PUB server running in a separate process.
    Responsible for encoding image data and publishing messages.
    """

    def __init__(self, server_host: str = "0.0.0.0", server_port: int = 5555):
        """
        Initialize the Publisher Server.

        Parameters
        ----------
        server_host : str, optional
            Host address to bind the server to (default: "0.0.0.0")
        server_port : int, optional
            Port number to bind the server to (default: 5555)
        """
        self.server_address = f"tcp://{server_host}:{server_port}"

        # ZeroMQ context and socket (initialized in child process)
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None

        # Server running flag
        self.running: EventType = Event()

        # Background worker process
        self.worker: Optional[Process] = None

        # Bounded queue for pending tasks
        self.task_queue: MPQueue = Queue(maxsize=256)

        # Start the background publishing process
        self.running.set()
        self.worker = Process(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _worker_loop(self) -> None:
        """
        Worker loop executed in the child process.
        Initializes ZeroMQ and continuously processes messages from the queue.
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self.server_address)

        # Ensure socket is fully bound before sending messages
        time.sleep(0.5)

        while self.running.is_set():
            try:
                # Use blocking get() to avoid busy-waiting
                message = self.task_queue.get(timeout=0.02)
                self._process_and_send_message(message)
            except _queue.Empty:  # Ignore queue empty timeout
                pass

        # Cleanup on exit
        if self.socket:
            self.socket.close(linger=0)
        if self.context:
            self.context.term()

    @staticmethod
    def encode_image(image: np.ndarray, name: str) -> Union[str, Dict[str, Any]]:
        """
        Encode an image as JPEG and convert to Base64 string.

        Parameters
        ----------
        image : np.ndarray
            Input image array
        name : str
            Image name (used for determining encoding method)

        Returns
        -------
        Union[str, dict]
            Base64-encoded image data. For RGB images returns a string.
            For depth images returns a dictionary containing the image and depth range.

        Raises
        ------
        ValueError
            If image encoding fails or image type is unknown
        """
        if "rgb" in name:
            # Encode RGB image as JPEG
            success, encoded_image = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not success:
                raise ValueError(f"Failed to encode image: {name}")
            return base64.b64encode(encoded_image.tobytes()).decode("utf-8")

        elif "depth" in name:
            # Normalize depth image and encode as 8-bit JPEG
            depth_min = float(np.min(image))
            depth_max = float(np.max(image))
            if depth_max == depth_min:
                depth_max = depth_min + 1.0
            depth_normalized = (image - depth_min) / (depth_max - depth_min)
            depth_8bit = (depth_normalized * 255).astype(np.uint8)
            success, encoded_image = cv2.imencode(".jpg", depth_8bit, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not success:
                raise ValueError(f"Failed to encode image: {name}")
            return {
                "image": base64.b64encode(encoded_image.tobytes()).decode("utf-8"),
                "depth_min": depth_min,
                "depth_max": depth_max,
            }
        else:
            raise ValueError(f"Unknown image type: {name}")

    def _process_and_send_message(self, message: Dict[str, Any]) -> None:
        """
        Process image message, encode images, and publish via ZeroMQ.

        Parameters
        ----------
        message : dict
            Dictionary containing image data and metadata
        """
        processed_message = message.copy()  # Create a copy to avoid modifying original

        # Encode all images in the message
        for name, image in message["Image"].items():
            processed_message["Image"][name] = PublisherServer.encode_image(image, name)

        # Convert numpy arrays to lists for JSON serialization
        for name, data in message["Data"].items():
            if isinstance(data, np.ndarray):
                processed_message["Data"][name] = data.tolist()

        # Publish the processed message
        if self.socket is not None:
            self.socket.send_string(json.dumps(processed_message))

    def send(self, message: Dict[str, Any]) -> None:
        """
        Push a message to the task queue for publishing.

        If the queue is full, the oldest message is discarded to make space.

        Parameters
        ----------
        message : dict
            Image message to be published
        """
        if self.task_queue.full():
            try:
                self.task_queue.get_nowait()  # Discard oldest message when queue is full
            except _queue.Empty:
                pass

        self.task_queue.put(message, block=False)

    def close(self) -> None:
        """
        Stop the worker process and clean up resources.
        """
        self.running.clear()

        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2.0)
            if self.worker.is_alive():
                self.worker.terminate()

        # Clear any pending tasks from the queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except _queue.Empty:
                break


class SubscriberClient(QObject):
    """
    ZeroMQ-based SUB client running in a separate process.
    Responsible for receiving and decoding image data from a publisher.
    """

    # Qt signals for inter-thread communication
    data_received = Signal(dict)  # Emitted when new data is received

    def __init__(self, server_host: str = "127.0.0.1", server_port: int = 5555):
        """
        Initialize the subscriber client.

        Parameters
        ----------
        server_host : str, optional
            Publisher server host address (default: "127.0.0.1")
        server_port : int, optional
            Publisher server port number (default: 5555)
        """
        super().__init__()
        self.server_address = f"tcp://{server_host}:{server_port}"

        # Client running flag
        self.running: EventType = Event()

        # Background worker process
        self.worker: Optional[Process] = None

        # Queue for received data
        self.data_queue: MPQueue = Queue()

        # Timer to process messages in main thread
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_message)

        # Start the background subscription process
        self.running.set()
        self.worker = Process(
            target=self.worker_loop, args=(self.server_address, self.running, self.data_queue), daemon=True
        )
        self.worker.start()
        self.timer.start(10)  # Check queue every 10ms

    @staticmethod
    def worker_loop(server_address: str, running: EventType, data_queue: MPQueue) -> None:
        """
        Worker loop executed in the child process.
        Initializes ZeroMQ and continuously receives messages from the publisher.

        Parameters
        ----------
        server_address : str
            ZeroMQ server address to connect to
        running : EventType
            Event flag to control the worker loop execution
        data_queue : MPQueue
            Queue for passing received data to the main process
        """
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(server_address)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages

        # Set up poller for non-blocking message reception
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        while running.is_set():
            # Poll for messages with 100ms timeout
            socks = dict(poller.poll(100))

            if socket in socks and socks[socket] == zmq.POLLIN:
                message = socket.recv_string()
                decoded_data = json.loads(message)

                # Decode all images in the received message
                for name, image_data in decoded_data["Image"].items():
                    decoded_data["Image"][name] = SubscriberClient.decode_image(image_data, name)

                # Put decoded data in queue for main thread processing
                if decoded_data is not None:
                    data_queue.put(decoded_data)

        # Cleanup on exit
        if socket:
            socket.close(linger=0)
        if context:
            context.term()

    @staticmethod
    def decode_image(encoded_data: Union[str, Dict[str, Any]], name: str) -> np.ndarray:
        """
        Decode a Base64-encoded JPEG image.

        Parameters
        ----------
        encoded_data : Union[str, dict]
            Base64-encoded image data. String for RGB images, dict for depth images
        name : str
            Image name (used to determine decoding method)

        Returns
        -------
        np.ndarray
            Decoded image array

        Raises
        ------
        ValueError
            If image type is unknown or decoding fails
        """
        if "rgb" in name:
            # Decode RGB image
            if not isinstance(encoded_data, str):
                raise ValueError(f"Expected string for RGB image, got {type(encoded_data)}")
            decoded_bytes = base64.b64decode(encoded_data)
            decoded_data_array = np.frombuffer(decoded_bytes, dtype=np.uint8)
            image = cv2.imdecode(decoded_data_array, cv2.IMREAD_COLOR)
        elif "depth" in name:
            # Decode depth image (with additional depth range metadata)
            if not isinstance(encoded_data, dict):
                raise ValueError(f"Expected dict for depth image, got {type(encoded_data)}")
            if "image" not in encoded_data:
                raise ValueError("Depth image data missing 'image' key")
            decoded_bytes = base64.b64decode(encoded_data["image"])
            decoded_data_array = np.frombuffer(decoded_bytes, dtype=np.uint8)
            image = cv2.imdecode(decoded_data_array, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError("Invalid image data")

        if image is not None:
            # Flip image vertically to correct orientation
            return cv2.flip(image, 0)
        else:
            raise ValueError(f"Failed to decode image: {name}")

    def process_message(self) -> None:
        """
        Process any pending messages in the queue by emitting the data_received signal.
        This method should be called regularly from the main thread.
        """
        try:
            # Process all available messages in the queue
            while not self.data_queue.empty():
                message = self.data_queue.get_nowait()
                self.data_received.emit(message)
        except _queue.Empty:
            pass  # Queue is empty, continue normal operation

    def close(self) -> None:
        """
        Stop the worker process and clean up resources.
        """
        self.running.clear()
        self.timer.stop()

        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2.0)
            if self.worker.is_alive():
                self.worker.terminate()

        # Clear any pending data from the queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except _queue.Empty:
                break
