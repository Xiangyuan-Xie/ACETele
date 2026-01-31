# import sys # Unused
# import cv2
import os

import numpy as np
from PySide6.QtCore import QDateTime, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class DataCollectorWindow(QMainWindow):
    def __init__(self, node, parent=None):
        super(DataCollectorWindow, self).__init__(parent)
        self.node = node
        self.setWindowTitle("Data Collector")
        self.resize(1400, 900)

        self.setup_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_view)
        self.timer.start(33)  # ~30 FPS

    def setup_ui(self):
        # Global Style
        self.setStyleSheet(
            """
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            QLineEdit { background-color: #2d2d2d; color: #ffffff; border: 1px solid #3e3e3e; padding: 6px; border-radius: 4px; font-size: 12px; }
            QPushButton { background-color: #007acc; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; font-size: 12px; }
            QPushButton:hover { background-color: #0098ff; }
            QPushButton:disabled { background-color: #444444; color: #888888; }
            QTextEdit { background-color: #1e1e1e; color: #dcdcdc; border: 1px solid #3e3e3e; font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; }
            QTableWidget { background-color: #252526; color: #e0e0e0; border: 1px solid #3e3e3e; gridline-color: #3e3e3e; }
            QHeaderView::section { background-color: #333333; color: #e0e0e0; padding: 4px; border: none; }
        """
        )

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # --- Header ---
        header_layout = QHBoxLayout()
        self.title_label = QLabel("ACETele Data Collector")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #007acc;")
        self.time_label = QLabel("00:00:00")
        self.time_label.setStyleSheet("font-size: 14px; color: #aaaaaa;")

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.time_label)
        main_layout.addLayout(header_layout)

        # --- Main Content Area ---
        content_layout = QHBoxLayout()

        # Left: RGB Stream
        left_frame = QFrame()
        left_frame.setStyleSheet("background-color: #252526; border-radius: 4px; border: 1px solid #3e3e3e;")
        rgb_layout = QVBoxLayout(left_frame)

        rgb_title = QLabel("RGB Stream")
        rgb_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; margin-bottom: 5px; border: none; background: transparent;"
        )

        self.rgb_view = QLabel()
        self.rgb_view.setStyleSheet("background-color: #000; border: 1px solid #444; border-radius: 0px;")
        self.rgb_view.setMinimumSize(640, 480)
        self.rgb_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.rgb_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rgb_view.setText("Waiting for RGB...")

        # Recording Status Label
        self.record_status_label = QLabel("")
        self.record_status_label.setStyleSheet(
            "color: #ff4444; font-weight: bold; font-size: 14px; border: none; background: transparent;"
        )

        rgb_layout.addWidget(rgb_title)
        rgb_layout.addWidget(self.record_status_label)
        rgb_layout.addWidget(self.rgb_view, 1)  # Stretch

        content_layout.addWidget(left_frame, 2)  # 2/3 width

        # Right: Depth & Status
        right_frame = QFrame()
        right_frame.setStyleSheet("background-color: #252526; border-radius: 4px; border: 1px solid #3e3e3e;")
        right_layout = QVBoxLayout(right_frame)

        # Depth
        depth_title = QLabel("Depth Stream")
        depth_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; margin-bottom: 5px; border: none; background: transparent;"
        )
        self.depth_view = QLabel()
        self.depth_view.setStyleSheet("background-color: #000; border: 1px solid #444; border-radius: 0px;")
        self.depth_view.setMinimumSize(320, 240)
        self.depth_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.depth_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_view.setText("Waiting for Depth...")

        right_layout.addWidget(depth_title)
        right_layout.addWidget(self.depth_view, 3)  # Weight 3

        # Status Table
        status_title = QLabel("System Status")
        status_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; margin-top: 10px; margin-bottom: 5px; border: none; background: transparent;"
        )

        self.status_table = QTableWidget(0, 2)
        self.status_table.setHorizontalHeaderLabels(["Topic", "Status"])
        self.status_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.status_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.status_table.verticalHeader().setVisible(False)
        self.status_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.status_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.status_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        right_layout.addWidget(status_title)
        right_layout.addWidget(self.status_table, 2)  # Weight 2

        content_layout.addWidget(right_frame, 1)  # 1/3 width
        main_layout.addLayout(content_layout, 1)  # Stretch vertical

        # --- Bottom Area ---
        bottom_layout = QHBoxLayout()

        # Metadata
        meta_layout = QVBoxLayout()
        meta_title = QLabel("RealSense Metadata")
        meta_title.setStyleSheet("font-size: 12px; font-weight: bold;")
        self.metadata_view = QTextEdit()
        self.metadata_view.setReadOnly(True)
        self.metadata_view.setMaximumHeight(150)

        meta_layout.addWidget(meta_title)
        meta_layout.addWidget(self.metadata_view)
        bottom_layout.addLayout(meta_layout, 3)  # 75% width

        # Controls
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(10, 0, 0, 0)
        control_title = QLabel("Data Recording")
        control_title.setStyleSheet("font-size: 12px; font-weight: bold;")

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Output Path...")
        self.path_input.setText(os.path.join(os.path.expanduser("~"), "data_collection"))

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setMinimumHeight(40)
        self.record_btn.clicked.connect(self.toggle_recording)

        control_layout.addWidget(control_title)
        control_layout.addWidget(self.path_input)
        control_layout.addWidget(self.record_btn)
        control_layout.addStretch()

        bottom_layout.addLayout(control_layout, 1)  # 25% width

        main_layout.addLayout(bottom_layout)

    def toggle_recording(self):
        if self.node.is_recording():
            self.node.stop_recording()
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("background-color: #007acc;")  # Blue
            self.path_input.setEnabled(True)
            self.record_status_label.setText("")
        else:
            base_path = self.path_input.text()
            now_str = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
            session_path = os.path.join(base_path, now_str)

            self.node.start_recording(session_path)
            self.record_btn.setText("Stop Recording")
            self.record_btn.setStyleSheet("background-color: #cc0000;")  # Red
            self.path_input.setEnabled(False)
            self.record_status_label.setText("● RECORDING")

    def update_view(self):
        # Update Time
        self.time_label.setText(QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"))

        # Get Images
        color, depth = self.node.get_latest_images()

        if color is not None:
            # Manual conversion (BGR -> RGB)
            # Assuming color is numpy array (H, W, 3)
            if len(color.shape) == 3 and color.shape[2] == 3:
                rgb = color[..., ::-1].copy()  # Ensure contiguous for QImage
                ch = 3
                fmt = QImage.Format.Format_RGB888
            else:
                rgb = color.copy()
                ch = 1
                fmt = QImage.Format.Format_Grayscale8

            h, w = rgb.shape[:2]
            bytes_per_line = ch * w

            qimg = QImage(rgb.data, w, h, bytes_per_line, fmt)
            pix = QPixmap.fromImage(qimg)

            # Scale to label size
            self.rgb_view.setPixmap(
                pix.scaled(
                    self.rgb_view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )
            self.rgb_view.setText("")

        if depth is not None:
            # Manual normalization for visualization
            min_val = np.min(depth)
            max_val = np.max(depth)

            if max_val > min_val:
                # Normalize to 0-255
                depth_norm = ((depth.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)

            h, w = depth_norm.shape
            bytes_per_line = w
            qimg = QImage(depth_norm.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg)

            self.depth_view.setPixmap(
                pix.scaled(
                    self.depth_view.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.depth_view.setText("")

        # Update Status
        status_info = self.node.get_status_info()
        self.update_status_table(status_info)

        # Update Metadata
        metadata = self.node.get_metadata_json()
        if metadata:
            self.metadata_view.setText(metadata)

    def update_status_table(self, status):
        self.status_table.setRowCount(len(status))
        for i, (key, value) in enumerate(status.items()):
            self.status_table.setItem(i, 0, QTableWidgetItem(key))
            self.status_table.setItem(i, 1, QTableWidgetItem(value))
