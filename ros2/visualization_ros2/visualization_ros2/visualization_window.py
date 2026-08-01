"""Qt operator monitor for camera streams, arm state, and topic health."""

from __future__ import annotations

import json
import math
import sys
import threading
from datetime import datetime
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import rclpy
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QGuiApplication, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from rclpy.executors import MultiThreadedExecutor
from visualization_ros2.visualization_node import VisualizationNode

topic_labels = {
    "front_color": "Front RGB",
    "front_depth": "Front depth",
    "wrist_color": "Wrist RGB",
    "wrist_depth": "Wrist depth",
    "front_metadata": "Front metadata",
    "wrist_metadata": "Wrist metadata",
    "arm_state": "Arm state",
}

app_stylesheet = """
QMainWindow {
    background: #eef1ef;
}
QWidget {
    color: #1d292d;
    font-family: "Inter", "Noto Sans", "Segoe UI", sans-serif;
    font-size: 13px;
    letter-spacing: 0px;
}
QWidget#topBar {
    background: #1d272b;
    border-bottom: 3px solid #2f9d78;
}
QLabel#brandLabel {
    color: #ffffff;
    font-size: 20px;
    font-weight: 700;
}
QLabel#productLabel {
    color: #aeb9bc;
    font-size: 12px;
}
QLabel#clockLabel {
    color: #d6dddf;
    font-family: "JetBrains Mono", "Roboto Mono", monospace;
    font-size: 12px;
}
QLabel#overallHealth {
    border-radius: 4px;
    padding: 5px 9px;
    font-size: 11px;
    font-weight: 700;
}
QLabel#overallHealth[health="online"] {
    background: #dcefe7;
    color: #176246;
}
QLabel#overallHealth[health="degraded"] {
    background: #fff0cc;
    color: #80540b;
}
QLabel#overallHealth[health="offline"] {
    background: #f6dede;
    color: #8c2d2d;
}
QLabel#workspaceTitle {
    color: #152126;
    font-size: 17px;
    font-weight: 700;
}
QLabel#sectionTitle {
    color: #334247;
    font-size: 12px;
    font-weight: 700;
}
QLabel#sectionValue {
    color: #69777b;
    font-family: "JetBrains Mono", "Roboto Mono", monospace;
    font-size: 11px;
}
QFrame#cameraPanel {
    background: #161d20;
    border: 1px solid #3a4549;
    border-radius: 6px;
}
QWidget#cameraHeader {
    background: #222d31;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}
QLabel#cameraTitle {
    color: #f2f5f4;
    font-size: 12px;
    font-weight: 650;
}
QLabel#streamBadge {
    border-radius: 3px;
    padding: 3px 6px;
    font-size: 10px;
    font-weight: 700;
}
QLabel#streamBadge[streamState="online"] {
    background: #2f9d78;
    color: #ffffff;
}
QLabel#streamBadge[streamState="offline"] {
    background: #b64f4f;
    color: #ffffff;
}
QLabel#streamBadge[streamState="waiting"] {
    background: #59666a;
    color: #eef2f1;
}
QWidget#telemetryPane {
    background: #ffffff;
    border-left: 1px solid #d2d9d7;
}
QTableWidget {
    background: #ffffff;
    alternate-background-color: #f5f7f6;
    border: 1px solid #d6dddb;
    border-radius: 5px;
    gridline-color: #e2e7e5;
    selection-background-color: #dbe9f5;
    selection-color: #172327;
}
QTableWidget::item {
    padding: 6px;
}
QHeaderView::section {
    background: #e8edeb;
    color: #445257;
    border: none;
    border-right: 1px solid #d4dcda;
    border-bottom: 1px solid #ccd5d2;
    padding: 7px;
    font-size: 11px;
    font-weight: 700;
}
QTabWidget::pane {
    background: #ffffff;
    border: 1px solid #d6dddb;
    border-radius: 5px;
}
QTabBar::tab {
    background: #e7ecea;
    color: #526065;
    border: 1px solid #d2dad7;
    border-bottom: none;
    padding: 7px 12px;
    margin-right: 2px;
    min-width: 74px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #176b90;
    font-weight: 700;
}
QPlainTextEdit {
    background: #ffffff;
    color: #354348;
    border: none;
    padding: 8px;
    font-family: "JetBrains Mono", "Roboto Mono", "Consolas", monospace;
    font-size: 11px;
}
QSplitter::handle {
    background: #d5dcda;
}
QSplitter::handle:horizontal {
    width: 1px;
}
QSplitter::handle:vertical {
    height: 5px;
}
"""


class AspectRatioLabel(QLabel):
    """Image viewport that preserves source aspect ratio during resize."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._original_pixmap = QPixmap()
        self.setMinimumSize(220, 124)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #111719; border: none;")

    def setPixmap(self, pixmap: QPixmap) -> None:
        """Store the source pixmap and display an aspect-preserving copy."""

        self._original_pixmap = pixmap
        self._update_pixmap()

    def resizeEvent(self, event) -> None:
        """Rescale from the source image rather than compounding prior scaling."""

        self._update_pixmap()
        super().resizeEvent(event)

    def _update_pixmap(self) -> None:
        if self._original_pixmap.isNull():
            return
        scaled = self._original_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        super().setPixmap(scaled)


class CameraPanel(QFrame):
    """Stable camera viewport with an explicit live/offline status badge."""

    def __init__(self, title: str, placeholder: QPixmap, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("cameraPanel")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget(self)
        header.setObjectName("cameraHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(11, 7, 9, 7)
        header_layout.setSpacing(8)

        title_label = QLabel(title, header)
        title_label.setObjectName("cameraTitle")
        self.status_badge = QLabel("WAITING", header)
        self.status_badge.setObjectName("streamBadge")
        self.status_badge.setProperty("streamState", "waiting")
        self.status_badge.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.status_badge)

        self.viewport = AspectRatioLabel(self)
        self.viewport.setPixmap(placeholder)
        layout.addWidget(header)
        layout.addWidget(self.viewport, 1)

    def set_frame(self, pixmap: QPixmap) -> None:
        """Display the newest decoded frame."""

        self.viewport.setPixmap(pixmap)

    def set_stream_status(self, online: bool) -> None:
        """Update the compact stream badge without rebuilding the panel."""

        state = "online" if online else "offline"
        text = "LIVE" if online else "OFFLINE"
        if self.status_badge.property("streamState") == state:
            return
        self.status_badge.setText(text)
        self.status_badge.setProperty("streamState", state)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)


class VisualizationWindow(QMainWindow):
    """Dense operator workspace with cameras prioritized over telemetry tables."""

    def __init__(self, node: VisualizationNode, parent=None) -> None:
        super().__init__(parent)
        self._node = node
        self._timer: QTimer

        self.setWindowTitle("ACETele Operator Monitor")
        self.setMinimumSize(1180, 720)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(1680, round(available.width() * 0.94)),
                min(1000, round(available.height() * 0.92)),
            )
        else:
            self.resize(1600, 900)

        self.setStyleSheet(app_stylesheet)
        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update_view)
        self._timer.start(33)

    def _setup_ui(self) -> None:
        """Compose one resizable camera workspace beside dense telemetry."""

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_top_bar())

        self.workspace_splitter = QSplitter(Qt.Horizontal, central)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.addWidget(self._build_camera_workspace())
        self.workspace_splitter.addWidget(self._build_telemetry_pane())
        self.workspace_splitter.setStretchFactor(0, 7)
        self.workspace_splitter.setStretchFactor(1, 3)
        self.workspace_splitter.setSizes([1120, 480])
        root_layout.addWidget(self.workspace_splitter, 1)

    def _build_top_bar(self) -> QWidget:
        """Build stable global health and clock indicators."""

        top_bar = QWidget(self)
        top_bar.setObjectName("topBar")
        top_bar.setFixedHeight(66)
        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(12)

        brand = QLabel("ACETele", top_bar)
        brand.setObjectName("brandLabel")
        product = QLabel("Operator Monitor", top_bar)
        product.setObjectName("productLabel")
        self.overall_health = QLabel("NO DATA", top_bar)
        self.overall_health.setObjectName("overallHealth")
        self.overall_health.setProperty("health", "offline")
        self.clock_label = QLabel(top_bar)
        self.clock_label.setObjectName("clockLabel")
        self.clock_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(brand)
        layout.addWidget(product)
        layout.addStretch(1)
        layout.addWidget(self.overall_health)
        layout.addWidget(self.clock_label)
        return top_bar

    def _build_camera_workspace(self) -> QWidget:
        """Prioritize front RGB while keeping all secondary views visible."""

        workspace = QWidget(self)
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(16, 14, 14, 16)
        layout.setSpacing(10)

        title = QLabel("Camera feeds", workspace)
        title.setObjectName("workspaceTitle")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.front_rgb_panel = CameraPanel(
            "Front RGB",
            self._create_placeholder("FRONT RGB", QColor("#2f789d")),
            workspace,
        )
        self.wrist_rgb_panel = CameraPanel(
            "Wrist RGB",
            self._create_placeholder("WRIST RGB", QColor("#2f9d78")),
            workspace,
        )
        self.front_depth_panel = CameraPanel(
            "Front depth",
            self._create_placeholder("FRONT DEPTH", QColor("#8b5fa7")),
            workspace,
        )
        self.wrist_depth_panel = CameraPanel(
            "Wrist depth",
            self._create_placeholder("WRIST DEPTH", QColor("#c4882d")),
            workspace,
        )

        grid.addWidget(self.front_rgb_panel, 0, 0, 1, 3)
        grid.addWidget(self.wrist_rgb_panel, 1, 0)
        grid.addWidget(self.front_depth_panel, 1, 1)
        grid.addWidget(self.wrist_depth_panel, 1, 2)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 1)
        for column in range(3):
            grid.setColumnStretch(column, 1)

        layout.addLayout(grid, 1)
        return workspace

    def _build_telemetry_pane(self) -> QWidget:
        """Stack health, joint state, and metadata in a resizable side pane."""

        pane = QWidget(self)
        pane.setObjectName("telemetryPane")
        pane.setMinimumWidth(390)
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title = QLabel("Telemetry", pane)
        title.setObjectName("workspaceTitle")
        self.stream_summary = QLabel("0 / 7 online", pane)
        self.stream_summary.setObjectName("sectionValue")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.stream_summary)
        layout.addLayout(title_row)

        details = QSplitter(Qt.Vertical, pane)
        details.setChildrenCollapsible(False)
        details.addWidget(self._build_status_section())
        details.addWidget(self._build_joint_section())
        details.addWidget(self._build_metadata_section())
        details.setStretchFactor(0, 3)
        details.setStretchFactor(1, 3)
        details.setStretchFactor(2, 2)
        details.setSizes([280, 260, 220])
        layout.addWidget(details, 1)
        return pane

    def _build_status_section(self) -> QWidget:
        """Build the fixed-column stream health table."""

        section = QWidget(self)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("STREAM HEALTH", section)
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.status_table = QTableWidget(0, 3, section)
        self.status_table.setHorizontalHeaderLabels(["Stream", "State", "Latency"])
        self._configure_table(self.status_table)
        header = self.status_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(self.status_table, 1)
        return section

    def _build_joint_section(self) -> QWidget:
        """Build a compact table whose rows follow incoming joint names."""

        section = QWidget(self)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        title = QLabel("JOINT STATE", section)
        title.setObjectName("sectionTitle")
        self.joint_sample = QLabel("WAITING", section)
        self.joint_sample.setObjectName("sectionValue")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.joint_sample)
        layout.addLayout(title_row)

        self.joint_table = QTableWidget(0, 4, section)
        self.joint_table.setHorizontalHeaderLabels(
            ["Joint", "Position", "Velocity", "Effort"]
        )
        self._configure_table(self.joint_table)
        header = self.joint_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        layout.addWidget(self.joint_table, 1)
        return section

    def _build_metadata_section(self) -> QWidget:
        """Separate verbose camera metadata from high-frequency telemetry."""

        section = QWidget(self)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("CAMERA METADATA", section)
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.metadata_tabs = QTabWidget(section)
        self.front_metadata_view = self._metadata_view("No front metadata")
        self.wrist_metadata_view = self._metadata_view("No wrist metadata")
        self.metadata_tabs.addTab(self.front_metadata_view, "Front")
        self.metadata_tabs.addTab(self.wrist_metadata_view, "Wrist")
        layout.addWidget(self.metadata_tabs, 1)
        return section

    @staticmethod
    def _configure_table(table: QTableWidget) -> None:
        """Apply the common read-only operator-table behavior."""

        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(31)
        table.setAlternatingRowColors(True)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setFocusPolicy(Qt.NoFocus)
        table.setShowGrid(False)

    @staticmethod
    def _metadata_view(placeholder: str) -> QPlainTextEdit:
        """Create a read-only metadata view with an explicit empty state."""

        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(placeholder)
        return view

    def update_view(self) -> None:
        """Refresh all widgets from one set of node-owned snapshots."""

        front_color, front_depth, wrist_color, wrist_depth = (
            self._node.get_latest_images()
        )

        if front_color is not None:
            self.front_rgb_panel.set_frame(self._color_pixmap(front_color))
        if wrist_color is not None:
            self.wrist_rgb_panel.set_frame(self._color_pixmap(wrist_color))
        if front_depth is not None:
            self.front_depth_panel.set_frame(self._depth_pixmap(front_depth))
        if wrist_depth is not None:
            self.wrist_depth_panel.set_frame(self._depth_pixmap(wrist_depth))

        status = self._node.get_status_info()
        self.update_status_table(status)
        self._update_camera_status(status)

        front_meta, wrist_meta = self._node.get_latest_metadata()
        self.update_metadata(front_meta, wrist_meta)
        self.update_joint_table(self._node.get_latest_arm_state())
        self.clock_label.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    def update_status_table(self, status: Mapping[str, str]) -> None:
        """Render stream health and latency without changing table geometry."""

        keys = list(topic_labels)
        keys.extend(sorted(set(status) - set(topic_labels)))
        self.status_table.setRowCount(len(keys))
        online_count = 0

        for row, key in enumerate(keys):
            value = status.get(key, "OFFLINE")
            online = value.startswith("ONLINE")
            latency = self._parse_latency(value)
            online_count += int(online)

            name_item = self._table_item(topic_labels.get(key, key))
            state_item = self._table_item(
                "Online" if online else "Offline",
                align=Qt.AlignCenter,
            )
            latency_item = self._table_item(
                "--" if latency is None else f"{latency:.1f} ms",
                align=Qt.AlignRight | Qt.AlignVCenter,
            )

            if not online:
                state_item.setForeground(QBrush(QColor("#a03636")))
                state_item.setBackground(QBrush(QColor("#f8e7e7")))
                latency_item.setForeground(QBrush(QColor("#8b9699")))
            elif latency is not None and latency > 150.0:
                state_item.setForeground(QBrush(QColor("#a03636")))
                latency_item.setForeground(QBrush(QColor("#a03636")))
                latency_item.setBackground(QBrush(QColor("#f8e7e7")))
            elif latency is not None and latency > 60.0:
                state_item.setForeground(QBrush(QColor("#80540b")))
                latency_item.setForeground(QBrush(QColor("#80540b")))
                latency_item.setBackground(QBrush(QColor("#fff2d6")))
            else:
                state_item.setForeground(QBrush(QColor("#176246")))
                state_item.setBackground(QBrush(QColor("#e3f2eb")))

            self.status_table.setItem(row, 0, name_item)
            self.status_table.setItem(row, 1, state_item)
            self.status_table.setItem(row, 2, latency_item)

        total = len(keys)
        self.stream_summary.setText(f"{online_count} / {total} online")
        if online_count == total and total:
            health, text = "online", "ALL SYSTEMS"
        elif online_count:
            health, text = "degraded", "DEGRADED"
        else:
            health, text = "offline", "NO DATA"
        self.overall_health.setText(text)
        if self.overall_health.property("health") != health:
            self.overall_health.setProperty("health", health)
            self.overall_health.style().unpolish(self.overall_health)
            self.overall_health.style().polish(self.overall_health)

    def update_joint_table(self, arm_state: Any) -> None:
        """Render the latest named arm state in stable position/velocity/effort columns."""

        names = list(arm_state.name)
        self.joint_table.setRowCount(len(names))
        if not names:
            self.joint_sample.setText("WAITING")
            return

        stamp = arm_state.header.stamp
        self.joint_sample.setText(f"{stamp.sec}.{stamp.nanosec:09d}")
        columns = (arm_state.position, arm_state.velocity, arm_state.effort)
        for row, name in enumerate(names):
            self.joint_table.setItem(row, 0, self._table_item(name))
            for column, values in enumerate(columns, start=1):
                self.joint_table.setItem(
                    row,
                    column,
                    self._table_item(
                        self._format_joint_value(values, row),
                        align=Qt.AlignRight | Qt.AlignVCenter,
                    ),
                )

    def update_metadata(self, front_json: str, wrist_json: str) -> None:
        """Pretty-print camera metadata while preserving non-JSON diagnostics."""

        self._set_metadata(
            self.front_metadata_view,
            front_json,
            "No front metadata",
        )
        self._set_metadata(
            self.wrist_metadata_view,
            wrist_json,
            "No wrist metadata",
        )

    def _update_camera_status(self, status: Mapping[str, str]) -> None:
        """Map topic health directly onto each camera panel badge."""

        for key, panel in (
            ("front_color", self.front_rgb_panel),
            ("wrist_color", self.wrist_rgb_panel),
            ("front_depth", self.front_depth_panel),
            ("wrist_depth", self.wrist_depth_panel),
        ):
            panel.set_stream_status(status.get(key, "OFFLINE").startswith("ONLINE"))

    @staticmethod
    def _table_item(text: str, *, align=Qt.AlignLeft | Qt.AlignVCenter) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(align)
        return item

    @staticmethod
    def _parse_latency(value: str) -> float | None:
        if "(" not in value or "ms" not in value:
            return None
        try:
            return float(value.split("(", 1)[1].split("ms", 1)[0].strip())
        except ValueError:
            return None

    @staticmethod
    def _format_joint_value(values: Sequence[float], index: int) -> str:
        if index >= len(values):
            return "--"
        value = float(values[index])
        return f"{value:.4f}" if math.isfinite(value) else "INVALID"

    @staticmethod
    def _set_metadata(view: QPlainTextEdit, raw: str, empty_text: str) -> None:
        if not raw:
            text = empty_text
        else:
            try:
                text = json.dumps(json.loads(raw), indent=2, sort_keys=True)
            except (TypeError, ValueError):
                text = raw
        if view.toPlainText() != text:
            view.setPlainText(text)

    def _color_pixmap(self, image: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return QPixmap.fromImage(self.mat_to_qimage(rgb))

    def _depth_pixmap(self, depth: np.ndarray) -> QPixmap:
        """Render depth robustly without letting invalid pixels dominate contrast."""

        depth_values = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(depth_values) & (depth_values > 0.0)
        normalized = np.zeros(depth_values.shape, dtype=np.uint8)
        if np.any(valid):
            # Per-frame robust percentiles keep the actual depth scene visible while
            # preventing a few invalid extremes from flattening the color range.
            low, high = np.percentile(depth_values[valid], (2.0, 98.0))
            if high <= low:
                high = low + 1.0
            scaled = (depth_values - low) * (255.0 / (high - low))
            normalized[valid] = np.clip(scaled[valid], 0.0, 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        colored[~valid] = (20, 25, 27)
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return QPixmap.fromImage(self.mat_to_qimage(rgb))

    @staticmethod
    def mat_to_qimage(mat: np.ndarray) -> QImage:
        """Copy a contiguous RGB array into Qt-owned image memory."""

        image = np.ascontiguousarray(mat)
        height, width, channels = image.shape
        return QImage(
            image.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()

    @staticmethod
    def _create_placeholder(text: str, accent: QColor) -> QPixmap:
        """Create a fixed-aspect placeholder that cannot shift the camera grid."""

        width, height = 1280, 720
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("#111719"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(0, height // 2 - 2, width, 4, accent)

        title_font = QFont(painter.font())
        title_font.setPointSize(22)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#e4e9e8"))
        painter.drawText(0, 0, width, height - 22, Qt.AlignCenter, text)

        status_font = QFont(painter.font())
        status_font.setPointSize(11)
        status_font.setBold(False)
        painter.setFont(status_font)
        painter.setPen(QColor("#8f9b9e"))
        painter.drawText(0, 34, width, height, Qt.AlignCenter, "WAITING FOR STREAM")
        painter.end()
        return pixmap


def main(args=None) -> int:
    """Run ROS callbacks in one thread and the Qt event loop in the main thread."""

    rclpy.init(args=args)
    node = VisualizationNode()
    app = QApplication(sys.argv)
    window = VisualizationWindow(node)
    window.show()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    # Qt requires its event loop on the main thread. ROS therefore owns a dedicated
    # executor thread and communicates with widgets only through copied snapshots.
    ros_thread = threading.Thread(target=executor.spin, daemon=False)
    ros_thread.start()

    try:
        result = app.exec()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        if ros_thread.is_alive():
            ros_thread.join()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
