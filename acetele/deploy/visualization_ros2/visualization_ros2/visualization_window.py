from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from typing import Any, Mapping

import cv2
import rclpy
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QGuiApplication, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from rclpy.executors import MultiThreadedExecutor
from visualization_ros2.visualization_node import VisualizationNode


class AspectRatioLabel(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._original_pixmap = QPixmap()
        self.setMinimumSize(300, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "AspectRatioLabel {"
            "   border: 2px solid #cccccc;"
            "   background-color: #f8f8f8;"
            "   border-radius: 5px;"
            "}"
        )

    def setPixmap(self, pixmap: QPixmap) -> None:
        self._original_pixmap = pixmap
        self._update_pixmap()

    def resizeEvent(self, event) -> None:
        self._update_pixmap()
        super().resizeEvent(event)

    def _update_pixmap(self) -> None:
        if not self._original_pixmap.isNull():
            scaled = self._original_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            super().setPixmap(scaled)


class VisualizationWindow(QMainWindow):
    def __init__(self, node: VisualizationNode, parent=None) -> None:
        super().__init__(parent)
        self._node = node
        self._timer: QTimer
        self._base_width = 1920.0
        self._base_height = 1080.0
        self.main_layout: QHBoxLayout
        self.left_layout: QVBoxLayout
        self.grid_layout: QGridLayout
        self.right_layout: QVBoxLayout
        self.image_title: QLabel
        self.front_rgb_text: QLabel
        self.wrist_rgb_text: QLabel
        self.front_depth_text: QLabel
        self.wrist_depth_text: QLabel
        self.front_rgb_view: AspectRatioLabel
        self.front_depth_view: AspectRatioLabel
        self.wrist_rgb_view: AspectRatioLabel
        self.wrist_depth_view: AspectRatioLabel
        self.data_title: QLabel
        self.status_label: QLabel
        self.status_header: QLabel
        self.status_table: QTableWidget
        self.meta_header: QLabel
        self.metadata_tabs: QTabWidget
        self.front_metadata_view: QTextEdit
        self.wrist_metadata_view: QTextEdit
        self.arm_state_view: QTextEdit

        self.setWindowTitle("ACETele Visualization")
        screen = QGuiApplication.primaryScreen()
        if screen:
            available_geometry = screen.availableGeometry()
            self.resize(available_geometry.size())
            self._base_width = float(available_geometry.width())
            self._base_height = float(available_geometry.height())
        else:
            self.resize(1920, 1080)

        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update_view)
        self._timer.start(33)

    def _setup_ui(self) -> None:
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(15, 15, 15, 15)

        left_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)
        self.image_title = QLabel("Camera Views (Front & Wrist)")
        self.image_title.setStyleSheet("font-size: 18pt; font-weight: bold; margin-bottom: 10px; color: #2c3e50;")
        self.left_layout.addWidget(self.image_title)

        grid_widget = QWidget()
        self.grid_layout = QGridLayout(grid_widget)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(5, 5, 5, 5)

        front_rgb_container = QWidget()
        front_rgb_layout = QVBoxLayout(front_rgb_container)
        front_rgb_layout.setSpacing(2)
        front_rgb_layout.setContentsMargins(5, 5, 5, 5)
        self.front_rgb_view = AspectRatioLabel()
        self.front_rgb_view.setPixmap(self.create_sample_image(QColor(52, 152, 219), "Front RGB"))
        self.front_rgb_text = QLabel("Front RGB")
        self.front_rgb_text.setAlignment(Qt.AlignCenter)
        self.front_rgb_text.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt;")
        front_rgb_layout.addWidget(self.front_rgb_view)
        front_rgb_layout.addWidget(self.front_rgb_text)
        self.grid_layout.addWidget(front_rgb_container, 0, 0)

        wrist_rgb_container = QWidget()
        wrist_rgb_layout = QVBoxLayout(wrist_rgb_container)
        wrist_rgb_layout.setSpacing(2)
        wrist_rgb_layout.setContentsMargins(5, 5, 5, 5)
        self.wrist_rgb_view = AspectRatioLabel()
        self.wrist_rgb_view.setPixmap(self.create_sample_image(QColor(46, 204, 113), "Wrist RGB"))
        self.wrist_rgb_text = QLabel("Wrist RGB")
        self.wrist_rgb_text.setAlignment(Qt.AlignCenter)
        self.wrist_rgb_text.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt;")
        wrist_rgb_layout.addWidget(self.wrist_rgb_view)
        wrist_rgb_layout.addWidget(self.wrist_rgb_text)
        self.grid_layout.addWidget(wrist_rgb_container, 0, 1)

        front_depth_container = QWidget()
        front_depth_layout = QVBoxLayout(front_depth_container)
        front_depth_layout.setSpacing(2)
        front_depth_layout.setContentsMargins(5, 5, 5, 5)
        self.front_depth_view = AspectRatioLabel()
        self.front_depth_view.setPixmap(self.create_sample_image(QColor(155, 89, 182), "Front Depth"))
        self.front_depth_text = QLabel("Front Depth")
        self.front_depth_text.setAlignment(Qt.AlignCenter)
        self.front_depth_text.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt;")
        front_depth_layout.addWidget(self.front_depth_view)
        front_depth_layout.addWidget(self.front_depth_text)
        self.grid_layout.addWidget(front_depth_container, 1, 0)

        wrist_depth_container = QWidget()
        wrist_depth_layout = QVBoxLayout(wrist_depth_container)
        wrist_depth_layout.setSpacing(2)
        wrist_depth_layout.setContentsMargins(5, 5, 5, 5)
        self.wrist_depth_view = AspectRatioLabel()
        self.wrist_depth_view.setPixmap(self.create_sample_image(QColor(241, 196, 15), "Wrist Depth"))
        self.wrist_depth_text = QLabel("Wrist Depth")
        self.wrist_depth_text.setAlignment(Qt.AlignCenter)
        self.wrist_depth_text.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt;")
        wrist_depth_layout.addWidget(self.wrist_depth_view)
        wrist_depth_layout.addWidget(self.wrist_depth_text)
        self.grid_layout.addWidget(wrist_depth_container, 1, 1)

        self.grid_layout.setRowStretch(0, 1)
        self.grid_layout.setRowStretch(1, 1)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)

        self.left_layout.addWidget(grid_widget, 1)
        self.main_layout.addWidget(left_widget, 4)

        right_widget = QWidget()
        self.right_layout = QVBoxLayout(right_widget)
        title_layout = QHBoxLayout()
        self.data_title = QLabel("System Info")
        self.data_title.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px; color: #2c3e50;")
        self.status_label = QLabel(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.status_label.setStyleSheet("color: #666666; font-style: italic;")
        title_layout.addWidget(self.data_title)
        title_layout.addStretch()
        title_layout.addWidget(self.status_label)
        self.right_layout.addLayout(title_layout)

        self.status_header = QLabel("Topic Status")
        self.status_header.setStyleSheet("font-weight: bold; color: #34495e; margin-top: 5px;")
        self.right_layout.addWidget(self.status_header)

        self.status_table = QTableWidget(0, 2)
        self.status_table.setHorizontalHeaderLabels(["Topic", "Status"])
        self.status_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.status_table.verticalHeader().setVisible(False)
        self.status_table.setAlternatingRowColors(True)
        self.status_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.status_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.status_table.setFocusPolicy(Qt.NoFocus)
        self.status_table.setStyleSheet(
            "QTableWidget {"
            "   gridline-color: #e0e0e0;"
            "   font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;"
            "   font-size: 12px;"
            "   border: 1px solid #d0d0d0;"
            "   border-radius: 6px;"
            "   background-color: #ffffff;"
            "   selection-background-color: #e8f0fe;"
            "   selection-color: #2c3e50;"
            "}"
            "QTableWidget::item {"
            "   padding: 10px;"
            "   border-bottom: 1px solid #f0f0f0;"
            "   color: #2c3e50;"
            "}"
            "QHeaderView::section {"
            "   background-color: #2c3e50;"
            "   color: #ffffff;"
            "   padding: 10px;"
            "   border: none;"
            "   font-weight: 600;"
            "   font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;"
            "   text-transform: uppercase;"
            "   letter-spacing: 1px;"
            "}"
        )
        self.right_layout.addWidget(self.status_table, 2)

        self.meta_header = QLabel("Metadata")
        self.meta_header.setStyleSheet("font-weight: bold; color: #34495e; margin-top: 10px;")
        self.right_layout.addWidget(self.meta_header)

        self.metadata_tabs = QTabWidget()
        self.metadata_tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #d0d0d0; border-radius: 6px; background-color: #fafafa; }"
            "QTabBar::tab { "
            "   background: #e0e0e0; "
            "   color: #555555; "
            "   padding: 8px 16px; "
            "   margin-right: 2px; "
            "   border-top-left-radius: 6px; "
            "   border-top-right-radius: 6px; "
            "   font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;"
            "}"
            "QTabBar::tab:selected { background: #fafafa; color: #2c3e50; font-weight: bold; "
            "border-bottom: 2px solid #2c3e50; }"
            "QTabBar::tab:hover { background: #ececec; }"
        )

        meta_style = (
            "QTextEdit {"
            "   border: none;"
            "   background-color: #fafafa;"
            "   color: #2c3e50;"
            "   font-family: 'JetBrains Mono', 'Fira Code', 'Roboto Mono', 'Consolas', monospace;"
            "   font-size: 12px;"
            "   padding: 10px;"
            "   line-height: 1.5;"
            "}"
        )

        self.front_metadata_view = QTextEdit()
        self.front_metadata_view.setReadOnly(True)
        self.front_metadata_view.setStyleSheet(meta_style)
        self.metadata_tabs.addTab(self.front_metadata_view, "Front Camera")

        self.wrist_metadata_view = QTextEdit()
        self.wrist_metadata_view.setReadOnly(True)
        self.wrist_metadata_view.setStyleSheet(meta_style)
        self.metadata_tabs.addTab(self.wrist_metadata_view, "Wrist Camera")

        self.arm_state_view = QTextEdit()
        self.arm_state_view.setReadOnly(True)
        self.arm_state_view.setStyleSheet(meta_style)
        self.metadata_tabs.addTab(self.arm_state_view, "Arm State")

        self.right_layout.addWidget(self.metadata_tabs, 3)
        self.main_layout.addWidget(right_widget, 1)

    def resizeEvent(self, event) -> None:
        self.update_fonts()
        super().resizeEvent(event)

    def update_fonts(self) -> None:
        scale_w = float(self.width()) / self._base_width
        scale_h = float(self.height()) / self._base_height
        scale = min(scale_w, scale_h)
        if scale < 0.5:
            scale = 0.5

        self.main_layout.setSpacing(int(15 * scale))
        self.main_layout.setContentsMargins(
            int(15 * scale),
            int(15 * scale),
            int(15 * scale),
            int(15 * scale),
        )
        self.grid_layout.setSpacing(int(10 * scale))
        self.grid_layout.setContentsMargins(
            int(5 * scale),
            int(5 * scale),
            int(5 * scale),
            int(5 * scale),
        )
        self.left_layout.setSpacing(int(6 * scale))
        self.right_layout.setSpacing(int(6 * scale))

        self.image_title.setStyleSheet(
            f"font-size: {int(18 * scale)}pt; font-weight: bold; margin-bottom: {int(10 * scale)}px; color: #2c3e50;"
        )

        view_label_style = f"font-weight: bold; color: #2c3e50; font-size: {int(10 * scale)}pt;"
        self.front_rgb_text.setStyleSheet(view_label_style)
        self.wrist_rgb_text.setStyleSheet(view_label_style)
        self.front_depth_text.setStyleSheet(view_label_style)
        self.wrist_depth_text.setStyleSheet(view_label_style)

        self.data_title.setStyleSheet(
            f"font-size: {int(16 * scale)}pt; font-weight: bold; margin-bottom: {int(10 * scale)}px; color: #2c3e50;"
        )
        self.status_label.setStyleSheet(f"color: #666666; font-style: italic; font-size: {int(10 * scale)}pt;")

        header_style = (
            f"font-weight: bold; color: #34495e; margin-top: {int(5 * scale)}px; font-size: {int(11 * scale)}pt;"
        )
        self.status_header.setStyleSheet(header_style)
        self.meta_header.setStyleSheet(header_style)

        self.status_table.verticalHeader().setDefaultSectionSize(int(36 * scale))
        self.status_table.setStyleSheet(
            "QTableWidget {"
            "   gridline-color: #e0e0e0;"
            "   font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;"
            f"   font-size: {int(12 * scale)}px;"
            "   border: 1px solid #d0d0d0;"
            "   border-radius: 6px;"
            "   background-color: #ffffff;"
            "   selection-background-color: #e8f0fe;"
            "   selection-color: #2c3e50;"
            "}"
            "QTableWidget::item {"
            f"   padding: {int(10 * scale)}px;"
            "   border-bottom: 1px solid #f0f0f0;"
            "   color: #2c3e50;"
            "}"
            "QHeaderView::section {"
            "   background-color: #2c3e50;"
            "   color: #ffffff;"
            f"   padding: {int(10 * scale)}px;"
            "   border: none;"
            "   font-weight: 600;"
            "   font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;"
            "   text-transform: uppercase;"
            "   letter-spacing: 1px;"
            f"   font-size: {int(12 * scale)}px;"
            "}"
        )

        meta_style = (
            "QTextEdit {"
            "   border: none;"
            "   background-color: #fafafa;"
            "   color: #2c3e50;"
            "   font-family: 'JetBrains Mono', 'Fira Code', 'Roboto Mono', 'Consolas', monospace;"
            f"   font-size: {int(12 * scale)}px;"
            f"   padding: {int(10 * scale)}px;"
            "   line-height: 1.5;"
            "}"
        )
        self.front_metadata_view.setStyleSheet(meta_style)
        self.wrist_metadata_view.setStyleSheet(meta_style)
        self.arm_state_view.setStyleSheet(meta_style)

        self.metadata_tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #d0d0d0; border-radius: 6px; background-color: #fafafa; }"
            "QTabBar::tab { "
            "   background: #e0e0e0; "
            "   color: #555555; "
            f"   padding: {int(8 * scale)}px {int(12 * scale)}px; "
            "   margin-right: 2px; "
            "   border-top-left-radius: 6px; "
            "   border-top-right-radius: 6px; "
            "   font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;"
            f"   font-size: {int(12 * scale)}px; "
            f"   min-width: {int(80 * scale)}px; "
            "}"
            "QTabBar::tab:selected { background: #fafafa; color: #2c3e50; font-weight: bold; "
            "border-bottom: 2px solid #2c3e50; }"
            "QTabBar::tab:hover { background: #ececec; }"
        )
        self.metadata_tabs.style().unpolish(self.metadata_tabs)
        self.metadata_tabs.style().polish(self.metadata_tabs)

    def update_view(self) -> None:
        front_color, front_depth, wrist_color, wrist_depth = self._node.get_latest_images()

        if front_color is not None:
            rgb = cv2.cvtColor(front_color, cv2.COLOR_BGR2RGB)
            self.front_rgb_view.setPixmap(QPixmap.fromImage(self.mat_to_qimage(rgb)))

        if wrist_color is not None:
            rgb = cv2.cvtColor(wrist_color, cv2.COLOR_BGR2RGB)
            self.wrist_rgb_view.setPixmap(QPixmap.fromImage(self.mat_to_qimage(rgb)))

        if front_depth is not None:
            depth_vis = cv2.normalize(front_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
            self.front_depth_view.setPixmap(QPixmap.fromImage(self.mat_to_qimage(depth_vis)))

        if wrist_depth is not None:
            depth_vis = cv2.normalize(wrist_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
            self.wrist_depth_view.setPixmap(QPixmap.fromImage(self.mat_to_qimage(depth_vis)))

        self.update_status_table(self._node.get_status_info())
        front_meta, wrist_meta = self._node.get_latest_metadata()
        self.update_metadata(front_meta, wrist_meta)

        arm_state = self._node.get_latest_arm_state()
        if len(arm_state.name) == 0:
            self.arm_state_view.setText("No Arm State")
        else:
            text = f"Timestamp: {arm_state.header.stamp.sec}.{arm_state.header.stamp.nanosec}\n\n"
            for i, joint_name in enumerate(arm_state.name):
                text += f"{joint_name}:\n"
                if i < len(arm_state.position):
                    text += f"  Pos: {arm_state.position[i]:.4f}\n"
                if i < len(arm_state.velocity):
                    text += f"  Vel: {arm_state.velocity[i]:.4f}\n"
                if i < len(arm_state.effort):
                    text += f"  Eff: {arm_state.effort[i]:.4f}\n"
                text += "\n"
            self.arm_state_view.setText(text)

        self.status_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def update_status_table(self, status: Mapping[str, str]) -> None:
        self.status_table.setRowCount(len(status))
        row = 0
        for key, val in status.items():
            key_item = QTableWidgetItem(str(key))
            val_item = QTableWidgetItem(str(val))
            key_item.setTextAlignment(Qt.AlignCenter)
            val_item.setTextAlignment(Qt.AlignCenter)

            if val.startswith("ONLINE"):
                latency = 0.0
                start = val.find("(")
                end = val.find(" ms)")
                has_latency = False
                if start != -1 and end != -1:
                    try:
                        latency = float(val[start + 1 : end])
                        has_latency = True
                    except Exception:
                        has_latency = False
                if has_latency:
                    if latency > 150.0:
                        val_item.setBackground(QColor(255, 200, 200))
                    elif latency > 60.0:
                        val_item.setBackground(QColor(255, 228, 181))
                    else:
                        val_item.setBackground(QColor(220, 255, 220))
                else:
                    val_item.setBackground(QColor(220, 255, 220))
                val_item.setForeground(QBrush(QColor(0, 0, 0)))
            elif val.startswith("OFFLINE"):
                val_item.setBackground(QColor(255, 200, 200))
                val_item.setForeground(QBrush(QColor(0, 0, 0)))

            self.status_table.setItem(row, 0, key_item)
            self.status_table.setItem(row, 1, val_item)
            row += 1

    def update_metadata(self, front_json: str, wrist_json: str) -> None:
        if len(front_json) == 0:
            self.front_metadata_view.setText("No Front Metadata")
        else:
            try:
                parsed = json.loads(front_json)
                self.front_metadata_view.setText(json.dumps(parsed, indent=2))
            except Exception:
                self.front_metadata_view.setText(front_json)

        if len(wrist_json) == 0:
            self.wrist_metadata_view.setText("No Wrist Metadata")
        else:
            try:
                parsed = json.loads(wrist_json)
                self.wrist_metadata_view.setText(json.dumps(parsed, indent=2))
            except Exception:
                self.wrist_metadata_view.setText(wrist_json)

    def mat_to_qimage(self, mat: Any) -> QImage:
        h, w, c = mat.shape
        bytes_per_line = c * w
        return QImage(mat.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    def create_sample_image(self, color: QColor, text: str) -> QPixmap:
        width = 800
        height = 450
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(240, 240, 240))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(color)
        painter.setPen(QColor(100, 100, 100))
        painter.drawRoundedRect(10, 10, width - 20, height - 20, 10, 10)
        painter.setPen(Qt.white)
        font = QFont(painter.font())
        font.setPointSize(20)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(0, 0, width, height, Qt.AlignCenter, text)
        font.setPointSize(12)
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(0, 30, width, height, Qt.AlignCenter, "Waiting for stream...")
        painter.end()
        return pixmap


def main(args=None) -> int:
    rclpy.init(args=args)
    node = VisualizationNode()
    app = QApplication(sys.argv)
    window = VisualizationWindow(node)
    window.show()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

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
