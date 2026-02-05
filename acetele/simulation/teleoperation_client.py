import sys
from datetime import datetime

import cv2
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial.transform import Rotation as R

from acetele.simulation.network import SubscriberClient


class AspectRatioLabel(QLabel):
    """
    Custom QLabel that maintains aspect ratio when scaling images.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            AspectRatioLabel {
                border: 2px solid #cccccc;
                background-color: #f8f8f8;
                border-radius: 5px;
            }
        """)
        self._pixmap = None

    def setPixmap(self, pixmap):
        """
        Set the pixmap to display while maintaining aspect ratio.

        Parameters
        ----------
        pixmap : QPixmap
            The pixmap to display
        """
        self._pixmap = pixmap
        self.updatePixmap()

    def updatePixmap(self):
        """Update the displayed pixmap with proper aspect ratio scaling."""
        if self._pixmap and not self._pixmap.isNull():
            scaled_pixmap = self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """
        Handle resize events to update the pixmap scaling.

        Parameters
        ----------
        event : QResizeEvent
            The resize event
        """
        self.updatePixmap()
        super().resizeEvent(event)


class TeleOperationApp(QMainWindow):
    """
    Main application window for teleoperation data visualization.
    Displays camera feeds and flight status data in real-time.
    """

    def __init__(self):
        super().__init__()
        self.initUI()
        self.initData()
        # Initialize ZMQ client for receiving data
        self.zmq_client = SubscriberClient()
        self.zmq_client.data_received.connect(self.updateData)

    def initUI(self):
        """Initialize the user interface components."""
        self.setWindowTitle("空中操作数据采集")
        self.setGeometry(100, 100, 1600, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Left panel - image display area
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Image grid title
        image_title = QLabel("相机视图")
        image_title.setStyleSheet("font-size: 18pt; font-weight: bold; margin-bottom: 10px; color: #2c3e50;")
        left_layout.addWidget(image_title)

        # Create 2x2 image grid
        grid_widget = QWidget()
        self.grid_layout = QGridLayout(grid_widget)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(5, 5, 5, 5)

        # Create containers for images with titles
        self.camera_view = {
            "front_camera_rgb": {
                "title": "前视相机(RGB)",
                "original_image": self.createSampleImage(QColor(52, 152, 219)),
                "display_label": AspectRatioLabel(),
            },
            "wrist_camera_rgb": {
                "title": "腕部相机(RGB)",
                "original_image": self.createSampleImage(QColor(46, 204, 113)),
                "display_label": AspectRatioLabel(),
            },
            "front_camera_depth": {
                "title": "前视相机(Depth)",
                "original_image": self.createSampleImage(QColor(155, 89, 182)),
                "display_label": AspectRatioLabel(),
            },
            "wrist_camera_depth": {
                "title": "腕部相机(Depth)",
                "original_image": self.createSampleImage(QColor(241, 196, 15)),
                "display_label": AspectRatioLabel(),
            },
        }

        # Add camera views to grid layout
        for i, (name, camera_info) in enumerate(self.camera_view.items()):
            image_container = QWidget()
            container_layout = QVBoxLayout(image_container)
            container_layout.setSpacing(2)
            container_layout.setContentsMargins(5, 5, 5, 5)

            # Image display label
            image_label = camera_info["display_label"]
            image_label.setPixmap(camera_info["original_image"])

            # Image title label
            title_label = QLabel(camera_info["title"])
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt; margin-top: 2px;")
            title_label.setMaximumHeight(20)

            container_layout.addWidget(image_label)
            container_layout.addWidget(title_label)

            self.grid_layout.addWidget(image_container, i // 2, i % 2)

        # Set grid layout row/column stretch factors
        self.grid_layout.setRowStretch(0, 1)
        self.grid_layout.setRowStretch(1, 1)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)

        left_layout.addWidget(grid_widget, 1)  # Image area is stretchable

        # Right panel - data display area
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Create horizontal layout for title and status label
        title_layout = QHBoxLayout()

        # Data table title
        data_title = QLabel("飞行状态数据")
        data_title.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px; color: #2c3e50;")

        # Status label
        self.status_label = QLabel("系统就绪")
        self.status_label.setStyleSheet("color: #666666; font-style: italic;")

        title_layout.addWidget(data_title)
        title_layout.addStretch()
        title_layout.addWidget(self.status_label)

        right_layout.addLayout(title_layout)

        # Create data table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["参数名称", "当前值", "状态"])

        # Configure table properties
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)  # Disable editing
        self.table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                font-size: 11px;
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)

        right_layout.addWidget(self.table, 1)  # Table area is stretchable

        # Add panels to main layout
        main_layout.addWidget(left_widget, 4)  # Left panel
        main_layout.addWidget(right_widget, 1)  # Right panel

    def createSampleImage(self, color: QColor) -> QPixmap:
        """
        Create a sample placeholder image.

        Parameters
        ----------
        color : QColor
            Background color for the placeholder image

        Returns
        -------
        QPixmap
            Generated placeholder image
        """
        width, height = 800, 450
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(240, 240, 240))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.setBrush(color)
        painter.setPen(QColor(100, 100, 100))
        painter.drawRoundedRect(10, 10, width - 20, height - 20, 10, 10)

        # Draw text
        painter.setPen(Qt.GlobalColor.white)
        font = painter.font()
        font.setPointSize(20)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(0, 0, width, height, Qt.AlignmentFlag.AlignCenter, "当前无图像")

        # Draw resolution info
        font.setPointSize(12)
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(0, 30, width, height, Qt.AlignmentFlag.AlignCenter, "1280×720")

        painter.end()
        return pixmap

    def initData(self):
        """Initialize data structure with default values."""
        self.data_items = {
            "位置X": {"value": 0.0, "unit": " m", "status": "正常"},
            "位置Y": {"value": 0.0, "unit": " m", "status": "正常"},
            "位置Z": {"value": 0.0, "unit": " m", "status": "正常"},
            "横滚角": {"value": 0.0, "unit": "°", "status": "正常"},
            "俯仰角": {"value": 0.0, "unit": "°", "status": "正常"},
            "偏航角": {"value": 0.0, "unit": "°", "status": "正常"},
            "线速度X": {"value": 0.0, "unit": " m/s", "status": "正常"},
            "线速度Y": {"value": 0.0, "unit": " m/s", "status": "正常"},
            "线速度Z": {"value": 0.0, "unit": " m/s", "status": "正常"},
            "角速度Roll": {"value": 0.0, "unit": " rad/s", "status": "正常"},
            "角速度Pitch": {"value": 0.0, "unit": " rad/s", "status": "正常"},
            "角速度Yaw": {"value": 0.0, "unit": " rad/s", "status": "正常"},
            "桨叶转速1": {"value": 0.0, "unit": " rad/s", "status": "正常"},
            "桨叶转速2": {"value": 0.0, "unit": " rad/s", "status": "正常"},
            "桨叶转速3": {"value": 0.0, "unit": " rad/s", "status": "正常"},
            "桨叶转速4": {"value": 0.0, "unit": " rad/s", "status": "正常"},
            "关节位置1": {"value": 0.0, "unit": " rad", "status": "正常"},
            "关节位置2": {"value": 0.0, "unit": " rad", "status": "正常"},
            "关节位置3": {"value": 0.0, "unit": " rad", "status": "正常"},
            "关节位置4": {"value": 0.0, "unit": " rad", "status": "正常"},
            "关节位置5": {"value": 0.0, "unit": " rad", "status": "正常"},
            "关节力矩1": {"value": 0.0, "unit": " N·m", "status": "正常"},
            "关节力矩2": {"value": 0.0, "unit": " N·m", "status": "正常"},
            "关节力矩3": {"value": 0.0, "unit": " N·m", "status": "正常"},
            "关节力矩4": {"value": 0.0, "unit": " N·m", "status": "正常"},
            "关节力矩5": {"value": 0.0, "unit": " N·m", "status": "正常"},
            "命令线速度X": {"value": 0.0, "unit": " m/s", "status": "正常"},
            "命令线速度Y": {"value": 0.0, "unit": " m/s", "status": "正常"},
            "命令线速度Z": {"value": 0.0, "unit": " m/s", "status": "正常"},
            "命令角速度Yaw": {"value": 0.0, "unit": " rad/s", "status": "正常"},
        }

        self.updateTable()

    def updateTable(self):
        """Update the table with current data values."""
        self.table.setRowCount(len(self.data_items))

        for row, (name, item) in enumerate(self.data_items.items()):
            # Parameter name
            name_item = QTableWidgetItem(name)
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, name_item)

            # Current value
            if isinstance(item["value"], (int, float)):
                value_text = f"{item['value']:.2f}{item['unit']}"
            else:
                value_text = str(item["value"])
            value_item = QTableWidgetItem(value_text)
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, value_item)

            # Status
            status = item["status"]
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if status == "正常":
                status_item.setBackground(QColor(220, 255, 220))  # Green
            elif status == "警告":
                status_item.setBackground(QColor(255, 255, 200))  # Yellow
            elif status == "异常":
                status_item.setBackground(QColor(255, 200, 200))  # Red
            else:
                status_item.setBackground(QColor(240, 240, 240))  # Default gray

            self.table.setItem(row, 2, status_item)

    @Slot(dict)
    def updateData(self, data: dict):
        """
        Update the UI with new data received from ZMQ.

        Parameters
        ----------
        data : dict
            Dictionary containing image and telemetry data
        """
        if not data["Image"]:
            return

        # Update camera images
        for i, (name, image) in enumerate(data["Image"].items()):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            if pixmap.isNull():
                pixmap = self.createSampleImage(QColor(200, 200, 200))

            self.camera_view[name]["original_image"] = pixmap
            self.camera_view[name]["display_label"].setPixmap(pixmap)

        if not data["Data"]:
            return

        # Define axis labels
        linear_axis = ["X", "Y", "Z"]
        rotation_axis_en = ["Roll", "Pitch", "Yaw"]
        rotation_axis_cn = ["横滚", "俯仰", "偏航"]

        # Update position data
        for name, value in zip(linear_axis, data["Data"]["relative_position"]):
            self.data_items[f"位置{name}"]["value"] = value

        # Update orientation data (convert quaternion to Euler angles)
        rotation = R.from_quat(quat=data["Data"]["orientation"], scalar_first=True)  # xyzw -> wxyz
        euler_angle = rotation.as_euler("xyz", degrees=True)
        for i, (name, value) in enumerate(zip(rotation_axis_cn, euler_angle)):
            self.data_items[f"{name}角"]["value"] = value

        # Update linear velocity
        for name, value in zip(linear_axis, data["Data"]["lin_vel_b"]):
            self.data_items[f"线速度{name}"]["value"] = value

        # Update angular velocity
        for name, value in zip(rotation_axis_en, data["Data"]["ang_vel_b"]):
            self.data_items[f"角速度{name}"]["value"] = value

        # Update rotor speeds
        for i, value in enumerate(data["Data"]["rotor_velocity"]):
            self.data_items[f"桨叶转速{i + 1}"]["value"] = value

        # Update joint positions
        for i, value in enumerate(data["Data"]["servo_position"]):
            self.data_items[f"关节位置{i + 1}"]["value"] = value

        # Update joint torques
        for i, value in enumerate(data["Data"]["servo_torque"]):
            self.data_items[f"关节力矩{i + 1}"]["value"] = value

        # Update command data
        for name, value in zip(linear_axis, data["Data"]["command"][:3]):
            self.data_items[f"命令线速度{name}"]["value"] = value
        self.data_items["命令角速度Yaw"]["value"] = data["Data"]["command"][-1]

        # Refresh table and update status
        self.updateTable()
        self.status_label.setText(f"数据已更新 - {datetime.now().strftime('%H:%M:%S')}")

    def resizeEvent(self, event):
        """
        Handle window resize events.

        Parameters
        ----------
        event : QResizeEvent
            The resize event
        """
        super().resizeEvent(event)


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = TeleOperationApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
