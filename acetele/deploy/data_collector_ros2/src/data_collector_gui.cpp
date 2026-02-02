#include "data_collector_gui.hpp"
#include <QDateTime>
#include <QDir>
#include <QHeaderView>
#include <QScrollBar>

DataCollectorWindow::DataCollectorWindow(std::shared_ptr<DataCollectorNode> node, QWidget *parent)
    : QMainWindow(parent), node_(node)
{
    this->setWindowTitle("Data Collector");
    this->resize(1400, 900);

    setupUI();

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &DataCollectorWindow::updateView);
    timer_->start(33); // ~30 FPS
}

DataCollectorWindow::~DataCollectorWindow()
{
}

void DataCollectorWindow::setupUI()
{
    // Global Style
    this->setStyleSheet(
        "QMainWindow { background-color: #1e1e1e; }"
        "QLabel { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }"
        "QLineEdit { background-color: #2d2d2d; color: #ffffff; border: 1px solid #3e3e3e; padding: 6px; border-radius: 4px; font-size: 12px; }"
        "QPushButton { background-color: #007acc; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; font-size: 12px; }"
        "QPushButton:hover { background-color: #0098ff; }"
        "QPushButton:disabled { background-color: #444444; color: #888888; }"
        "QTextEdit { background-color: #1e1e1e; color: #dcdcdc; border: 1px solid #3e3e3e; font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; }"
        "QTableWidget { background-color: #252526; color: #e0e0e0; border: 1px solid #3e3e3e; gridline-color: #3e3e3e; }"
        "QHeaderView::section { background-color: #333333; color: #e0e0e0; padding: 4px; border: none; }"
    );

    QWidget *centralWidget = new QWidget(this);
    this->setCentralWidget(centralWidget);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setSpacing(10);
    mainLayout->setContentsMargins(15, 15, 15, 15);

    // --- Header ---
    QHBoxLayout *headerLayout = new QHBoxLayout();
    title_label_ = new QLabel("ACETele Data Collector");
    title_label_->setStyleSheet("font-size: 20px; font-weight: bold; color: #007acc;");
    time_label_ = new QLabel("00:00:00");
    time_label_->setStyleSheet("font-size: 14px; color: #aaaaaa;");

    headerLayout->addWidget(title_label_);
    headerLayout->addStretch();
    headerLayout->addWidget(time_label_);
    mainLayout->addLayout(headerLayout);

    // --- Main Content Area ---
    QHBoxLayout *contentLayout = new QHBoxLayout();

    // Left: RGB Stream
    QFrame *leftFrame = new QFrame();
    leftFrame->setStyleSheet("background-color: #252526; border-radius: 4px; border: 1px solid #3e3e3e;");
    QVBoxLayout *rgbLayout = new QVBoxLayout(leftFrame);

    QLabel *rgbTitle = new QLabel("RGB Stream");
    rgbTitle->setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 5px; border: none; background: transparent;");
    rgb_view_ = new QLabel();
    rgb_view_->setStyleSheet("background-color: #000; border: 1px solid #444; border-radius: 0px;");
    rgb_view_->setMinimumSize(640, 480);
    rgb_view_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    rgb_view_->setAlignment(Qt::AlignCenter);
    rgb_view_->setText("Waiting for RGB...");

    // Recording Status Overlay (Label on top of RGB view via layout trick or separate label)
    record_status_label_ = new QLabel("");
    record_status_label_->setStyleSheet("color: #ff4444; font-weight: bold; font-size: 14px; border: none; background: transparent;");

    rgbLayout->addWidget(rgbTitle);
    rgbLayout->addWidget(record_status_label_);
    rgbLayout->addWidget(rgb_view_, 1); // Stretch

    contentLayout->addWidget(leftFrame, 2); // 2/3 width

    // Right: Depth & Status
    QFrame *rightFrame = new QFrame();
    rightFrame->setStyleSheet("background-color: #252526; border-radius: 4px; border: 1px solid #3e3e3e;");
    QVBoxLayout *rightLayout = new QVBoxLayout(rightFrame);

    // Depth
    QLabel *depthTitle = new QLabel("Depth Stream");
    depthTitle->setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 5px; border: none; background: transparent;");
    depth_view_ = new QLabel();
    depth_view_->setStyleSheet("background-color: #000; border: 1px solid #444; border-radius: 0px;");
    depth_view_->setMinimumSize(320, 240);
    depth_view_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    depth_view_->setAlignment(Qt::AlignCenter);
    depth_view_->setText("Waiting for Depth...");

    rightLayout->addWidget(depthTitle);
    rightLayout->addWidget(depth_view_, 3); // Weight 3

    // Status Table
    QLabel *statusTitle = new QLabel("System Status");
    statusTitle->setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px; margin-bottom: 5px; border: none; background: transparent;");

    status_table_ = new QTableWidget(0, 2);
    status_table_->setHorizontalHeaderLabels({"Topic", "Status"});
    status_table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    status_table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    status_table_->verticalHeader()->setVisible(false);
    status_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    status_table_->setSelectionMode(QAbstractItemView::NoSelection);
    status_table_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    // Removed fixed height to allow alignment
    // status_table_->setFixedHeight(200);

    rightLayout->addWidget(statusTitle);
    rightLayout->addWidget(status_table_, 2); // Weight 2

    contentLayout->addWidget(rightFrame, 1); // 1/3 width
    mainLayout->addLayout(contentLayout, 1); // Stretch vertical

    // --- Bottom Area ---
    QHBoxLayout *bottomLayout = new QHBoxLayout();

    // Metadata
    QVBoxLayout *metaLayout = new QVBoxLayout();
    QLabel *metaTitle = new QLabel("RealSense Metadata");
    metaTitle->setStyleSheet("font-size: 12px; font-weight: bold;");
    metadata_view_ = new QTextEdit();
    metadata_view_->setReadOnly(true);
    metadata_view_->setMaximumHeight(150);

    metaLayout->addWidget(metaTitle);
    metaLayout->addWidget(metadata_view_);
    bottomLayout->addLayout(metaLayout, 3); // 75% width

    // Controls
    QVBoxLayout *controlLayout = new QVBoxLayout();
    controlLayout->setContentsMargins(10, 0, 0, 0);
    QLabel *controlTitle = new QLabel("Data Recording");
    controlTitle->setStyleSheet("font-size: 12px; font-weight: bold;");

    path_input_ = new QLineEdit();
    path_input_->setPlaceholderText("Output Path...");
    QString home = QDir::homePath();
    path_input_->setText(home + "/data_collection");

    record_btn_ = new QPushButton("Start Recording");
    record_btn_->setMinimumHeight(40);
    connect(record_btn_, &QPushButton::clicked, this, &DataCollectorWindow::toggleRecording);

    controlLayout->addWidget(controlTitle);
    controlLayout->addWidget(path_input_);
    controlLayout->addWidget(record_btn_);
    controlLayout->addStretch();

    bottomLayout->addLayout(controlLayout, 1); // 25% width

    mainLayout->addLayout(bottomLayout);
}

void DataCollectorWindow::toggleRecording()
{
    if (node_->is_recording()) {
        node_->stop_recording();
        record_btn_->setText("Start Recording");
        record_btn_->setStyleSheet("background-color: #007acc;"); // Blue
        path_input_->setEnabled(true);
        record_status_label_->setText("");
    } else {
        std::string base_path = path_input_->text().toStdString();
        auto now = QDateTime::currentDateTime();
        std::string session_path = base_path + "/" + now.toString("yyyyMMdd_HHmmss").toStdString();

        node_->start_recording(session_path);
        record_btn_->setText("Stop Recording");
        record_btn_->setStyleSheet("background-color: #cc0000;"); // Red
        path_input_->setEnabled(false);
        record_status_label_->setText("● RECORDING");
    }
}

void DataCollectorWindow::updateView()
{
    // Update Time
    time_label_->setText(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss"));

    // Get Images
    cv::Mat color, depth;
    node_->get_latest_images(color, depth);

    if (!color.empty()) {
        cv::Mat rgb;
        cv::cvtColor(color, rgb, cv::COLOR_BGR2RGB);
        // Resize to fit label while maintaining aspect ratio
        QImage qimg = matToQImage(rgb);
        QPixmap pix = QPixmap::fromImage(qimg);

        // Scale nicely
        int w = rgb_view_->width();
        int h = rgb_view_->height();
        rgb_view_->setPixmap(pix.scaled(w, h, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    if (!depth.empty()) {
        cv::Mat depth_vis;
        cv::normalize(depth, depth_vis, 0, 255, cv::NORM_MINMAX, CV_8U);

        QImage qimg = matToQImage(depth_vis);
        QPixmap pix = QPixmap::fromImage(qimg);

        int w = depth_view_->width();
        int h = depth_view_->height();
        depth_view_->setPixmap(pix.scaled(w, h, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    // Update Info
    updateStatusTable(node_->get_status_info());
    updateMetadata(node_->get_metadata_json());

    // Update Recording Status Text
    if (node_->is_recording()) {
        size_t count = node_->get_recorded_frame_count();
        record_status_label_->setText(QString("● RECORDING - Frames: %1").arg(count));
    }
}

void DataCollectorWindow::updateStatusTable(const std::map<std::string, std::string>& status)
{
    status_table_->setRowCount(status.size());
    int row = 0;
    for (auto const& [key, val] : status) {
        QTableWidgetItem *keyItem = new QTableWidgetItem(QString::fromStdString(key));
        QTableWidgetItem *valItem = new QTableWidgetItem(QString::fromStdString(val));

        if (val.find("ONLINE") == 0) {
            // Parse latency for color coding
            double latency = 0.0;
            size_t start = val.find('(');
            size_t end = val.find(" ms)");
            bool has_latency = (start != std::string::npos && end != std::string::npos);

            if (has_latency) {
                try {
                    latency = std::stod(val.substr(start + 1, end - start - 1));
                } catch (...) {}
            }

            if (!has_latency || latency < 50.0) {
                valItem->setForeground(QBrush(QColor("#4caf50"))); // Green
            } else if (latency < 150.0) {
                valItem->setForeground(QBrush(QColor("#ff9800"))); // Orange
            } else {
                valItem->setForeground(QBrush(QColor("#ff5252"))); // Light Red
            }
        } else {
            valItem->setForeground(QBrush(QColor("#f44336"))); // Red
        }

        status_table_->setItem(row, 0, keyItem);
        status_table_->setItem(row, 1, valItem);
        row++;
    }
}

void DataCollectorWindow::updateMetadata(const std::string& json_str)
{
    if (json_str.empty()) {
        metadata_view_->setText("No Metadata");
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(json_str));
    if (doc.isNull()) {
        metadata_view_->setText(QString::fromStdString(json_str)); // Fallback
    } else {
        metadata_view_->setText(doc.toJson(QJsonDocument::Indented));
    }
}

QImage DataCollectorWindow::matToQImage(const cv::Mat& mat)
{
    return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).copy();
}
