#include "data_collector_gui.hpp"
#include <QDateTime>
#include <QDir>
#include <QHeaderView>

// --- AspectRatioLabel Implementation ---

AspectRatioLabel::AspectRatioLabel(QWidget *parent)
    : QLabel(parent)
{
    this->setMinimumSize(300, 200);
    this->setAlignment(Qt::AlignCenter);
    this->setStyleSheet(
        "AspectRatioLabel {"
        "   border: 2px solid #cccccc;"
        "   background-color: #f8f8f8;"
        "   border-radius: 5px;"
        "}"
    );
}

void AspectRatioLabel::setPixmap(const QPixmap &pixmap)
{
    original_pixmap_ = pixmap;
    updatePixmap();
}

void AspectRatioLabel::updatePixmap()
{
    if (!original_pixmap_.isNull()) {
        QPixmap scaled = original_pixmap_.scaled(
            this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation
        );
        QLabel::setPixmap(scaled);
    }
}

void AspectRatioLabel::resizeEvent(QResizeEvent *event)
{
    updatePixmap();
    QLabel::resizeEvent(event);
}

// --- DataCollectorWindow Implementation ---

DataCollectorWindow::DataCollectorWindow(std::shared_ptr<DataCollectorNode> node, QWidget *parent)
    : QMainWindow(parent), node_(node)
{
    this->setWindowTitle("ACETele Data Collector");
    this->resize(1600, 900);

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
    // Main Window Style
    this->setStyleSheet("QMainWindow { background-color: #ffffff; }");

    QWidget *centralWidget = new QWidget(this);
    this->setCentralWidget(centralWidget);
    QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(15, 15, 15, 15);

    // --- Left Panel (Camera Views) ---
    QWidget *leftWidget = new QWidget();
    QVBoxLayout *leftLayout = new QVBoxLayout(leftWidget);

    QLabel *imageTitle = new QLabel("Camera Views (Front & Wrist)");
    imageTitle->setStyleSheet("font-size: 18pt; font-weight: bold; margin-bottom: 10px; color: #2c3e50;");
    leftLayout->addWidget(imageTitle);

    QWidget *gridWidget = new QWidget();
    QGridLayout *gridLayout = new QGridLayout(gridWidget);
    gridLayout->setSpacing(10);
    gridLayout->setContentsMargins(5, 5, 5, 5);

    // Grid Layout Strategy:
    // (0,0) Front RGB   (0,1) Wrist RGB
    // (1,0) Front Depth (1,1) Wrist Depth

    // 1. Front RGB (0,0)
    QWidget *frontRgbContainer = new QWidget();
    QVBoxLayout *frontRgbLayout = new QVBoxLayout(frontRgbContainer);
    frontRgbLayout->setSpacing(2);
    frontRgbLayout->setContentsMargins(5, 5, 5, 5);

    front_rgb_view_ = new AspectRatioLabel();
    front_rgb_view_->setPixmap(createSampleImage(QColor(52, 152, 219), "Front RGB"));
    QLabel *frontRgbText = new QLabel("Front RGB");
    frontRgbText->setAlignment(Qt::AlignCenter);
    frontRgbText->setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt;");
    frontRgbLayout->addWidget(front_rgb_view_);
    frontRgbLayout->addWidget(frontRgbText);
    gridLayout->addWidget(frontRgbContainer, 0, 0);

    // 2. Wrist RGB (0,1)
    QWidget *wristRgbContainer = new QWidget();
    QVBoxLayout *wristRgbLayout = new QVBoxLayout(wristRgbContainer);
    wristRgbLayout->setSpacing(2);
    wristRgbLayout->setContentsMargins(5, 5, 5, 5);

    wrist_rgb_view_ = new AspectRatioLabel();
    wrist_rgb_view_->setPixmap(createSampleImage(QColor(46, 204, 113), "Wrist RGB"));
    QLabel *wristRgbText = new QLabel("Wrist RGB");
    wristRgbText->setAlignment(Qt::AlignCenter);
    wristRgbText->setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt;");
    wristRgbLayout->addWidget(wrist_rgb_view_);
    wristRgbLayout->addWidget(wristRgbText);
    gridLayout->addWidget(wristRgbContainer, 0, 1);

    // 3. Front Depth (1,0)
    QWidget *frontDepthContainer = new QWidget();
    QVBoxLayout *frontDepthLayout = new QVBoxLayout(frontDepthContainer);
    frontDepthLayout->setSpacing(2);
    frontDepthLayout->setContentsMargins(5, 5, 5, 5);

    front_depth_view_ = new AspectRatioLabel();
    front_depth_view_->setPixmap(createSampleImage(QColor(155, 89, 182), "Front Depth"));
    QLabel *frontDepthText = new QLabel("Front Depth");
    frontDepthText->setAlignment(Qt::AlignCenter);
    frontDepthText->setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt;");
    frontDepthLayout->addWidget(front_depth_view_);
    frontDepthLayout->addWidget(frontDepthText);
    gridLayout->addWidget(frontDepthContainer, 1, 0);

    // 4. Wrist Depth (1,1)
    QWidget *wristDepthContainer = new QWidget();
    QVBoxLayout *wristDepthLayout = new QVBoxLayout(wristDepthContainer);
    wristDepthLayout->setSpacing(2);
    wristDepthLayout->setContentsMargins(5, 5, 5, 5);

    wrist_depth_view_ = new AspectRatioLabel();
    wrist_depth_view_->setPixmap(createSampleImage(QColor(241, 196, 15), "Wrist Depth"));
    QLabel *wristDepthText = new QLabel("Wrist Depth");
    wristDepthText->setAlignment(Qt::AlignCenter);
    wristDepthText->setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt;");
    wristDepthLayout->addWidget(wrist_depth_view_);
    wristDepthLayout->addWidget(wristDepthText);
    gridLayout->addWidget(wristDepthContainer, 1, 1);

    // Stretch factors
    gridLayout->setRowStretch(0, 1);
    gridLayout->setRowStretch(1, 1);
    gridLayout->setColumnStretch(0, 1);
    gridLayout->setColumnStretch(1, 1);

    leftLayout->addWidget(gridWidget, 1);
    mainLayout->addWidget(leftWidget, 4); // Left panel takes 4/5 width

    // --- Right Panel (Status & Controls) ---
    QWidget *rightWidget = new QWidget();
    QVBoxLayout *rightLayout = new QVBoxLayout(rightWidget);

    // Title & Status
    QHBoxLayout *titleLayout = new QHBoxLayout();
    QLabel *dataTitle = new QLabel("System Info");
    dataTitle->setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px; color: #2c3e50;");
    status_label_ = new QLabel(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss"));
    status_label_->setStyleSheet("color: #666666; font-style: italic;");
    titleLayout->addWidget(dataTitle);
    titleLayout->addStretch();
    titleLayout->addWidget(status_label_);
    rightLayout->addLayout(titleLayout);

    // Merged Information Section (Table for status + Metadata below)
    // We use a vertical splitter or just layout

    // Status Table
    QLabel *statusHeader = new QLabel("Topic Status");
    statusHeader->setStyleSheet("font-weight: bold; color: #34495e; margin-top: 5px;");
    rightLayout->addWidget(statusHeader);

    status_table_ = new QTableWidget(0, 2);
    status_table_->setHorizontalHeaderLabels({"Topic", "Status"});
    status_table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    status_table_->verticalHeader()->setVisible(false);
    status_table_->setAlternatingRowColors(true);
    status_table_->setSelectionMode(QAbstractItemView::NoSelection);
    status_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    status_table_->setStyleSheet(
        "QTableWidget {"
        "   gridline-color: #d0d0d0;"
        "   font-size: 11px;"
        "   border: 1px solid #cccccc;"
        "   border-radius: 5px;"
        "}"
        "QTableWidget::item {"
        "   padding: 8px;"
        "   border-bottom: 1px solid #f0f0f0;"
        "}"
        "QHeaderView::section {"
        "   background-color: #3498db;"
        "   color: white;"
        "   padding: 8px;"
        "   border: none;"
        "   font-weight: bold;"
        "}"
    );
    rightLayout->addWidget(status_table_, 2); // Takes 2/5 of available vertical space

    // Metadata View (Tabbed)
    QLabel *metaHeader = new QLabel("Metadata");
    metaHeader->setStyleSheet("font-weight: bold; color: #34495e; margin-top: 10px;");
    rightLayout->addWidget(metaHeader);

    metadata_tabs_ = new QTabWidget();
    metadata_tabs_->setStyleSheet(
        "QTabWidget::pane { border: 1px solid #cccccc; border-radius: 4px; }"
        "QTabBar::tab { background: #e0e0e0; padding: 5px 10px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }"
        "QTabBar::tab:selected { background: #f8f8f8; font-weight: bold; }"
    );

    // Front Metadata Tab
    front_metadata_view_ = new QTextEdit();
    front_metadata_view_->setReadOnly(true);
    front_metadata_view_->setStyleSheet(
        "QTextEdit {"
        "   border: none;"
        "   background-color: #f8f8f8;"
        "   color: #333333;"
        "   font-family: 'Consolas', monospace;"
        "   font-size: 11px;"
        "}"
    );
    metadata_tabs_->addTab(front_metadata_view_, "Front Camera");

    // Wrist Metadata Tab
    wrist_metadata_view_ = new QTextEdit();
    wrist_metadata_view_->setReadOnly(true);
    wrist_metadata_view_->setStyleSheet(
        "QTextEdit {"
        "   border: none;"
        "   background-color: #f8f8f8;"
        "   color: #333333;"
        "   font-family: 'Consolas', monospace;"
        "   font-size: 11px;"
        "}"
    );
    metadata_tabs_->addTab(wrist_metadata_view_, "Wrist Camera");

    rightLayout->addWidget(metadata_tabs_, 2); // Takes 2/5 of available vertical space

    // Controls Section (Bottom)
    QLabel *controlTitle = new QLabel("Data Recording");
    controlTitle->setStyleSheet("font-size: 14pt; font-weight: bold; margin-top: 20px; color: #2c3e50;");
    rightLayout->addWidget(controlTitle);

    path_input_ = new QLineEdit();
    path_input_->setPlaceholderText("Output Path...");
    path_input_->setText(QDir::homePath() + "/data_collection");
    path_input_->setStyleSheet("padding: 8px; border: 1px solid #cccccc; border-radius: 4px; color: #333333;");
    rightLayout->addWidget(path_input_);

    record_btn_ = new QPushButton("Start Recording");
    record_btn_->setMinimumHeight(40);
    record_btn_->setStyleSheet(
        "QPushButton {"
        "   background-color: #3498db;"
        "   color: white;"
        "   border: none;"
        "   border-radius: 4px;"
        "   font-weight: bold;"
        "   font-size: 14px;"
        "}"
        "QPushButton:hover { background-color: #2980b9; }"
        "QPushButton:disabled { background-color: #95a5a6; }"
    );
    connect(record_btn_, &QPushButton::clicked, this, &DataCollectorWindow::toggleRecording);
    rightLayout->addWidget(record_btn_);

    mainLayout->addWidget(rightWidget, 1); // Right panel takes 1/5 width
}

void DataCollectorWindow::toggleRecording()
{
    if (node_->is_recording()) {
        node_->stop_recording();
        record_btn_->setText("Start Recording");
        record_btn_->setStyleSheet(
            "QPushButton {"
            "   background-color: #3498db;"
            "   color: white;"
            "   border: none;"
            "   border-radius: 4px;"
            "   font-weight: bold;"
            "   font-size: 14px;"
            "}"
            "QPushButton:hover { background-color: #2980b9; }"
        );
        path_input_->setEnabled(true);
        status_label_->setText(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss"));
        status_label_->setStyleSheet("color: #666666; font-style: italic;");
    } else {
        std::string base_path = path_input_->text().toStdString();
        auto now = QDateTime::currentDateTime();
        std::string session_path = base_path + "/" + now.toString("yyyyMMdd_HHmmss").toStdString();

        node_->start_recording(session_path);
        record_btn_->setText("Stop Recording");
        record_btn_->setStyleSheet(
            "QPushButton {"
            "   background-color: #e74c3c;"
            "   color: white;"
            "   border: none;"
            "   border-radius: 4px;"
            "   font-weight: bold;"
            "   font-size: 14px;"
            "}"
            "QPushButton:hover { background-color: #c0392b; }"
        );
        path_input_->setEnabled(false);
        status_label_->setText("● Recording");
        status_label_->setStyleSheet("color: #e74c3c; font-weight: bold;");
    }
}

void DataCollectorWindow::updateView()
{
    // Get Images
    cv::Mat front_color, front_depth, wrist_color, wrist_depth;
    node_->get_latest_images(front_color, front_depth, wrist_color, wrist_depth);

    // Update Front RGB
    if (!front_color.empty()) {
        cv::Mat rgb;
        cv::cvtColor(front_color, rgb, cv::COLOR_BGR2RGB);
        QImage qimg = matToQImage(rgb);
        front_rgb_view_->setPixmap(QPixmap::fromImage(qimg));
    }

    // Update Wrist RGB
    if (!wrist_color.empty()) {
        cv::Mat rgb;
        cv::cvtColor(wrist_color, rgb, cv::COLOR_BGR2RGB);
        QImage qimg = matToQImage(rgb);
        wrist_rgb_view_->setPixmap(QPixmap::fromImage(qimg));
    }

    // Update Front Depth
    if (!front_depth.empty()) {
        cv::Mat depth_vis;
        cv::normalize(front_depth, depth_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
        // Changed to Grayscale per user request
        cv::cvtColor(depth_vis, depth_vis, cv::COLOR_GRAY2RGB);

        QImage qimg = matToQImage(depth_vis);
        front_depth_view_->setPixmap(QPixmap::fromImage(qimg));
    }

    // Update Wrist Depth
    if (!wrist_depth.empty()) {
        cv::Mat depth_vis;
        cv::normalize(wrist_depth, depth_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
        // Changed to Grayscale per user request
        cv::cvtColor(depth_vis, depth_vis, cv::COLOR_GRAY2RGB);

        QImage qimg = matToQImage(depth_vis);
        wrist_depth_view_->setPixmap(QPixmap::fromImage(qimg));
    }

    // Update Info
    updateStatusTable(node_->get_status_info());

    std::string front_meta, wrist_meta;
    node_->get_latest_metadata(front_meta, wrist_meta);
    updateMetadata(front_meta, wrist_meta);

    // Update Recording Status Info
    if (node_->is_recording()) {
        size_t count = node_->get_recorded_frame_count();
        status_label_->setText(QString("● Recording - Frames: %1").arg(count));
    } else {
        status_label_->setText(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss"));
    }
}

void DataCollectorWindow::updateStatusTable(const std::map<std::string, std::string>& status)
{
    status_table_->setRowCount(status.size());
    int row = 0;
    for (auto const& [key, val] : status) {
        QTableWidgetItem *keyItem = new QTableWidgetItem(QString::fromStdString(key));
        QTableWidgetItem *valItem = new QTableWidgetItem(QString::fromStdString(val));

        // Center alignment
        keyItem->setTextAlignment(Qt::AlignCenter);
        valItem->setTextAlignment(Qt::AlignCenter);

        // Status Color Logic
        if (val.find("ONLINE") == 0) {
            // Check for latency to determine color
            double latency = 0.0;
            size_t start = val.find("(");
            size_t end = val.find(" ms)");
            bool has_latency = false;

            if (start != std::string::npos && end != std::string::npos) {
                try {
                    latency = std::stod(val.substr(start + 1, end - start - 1));
                    has_latency = true;
                } catch (...) {}
            }

            if (has_latency) {
                if (latency > 150.0) {
                     // High latency - Red
                     valItem->setBackground(QColor(255, 200, 200));
                } else if (latency > 60.0) {
                     // Medium latency - Orange/Yellow
                     valItem->setBackground(QColor(255, 228, 181));
                } else {
                     // Low latency - Green
                     valItem->setBackground(QColor(220, 255, 220));
                }
            } else {
                // Default Green if online but no latency info (e.g. metadata)
                valItem->setBackground(QColor(220, 255, 220));
            }

            valItem->setForeground(QBrush(QColor(0, 0, 0)));
        } else if (val.find("OFFLINE") == 0) {
            valItem->setBackground(QColor(255, 200, 200)); // Light Red
            valItem->setForeground(QBrush(QColor(0, 0, 0)));
        }

        status_table_->setItem(row, 0, keyItem);
        status_table_->setItem(row, 1, valItem);
        row++;
    }
}

void DataCollectorWindow::updateMetadata(const std::string& front_json, const std::string& wrist_json)
{
    // Update Front Metadata
    if (front_json.empty()) {
        front_metadata_view_->setText("No Front Metadata");
    } else {
        QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(front_json));
        if (doc.isNull()) {
            front_metadata_view_->setText(QString::fromStdString(front_json));
        } else {
            front_metadata_view_->setText(doc.toJson(QJsonDocument::Indented));
        }
    }

    // Update Wrist Metadata
    if (wrist_json.empty()) {
        wrist_metadata_view_->setText("No Wrist Metadata");
    } else {
        QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(wrist_json));
        if (doc.isNull()) {
            wrist_metadata_view_->setText(QString::fromStdString(wrist_json));
        } else {
            wrist_metadata_view_->setText(doc.toJson(QJsonDocument::Indented));
        }
    }
}

QImage DataCollectorWindow::matToQImage(const cv::Mat& mat)
{
    return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).copy();
}

QPixmap DataCollectorWindow::createSampleImage(const QColor& color, const QString& text)
{
    int width = 800;
    int height = 450;
    QPixmap pixmap(width, height);
    pixmap.fill(QColor(240, 240, 240));

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);

    // Draw background
    painter.setBrush(color);
    painter.setPen(QColor(100, 100, 100));
    painter.drawRoundedRect(10, 10, width - 20, height - 20, 10, 10);

    // Draw text
    painter.setPen(Qt::white);
    QFont font = painter.font();
    font.setPointSize(20);
    font.setBold(true);
    painter.setFont(font);
    painter.drawText(0, 0, width, height, Qt::AlignCenter, text);

    // Resolution
    font.setPointSize(12);
    font.setBold(false);
    painter.setFont(font);
    painter.drawText(0, 30, width, height, Qt::AlignCenter, "Waiting for stream...");

    return pixmap;
}
