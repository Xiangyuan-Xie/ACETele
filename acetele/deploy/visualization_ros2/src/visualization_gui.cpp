#include "visualization_gui.hpp"
#include <QDateTime>
#include <QDir>
#include <QHeaderView>
#include <QResizeEvent>

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

// --- VisualizationWindow Implementation ---

VisualizationWindow::VisualizationWindow(std::shared_ptr<VisualizationNode> node, QWidget *parent)
    : QMainWindow(parent), node_(node)
{
    this->setWindowTitle("ACETele Visualization");
    this->resize(1600, 900);

    setupUI();

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &VisualizationWindow::updateView);
    timer_->start(33); // ~30 FPS
}

VisualizationWindow::~VisualizationWindow()
{
}

void VisualizationWindow::setupUI()
{
    // Main Window Style
    this->setStyleSheet("QMainWindow { background-color: #ffffff; }");

    QWidget *centralWidget = new QWidget(this);
    this->setCentralWidget(centralWidget);
    main_layout_ = new QHBoxLayout(centralWidget);
    main_layout_->setSpacing(15);
    main_layout_->setContentsMargins(15, 15, 15, 15);

    // --- Left Panel (Camera Views) ---
    QWidget *leftWidget = new QWidget();
    left_layout_ = new QVBoxLayout(leftWidget);

    image_title_ = new QLabel("Camera Views (Front & Wrist)");
    image_title_->setStyleSheet("font-size: 18pt; font-weight: bold; margin-bottom: 10px; color: #2c3e50;");
    left_layout_->addWidget(image_title_);

    QWidget *gridWidget = new QWidget();
    grid_layout_ = new QGridLayout(gridWidget);
    grid_layout_->setSpacing(10);
    grid_layout_->setContentsMargins(5, 5, 5, 5);

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
    front_rgb_text_ = new QLabel("Front RGB");
    front_rgb_text_->setAlignment(Qt::AlignCenter);
    front_rgb_text_->setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt;");
    frontRgbLayout->addWidget(front_rgb_view_);
    frontRgbLayout->addWidget(front_rgb_text_);
    grid_layout_->addWidget(frontRgbContainer, 0, 0);

    // 2. Wrist RGB (0,1)
    QWidget *wristRgbContainer = new QWidget();
    QVBoxLayout *wristRgbLayout = new QVBoxLayout(wristRgbContainer);
    wristRgbLayout->setSpacing(2);
    wristRgbLayout->setContentsMargins(5, 5, 5, 5);

    wrist_rgb_view_ = new AspectRatioLabel();
    wrist_rgb_view_->setPixmap(createSampleImage(QColor(46, 204, 113), "Wrist RGB"));
    wrist_rgb_text_ = new QLabel("Wrist RGB");
    wrist_rgb_text_->setAlignment(Qt::AlignCenter);
    wrist_rgb_text_->setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt;");
    wristRgbLayout->addWidget(wrist_rgb_view_);
    wristRgbLayout->addWidget(wrist_rgb_text_);
    grid_layout_->addWidget(wristRgbContainer, 0, 1);

    // 3. Front Depth (1,0)
    QWidget *frontDepthContainer = new QWidget();
    QVBoxLayout *frontDepthLayout = new QVBoxLayout(frontDepthContainer);
    frontDepthLayout->setSpacing(2);
    frontDepthLayout->setContentsMargins(5, 5, 5, 5);

    front_depth_view_ = new AspectRatioLabel();
    front_depth_view_->setPixmap(createSampleImage(QColor(155, 89, 182), "Front Depth"));
    front_depth_text_ = new QLabel("Front Depth");
    front_depth_text_->setAlignment(Qt::AlignCenter);
    front_depth_text_->setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt;");
    frontDepthLayout->addWidget(front_depth_view_);
    frontDepthLayout->addWidget(front_depth_text_);
    grid_layout_->addWidget(frontDepthContainer, 1, 0);

    // 4. Wrist Depth (1,1)
    QWidget *wristDepthContainer = new QWidget();
    QVBoxLayout *wristDepthLayout = new QVBoxLayout(wristDepthContainer);
    wristDepthLayout->setSpacing(2);
    wristDepthLayout->setContentsMargins(5, 5, 5, 5);

    wrist_depth_view_ = new AspectRatioLabel();
    wrist_depth_view_->setPixmap(createSampleImage(QColor(241, 196, 15), "Wrist Depth"));
    wrist_depth_text_ = new QLabel("Wrist Depth");
    wrist_depth_text_->setAlignment(Qt::AlignCenter);
    wrist_depth_text_->setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 10pt;");
    wristDepthLayout->addWidget(wrist_depth_view_);
    wristDepthLayout->addWidget(wrist_depth_text_);
    grid_layout_->addWidget(wristDepthContainer, 1, 1);

    // Stretch factors
    grid_layout_->setRowStretch(0, 1);
    grid_layout_->setRowStretch(1, 1);
    grid_layout_->setColumnStretch(0, 1);
    grid_layout_->setColumnStretch(1, 1);

    left_layout_->addWidget(gridWidget, 1);
    main_layout_->addWidget(leftWidget, 4); // Left panel takes 4/5 width

    // --- Right Panel (Status & Controls) ---
    QWidget *rightWidget = new QWidget();
    right_layout_ = new QVBoxLayout(rightWidget);

    // Title & Status
    QHBoxLayout *titleLayout = new QHBoxLayout();
    data_title_ = new QLabel("System Info");
    data_title_->setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px; color: #2c3e50;");
    status_label_ = new QLabel(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss"));
    status_label_->setStyleSheet("color: #666666; font-style: italic;");
    titleLayout->addWidget(data_title_);
    titleLayout->addStretch();
    titleLayout->addWidget(status_label_);
    right_layout_->addLayout(titleLayout);

    // Status Table
    status_header_ = new QLabel("Topic Status");
    status_header_->setStyleSheet("font-weight: bold; color: #34495e; margin-top: 5px;");
    right_layout_->addWidget(status_header_);

    status_table_ = new QTableWidget(0, 2);
    status_table_->setHorizontalHeaderLabels({"Topic", "Status"});
    status_table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    status_table_->verticalHeader()->setVisible(false);
    status_table_->setAlternatingRowColors(true);
    status_table_->setSelectionMode(QAbstractItemView::NoSelection);
    status_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    status_table_->setFocusPolicy(Qt::NoFocus);
    status_table_->setStyleSheet(
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
    );
    right_layout_->addWidget(status_table_, 2); // Takes 2/5 of available vertical space

    // Metadata View (Tabbed)
    meta_header_ = new QLabel("Metadata");
    meta_header_->setStyleSheet("font-weight: bold; color: #34495e; margin-top: 10px;");
    right_layout_->addWidget(meta_header_);

    metadata_tabs_ = new QTabWidget();
    metadata_tabs_->setStyleSheet(
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
        "QTabBar::tab:selected { background: #fafafa; color: #2c3e50; font-weight: bold; border-bottom: 2px solid #2c3e50; }"
        "QTabBar::tab:hover { background: #ececec; }"
    );

    // Common style for metadata views
    QString metaStyle =
        "QTextEdit {"
        "   border: none;"
        "   background-color: #fafafa;"
        "   color: #2c3e50;"
        "   font-family: 'JetBrains Mono', 'Fira Code', 'Roboto Mono', 'Consolas', monospace;"
        "   font-size: 12px;"
        "   padding: 10px;"
        "   line-height: 1.5;"
        "}";

    // Front Metadata Tab
    front_metadata_view_ = new QTextEdit();
    front_metadata_view_->setReadOnly(true);
    front_metadata_view_->setStyleSheet(metaStyle);
    metadata_tabs_->addTab(front_metadata_view_, "Front Camera");

    // Wrist Metadata Tab
    wrist_metadata_view_ = new QTextEdit();
    wrist_metadata_view_->setReadOnly(true);
    wrist_metadata_view_->setStyleSheet(metaStyle);
    metadata_tabs_->addTab(wrist_metadata_view_, "Wrist Camera");

    // Arm State Tab
    arm_state_view_ = new QTextEdit();
    arm_state_view_->setReadOnly(true);
    arm_state_view_->setStyleSheet(metaStyle);
    metadata_tabs_->addTab(arm_state_view_, "Arm State");

    right_layout_->addWidget(metadata_tabs_, 3); // Takes more space now that controls are gone

    main_layout_->addWidget(rightWidget, 1); // Right panel takes 1/5 width
}

void VisualizationWindow::resizeEvent(QResizeEvent *event)
{
    updateFonts();
    QMainWindow::resizeEvent(event);
}

void VisualizationWindow::updateFonts()
{
    // Base resolution is 1600x900
    // We scale based on the smaller ratio to ensure fit
    double scale_w = (double)this->width() / base_width_;
    double scale_h = (double)this->height() / base_height_;
    double scale = std::min(scale_w, scale_h);

    // Minimum scale to prevent text becoming unreadable
    if (scale < 0.5) scale = 0.5;

    // --- 1. Scale Layouts ---
    if (main_layout_) {
        main_layout_->setSpacing(int(15 * scale));
        main_layout_->setContentsMargins(int(15 * scale), int(15 * scale), int(15 * scale), int(15 * scale));
    }
    if (grid_layout_) {
        grid_layout_->setSpacing(int(10 * scale));
        grid_layout_->setContentsMargins(int(5 * scale), int(5 * scale), int(5 * scale), int(5 * scale));
    }
    if (left_layout_) left_layout_->setSpacing(int(6 * scale));
    if (right_layout_) right_layout_->setSpacing(int(6 * scale));

    // --- 2. Update Stylesheets ---

    // Image Title (Base 18pt)
    image_title_->setStyleSheet(QString(
        "font-size: %1pt; font-weight: bold; margin-bottom: %2px; color: #2c3e50;"
    ).arg(int(18 * scale)).arg(int(10 * scale)));

    // View Labels (Base 10pt)
    QString viewLabelStyle = QString(
        "font-weight: bold; color: #2c3e50; font-size: %1pt;"
    ).arg(int(10 * scale));
    front_rgb_text_->setStyleSheet(viewLabelStyle);
    wrist_rgb_text_->setStyleSheet(viewLabelStyle);
    front_depth_text_->setStyleSheet(viewLabelStyle);
    wrist_depth_text_->setStyleSheet(viewLabelStyle);

    // Data Title (Base 16pt)
    data_title_->setStyleSheet(QString(
        "font-size: %1pt; font-weight: bold; margin-bottom: %2px; color: #2c3e50;"
    ).arg(int(16 * scale)).arg(int(10 * scale)));

    // Status Label (Base 10pt)
    status_label_->setStyleSheet(QString(
        "color: #666666; font-style: italic; font-size: %1pt;"
    ).arg(int(10 * scale)));

    // Headers (Base 11pt bold)
    QString headerStyle = QString(
        "font-weight: bold; color: #34495e; margin-top: %2px; font-size: %1pt;"
    ).arg(int(11 * scale)).arg(int(5 * scale));
    status_header_->setStyleSheet(headerStyle);
    meta_header_->setStyleSheet(headerStyle);

    // Status Table (Base 12px)
    // Scale row height explicitly
    status_table_->verticalHeader()->setDefaultSectionSize(int(36 * scale));

    status_table_->setStyleSheet(QString(
        "QTableWidget {"
        "   gridline-color: #e0e0e0;"
        "   font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;"
        "   font-size: %1px;"
        "   border: 1px solid #d0d0d0;"
        "   border-radius: 6px;"
        "   background-color: #ffffff;"
        "   selection-background-color: #e8f0fe;"
        "   selection-color: #2c3e50;"
        "}"
        "QTableWidget::item {"
        "   padding: %2px;"
        "   border-bottom: 1px solid #f0f0f0;"
        "   color: #2c3e50;"
        "}"
        "QHeaderView::section {"
        "   background-color: #2c3e50;"
        "   color: #ffffff;"
        "   padding: %2px;"
        "   border: none;"
        "   font-weight: 600;"
        "   font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;"
        "   text-transform: uppercase;"
        "   letter-spacing: 1px;"
        "   font-size: %3px;"
        "}"
    ).arg(int(12 * scale))
     .arg(int(10 * scale))
     .arg(int(12 * scale)));

    // Metadata Views (Base 12px)
    QString metaStyle = QString(
        "QTextEdit {"
        "   border: none;"
        "   background-color: #fafafa;"
        "   color: #2c3e50;"
        "   font-family: 'JetBrains Mono', 'Fira Code', 'Roboto Mono', 'Consolas', monospace;"
        "   font-size: %1px;"
        "   padding: %2px;"
        "   line-height: 1.5;"
        "}"
    ).arg(int(12 * scale)).arg(int(10 * scale));

    front_metadata_view_->setStyleSheet(metaStyle);
    wrist_metadata_view_->setStyleSheet(metaStyle);
    arm_state_view_->setStyleSheet(metaStyle);

    // Tab Widget (Tabs themselves)
    // Added min-width to prevent text truncation
    metadata_tabs_->setStyleSheet(QString(
        "QTabWidget::pane { border: 1px solid #d0d0d0; border-radius: 6px; background-color: #fafafa; }"
        "QTabBar::tab { "
        "   background: #e0e0e0; "
        "   color: #555555; "
        "   padding: %2px %3px; "
        "   margin-right: 2px; "
        "   border-top-left-radius: 6px; "
        "   border-top-right-radius: 6px; "
        "   font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;"
        "   font-size: %1px; "
        "   min-width: %4px; "
        "}"
        "QTabBar::tab:selected { background: #fafafa; color: #2c3e50; font-weight: bold; border-bottom: 2px solid #2c3e50; }"
        "QTabBar::tab:hover { background: #ececec; }"
    ).arg(int(12 * scale))
     .arg(int(8 * scale))
     .arg(int(12 * scale)) // Reduced horizontal padding slightly for better fit
     .arg(int(80 * scale))); // min-width ensures text space

    // Force style update
    metadata_tabs_->style()->unpolish(metadata_tabs_);
    metadata_tabs_->style()->polish(metadata_tabs_);
}

void VisualizationWindow::updateView()
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

    // Update Arm State
    sensor_msgs::msg::JointState arm_state;
    node_->get_latest_arm_state(arm_state);
    if (arm_state.name.empty()) {
        arm_state_view_->setText("No Arm State");
    } else {
        QString text = "Timestamp: " + QString::number(arm_state.header.stamp.sec) + "." + QString::number(arm_state.header.stamp.nanosec) + "\n\n";
        for (size_t i = 0; i < arm_state.name.size(); ++i) {
            text += QString::fromStdString(arm_state.name[i]) + ":\n";
            if (i < arm_state.position.size()) text += "  Pos: " + QString::number(arm_state.position[i], 'f', 4) + "\n";
            if (i < arm_state.velocity.size()) text += "  Vel: " + QString::number(arm_state.velocity[i], 'f', 4) + "\n";
            if (i < arm_state.effort.size())   text += "  Eff: " + QString::number(arm_state.effort[i], 'f', 4) + "\n";
            text += "\n";
        }
        arm_state_view_->setText(text);
    }

    status_label_->setText(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss"));
}

void VisualizationWindow::updateStatusTable(const std::map<std::string, std::string>& status)
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

void VisualizationWindow::updateMetadata(const std::string& front_json, const std::string& wrist_json)
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

QImage VisualizationWindow::matToQImage(const cv::Mat& mat)
{
    return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).copy();
}

QPixmap VisualizationWindow::createSampleImage(const QColor& color, const QString& text)
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
