#ifndef DATA_COLLECTOR_GUI_HPP
#define DATA_COLLECTOR_GUI_HPP

#include <QMainWindow>
#include <QLabel>
#include <QTimer>
#include <QImage>
#include <QPainter>
#include <QPushButton>
#include <QLineEdit>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QTableWidget>
#include <QJsonDocument>
#include <QResizeEvent>
#include <QTabWidget>
#include <memory>
#include "data_collector_node.hpp"

class AspectRatioLabel : public QLabel {
    Q_OBJECT

public:
    explicit AspectRatioLabel(QWidget *parent = nullptr);
    void setPixmap(const QPixmap &pixmap);

protected:
    void resizeEvent(QResizeEvent *event) override;

private:
    QPixmap original_pixmap_;
    void updatePixmap();
};

class DataCollectorWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit DataCollectorWindow(std::shared_ptr<DataCollectorNode> node, QWidget *parent = nullptr);
    ~DataCollectorWindow();

private slots:
    void updateView();
    void toggleRecording();

private:
    std::shared_ptr<DataCollectorNode> node_;
    QTimer *timer_;

    // UI Elements
    // Left Panel
    AspectRatioLabel* front_rgb_view_;
    AspectRatioLabel* wrist_rgb_view_;
    AspectRatioLabel* front_depth_view_;
    AspectRatioLabel* wrist_depth_view_;

    // Right Panel
    QLabel* status_label_;
    QTableWidget* status_table_;

    // Metadata Panel (Tabbed)
    QTabWidget* metadata_tabs_;
    QTextEdit* front_metadata_view_;
    QTextEdit* wrist_metadata_view_;

    // Controls (Right Panel Bottom)
    QLineEdit* path_input_;
    QPushButton* record_btn_;

    // Helpers
    QImage matToQImage(const cv::Mat& mat);
    QPixmap createSampleImage(const QColor& color, const QString& text);
    void setupUI();
    void updateStatusTable(const std::map<std::string, std::string>& status);
    void updateMetadata(const std::string& front_json, const std::string& wrist_json);
};

#endif // DATA_COLLECTOR_GUI_HPP
