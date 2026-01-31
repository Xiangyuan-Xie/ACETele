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
#include <QJsonObject>
#include <memory>
#include "data_collector_node.hpp"

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
    QLabel* title_label_;
    QLabel* time_label_;
    QLabel* rgb_view_;
    QLabel* depth_view_;

    QTableWidget* status_table_;
    QTextEdit* metadata_view_;

    QLineEdit* path_input_;
    QPushButton* record_btn_;
    QLabel* record_status_label_;

    // Helpers
    QImage matToQImage(const cv::Mat& mat);
    void setupUI();
    void updateStatusTable(const std::map<std::string, std::string>& status);
    void updateMetadata(const std::string& json_str);
};

#endif // DATA_COLLECTOR_GUI_HPP
