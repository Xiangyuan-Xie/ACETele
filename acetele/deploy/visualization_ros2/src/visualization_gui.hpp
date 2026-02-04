#ifndef VISUALIZATION_GUI_HPP
#define VISUALIZATION_GUI_HPP

#include <QMainWindow>
#include <QLabel>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QTableWidget>
#include <QTextEdit>
#include <QTabWidget>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <memory>
#include "visualization_node.hpp"

// Custom Label to maintain aspect ratio
class AspectRatioLabel : public QLabel
{
    Q_OBJECT
public:
    explicit AspectRatioLabel(QWidget *parent = nullptr);
    void setPixmap(const QPixmap &pixmap);

protected:
    void resizeEvent(QResizeEvent *event) override;

private:
    void updatePixmap();
    QPixmap original_pixmap_;
};

class VisualizationWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit VisualizationWindow(std::shared_ptr<VisualizationNode> node, QWidget *parent = nullptr);
    ~VisualizationWindow();

private slots:
    void updateView();

private:
    void setupUI();
    void updateStatusTable(const std::map<std::string, std::string>& status);
    void updateMetadata(const std::string& front_json, const std::string& wrist_json);
    QImage matToQImage(const cv::Mat& mat);
    QPixmap createSampleImage(const QColor& color, const QString& text);

    std::shared_ptr<VisualizationNode> node_;
    QTimer *timer_;

    // UI Elements
    AspectRatioLabel *front_rgb_view_;
    AspectRatioLabel *front_depth_view_;
    AspectRatioLabel *wrist_rgb_view_;
    AspectRatioLabel *wrist_depth_view_;

    QLabel *status_label_;
    QTableWidget *status_table_;

    QTabWidget *metadata_tabs_;
    QTextEdit *front_metadata_view_;
    QTextEdit *wrist_metadata_view_;
    QTextEdit *arm_state_view_;
};

#endif // VISUALIZATION_GUI_HPP
