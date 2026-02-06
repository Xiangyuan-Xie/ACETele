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

protected:
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void updateView();

private:
    void setupUI();
    void updateFonts();
    void updateStatusTable(const std::map<std::string, std::string>& status);
    void updateMetadata(const std::string& front_json, const std::string& wrist_json);
    QImage matToQImage(const cv::Mat& mat);
    QPixmap createSampleImage(const QColor& color, const QString& text);

    std::shared_ptr<VisualizationNode> node_;
    QTimer *timer_;
    double base_width_ = 1600.0;
    double base_height_ = 900.0;

    // UI Elements
    QLabel *image_title_;
    QLabel *front_rgb_text_;
    QLabel *wrist_rgb_text_;
    QLabel *front_depth_text_;
    QLabel *wrist_depth_text_;

    AspectRatioLabel *front_rgb_view_;
    AspectRatioLabel *front_depth_view_;
    AspectRatioLabel *wrist_rgb_view_;
    AspectRatioLabel *wrist_depth_view_;

    QLabel *data_title_;
    QLabel *status_label_;
    QLabel *status_header_;
    QTableWidget *status_table_;

    QLabel *meta_header_;
    QTabWidget *metadata_tabs_;
    QTextEdit *front_metadata_view_;
    QTextEdit *wrist_metadata_view_;
    QTextEdit *arm_state_view_;

    // Layouts
    QHBoxLayout *main_layout_;
    QVBoxLayout *left_layout_;
    QGridLayout *grid_layout_;
    QVBoxLayout *right_layout_;
};

#endif // VISUALIZATION_GUI_HPP
