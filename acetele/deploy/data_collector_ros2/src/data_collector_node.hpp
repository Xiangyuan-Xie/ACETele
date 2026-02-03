#ifndef DATA_COLLECTOR_NODE_HPP
#define DATA_COLLECTOR_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <realsense2_camera_msgs/msg/metadata.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <map>
#include <chrono>
#include <atomic>
#include <filesystem>

using std::placeholders::_1;

class DataCollectorNode : public rclcpp::Node
{
public:
    DataCollectorNode();
    ~DataCollectorNode();

    // Thread-safe accessors for GUI
    void get_latest_images(cv::Mat& front_color, cv::Mat& front_depth,
                          cv::Mat& wrist_color, cv::Mat& wrist_depth);
    std::map<std::string, std::string> get_status_info();
    void get_latest_metadata(std::string& front_meta, std::string& wrist_meta);

    // Data Collection Interface
    void start_recording(const std::string& output_dir);
    void stop_recording();
    bool is_recording() const;
    std::string get_current_recording_dir() const;
    size_t get_recorded_frame_count() const;

private:
    void color_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void wrist_color_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void wrist_depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void front_metadata_callback(const realsense2_camera_msgs::msg::Metadata::SharedPtr msg);
    void wrist_metadata_callback(const realsense2_camera_msgs::msg::Metadata::SharedPtr msg);

    void update_status(const std::string& key, double latency_ms = 0.0);
    void save_data_worker();

    image_transport::Subscriber depth_sub_;
    image_transport::Subscriber color_sub_;
    image_transport::Subscriber wrist_depth_sub_;
    image_transport::Subscriber wrist_color_sub_;
    rclcpp::Subscription<realsense2_camera_msgs::msg::Metadata>::SharedPtr front_metadata_sub_;
    rclcpp::Subscription<realsense2_camera_msgs::msg::Metadata>::SharedPtr wrist_metadata_sub_;

    std::mutex data_mutex_;
    cv::Mat latest_color_;
    cv::Mat latest_depth_;
    cv::Mat latest_wrist_color_;
    cv::Mat latest_wrist_depth_;
    std::string last_front_metadata_json_;
    std::string last_wrist_metadata_json_;
    std::map<std::string, rclcpp::Time> topic_status_;
    std::map<std::string, double> topic_smoothed_latency_;

    // Recording State
    std::atomic<bool> is_recording_;
    std::string recording_dir_;
    std::thread recording_thread_;
    std::atomic<size_t> recorded_frame_count_;
};

#endif // DATA_COLLECTOR_NODE_HPP
