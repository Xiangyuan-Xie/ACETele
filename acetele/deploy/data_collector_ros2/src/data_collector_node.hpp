#ifndef DATA_COLLECTOR_NODE_HPP
#define DATA_COLLECTOR_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <realsense2_camera_msgs/msg/metadata.hpp>
#include <realsense2_camera_msgs/msg/extrinsics.hpp>
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
    void get_latest_images(cv::Mat& color, cv::Mat& depth);
    std::map<std::string, std::string> get_status_info();
    std::string get_metadata_json();

    // Data Collection Interface
    void start_recording(const std::string& output_dir);
    void stop_recording();
    bool is_recording() const;
    std::string get_current_recording_dir() const;
    size_t get_recorded_frame_count() const;

private:
    void color_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

    void color_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void color_metadata_callback(const realsense2_camera_msgs::msg::Metadata::SharedPtr msg);
    void depth_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void depth_metadata_callback(const realsense2_camera_msgs::msg::Metadata::SharedPtr msg);
    void depth_to_color_ext_callback(const realsense2_camera_msgs::msg::Extrinsics::SharedPtr msg);
    void depth_to_depth_ext_callback(const realsense2_camera_msgs::msg::Extrinsics::SharedPtr msg);

    void update_status(const std::string& key);
    void save_data_worker();

    image_transport::Subscriber depth_sub_;
    image_transport::Subscriber color_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr color_info_sub_;
    rclcpp::Subscription<realsense2_camera_msgs::msg::Metadata>::SharedPtr color_metadata_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_;
    rclcpp::Subscription<realsense2_camera_msgs::msg::Metadata>::SharedPtr depth_metadata_sub_;
    rclcpp::Subscription<realsense2_camera_msgs::msg::Extrinsics>::SharedPtr depth_to_color_ext_sub_;
    rclcpp::Subscription<realsense2_camera_msgs::msg::Extrinsics>::SharedPtr depth_to_depth_ext_sub_;

    std::mutex data_mutex_;
    cv::Mat latest_color_;
    cv::Mat latest_depth_;
    std::string last_metadata_json_;
    std::map<std::string, rclcpp::Time> topic_status_;

    // Recording State
    std::atomic<bool> is_recording_;
    std::string recording_dir_;
    std::thread recording_thread_;
    std::atomic<size_t> recorded_frame_count_;
};

#endif // DATA_COLLECTOR_NODE_HPP
