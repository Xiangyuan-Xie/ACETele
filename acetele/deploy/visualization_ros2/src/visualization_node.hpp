#ifndef VISUALIZATION_NODE_HPP
#define VISUALIZATION_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <realsense2_camera_msgs/msg/metadata.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <map>
#include <chrono>
#include <atomic>

using std::placeholders::_1;

class VisualizationNode : public rclcpp::Node
{
public:
    VisualizationNode();
    ~VisualizationNode();

    // Thread-safe accessors for GUI
    void get_latest_images(cv::Mat& front_color, cv::Mat& front_depth,
                          cv::Mat& wrist_color, cv::Mat& wrist_depth);
    std::map<std::string, std::string> get_status_info();
    void get_latest_metadata(std::string& front_meta, std::string& wrist_meta);
    void get_latest_arm_state(sensor_msgs::msg::JointState& arm_state);

private:
    void color_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void wrist_color_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void wrist_depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void front_metadata_callback(const realsense2_camera_msgs::msg::Metadata::SharedPtr msg);
    void wrist_metadata_callback(const realsense2_camera_msgs::msg::Metadata::SharedPtr msg);
    void arm_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg);

    void update_status(const std::string& key, double latency_ms = 0.0);

    image_transport::Subscriber depth_sub_;
    image_transport::Subscriber color_sub_;
    image_transport::Subscriber wrist_depth_sub_;
    image_transport::Subscriber wrist_color_sub_;
    rclcpp::Subscription<realsense2_camera_msgs::msg::Metadata>::SharedPtr front_metadata_sub_;
    rclcpp::Subscription<realsense2_camera_msgs::msg::Metadata>::SharedPtr wrist_metadata_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr arm_state_sub_;

    std::mutex data_mutex_;
    cv::Mat latest_color_;
    cv::Mat latest_depth_;
    cv::Mat latest_wrist_color_;
    cv::Mat latest_wrist_depth_;
    sensor_msgs::msg::JointState latest_arm_state_;
    std::string last_front_metadata_json_;
    std::string last_wrist_metadata_json_;
    std::map<std::string, rclcpp::Time> topic_status_;
    std::map<std::string, double> topic_smoothed_latency_;
};

#endif // VISUALIZATION_NODE_HPP
