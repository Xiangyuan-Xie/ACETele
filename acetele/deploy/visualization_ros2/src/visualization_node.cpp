#include "visualization_node.hpp"
#include <fstream>
#include <iomanip>

VisualizationNode::VisualizationNode() : Node("visualization")
{
    // Parameters for topics
    this->declare_parameter("front_color_topic", "/camera/front/color/image_raw");
    this->declare_parameter("front_depth_topic", "/camera/front/aligned_depth_to_color/image_raw");
    this->declare_parameter("wrist_color_topic", "/camera/wrist/color/image_raw");
    this->declare_parameter("wrist_depth_topic", "/camera/wrist/aligned_depth_to_color/image_raw");
    this->declare_parameter("front_color_metadata_topic", "/camera/front/color/metadata");
    this->declare_parameter("wrist_color_metadata_topic", "/camera/wrist/color/metadata");
    this->declare_parameter("arm_state_topic", "/arm/state");

    // Transport parameters
    this->declare_parameter("color_transport", "compressed");
    this->declare_parameter("depth_transport", "compressedDepth");

    std::string front_color_topic = this->get_parameter("front_color_topic").as_string();
    std::string front_depth_topic = this->get_parameter("front_depth_topic").as_string();
    std::string wrist_color_topic = this->get_parameter("wrist_color_topic").as_string();
    std::string wrist_depth_topic = this->get_parameter("wrist_depth_topic").as_string();
    std::string front_color_metadata_topic = this->get_parameter("front_color_metadata_topic").as_string();
    std::string wrist_color_metadata_topic = this->get_parameter("wrist_color_metadata_topic").as_string();
    std::string arm_state_topic = this->get_parameter("arm_state_topic").as_string();

    std::string color_transport = this->get_parameter("color_transport").as_string();
    std::string depth_transport = this->get_parameter("depth_transport").as_string();

    // Setup ROS 2 Subscribers
    auto qos = rclcpp::SensorDataQoS();

    // Always use Best Effort QoS for image transport to prevent disconnection
    // especially for compressed streams over network/WiFi
    RCLCPP_INFO(this->get_logger(), "Using SensorDataQoS (Best Effort) for color transport: %s", color_transport.c_str());
    color_sub_ = image_transport::create_subscription(
        this, front_color_topic, std::bind(&VisualizationNode::color_callback, this, _1), color_transport, qos.get_rmw_qos_profile());

    RCLCPP_INFO(this->get_logger(), "Using SensorDataQoS (Best Effort) for depth transport: %s", depth_transport.c_str());
    depth_sub_ = image_transport::create_subscription(
        this, front_depth_topic, std::bind(&VisualizationNode::depth_callback, this, _1), depth_transport, qos.get_rmw_qos_profile());

    RCLCPP_INFO(this->get_logger(), "Using SensorDataQoS (Best Effort) for wrist color transport: %s", color_transport.c_str());
    wrist_color_sub_ = image_transport::create_subscription(
        this, wrist_color_topic, std::bind(&VisualizationNode::wrist_color_callback, this, _1), color_transport, qos.get_rmw_qos_profile());

    RCLCPP_INFO(this->get_logger(), "Using SensorDataQoS (Best Effort) for wrist depth transport: %s", depth_transport.c_str());
    wrist_depth_sub_ = image_transport::create_subscription(
        this, wrist_depth_topic, std::bind(&VisualizationNode::wrist_depth_callback, this, _1), depth_transport, qos.get_rmw_qos_profile());

    front_metadata_sub_ = this->create_subscription<realsense2_camera_msgs::msg::Metadata>(
        front_color_metadata_topic, qos, std::bind(&VisualizationNode::front_metadata_callback, this, _1));

    wrist_metadata_sub_ = this->create_subscription<realsense2_camera_msgs::msg::Metadata>(
        wrist_color_metadata_topic, qos, std::bind(&VisualizationNode::wrist_metadata_callback, this, _1));

    arm_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        arm_state_topic, qos, std::bind(&VisualizationNode::arm_state_callback, this, _1));

    RCLCPP_INFO(this->get_logger(), "Visualization Node Started.");
    RCLCPP_INFO(this->get_logger(), "Transport - Color: %s, Depth: %s", color_transport.c_str(), depth_transport.c_str());
}

VisualizationNode::~VisualizationNode()
{
}

void VisualizationNode::get_latest_images(cv::Mat& front_color, cv::Mat& front_depth,
                                          cv::Mat& wrist_color, cv::Mat& wrist_depth)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!latest_color_.empty()) {
        front_color = latest_color_;
    }
    if (!latest_depth_.empty()) {
        front_depth = latest_depth_;
    }
    if (!latest_wrist_color_.empty()) {
        wrist_color = latest_wrist_color_;
    }
    if (!latest_wrist_depth_.empty()) {
        wrist_depth = latest_wrist_depth_;
    }
}

void VisualizationNode::get_latest_arm_state(sensor_msgs::msg::JointState& arm_state)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    arm_state = latest_arm_state_;
}

std::map<std::string, std::string> VisualizationNode::get_status_info()
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    std::map<std::string, std::string> info;
    auto now = this->now();

    for (auto const& [key, time] : topic_status_) {
        double diff = (now - time).seconds();
        if (diff < 2.0) {
            std::string status = "ONLINE";
            if (topic_smoothed_latency_.find(key) != topic_smoothed_latency_.end()) {
                std::stringstream ss;
                ss << "ONLINE (" << std::fixed << std::setprecision(1) << topic_smoothed_latency_[key] << "ms)";
                status = ss.str();
            }
            info[key] = status;
        } else {
            info[key] = "OFFLINE";
        }
    }
    return info;
}

void VisualizationNode::get_latest_metadata(std::string& front_meta, std::string& wrist_meta)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    front_meta = last_front_metadata_json_;
    wrist_meta = last_wrist_metadata_json_;
}

void VisualizationNode::update_status(const std::string& key, double latency_ms)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    topic_status_[key] = this->now();

    if (topic_smoothed_latency_.find(key) == topic_smoothed_latency_.end()) {
        topic_smoothed_latency_[key] = latency_ms;
    } else {
        // Alpha determines the weight of new data.
        double alpha = 0.05;
        topic_smoothed_latency_[key] = alpha * latency_ms + (1.0 - alpha) * topic_smoothed_latency_[key];
    }
}

void VisualizationNode::color_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    try {
        cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
        double latency = (this->now() - msg->header.stamp).seconds() * 1000.0;
        if (latency < 0.0) latency = 0.0;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            latest_color_ = image;
        }
        update_status("front_color", latency);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in color_callback: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown error in color_callback");
    }
}

void VisualizationNode::depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    try {
        cv::Mat image = cv_bridge::toCvCopy(msg, "16UC1")->image;
        double latency = (this->now() - rclcpp::Time(msg->header.stamp)).seconds() * 1000.0;
        if (latency < 0.0) latency = 0.0;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            latest_depth_ = image;
        }
        update_status("front_depth", latency);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in depth_callback: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown error in depth_callback");
    }
}

void VisualizationNode::wrist_color_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    try {
        cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
        double latency = (this->now() - msg->header.stamp).seconds() * 1000.0;
        if (latency < 0.0) latency = 0.0;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            latest_wrist_color_ = image;
        }
        update_status("wrist_color", latency);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in wrist_color_callback: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown error in wrist_color_callback");
    }
}

void VisualizationNode::wrist_depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    try {
        cv::Mat image = cv_bridge::toCvCopy(msg, "16UC1")->image;
        double latency = (this->now() - rclcpp::Time(msg->header.stamp)).seconds() * 1000.0;
        if (latency < 0.0) latency = 0.0;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            latest_wrist_depth_ = image;
        }
        update_status("wrist_depth", latency);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in wrist_depth_callback: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown error in wrist_depth_callback");
    }
}

void VisualizationNode::front_metadata_callback(const realsense2_camera_msgs::msg::Metadata::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    topic_status_["front_metadata"] = this->now();
    last_front_metadata_json_ = msg->json_data;
}

void VisualizationNode::wrist_metadata_callback(const realsense2_camera_msgs::msg::Metadata::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    topic_status_["wrist_metadata"] = this->now();
    last_wrist_metadata_json_ = msg->json_data;
}

void VisualizationNode::arm_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_arm_state_ = *msg;
    topic_status_["arm_state"] = this->now();
}
