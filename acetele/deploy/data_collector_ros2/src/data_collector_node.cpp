#include "data_collector_node.hpp"
#include <fstream>
#include <iomanip>

DataCollectorNode::DataCollectorNode() : Node("data_collector"), is_recording_(false), recorded_frame_count_(0)
{
    // Parameters for topics
    this->declare_parameter("color_topic", "/camera/camera/color/image_raw");
    this->declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw");
    this->declare_parameter("color_metadata_topic", "/camera/camera/color/metadata");

    // Transport parameters
    this->declare_parameter("color_transport", "compressed");
    this->declare_parameter("depth_transport", "compressedDepth");

    std::string color_topic = this->get_parameter("color_topic").as_string();
    std::string depth_topic = this->get_parameter("depth_topic").as_string();
    std::string color_metadata_topic = this->get_parameter("color_metadata_topic").as_string();

    std::string color_transport = this->get_parameter("color_transport").as_string();
    std::string depth_transport = this->get_parameter("depth_transport").as_string();

    // Setup ROS 2 Subscribers
    auto qos = rclcpp::SensorDataQoS();

    // Always use Best Effort QoS for image transport to prevent disconnection
    // especially for compressed streams over network/WiFi
    RCLCPP_INFO(this->get_logger(), "Using SensorDataQoS (Best Effort) for color transport: %s", color_transport.c_str());
    color_sub_ = image_transport::create_subscription(
        this, color_topic, std::bind(&DataCollectorNode::color_callback, this, _1), color_transport, qos.get_rmw_qos_profile());

    RCLCPP_INFO(this->get_logger(), "Using SensorDataQoS (Best Effort) for depth transport: %s", depth_transport.c_str());
    depth_sub_ = image_transport::create_subscription(
        this, depth_topic, std::bind(&DataCollectorNode::depth_callback, this, _1), depth_transport, qos.get_rmw_qos_profile());

    color_metadata_sub_ = this->create_subscription<realsense2_camera_msgs::msg::Metadata>(
        color_metadata_topic, qos, std::bind(&DataCollectorNode::color_metadata_callback, this, _1));

    RCLCPP_INFO(this->get_logger(), "Data Collector Started.");
    RCLCPP_INFO(this->get_logger(), "Transport - Color: %s, Depth: %s", color_transport.c_str(), depth_transport.c_str());

    // Start Recording Thread
    recording_thread_ = std::thread(&DataCollectorNode::save_data_worker, this);
}

DataCollectorNode::~DataCollectorNode()
{
    is_recording_ = false;
    if (recording_thread_.joinable()) {
        recording_thread_.join();
    }
}

void DataCollectorNode::get_latest_images(cv::Mat& color, cv::Mat& depth)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!latest_color_.empty()) {
        color = latest_color_;
    }
    if (!latest_depth_.empty()) {
        depth = latest_depth_;
    }
}

std::map<std::string, std::string> DataCollectorNode::get_status_info()
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    std::map<std::string, std::string> info;
    auto now = this->now();

    for (auto const& [key, time] : topic_status_) {
        double diff = (now - time).seconds();
        if (diff < 2.0) {
            info[key] = "ONLINE";
        } else {
            info[key] = "OFFLINE (" + std::to_string((int)diff) + "s)";
        }
    }
    return info;
}

std::string DataCollectorNode::get_metadata_json()
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    return last_metadata_json_;
}

void DataCollectorNode::start_recording(const std::string& output_dir)
{
    if (is_recording_) return;

    recording_dir_ = output_dir;
    std::filesystem::create_directories(recording_dir_);
    recorded_frame_count_ = 0;
    is_recording_ = true;
    RCLCPP_INFO(this->get_logger(), "Started recording to: %s", recording_dir_.c_str());
}

void DataCollectorNode::stop_recording()
{
    if (!is_recording_) return;

    is_recording_ = false;
    RCLCPP_INFO(this->get_logger(), "Stopped recording. Total frames: %zu", recorded_frame_count_.load());
}

bool DataCollectorNode::is_recording() const
{
    return is_recording_;
}

std::string DataCollectorNode::get_current_recording_dir() const
{
    return recording_dir_;
}

size_t DataCollectorNode::get_recorded_frame_count() const
{
    return recorded_frame_count_;
}

void DataCollectorNode::save_data_worker()
{
    while (rclcpp::ok()) {
        if (!is_recording_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        cv::Mat color_snap, depth_snap;
        std::string meta_snap;

        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (latest_color_.empty() || latest_depth_.empty()) continue;
            color_snap = latest_color_;
            depth_snap = latest_depth_;
            meta_snap = last_metadata_json_;
        }

        // Generate timestamp for filenames
        auto now = std::chrono::system_clock::now();
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::string timestamp = std::to_string(now_ms);

        std::string color_path = recording_dir_ + "/" + timestamp + "_color.jpg";
        std::string depth_path = recording_dir_ + "/" + timestamp + "_depth.png";
        std::string meta_path = recording_dir_ + "/" + timestamp + "_meta.json";

        // Save Color
        try {
            cv::imwrite(color_path, color_snap);
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "Failed to save color image");
        }

        // Save Depth
        try {
            cv::imwrite(depth_path, depth_snap);
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "Failed to save depth image");
        }

        // Save Metadata
        if (!meta_snap.empty()) {
            std::ofstream meta_file(meta_path);
            if (meta_file.is_open()) {
                meta_file << meta_snap;
                meta_file.close();
            }
        }

        recorded_frame_count_++;

        // Rate limit recording to ~10Hz or similar to avoid disk flooding
        std::this_thread::sleep_for(std::chrono::milliseconds(66));
    }
}

void DataCollectorNode::update_status(const std::string& key, double latency_ms) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    topic_status_[key] = this->now();
    topic_latency_[key] = latency_ms;
}

void DataCollectorNode::color_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    try {
        cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
        double latency = (this->now() - msg->header.stamp).seconds() * 1000.0;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            latest_color_ = image;
        }
        update_status("color", latency);
    } catch (...) {}
}

void DataCollectorNode::depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    try {
        cv::Mat image = cv_bridge::toCvCopy(msg, "16UC1")->image;
        double latency = (this->now() - rclcpp::Time(msg->header.stamp)).seconds() * 1000.0;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            latest_depth_ = image;
        }
        update_status("depth", latency);
    } catch (...) {}
}

void DataCollectorNode::color_metadata_callback(const realsense2_camera_msgs::msg::Metadata::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    topic_status_["color_metadata"] = this->now();
    last_metadata_json_ = msg->json_data;
}
