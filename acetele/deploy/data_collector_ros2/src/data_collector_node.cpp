#include "data_collector_node.hpp"
#include <fstream>
#include <iomanip>

DataCollectorNode::DataCollectorNode() : Node("data_collector"), is_recording_(false), recorded_frame_count_(0)
{
    // Parameters for topics
    this->declare_parameter("front_color_topic", "/camera/front/color/image_raw");
    this->declare_parameter("front_depth_topic", "/camera/front/aligned_depth_to_color/image_raw");
    this->declare_parameter("wrist_color_topic", "/camera/wrist/color/image_raw");
    this->declare_parameter("wrist_depth_topic", "/camera/wrist/aligned_depth_to_color/image_raw");
    this->declare_parameter("front_color_metadata_topic", "/camera/front/color/metadata");
    this->declare_parameter("wrist_color_metadata_topic", "/camera/wrist/color/metadata");

    // Transport parameters
    this->declare_parameter("color_transport", "compressed");
    this->declare_parameter("depth_transport", "compressedDepth");

    std::string front_color_topic = this->get_parameter("front_color_topic").as_string();
    std::string front_depth_topic = this->get_parameter("front_depth_topic").as_string();
    std::string wrist_color_topic = this->get_parameter("wrist_color_topic").as_string();
    std::string wrist_depth_topic = this->get_parameter("wrist_depth_topic").as_string();
    std::string front_color_metadata_topic = this->get_parameter("front_color_metadata_topic").as_string();
    std::string wrist_color_metadata_topic = this->get_parameter("wrist_color_metadata_topic").as_string();

    std::string color_transport = this->get_parameter("color_transport").as_string();
    std::string depth_transport = this->get_parameter("depth_transport").as_string();

    // Setup ROS 2 Subscribers
    auto qos = rclcpp::SensorDataQoS();

    // Always use Best Effort QoS for image transport to prevent disconnection
    // especially for compressed streams over network/WiFi
    RCLCPP_INFO(this->get_logger(), "Using SensorDataQoS (Best Effort) for color transport: %s", color_transport.c_str());
    color_sub_ = image_transport::create_subscription(
        this, front_color_topic, std::bind(&DataCollectorNode::color_callback, this, _1), color_transport, qos.get_rmw_qos_profile());

    RCLCPP_INFO(this->get_logger(), "Using SensorDataQoS (Best Effort) for depth transport: %s", depth_transport.c_str());
    depth_sub_ = image_transport::create_subscription(
        this, front_depth_topic, std::bind(&DataCollectorNode::depth_callback, this, _1), depth_transport, qos.get_rmw_qos_profile());

    RCLCPP_INFO(this->get_logger(), "Using SensorDataQoS (Best Effort) for wrist color transport: %s", color_transport.c_str());
    wrist_color_sub_ = image_transport::create_subscription(
        this, wrist_color_topic, std::bind(&DataCollectorNode::wrist_color_callback, this, _1), color_transport, qos.get_rmw_qos_profile());

    RCLCPP_INFO(this->get_logger(), "Using SensorDataQoS (Best Effort) for wrist depth transport: %s", depth_transport.c_str());
    wrist_depth_sub_ = image_transport::create_subscription(
        this, wrist_depth_topic, std::bind(&DataCollectorNode::wrist_depth_callback, this, _1), depth_transport, qos.get_rmw_qos_profile());

    front_metadata_sub_ = this->create_subscription<realsense2_camera_msgs::msg::Metadata>(
        front_color_metadata_topic, qos, std::bind(&DataCollectorNode::front_metadata_callback, this, _1));

    wrist_metadata_sub_ = this->create_subscription<realsense2_camera_msgs::msg::Metadata>(
        wrist_color_metadata_topic, qos, std::bind(&DataCollectorNode::wrist_metadata_callback, this, _1));

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

void DataCollectorNode::get_latest_images(cv::Mat& front_color, cv::Mat& front_depth,
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

std::map<std::string, std::string> DataCollectorNode::get_status_info()
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
                ss << std::fixed << std::setprecision(1) << topic_smoothed_latency_[key];
                status += " (" + ss.str() + " ms)";
            }
            info[key] = status;
        } else {
            info[key] = "OFFLINE (" + std::to_string((int)diff) + "s)";
        }
    }
    return info;
}

void DataCollectorNode::get_latest_metadata(std::string& front_meta, std::string& wrist_meta)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    front_meta = last_front_metadata_json_;
    wrist_meta = last_wrist_metadata_json_;
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

        cv::Mat color_snap, depth_snap, wrist_color_snap, wrist_depth_snap;
        std::string front_meta_snap, wrist_meta_snap;

        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            // At least one camera should have data to record, but we are robust
            // We check if we have ANY data. If totally empty, maybe skip or record empty?
            // Usually we want aligned data. Let's assume front camera is primary, but we should be flexible.
            // If both are empty, skip.
            bool front_has_data = !latest_color_.empty() && !latest_depth_.empty();
            bool wrist_has_data = !latest_wrist_color_.empty() && !latest_wrist_depth_.empty();

            if (!front_has_data && !wrist_has_data) continue;

            if (front_has_data) {
                color_snap = latest_color_;
                depth_snap = latest_depth_;
                front_meta_snap = last_front_metadata_json_;
            }
            if (wrist_has_data) {
                wrist_color_snap = latest_wrist_color_;
                wrist_depth_snap = latest_wrist_depth_;
                wrist_meta_snap = last_wrist_metadata_json_;
            }
        }

        // Generate timestamp for filenames
        auto now = std::chrono::system_clock::now();
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::string timestamp = std::to_string(now_ms);

        std::string color_path = recording_dir_ + "/" + timestamp + "_front_color.jpg";
        std::string depth_path = recording_dir_ + "/" + timestamp + "_front_depth.png";
        std::string wrist_color_path = recording_dir_ + "/" + timestamp + "_wrist_color.jpg";
        std::string wrist_depth_path = recording_dir_ + "/" + timestamp + "_wrist_depth.png";
        std::string front_meta_path = recording_dir_ + "/" + timestamp + "_front_meta.json";
        std::string wrist_meta_path = recording_dir_ + "/" + timestamp + "_wrist_meta.json";

        // Save Front Data
        if (!color_snap.empty()) {
            try { cv::imwrite(color_path, color_snap); } catch (...) {}
            try { cv::imwrite(depth_path, depth_snap); } catch (...) {}
            if (!front_meta_snap.empty()) {
                std::ofstream meta_file(front_meta_path);
                if (meta_file.is_open()) {
                    meta_file << front_meta_snap;
                    meta_file.close();
                }
            }
        }

        // Save Wrist Data
        if (!wrist_color_snap.empty()) {
            try { cv::imwrite(wrist_color_path, wrist_color_snap); } catch (...) {}
            try { cv::imwrite(wrist_depth_path, wrist_depth_snap); } catch (...) {}
            if (!wrist_meta_snap.empty()) {
                std::ofstream meta_file(wrist_meta_path);
                if (meta_file.is_open()) {
                    meta_file << wrist_meta_snap;
                    meta_file.close();
                }
            }
        }

        recorded_frame_count_++;

        // Rate limit recording to ~10Hz
        std::this_thread::sleep_for(std::chrono::milliseconds(66));
    }
}

void DataCollectorNode::update_status(const std::string& key, double latency_ms) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    topic_status_[key] = this->now();

    // EMA Smoothing
    if (topic_smoothed_latency_.find(key) == topic_smoothed_latency_.end()) {
        topic_smoothed_latency_[key] = latency_ms;
    } else {
        // Alpha determines the weight of new data.
        // Lower alpha = more smoothing (slower response)
        // Higher alpha = less smoothing (faster response)
        // 0.05 is quite smooth.
        double alpha = 0.05;
        topic_smoothed_latency_[key] = alpha * latency_ms + (1.0 - alpha) * topic_smoothed_latency_[key];
    }
}

void DataCollectorNode::color_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
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
    } catch (...) {}
}

void DataCollectorNode::depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
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
    } catch (...) {}
}

void DataCollectorNode::wrist_color_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
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
    } catch (...) {}
}

void DataCollectorNode::wrist_depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
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
    } catch (...) {}
}

void DataCollectorNode::front_metadata_callback(const realsense2_camera_msgs::msg::Metadata::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    topic_status_["front_metadata"] = this->now();
    last_front_metadata_json_ = msg->json_data;
}

void DataCollectorNode::wrist_metadata_callback(const realsense2_camera_msgs::msg::Metadata::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    topic_status_["wrist_metadata"] = this->now();
    last_wrist_metadata_json_ = msg->json_data;
}
