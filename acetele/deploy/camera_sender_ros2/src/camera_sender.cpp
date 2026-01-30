#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.h>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <iostream>

using namespace std::chrono_literals;

class CameraSender : public rclcpp::Node
{
public:
    CameraSender() : Node("camera_sender")
    {
        // 1. Setup ROS 2 Publishers
        depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/depth", 10);

        // Metadata publisher for sync (Sequence ID -> Timestamp)
        // We use a custom string or just Header to send the timestamp associated with the current frame
        // Ideally we would send a custom msg {seq, timestamp}, but Header works if we use frame_id as seq
        metadata_pub_ = this->create_publisher<std_msgs::msg::Header>("/camera/rgb/metadata", 10);

        // 2. Setup RealSense
        try {
            cfg_.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
            cfg_.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
            pipe_.start(cfg_);

            // Align depth to color
            align_to_color_ = std::make_shared<rs2::align>(RS2_STREAM_COLOR);

            RCLCPP_INFO(this->get_logger(), "RealSense Initialized");
        } catch (const rs2::error & e) {
            RCLCPP_ERROR(this->get_logger(), "RealSense Error: %s", e.what());
            return;
        }

        // 3. Setup GStreamer Writer (Video Push)
        // Using x264enc for H.264 encoding and sending via UDP
        // tune=zerolatency is critical for real-time
        // speed-preset=ultrafast minimizes CPU usage
        std::string gst_out = "appsrc ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2048 ! rtph264pay ! udpsink host=127.0.0.1 port=5600";

        writer_.open(gst_out, cv::CAP_GSTREAMER, 0, 30.0, cv::Size(640, 480), true);

        if (!writer_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open GStreamer VideoWriter. Check if GStreamer is installed.");
            // We might continue just to publish depth, but usually this is fatal for the task
        } else {
            RCLCPP_INFO(this->get_logger(), "GStreamer Stream Started on udp://127.0.0.1:5600");
        }

        // 4. Timer Loop
        timer_ = this->create_wall_timer(33ms, std::bind(&CameraSender::timer_callback, this));
    }

    ~CameraSender()
    {
        pipe_.stop();
        if (writer_.isOpened()) {
            writer_.release();
        }
    }

private:
    void timer_callback()
    {
        rs2::frameset frames;
        if (pipe_.poll_for_frames(&frames)) {
            // Align
            frames = align_to_color_->process(frames);

            auto color_frame = frames.get_color_frame();
            auto depth_frame = frames.get_depth_frame();

            if (!color_frame || !depth_frame) return;

            // Get Timestamp (from RealSense or ROS Time)
            // Ideally use RealSense timestamp, but for simplicity and ROS sync, we use ROS time
            // We capture the time NOW as the "timestamp" for both
            rclcpp::Time now = this->get_clock()->now();

            // --- Process Depth (ROS 2 Topic) ---
            // Convert to OpenCV
            cv::Mat depth_mat(cv::Size(640, 480), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

            std_msgs::msg::Header header;
            header.stamp = now;
            header.frame_id = "camera_link";

            sensor_msgs::msg::Image::SharedPtr depth_msg = cv_bridge::CvImage(header, "16UC1", depth_mat).toImageMsg();
            depth_pub_->publish(*depth_msg);

            // --- Process RGB (H.264 Stream) ---
            cv::Mat color_mat(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

            if (writer_.isOpened()) {
                // Publish Metadata FIRST so receiver might buffer it before frame arrives
                std_msgs::msg::Header meta_header;
                meta_header.stamp = now;
                meta_header.frame_id = std::to_string(frame_count_); // Use frame_id as sequence number
                metadata_pub_->publish(meta_header);

                // Write to stream
                writer_.write(color_mat);
                frame_count_++;
            }
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr metadata_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    rs2::pipeline pipe_;
    rs2::config cfg_;
    std::shared_ptr<rs2::align> align_to_color_;

    cv::VideoWriter writer_;
    size_t frame_count_ = 0;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraSender>());
    rclcpp::shutdown();
    return 0;
}
