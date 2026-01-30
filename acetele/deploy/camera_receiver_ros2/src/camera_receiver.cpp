#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

using std::placeholders::_1;

class CameraReceiver : public rclcpp::Node
{
public:
    CameraReceiver() : Node("camera_receiver")
    {
        // 1. Setup ROS 2 Subscribers
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth", 10, std::bind(&CameraReceiver::depth_callback, this, _1));

        metadata_sub_ = this->create_subscription<std_msgs::msg::Header>(
            "/camera/rgb/metadata", 10, std::bind(&CameraReceiver::metadata_callback, this, _1));

        // 2. Setup Video Receiver Thread
        running_ = true;
        video_thread_ = std::thread(&CameraReceiver::receive_video_loop, this);

        RCLCPP_INFO(this->get_logger(), "Camera Receiver Started. Waiting for stream on udp://127.0.0.1:5600");
    }

    ~CameraReceiver()
    {
        running_ = false;
        if (video_thread_.joinable()) {
            video_thread_.join();
        }
        cv::destroyAllWindows();
    }

private:
    // Store RGB frame with its arrival time or matched timestamp
    struct RgbFrame {
        cv::Mat image;
        rclcpp::Time timestamp;
        uint64_t seq_id;
    };

    void receive_video_loop()
    {
        // GStreamer pipeline to receive H.264 from UDP
        // sync=false to minimize latency
        std::string gst_in = "udpsrc port=5600 ! application/x-rtp, payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=false";

        cv::VideoCapture cap(gst_in, cv::CAP_GSTREAMER);

        if (!cap.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open GStreamer VideoCapture");
            return;
        }

        cv::Mat frame;
        uint64_t current_seq = 0;

        while (running_ && rclcpp::ok()) {
            if (cap.read(frame)) {
                if (frame.empty()) continue;

                // Sync Strategy:
                // We have a stream of frames and a stream of timestamps (via metadata topic).
                // We assume reliable ordering (UDP might drop, but GStreamer usually handles small packet loss or tears).
                // If we lose a frame, the sequences drift.
                // A robust implementation would need embedded timestamps in the video stream.
                // For this implementation, we will try to match the *latest* metadata received.

                rclcpp::Time estimated_ts;

                {
                    std::lock_guard<std::mutex> lock(meta_mutex_);
                    if (!metadata_queue_.empty()) {
                        // Pop the oldest metadata? Or try to match?
                        // Simple approach: FIFO.
                        // Assuming 1-to-1:
                        estimated_ts = metadata_queue_.front();
                        metadata_queue_.pop_front();
                        current_seq++;
                    } else {
                        // Fallback: Use current time or extrapolation
                        estimated_ts = this->now();
                    }
                }

                // Push to RGB Buffer
                {
                    std::lock_guard<std::mutex> lock(rgb_mutex_);
                    RgbFrame rgb_frame;
                    rgb_frame.image = frame.clone();
                    rgb_frame.timestamp = estimated_ts;
                    rgb_frame.seq_id = current_seq;

                    rgb_buffer_.push_back(rgb_frame);

                    // Keep buffer small
                    if (rgb_buffer_.size() > 30) {
                        rgb_buffer_.pop_front();
                    }
                }
            } else {
                // Wait a bit if no frame
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }

    void metadata_callback(const std_msgs::msg::Header::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(meta_mutex_);
        metadata_queue_.push_back(msg->stamp);
        // Limit queue size
        if (metadata_queue_.size() > 60) {
            metadata_queue_.pop_front();
        }
    }

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // 1. Get Depth Timestamp
        rclcpp::Time depth_ts = msg->header.stamp;

        // 2. Find closest RGB in buffer
        cv::Mat best_rgb;
        double min_diff_ns = 1e12; // Large initial value
        bool found = false;

        {
            std::lock_guard<std::mutex> lock(rgb_mutex_);
            if (rgb_buffer_.empty()) return;

            // Search for closest timestamp
            for (const auto& item : rgb_buffer_) {
                double diff = std::abs((item.timestamp - depth_ts).nanoseconds());
                if (diff < min_diff_ns) {
                    min_diff_ns = diff;
                    best_rgb = item.image;
                    found = true;
                }
            }

            // Cleanup old frames
            // Remove frames significantly older than this depth frame (e.g., > 1 second old)
            // But be careful not to remove frames that might match future depth frames if depth is delayed
            while (!rgb_buffer_.empty() && (depth_ts - rgb_buffer_.front().timestamp).seconds() > 1.0) {
                rgb_buffer_.pop_front();
            }
        }

        if (found && !best_rgb.empty()) {
            try {
                // Convert Depth to CV
                cv::Mat depth_cv = cv_bridge::toCvCopy(msg, "16UC1")->image;

                // Visualize
                visualize(best_rgb, depth_cv, min_diff_ns / 1e6); // diff in ms

            } catch (cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            }
        }
    }

    void visualize(const cv::Mat& rgb, const cv::Mat& depth, double diff_ms)
    {
        // Normalize depth
        cv::Mat depth_vis;
        // Clip to 3m
        cv::threshold(depth, depth_vis, 3000, 3000, cv::THRESH_TRUNC);
        cv::normalize(depth_vis, depth_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);

        // Resize if needed
        if (rgb.size() != depth_vis.size()) {
            cv::resize(depth_vis, depth_vis, rgb.size());
        }

        cv::Mat combined;
        cv::hconcat(rgb, depth_vis, combined);

        std::string info = "Sync Diff: " + std::to_string(diff_ms) + " ms";
        cv::putText(combined, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow("RGB (H.264 Stream) + Depth (ROS2)", combined);
        cv::waitKey(1);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr metadata_sub_;

    std::thread video_thread_;
    std::atomic<bool> running_;

    std::deque<RgbFrame> rgb_buffer_;
    std::mutex rgb_mutex_;

    std::deque<rclcpp::Time> metadata_queue_;
    std::mutex meta_mutex_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraReceiver>());
    rclcpp::shutdown();
    return 0;
}
