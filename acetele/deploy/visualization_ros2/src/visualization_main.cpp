#include <QApplication>
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include "visualization_node.hpp"
#include "visualization_gui.hpp"

int main(int argc, char * argv[])
{
    // 1. Initialize ROS
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisualizationNode>();

    // 2. Initialize Qt
    QApplication app(argc, argv);

    // 3. Create Window
    VisualizationWindow window(node);
    window.show();

    // 4. Run ROS in a separate thread
    std::thread ros_thread([node]() {
        rclcpp::executors::MultiThreadedExecutor exec;
        exec.add_node(node);
        exec.spin();
    });

    // 5. Run Qt Event Loop
    int result = app.exec();

    // 6. Cleanup
    rclcpp::shutdown();

    if (ros_thread.joinable()) {
        ros_thread.join();
    }

    return result;
}
