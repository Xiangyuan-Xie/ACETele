cmake_minimum_required(VERSION 3.8)
project(camera_sender_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)

add_executable(camera_sender src/camera_sender.cpp)
target_include_directories(camera_sender PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(camera_sender
  ${OpenCV_LIBRARIES}
  ${realsense2_LIBRARY}
)

ament_target_dependencies(camera_sender
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

install(TARGETS camera_sender
  DESTINATION lib/${PROJECT_NAME})

ament_package()
