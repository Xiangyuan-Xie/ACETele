cmake_minimum_required(VERSION 3.8)
project(camera_receiver_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(camera_receiver src/camera_receiver.cpp)
target_include_directories(camera_receiver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(camera_receiver
  ${OpenCV_LIBRARIES}
)

ament_target_dependencies(camera_receiver
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

install(TARGETS camera_receiver
  DESTINATION lib/${PROJECT_NAME})

ament_package()
