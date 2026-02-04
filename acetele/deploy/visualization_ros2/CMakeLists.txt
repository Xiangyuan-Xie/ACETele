cmake_minimum_required(VERSION 3.8)
project(visualization_ros2)

# Enable Qt MOC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(realsense2_camera_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)

add_executable(visualization
  src/visualization_main.cpp
  src/visualization_node.cpp
  src/visualization_gui.cpp
)

target_include_directories(visualization PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(visualization
  ${OpenCV_LIBRARIES}
  Qt5::Widgets
  Qt5::Core
  Qt5::Gui
)

ament_target_dependencies(visualization
  rclcpp
  std_msgs
  sensor_msgs
  realsense2_camera_msgs
  cv_bridge
  image_transport
)

install(TARGETS visualization
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
