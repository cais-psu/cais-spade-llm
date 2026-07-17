cmake_minimum_required(VERSION 3.8)
project(cais_lab_robotics)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    config
    launch
    models
    rviz
    scripts
    sensor
    worlds
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cad_models")
  install(
    DIRECTORY cad_models
    DESTINATION share/${PROJECT_NAME}
  )
endif()

install(
  PROGRAMS
    launch/auto_link_attacher_node.py
    sensor/gazebo_camera_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
