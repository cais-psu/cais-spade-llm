cmake_minimum_required(VERSION 3.8)
project(cais_lab_robotics)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/DockKMR.action"
  "action/MoveUR5eCartesian.action"
  "action/MoveUR5eInsert.action"
  "action/MoveUR5eJointJog.action"
  "action/MoveUR5eRelativeCartesian.action"
  "action/RecordUR5eInsertionDemonstration.action"
  "srv/SetUR5eCartesianJog.srv"
  DEPENDENCIES geometry_msgs
)

ament_environment_hooks(env-hooks/cais_lab_robotics_gazebo.dsv.in)

install(
  DIRECTORY
    config
    launch
    models
    rviz
    scripts
    sensor
    urdf
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
  FILES ../../cais_spade_llm/initialization/recovery_framework_gazebo.json
  DESTINATION share/${PROJECT_NAME}/config
)

install(
  PROGRAMS
    launch/auto_link_attacher_node.py
    sensor/gazebo_camera_detector.py
    scripts/kmr_base_controller.py
    scripts/physical_part_twin_sync.py
    scripts/recovery_drag_markers.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
