cmake_minimum_required(VERSION 3.8)
project(cais_lab_robotics)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(gazebo REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(joint_trajectory_controller REQUIRED)
find_package(pluginlib REQUIRED)
find_package(std_srvs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(fcl REQUIRED)
find_library(CCD_LIBRARY ccd REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(rosidl_typesupport_introspection_cpp REQUIRED)

add_library(cais_recovery_joint_trajectory_controller SHARED src/recovery_joint_trajectory_controller.cpp)
ament_target_dependencies(cais_recovery_joint_trajectory_controller
  joint_trajectory_controller pluginlib std_srvs trajectory_msgs control_msgs rclcpp_action lifecycle_msgs)
target_compile_features(cais_recovery_joint_trajectory_controller PRIVATE cxx_std_17)
pluginlib_export_plugin_description_file(controller_interface recovery_controllers.xml)
install(TARGETS cais_recovery_joint_trajectory_controller LIBRARY DESTINATION lib)

add_library(cais_recovery_render_rate SHARED src/recovery_render_rate.cpp)
target_include_directories(cais_recovery_render_rate PRIVATE ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(cais_recovery_render_rate ${GAZEBO_LIBRARIES} Qt5::Widgets)
target_compile_features(cais_recovery_render_rate PRIVATE cxx_std_17)
install(TARGETS cais_recovery_render_rate LIBRARY DESTINATION lib)

add_library(cais_kmr_rigid_base_motion SHARED src/kmr_rigid_base_motion.cpp)
target_include_directories(cais_kmr_rigid_base_motion PRIVATE ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(cais_kmr_rigid_base_motion ${GAZEBO_LIBRARIES})
target_compile_features(cais_kmr_rigid_base_motion PRIVATE cxx_std_17)
install(TARGETS cais_kmr_rigid_base_motion LIBRARY DESTINATION lib)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/PreviewRecoveryMotion.action"
  "action/DockKMR.action"
  "action/MoveBaseKMR.action"
  "action/MoveUR5eCartesian.action"
  "action/MoveUR5eInsert.action"
  "action/MoveUR5eJointJog.action"
  "action/MoveUR5eRelativeCartesian.action"
  "action/RecordUR5eInsertionDemonstration.action"
  "srv/SetUR5eCartesianJog.srv"
  "srv/SetRecoveryMotionContract.srv"
  DEPENDENCIES geometry_msgs trajectory_msgs moveit_msgs sensor_msgs
)
rosidl_get_typesupport_target(cais_recovery_typesupport ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(cais_recovery_joint_trajectory_controller "${cais_recovery_typesupport}")

add_executable(preview_recovery_motion src/preview_recovery_motion.cpp)
ament_target_dependencies(preview_recovery_motion
  rclcpp rclcpp_action moveit_core moveit_ros_planning moveit_msgs sensor_msgs geometry_msgs rosidl_typesupport_introspection_cpp)
target_compile_features(preview_recovery_motion PRIVATE cxx_std_17)
target_link_libraries(preview_recovery_motion "${cais_recovery_typesupport}" OpenSSL::Crypto nlohmann_json::nlohmann_json fcl "${CCD_LIBRARY}")
install(TARGETS preview_recovery_motion DESTINATION lib/${PROJECT_NAME})

ament_environment_hooks(env-hooks/cais_lab_robotics_gazebo.dsv.in)

install(
  DIRECTORY
    config
    launch
    models
    rviz
    scripts
    src
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
    scripts/recovery_controller_spawner.py
    scripts/recovery_nav2_lifecycle.py
    scripts/recovery_drag_markers.py
    scripts/simulation_performance.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(preview_recovery_motion_test test/test_preview_recovery_motion.py
    TIMEOUT 90
    ENV "PREVIEW_RECOVERY_MOTION_TEST_SERVER=$<TARGET_FILE:preview_recovery_motion>"
        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_py:$ENV{PYTHONPATH}"
        "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_py/cais_lab_robotics:${CMAKE_CURRENT_BINARY_DIR}:$ENV{LD_LIBRARY_PATH}")
endif()

ament_package()
