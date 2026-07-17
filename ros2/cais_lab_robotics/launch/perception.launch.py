#!/usr/bin/env python3
"""
Launch the perception node for vision-based part detection.

Runs gazebo_camera_detector.py which subscribes to the 4 depth cameras
(cam_mk3, cam_mk4_1, cam_mk4_2, cam_assembly) and provides
/detect_part and /detect_all services.

Usage:
    # After starting the simulation:
    ros2 launch cais_lab_robotics perception.launch.py

    # View debug image:
    ros2 run rqt_image_view rqt_image_view /perception/debug_image

    # Test detection:
    ros2 param set /perception_node target_part SG
    ros2 service call /detect_part std_srvs/srv/Trigger "{}"
    ros2 service call /detect_all std_srvs/srv/Trigger "{}"
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cais_lab_robotics",
            executable="gazebo_camera_detector.py",
            name="perception_node",
            output="screen",
            parameters=[{
                "target_part": "",
            }],
        ),
    ])
