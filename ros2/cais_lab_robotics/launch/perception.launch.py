#!/usr/bin/env python3
"""
Launch the perception node for vision-based part detection.

Runs gazebo_camera_detector.py which subscribes to the 4 depth cameras
(cam_mk3, cam_mk4_1, cam_mk4_2, cam_assembly) and provides
/detect_part and /detect_all services.

Usage:
    # After starting the simulation:
    ros2 launch xarm_gazebo perception.launch.py

    # View debug image:
    ros2 run rqt_image_view rqt_image_view /perception/debug_image

    # Test detection:
    ros2 param set /perception_node target_part SG
    ros2 service call /detect_part std_srvs/srv/Trigger "{}"
    ros2 service call /detect_all std_srvs/srv/Trigger "{}"
"""

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    perception_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sensor",
        "gazebo_camera_detector.py",
    )

    return LaunchDescription([
        Node(
            package=None,
            executable=perception_script,
            name="perception_node",
            output="screen",
            parameters=[{
                "target_part": "",
            }],
        ),
    ])
