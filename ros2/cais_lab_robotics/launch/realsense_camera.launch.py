#!/usr/bin/env python3
"""Start the wrist RealSense with color-aligned depth and its TF tree."""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Build the RealSense driver launch description."""
    serial_default = str(os.environ.get("REALSENSE_SERIAL", "")).strip()
    serial_no = LaunchConfiguration("serial_no")
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments={
            "serial_no": serial_no,
            "enable_color": "true",
            "enable_depth": "true",
            "align_depth.enable": "true",
            "enable_sync": "true",
            "publish_tf": "true",
            "pointcloud.enable": "false",
        }.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "serial_no",
                default_value=serial_default,
                description="RealSense serial number; empty selects the connected camera",
            ),
            camera_launch,
        ]
    )
