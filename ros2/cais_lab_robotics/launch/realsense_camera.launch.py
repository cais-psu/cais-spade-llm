#!/usr/bin/env python3
"""Start the wrist RealSense with color-aligned depth and its TF tree."""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Build the RealSense driver launch description."""
    serial_default = str(os.environ.get("REALSENSE_SERIAL", "")).strip()
    serial_no = LaunchConfiguration("serial_no")
    camera_name = LaunchConfiguration("camera_name")
    camera_namespace = LaunchConfiguration("camera_namespace")
    color_profile = LaunchConfiguration("rgb_camera.color_profile")
    depth_profile = LaunchConfiguration("depth_module.depth_profile")
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments={
            # The RealSense wrapper parses launch values as YAML. Preserve the
            # all-numeric device serial as a string instead of an integer.
            "serial_no": [
                TextSubstitution(text="'"),
                serial_no,
                TextSubstitution(text="'"),
            ],
            "camera_name": camera_name,
            "camera_namespace": camera_namespace,
            "enable_color": "true",
            "enable_depth": "true",
            "rgb_camera.color_profile": color_profile,
            "depth_module.depth_profile": depth_profile,
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
            DeclareLaunchArgument(
                "camera_name",
                default_value="camera",
                description="Unique RealSense ROS node and TF frame prefix",
            ),
            DeclareLaunchArgument(
                "camera_namespace",
                default_value="camera",
                description="Unique RealSense ROS namespace",
            ),
            DeclareLaunchArgument(
                "rgb_camera.color_profile",
                default_value="640x480x6",
                description="Color profile chosen for WSL USB bandwidth",
            ),
            DeclareLaunchArgument(
                "depth_module.depth_profile",
                default_value="640x480x6",
                description="Depth profile chosen for WSL USB bandwidth",
            ),
            camera_launch,
        ]
    )
