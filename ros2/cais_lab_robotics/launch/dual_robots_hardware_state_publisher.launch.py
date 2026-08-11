#!/usr/bin/env python3
"""Launch only the combined xArm6 + UR5e hardware robot state publisher."""

from __future__ import annotations

from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description() -> LaunchDescription:
    """Return the hardware model launch with MoveIt and RViz disabled."""
    launch_file = (
        Path(get_package_share_directory("cais_lab_robotics"))
        / "launch"
        / "dual_robots_hardware_moveit.launch.py"
    )
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(launch_file)),
                launch_arguments={
                    "launch_move_group": "false",
                    "launch_rviz": "false",
                }.items(),
            )
        ]
    )
