#!/usr/bin/env python3
"""
Gazebo Classic launch: xArm6 + xArm gripper in a single-table world.

Usage:
    ros2 launch xarm_gazebo xarm6_single_gazebo.launch.py
"""

import os
from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file, get_xacro_content


def _build_xarm6_description(context, prefix):
    ros2_control_params = generate_ros2_control_params_temp_file(
        os.path.join(
            get_package_share_directory('xarm_controller'),
            'config',
            'xarm6_controllers.yaml',
        ),
        prefix=prefix,
        add_gripper=True,
        ros_namespace='',
        update_rate=1000,
        use_sim_time=True,
        robot_type='xarm',
    )

    description = get_xacro_content(
        context,
        xacro_file=Path(get_package_share_directory('xarm_description')) / 'urdf' / 'xarm_device.urdf.xacro',
        dof='6',
        robot_type='xarm',
        prefix=prefix,
        hw_ns='xarm',
        limited=True,
        effort_control=False,
        velocity_control=False,
        attach_to='world',
        attach_xyz='"0 0 1.021"',
        attach_rpy='"0 0 3.142"',
        add_gripper='true',
        ros2_control_plugin='gazebo_ros2_control/GazeboSystem',
        ros2_control_params=ros2_control_params,
    )

    return description.replace(
        'package://xarm_description',
        f"file://{get_package_share_directory('xarm_description')}",
    )


def launch_setup(context, *args, **kwargs):
    prefix = 'xarm6_'

    gazebo_world = PathJoinSubstitution([FindPackageShare('xarm_gazebo'), 'worlds', 'single_table.world'])
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'])
        ),
        launch_arguments={
            'world': gazebo_world,
            'server_required': 'true',
            'gui_required': 'false',
        }.items(),
    )

    robot_description = _build_xarm6_description(context, prefix)

    state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True, 'robot_description': robot_description}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-topic', '/robot_description',
            '-entity', 'xarm6',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0',
        ],
    )

    controllers = [
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[f'{prefix}xarm6_traj_controller', '--controller-manager', '/controller_manager'],
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[f'{prefix}xarm_gripper_traj_controller', '--controller-manager', '/controller_manager'],
            parameters=[{'use_sim_time': True}],
        ),
    ]

    return [
        gazebo,
        state_publisher,
        TimerAction(period=25.0, actions=[spawn]),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn,
                on_exit=controllers,
            )
        ),
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
