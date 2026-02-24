#!/usr/bin/env python3
"""
All-in-one: xArm6 + xArm gripper in Gazebo Classic with MoveIt2.

Usage:
    ros2 launch xarm_gazebo xarm6_moveit_single_gazebo.launch.py
"""

import os
import yaml

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file

SPEED_LIMIT_SCALE = 2.5
ACC_LIMIT_SCALE = 2.0
DEFAULT_VELOCITY_SCALING = 1.0
DEFAULT_ACCELERATION_SCALING = 1.0


def _scale_joint_limits(joint_limits):
    for limits in joint_limits.values():
        if limits.get('has_velocity_limits', False) and 'max_velocity' in limits:
            limits['max_velocity'] = float(limits['max_velocity']) * SPEED_LIMIT_SCALE
        if limits.get('has_acceleration_limits', False) and 'max_acceleration' in limits:
            limits['max_acceleration'] = float(limits['max_acceleration']) * ACC_LIMIT_SCALE


def launch_setup(context, *args, **kwargs):
    prefix = 'xarm6_'

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

    moveit_config = MoveItConfigsBuilder(
        context=context,
        controllers_name='fake_controllers',
        dof='6',
        robot_type='xarm',
        prefix=prefix,
        hw_ns='xarm',
        limited=True,
        effort_control=False,
        velocity_control=False,
        model1300=False,
        robot_sn='',
        attach_to='world',
        attach_xyz='"0 0 1.021"',
        attach_rpy='"0 0 3.142"',
        mesh_suffix='stl',
        kinematics_suffix='',
        ros2_control_plugin='gazebo_ros2_control/GazeboSystem',
        ros2_control_params=ros2_control_params,
        add_gripper=True,
        add_vacuum_gripper=False,
        add_bio_gripper=False,
        add_realsense_d435i=False,
        add_d435i_links=True,
        add_other_geometry=False,
        geometry_type='box',
        geometry_mass=0.1,
        geometry_height=0.1,
        geometry_radius=0.1,
        geometry_length=0.1,
        geometry_width=0.1,
        geometry_mesh_filename='',
        geometry_mesh_origin_xyz='"0 0 0"',
        geometry_mesh_origin_rpy='"0 0 0"',
        geometry_mesh_tcp_xyz='"0 0 0"',
        geometry_mesh_tcp_rpy='"0 0 0"',
    ).to_moveit_configs()

    moveit_config_dict = moveit_config.to_dict()
    planning_cfg = moveit_config_dict.setdefault('robot_description_planning', {})
    planning_cfg['default_velocity_scaling_factor'] = DEFAULT_VELOCITY_SCALING
    planning_cfg['default_acceleration_scaling_factor'] = DEFAULT_ACCELERATION_SCALING
    if 'joint_limits' in planning_cfg:
        _scale_joint_limits(planning_cfg['joint_limits'])

    moveit_config.trajectory_execution.update({
        'trajectory_execution.allowed_start_tolerance': 0.0,
        'trajectory_execution.allowed_execution_duration_scaling': 2.0,
        'trajectory_execution.allowed_goal_duration_margin': 1.0,
        'trajectory_execution.execution_duration_monitoring': False,
    })
    moveit_config_dict['trajectory_execution'] = moveit_config.trajectory_execution
    moveit_config_dump = yaml.dump(moveit_config_dict)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('xarm_gazebo'), 'launch', 'xarm6_single_gazebo.launch.py'])
        ),
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('xarm_moveit_config'), 'launch', '_robot_moveit_common2.launch.py'])
        ),
        launch_arguments={
            'prefix': prefix,
            'attach_to': 'world',
            'attach_xyz': '"0 0 1.021"',
            'attach_rpy': '"0 0 3.142"',
            'no_gui_ctrl': 'false',
            'show_rviz': 'true',
            'use_sim_time': 'true',
            'moveit_config_dump': moveit_config_dump,
        }.items(),
    )

    return [
        gazebo,
        TimerAction(period=35.0, actions=[moveit]),
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
