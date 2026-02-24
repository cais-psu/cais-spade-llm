#!/usr/bin/env python3
"""
All-in-one: UR5e + OnRobot RG2 in Gazebo Classic with MoveIt2.

Usage:
    ros2 launch xarm_gazebo ur5e_rg2_moveit_gazebo.launch.py
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

SPEED_LIMIT_SCALE = 2.5
ACC_LIMIT_SCALE = 2.0
DEFAULT_VELOCITY_SCALING = 1.0
DEFAULT_ACCELERATION_SCALING = 1.0
RG2_MAX_VELOCITY = 1.5
RG2_MAX_ACCELERATION = 8.0


def _scale_joint_limits(joint_limits):
    scaled = {}
    for joint_name, limits in joint_limits.items():
        jl = dict(limits)
        if jl.get('has_velocity_limits', False) and 'max_velocity' in jl:
            jl['max_velocity'] = float(jl['max_velocity']) * SPEED_LIMIT_SCALE
        if jl.get('has_acceleration_limits', False) and 'max_acceleration' in jl:
            jl['max_acceleration'] = float(jl['max_acceleration']) * ACC_LIMIT_SCALE
        scaled[joint_name] = jl
    return scaled


def _strip_world_and_ground(root):
    strip_names = {'world', 'ground_plane'}
    for joint in list(root.findall('joint')):
        parent = joint.find('parent')
        child = joint.find('child')
        p_name = parent.get('link') if parent is not None else ''
        c_name = child.get('link') if child is not None else ''
        if p_name in strip_names or c_name in strip_names:
            root.remove(joint)
    for link in list(root.findall('link')):
        if link.get('name') in strip_names:
            root.remove(link)


def _build_urdf(prefix):
    ur5e_raw = subprocess.check_output([
        'xacro',
        str(Path(get_package_share_directory('ur_description')) / 'urdf' / 'ur.urdf.xacro'),
        'ur_type:=ur5e',
        'name:=ur5e',
        f'tf_prefix:={prefix}',
        'use_fake_hardware:=true',
    ]).decode('utf-8')
    ur5e_root = ET.fromstring(ur5e_raw)
    _strip_world_and_ground(ur5e_root)

    onrobot_prefix = f'{prefix}rg2_'
    onrobot_raw = subprocess.check_output([
        'xacro',
        str(Path(get_package_share_directory('onrobot_description')) / 'urdf' / 'onrobot.urdf.xacro'),
        'onrobot_type:=rg2',
        'name:=rg2',
        f'prefix:={onrobot_prefix}',
        'sim_gazebo:=false',
    ]).decode('utf-8')
    onrobot_root = ET.fromstring(onrobot_raw)
    _strip_world_and_ground(onrobot_root)

    for elem in list(onrobot_root):
        ur5e_root.append(elem)

    mounting_joint = ET.Element('joint', {'name': f'{prefix}gripper_mount_joint', 'type': 'fixed'})
    ET.SubElement(mounting_joint, 'parent', {'link': f'{prefix}tool0'})
    ET.SubElement(mounting_joint, 'child', {'link': f'{onrobot_prefix}onrobot_base_link'})
    ET.SubElement(mounting_joint, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
    ur5e_root.append(mounting_joint)

    combined = ET.Element('robot', {'name': 'ur5e_rg2_moveit'})
    ET.SubElement(combined, 'link', {'name': 'world'})

    world_joint = ET.SubElement(combined, 'joint', {'name': f'{prefix}world_joint', 'type': 'fixed'})
    ET.SubElement(world_joint, 'parent', {'link': 'world'})
    ET.SubElement(world_joint, 'child', {'link': f'{prefix}base_link'})
    ET.SubElement(world_joint, 'origin', {'xyz': '0.0 0.0 1.021', 'rpy': '0 0 3.142'})

    for elem in list(ur5e_root):
        combined.append(elem)
    return ET.tostring(combined, encoding='unicode')


def _build_srdf(prefix):
    ur_srdf_raw = subprocess.check_output([
        'xacro',
        str(Path(get_package_share_directory('ur_moveit_config')) / 'srdf' / 'ur.srdf.xacro'),
        'name:=ur',
        f'prefix:={prefix}',
    ]).decode('utf-8')
    srdf_root = ET.fromstring(ur_srdf_raw)
    srdf_root.attrib['name'] = 'ur5e_rg2_moveit'

    onrobot_prefix = f'{prefix}rg2_'
    rg2_group = ET.SubElement(srdf_root, 'group', {'name': f'{prefix}rg2_gripper'})
    ET.SubElement(rg2_group, 'joint', {'name': f'{onrobot_prefix}finger_width'})

    open_state = ET.SubElement(srdf_root, 'group_state', {'name': 'open', 'group': f'{prefix}rg2_gripper'})
    ET.SubElement(open_state, 'joint', {'name': f'{onrobot_prefix}finger_width', 'value': '0.11'})

    close_state = ET.SubElement(srdf_root, 'group_state', {'name': 'close', 'group': f'{prefix}rg2_gripper'})
    ET.SubElement(close_state, 'joint', {'name': f'{onrobot_prefix}finger_width', 'value': '0.0'})

    rg2_links = [
        f'{onrobot_prefix}onrobot_base_link',
        f'{onrobot_prefix}left_outer_knuckle',
        f'{onrobot_prefix}right_outer_knuckle',
        f'{onrobot_prefix}left_inner_knuckle',
        f'{onrobot_prefix}right_inner_knuckle',
        f'{onrobot_prefix}left_inner_finger',
        f'{onrobot_prefix}right_inner_finger',
        f'{onrobot_prefix}left_finger_tip',
        f'{onrobot_prefix}right_finger_tip',
        f'{onrobot_prefix}finger_width_mock_link',
        f'{onrobot_prefix}gripper_tcp',
    ]
    ur_links = [
        f'{prefix}tool0',
        f'{prefix}wrist_3_link',
        f'{prefix}wrist_2_link',
    ]
    disabled_links = rg2_links + ur_links
    for index, link1 in enumerate(disabled_links):
        for link2 in disabled_links[index + 1:]:
            ET.SubElement(
                srdf_root,
                'disable_collisions',
                {'link1': link1, 'link2': link2, 'reason': 'Never'},
            )

    return ET.tostring(srdf_root, encoding='unicode')


def _build_moveit_params(prefix, urdf, srdf):
    from ur_moveit_config.launch_common import load_yaml

    kinematics = {
        f'{prefix}ur_manipulator': {
            'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
            'kinematics_solver_search_resolution': 0.005,
            'kinematics_solver_timeout': 0.005,
            'kinematics_solver_attempts': 3,
        }
    }

    prefixed_limits = {}
    joint_limits = load_yaml('ur_moveit_config', 'config/joint_limits.yaml')
    if 'joint_limits' in joint_limits:
        for joint_name, limits in _scale_joint_limits(joint_limits['joint_limits']).items():
            prefixed_limits[f'{prefix}{joint_name}'] = limits

    prefixed_limits[f'{prefix}rg2_finger_width'] = {
        'has_velocity_limits': True,
        'max_velocity': RG2_MAX_VELOCITY,
        'has_acceleration_limits': True,
        'max_acceleration': RG2_MAX_ACCELERATION,
    }

    ur_ompl = load_yaml('ur_moveit_config', 'config/ompl_planning.yaml')
    planner_configs = ur_ompl.get(
        'planner_configs',
        {'RRTConnectkConfigDefault': {'type': 'geometric::RRTConnect'}},
    )

    ompl = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': (
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/FixStartStatePathConstraints'
            ),
            'start_state_max_bounds_error': 0.1,
            'planner_configs': planner_configs,
        }
    }
    if 'ur_manipulator' in ur_ompl:
        ompl['move_group'][f'{prefix}ur_manipulator'] = ur_ompl['ur_manipulator']
    ompl['move_group'][f'{prefix}rg2_gripper'] = {
        'planner_configs': ['RRTConnectkConfigDefault'],
        'projection_evaluator': f'joints({prefix}rg2_finger_width)',
        'longest_valid_segment_fraction': 0.005,
    }

    controllers = {
        'moveit_simple_controller_manager': {
            'controller_names': [
                'ur5e_joint_trajectory_controller',
                'ur5e_rg2_gripper_traj_controller',
            ],
            'ur5e_joint_trajectory_controller': {
                'action_ns': 'follow_joint_trajectory',
                'type': 'FollowJointTrajectory',
                'default': True,
                'joints': [
                    f'{prefix}shoulder_pan_joint',
                    f'{prefix}shoulder_lift_joint',
                    f'{prefix}elbow_joint',
                    f'{prefix}wrist_1_joint',
                    f'{prefix}wrist_2_joint',
                    f'{prefix}wrist_3_joint',
                ],
            },
            'ur5e_rg2_gripper_traj_controller': {
                'action_ns': 'follow_joint_trajectory',
                'type': 'FollowJointTrajectory',
                'default': True,
                'joints': [f'{prefix}rg2_finger_width'],
            },
        },
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
    }

    trajectory_execution = {
        'moveit_manage_controllers': False,
        'trajectory_execution.allowed_execution_duration_scaling': 2.0,
        'trajectory_execution.allowed_goal_duration_margin': 1.0,
        'trajectory_execution.allowed_start_tolerance': 0.0,
        'trajectory_execution.execution_duration_monitoring': False,
    }

    planning_scene_monitor = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    config = {
        'robot_description': urdf,
        'robot_description_semantic': srdf,
        'robot_description_kinematics': kinematics,
        'robot_description_planning': {
            'default_velocity_scaling_factor': DEFAULT_VELOCITY_SCALING,
            'default_acceleration_scaling_factor': DEFAULT_ACCELERATION_SCALING,
            'joint_limits': prefixed_limits,
        },
    }
    config.update(ompl)
    config.update(controllers)
    config.update(trajectory_execution)
    config.update(planning_scene_monitor)
    return config


def launch_setup(context, *args, **kwargs):
    prefix = 'ur5e_'

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('xarm_gazebo'), 'launch', 'ur5e_rg2_gazebo.launch.py'])
        ),
    )

    urdf = _build_urdf(prefix)
    srdf = _build_srdf(prefix)
    moveit_config = _build_moveit_params(prefix, urdf, srdf)

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group',
        output='screen',
        parameters=[moveit_config, {'use_sim_time': True}],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=[
            '-d',
            PathJoinSubstitution([FindPackageShare('ur_moveit_config'), 'rviz', 'view_robot.rviz']),
        ],
        parameters=[
            {
                'robot_description': urdf,
                'robot_description_semantic': srdf,
                'robot_description_kinematics': moveit_config['robot_description_kinematics'],
                'robot_description_planning': moveit_config['robot_description_planning'],
                'use_sim_time': True,
            },
        ],
    )

    return [
        gazebo,
        TimerAction(period=40.0, actions=[move_group, rviz]),
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
