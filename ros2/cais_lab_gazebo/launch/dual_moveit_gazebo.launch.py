#!/usr/bin/env python3
"""
All-in-one: Gazebo Classic + MoveIt2 for BOTH xArm6 and UR5e in ONE move_group.

A single MoveIt move_group manages both robots. In RViz, toggle between
planning groups to control each robot:
  - xarm6_xarm6           → xArm6 arm (6-DOF)
  - xarm6_xarm_gripper    → xArm6 gripper
  - ur5e_ur_manipulator   → UR5e arm (6-DOF)

Usage:
    ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ROBOT_BASE_Y = 0.50
from uf_ros_lib.uf_robot_utils import (
    get_xacro_content,
    generate_ros2_control_params_temp_file,
)

SPEED_LIMIT_SCALE = 3.0
ACC_LIMIT_SCALE = 2.5
DEFAULT_VELOCITY_SCALING = 1.0
DEFAULT_ACCELERATION_SCALING = 1.0
RG2_MAX_VELOCITY = 0.40
RG2_MAX_ACCELERATION = 1.50


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
    """Remove world/ground_plane links and their joints from a URDF XML tree."""
    strip = {'world', 'ground_plane'}
    for joint in list(root.findall('joint')):
        parent = joint.find('parent')
        child = joint.find('child')
        p = parent.get('link') if parent is not None else ''
        c = child.get('link') if child is not None else ''
        if p in strip or c in strip:
            root.remove(joint)
    for link in list(root.findall('link')):
        if link.get('name') in strip:
            root.remove(link)


def _build_combined_urdf(context, xarm_prefix, ur5e_prefix):
    """Build a combined URDF with both robots (kinematics for MoveIt)."""

    # ── xArm6 URDF fragment ─────────────────────────────────────────────────
    xarm_ros2_ctrl = generate_ros2_control_params_temp_file(
        os.path.join(
            get_package_share_directory('xarm_controller'),
            'config', 'xarm6_controllers.yaml',
        ),
        prefix=xarm_prefix,
        add_gripper=True,
        ros_namespace='',
        update_rate=1000,
        use_sim_time=True,
        robot_type='xarm',
    )
    xarm_desc = get_xacro_content(
        context,
        xacro_file=Path(get_package_share_directory('xarm_description'))
        / 'urdf' / 'xarm_device.urdf.xacro',
        dof='6',
        robot_type='xarm',
        prefix=xarm_prefix,
        hw_ns='xarm',
        limited=True,
        effort_control=False,
        velocity_control=False,
        add_gripper='true',
        ros2_control_plugin='gazebo_ros2_control/GazeboSystem',
        ros2_control_params=xarm_ros2_ctrl,
    )
    xarm_root = ET.fromstring(xarm_desc)
    _strip_world_and_ground(xarm_root)

    # ── UR5e URDF fragment ───────────────────────────────────────────────────
    ur5e_raw = subprocess.check_output([
        'xacro',
        str(Path(get_package_share_directory('ur_description'))
            / 'urdf' / 'ur.urdf.xacro'),
        'ur_type:=ur5e', 'name:=ur5e',
        f'tf_prefix:={ur5e_prefix}',
        'use_fake_hardware:=true',
    ]).decode('utf-8')
    ur5e_root = ET.fromstring(ur5e_raw)
    _strip_world_and_ground(ur5e_root)

    # Inject OnRobot RG2 gripper onto UR5e
    onrobot_prefix = f'{ur5e_prefix}rg2_'
    try:
        onrobot_raw = subprocess.check_output([
            'xacro',
            str(Path(get_package_share_directory('onrobot_description'))
                / 'urdf' / 'onrobot.urdf.xacro'),
            'onrobot_type:=rg2', 'name:=rg2',
            f'prefix:={onrobot_prefix}',
            'sim_gazebo:=true',
        ]).decode('utf-8')
        onrobot_root = ET.fromstring(onrobot_raw)

        # Gazebo drops massless links, which breaks the finger_width mock joint and mimic plugin.
        # Inject a small mass into the mock link so Gazebo keeps it.
        for link in onrobot_root.findall('link'):
            if 'finger_width_mock_link' in link.get('name', ''):
                if link.find('inertial') is None:
                    inertial = ET.SubElement(link, 'inertial')
                    ET.SubElement(inertial, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
                    ET.SubElement(inertial, 'mass', {'value': '0.01'})
                    ET.SubElement(inertial, 'inertia', {'ixx': '0.0001', 'ixy': '0.0', 'ixz': '0.0', 'iyy': '0.0001', 'iyz': '0.0', 'izz': '0.0001'})

        # Strip duplicate gazebo plugins
        for gazebo_elem in list(onrobot_root.findall('gazebo')):
            plugin = gazebo_elem.find('plugin')
            if plugin is not None and 'gazebo_ros2_control' in (plugin.get('filename', '') + plugin.get('name', '')):
                onrobot_root.remove(gazebo_elem)

        for link in list(onrobot_root.findall('link')):
            if link.get('name') == 'world':
                onrobot_root.remove(link)
        for joint in list(onrobot_root.findall('joint')):
            p = joint.find('parent')
            c = joint.find('child')
            if ((p is not None and p.get('link') == 'world')
                    or (c is not None and c.get('link') == 'world')):
                onrobot_root.remove(joint)
        # Do not convert RG2 joints to fixed because we want MoveIt to control them!
        
        for elem in list(onrobot_root):
            ur5e_root.append(elem)
        mj = ET.Element('joint', {
            'name': f'{ur5e_prefix}gripper_mount_joint', 'type': 'fixed'})
        ET.SubElement(mj, 'parent', {'link': f'{ur5e_prefix}tool0'})
        ET.SubElement(mj, 'child',
                      {'link': f'{onrobot_prefix}onrobot_base_link'})
        ET.SubElement(mj, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 -1.57079632679'})
        ur5e_root.append(mj)
    except Exception:
        pass

    # ── Merge into combined URDF ─────────────────────────────────────────────
    combined = ET.Element('robot', {'name': 'dual_robot_moveit'})
    ET.SubElement(combined, 'link', {'name': 'world'})

    # xArm6 at (0, -0.62, 1.021) with 180° yaw
    xj = ET.SubElement(combined, 'joint', {
        'name': f'{xarm_prefix}world_joint', 'type': 'fixed'})
    ET.SubElement(xj, 'parent', {'link': 'world'})
    ET.SubElement(xj, 'child', {'link': f'{xarm_prefix}link_base'})
    ET.SubElement(xj, 'origin', {'xyz': f'0.0 {-ROBOT_BASE_Y} 1.021', 'rpy': '0 0 3.142'})
    for elem in list(xarm_root):
        combined.append(elem)

    # UR5e at (0, 0.62, 1.021) with 180° yaw
    uj = ET.SubElement(combined, 'joint', {
        'name': f'{ur5e_prefix}world_joint', 'type': 'fixed'})
    ET.SubElement(uj, 'parent', {'link': 'world'})
    ET.SubElement(uj, 'child', {'link': f'{ur5e_prefix}base_link'})
    ET.SubElement(uj, 'origin', {'xyz': f'0.0 {ROBOT_BASE_Y} 1.021', 'rpy': '0 0 3.142'})
    for elem in list(ur5e_root):
        combined.append(elem)

    return ET.tostring(combined, encoding='unicode')


def _build_combined_srdf(xarm_prefix, ur5e_prefix):
    """Build combined SRDF by merging xArm6 and UR5e SRDFs."""
    xarm_srdf_raw = subprocess.check_output([
        'xacro',
        str(Path(get_package_share_directory('xarm_moveit_config'))
            / 'srdf' / 'xarm.srdf.xacro'),
        f'prefix:={xarm_prefix}',
        'dof:=6', 'robot_type:=xarm',
        'add_gripper:=true', 'add_vacuum_gripper:=false',
        'add_bio_gripper:=false', 'add_other_geometry:=false',
    ]).decode('utf-8')

    ur5e_srdf_raw = subprocess.check_output([
        'xacro',
        str(Path(get_package_share_directory('ur_moveit_config'))
            / 'srdf' / 'ur.srdf.xacro'),
        'name:=ur', f'prefix:={ur5e_prefix}',
    ]).decode('utf-8')

    xarm_srdf = ET.fromstring(xarm_srdf_raw)
    ur5e_srdf = ET.fromstring(ur5e_srdf_raw)

    merged = ET.Element('robot', {'name': 'dual_robot_moveit'})
    for elem in list(xarm_srdf):
        merged.append(elem)
    for elem in list(ur5e_srdf):
        merged.append(elem)

    # ── Inject RG2 planning group ───────────────────────────────────────────
    onrobot_prefix = f'{ur5e_prefix}rg2_'
    rg2_group = ET.SubElement(merged, 'group', {'name': f'{ur5e_prefix}rg2_gripper'})
    ET.SubElement(rg2_group, 'joint', {'name': f'{onrobot_prefix}finger_width'})

    # Predefined "open" state
    state_open = ET.SubElement(merged, 'group_state', {'name': 'open', 'group': f'{ur5e_prefix}rg2_gripper'})
    ET.SubElement(state_open, 'joint', {'name': f'{onrobot_prefix}finger_width', 'value': '0.11'})

    # Predefined "close" state. Keep a non-zero gap so RViz "close" doesn't overcrush parts.
    state_close = ET.SubElement(merged, 'group_state', {'name': 'close', 'group': f'{ur5e_prefix}rg2_gripper'})
    ET.SubElement(state_close, 'joint', {'name': f'{onrobot_prefix}finger_width', 'value': '0.020'})

    # Disable collisions for all OnRobot RG2 gripper link pairs
    onrobot_prefix = f'{ur5e_prefix}rg2_'
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
    # Also disable collisions between RG2 links and UR5e tool links
    ur5e_adjacent = [
        f'{ur5e_prefix}tool0',
        f'{ur5e_prefix}wrist_3_link',
        f'{ur5e_prefix}wrist_2_link',
    ]
    all_rg2_related = rg2_links + ur5e_adjacent
    for i, link1 in enumerate(all_rg2_related):
        for link2 in all_rg2_related[i + 1:]:
            ET.SubElement(merged, 'disable_collisions', {
                'link1': link1, 'link2': link2, 'reason': 'Never'})

    return ET.tostring(merged, encoding='unicode')


def _build_moveit_params(xarm_prefix, ur5e_prefix, urdf, srdf):
    """Build all MoveIt parameters for the combined dual-robot setup."""
    from ur_moveit_config.launch_common import load_yaml

    # ── Kinematics (one entry per planning group) ────────────────────────────
    kinematics = {
        f'{xarm_prefix}xarm6': {
            'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
            'kinematics_solver_search_resolution': 0.005,
            'kinematics_solver_timeout': 0.005,
            'kinematics_solver_attempts': 3,
        },
        f'{ur5e_prefix}ur_manipulator': {
            'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
            'kinematics_solver_search_resolution': 0.005,
            'kinematics_solver_timeout': 0.005,
            'kinematics_solver_attempts': 3,
        },
    }

    # ── Joint limits ─────────────────────────────────────────────────────────
    combined_limits = {}
    xarm_lim = load_yaml('xarm_moveit_config', 'config/xarm6/joint_limits.yaml')
    if 'joint_limits' in xarm_lim:
        for j, v in _scale_joint_limits(xarm_lim['joint_limits']).items():
            combined_limits[f'{xarm_prefix}{j}'] = v
    xarm_gripper_lim = load_yaml('xarm_moveit_config', 'config/xarm_gripper/joint_limits.yaml')
    if 'joint_limits' in xarm_gripper_lim:
        for j, v in _scale_joint_limits(xarm_gripper_lim['joint_limits']).items():
            combined_limits[f'{xarm_prefix}{j}'] = v
    ur5e_lim = load_yaml('ur_moveit_config', 'config/joint_limits.yaml')
    if 'joint_limits' in ur5e_lim:
        for j, v in _scale_joint_limits(ur5e_lim['joint_limits']).items():
            combined_limits[f'{ur5e_prefix}{j}'] = v
    combined_limits[f'{ur5e_prefix}rg2_finger_width'] = {
        'has_position_limits': True,
        'min_position': 0.015,
        'max_position': 0.110,
        'has_velocity_limits': True,
        'max_velocity': RG2_MAX_VELOCITY,
        'has_acceleration_limits': True,
        'max_acceleration': RG2_MAX_ACCELERATION,
    }

    # ── OMPL planning pipeline ───────────────────────────────────────────────
    # Merge planner type definitions from both packages
    xarm_ompl_defs = load_yaml(
        'xarm_moveit_config', 'config/moveit_configs/ompl_defaults.yaml')
    ur5e_ompl = load_yaml('ur_moveit_config', 'config/ompl_planning.yaml')

    planner_configs = {}
    if 'planner_configs' in xarm_ompl_defs:
        planner_configs.update(xarm_ompl_defs['planner_configs'])
    if 'planner_configs' in ur5e_ompl:
        planner_configs.update(ur5e_ompl['planner_configs'])

    # Per-group OMPL configs
    xarm_ompl_group = load_yaml(
        'xarm_moveit_config', 'config/xarm6/ompl_planning.yaml')
    xarm_gripper_ompl = load_yaml(
        'xarm_moveit_config', 'config/xarm_gripper/ompl_planning.yaml')

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
    if 'xarm6' in xarm_ompl_group:
        ompl['move_group'][f'{xarm_prefix}xarm6'] = xarm_ompl_group['xarm6']
    if 'xarm_gripper' in xarm_gripper_ompl:
        ompl['move_group'][f'{xarm_prefix}xarm_gripper'] = \
            xarm_gripper_ompl['xarm_gripper']
    if 'ur_manipulator' in ur5e_ompl:
        ompl['move_group'][f'{ur5e_prefix}ur_manipulator'] = \
            ur5e_ompl['ur_manipulator']
    if 'xarm_gripper' in xarm_gripper_ompl:
        ompl['move_group'][f'{ur5e_prefix}rg2_gripper'] = xarm_gripper_ompl['xarm_gripper']

    # ── MoveIt controllers ───────────────────────────────────────────────────
    xarm_joints = [f'{xarm_prefix}joint{i}' for i in range(1, 7)]
    ur5e_joints = [
        f'{ur5e_prefix}shoulder_pan_joint',
        f'{ur5e_prefix}shoulder_lift_joint',
        f'{ur5e_prefix}elbow_joint',
        f'{ur5e_prefix}wrist_1_joint',
        f'{ur5e_prefix}wrist_2_joint',
        f'{ur5e_prefix}wrist_3_joint',
    ]

    controllers = {
        'moveit_simple_controller_manager': {
            'controller_names': [
                f'{xarm_prefix}xarm6_traj_controller',
                f'{xarm_prefix}xarm_gripper_traj_controller',
                'ur5e_joint_trajectory_controller',
                'ur5e_rg2_gripper_traj_controller',
            ],
            f'{xarm_prefix}xarm6_traj_controller': {
                'action_ns': 'follow_joint_trajectory',
                'type': 'FollowJointTrajectory',
                'default': True,
                'joints': xarm_joints,
            },
            f'{xarm_prefix}xarm_gripper_traj_controller': {
                'action_ns': 'follow_joint_trajectory',
                'type': 'FollowJointTrajectory',
                'default': True,
                'joints': [f'{xarm_prefix}drive_joint'],
            },
            'ur5e_joint_trajectory_controller': {
                'action_ns': 'follow_joint_trajectory',
                'type': 'FollowJointTrajectory',
                'default': True,
                'joints': ur5e_joints,
            },
            'ur5e_rg2_gripper_traj_controller': {
                'action_ns': 'follow_joint_trajectory',
                'type': 'FollowJointTrajectory',
                'default': True,
                'joints': [f'{ur5e_prefix}rg2_finger_width'],
            },
        },
        'moveit_controller_manager':
            'moveit_simple_controller_manager/MoveItSimpleControllerManager',
    }

    # ── Trajectory execution ─────────────────────────────────────────────────
    traj_exec = {
        'moveit_manage_controllers': False,
        'trajectory_execution.allowed_execution_duration_scaling': 2.0,
        'trajectory_execution.allowed_goal_duration_margin': 1.0,
        'trajectory_execution.allowed_start_tolerance': 0.0,
        'trajectory_execution.execution_duration_monitoring': False,
    }

    # ── Planning scene monitor ───────────────────────────────────────────────
    psm = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    # ── Assemble ─────────────────────────────────────────────────────────────
    config = {
        'robot_description': urdf,
        'robot_description_semantic': srdf,
        'robot_description_kinematics': kinematics,
        'robot_description_planning': {
            'default_velocity_scaling_factor': DEFAULT_VELOCITY_SCALING,
            'default_acceleration_scaling_factor': DEFAULT_ACCELERATION_SCALING,
            'joint_limits': combined_limits,
        },
    }
    config.update(ompl)
    config.update(controllers)
    config.update(traj_exec)
    config.update(psm)
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════════

def _launch_arg_enabled(context, name, default='false'):
    value = LaunchConfiguration(name, default=default).perform(context)
    return str(value or '').strip().lower() in {'1', 'true', 'yes', 'on'}


def launch_setup(context, *args, **kwargs):
    xarm_prefix = 'xarm6_'
    ur5e_prefix = 'ur5e_'
    fast_sim = LaunchConfiguration('fast_sim')
    launch_rviz = _launch_arg_enabled(context, 'launch_rviz', default='true')

    # 1. Gazebo with combined URDF (both robots have physics)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('xarm_gazebo'), 'launch',
                'xarm6_ur5e_gazebo.launch.py',
            ])
        ),
        launch_arguments={
            'fast_sim': fast_sim,
        }.items(),
    )

    # 2. Build combined MoveIt config
    combined_urdf = _build_combined_urdf(context, xarm_prefix, ur5e_prefix)
    combined_srdf = _build_combined_srdf(xarm_prefix, ur5e_prefix)
    moveit_config = _build_moveit_params(
        xarm_prefix, ur5e_prefix, combined_urdf, combined_srdf)

    # 3. Single move_group at ROOT namespace (no namespace prefix issues)
    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group',
        output='screen',
        parameters=[
            moveit_config,
            {'use_sim_time': True},
        ],
    )

    # 4. Single RViz — toggle Planning Group dropdown to switch robots:
    #    xarm6_xarm6, xarm6_xarm_gripper, ur5e_ur_manipulator
    rviz_config = PathJoinSubstitution([
        FindPackageShare('xarm_gazebo'), 'rviz', 'dual_moveit.rviz',
    ])
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config],
        parameters=[
            {
                'robot_description': combined_urdf,
                'robot_description_semantic': combined_srdf,
                'robot_description_kinematics':
                    moveit_config['robot_description_kinematics'],
                'robot_description_planning':
                    moveit_config.get('robot_description_planning', {}),
                'use_sim_time': True,
            },
        ],
    )

    launch_actions = [
        gazebo,
        move_group,
    ]
    if launch_rviz:
        launch_actions.append(rviz)
    return launch_actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'fast_sim',
            default_value='false',
            description='Use the fast Gazebo world timing profile.',
        ),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
            description='Launch RViz alongside Gazebo and MoveIt.',
        ),
        OpaqueFunction(function=launch_setup),
    ])
