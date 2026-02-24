#!/usr/bin/env python3
"""
Combined Gazebo Classic launch: xArm6 (with gripper) + UR5e in the same world.

Namespaces / TF prefixes:
  xArm6 → namespace: xarm6,  tf_prefix: xarm6_,  pos: (0.0, -0.7, 1.021) [180° yaw (3.142)]
  UR5e  → namespace: ur5e,   tf_prefix: ur5e_,   pos: (0.0, 0.7, 1.021)  [0° yaw (0.0)]
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
from uf_ros_lib.uf_robot_utils import get_xacro_content, generate_ros2_control_params_temp_file


def launch_setup(context, *args, **kwargs):

    # ── Gazebo Classic ────────────────────────────────────────────────────────
    gazebo_world = PathJoinSubstitution(
        [FindPackageShare('xarm_gazebo'), 'worlds', 'table.world']
    )
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'])
        ),
        launch_arguments={
            'world': gazebo_world,
            'server_required': 'true',
            'gui_required': 'false',
        }.items(),
    )

    # ── xArm6 ────────────────────────────────────────────────────────────────
    xarm_prefix = 'xarm6_'
    xarm_ros2_control_params = generate_ros2_control_params_temp_file(
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
    xarm_description = get_xacro_content(
        context,
        xacro_file=Path(get_package_share_directory('xarm_description')) / 'urdf' / 'xarm_device.urdf.xacro',
        dof='6',
        robot_type='xarm',
        prefix=xarm_prefix,
        hw_ns='xarm',
        limited=False,
        effort_control=False,
        velocity_control=False,
        add_gripper='true',
        ros2_control_plugin='gazebo_ros2_control/GazeboSystem',
        ros2_control_params=xarm_ros2_control_params,
    )
    xarm_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True, 'robot_description': xarm_description}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )
    xarm_spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-topic', '/robot_description',
            '-entity', 'xarm6',
            '-x', '0.0', '-y', '-0.7', '-z', '1.021', '-Y', '3.142',
        ],
    )
    # gazebo_ros2_control starts in namespace '/', so controller_manager is at /controller_manager
    xarm_controller_nodes = [
        Node(
            package='controller_manager',
            executable='spawner',
            namespace='xarm6',
            arguments=['joint_state_broadcaster',
                       '--controller-manager', '/controller_manager'],
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            namespace='xarm6',
            arguments=[f'{xarm_prefix}xarm6_traj_controller',
                       '--controller-manager', '/controller_manager'],
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            namespace='xarm6',
            arguments=[f'{xarm_prefix}xarm_gripper_traj_controller',
                       '--controller-manager', '/controller_manager'],
            parameters=[{'use_sim_time': True}],
        ),
    ]

    # ── UR5e ─────────────────────────────────────────────────────────────────
    ur5e_prefix = 'ur5e_'
    ur5e_raw = subprocess.check_output([
        'xacro',
        str(Path(get_package_share_directory('ur_description')) / 'urdf' / 'ur.urdf.xacro'),
        'ur_type:=ur5e',
        'name:=ur5e',
        f'tf_prefix:={ur5e_prefix}',
        'use_fake_hardware:=true',
    ]).decode('utf-8')

    # Strip 'ground_plane' and 'world' links and their joints — Gazebo already has a ground_plane
    # model in the world, and 'world' forces the robot to 0,0,0 defying the spawn position.
    root = ET.fromstring(ur5e_raw)
    
    # Inject <gazebo><static>true</static></gazebo> to freeze joints and fix to world so it doesn't fall!
    gazebo_elem = ET.Element('gazebo')
    static_elem = ET.Element('static')
    static_elem.text = 'true'
    gazebo_elem.append(static_elem)
    root.append(gazebo_elem)
    
    for joint in root.findall('joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        p_name = parent.get('link') if parent is not None else ''
        c_name = child.get('link') if child is not None else ''
        if p_name in ['ground_plane', 'world'] or c_name in ['ground_plane', 'world']:
            root.remove(joint)
    for link in root.findall('link'):
        l_name = link.get('name')
        if l_name in ['ground_plane', 'world']:
            root.remove(link)

    # ── Inject OnRobot RG2 Gripper ───────────────────────────────────────────
    onrobot_prefix = f'{ur5e_prefix}rg2_'
    onrobot_raw = subprocess.check_output([
        'xacro',
        str(Path(get_package_share_directory('onrobot_description')) / 'urdf' / 'onrobot.urdf.xacro'),
        'onrobot_type:=rg2',
        'name:=rg2',
        f'prefix:={onrobot_prefix}',
        'sim_gazebo:=false',
    ]).decode('utf-8')
    onrobot_root = ET.fromstring(onrobot_raw)
    
    # Strip world from RG2
    for link in onrobot_root.findall('link'):
        if link.get('name') == 'world':
            onrobot_root.remove(link)
    for joint in onrobot_root.findall('joint'):
        p = joint.find('parent')
        c = joint.find('child')
        if (p is not None and p.get('link') == 'world') or (c is not None and c.get('link') == 'world'):
            onrobot_root.remove(joint)
                
    # Append RG2 elements to UR5e tree
    for elem in list(onrobot_root):
        root.append(elem)

    # Create connecting joint: UR5e tool0 -> RG2 base_link
    mounting_joint = ET.Element('joint', {'name': f'{ur5e_prefix}gripper_mount_joint', 'type': 'fixed'})
    ET.SubElement(mounting_joint, 'parent', {'link': f'{ur5e_prefix}tool0'})
    ET.SubElement(mounting_joint, 'child', {'link': f'{onrobot_prefix}onrobot_base_link'})
    ET.SubElement(mounting_joint, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
    root.append(mounting_joint)
    # ─────────────────────────────────────────────────────────────────────────

    ur5e_description = ET.tostring(root, encoding='unicode')
    # Resolve package:// paths to absolute file:// paths so Gazebo can find the meshes
    ur_desc_path = get_package_share_directory('ur_description')
    onrobot_desc_path = get_package_share_directory('onrobot_description')
    ur5e_description = ur5e_description.replace('package://ur_description', f'file://{ur_desc_path}')
    ur5e_description = ur5e_description.replace('package://onrobot_description', f'file://{onrobot_desc_path}')

    ur5e_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='ur5e',
        output='screen',
        parameters=[{'use_sim_time': True, 'robot_description': ur5e_description}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )
    ur5e_spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-topic', '/ur5e/robot_description',
            '-entity', 'ur5e',
            '-x', '0.0', '-y', '0.7', '-z', '1.021', '-Y', '0.0',
        ],
    )
    return [
        gazebo_launch,
        xarm_rsp,
        ur5e_rsp,
        # Delay spawns 30 s to give Gazebo (WSL) time to fully initialize
        TimerAction(period=30.0, actions=[xarm_spawn]),
        TimerAction(period=32.0, actions=[ur5e_spawn]),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=xarm_spawn,
                on_exit=xarm_controller_nodes,
            )
        ),
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
