#!/usr/bin/env python3
"""
Combined Gazebo Classic launch: xArm6 (with gripper) + UR5e in the same world.

Both robots share a single gazebo_ros2_control plugin and controller_manager,
so both have full physics simulation in Gazebo. Robot positions are encoded as
fixed joints in the combined URDF (not as spawn_entity arguments).

TF prefixes:
  xArm6 → prefix: xarm6_,  pos: (0.0, -0.7, 1.021) [180° yaw (3.142)]
  UR5e  → prefix: ur5e_,   pos: (0.0, 0.7, 1.021)  [180° yaw (3.142)]
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


def _strip_gazebo_ros2_control_plugin(root):
    """Remove <gazebo><plugin filename='...gazebo_ros2_control...'> from an XML tree."""
    for gazebo_elem in root.findall('gazebo'):
        plugin = gazebo_elem.find('plugin')
        if plugin is not None and 'gazebo_ros2_control' in (plugin.get('filename', '') + plugin.get('name', '')):
            root.remove(gazebo_elem)


def _inject_mimic_plugins(root):
    """Auto-inject gazebo_mimic_joint_plugin tags for all mimic joints in the URDF tree."""
    for joint in list(root.findall('joint')):
        mimic = joint.find('mimic')
        if mimic is not None:
            plugin_name = f"mimic_plugin_{joint.get('name')}"
            gazebo_elem = ET.SubElement(root, 'gazebo')
            plugin_elem = ET.SubElement(gazebo_elem, 'plugin', {'name': plugin_name, 'filename': 'libgazebo_mimic_joint_plugin.so'})
            # Match xarm_description's working mimic plugin convention:
            #   joint      = driving joint
            #   mimicJoint = follower joint
            ET.SubElement(plugin_elem, 'joint').text = mimic.get('joint')
            ET.SubElement(plugin_elem, 'mimicJoint').text = joint.get('name')
            ET.SubElement(plugin_elem, 'multiplier').text = mimic.get('multiplier', '1.0')
            ET.SubElement(plugin_elem, 'offset').text = mimic.get('offset', '0.0')
            ET.SubElement(plugin_elem, 'sensitiveness').text = '0.0'
            ET.SubElement(plugin_elem, 'maxEffort').text = '100.0'



def _strip_world_links_and_joints(root, extra_link_names=None):
    """Remove world/ground_plane links and their joints from a URDF XML tree."""
    strip_names = {'world', 'ground_plane'}
    if extra_link_names:
        strip_names.update(extra_link_names)
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

    # ── Combined controllers YAML ─────────────────────────────────────────────
    combined_controllers_yaml = os.path.join(
        get_package_share_directory('xarm_gazebo'),
        'config', 'xarm6_ur5e_controllers.yaml',
    )

    # ══════════════════════════════════════════════════════════════════════════
    # xArm6 URDF fragment
    # ══════════════════════════════════════════════════════════════════════════
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
    xarm_description_str = get_xacro_content(
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
    xarm_root = ET.fromstring(xarm_description_str)

    # Strip world link, world_joint, and the gazebo_ros2_control plugin
    # (combined URDF provides its own world link and single plugin)
    _strip_world_links_and_joints(xarm_root)
    _strip_gazebo_ros2_control_plugin(xarm_root)

    # Resolve xarm package:// paths to file:// for Gazebo mesh loading
    xarm_desc_path = get_package_share_directory('xarm_description')

    # ══════════════════════════════════════════════════════════════════════════
    # UR5e URDF fragment (with sim_gazebo for GazeboSystem hardware interface)
    # ══════════════════════════════════════════════════════════════════════════
    ur5e_prefix = 'ur5e_'
    ur5e_raw = subprocess.check_output([
        'xacro',
        str(Path(get_package_share_directory('ur_description')) / 'urdf' / 'ur.urdf.xacro'),
        'ur_type:=ur5e',
        'name:=ur5e',
        f'tf_prefix:={ur5e_prefix}',
        'sim_gazebo:=true',
        f'simulation_controllers:={combined_controllers_yaml}',
    ]).decode('utf-8')
    ur5e_root = ET.fromstring(ur5e_raw)

    # Strip world/ground_plane links and joints
    _strip_world_links_and_joints(ur5e_root)
    # Strip the gazebo_ros2_control plugin injected by sim_gazebo:=true
    _strip_gazebo_ros2_control_plugin(ur5e_root)
    # Do NOT inject <static>true</static> — UR5e now has physics!

    # ── Inject OnRobot RG2 Gripper onto UR5e ──────────────────────────────────
    onrobot_prefix = f'{ur5e_prefix}rg2_'
    onrobot_raw = subprocess.check_output([
        'xacro',
        str(Path(get_package_share_directory('onrobot_description')) / 'urdf' / 'onrobot.urdf.xacro'),
        'onrobot_type:=rg2',
        'name:=rg2',
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

    # Strip the duplicate gazebo_ros2_control plugin from RG2
    _strip_gazebo_ros2_control_plugin(onrobot_root)
    # Inject Gazebo mimic plugins for RG2 mimic joints
    _inject_mimic_plugins(onrobot_root)

    # Strip world from RG2
    for link in list(onrobot_root.findall('link')):
        if link.get('name') == 'world':
            onrobot_root.remove(link)
    for joint in list(onrobot_root.findall('joint')):
        p = joint.find('parent')
        c = joint.find('child')
        if (p is not None and p.get('link') == 'world') or (c is not None and c.get('link') == 'world'):
            onrobot_root.remove(joint)

    # Append RG2 elements to UR5e tree
    for elem in list(onrobot_root):
        ur5e_root.append(elem)

    # Create connecting joint: UR5e tool0 -> RG2 base_link
    mounting_joint = ET.Element('joint', {'name': f'{ur5e_prefix}gripper_mount_joint', 'type': 'fixed'})
    ET.SubElement(mounting_joint, 'parent', {'link': f'{ur5e_prefix}tool0'})
    ET.SubElement(mounting_joint, 'child', {'link': f'{onrobot_prefix}onrobot_base_link'})
    ET.SubElement(mounting_joint, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
    ur5e_root.append(mounting_joint)

    # ══════════════════════════════════════════════════════════════════════════
    # Merge into combined URDF
    # ══════════════════════════════════════════════════════════════════════════
    combined_root = ET.Element('robot', {'name': 'xarm6_ur5e_combined'})

    # World link (root of the combined kinematic tree)
    ET.SubElement(combined_root, 'link', {'name': 'world'})

    # xArm6 attachment: world -> xarm6_link_base
    xarm_attach = ET.SubElement(combined_root, 'joint',
                                {'name': f'{xarm_prefix}world_joint', 'type': 'fixed'})
    ET.SubElement(xarm_attach, 'parent', {'link': 'world'})
    ET.SubElement(xarm_attach, 'child', {'link': f'{xarm_prefix}link_base'})
    ET.SubElement(xarm_attach, 'origin', {'xyz': '0.0 -0.7 1.021', 'rpy': '0 0 3.142'})

    # Copy all xArm6 elements (links, joints, ros2_control, gazebo material tags)
    for elem in list(xarm_root):
        combined_root.append(elem)

    # UR5e attachment: world -> ur5e_base_link
    ur5e_attach = ET.SubElement(combined_root, 'joint',
                                {'name': f'{ur5e_prefix}world_joint', 'type': 'fixed'})
    ET.SubElement(ur5e_attach, 'parent', {'link': 'world'})
    ET.SubElement(ur5e_attach, 'child', {'link': f'{ur5e_prefix}base_link'})
    ET.SubElement(ur5e_attach, 'origin', {'xyz': '0.0 0.7 1.021', 'rpy': '0 0 3.142'})

    # Copy all UR5e elements (links, joints, ros2_control, gripper)
    for elem in list(ur5e_root):
        combined_root.append(elem)

    # Single gazebo_ros2_control plugin for both robots
    gazebo_plugin_elem = ET.SubElement(combined_root, 'gazebo')
    plugin_elem = ET.SubElement(gazebo_plugin_elem, 'plugin',
                                {'filename': 'libgazebo_ros2_control.so',
                                 'name': 'gazebo_ros2_control'})
    params_elem = ET.SubElement(plugin_elem, 'parameters')
    params_elem.text = combined_controllers_yaml

    # Serialize combined URDF
    combined_description = ET.tostring(combined_root, encoding='unicode')

    # Resolve package:// paths to file:// for Gazebo mesh loading
    ur_desc_path = get_package_share_directory('ur_description')
    onrobot_desc_path = get_package_share_directory('onrobot_description')
    combined_description = combined_description.replace('package://xarm_description', f'file://{xarm_desc_path}')
    combined_description = combined_description.replace('package://ur_description', f'file://{ur_desc_path}')
    combined_description = combined_description.replace('package://onrobot_description', f'file://{onrobot_desc_path}')

    # ══════════════════════════════════════════════════════════════════════════
    # Launch nodes
    # ══════════════════════════════════════════════════════════════════════════

    # Single robot_state_publisher for the combined URDF
    combined_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True, 'robot_description': combined_description}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    # Single spawn — positions are in the URDF fixed joints, not spawn args
    combined_spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-topic', '/robot_description',
            '-entity', 'dual_robot',
            '-x', '0.0', '-y', '0.0', '-z', '0.0',
        ],
    )

    # All controllers under the single /controller_manager
    controller_nodes = [
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster',
                       '--controller-manager', '/controller_manager'],
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[f'{xarm_prefix}xarm6_traj_controller',
                       '--controller-manager', '/controller_manager'],
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[f'{xarm_prefix}xarm_gripper_traj_controller',
                       '--controller-manager', '/controller_manager'],
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['ur5e_joint_trajectory_controller',
                       '--controller-manager', '/controller_manager'],
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['ur5e_rg2_gripper_traj_controller',
                       '--controller-manager', '/controller_manager'],
            parameters=[{'use_sim_time': True}],
        ),
    ]

    return [
        gazebo_launch,
        combined_rsp,
        # Delay spawn 30s to give Gazebo (WSL) time to fully initialize
        TimerAction(period=30.0, actions=[combined_spawn]),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=combined_spawn,
                on_exit=controller_nodes,
            )
        ),
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
