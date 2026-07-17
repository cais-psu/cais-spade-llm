#!/usr/bin/env python3
"""
Gazebo Classic launch: UR5e + OnRobot RG2 in a single-table world.

Usage:
    ros2 launch cais_lab_robotics ur5e_rg2_gazebo.launch.py
    ros2 launch cais_lab_robotics ur5e_rg2_gazebo.launch.py passive:=true run_perception:=false
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

CONTROLLER_MANAGER_TIMEOUT_SEC = '60.0'
CONTROLLER_SERVICE_CALL_TIMEOUT_SEC = '20.0'
CONTROLLER_SWITCH_TIMEOUT_SEC = '20.0'
RG2_FINGER_WIDTH_EFFORT = '18'
RG2_FINGER_WIDTH_VELOCITY = '0.40'
UR5E_BASE_XYZ = '0.0 0.0 1.021'
UR5E_BASE_RPY = '0 0 3.142'
UR5E_HOME_RAD = {
    'shoulder_pan_joint': 1.637161,
    'shoulder_lift_joint': -2.150816,
    'elbow_joint': 2.028921,
    'wrist_1_joint': -1.452287,
    'wrist_2_joint': -1.561075,
    'wrist_3_joint': 1.637331,
}


def _strip_world_links_and_joints(root):
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


def _strip_gazebo_ros2_control_plugin(root):
    for gazebo_elem in list(root.findall('gazebo')):
        plugin = gazebo_elem.find('plugin')
        if plugin is None:
            continue
        plugin_id = (plugin.get('filename', '') + plugin.get('name', ''))
        if 'gazebo_ros2_control' in plugin_id:
            root.remove(gazebo_elem)


def _set_ros2_control_initial_positions(root, joint_positions):
    for ros2_control in root.findall('ros2_control'):
        for joint in ros2_control.findall('joint'):
            joint_name = joint.get('name', '')
            if joint_name not in joint_positions:
                continue
            position_state = None
            for state_interface in joint.findall('state_interface'):
                if state_interface.get('name') == 'position':
                    position_state = state_interface
                    break
            if position_state is None:
                position_state = ET.SubElement(joint, 'state_interface', {'name': 'position'})
            initial_value = position_state.find("param[@name='initial_value']")
            if initial_value is None:
                initial_value = ET.SubElement(position_state, 'param', {'name': 'initial_value'})
            initial_value.text = str(joint_positions[joint_name])


def _set_or_update_text(parent, tag_name, value):
    child = parent.find(tag_name)
    if child is None:
        child = ET.SubElement(parent, tag_name)
    child.text = str(value)


def _strip_grasp_fix_plugins(root):
    for gazebo_elem in list(root.findall('gazebo')):
        plugin = gazebo_elem.find('plugin')
        if plugin is None:
            continue
        name = plugin.get('name', '')
        filename = plugin.get('filename', '')
        if 'grasp_fix' in name or 'libgazebo_grasp_fix' in filename:
            root.remove(gazebo_elem)


def _inject_mimic_plugins(root, max_effort='3.0', sensitiveness='0.003'):
    for joint in list(root.findall('joint')):
        mimic = joint.find('mimic')
        if mimic is None:
            continue
        gazebo_elem = ET.SubElement(root, 'gazebo')
        plugin_elem = ET.SubElement(
            gazebo_elem,
            'plugin',
            {
                'name': f"mimic_plugin_{joint.get('name')}",
                'filename': 'libgazebo_mimic_joint_plugin.so',
            },
        )
        ET.SubElement(plugin_elem, 'joint').text = mimic.get('joint')
        ET.SubElement(plugin_elem, 'mimicJoint').text = joint.get('name')
        ET.SubElement(plugin_elem, 'multiplier').text = mimic.get('multiplier', '1.0')
        ET.SubElement(plugin_elem, 'offset').text = mimic.get('offset', '0.0')
        ET.SubElement(plugin_elem, 'sensitiveness').text = sensitiveness
        ET.SubElement(plugin_elem, 'maxEffort').text = max_effort


def _tune_rg2_joint_dynamics(root, prefix):
    for joint in root.findall('joint'):
        if joint.get('name') != f'{prefix}finger_width':
            continue
        limit = joint.find('limit')
        if limit is None:
            continue
        limit.set('effort', RG2_FINGER_WIDTH_EFFORT)
        limit.set('velocity', RG2_FINGER_WIDTH_VELOCITY)


def _tune_rg2_contact_properties(root, prefix):
    finger_refs = {
        f'{prefix}left_inner_finger',
        f'{prefix}right_inner_finger',
        f'{prefix}left_inner_knuckle',
        f'{prefix}right_inner_knuckle',
    }
    for gazebo_elem in root.findall('gazebo'):
        ref = gazebo_elem.get('reference', '')
        if ref not in finger_refs:
            continue
        _set_or_update_text(gazebo_elem, 'kp', '12000.0')
        _set_or_update_text(gazebo_elem, 'kd', '30.0')
        _set_or_update_text(gazebo_elem, 'mu1', '200.0')
        _set_or_update_text(gazebo_elem, 'mu2', '200.0')
        _set_or_update_text(gazebo_elem, 'minDepth', '0.002')


def _make_controller_spawner(controller_names):
    return Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            *controller_names,
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', CONTROLLER_MANAGER_TIMEOUT_SEC,
            '--service-call-timeout', CONTROLLER_SERVICE_CALL_TIMEOUT_SEC,
            '--switch-timeout', CONTROLLER_SWITCH_TIMEOUT_SEC,
            '--activate-as-group',
        ],
        parameters=[{'use_sim_time': True}],
    )


def _build_ur5e_rg2_description(controllers_yaml):
    ur5e_prefix = 'ur5e_'

    ur5e_raw = subprocess.check_output([
        'xacro',
        str(Path(get_package_share_directory('ur_description')) / 'urdf' / 'ur.urdf.xacro'),
        'ur_type:=ur5e',
        'name:=ur5e',
        f'tf_prefix:={ur5e_prefix}',
        'sim_gazebo:=true',
        f'simulation_controllers:={controllers_yaml}',
    ]).decode('utf-8')
    ur5e_root = ET.fromstring(ur5e_raw)
    _set_ros2_control_initial_positions(
        ur5e_root,
        {f'{ur5e_prefix}{joint}': value for joint, value in UR5E_HOME_RAD.items()},
    )
    _strip_world_links_and_joints(ur5e_root)
    _strip_gazebo_ros2_control_plugin(ur5e_root)

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

    _set_ros2_control_initial_positions(
        onrobot_root,
        {f'{onrobot_prefix}finger_width': 0.11},
    )

    # Keep this mock link in Gazebo so the RG2 driving joint survives physics.
    for link in onrobot_root.findall('link'):
        if 'finger_width_mock_link' not in link.get('name', ''):
            continue
        if link.find('inertial') is not None:
            continue
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
        ET.SubElement(inertial, 'mass', {'value': '0.01'})
        ET.SubElement(
            inertial,
            'inertia',
            {
                'ixx': '0.0001',
                'ixy': '0.0',
                'ixz': '0.0',
                'iyy': '0.0001',
                'iyz': '0.0',
                'izz': '0.0001',
            },
        )

    _strip_gazebo_ros2_control_plugin(onrobot_root)
    _inject_mimic_plugins(onrobot_root, max_effort='3.0', sensitiveness='0.003')
    _tune_rg2_joint_dynamics(onrobot_root, onrobot_prefix)
    _tune_rg2_contact_properties(onrobot_root, onrobot_prefix)
    _strip_grasp_fix_plugins(onrobot_root)
    _strip_world_links_and_joints(onrobot_root)

    for elem in list(onrobot_root):
        ur5e_root.append(elem)

    mounting_joint = ET.Element('joint', {'name': f'{ur5e_prefix}gripper_mount_joint', 'type': 'fixed'})
    ET.SubElement(mounting_joint, 'parent', {'link': f'{ur5e_prefix}tool0'})
    ET.SubElement(mounting_joint, 'child', {'link': f'{onrobot_prefix}onrobot_base_link'})
    ET.SubElement(mounting_joint, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 -1.57079632679'})
    ur5e_root.append(mounting_joint)

    combined_root = ET.Element('robot', {'name': 'ur5e_rg2'})
    ET.SubElement(combined_root, 'link', {'name': 'world'})

    world_joint = ET.SubElement(
        combined_root,
        'joint',
        {'name': f'{ur5e_prefix}world_joint', 'type': 'fixed'},
    )
    ET.SubElement(world_joint, 'parent', {'link': 'world'})
    ET.SubElement(world_joint, 'child', {'link': f'{ur5e_prefix}base_link'})
    ET.SubElement(world_joint, 'origin', {'xyz': UR5E_BASE_XYZ, 'rpy': UR5E_BASE_RPY})

    for elem in list(ur5e_root):
        combined_root.append(elem)

    gazebo_plugin_elem = ET.SubElement(combined_root, 'gazebo')
    plugin_elem = ET.SubElement(
        gazebo_plugin_elem,
        'plugin',
        {'filename': 'libgazebo_ros2_control.so', 'name': 'gazebo_ros2_control'},
    )
    ET.SubElement(plugin_elem, 'parameters').text = controllers_yaml

    description = ET.tostring(combined_root, encoding='unicode')
    description = description.replace(
        'package://ur_description',
        f"file://{get_package_share_directory('ur_description')}",
    )
    description = description.replace(
        'package://onrobot_description',
        f"file://{get_package_share_directory('onrobot_description')}",
    )
    return description


def launch_setup(context, *args, **kwargs):
    run_perception = LaunchConfiguration('run_perception')
    passive = LaunchConfiguration('passive').perform(context).strip().lower() in {
        '1', 'true', 'yes', 'on',
    }

    controllers_yaml = os.path.join(
        get_package_share_directory('cais_lab_robotics'),
        'config',
        'gazebo_ros2_control',
        'ur5e_rg2_gazebo_ros2_control_controllers.yaml',
    )

    gazebo_world = PathJoinSubstitution([FindPackageShare('cais_lab_robotics'), 'worlds', 'single_table.world'])
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

    robot_description = _build_ur5e_rg2_description(controllers_yaml)

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
            '-entity', 'ur5e_rg2',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0',
        ],
    )

    perception_candidates = [
        Path(__file__).resolve().parents[1] / 'sensor' / 'gazebo_camera_detector.py',
        Path(os.path.expanduser('~/projects/cais-spade-llm/ros2/cais_lab_robotics/sensor/gazebo_camera_detector.py')),
    ]
    perception_script = next((str(p) for p in perception_candidates if p.is_file()), None)
    post_controller_actions = []
    perception_log = None
    if perception_script:
        post_controller_actions.append(
            ExecuteProcess(
                cmd=[
                    'bash',
                    '-lc',
                    [
                        'source /opt/ros/humble/setup.bash && '
                        'source ',
                        os.path.expanduser('~/ros2_ws/install/setup.bash'),
                        ' && python3.10 ',
                        perception_script,
                        ' --ros-args -p use_sim_time:=true',
                    ],
                ],
                output='screen',
                condition=IfCondition(run_perception),
            )
        )
    else:
        perception_log = LogInfo(
            msg='[cais_lab_robotics] gazebo_camera_detector.py not found. '
                'Skipping automatic perception startup.'
        )

    launch_actions = [
        AppendEnvironmentVariable(
            name='GAZEBO_MODEL_PATH',
            value=str(Path(get_package_share_directory('cais_lab_robotics')) / 'models'),
            prepend=True,
        ),
        gazebo,
        state_publisher,
        spawn,
    ]

    if passive:
        # Mirror mode: bring up only the controllers the digital-twin sync needs.
        # joint_state_broadcaster publishes /joint_states (so gazebo -> hardware can
        # read the gazebo pose), while the arm and gripper trajectory controllers
        # actively hold the streamed mirror pose against gravity. No perception.
        controller_spawner = _make_controller_spawner([
            'joint_state_broadcaster',
            'ur5e_joint_trajectory_controller',
            'ur5e_rg2_gripper_traj_controller',
        ])
        launch_actions.append(TimerAction(period=2.0, actions=[controller_spawner]))
    else:
        controller_spawner = _make_controller_spawner([
            'joint_state_broadcaster',
            'ur5e_joint_trajectory_controller',
            'ur5e_rg2_gripper_traj_controller',
        ])
        launch_actions.extend([
            TimerAction(period=2.0, actions=[controller_spawner]),
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=controller_spawner,
                    on_exit=post_controller_actions,
                )
            ),
        ])
        if perception_log is not None:
            launch_actions.append(perception_log)
    return launch_actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'run_perception',
            default_value='true',
            description='Automatically start gazebo_camera_detector for /detect_part and /detect_all.',
        ),
        DeclareLaunchArgument(
            'passive',
            default_value='false',
            description='Spawn Gazebo as a passive mirror with holding trajectory controllers.',
        ),
        OpaqueFunction(function=launch_setup),
    ])
