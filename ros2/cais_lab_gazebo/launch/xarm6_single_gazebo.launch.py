#!/usr/bin/env python3
"""
Gazebo Classic launch: xArm6 + xArm gripper in a single-table world.

Usage:
    ros2 launch xarm_gazebo xarm6_single_gazebo.launch.py
    ros2 launch xarm_gazebo xarm6_single_gazebo.launch.py passive:=true
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file, get_xacro_content

CONTROLLER_MANAGER_TIMEOUT_SEC = '60.0'
CONTROLLER_SERVICE_CALL_TIMEOUT_SEC = '20.0'
CONTROLLER_SWITCH_TIMEOUT_SEC = '20.0'

# Startup posture (radians, per-joint, before prefix) — matches the dual-robot sim's xArm6
# home (robot_xarm6.json named_positions.home). Spawns the arm lifted off the table instead
# of collapsed into the ground. Tweak these to reposition the startup pose.
XARM6_HOME_RAD = {
    'joint1': -1.572631,
    'joint2': -1.054702,
    'joint3': -0.385494,
    'joint4': 0.000322,
    'joint5': 1.440603,
    'joint6': -1.572544,
    'drive_joint': 0.85,  # gripper fully open at startup
}


def _set_ros2_control_initial_positions(root, joint_positions):
    """Inject per-joint <state_interface name="position"> initial_value into ros2_control.

    Sets the Gazebo startup posture without moving the robot base frame.
    """
    for ros2_control in root.findall('ros2_control'):
        for joint in ros2_control.findall('joint'):
            joint_name = joint.get('name', '')
            if joint_name not in joint_positions:
                continue
            position_si = None
            for si in joint.findall('state_interface'):
                if si.get('name') == 'position':
                    position_si = si
                    break
            if position_si is None:
                position_si = ET.SubElement(joint, 'state_interface', {'name': 'position'})
            initial_param = position_si.find("param[@name='initial_value']")
            if initial_param is None:
                initial_param = ET.SubElement(position_si, 'param', {'name': 'initial_value'})
            initial_param.text = str(joint_positions[joint_name])


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

    # Inject the stable startup posture into the ros2_control joints.
    root = ET.fromstring(description)
    _set_ros2_control_initial_positions(
        root, {f'{prefix}{joint}': value for joint, value in XARM6_HOME_RAD.items()}
    )
    description = ET.tostring(root, encoding='unicode')

    return description.replace(
        'package://xarm_description',
        f"file://{get_package_share_directory('xarm_description')}",
    )


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


def launch_setup(context, *args, **kwargs):
    prefix = 'xarm6_'
    passive = LaunchConfiguration('passive').perform(context).strip().lower() in {
        '1', 'true', 'yes', 'on',
    }

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

    launch_actions = [
        gazebo,
        state_publisher,
        spawn,
    ]

    if passive:
        # Mirror mode: bring up the controllers the digital-twin sync needs.
        # joint_state_broadcaster publishes /joint_states (so gazebo -> hardware can
        # read the gazebo pose); the arm and gripper trajectory controllers actively
        # hold the streamed mirror poses against gravity.
        controller_spawner = _make_controller_spawner([
            'joint_state_broadcaster',
            f'{prefix}xarm6_traj_controller',
            f'{prefix}xarm_gripper_traj_controller',
        ])
    else:
        controller_spawner = _make_controller_spawner([
            'joint_state_broadcaster',
            f'{prefix}xarm6_traj_controller',
            f'{prefix}xarm_gripper_traj_controller',
        ])

    launch_actions.append(
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn,
                on_exit=[controller_spawner],
            )
        )
    )

    return launch_actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'passive',
            default_value='false',
            description='Spawn Gazebo as a passive mirror without active trajectory controller spawners.',
        ),
        OpaqueFunction(function=launch_setup),
    ])
