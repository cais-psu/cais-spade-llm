"""Launch the recovery framework NIST world with independent UR5e/RG2 arms."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

UR5E_JOINTS = (
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
)
KMR_ARM_JOINTS = tuple(f'joint_a{index}' for index in range(1, 8))
KMR_BASE_STATE_JOINTS = (
    'KMR_base_x_joint', 'KMR_base_y_joint', 'KMR_base_yaw_joint',
)
KMR_SDF_VISUAL_COLORS = (
    ('safety_scanner_window', '0.12 0.62 0.78 1'),
    ('emergency_stop', '0.85 0.02 0.02 1'),
    ('_roller_', '0.48 0.50 0.52 1'),
    ('mobile_platform_lower', '0.08 0.09 0.10 1'),
    ('mobile_platform_upper', '0.76 0.78 0.80 1'),
    ('top_deck', '0.19 0.21 0.23 1'),
    ('clear_deck', '0.60 0.62 0.64 1'),
    ('iiwa_mounting_plate', '0.19 0.21 0.23 1'),
    ('KUKA_panel', '0.96 0.36 0.03 1'),
    ('RGB_LED_band', '0.14 0.58 0.92 1'),
    ('safety_scanner', '0.08 0.09 0.10 1'),
    ('led_sound_buzzer', '0.96 0.36 0.03 1'),
    ('ultrasonic_', '0.08 0.09 0.10 1'),
    ('wheel_', '0.08 0.09 0.10 1'),
    ('rg2_adapter', '0.08 0.09 0.10 1'),
    ('rg2_base_link', '0.8 0.8 0.8 1'),
    ('rg2_left_outer_knuckle', '0.8 0.8 0.8 1'),
    ('rg2_right_outer_knuckle', '0.8 0.8 0.8 1'),
    ('rg2_left_inner_knuckle', '0.8 0.8 0.8 1'),
    ('rg2_right_inner_knuckle', '0.8 0.8 0.8 1'),
    ('rg2_left_inner_finger', '0.1 0.1 0.1 1'),
    ('rg2_right_inner_finger', '0.1 0.1 0.1 1'),
)
KMR_SDF_EXPECTED_COLOR_COUNT = 52
RECOVERY_VELOCITY_SCALING = 0.4
RECOVERY_ACCELERATION_SCALING = 0.3


def _load_robots(path: Path) -> list[dict[str, Any]]:
    robots = json.loads(path.read_text(encoding='utf-8'))['robots']
    if not isinstance(robots, list) or len(robots) != 4:
        raise ValueError('The current recovery framework scene requires four UR5e instances.')
    for robot in robots:
        if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*_', robot['prefix']):
            raise ValueError('Each UR5e prefix must be a distinct ROS identifier ending in _.')
        for key in ('base_xyz', 'base_rpy'):
            if len(robot[key]) != 3 or not all(math.isfinite(value) for value in robot[key]):
                raise ValueError(f'{robot["resource_id"]}: {key} must contain three finite values.')
        positions = robot['initial_joint_positions']
        if set(positions) != set(UR5E_JOINTS) or not all(math.isfinite(value) for value in positions.values()):
            raise ValueError(f'{robot["resource_id"]}: all six UR5e initial joints are required.')
    for key in ('prefix', 'resource_id'):
        if len({robot[key] for robot in robots}) != len(robots):
            raise ValueError(f'UR5e instances must have distinct {key} values.')
    return robots


def _load_launch_module(filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load UR5e simulation support: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _controller_config(robots: list[dict[str, Any]]) -> dict[str, Any]:
    manager = {
        'update_rate': 1000, 'use_sim_time': True,
        'joint_state_broadcaster': {'type': 'joint_state_broadcaster/JointStateBroadcaster'},
    }
    config = {'controller_manager': {'ros__parameters': manager}}
    for robot in robots:
        prefix = robot['prefix']
        for controller, joints in (
            (f'{prefix}joint_trajectory_controller', [prefix + joint for joint in UR5E_JOINTS]),
            (f'{prefix}rg2_gripper_traj_controller', [f'{prefix}rg2_finger_width']),
        ):
            manager[controller] = {'type': 'joint_trajectory_controller/JointTrajectoryController'}
            config[controller] = {'ros__parameters': {
                'joints': joints, 'command_interfaces': ['position'],
                'state_interfaces': ['position', 'velocity'] if len(joints) == 6 else ['position'],
                'state_publish_rate': 100.0, 'action_monitor_rate': 20.0,
                'allow_partial_joints_goal': False,
                'constraints': {'stopped_velocity_tolerance': 0.2, 'goal_time': 0.0},
            }}
    return config


def _load_scene(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload.get('KMR'), dict):
        raise ValueError('The recovery framework configuration requires KMR.')
    return payload


def _kmr_controller_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('KMR controller configuration must be a map.')
    return config


def _build_description(robots: list[dict[str, Any]], controllers_yaml: str) -> str:
    ur5e = _load_launch_module('ur5e_rg2_gazebo.launch.py')
    combined = ET.Element('robot', {'name': 'dual_robot'})
    ET.SubElement(combined, 'link', {'name': 'world'})
    control = ET.SubElement(combined, 'ros2_control', {'name': 'dual_robot', 'type': 'system'})
    hardware = ET.SubElement(control, 'hardware')
    ET.SubElement(hardware, 'plugin').text = 'gazebo_ros2_control/GazeboSystem'
    for robot in robots:
        prefix = robot['prefix']
        fragment = ur5e._build_ur5e_rg2_description(
            controllers_yaml, ur5e_prefix=prefix,
            base_xyz=' '.join(str(value) for value in robot['base_xyz']),
            base_rpy=' '.join(str(value) for value in robot['base_rpy']),
            initial_positions=robot['initial_joint_positions'],
        )
        root = ET.fromstring(fragment)
        ur5e._strip_gazebo_ros2_control_plugin(root)
        # One GazeboSystem avoids redeclaring hold_joints on the shared plugin
        # node. Controller ownership remains separate for each arm and gripper.
        for component in root.findall('ros2_control'):
            if component.findtext('hardware/plugin') != 'gazebo_ros2_control/GazeboSystem':
                raise ValueError('Recovery framework robot joints require GazeboSystem.')
            control.extend(component.findall('joint'))
            root.remove(component)
        for element in root:
            if element.tag != 'link' or element.get('name') != 'world':
                combined.append(element)
    plugin = ET.SubElement(ET.SubElement(combined, 'gazebo'), 'plugin', {
        'filename': 'libgazebo_ros2_control.so', 'name': 'gazebo_ros2_control',
    })
    ET.SubElement(plugin, 'robot_param').text = 'robot_description'
    ET.SubElement(plugin, 'robot_param_node').text = 'ur_gazebo_robot_state_publisher'
    ET.SubElement(plugin, 'parameters').text = controllers_yaml
    return ET.tostring(combined, encoding='unicode')


def _build_kmr_description(share: Path, controllers_yaml: Path) -> str:
    description = subprocess.check_output([
        'xacro', str(share / 'urdf' / 'KMR_recovery.urdf.xacro'),
        f'controllers_file:={controllers_yaml}',
    ], text=True)
    root = ET.fromstring(description)
    if root.get('name') != 'KMR':
        raise ValueError('KMR runtime description must retain robot name KMR.')
    controlled = root.findall('ros2_control/joint')
    if {joint.get('name') for joint in controlled} != {
        *KMR_ARM_JOINTS, 'KMR_rg2_finger_width',
    }:
        raise ValueError('KMR requires seven iiwa joints and one RG2 joint.')
    return description


def _build_planning_description(
    ur_description: str,
    kmr_description: str,
) -> str:
    combined = ET.fromstring(ur_description)
    combined.set('name', 'dual_robot')
    for element in list(combined.findall('ros2_control')) + list(combined.findall('gazebo')):
        combined.remove(element)
    kmr = ET.fromstring(kmr_description)
    for element in kmr:
        if element.tag not in {'ros2_control', 'gazebo'}:
            combined.append(element)
    for name in ('KMR_base_x_link', 'KMR_base_y_link'):
        ET.SubElement(combined, 'link', {'name': name})
    base_x = ET.SubElement(combined, 'joint', {'name': KMR_BASE_STATE_JOINTS[0], 'type': 'prismatic'})
    ET.SubElement(base_x, 'parent', {'link': 'world'})
    ET.SubElement(base_x, 'child', {'link': 'KMR_base_x_link'})
    ET.SubElement(base_x, 'axis', {'xyz': '1 0 0'})
    ET.SubElement(base_x, 'limit', {'lower': '-20', 'upper': '20', 'effort': '100', 'velocity': '1'})
    base_y = ET.SubElement(combined, 'joint', {'name': KMR_BASE_STATE_JOINTS[1], 'type': 'prismatic'})
    ET.SubElement(base_y, 'parent', {'link': 'KMR_base_x_link'})
    ET.SubElement(base_y, 'child', {'link': 'KMR_base_y_link'})
    ET.SubElement(base_y, 'axis', {'xyz': '0 1 0'})
    ET.SubElement(base_y, 'limit', {'lower': '-20', 'upper': '20', 'effort': '100', 'velocity': '1'})
    base_yaw = ET.SubElement(combined, 'joint', {'name': KMR_BASE_STATE_JOINTS[2], 'type': 'continuous'})
    ET.SubElement(base_yaw, 'parent', {'link': 'KMR_base_y_link'})
    ET.SubElement(base_yaw, 'child', {'link': 'KMR_base_link'})
    ET.SubElement(base_yaw, 'axis', {'xyz': '0 0 1'})
    ET.SubElement(base_yaw, 'limit', {'effort': '100', 'velocity': '1'})
    names = [element.get('name') for tag in ('link', 'joint') for element in combined.findall(tag)]
    if len(names) != len(set(names)):
        raise ValueError('Combined recovery planning description contains duplicate names.')
    return ET.tostring(combined, encoding='unicode')


def _build_srdf(robots: list[dict[str, Any]], description: str, ur_moveit_share: Path) -> str:
    merged = ET.Element('robot', {'name': 'dual_robot'})
    all_robots = ET.SubElement(merged, 'group', {'name': 'all_robots'})
    model = ET.fromstring(description)
    for robot in robots:
        prefix = robot['prefix']
        arm_group = f'{prefix}ur_manipulator'
        gripper_group = f'{prefix}rg2_gripper'
        srdf = ET.fromstring(subprocess.check_output([
            'xacro', str(ur_moveit_share / 'srdf' / 'ur.srdf.xacro'),
            'name:=ur', f'prefix:={prefix}',
        ], text=True))
        for element in srdf:
            if element.tag == 'group_state' and element.get('name') == f'{prefix}home':
                for joint in element.findall('joint'):
                    joint.set('value', str(robot['initial_joint_positions'][joint.attrib['name'][len(prefix):]]))
            merged.append(element)
        ET.SubElement(all_robots, 'group', {'name': arm_group})
        gripper = ET.SubElement(merged, 'group', {'name': gripper_group})
        ET.SubElement(gripper, 'joint', {'name': f'{prefix}rg2_finger_width'})
        for name, value in (('open', 0.11), ('close', 0.02)):
            state = ET.SubElement(merged, 'group_state', {'name': name, 'group': gripper_group})
            ET.SubElement(state, 'joint', {'name': f'{prefix}rg2_finger_width', 'value': str(value)})
        ET.SubElement(merged, 'end_effector', {
            'name': f'{prefix}rg2', 'parent_link': f'{prefix}tool0',
            'group': gripper_group, 'parent_group': arm_group,
        })
        # Retain the established RG2 mount/linkage exclusions within each arm.
        # No exclusion is added between separate robot instances.
        related = [link.attrib['name'] for link in model.findall('link')
                   if link.attrib['name'].startswith(f'{prefix}rg2_')]
        related += [f'{prefix}tool0', f'{prefix}wrist_3_link', f'{prefix}wrist_2_link']
        for index, link1 in enumerate(related):
            for link2 in related[index + 1:]:
                ET.SubElement(merged, 'disable_collisions', {
                    'link1': link1, 'link2': link2, 'reason': 'Never',
                })
    kmr_arm = ET.SubElement(merged, 'group', {'name': 'KMR_iiwa_arm'})
    ET.SubElement(kmr_arm, 'chain', {'base_link': 'iiwa_link_0', 'tip_link': 'iiwa_tool0'})
    ET.SubElement(all_robots, 'group', {'name': 'KMR_iiwa_arm'})
    kmr_gripper = ET.SubElement(merged, 'group', {'name': 'KMR_rg2_gripper'})
    ET.SubElement(kmr_gripper, 'joint', {'name': 'KMR_rg2_finger_width'})
    upright = ET.SubElement(merged, 'group_state', {
        'name': 'upright', 'group': 'KMR_iiwa_arm',
    })
    for joint in KMR_ARM_JOINTS:
        ET.SubElement(upright, 'joint', {'name': joint, 'value': '0'})
    for name, value in (('open', 0.11), ('close', 0.02)):
        state = ET.SubElement(merged, 'group_state', {'name': name, 'group': 'KMR_rg2_gripper'})
        ET.SubElement(state, 'joint', {'name': 'KMR_rg2_finger_width', 'value': str(value)})
    ET.SubElement(merged, 'end_effector', {
        'name': 'KMR_rg2', 'parent_link': 'iiwa_tool0',
        'group': 'KMR_rg2_gripper', 'parent_group': 'KMR_iiwa_arm',
    })
    kmr_related = [
        'iiwa_link_0', 'iiwa_link_1', 'iiwa_link_2', 'iiwa_link_3',
        'iiwa_link_4', 'iiwa_link_5', 'iiwa_link_6', 'iiwa_link_7',
        'iiwa_tool0', 'rg2_adapter', 'rg2_base_link',
        'rg2_left_outer_knuckle', 'rg2_right_outer_knuckle',
        'rg2_left_inner_knuckle', 'rg2_right_inner_knuckle',
        'rg2_left_inner_finger', 'rg2_right_inner_finger',
    ]
    adjacent = {
        ('iiwa_link_0', 'iiwa_link_1'), ('iiwa_link_1', 'iiwa_link_2'),
        ('iiwa_link_2', 'iiwa_link_3'), ('iiwa_link_3', 'iiwa_link_4'),
        ('iiwa_link_4', 'iiwa_link_5'), ('iiwa_link_5', 'iiwa_link_6'),
        ('iiwa_link_6', 'iiwa_link_7'), ('iiwa_link_7', 'iiwa_tool0'),
        ('iiwa_tool0', 'rg2_adapter'), ('rg2_adapter', 'rg2_base_link'),
    }
    for link1, link2 in adjacent:
        ET.SubElement(merged, 'disable_collisions', {
            'link1': link1, 'link2': link2, 'reason': 'Adjacent',
        })
    gripper_links = [link for link in kmr_related if link.startswith('rg2_')]
    for index, link1 in enumerate(gripper_links):
        for link2 in gripper_links[index + 1:]:
            ET.SubElement(merged, 'disable_collisions', {
                'link1': link1, 'link2': link2, 'reason': 'Never',
            })
    return ET.tostring(merged, encoding='unicode')


def _moveit_parameters(
    robots: list[dict[str, Any]], description: str, srdf: str, ur_moveit_share: Path,
) -> dict[str, Any]:
    limits = yaml.safe_load((ur_moveit_share / 'config/joint_limits.yaml').read_text())['joint_limits']
    ompl_defaults = yaml.safe_load((ur_moveit_share / 'config/ompl_planning.yaml').read_text())
    joint_limits, kinematics, controllers = {}, {}, {'controller_names': []}
    ompl = {
        'planning_plugin': 'ompl_interface/OMPLPlanner',
        'request_adapters': (
            'default_planner_request_adapters/AddTimeOptimalParameterization '
            'default_planner_request_adapters/FixWorkspaceBounds '
            'default_planner_request_adapters/FixStartStateBounds '
            'default_planner_request_adapters/FixStartStateCollision '
            'default_planner_request_adapters/FixStartStatePathConstraints'
        ),
        'start_state_max_bounds_error': 0.1,
        'planner_configs': ompl_defaults['planner_configs'],
        'all_robots': {'planner_configs': ['RRTConnectkConfigDefault']},
    }
    for robot in robots:
        prefix = robot['prefix']
        kinematics[f'{prefix}ur_manipulator'] = {
            'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
            'kinematics_solver_search_resolution': 0.005,
            'kinematics_solver_timeout': 0.05,
        }
        joint_limits.update({prefix + name: values for name, values in limits.items()})
        joint_limits[f'{prefix}rg2_finger_width'] = {
            'has_position_limits': True, 'min_position': 0.015, 'max_position': 0.11,
            'has_velocity_limits': True, 'max_velocity': 0.4,
            'has_acceleration_limits': True, 'max_acceleration': 1.5,
        }
        for group in (f'{prefix}ur_manipulator', f'{prefix}rg2_gripper'):
            ompl[group] = {'planner_configs': ['RRTConnectkConfigDefault']}
        for name, joints in (
            (f'{prefix}joint_trajectory_controller', [prefix + joint for joint in UR5E_JOINTS]),
            (f'{prefix}rg2_gripper_traj_controller', [f'{prefix}rg2_finger_width']),
        ):
            controllers['controller_names'].append(name)
            controllers[name] = {
                'action_ns': 'follow_joint_trajectory', 'type': 'FollowJointTrajectory',
                'default': True, 'joints': joints,
            }
    kinematics['KMR_iiwa_arm'] = {
        'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
        'kinematics_solver_search_resolution': 0.005,
        'kinematics_solver_timeout': 0.05,
    }
    kmr_limits = {
        'joint_a1': (1.4835, 3.75), 'joint_a2': (1.4835, 1.875),
        'joint_a3': (1.7453, 2.5), 'joint_a4': (1.309, 3.125),
        'joint_a5': (2.2689, 3.75), 'joint_a6': (2.3562, 5.0),
        'joint_a7': (2.3562, 5.0),
    }
    for joint, (velocity, acceleration) in kmr_limits.items():
        joint_limits[joint] = {
            'has_velocity_limits': True, 'max_velocity': velocity,
            'has_acceleration_limits': True, 'max_acceleration': acceleration,
        }
    joint_limits['KMR_rg2_finger_width'] = {
        'has_position_limits': True, 'min_position': 0.015, 'max_position': 0.11,
        'has_velocity_limits': True, 'max_velocity': 0.127,
        'has_acceleration_limits': True, 'max_acceleration': 1.0,
    }
    for group in ('KMR_iiwa_arm', 'KMR_rg2_gripper'):
        ompl[group] = {'planner_configs': ['RRTConnectkConfigDefault']}
    for name, action_ns, joints in (
        (
            'KMR/KMR_iiwa_joint_trajectory_controller',
            'follow_joint_trajectory',
            list(KMR_ARM_JOINTS),
        ),
        (
            'KMR/KMR_rg2_gripper_traj_controller',
            'follow_joint_trajectory',
            ['KMR_rg2_finger_width'],
        ),
    ):
        controllers['controller_names'].append(name)
        controllers[name] = {
            'action_ns': action_ns, 'type': 'FollowJointTrajectory',
            'default': True, 'joints': joints,
        }
    return {
        'robot_description': description, 'robot_description_semantic': srdf,
        'robot_description_kinematics': kinematics,
        'robot_description_planning': {
            'default_velocity_scaling_factor': RECOVERY_VELOCITY_SCALING,
            'default_acceleration_scaling_factor': RECOVERY_ACCELERATION_SCALING,
            'joint_limits': joint_limits,
        },
        'move_group': ompl, 'moveit_simple_controller_manager': controllers,
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
        'moveit_manage_controllers': False,
        'trajectory_execution.allowed_execution_duration_scaling': 2.0,
        'trajectory_execution.allowed_goal_duration_margin': 1.0,
        'trajectory_execution.allowed_start_tolerance': 0.01,
        'publish_planning_scene': True, 'publish_geometry_updates': True,
        'publish_state_updates': True, 'publish_transforms_updates': True,
        'use_sim_time': True,
    }


def _runtime_world_without_static_kmr(world_path: Path) -> Path:
    tree = ET.parse(world_path)
    root = tree.getroot()
    world = root.find('world')
    if world is None:
        raise ValueError(f'Gazebo world element is missing: {world_path}')
    matches = [
        include for include in world.findall('include')
        if str(include.findtext('name') or '').strip() == 'KMR'
    ]
    if len(matches) != 1:
        raise ValueError('Recovery source world must contain exactly one static KMR include.')
    world.remove(matches[0])
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', suffix='.world',
        prefix='cais_recovery_runtime_', delete=False,
    ) as file:
        tree.write(file, encoding='unicode', xml_declaration=True)
        return Path(file.name)


def _write_temporary_urdf(description: str, prefix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', suffix='.urdf', prefix=prefix, delete=False,
    ) as file:
        file.write(description)
        return Path(file.name)


def _apply_kmr_sdf_materials(description: str) -> str:
    """Add explicit Gazebo materials to KMR primitive visuals."""

    root = ET.fromstring(description)
    mapped = set()
    for visual in root.findall('.//model/link/visual'):
        name = str(visual.get('name') or '')
        color = next((rgba for token, rgba in KMR_SDF_VISUAL_COLORS if token in name), None)
        if color is None:
            continue
        material = visual.find('material')
        if material is None:
            material = ET.SubElement(visual, 'material')
        for tag in ('ambient', 'diffuse'):
            value = material.find(tag)
            if value is None:
                value = ET.SubElement(material, tag)
            value.text = color
        mapped.add(name)
    if len(mapped) != KMR_SDF_EXPECTED_COLOR_COUNT:
        raise ValueError(
            f'Expected {KMR_SDF_EXPECTED_COLOR_COUNT} colored KMR visuals, found {len(mapped)}.'
        )
    return ET.tostring(root, encoding='unicode')


def _write_kmr_runtime_sdf(urdf_path: Path) -> Path:
    """Convert the articulated KMR URDF and preserve its Gazebo colors."""

    try:
        description = subprocess.check_output(
            ['gz', 'sdf', '-p', str(urdf_path)], text=True, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'Could not convert the KMR URDF to SDF: {exc.stderr}') from exc
    colored = _apply_kmr_sdf_materials(description)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', suffix='.sdf', prefix='cais_recovery_KMR_', delete=False,
    ) as file:
        file.write(colored)
        return Path(file.name)


def launch_setup(context: Any, *args: Any, **kwargs: Any) -> list[Any]:
    """Create the recovery world with four UR5e arms and articulated KMR."""
    from ament_index_python import get_package_prefix, get_package_share_directory
    from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription, RegisterEventHandler, TimerAction
    from launch.event_handlers import OnProcessExit, OnProcessIO, OnShutdown
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from launch_ros.descriptions import ParameterFile

    def enabled(name: str) -> bool:
        return LaunchConfiguration(name).perform(context) == 'true'

    if enabled('run_perception'):
        raise RuntimeError('The recovery framework environment currently requires run_perception:=false.')
    share = Path(get_package_share_directory('cais_lab_robotics'))
    scene_path = Path(LaunchConfiguration('robots_file').perform(context))
    scene_config = _load_scene(scene_path)
    robots = _load_robots(scene_path)
    kmr = scene_config['KMR']
    controllers = _controller_config(robots)
    temporary_paths: list[Path] = []
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', prefix='cais_recovery_controllers_', delete=False) as file:
        yaml.safe_dump(controllers, file, sort_keys=False)
        controllers_path = file.name
    temporary_paths.append(Path(controllers_path))
    ur_description = _build_description(robots, controllers_path)
    kmr_controllers_path = share / 'config' / 'recovery_framework_kmr_controllers.yaml'
    _kmr_controller_config(kmr_controllers_path)
    kmr_description = _build_kmr_description(share, kmr_controllers_path)
    planning_description = _build_planning_description(ur_description, kmr_description)
    ur_urdf_path = _write_temporary_urdf(ur_description, 'cais_recovery_ur5e_')
    kmr_urdf_path = _write_temporary_urdf(kmr_description, 'cais_recovery_KMR_')
    kmr_sdf_path = _write_kmr_runtime_sdf(kmr_urdf_path)
    temporary_paths += [ur_urdf_path, kmr_urdf_path, kmr_sdf_path]
    moveit_share = Path(get_package_share_directory('ur_moveit_config'))
    srdf = _build_srdf(robots, planning_description, moveit_share)
    moveit = _moveit_parameters(robots, planning_description, srdf, moveit_share)
    actions: list[Any] = []
    if enabled('launch_gazebo'):
        support = _load_launch_module('xarm6_ur5e_gazebo.launch.py')
        world = support._package_world_path(share, LaunchConfiguration('world_file').perform(context))
        if not enabled('include_assembly_parts') or not enabled('include_loose_parts'):
            world = Path(support._filtered_world(
                world, include_assembly_parts=enabled('include_assembly_parts'),
                include_loose_parts=enabled('include_loose_parts'),
            ))
            temporary_paths.append(world)
        world = _runtime_world_without_static_kmr(world)
        temporary_paths.append(world)
        initial_pose = [str(value) for value in kmr['initial_pose']]
        ur_spawn = Node(
            package='gazebo_ros', executable='spawn_entity.py', output='screen',
            arguments=['-file', str(ur_urdf_path), '-entity', 'dual_robot', '-timeout', '120'],
        )
        ur_spawner = Node(
            package='controller_manager', executable='spawner', output='screen',
            arguments=[
                'joint_state_broadcaster', *[name for name in controllers if name != 'controller_manager'],
                '--controller-manager', '/controller_manager', '--controller-manager-timeout', '60.0',
                '--service-call-timeout', '60.0', '--switch-timeout', '60.0', '--activate-as-group',
            ], parameters=[{'use_sim_time': True}],
        )
        kmr_spawn = Node(
            package='gazebo_ros', executable='spawn_entity.py', output='screen',
            arguments=[
                '-file', str(kmr_sdf_path), '-entity', 'KMR',
                '-x', initial_pose[0], '-y', initial_pose[1], '-z', initial_pose[2],
                '-R', initial_pose[3], '-P', initial_pose[4], '-Y', initial_pose[5],
                '-timeout', '120',
            ],
        )
        kmr_spawner = Node(
            package='controller_manager', executable='spawner', output='screen',
            arguments=[
                'joint_state_broadcaster',
                'KMR_iiwa_joint_trajectory_controller',
                'KMR_rg2_gripper_traj_controller',
                '--controller-manager', '/KMR/controller_manager',
                '--controller-manager-timeout', '60.0',
                '--service-call-timeout', '60.0', '--switch-timeout', '60.0',
                '--activate-as-group',
            ], parameters=[{'use_sim_time': True}],
        )
        nav2_actions: list[Any] = []
        if enabled('launch_nav2'):
            try:
                get_package_share_directory('nav2_bringup')
                get_package_share_directory('nav2_msgs')
            except Exception as exc:
                raise RuntimeError(
                    'Recovery KMR navigation requires ROS 2 Humble Nav2. Install it with: '
                    'sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup'
                ) from exc
            from nav2_common.launch import RewrittenYaml

            nav2_params = share / 'config/recovery_framework_nav2.yaml'
            map_yaml = share / 'config/recovery_framework_map.yaml'
            for required in (nav2_params, map_yaml, map_yaml.with_suffix('.pgm')):
                if not required.is_file():
                    raise RuntimeError(f'Recovery KMR navigation asset is missing: {required}')
            configured_nav2 = ParameterFile(
                RewrittenYaml(
                    source_file=str(nav2_params),
                    root_key='KMR',
                    param_rewrites={
                        'use_sim_time': 'true',
                        'yaml_filename': str(map_yaml),
                        'default_nav_to_pose_bt_xml': str(
                            share / 'config' / 'recovery_framework_navigate_to_pose.xml'
                        ),
                        'default_nav_through_poses_bt_xml': str(
                            share / 'config' / 'recovery_framework_navigate_through_poses.xml'
                        ),
                    },
                    convert_types=True,
                ),
                allow_substs=True,
            )
            nav2_node_specs = (
                ('nav2_map_server', 'map_server', 'map_server', []),
                (
                    'nav2_controller', 'controller_server', 'controller_server',
                    [('cmd_vel', 'nav_cmd_vel')],
                ),
                ('nav2_planner', 'planner_server', 'planner_server', []),
                (
                    'nav2_behaviors', 'behavior_server', 'behavior_server',
                    [('cmd_vel', 'nav_cmd_vel')],
                ),
                ('nav2_bt_navigator', 'bt_navigator', 'bt_navigator', []),
            )
            nav2_actions.extend(
                Node(
                    package=package,
                    executable=executable,
                    namespace='KMR',
                    name=name,
                    output='screen',
                    parameters=[configured_nav2],
                    remappings=remappings,
                )
                for package, executable, name, remappings in nav2_node_specs
            )
            nav2_actions.append(Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                namespace='KMR',
                name='lifecycle_manager_navigation',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'autostart': True,
                    'node_names': [
                        'map_server', 'controller_server', 'planner_server',
                        'behavior_server', 'bt_navigator',
                    ],
                }],
            ))
        actions += [
            AppendEnvironmentVariable(name='GAZEBO_MODEL_PATH', value=str(share / 'models'), prepend=True),
            AppendEnvironmentVariable(name='GAZEBO_MODEL_PATH', value=str(share), prepend=True),
            # URDF package:// URIs become model://cais_lab_robotics/... when
            # Gazebo converts them to SDF. Include the parent share directory
            # so the articulated KMR meshes resolve in the Gazebo client.
            AppendEnvironmentVariable(name='GAZEBO_MODEL_PATH', value=str(share.parent), prepend=True),
            AppendEnvironmentVariable(name='GAZEBO_PLUGIN_PATH', value=str(Path(get_package_prefix('ros2_linkattacher')) / 'lib'), prepend=True),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(Path(get_package_share_directory('gazebo_ros')) / 'launch/gazebo.launch.py')),
                launch_arguments={
                    'world': str(world),
                    'gui': LaunchConfiguration('launch_gazebo_gui'),
                    'server_required': 'true',
                    'gui_required': 'false',
                }.items(),
            ),
            Node(
                package='robot_state_publisher', executable='robot_state_publisher',
                name='robot_state_publisher',
                parameters=[{'robot_description': planning_description, 'use_sim_time': True}],
                output='screen',
            ),
            Node(
                package='robot_state_publisher', executable='robot_state_publisher',
                name='ur_gazebo_robot_state_publisher',
                parameters=[{'robot_description': ur_description, 'use_sim_time': True}],
                remappings=[('/tf', '/ur_gazebo_tf_unused'), ('/tf_static', '/ur_gazebo_tf_static_unused')],
                output='log',
            ),
            Node(
                package='robot_state_publisher', executable='robot_state_publisher',
                namespace='KMR', name='robot_state_publisher',
                parameters=[{'robot_description': kmr_description, 'use_sim_time': True}],
                remappings=[('/tf', '/KMR/gazebo_tf_unused'), ('/tf_static', '/KMR/gazebo_tf_static_unused')],
                output='log',
            ),
            ur_spawn,
            # Gazebo can finish inserting the model before spawn_entity.py
            # receives its service response. Start the controller spawner now;
            # it waits for /controller_manager and avoids withholding all live
            # joint state (and therefore the RViz markers) during that delay.
            ur_spawner,
            RegisterEventHandler(OnProcessExit(
                # Serialize Gazebo factory requests without waiting for the UR
                # spawn client response: active UR controllers prove that the
                # first model is inserted and the KMR request can start safely.
                target_action=ur_spawner, on_exit=[kmr_spawn, kmr_spawner],
            )),
            *(
                [RegisterEventHandler(OnProcessExit(
                    target_action=kmr_spawner,
                    on_exit=nav2_actions,
                ))]
                if nav2_actions else []
            ),
            Node(
                package='cais_lab_robotics', executable='kmr_base_controller.py',
                name='KMR_base_controller', output='screen',
                parameters=[{'config_file': str(scene_path), 'use_sim_time': True}],
            ),
        ]
    recovery_markers = None
    if enabled('launch_moveit'):
        # Preserve separate parameter services for MoveIt and its controller manager.
        recovery_markers = Node(
            package='cais_lab_robotics', executable='recovery_drag_markers.py',
            output='screen', parameters=[{'use_sim_time': True}],
        )
        actions += [
            TimerAction(period=4.0, actions=[Node(
                package='moveit_ros_move_group', executable='move_group',
                parameters=[moveit], output='screen',
            )]),
            TimerAction(period=5.0, actions=[recovery_markers]),
        ]
    if enabled('launch_rviz'):
        rviz_environment = (
            {'LIBGL_ALWAYS_SOFTWARE': '1'}
            if os.environ.get('WSL_DISTRO_NAME')
            else {}
        )
        rviz = Node(
            package='rviz2', executable='rviz2', output='log',
            arguments=['-d', str(share / 'rviz/recovery_framework.rviz')],
            parameters=[moveit],
            additional_env=rviz_environment,
            remappings=[
                ('navigate_to_pose', '/KMR/validated_navigate_to_pose'),
            ],
        )
        if recovery_markers is None:
            actions.append(rviz)
        else:
            rviz_started = {'value': False}

            def start_rviz_from_live_state(event: Any) -> list[Any]:
                text = event.text.decode(errors='replace')
                if (
                    rviz_started['value']
                    or 'Recovery markers initialized from live joint and KMR odometry state'
                    not in text
                ):
                    return []
                rviz_started['value'] = True
                return [rviz]

            actions.append(RegisterEventHandler(OnProcessIO(
                target_action=recovery_markers,
                on_stdout=start_rviz_from_live_state,
                on_stderr=start_rviz_from_live_state,
            )))

    def cleanup(event: Any, launch_context: Any) -> None:
        for path in temporary_paths:
            path.unlink(missing_ok=True)

    actions.append(RegisterEventHandler(OnShutdown(on_shutdown=cleanup)))
    return actions


def generate_launch_description() -> Any:
    """Declare the recovery framework's four-UR5e and KMR Gazebo entry point."""
    from ament_index_python import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, OpaqueFunction

    share = Path(get_package_share_directory('cais_lab_robotics'))
    defaults = {
        'world_file': 'table_recovery_framework.world',
        'robots_file': str(share / 'config/recovery_framework_gazebo.json'),
        'run_perception': 'false', 'include_assembly_parts': 'true', 'include_loose_parts': 'true',
        'launch_gazebo': 'true', 'launch_gazebo_gui': 'true',
        'launch_moveit': 'true', 'launch_rviz': 'true',
        'launch_nav2': 'true',
    }
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()],
        OpaqueFunction(function=launch_setup),
    ])
