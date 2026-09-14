#!/usr/bin/env python3
"""
Combined Gazebo Classic launch: xArm6 (with gripper) + UR5e in the same world.

Both robots share a single gazebo_ros2_control plugin and controller_manager,
so both have full physics simulation in Gazebo. Robot positions are encoded as
fixed joints in the combined URDF (not as spawn_entity arguments).

TF prefixes:
  xArm6 → prefix: xarm6_,  pos: (0.0, -0.62, 1.021) [180° yaw (3.142)]
  UR5e  → prefix: ur5e_,   pos: (0.0, 0.62, 1.021)  [180° yaw (3.142)]
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from ament_index_python import get_package_prefix, get_package_share_directory
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
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file, get_xacro_content

ROBOT_BASE_Y = 0.50
CONTROLLER_MANAGER_TIMEOUT_SEC = '60.0'
CONTROLLER_SERVICE_CALL_TIMEOUT_SEC = '20.0'
CONTROLLER_SWITCH_TIMEOUT_SEC = '20.0'
RG2_FINGER_WIDTH_EFFORT = '18'
RG2_FINGER_WIDTH_VELOCITY = '0.40'
XARM_GRIPPER_EFFORT = '12'
XARM_GRIPPER_VELOCITY = '0.60'
GAZEBO_TABLE_SURFACE_Z_M = 1.015
ASSEMBLY_PART_MODELS = {
    'table_xarm6',
    'table_ur5e',
    'gear_small',
    'rect_pin_small',
    'circ_pin_small',
    'gear_medium',
    'rect_pin_medium',
    'circ_pin_medium',
    'gear_large',
    'rect_pin_large',
    'circ_pin_large',
    'assembly_board_v1',
    'prusa_mk3',
    'prusa_mk4_1',
    'prusa_mk4_2',
    'cam_mk3',
    'cam_mk4_1',
    'cam_mk4_2',
    'cam_assembly',
}
LOOSE_PART_MODELS = {
    'gear_small',
    'rect_pin_small',
    'circ_pin_small',
    'gear_medium',
    'rect_pin_medium',
    'circ_pin_medium',
    'gear_large',
    'rect_pin_large',
    'circ_pin_large',
}
PRUSA_PRINTERS_AND_ASSEMBLY_BOARD_MODELS = {
    'assembly_board_v1',
    'prusa_mk3',
    'prusa_mk4_1',
    'prusa_mk4_2',
}


def _strip_gazebo_ros2_control_plugin(root):
    """Remove <gazebo><plugin filename='...gazebo_ros2_control...'> from an XML tree."""
    for gazebo_elem in root.findall('gazebo'):
        plugin = gazebo_elem.find('plugin')
        if plugin is not None and 'gazebo_ros2_control' in (plugin.get('filename', '') + plugin.get('name', '')):
            root.remove(gazebo_elem)


def _strip_passive_ur5e_mount_collision(root, prefix):
    """Remove the passive mirror's table-intersecting UR5e base collision."""
    link_name = f'{prefix}base_link_inertia'
    link = next((item for item in root.findall('link') if item.get('name') == link_name), None)
    if link is None:
        raise RuntimeError(f'UR5e mount link is missing: {link_name}')
    collisions = list(link.findall('collision'))
    if not collisions:
        raise RuntimeError(f'UR5e mount collision is missing: {link_name}')
    for collision in collisions:
        link.remove(collision)


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


def _inject_mimic_plugins(root, max_effort='5.0', sensitiveness='0.001'):
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
            ET.SubElement(plugin_elem, 'sensitiveness').text = sensitiveness
            ET.SubElement(plugin_elem, 'maxEffort').text = max_effort


def _set_ros2_control_initial_positions(root, joint_positions):
    """
    Inject per-joint state_interface initial_value into ros2_control joints.
    This sets startup posture without moving the robot base frame.
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


def _load_prefixed_joint_positions(path, prefix):
    """Load unprefixed Gazebo initial joint positions and apply the URDF prefix."""
    with open(path, encoding='utf-8') as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f'Initial joint positions file must contain a map: {path}')
    return {f'{prefix}{str(name)}': value for name, value in loaded.items()}


def _set_or_update_text(parent, tag_name, value):
    child = parent.find(tag_name)
    if child is None:
        child = ET.SubElement(parent, tag_name)
    child.text = str(value)


def _strip_grasp_fix_plugins(root):
    """Remove libgazebo_grasp_fix plugins to avoid Gazebo crashes."""
    for gazebo_elem in list(root.findall('gazebo')):
        plugin = gazebo_elem.find('plugin')
        if plugin is None:
            continue
        name = plugin.get('name', '')
        filename = plugin.get('filename', '')
        if 'grasp_fix' in name or 'libgazebo_grasp_fix' in filename:
            root.remove(gazebo_elem)


def _tune_rg2_joint_dynamics(root, prefix):
    """
    Keep RG2 drive-joint motion moderate; avoid over-constraining mimic joints.
    """
    for joint in root.findall('joint'):
        name = joint.get('name', '')
        if not name.startswith(prefix):
            continue
        limit = joint.find('limit')
        if limit is None:
            continue

        if name == f'{prefix}finger_width':
            limit.set('effort', RG2_FINGER_WIDTH_EFFORT)
            limit.set('velocity', RG2_FINGER_WIDTH_VELOCITY)


def _tune_rg2_contact_properties(root, prefix):
    """Soften/damp RG2 finger contact to reduce part ejection during closure."""
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


def _tune_xarm_gripper_contact_properties(root, prefix):
    """Reduce xArm gripper impulse and increase frictional hold on parts."""
    refs = {
        f'{prefix}left_finger',
        f'{prefix}right_finger',
        f'{prefix}left_inner_knuckle',
        f'{prefix}right_inner_knuckle',
    }
    for gazebo_elem in root.findall('gazebo'):
        ref = gazebo_elem.get('reference', '')
        if ref not in refs:
            continue
        _set_or_update_text(gazebo_elem, 'mu1', '20.0')
        _set_or_update_text(gazebo_elem, 'mu2', '20.0')
        _set_or_update_text(gazebo_elem, 'kp', '8000.0')
        _set_or_update_text(gazebo_elem, 'kd', '8.0')
        _set_or_update_text(gazebo_elem, 'minDepth', '0.001')


def _tune_xarm_gripper_joint_dynamics(root, prefix):
    """Limit xArm gripper closing speed/effort to prevent ODE instability."""
    drive_joint = f'{prefix}drive_joint'
    for joint in root.findall('joint'):
        if joint.get('name') != drive_joint:
            continue
        limit = joint.find('limit')
        if limit is None:
            continue
        # Restore snappier motion; attach will be handled by IFRA LinkAttacher.
        limit.set('effort', XARM_GRIPPER_EFFORT)
        limit.set('velocity', XARM_GRIPPER_VELOCITY)


def _configure_spec2primitives_gripper_followers(
    root: ET.Element, prefix: str, *, world_file: str, passive: bool,
) -> None:
    """Select direct finger following for the active ICRA attachment simulation."""
    if world_file != 'table_spec2primitives.world' or passive:
        return
    expected = {
        f'{prefix}{name}' for name in (
            'left_finger_joint', 'left_inner_knuckle_joint',
            'right_outer_knuckle_joint', 'right_finger_joint',
            'right_inner_knuckle_joint',
        )
    }
    plugins = [
        plugin for plugin in root.findall('./gazebo/plugin')
        if plugin.get('filename') == 'libgazebo_mimic_joint_plugin.so'
        and plugin.findtext('joint') == f'{prefix}drive_joint'
    ]
    if len(plugins) != len(expected) or {
        plugin.findtext('mimicJoint') for plugin in plugins
    } != expected:
        raise RuntimeError('Expected exactly five xArm gripper follower plugins for Spec2Primitives.')
    for plugin in plugins:
        if len(plugin.findall('hasPID')) != 1:
            raise RuntimeError('The Spec2Primitives xArm gripper follower configuration is ambiguous.')
    # This scope uses acknowledged attachment for custody. Direct following keeps
    # the vendor's finger linkage aligned without claiming frictional grasping.
    for plugin in plugins:
        plugin.remove(plugin.find('hasPID'))


def _xarm_gripper_initial_position(*, world_file: str, passive: bool) -> float:
    """Return the xArm drive-joint startup position for this simulation mode."""
    if world_file == 'table_spec2primitives.world' and not passive:
        return 0.0
    return 0.85


def _tune_xarm_grasp_fix_plugin(root):
    """Avoid repulsive impulses after grasp attach on xArm gripper."""
    for gazebo_elem in root.findall('gazebo'):
        plugin = gazebo_elem.find("plugin[@name='xarm_gazebo_grasp_fix']")
        if plugin is None:
            continue
        _set_or_update_text(plugin, 'update_rate', '30')
        _set_or_update_text(plugin, 'grip_count_threshold', '1')
        _set_or_update_text(plugin, 'max_grip_count', '6')
        # Keep contact pairs active after attach for Gazebo stability.
        _set_or_update_text(plugin, 'disable_collisions_on_attach', 'false')
        _set_or_update_text(plugin, 'release_tolerance', '0.005')


def _add_rg2_grasp_fix_plugin(root, prefix):
    """
    Ensure RG2 grasp-fix plugin exists and uses stable attach parameters.
    """
    plugin = None
    for gazebo_elem in root.findall('gazebo'):
        plugin = gazebo_elem.find("plugin[@name='onrobot_gazebo_grasp_fix']")
        if plugin is not None:
            break

    if plugin is None:
        gazebo_elem = ET.SubElement(root, 'gazebo')
        plugin = ET.SubElement(
            gazebo_elem,
            'plugin',
            {'name': 'onrobot_gazebo_grasp_fix', 'filename': 'libgazebo_grasp_fix.so'},
        )
        arm = ET.SubElement(plugin, 'arm')
        ET.SubElement(arm, 'arm_name').text = 'onrobot_gripper'
        ET.SubElement(arm, 'palm_link').text = f'{prefix}onrobot_base_link'
        ET.SubElement(arm, 'gripper_link').text = f'{prefix}left_inner_finger'
        ET.SubElement(arm, 'gripper_link').text = f'{prefix}right_inner_finger'
        ET.SubElement(arm, 'gripper_link').text = f'{prefix}left_inner_knuckle'
        ET.SubElement(arm, 'gripper_link').text = f'{prefix}right_inner_knuckle'

    _set_or_update_text(plugin, 'forces_angle_tolerance', '120')
    _set_or_update_text(plugin, 'update_rate', '30')
    _set_or_update_text(plugin, 'grip_count_threshold', '1')
    _set_or_update_text(plugin, 'max_grip_count', '8')
    _set_or_update_text(plugin, 'release_tolerance', '0.01')
    _set_or_update_text(plugin, 'disable_collisions_on_attach', 'true')
    _set_or_update_text(plugin, 'contact_topic', '__default_topic__')



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


def _launch_arg_enabled(context, name, default='false'):
    value = LaunchConfiguration(name, default=default).perform(context)
    return str(value or '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _physical_table_surface_z():
    calibration_path = Path(
        os.path.expanduser(
            os.environ.get(
                'REALSENSE_HAND_EYE_CONFIG',
                '~/.config/cais-spade-llm/ur5e_realsense_hand_eye.yaml',
            )
        )
    )
    try:
        payload = yaml.safe_load(calibration_path.read_text(encoding='utf-8')) or {}
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return None
    table_plane = payload.get('table_plane')
    if not isinstance(table_plane, dict) or not bool(table_plane.get('accepted', False)):
        return None
    if str(table_plane.get('world_frame') or '') != 'world':
        raise RuntimeError('table-plane calibration must use world_frame=world')
    try:
        surface_z_m = float(table_plane['surface_z_m'])
        frame_count = int(table_plane['frame_count'])
        mad_m = float(table_plane['mad_m'])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('table-plane calibration is incomplete') from exc
    if frame_count < 10 or mad_m > 0.002:
        raise RuntimeError('table-plane calibration fails the 10-frame / 2 mm MAD contract')
    return surface_z_m


def _filtered_world(
    world_path,
    *,
    include_assembly_parts,
    include_loose_parts,
    include_prusa_printers_and_assembly_board=True,
    table_surface_z_m=None,
):
    tree = ET.parse(world_path)
    root = tree.getroot()
    for world in root.findall('world'):
        for element in [*list(world.findall('model')), *list(world.findall('include'))]:
            model_name = (
                element.get('name')
                if element.tag == 'model'
                else str(element.findtext('name') or '').strip()
            )
            remove_assembly = not include_assembly_parts and model_name in ASSEMBLY_PART_MODELS
            remove_loose = not include_loose_parts and model_name in LOOSE_PART_MODELS
            remove_prusa_printers_and_assembly_board = (
                not include_prusa_printers_and_assembly_board
                and model_name in PRUSA_PRINTERS_AND_ASSEMBLY_BOARD_MODELS
            )
            if (
                remove_assembly
                or remove_loose
                or remove_prusa_printers_and_assembly_board
            ):
                world.remove(element)
                continue
            if element.tag == 'include' and model_name == 'table_ur5e' and table_surface_z_m is not None:
                pose = element.find('pose')
                if pose is None:
                    pose = ET.SubElement(element, 'pose')
                table_model_z = float(table_surface_z_m) - GAZEBO_TABLE_SURFACE_Z_M
                pose.text = f'0.0 -0.40 {table_model_z:.9f} 0 0 0'
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        prefix='cais_dual_passive_no_assembly_parts_',
        suffix='.world',
        delete=False,
    ) as temporary:
        tree.write(temporary, encoding='unicode', xml_declaration=True)
        temporary_path = temporary.name
    return temporary_path


def _package_world_path(package_share, world_file):
    """Resolve one package-local world filename and reject external paths."""
    world_file = str(world_file or '').strip()
    if (
        not world_file
        or Path(world_file).name != world_file
        or not world_file.endswith('.world')
    ):
        raise RuntimeError('world_file must name a .world file under cais_lab_robotics/worlds')
    world_path = Path(package_share) / 'worlds' / world_file
    if not world_path.is_file():
        raise RuntimeError(f'cais_lab_robotics world file is missing: {world_file}')
    return world_path


def launch_setup(context, *args, **kwargs):
    run_perception = LaunchConfiguration('run_perception')
    passive = _launch_arg_enabled(context, 'passive')
    include_assembly_parts = _launch_arg_enabled(context, 'include_assembly_parts', default='true')
    include_loose_parts = _launch_arg_enabled(context, 'include_loose_parts', default='true')
    include_prusa_printers_and_assembly_board = _launch_arg_enabled(
        context,
        'include_prusa_printers_and_assembly_board',
        default='true',
    )
    table_surface_z_m = _physical_table_surface_z() if passive else None

    # Ensure Gazebo can resolve IFRA LinkAttacher shared library.
    append_gazebo_plugin_path = None
    attacher_candidates = [os.path.expanduser('~/ros2_ws/install/ros2_linkattacher/lib')]
    try:
        attacher_candidates.insert(0, os.path.join(get_package_prefix('ros2_linkattacher'), 'lib'))
    except Exception:
        pass
    attacher_lib_dir = next((p for p in attacher_candidates if os.path.isdir(p)), None)
    if attacher_lib_dir:
        append_gazebo_plugin_path = AppendEnvironmentVariable(
            name='GAZEBO_PLUGIN_PATH',
            value=attacher_lib_dir,
            prepend=True,
        )

    # ── Gazebo Classic ────────────────────────────────────────────────────────
    cais_lab_robotics_share = Path(get_package_share_directory('cais_lab_robotics'))
    gazebo_world_path = _package_world_path(
        cais_lab_robotics_share,
        LaunchConfiguration('world_file').perform(context),
    )
    gazebo_world = (
        str(gazebo_world_path)
        if (
            include_assembly_parts
            and include_loose_parts
            and include_prusa_printers_and_assembly_board
            and table_surface_z_m is None
        )
        else _filtered_world(
            gazebo_world_path,
            include_assembly_parts=include_assembly_parts,
            include_loose_parts=include_loose_parts,
            include_prusa_printers_and_assembly_board=(
                include_prusa_printers_and_assembly_board
            ),
            table_surface_z_m=table_surface_z_m,
        )
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
        get_package_share_directory('cais_lab_robotics'),
        'config',
        'gazebo_ros2_control',
        'xarm6_ur5e_gazebo_ros2_control_controllers.yaml',
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

    # Home position from robot_xarm6.json named_positions.home
    xarm_initial_positions = {
        f'{xarm_prefix}joint1': -1.572631,
        f'{xarm_prefix}joint2': -1.054702,
        f'{xarm_prefix}joint3': -0.385494,
        f'{xarm_prefix}joint4': 0.000322,
        f'{xarm_prefix}joint5': 1.440603,
        f'{xarm_prefix}joint6': -1.572544,
        f'{xarm_prefix}drive_joint': _xarm_gripper_initial_position(
            world_file=gazebo_world_path.name,
            passive=passive,
        ),
    }
    _set_ros2_control_initial_positions(xarm_root, xarm_initial_positions)
    _tune_xarm_gripper_joint_dynamics(xarm_root, xarm_prefix)
    _tune_xarm_gripper_contact_properties(xarm_root, xarm_prefix)
    _strip_grasp_fix_plugins(xarm_root)
    _configure_spec2primitives_gripper_followers(
        xarm_root, xarm_prefix, world_file=gazebo_world_path.name, passive=passive,
    )

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
    ur5e_initial_positions_file = os.path.join(
        get_package_share_directory('cais_lab_robotics'),
        'config',
        'gazebo_initial_joint_positions',
        'ur5e_gazebo_initial_joint_positions.yaml',
    )
    ur5e_raw = subprocess.check_output([
        'xacro',
        str(Path(get_package_share_directory('ur_description')) / 'urdf' / 'ur.urdf.xacro'),
        'ur_type:=ur5e',
        'name:=ur5e',
        f'tf_prefix:={ur5e_prefix}',
        'sim_gazebo:=true',
        f'simulation_controllers:={combined_controllers_yaml}',
        f'initial_positions_file:={ur5e_initial_positions_file}',
    ]).decode('utf-8')
    ur5e_root = ET.fromstring(ur5e_raw)

    ur5e_initial_positions = _load_prefixed_joint_positions(
        ur5e_initial_positions_file,
        ur5e_prefix,
    )
    _set_ros2_control_initial_positions(ur5e_root, ur5e_initial_positions)

    # Strip world/ground_plane links and joints
    _strip_world_links_and_joints(ur5e_root)
    # Strip the gazebo_ros2_control plugin injected by sim_gazebo:=true
    _strip_gazebo_ros2_control_plugin(ur5e_root)
    _strip_grasp_fix_plugins(ur5e_root)
    if passive:
        _strip_passive_ur5e_mount_collision(ur5e_root, ur5e_prefix)
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

    # Start RG2 opened so first grasp comes from an explicit close command.
    _set_ros2_control_initial_positions(
        onrobot_root,
        {f'{onrobot_prefix}finger_width': 0.11},
    )

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
    _inject_mimic_plugins(onrobot_root, max_effort='3.0', sensitiveness='0.003')
    _tune_rg2_joint_dynamics(onrobot_root, onrobot_prefix)
    _tune_rg2_contact_properties(onrobot_root, onrobot_prefix)
    _strip_grasp_fix_plugins(onrobot_root)

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
    ET.SubElement(mounting_joint, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 -1.57079632679'})
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
    ET.SubElement(xarm_attach, 'origin', {'xyz': f'0.0 {-ROBOT_BASE_Y} 1.021', 'rpy': '0 0 3.142'})

    # Copy all xArm6 elements (links, joints, ros2_control, gazebo material tags)
    for elem in list(xarm_root):
        combined_root.append(elem)

    # UR5e attachment: world -> ur5e_base_link
    ur5e_attach = ET.SubElement(combined_root, 'joint',
                                {'name': f'{ur5e_prefix}world_joint', 'type': 'fixed'})
    ET.SubElement(ur5e_attach, 'parent', {'link': 'world'})
    ET.SubElement(ur5e_attach, 'child', {'link': f'{ur5e_prefix}base_link'})
    ET.SubElement(ur5e_attach, 'origin', {'xyz': f'0.0 {ROBOT_BASE_Y} 1.021', 'rpy': '0 0 3.142'})

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
            '-timeout', '120',
        ],
    )

    # Run through bash and source overlays so linkattacher Python modules are always visible.
    auto_link_attacher = ExecuteProcess(
        cmd=[
            'bash',
            '-lc',
            [
                'source /opt/ros/humble/setup.bash && '
                'source ', os.path.expanduser('~/ros2_ws/install/setup.bash'),
                ' && python3 ',
                PathJoinSubstitution([FindPackageShare('cais_lab_robotics'), 'launch', 'auto_link_attacher_node.py']),
                ' --ros-args -p use_sim_time:=true',
                ' -p attach_distance_threshold:=0.06',
                ' -p finger_distance_threshold:=0.04',
                ' -p attach_distance_threshold_ur5e:=0.04',
                ' -p finger_distance_threshold_ur5e:=0.03',
                ' -p require_finger_consensus:=true',
                ' -p allow_tcp_fallback:=false',
                ' -p tcp_fallback_distance_threshold:=0.03',
            ],
        ],
        output='screen',
    )

    # Start perception node so /detect_part and /detect_all are available to SPADE camera clients.
    perception_candidates = [
        Path(__file__).resolve().parents[1] / 'sensor' / 'gazebo_camera_detector.py',
        Path(os.path.expanduser('~/projects/cais-spade-llm/ros2/cais_lab_robotics/sensor/gazebo_camera_detector.py')),
    ]
    perception_script = next((str(p) for p in perception_candidates if p.is_file()), None)
    post_controller_actions = [] if passive else [auto_link_attacher]
    perception_log = None
    if perception_script and not passive:
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

    controller_spawner = _make_controller_spawner([
        'joint_state_broadcaster',
        f'{xarm_prefix}xarm6_traj_controller',
        f'{xarm_prefix}xarm_gripper_traj_controller',
        'ur5e_joint_trajectory_controller',
        'ur5e_rg2_gripper_traj_controller',
    ])

    launch_actions = [
        AppendEnvironmentVariable(
            name='GAZEBO_MODEL_PATH',
            value=str(cais_lab_robotics_share / 'models'),
            prepend=True,
        ),
        AppendEnvironmentVariable(
            name='GAZEBO_MODEL_PATH',
            value=str(cais_lab_robotics_share),
            prepend=True,
        ),
        gazebo_launch,
        combined_rsp,
        combined_spawn,
        TimerAction(period=2.0, actions=[controller_spawner]),
    ]
    if passive and table_surface_z_m is None:
        launch_actions.append(
            LogInfo(
                msg='[cais_lab_robotics] Passive table alignment is unavailable. '
                    'Run calibrate_hand_eye table-plane before mirroring physical gears.'
            )
        )
    elif passive:
        launch_actions.append(
            LogInfo(
                msg=f'[cais_lab_robotics] Passive table_ur5e surface aligned to '
                    f'z={table_surface_z_m:.6f} m.'
            )
        )
    if post_controller_actions:
        launch_actions.append(
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=controller_spawner,
                    on_exit=post_controller_actions,
                )
            )
        )
    if perception_log is not None and not passive:
        launch_actions.append(perception_log)
    if append_gazebo_plugin_path is not None:
        launch_actions.insert(0, append_gazebo_plugin_path)
    return launch_actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'world_file',
            default_value='table.world',
            description='Package-local Gazebo world filename under cais_lab_robotics/worlds.',
        ),
        DeclareLaunchArgument(
            'run_perception',
            default_value='true',
            description='Automatically start gazebo_camera_detector for /detect_part and /detect_all.',
        ),
        DeclareLaunchArgument(
            'passive',
            default_value='false',
            description='Spawn Gazebo as a passive mirror without perception or auto_link_attacher.',
        ),
        DeclareLaunchArgument(
            'include_assembly_parts',
            default_value='true',
            description='Spawn assembly board, gray/black boards, fixtures, and loose parts in the dual Gazebo world.',
        ),
        DeclareLaunchArgument(
            'include_loose_parts',
            default_value='true',
            description='Spawn loose gears and pins. Digital twin perception sets this false.',
        ),
        DeclareLaunchArgument(
            'include_prusa_printers_and_assembly_board',
            default_value='true',
            description=(
                'Spawn prusa_mk3, prusa_mk4_1, prusa_mk4_2, and assembly_board_v1.'
            ),
        ),
        OpaqueFunction(function=launch_setup),
    ])
