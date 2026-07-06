#!/usr/bin/env python3
"""xArm6 hardware MoveIt2/RViz configured by xarm6_ur5e_hardware_runtime.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file


def _load_hardware_arms_config() -> dict[str, Any]:
    path = (
        Path(get_package_share_directory("xarm_gazebo"))
        / "config"
        / "hardware_runtime"
        / "xarm6_ur5e_hardware_runtime.yaml"
    )
    with path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return dict(loaded) if isinstance(loaded, dict) else {}


def _nested(config: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _float(config: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    try:
        return float(_nested(config, keys, default))
    except (TypeError, ValueError):
        return float(default)


def _bool_text(context, name: str, default: str) -> str:
    value = LaunchConfiguration(name, default=default).perform(context)
    return "true" if str(value or "").strip().lower() in {"1", "true", "yes", "on"} else "false"


def _config_bool_text(config: dict[str, Any], keys: tuple[str, ...], default: bool) -> str:
    value = _nested(config, keys, default)
    if isinstance(value, bool):
        return "true" if value else "false"
    return "true" if str(value or "").strip().lower() in {"1", "true", "yes", "on"} else "false"


def _apply_xarm6_moveit_config(moveit_config_dict: dict[str, Any], hardware_config: dict[str, Any]) -> None:
    xarm6_config = dict(hardware_config.get("xarm6") or {})
    planning = dict(moveit_config_dict.get("robot_description_planning") or {})
    joint_limits = dict(planning.get("joint_limits") or {})
    arm_max_velocity = _float(xarm6_config, ("moveit", "arm_joint_limits", "max_velocity"), 2.14)
    arm_max_acceleration = _float(
        xarm6_config,
        ("moveit", "arm_joint_limits", "max_acceleration"),
        10.0,
    )
    for joint_name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
        joint_limit = dict(joint_limits.get(joint_name) or {})
        joint_limit.update(
            {
                "has_velocity_limits": True,
                "max_velocity": arm_max_velocity,
                "has_acceleration_limits": True,
                "max_acceleration": arm_max_acceleration,
            }
        )
        joint_limits[joint_name] = joint_limit

    gripper_max_velocity = _float(
        xarm6_config,
        ("moveit", "gripper_joint_limits", "max_velocity"),
        3.14,
    )
    gripper_max_acceleration = _float(
        xarm6_config,
        ("moveit", "gripper_joint_limits", "max_acceleration"),
        10.0,
    )
    for joint_name in (
        "drive_joint",
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "right_finger_joint",
        "right_inner_knuckle_joint",
        "right_outer_knuckle_joint",
    ):
        if joint_name not in joint_limits:
            continue
        joint_limit = dict(joint_limits.get(joint_name) or {})
        joint_limit.update(
            {
                "has_velocity_limits": True,
                "max_velocity": gripper_max_velocity,
                "has_acceleration_limits": True,
                "max_acceleration": gripper_max_acceleration,
            }
        )
        joint_limits[joint_name] = joint_limit

    planning["default_velocity_scaling_factor"] = _float(
        xarm6_config,
        ("moveit", "default_velocity_scaling"),
        0.50,
    )
    planning["default_acceleration_scaling_factor"] = _float(
        xarm6_config,
        ("moveit", "default_acceleration_scaling"),
        0.50,
    )
    planning["joint_limits"] = joint_limits
    moveit_config_dict["robot_description_planning"] = planning


def launch_setup(context, *args, **kwargs):
    hardware_config = _load_hardware_arms_config()
    xarm6_config = dict(hardware_config.get("xarm6") or {})
    robot_ip = LaunchConfiguration(
        "robot_ip",
        default=str(_nested(xarm6_config, ("robot_ip",), "192.168.1.240")),
    )
    report_type = LaunchConfiguration("report_type", default="normal")
    baud_checkset = LaunchConfiguration("baud_checkset", default=True)
    default_gripper_baud = LaunchConfiguration("default_gripper_baud", default=2000000)

    dof = LaunchConfiguration("dof", default=str(_nested(xarm6_config, ("moveit", "dof"), 6)))
    robot_type = LaunchConfiguration(
        "robot_type",
        default=str(_nested(xarm6_config, ("moveit", "robot_type"), "xarm")),
    )
    prefix = LaunchConfiguration("prefix", default="")
    hw_ns = LaunchConfiguration("hw_ns", default=str(_nested(xarm6_config, ("moveit", "hw_ns"), "xarm")))
    limited = LaunchConfiguration("limited", default=True)
    effort_control = LaunchConfiguration("effort_control", default=False)
    velocity_control = LaunchConfiguration("velocity_control", default=False)
    model1300 = LaunchConfiguration("model1300", default=False)
    robot_sn = LaunchConfiguration("robot_sn", default="")
    attach_to = LaunchConfiguration("attach_to", default="world")
    attach_xyz = LaunchConfiguration("attach_xyz", default='"0 0 0"')
    attach_rpy = LaunchConfiguration("attach_rpy", default='"0 0 0"')
    mesh_suffix = LaunchConfiguration("mesh_suffix", default="stl")
    kinematics_suffix = LaunchConfiguration("kinematics_suffix", default="")

    add_gripper = LaunchConfiguration(
        "add_gripper",
        default=_config_bool_text(xarm6_config, ("moveit", "add_gripper"), True),
    )
    add_vacuum_gripper = LaunchConfiguration("add_vacuum_gripper", default=False)
    add_bio_gripper = LaunchConfiguration("add_bio_gripper", default=False)
    add_realsense_d435i = LaunchConfiguration("add_realsense_d435i", default=False)
    add_d435i_links = LaunchConfiguration("add_d435i_links", default=True)
    add_other_geometry = LaunchConfiguration("add_other_geometry", default=False)
    geometry_type = LaunchConfiguration("geometry_type", default="box")
    geometry_mass = LaunchConfiguration("geometry_mass", default=0.1)
    geometry_height = LaunchConfiguration("geometry_height", default=0.1)
    geometry_radius = LaunchConfiguration("geometry_radius", default=0.1)
    geometry_length = LaunchConfiguration("geometry_length", default=0.1)
    geometry_width = LaunchConfiguration("geometry_width", default=0.1)
    geometry_mesh_filename = LaunchConfiguration("geometry_mesh_filename", default="")
    geometry_mesh_origin_xyz = LaunchConfiguration("geometry_mesh_origin_xyz", default='"0 0 0"')
    geometry_mesh_origin_rpy = LaunchConfiguration("geometry_mesh_origin_rpy", default='"0 0 0"')
    geometry_mesh_tcp_xyz = LaunchConfiguration("geometry_mesh_tcp_xyz", default='"0 0 0"')
    geometry_mesh_tcp_rpy = LaunchConfiguration("geometry_mesh_tcp_rpy", default='"0 0 0"')

    no_gui_ctrl = LaunchConfiguration("no_gui_ctrl", default=False)
    ros_namespace = LaunchConfiguration("ros_namespace", default="").perform(context)
    show_rviz = _bool_text(context, "show_rviz", _bool_text(context, "launch_rviz", "true"))

    ros2_control_plugin = "uf_robot_hardware/UFRobotSystemHardware"
    controllers_name = "controllers"
    xarm_type = "{}{}".format(
        robot_type.perform(context),
        dof.perform(context) if robot_type.perform(context) in ("xarm", "lite") else "",
    )

    ros2_control_params = generate_ros2_control_params_temp_file(
        os.path.join(
            get_package_share_directory("xarm_controller"),
            "config",
            f"{xarm_type}_controllers.yaml",
        ),
        prefix=prefix.perform(context),
        add_gripper=add_gripper.perform(context) in ("True", "true"),
        add_bio_gripper=add_bio_gripper.perform(context) in ("True", "true"),
        ros_namespace=ros_namespace,
        robot_type=robot_type.perform(context),
    )

    moveit_config = MoveItConfigsBuilder(
        context=context,
        controllers_name=controllers_name,
        robot_ip=robot_ip,
        report_type=report_type,
        baud_checkset=baud_checkset,
        default_gripper_baud=default_gripper_baud,
        dof=dof,
        robot_type=robot_type,
        prefix=prefix,
        hw_ns=hw_ns,
        limited=limited,
        effort_control=effort_control,
        velocity_control=velocity_control,
        model1300=model1300,
        robot_sn=robot_sn,
        attach_to=attach_to,
        attach_xyz=attach_xyz,
        attach_rpy=attach_rpy,
        mesh_suffix=mesh_suffix,
        kinematics_suffix=kinematics_suffix,
        ros2_control_plugin=ros2_control_plugin,
        ros2_control_params=ros2_control_params,
        add_gripper=add_gripper,
        add_vacuum_gripper=add_vacuum_gripper,
        add_bio_gripper=add_bio_gripper,
        add_realsense_d435i=add_realsense_d435i,
        add_d435i_links=add_d435i_links,
        add_other_geometry=add_other_geometry,
        geometry_type=geometry_type,
        geometry_mass=geometry_mass,
        geometry_height=geometry_height,
        geometry_radius=geometry_radius,
        geometry_length=geometry_length,
        geometry_width=geometry_width,
        geometry_mesh_filename=geometry_mesh_filename,
        geometry_mesh_origin_xyz=geometry_mesh_origin_xyz,
        geometry_mesh_origin_rpy=geometry_mesh_origin_rpy,
        geometry_mesh_tcp_xyz=geometry_mesh_tcp_xyz,
        geometry_mesh_tcp_rpy=geometry_mesh_tcp_rpy,
    ).to_moveit_configs()

    moveit_config_dict = moveit_config.to_dict()
    _apply_xarm6_moveit_config(moveit_config_dict, hardware_config)

    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("xarm_description"), "launch", "_robot_description.launch.py"]
            )
        ),
        launch_arguments={
            "robot_description": yaml.dump(moveit_config.robot_description),
        }.items(),
    )

    robot_moveit_common_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("xarm_moveit_config"), "launch", "_robot_moveit_common2.launch.py"]
            )
        ),
        launch_arguments={
            "prefix": prefix,
            "attach_to": attach_to,
            "attach_xyz": attach_xyz,
            "attach_rpy": attach_rpy,
            "no_gui_ctrl": no_gui_ctrl,
            "show_rviz": show_rviz,
            "use_sim_time": "false",
            "moveit_config_dump": yaml.dump(moveit_config_dict),
        }.items(),
    )

    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"source_list": [f"{prefix.perform(context)}{hw_ns.perform(context)}/joint_states"]}],
        remappings=[
            (
                "follow_joint_trajectory",
                f"{prefix.perform(context)}{xarm_type}_traj_controller/follow_joint_trajectory",
            ),
        ],
    )

    ros2_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("xarm_controller"), "launch", "_ros2_control.launch.py"])
        ),
        launch_arguments={
            "robot_description": yaml.dump(moveit_config.robot_description),
            "ros2_control_params": ros2_control_params,
        }.items(),
    )

    control_node = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            f"{prefix.perform(context)}{xarm_type}_traj_controller",
            "--controller-manager",
            f"{ros_namespace}/controller_manager",
        ],
    )

    return [
        robot_description_launch,
        robot_moveit_common_launch,
        joint_state_publisher_node,
        ros2_control_launch,
        control_node,
    ]


def generate_launch_description():
    hardware_config = _load_hardware_arms_config()
    xarm6_config = dict(hardware_config.get("xarm6") or {})
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_ip",
                default_value=str(_nested(xarm6_config, ("robot_ip",), "192.168.1.240")),
                description="xArm6 controller IP address.",
            ),
            DeclareLaunchArgument(
                "show_rviz",
                default_value="true",
                description="Launch RViz alongside xArm6 hardware MoveIt.",
            ),
            DeclareLaunchArgument(
                "launch_rviz",
                default_value="true",
                description="Alias for show_rviz.",
            ),
            DeclareLaunchArgument(
                "add_gripper",
                default_value=_config_bool_text(xarm6_config, ("moveit", "add_gripper"), True),
                description="Attach the xArm gripper to the xArm6 hardware MoveIt model.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
