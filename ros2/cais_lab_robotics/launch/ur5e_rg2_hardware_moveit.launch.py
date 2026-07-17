#!/usr/bin/env python3
"""
UR5e hardware MoveIt2 with an attached OnRobot RG2.

Usage:
    ros2 launch cais_lab_robotics ur5e_rg2_hardware_moveit.launch.py
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

RG2_PREFIX = "ur5e_rg2_"
UR5E_BASE_XYZ = "0.0 0.0 1.021"
UR5E_BASE_RPY = "0 0 3.142"


def _load_hardware_arms_config() -> dict[str, Any]:
    path = (
        Path(get_package_share_directory("cais_lab_robotics"))
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


def _str(config: dict[str, Any], keys: tuple[str, ...], default: str) -> str:
    value = str(_nested(config, keys, default) or "").strip()
    return value or str(default)


HARDWARE_ARMS_CONFIG = _load_hardware_arms_config()
RG2_MAX_VELOCITY = _float(
    HARDWARE_ARMS_CONFIG,
    ("ur5e", "moveit", "gripper_joint_limits", "max_velocity"),
    0.40,
)
RG2_MAX_ACCELERATION = _float(
    HARDWARE_ARMS_CONFIG,
    ("ur5e", "moveit", "gripper_joint_limits", "max_acceleration"),
    1.50,
)
UR5E_MAX_VELOCITY = _float(
    HARDWARE_ARMS_CONFIG,
    ("ur5e", "moveit", "arm_joint_limits", "max_velocity"),
    0.90,
)
UR5E_MAX_ACCELERATION = _float(
    HARDWARE_ARMS_CONFIG,
    ("ur5e", "moveit", "arm_joint_limits", "max_acceleration"),
    1.80,
)
DEFAULT_VELOCITY_SCALING = _float(
    HARDWARE_ARMS_CONFIG,
    ("ur5e", "moveit", "default_velocity_scaling"),
    0.50,
)
DEFAULT_ACCELERATION_SCALING = _float(
    HARDWARE_ARMS_CONFIG,
    ("ur5e", "moveit", "default_acceleration_scaling"),
    0.50,
)
RTDE_ALLOWED_EXECUTION_DURATION_SCALING = _float(
    HARDWARE_ARMS_CONFIG,
    ("ur5e", "moveit", "rtde_allowed_execution_duration_scaling"),
    8.0,
)
RTDE_ALLOWED_GOAL_DURATION_MARGIN = _float(
    HARDWARE_ARMS_CONFIG,
    ("ur5e", "moveit", "rtde_allowed_goal_duration_margin"),
    20.0,
)
UR5E_RTDE_TRAJECTORY_CONTROLLER = _str(
    HARDWARE_ARMS_CONFIG,
    ("ur5e", "moveit", "rtde_trajectory_controller"),
    "cais_ur5e_rtde_trajectory_controller",
)


def _strip_world_and_ground(root):
    strip_names = {"world", "ground_plane"}
    for joint in list(root.findall("joint")):
        parent = joint.find("parent")
        child = joint.find("child")
        p_name = parent.get("link") if parent is not None else ""
        c_name = child.get("link") if child is not None else ""
        if p_name in strip_names or c_name in strip_names:
            root.remove(joint)
    for link in list(root.findall("link")):
        if link.get("name") in strip_names:
            root.remove(link)


def _launch_arg_enabled(context, name, default="false"):
    value = LaunchConfiguration(name, default=default).perform(context)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _build_urdf():
    ur5e_raw = subprocess.check_output([
        "xacro",
        str(Path(get_package_share_directory("ur_description")) / "urdf" / "ur.urdf.xacro"),
        "ur_type:=ur5e",
        "name:=ur",
        "tf_prefix:=",
        "safety_limits:=true",
        "use_fake_hardware:=false",
    ]).decode("utf-8")
    ur5e_root = ET.fromstring(ur5e_raw)
    for joint in ur5e_root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if (
            parent is not None
            and child is not None
            and parent.get("link") == "world"
            and child.get("link") == "base_link"
        ):
            origin = joint.find("origin")
            if origin is None:
                origin = ET.SubElement(joint, "origin")
            origin.set("xyz", UR5E_BASE_XYZ)
            origin.set("rpy", UR5E_BASE_RPY)
            break

    onrobot_raw = subprocess.check_output([
        "xacro",
        str(Path(get_package_share_directory("onrobot_description")) / "urdf" / "onrobot.urdf.xacro"),
        "onrobot_type:=rg2",
        "name:=rg2",
        f"prefix:={RG2_PREFIX}",
        "sim_gazebo:=false",
    ]).decode("utf-8")
    onrobot_root = ET.fromstring(onrobot_raw)
    _strip_world_and_ground(onrobot_root)

    for elem in list(onrobot_root):
        ur5e_root.append(elem)

    mounting_joint = ET.Element("joint", {"name": "ur5e_rg2_gripper_mount_joint", "type": "fixed"})
    ET.SubElement(mounting_joint, "parent", {"link": "tool0"})
    ET.SubElement(mounting_joint, "child", {"link": f"{RG2_PREFIX}onrobot_base_link"})
    ET.SubElement(mounting_joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 -1.57079632679"})
    ur5e_root.append(mounting_joint)

    ur5e_root.attrib["name"] = "ur5e_rg2_hardware"
    return ET.tostring(ur5e_root, encoding="unicode")


def _build_srdf():
    srdf_raw = subprocess.check_output([
        "xacro",
        str(Path(get_package_share_directory("ur_moveit_config")) / "srdf" / "ur.srdf.xacro"),
        "name:=ur",
        "prefix:=",
    ]).decode("utf-8")
    srdf_root = ET.fromstring(srdf_raw)
    srdf_root.attrib["name"] = "ur5e_rg2_hardware"

    rg2_group = ET.SubElement(srdf_root, "group", {"name": "ur5e_rg2_gripper"})
    ET.SubElement(rg2_group, "joint", {"name": "ur5e_rg2_finger_width"})

    open_state = ET.SubElement(srdf_root, "group_state", {"name": "open", "group": "ur5e_rg2_gripper"})
    ET.SubElement(open_state, "joint", {"name": "ur5e_rg2_finger_width", "value": "0.11"})

    close_state = ET.SubElement(srdf_root, "group_state", {"name": "close", "group": "ur5e_rg2_gripper"})
    ET.SubElement(close_state, "joint", {"name": "ur5e_rg2_finger_width", "value": "0.020"})

    rg2_links = [
        f"{RG2_PREFIX}onrobot_base_link",
        f"{RG2_PREFIX}left_outer_knuckle",
        f"{RG2_PREFIX}right_outer_knuckle",
        f"{RG2_PREFIX}left_inner_knuckle",
        f"{RG2_PREFIX}right_inner_knuckle",
        f"{RG2_PREFIX}left_inner_finger",
        f"{RG2_PREFIX}right_inner_finger",
        f"{RG2_PREFIX}left_finger_tip",
        f"{RG2_PREFIX}right_finger_tip",
        f"{RG2_PREFIX}finger_width_mock_link",
        f"{RG2_PREFIX}gripper_tcp",
    ]
    adjacent_links = [
        "tool0",
        "wrist_3_link",
        "wrist_2_link",
    ]
    disabled_links = rg2_links + adjacent_links
    for index, link1 in enumerate(disabled_links):
        for link2 in disabled_links[index + 1:]:
            ET.SubElement(
                srdf_root,
                "disable_collisions",
                {"link1": link1, "link2": link2, "reason": "Never"},
            )

    return ET.tostring(srdf_root, encoding="unicode")


def _build_moveit_params(urdf, srdf):
    from ur_moveit_config.launch_common import load_yaml

    robot_description_kinematics = {
        "ur_manipulator": {
            "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
            "kinematics_solver_search_resolution": 0.005,
            "kinematics_solver_timeout": 0.005,
            "kinematics_solver_attempts": 3,
        }
    }

    joint_limits_yaml = load_yaml("ur_moveit_config", "config/joint_limits.yaml")
    joint_limits = dict(joint_limits_yaml.get("joint_limits", {}))
    for joint_name in (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ):
        joint_limit = dict(joint_limits.get(joint_name, {}))
        joint_limit.update({
            "has_velocity_limits": True,
            "max_velocity": UR5E_MAX_VELOCITY,
            "has_acceleration_limits": True,
            "max_acceleration": UR5E_MAX_ACCELERATION,
        })
        joint_limits[joint_name] = joint_limit
    joint_limits["ur5e_rg2_finger_width"] = {
        "has_position_limits": True,
        "min_position": 0.015,
        "max_position": 0.110,
        "has_velocity_limits": True,
        "max_velocity": RG2_MAX_VELOCITY,
        "has_acceleration_limits": True,
        "max_acceleration": RG2_MAX_ACCELERATION,
    }

    ur_ompl = load_yaml("ur_moveit_config", "config/ompl_planning.yaml")
    planner_configs = ur_ompl.get(
        "planner_configs",
        {"RRTConnectkConfigDefault": {"type": "geometric::RRTConnect"}},
    )
    ompl = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": (
                "default_planner_request_adapters/AddTimeOptimalParameterization "
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ),
            "start_state_max_bounds_error": 0.1,
            "planner_configs": planner_configs,
        }
    }
    if "ur_manipulator" in ur_ompl:
        ompl["move_group"]["ur_manipulator"] = ur_ompl["ur_manipulator"]
    ompl["move_group"]["ur5e_rg2_gripper"] = {
        "planner_configs": ["RRTConnectkConfigDefault"],
        "projection_evaluator": "joints(ur5e_rg2_finger_width)",
        "longest_valid_segment_fraction": 0.005,
    }

    ur5e_arm_controller = UR5E_RTDE_TRAJECTORY_CONTROLLER
    controllers = {
        "moveit_simple_controller_manager": {
            "controller_names": [
                ur5e_arm_controller,
                "joint_trajectory_controller",
                "ur5e_rg2_gripper_traj_controller",
            ],
            ur5e_arm_controller: {
                "action_ns": "follow_joint_trajectory",
                "type": "FollowJointTrajectory",
                "default": True,
                "joints": [
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
            },
            "joint_trajectory_controller": {
                "action_ns": "follow_joint_trajectory",
                "type": "FollowJointTrajectory",
                "default": False,
                "joints": [
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
            },
            "ur5e_rg2_gripper_traj_controller": {
                "action_ns": "follow_joint_trajectory",
                "type": "FollowJointTrajectory",
                "default": True,
                "joints": ["ur5e_rg2_finger_width"],
            },
        },
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    trajectory_execution = {
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": RTDE_ALLOWED_EXECUTION_DURATION_SCALING,
        "trajectory_execution.allowed_goal_duration_margin": RTDE_ALLOWED_GOAL_DURATION_MARGIN,
        "trajectory_execution.allowed_start_tolerance": 0.01,
        "trajectory_execution.execution_duration_monitoring": False,
    }

    planning_scene_monitor = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    config = {
        "robot_description": urdf,
        "robot_description_semantic": srdf,
        "robot_description_kinematics": robot_description_kinematics,
        "robot_description_planning": {
            "default_velocity_scaling_factor": DEFAULT_VELOCITY_SCALING,
            "default_acceleration_scaling_factor": DEFAULT_ACCELERATION_SCALING,
            "joint_limits": joint_limits,
        },
    }
    config.update(ompl)
    config.update(controllers)
    config.update(trajectory_execution)
    config.update(planning_scene_monitor)
    return config


def launch_setup(context, *args, **kwargs):
    launch_rviz = _launch_arg_enabled(context, "launch_rviz", default="true")

    urdf = _build_urdf()
    srdf = _build_srdf()
    moveit_config = _build_moveit_params(urdf, srdf)

    rg2_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="ur5e_rg2_robot_state_publisher",
        output="log",
        parameters=[
            {
                "robot_description": urdf,
                "use_sim_time": False,
            },
        ],
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config, {"use_sim_time": False}],
    )

    rviz_config = PathJoinSubstitution([
        FindPackageShare("cais_lab_robotics"),
        "rviz",
        "ur5e_rg2_hardware_moveit.rviz",
    ])
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            {
                "robot_description": urdf,
                "robot_description_semantic": srdf,
                "robot_description_kinematics": moveit_config["robot_description_kinematics"],
                "robot_description_planning": moveit_config["robot_description_planning"],
                "use_sim_time": False,
            },
        ],
    )

    launch_actions = [rg2_robot_state_publisher, move_group]
    if launch_rviz:
        launch_actions.append(rviz)
    return launch_actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz alongside UR5e RG2 hardware MoveIt.",
        ),
        OpaqueFunction(function=launch_setup),
    ])
