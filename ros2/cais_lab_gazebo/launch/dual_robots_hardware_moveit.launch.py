#!/usr/bin/env python3
"""
Combined hardware MoveIt2/RViz for the real xArm6 + UR5e RG2 setup.

Both robots are represented in one robot_description and one root move_group.
The xArm6 controller_manager is expected under /xarm6/controller_manager; the
UR5e hardware driver and RG2 bridge stay at root, matching the single UR5e
hardware stack.
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from uf_ros_lib.uf_robot_utils import get_xacro_content


ROBOT_BASE_Y = 0.50
RG2_PREFIX = "ur5e_rg2_"
RG2_MAX_VELOCITY = 0.40
RG2_MAX_ACCELERATION = 1.50
UR5E_MAX_VELOCITY = 0.50
UR5E_MAX_ACCELERATION = 0.80
DEFAULT_VELOCITY_SCALING = 0.20
DEFAULT_ACCELERATION_SCALING = 0.20
RTDE_ALLOWED_EXECUTION_DURATION_SCALING = 8.0
RTDE_ALLOWED_GOAL_DURATION_MARGIN = 20.0
UR5E_RTDE_TRAJECTORY_CONTROLLER = "cais_ur5e_rtde_trajectory_controller"


def _strip_world_and_ground(root: ET.Element) -> None:
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


def _launch_arg_enabled(context, name: str, default: str = "false") -> bool:
    value = LaunchConfiguration(name, default=default).perform(context)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _scale_joint_limits(joint_limits: dict) -> dict:
    scaled = {}
    for joint_name, limits in joint_limits.items():
        scaled[joint_name] = dict(limits)
    return scaled


def _build_urdf(context) -> str:
    xarm_raw = get_xacro_content(
        context,
        xacro_file=Path(get_package_share_directory("xarm_description"))
        / "urdf"
        / "xarm_device.urdf.xacro",
        dof="6",
        robot_type="xarm",
        prefix="",
        hw_ns="xarm",
        limited=True,
        effort_control=False,
        velocity_control=False,
        add_gripper="true",
        ros2_control_plugin="uf_robot_hardware/UFRobotSystemHardware",
    )
    xarm_root = ET.fromstring(xarm_raw)
    _strip_world_and_ground(xarm_root)

    ur5e_raw = subprocess.check_output(
        [
            "xacro",
            str(Path(get_package_share_directory("ur_description")) / "urdf" / "ur.urdf.xacro"),
            "ur_type:=ur5e",
            "name:=ur",
            "tf_prefix:=",
            "safety_limits:=true",
            "use_fake_hardware:=false",
        ]
    ).decode("utf-8")
    ur5e_root = ET.fromstring(ur5e_raw)
    _strip_world_and_ground(ur5e_root)

    onrobot_raw = subprocess.check_output(
        [
            "xacro",
            str(Path(get_package_share_directory("onrobot_description")) / "urdf" / "onrobot.urdf.xacro"),
            "onrobot_type:=rg2",
            "name:=rg2",
            f"prefix:={RG2_PREFIX}",
            "sim_gazebo:=false",
        ]
    ).decode("utf-8")
    onrobot_root = ET.fromstring(onrobot_raw)
    _strip_world_and_ground(onrobot_root)
    for elem in list(onrobot_root):
        ur5e_root.append(elem)

    mounting_joint = ET.Element("joint", {"name": "ur5e_rg2_gripper_mount_joint", "type": "fixed"})
    ET.SubElement(mounting_joint, "parent", {"link": "tool0"})
    ET.SubElement(mounting_joint, "child", {"link": f"{RG2_PREFIX}onrobot_base_link"})
    ET.SubElement(mounting_joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 -1.57079632679"})
    ur5e_root.append(mounting_joint)

    combined = ET.Element("robot", {"name": "dual_robots_hardware"})
    ET.SubElement(combined, "link", {"name": "world"})

    xarm_joint = ET.SubElement(combined, "joint", {"name": "xarm6_world_joint", "type": "fixed"})
    ET.SubElement(xarm_joint, "parent", {"link": "world"})
    ET.SubElement(xarm_joint, "child", {"link": "link_base"})
    ET.SubElement(xarm_joint, "origin", {"xyz": f"0.0 {-ROBOT_BASE_Y} 1.021", "rpy": "0 0 3.142"})
    for elem in list(xarm_root):
        combined.append(elem)

    ur5e_joint = ET.SubElement(combined, "joint", {"name": "ur5e_world_joint", "type": "fixed"})
    ET.SubElement(ur5e_joint, "parent", {"link": "world"})
    ET.SubElement(ur5e_joint, "child", {"link": "base_link"})
    ET.SubElement(ur5e_joint, "origin", {"xyz": f"0.0 {ROBOT_BASE_Y} 1.021", "rpy": "0 0 3.142"})
    for elem in list(ur5e_root):
        combined.append(elem)

    return ET.tostring(combined, encoding="unicode")


def _build_srdf() -> str:
    xarm_srdf_raw = subprocess.check_output(
        [
            "xacro",
            str(Path(get_package_share_directory("xarm_moveit_config")) / "srdf" / "xarm.srdf.xacro"),
            "prefix:=",
            "dof:=6",
            "robot_type:=xarm",
            "add_gripper:=true",
            "add_vacuum_gripper:=false",
            "add_bio_gripper:=false",
            "add_other_geometry:=false",
        ]
    ).decode("utf-8")
    ur5e_srdf_raw = subprocess.check_output(
        [
            "xacro",
            str(Path(get_package_share_directory("ur_moveit_config")) / "srdf" / "ur.srdf.xacro"),
            "name:=ur",
            "prefix:=",
        ]
    ).decode("utf-8")

    xarm_srdf = ET.fromstring(xarm_srdf_raw)
    ur5e_srdf = ET.fromstring(ur5e_srdf_raw)
    merged = ET.Element("robot", {"name": "dual_robots_hardware"})
    for elem in list(xarm_srdf):
        merged.append(elem)
    for elem in list(ur5e_srdf):
        merged.append(elem)

    dual_group = ET.SubElement(merged, "group", {"name": "dual_robots"})
    ET.SubElement(dual_group, "group", {"name": "xarm6"})
    ET.SubElement(dual_group, "group", {"name": "ur_manipulator"})

    rg2_group = ET.SubElement(merged, "group", {"name": "ur5e_rg2_gripper"})
    ET.SubElement(rg2_group, "joint", {"name": "ur5e_rg2_finger_width"})

    open_state = ET.SubElement(merged, "group_state", {"name": "open", "group": "ur5e_rg2_gripper"})
    ET.SubElement(open_state, "joint", {"name": "ur5e_rg2_finger_width", "value": "0.11"})

    close_state = ET.SubElement(merged, "group_state", {"name": "close", "group": "ur5e_rg2_gripper"})
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
    disabled_links = rg2_links + ["tool0", "wrist_3_link", "wrist_2_link"]
    for index, link1 in enumerate(disabled_links):
        for link2 in disabled_links[index + 1 :]:
            ET.SubElement(merged, "disable_collisions", {"link1": link1, "link2": link2, "reason": "Never"})

    return ET.tostring(merged, encoding="unicode")


def _build_moveit_params(
    urdf: str,
    srdf: str,
) -> dict:
    from ur_moveit_config.launch_common import load_yaml

    kinematics = {
        "xarm6": {
            "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
            "kinematics_solver_search_resolution": 0.005,
            "kinematics_solver_timeout": 0.005,
            "kinematics_solver_attempts": 3,
        },
        "ur_manipulator": {
            "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
            "kinematics_solver_search_resolution": 0.005,
            "kinematics_solver_timeout": 0.005,
            "kinematics_solver_attempts": 3,
        },
    }

    combined_limits: dict = {}
    xarm_limits = load_yaml("xarm_moveit_config", "config/xarm6/joint_limits.yaml")
    if "joint_limits" in xarm_limits:
        combined_limits.update(_scale_joint_limits(xarm_limits["joint_limits"]))
    xarm_gripper_limits = load_yaml("xarm_moveit_config", "config/xarm_gripper/joint_limits.yaml")
    if "joint_limits" in xarm_gripper_limits:
        combined_limits.update(_scale_joint_limits(xarm_gripper_limits["joint_limits"]))
    ur_limits = load_yaml("ur_moveit_config", "config/joint_limits.yaml")
    if "joint_limits" in ur_limits:
        combined_limits.update(_scale_joint_limits(ur_limits["joint_limits"]))
    for joint_name in (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ):
        joint_limit = dict(combined_limits.get(joint_name, {}))
        joint_limit.update(
            {
                "has_velocity_limits": True,
                "max_velocity": UR5E_MAX_VELOCITY,
                "has_acceleration_limits": True,
                "max_acceleration": UR5E_MAX_ACCELERATION,
            }
        )
        combined_limits[joint_name] = joint_limit
    combined_limits["ur5e_rg2_finger_width"] = {
        "has_position_limits": True,
        "min_position": 0.015,
        "max_position": 0.110,
        "has_velocity_limits": True,
        "max_velocity": RG2_MAX_VELOCITY,
        "has_acceleration_limits": True,
        "max_acceleration": RG2_MAX_ACCELERATION,
    }

    xarm_ompl_defs = load_yaml("xarm_moveit_config", "config/moveit_configs/ompl_defaults.yaml")
    ur_ompl = load_yaml("ur_moveit_config", "config/ompl_planning.yaml")
    planner_configs = {}
    planner_configs.update(xarm_ompl_defs.get("planner_configs", {}))
    planner_configs.update(ur_ompl.get("planner_configs", {}))

    xarm_ompl = load_yaml("xarm_moveit_config", "config/xarm6/ompl_planning.yaml")
    xarm_gripper_ompl = load_yaml("xarm_moveit_config", "config/xarm_gripper/ompl_planning.yaml")

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
    if "xarm6" in xarm_ompl:
        ompl["move_group"]["xarm6"] = xarm_ompl["xarm6"]
    if "xarm_gripper" in xarm_gripper_ompl:
        ompl["move_group"]["xarm_gripper"] = xarm_gripper_ompl["xarm_gripper"]
    if "ur_manipulator" in ur_ompl:
        ompl["move_group"]["ur_manipulator"] = ur_ompl["ur_manipulator"]
    ompl["move_group"]["dual_robots"] = {
        "planner_configs": ["RRTConnectkConfigDefault"],
        "projection_evaluator": "joints(joint1,joint2,shoulder_pan_joint,shoulder_lift_joint)",
        "longest_valid_segment_fraction": 0.005,
    }
    ompl["move_group"]["ur5e_rg2_gripper"] = {
        "planner_configs": ["RRTConnectkConfigDefault"],
        "projection_evaluator": "joints(ur5e_rg2_finger_width)",
        "longest_valid_segment_fraction": 0.005,
    }

    ur5e_arm_controller = UR5E_RTDE_TRAJECTORY_CONTROLLER
    controllers = {
        "moveit_simple_controller_manager": {
            "controller_names": [
                "xarm6/xarm6_traj_controller",
                "xarm6/xarm_gripper",
                ur5e_arm_controller,
                "ur5e_rg2_gripper_traj_controller",
            ],
            "xarm6/xarm6_traj_controller": {
                "action_ns": "follow_joint_trajectory",
                "type": "FollowJointTrajectory",
                "default": True,
                "joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            },
            "xarm6/xarm_gripper": {
                "action_ns": "gripper_action",
                "type": "GripperCommand",
                "default": True,
                "joints": ["drive_joint"],
            },
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
        "robot_description_kinematics": kinematics,
        "robot_description_planning": {
            "default_velocity_scaling_factor": DEFAULT_VELOCITY_SCALING,
            "default_acceleration_scaling_factor": DEFAULT_ACCELERATION_SCALING,
            "joint_limits": combined_limits,
        },
    }
    config.update(ompl)
    config.update(controllers)
    config.update(trajectory_execution)
    config.update(planning_scene_monitor)
    return config


def launch_setup(context, *args, **kwargs):
    launch_rviz = _launch_arg_enabled(context, "launch_rviz", default="true")
    urdf = _build_urdf(context)
    srdf = _build_srdf()
    moveit_config = _build_moveit_params(urdf, srdf)

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[moveit_config, {"use_sim_time": False}],
    )

    rviz_config = PathJoinSubstitution(
        [FindPackageShare("xarm_gazebo"), "rviz", "dual_robots_hardware_moveit.rviz"]
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
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

    launch_actions = [move_group]
    if launch_rviz:
        launch_actions.append(rviz)
    return launch_actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "launch_rviz",
                default_value="true",
                description="Launch RViz alongside the combined dual robots hardware MoveIt stack.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
