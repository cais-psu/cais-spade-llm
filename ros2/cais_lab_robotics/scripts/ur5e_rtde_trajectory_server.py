#!/usr/bin/env python3.10
"""UR5e RTDE-backed joint and Cartesian action server."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import rclpy
import yaml
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

try:
    from cais_lab_robotics.action import (
        MoveUR5eCartesian,
        MoveUR5eJointJog,
        MoveUR5eRelativeCartesian,
        RecordUR5eInsertionDemonstration,
    )
    from cais_lab_robotics.srv import SetUR5eCartesianJog
except ImportError:  # Installed ROS interfaces may not be rebuilt yet.
    MoveUR5eCartesian = None
    MoveUR5eJointJog = None
    MoveUR5eRelativeCartesian = None
    RecordUR5eInsertionDemonstration = None
    SetUR5eCartesianJog = None

try:
    from cais_lab_robotics.action import MoveUR5eInsert
except ImportError:  # Installed ROS interfaces may not be rebuilt yet.
    MoveUR5eInsert = None

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
ACTION_NAME = "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
CARTESIAN_ACTION_NAME = "/cais_ur5e_rtde_cartesian_controller/move_cartesian"
INSERT_ACTION_NAME = "/cais_ur5e_rtde_cartesian_controller/move_insert"
INSERT_DEMONSTRATION_ACTION_NAME = (
    "/cais_ur5e_rtde_cartesian_controller/record_insertion_demonstration"
)
INSERT_SUPPORTED_PART_NAMES = ("SG", "MG", "LG", "SCP", "MCP", "LCP")
INSERT_FORCE_DEPTH_PROFILE_POINTS = 16
INSERT_MG_HARD_CAP_FIELDS = (
    "insert_max_insertion_force_n",
    "insert_max_axial_force_n",
    "insert_max_lateral_force_n",
    "insert_max_torque_nm",
    "insert_max_tool_flange_torque_nm",
    "insert_max_relief_retreat_m",
    "insert_max_contact_search_radius_m",
    "insert_max_disengagement_cycles",
    "insert_search_peck_retreat_m",
    "insert_search_peck_interval_sec",
)
RELATIVE_CARTESIAN_ACTION_NAME = "/cais_ur5e_rtde_cartesian_controller/move_relative_cartesian"
CARTESIAN_JOG_SERVICE_NAME = "/cais_ur5e_rtde_cartesian_controller/set_cartesian_jog"
JOINT_JOG_ACTION_NAME = "/cais_ur5e_rtde_trajectory_controller/move_joint_jog"
DEFAULT_STATUS_FILE = Path("/tmp") / "cais_ur5e_rtde_trajectory_status.json"
INSERT_DEMONSTRATION_TRACE_ROOT = Path("/tmp") / "cais_ur5e_insertion_demonstrations"
INSERT_TRIAL_TRACE_ROOT = Path("/tmp") / "cais_ur5e_insert_trials"
DEFAULT_CONFIG_FILE = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "hardware_runtime"
    / "xarm6_ur5e_hardware_runtime.yaml"
)
HARDWARE_ARMS_CONFIG_FILE = str(DEFAULT_CONFIG_FILE)


def _load_hardware_arms_config(path: Path) -> dict[str, Any]:
    with Path(path).expanduser().open(encoding="utf-8") as f:
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


def _optional_float(
    config: dict[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    value = _nested(config, keys, None)
    if isinstance(value, bool):
        return None
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _required_finite_float(
    config: dict[str, Any],
    keys: tuple[str, ...],
) -> float:
    value = _nested(config, keys, None)
    if value is None or isinstance(value, bool):
        raise ValueError(f"{'.'.join(keys)} is missing or is not a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{'.'.join(keys)} is missing or is not a finite number"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{'.'.join(keys)} is missing or is not a finite number")
    return parsed


def _apply_hardware_arms_config(config_path: Path) -> None:  # noqa: PLR0915
    """Load UR5e RTDE runtime limits from xarm6_ur5e_hardware_runtime.yaml."""
    global ACTION_NAME
    global CARTESIAN_ACTION_NAME
    global INSERT_ACTION_NAME
    global INSERT_DEMONSTRATION_ACTION_NAME
    global RELATIVE_CARTESIAN_ACTION_NAME
    global CARTESIAN_JOG_SERVICE_NAME
    global UR5E_RTDE_MAX_JOINT_VEL_RAD_S
    global UR5E_RTDE_MAX_JOINT_ACCEL_RAD_S2
    global UR5E_RTDE_MAX_JOINT_JERK_RAD_S3
    global UR5E_RTDE_GOAL_TOLERANCE_RAD
    global UR5E_RTDE_SHOULDER_PAN_EXTRA_SCALE
    global UR5E_RTDE_MOVEJ_SPEED_RAD_S
    global UR5E_RTDE_MOVEJ_ACCEL_RAD_S2
    global UR5E_RTDE_CONTROL_PROGRAM_START_TIMEOUT_SEC
    global UR5E_RTDE_MOTION_START_TIMEOUT_SEC
    global UR5E_RTDE_MOTION_START_DELTA_RAD
    global UR5E_RTDE_FEEDBACK_RECONNECT_AFTER_SEC
    global UR5E_RTDE_FEEDBACK_RECONNECT_RETRY_SEC
    global UR5E_RTDE_FEEDBACK_RECOVERY_TIMEOUT_SEC
    global UR5E_RTDE_FREQUENCY_HZ
    global UR5E_RTDE_STATIONARY_HOLD_SEC
    global UR5E_RTDE_STOPPED_AWAY_HOLD_SEC
    global UR5E_RTDE_ALLOWED_EXECUTION_DURATION_SCALING
    global UR5E_RTDE_RESULT_MARGIN_SEC
    global UR5E_RTDE_CARTESIAN_SPEED_M_S
    global UR5E_RTDE_CARTESIAN_MAX_SPEED_M_S
    global UR5E_RTDE_CARTESIAN_ACCEL_M_S2
    global UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M
    global UR5E_RTDE_CARTESIAN_ORIENTATION_TOLERANCE_RAD
    global UR5E_RTDE_CARTESIAN_WORLD_BASE
    global UR5E_RTDE_CARTESIAN_WORLD_BASE_CONFIG_ERROR
    global UR5E_RTDE_CARTESIAN_WORKSPACE_BOUNDS
    global UR5E_RTDE_CARTESIAN_REACH_ORIGIN
    global UR5E_RTDE_CARTESIAN_REACH_RADIUS_M
    global UR5E_RTDE_INSERT_MAX_CONTACT_SPEED_M_S
    global UR5E_RTDE_INSERT_MAX_CONTACT_FORCE_DELTA_N
    global UR5E_RTDE_INSERT_MAX_ENGAGEMENT_PROGRESS_M
    global UR5E_RTDE_INSERT_MAX_INSERTION_FORCE_N
    global UR5E_RTDE_INSERT_MAX_SPIRAL_RADIUS_M
    global UR5E_RTDE_INSERT_MAX_SPIRAL_PITCH_M
    global UR5E_RTDE_INSERT_MAX_SPIRAL_SPEED_M_S
    global UR5E_RTDE_INSERT_MAX_SPIRAL_ACCELERATION_M_S2
    global UR5E_RTDE_INSERT_MAX_AXIAL_FORCE_N
    global UR5E_RTDE_INSERT_MAX_LATERAL_FORCE_N
    global UR5E_RTDE_INSERT_MAX_TORQUE_NM
    global UR5E_RTDE_INSERT_MAX_TOOL_FLANGE_TORQUE_NM
    global UR5E_RTDE_INSERT_SG_HARD_CAPS
    global UR5E_RTDE_INSERT_MG_HARD_CAPS
    global UR5E_RTDE_INSERT_LG_HARD_CAPS
    global UR5E_RTDE_INSERT_SCP_HARD_CAPS
    global UR5E_RTDE_INSERT_MCP_HARD_CAPS
    global UR5E_RTDE_INSERT_LCP_HARD_CAPS
    global UR5E_RTDE_INSERT_SOFT_FILTER_WINDOW_SEC
    global UR5E_RTDE_INSERT_SOFT_OVERLOAD_HOLD_SEC
    global UR5E_RTDE_INSERT_RELIEF_UNLOAD_DWELL_SEC
    global UR5E_RTDE_INSERT_RELIEF_CLEAR_DWELL_SEC
    global UR5E_RTDE_INSERT_RELIEF_CLEAR_HYSTERESIS_RATIO
    global UR5E_RTDE_INSERT_RELIEF_TIMEOUT_SEC
    global UR5E_RTDE_INSERT_RELIEF_AXIAL_FORCE_RATIO
    global UR5E_RTDE_INSERT_RELIEF_REVERSE_FORCE_RATIO
    global UR5E_RTDE_INSERT_RELIEF_RESUME_RAMP_SEC
    global UR5E_RTDE_INSERT_RELIEF_SEARCH_FORCE_RATIO
    global UR5E_RTDE_INSERT_RELIEF_SEARCH_SPEED_RATIO
    global UR5E_RTDE_INSERT_RELIEF_BACKOFF_STEP_M
    global UR5E_RTDE_INSERT_MAX_RELIEF_RETREAT_M
    global UR5E_RTDE_INSERT_RELIEF_STATIONARY_SPEED_M_S
    global UR5E_RTDE_INSERT_RELIEF_STATIONARY_ANGULAR_SPEED_RAD_S
    global UR5E_RTDE_INSERT_MAX_RELIEF_CYCLES
    global UR5E_RTDE_INSERT_MAX_TILT_TOLERANCE_RAD
    global UR5E_RTDE_INSERT_MAX_SEATED_DEPTH_TOLERANCE_M
    global UR5E_RTDE_INSERT_MAX_SETTLE_TIME_SEC
    global UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC
    global UR5E_RTDE_INSERT_MAX_TRAVEL_M
    global UR5E_RTDE_INSERT_START_POSITION_TOLERANCE_M
    global UR5E_RTDE_INSERT_START_ORIENTATION_TOLERANCE_RAD
    global UR5E_RTDE_INSERT_DEMONSTRATION_MAX_DURATION_SEC
    global UR5E_RTDE_INSERT_DEMONSTRATION_BASELINE_SEC
    global UR5E_RTDE_INSERT_DEMONSTRATION_STATIONARY_SPEED_M_S
    global UR5E_RTDE_INSERT_DEMONSTRATION_STATIONARY_ANGULAR_SPEED_RAD_S
    global HARDWARE_ARMS_CONFIG_FILE

    HARDWARE_ARMS_CONFIG_FILE = str(Path(config_path).expanduser())
    config = _load_hardware_arms_config(config_path)
    ACTION_NAME = _str(
        config,
        ("ur5e", "hardware_trajectory_action"),
        ACTION_NAME,
    )
    CARTESIAN_ACTION_NAME = _str(
        config,
        ("ur5e", "hardware_cartesian_action"),
        CARTESIAN_ACTION_NAME,
    )
    INSERT_ACTION_NAME = _str(
        config,
        ("ur5e", "hardware_insert_action"),
        INSERT_ACTION_NAME,
    )
    INSERT_DEMONSTRATION_ACTION_NAME = _str(
        config,
        ("ur5e", "hardware_insertion_demonstration_action"),
        INSERT_DEMONSTRATION_ACTION_NAME,
    )
    RELATIVE_CARTESIAN_ACTION_NAME = _str(
        config,
        ("ur5e", "hardware_relative_cartesian_action"),
        RELATIVE_CARTESIAN_ACTION_NAME,
    )
    CARTESIAN_JOG_SERVICE_NAME = _str(
        config,
        ("ur5e", "hardware_cartesian_jog_service"),
        CARTESIAN_JOG_SERVICE_NAME,
    )
    UR5E_RTDE_MAX_JOINT_VEL_RAD_S = _float(
        config,
        ("ur5e", "rtde", "max_joint_vel_rad_s"),
        UR5E_RTDE_MAX_JOINT_VEL_RAD_S,
    )
    UR5E_RTDE_MAX_JOINT_ACCEL_RAD_S2 = _float(
        config,
        ("ur5e", "rtde", "max_joint_accel_rad_s2"),
        UR5E_RTDE_MAX_JOINT_ACCEL_RAD_S2,
    )
    UR5E_RTDE_MAX_JOINT_JERK_RAD_S3 = _float(
        config,
        ("ur5e", "rtde", "max_joint_jerk_rad_s3"),
        UR5E_RTDE_MAX_JOINT_JERK_RAD_S3,
    )
    UR5E_RTDE_GOAL_TOLERANCE_RAD = max(
        1e-6,
        _float(
            config,
            ("ur5e", "rtde", "joint_goal_tolerance_rad"),
            UR5E_RTDE_GOAL_TOLERANCE_RAD,
        ),
    )
    UR5E_RTDE_SHOULDER_PAN_EXTRA_SCALE = _float(
        config,
        ("ur5e", "rtde", "shoulder_pan_extra_scale"),
        UR5E_RTDE_SHOULDER_PAN_EXTRA_SCALE,
    )
    UR5E_RTDE_MOVEJ_SPEED_RAD_S = _float(
        config,
        ("ur5e", "rtde", "movej_speed_rad_s"),
        UR5E_RTDE_MOVEJ_SPEED_RAD_S,
    )
    UR5E_RTDE_MOVEJ_ACCEL_RAD_S2 = _float(
        config,
        ("ur5e", "rtde", "movej_accel_rad_s2"),
        UR5E_RTDE_MOVEJ_ACCEL_RAD_S2,
    )
    UR5E_RTDE_CONTROL_PROGRAM_START_TIMEOUT_SEC = max(
        0.1,
        _float(
            config,
            ("ur5e", "rtde", "control_program_start_timeout_sec"),
            UR5E_RTDE_CONTROL_PROGRAM_START_TIMEOUT_SEC,
        ),
    )
    UR5E_RTDE_MOTION_START_TIMEOUT_SEC = max(
        0.1,
        _float(
            config,
            ("ur5e", "rtde", "motion_start_timeout_sec"),
            UR5E_RTDE_MOTION_START_TIMEOUT_SEC,
        ),
    )
    UR5E_RTDE_MOTION_START_DELTA_RAD = max(
        0.0,
        _float(
            config,
            ("ur5e", "rtde", "motion_start_delta_rad"),
            UR5E_RTDE_MOTION_START_DELTA_RAD,
        ),
    )
    UR5E_RTDE_FEEDBACK_RECONNECT_AFTER_SEC = max(
        0.1,
        _float(
            config,
            ("ur5e", "rtde", "feedback_reconnect_after_sec"),
            UR5E_RTDE_FEEDBACK_RECONNECT_AFTER_SEC,
        ),
    )
    UR5E_RTDE_FEEDBACK_RECOVERY_TIMEOUT_SEC = max(
        UR5E_RTDE_FEEDBACK_RECONNECT_AFTER_SEC,
        _float(
            config,
            ("ur5e", "rtde", "feedback_recovery_timeout_sec"),
            UR5E_RTDE_FEEDBACK_RECOVERY_TIMEOUT_SEC,
        ),
    )
    UR5E_RTDE_FEEDBACK_RECONNECT_RETRY_SEC = max(
        UR5E_RTDE_FEEDBACK_RECONNECT_AFTER_SEC,
        _float(
            config,
            ("ur5e", "rtde", "feedback_reconnect_retry_sec"),
            UR5E_RTDE_FEEDBACK_RECONNECT_RETRY_SEC,
        ),
    )
    UR5E_RTDE_FREQUENCY_HZ = max(
        1.0,
        min(
            500.0,
            _float(
                config,
                ("ur5e", "rtde", "frequency_hz"),
                UR5E_RTDE_FREQUENCY_HZ,
            ),
        ),
    )
    UR5E_RTDE_STATIONARY_HOLD_SEC = max(
        0.05,
        _float(
            config,
            ("ur5e", "rtde", "stationary_hold_sec"),
            UR5E_RTDE_STATIONARY_HOLD_SEC,
        ),
    )
    UR5E_RTDE_STOPPED_AWAY_HOLD_SEC = max(
        UR5E_RTDE_STATIONARY_HOLD_SEC,
        _float(
            config,
            ("ur5e", "rtde", "stopped_away_hold_sec"),
            UR5E_RTDE_STOPPED_AWAY_HOLD_SEC,
        ),
    )
    UR5E_RTDE_ALLOWED_EXECUTION_DURATION_SCALING = max(
        1.0,
        _float(
            config,
            ("ur5e", "moveit", "rtde_allowed_execution_duration_scaling"),
            UR5E_RTDE_ALLOWED_EXECUTION_DURATION_SCALING,
        ),
    )
    UR5E_RTDE_RESULT_MARGIN_SEC = max(
        0.0,
        _float(
            config,
            ("ur5e", "moveit", "rtde_allowed_goal_duration_margin"),
            UR5E_RTDE_RESULT_MARGIN_SEC,
        ),
    )
    UR5E_RTDE_CARTESIAN_SPEED_M_S = max(
        0.001,
        _float(
            config,
            ("ur5e", "rtde", "cartesian_speed_m_s"),
            UR5E_RTDE_CARTESIAN_SPEED_M_S,
        ),
    )
    UR5E_RTDE_CARTESIAN_MAX_SPEED_M_S = max(
        UR5E_RTDE_CARTESIAN_SPEED_M_S,
        _float(
            config,
            ("ur5e", "rtde", "cartesian_max_speed_m_s"),
            UR5E_RTDE_CARTESIAN_MAX_SPEED_M_S,
        ),
    )
    UR5E_RTDE_CARTESIAN_ACCEL_M_S2 = max(
        0.001,
        _float(
            config,
            ("ur5e", "rtde", "cartesian_accel_m_s2"),
            UR5E_RTDE_CARTESIAN_ACCEL_M_S2,
        ),
    )
    UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M = max(
        0.0001,
        _float(
            config,
            ("ur5e", "rtde", "cartesian_position_tolerance_m"),
            UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M,
        ),
    )
    UR5E_RTDE_CARTESIAN_ORIENTATION_TOLERANCE_RAD = max(
        0.001,
        _float(
            config,
            ("ur5e", "rtde", "cartesian_orientation_tolerance_rad"),
            UR5E_RTDE_CARTESIAN_ORIENTATION_TOLERANCE_RAD,
        ),
    )
    cartesian_world_base_keys = (
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
    )
    try:
        UR5E_RTDE_CARTESIAN_WORLD_BASE = tuple(
            _required_finite_float(
                config,
                ("ur5e", "rtde", "cartesian_world_base", field_name),
            )
            for field_name in cartesian_world_base_keys
        )
        UR5E_RTDE_CARTESIAN_WORLD_BASE_CONFIG_ERROR = ""
    except ValueError as exc:
        UR5E_RTDE_CARTESIAN_WORLD_BASE = None
        UR5E_RTDE_CARTESIAN_WORLD_BASE_CONFIG_ERROR = str(exc)
    bounds = _nested(config, ("ur5e", "rtde", "cartesian_workspace_bounds"), {})
    if isinstance(bounds, dict):
        UR5E_RTDE_CARTESIAN_WORKSPACE_BOUNDS = {
            key: _float(
                bounds,
                (key,),
                UR5E_RTDE_CARTESIAN_WORKSPACE_BOUNDS[key],
            )
            for key in UR5E_RTDE_CARTESIAN_WORKSPACE_BOUNDS
        }
    reach = _nested(config, ("ur5e", "rtde", "cartesian_reach"), {})
    if isinstance(reach, dict):
        UR5E_RTDE_CARTESIAN_REACH_ORIGIN = (
            _float(reach, ("origin_x_m",), UR5E_RTDE_CARTESIAN_REACH_ORIGIN[0]),
            _float(reach, ("origin_y_m",), UR5E_RTDE_CARTESIAN_REACH_ORIGIN[1]),
            _float(reach, ("origin_z_m",), UR5E_RTDE_CARTESIAN_REACH_ORIGIN[2]),
        )
        UR5E_RTDE_CARTESIAN_REACH_RADIUS_M = max(
            0.01,
            _float(
                reach,
                ("max_xy_radius_m",),
                UR5E_RTDE_CARTESIAN_REACH_RADIUS_M,
            ),
        )
    UR5E_RTDE_INSERT_MAX_CONTACT_SPEED_M_S = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_contact_speed_m_s"),
    )
    UR5E_RTDE_INSERT_MAX_CONTACT_FORCE_DELTA_N = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_contact_force_delta_n"),
    )
    UR5E_RTDE_INSERT_MAX_ENGAGEMENT_PROGRESS_M = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_engagement_progress_m"),
    )
    UR5E_RTDE_INSERT_MAX_INSERTION_FORCE_N = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_insertion_force_n"),
    )
    UR5E_RTDE_INSERT_MAX_SPIRAL_RADIUS_M = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_spiral_radius_m"),
    )
    UR5E_RTDE_INSERT_MAX_SPIRAL_PITCH_M = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_spiral_pitch_m"),
    )
    UR5E_RTDE_INSERT_MAX_SPIRAL_SPEED_M_S = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_spiral_speed_m_s"),
    )
    UR5E_RTDE_INSERT_MAX_SPIRAL_ACCELERATION_M_S2 = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_spiral_acceleration_m_s2"),
    )
    UR5E_RTDE_INSERT_MAX_AXIAL_FORCE_N = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_axial_force_n"),
    )
    UR5E_RTDE_INSERT_MAX_LATERAL_FORCE_N = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_lateral_force_n"),
    )
    UR5E_RTDE_INSERT_MAX_TORQUE_NM = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_torque_nm"),
    )
    UR5E_RTDE_INSERT_MAX_TOOL_FLANGE_TORQUE_NM = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_tool_flange_torque_nm"),
    )
    mg_hard_caps = _nested(config, ("ur5e", "rtde", "MG"), {})
    UR5E_RTDE_INSERT_MG_HARD_CAPS = {
        field_name: _optional_float(mg_hard_caps, (field_name,))
        for field_name in INSERT_MG_HARD_CAP_FIELDS
        if isinstance(mg_hard_caps, dict) and field_name in mg_hard_caps
    }
    sg_hard_caps = _nested(config, ("ur5e", "rtde", "SG"), {})
    UR5E_RTDE_INSERT_SG_HARD_CAPS = {
        field_name: _optional_float(sg_hard_caps, (field_name,))
        for field_name in INSERT_MG_HARD_CAP_FIELDS
        if isinstance(sg_hard_caps, dict) and field_name in sg_hard_caps
    }
    lg_hard_caps = _nested(config, ("ur5e", "rtde", "LG"), {})
    UR5E_RTDE_INSERT_LG_HARD_CAPS = {
        field_name: _optional_float(lg_hard_caps, (field_name,))
        for field_name in INSERT_MG_HARD_CAP_FIELDS
        if isinstance(lg_hard_caps, dict) and field_name in lg_hard_caps
    }
    scp_hard_caps = _nested(config, ("ur5e", "rtde", "SCP"), {})
    UR5E_RTDE_INSERT_SCP_HARD_CAPS = {
        field_name: _optional_float(scp_hard_caps, (field_name,))
        for field_name in INSERT_MG_HARD_CAP_FIELDS
        if isinstance(scp_hard_caps, dict) and field_name in scp_hard_caps
    }
    mcp_hard_caps = _nested(config, ("ur5e", "rtde", "MCP"), {})
    UR5E_RTDE_INSERT_MCP_HARD_CAPS = {
        field_name: _optional_float(mcp_hard_caps, (field_name,))
        for field_name in INSERT_MG_HARD_CAP_FIELDS
        if isinstance(mcp_hard_caps, dict) and field_name in mcp_hard_caps
    }
    lcp_hard_caps = _nested(config, ("ur5e", "rtde", "LCP"), {})
    UR5E_RTDE_INSERT_LCP_HARD_CAPS = {
        field_name: _optional_float(lcp_hard_caps, (field_name,))
        for field_name in INSERT_MG_HARD_CAP_FIELDS
        if isinstance(lcp_hard_caps, dict) and field_name in lcp_hard_caps
    }
    UR5E_RTDE_INSERT_SOFT_FILTER_WINDOW_SEC = _optional_float(
        config,
        ("ur5e", "rtde", "insert_soft_filter_window_sec"),
    )
    UR5E_RTDE_INSERT_SOFT_OVERLOAD_HOLD_SEC = _optional_float(
        config,
        ("ur5e", "rtde", "insert_soft_overload_hold_sec"),
    )
    UR5E_RTDE_INSERT_RELIEF_UNLOAD_DWELL_SEC = _optional_float(
        config,
        ("ur5e", "rtde", "insert_relief_unload_dwell_sec"),
    )
    UR5E_RTDE_INSERT_RELIEF_CLEAR_DWELL_SEC = _optional_float(
        config,
        ("ur5e", "rtde", "insert_relief_clear_dwell_sec"),
    )
    UR5E_RTDE_INSERT_RELIEF_CLEAR_HYSTERESIS_RATIO = _optional_float(
        config,
        ("ur5e", "rtde", "insert_relief_clear_hysteresis_ratio"),
    )
    UR5E_RTDE_INSERT_RELIEF_TIMEOUT_SEC = _optional_float(
        config,
        ("ur5e", "rtde", "insert_relief_timeout_sec"),
    )
    UR5E_RTDE_INSERT_RELIEF_AXIAL_FORCE_RATIO = _optional_float(
        config,
        ("ur5e", "rtde", "insert_relief_axial_force_ratio"),
    )
    UR5E_RTDE_INSERT_RELIEF_REVERSE_FORCE_RATIO = _optional_float(
        config,
        ("ur5e", "rtde", "insert_relief_reverse_force_ratio"),
    )
    UR5E_RTDE_INSERT_RELIEF_RESUME_RAMP_SEC = _optional_float(
        config,
        ("ur5e", "rtde", "insert_relief_resume_ramp_sec"),
    )
    UR5E_RTDE_INSERT_RELIEF_SEARCH_FORCE_RATIO = _optional_float(
        config,
        ("ur5e", "rtde", "insert_relief_search_force_ratio"),
    )
    UR5E_RTDE_INSERT_RELIEF_SEARCH_SPEED_RATIO = _optional_float(
        config,
        ("ur5e", "rtde", "insert_relief_search_speed_ratio"),
    )
    UR5E_RTDE_INSERT_RELIEF_BACKOFF_STEP_M = _optional_float(
        config,
        ("ur5e", "rtde", "insert_relief_backoff_step_m"),
    )
    UR5E_RTDE_INSERT_MAX_RELIEF_RETREAT_M = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_relief_retreat_m"),
    )
    UR5E_RTDE_INSERT_RELIEF_STATIONARY_SPEED_M_S = _optional_float(
        config,
        ("ur5e", "rtde", "insert_relief_stationary_speed_m_s"),
    )
    UR5E_RTDE_INSERT_RELIEF_STATIONARY_ANGULAR_SPEED_RAD_S = _optional_float(
        config,
        ("ur5e", "rtde", "insert_relief_stationary_angular_speed_rad_s"),
    )
    relief_cycles = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_relief_cycles"),
    )
    UR5E_RTDE_INSERT_MAX_RELIEF_CYCLES = (
        int(relief_cycles)
        if relief_cycles is not None and relief_cycles.is_integer()
        else None
    )
    UR5E_RTDE_INSERT_MAX_TILT_TOLERANCE_RAD = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_tilt_tolerance_rad"),
    )
    UR5E_RTDE_INSERT_MAX_SEATED_DEPTH_TOLERANCE_M = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_seated_depth_tolerance_m"),
    )
    UR5E_RTDE_INSERT_MAX_SETTLE_TIME_SEC = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_settle_time_sec"),
    )
    UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_timeout_sec"),
    )
    UR5E_RTDE_INSERT_MAX_TRAVEL_M = _optional_float(
        config,
        ("ur5e", "rtde", "insert_max_travel_m"),
    )
    UR5E_RTDE_INSERT_START_POSITION_TOLERANCE_M = _optional_float(
        config,
        ("ur5e", "rtde", "insert_start_position_tolerance_m"),
    )
    UR5E_RTDE_INSERT_START_ORIENTATION_TOLERANCE_RAD = _optional_float(
        config,
        ("ur5e", "rtde", "insert_start_orientation_tolerance_rad"),
    )
    UR5E_RTDE_INSERT_DEMONSTRATION_MAX_DURATION_SEC = max(
        1.0,
        _float(
            config,
            ("ur5e", "rtde", "insert_demonstration_max_duration_sec"),
            UR5E_RTDE_INSERT_DEMONSTRATION_MAX_DURATION_SEC,
        ),
    )
    UR5E_RTDE_INSERT_DEMONSTRATION_BASELINE_SEC = max(
        0.25,
        _float(
            config,
            ("ur5e", "rtde", "insert_demonstration_baseline_sec"),
            UR5E_RTDE_INSERT_DEMONSTRATION_BASELINE_SEC,
        ),
    )
    UR5E_RTDE_INSERT_DEMONSTRATION_STATIONARY_SPEED_M_S = max(
        1e-5,
        _float(
            config,
            ("ur5e", "rtde", "insert_demonstration_stationary_speed_m_s"),
            UR5E_RTDE_INSERT_DEMONSTRATION_STATIONARY_SPEED_M_S,
        ),
    )
    UR5E_RTDE_INSERT_DEMONSTRATION_STATIONARY_ANGULAR_SPEED_RAD_S = max(
        1e-5,
        _float(
            config,
            (
                "ur5e",
                "rtde",
                "insert_demonstration_stationary_angular_speed_rad_s",
            ),
            UR5E_RTDE_INSERT_DEMONSTRATION_STATIONARY_ANGULAR_SPEED_RAD_S,
        ),
    )


UR5E_RTDE_CURRENT_HOLD_SEC = 0.25
UR5E_RTDE_MIN_POINT_SPACING_SEC = 0.10
UR5E_RTDE_MAX_JOINT_VEL_RAD_S = 0.324
UR5E_RTDE_MAX_JOINT_ACCEL_RAD_S2 = 0.486
UR5E_RTDE_MAX_JOINT_JERK_RAD_S3 = 2.025
UR5E_RTDE_SHOULDER_PAN_EXTRA_SCALE = 1.0
UR5E_RTDE_START_TOLERANCE_RAD = 0.15
UR5E_RTDE_GOAL_TOLERANCE_RAD = 0.025
UR5E_RTDE_STATIONARY_MAX_JOINT_VEL_RAD_S = 0.01
UR5E_RTDE_STATIONARY_HOLD_SEC = 0.25
UR5E_RTDE_JOINT_JOG_TOLERANCE_RAD = math.radians(0.20)
UR5E_RTDE_JOINT_JOG_MAX_DELTA_RAD = math.radians(30.0)
UR5E_RTDE_MOVEJ_SPEED_RAD_S = 0.486
UR5E_RTDE_MOVEJ_ACCEL_RAD_S2 = 0.81
UR5E_RTDE_CONTROL_PROGRAM_START_TIMEOUT_SEC = 2.0
UR5E_RTDE_MOTION_START_TIMEOUT_SEC = 2.0
UR5E_RTDE_MOTION_START_DELTA_RAD = 0.001
UR5E_RTDE_FEEDBACK_RECONNECT_AFTER_SEC = 0.5
UR5E_RTDE_FEEDBACK_RECOVERY_TIMEOUT_SEC = 2.0
UR5E_RTDE_FEEDBACK_RECONNECT_RETRY_SEC = 1.0
UR5E_RTDE_FREQUENCY_HZ = 125.0
UR5E_RTDE_RECEIVE_VARIABLES = (
    "timestamp",
    "actual_q",
    "actual_qd",
    "actual_TCP_pose",
    "actual_TCP_force",
    "actual_TCP_speed",
)
UR5E_RTDE_STOPPED_AWAY_HOLD_SEC = 0.5
UR5E_RTDE_INTERMEDIATE_BLEND_RAD = 0.005
UR5E_RTDE_STOP_ACCEL_RAD_S2 = 0.50
UR5E_RTDE_FEEDBACK_STALE_SEC = 2.0
UR5E_RTDE_ALLOWED_EXECUTION_DURATION_SCALING = 8.0
UR5E_RTDE_RESULT_MARGIN_SEC = 20.0
UR5E_RTDE_CARTESIAN_SPEED_M_S = 0.08
UR5E_RTDE_CARTESIAN_MAX_SPEED_M_S = 0.10
UR5E_RTDE_CARTESIAN_ACCEL_M_S2 = 0.10
UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M = 0.002
UR5E_RTDE_CARTESIAN_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)
UR5E_RTDE_CARTESIAN_WORLD_BASE: tuple[
    float,
    float,
    float,
    float,
    float,
    float,
] | None = None
UR5E_RTDE_CARTESIAN_WORLD_BASE_CONFIG_ERROR = (
    "ur5e.rtde.cartesian_world_base has not been loaded"
)
UR5E_RTDE_CARTESIAN_FRAME_POSITION_TOLERANCE_M = 0.005
UR5E_RTDE_CARTESIAN_FRAME_ORIENTATION_TOLERANCE_RAD = math.radians(3.0)
UR5E_RTDE_CARTESIAN_JOG_ORIENTATION_DRIFT_RAD = math.radians(1.0)
UR5E_RTDE_CARTESIAN_JOG_MAX_STEP_M = 0.10
UR5E_RTDE_CARTESIAN_JOG_MIN_WATCHDOG_SEC = 0.10
UR5E_RTDE_CARTESIAN_JOG_MAX_WATCHDOG_SEC = 0.50
UR5E_RTDE_CARTESIAN_WORKSPACE_BOUNDS = {
    "x_min_m": -0.7,
    "x_max_m": 0.7,
    "y_min_m": -0.35,
    "y_max_m": 1.1,
    "z_min_m": 0.85,
    "z_max_m": 1.6,
}
UR5E_RTDE_CARTESIAN_REACH_ORIGIN = (0.0, 0.5, 1.021)
UR5E_RTDE_CARTESIAN_REACH_RADIUS_M = 0.7
UR5E_RTDE_INSERT_MAX_CONTACT_SPEED_M_S: float | None = None
UR5E_RTDE_INSERT_MAX_CONTACT_FORCE_DELTA_N: float | None = None
UR5E_RTDE_INSERT_MAX_ENGAGEMENT_PROGRESS_M: float | None = None
UR5E_RTDE_INSERT_MAX_INSERTION_FORCE_N: float | None = None
UR5E_RTDE_INSERT_MAX_SPIRAL_RADIUS_M: float | None = None
UR5E_RTDE_INSERT_MAX_SPIRAL_PITCH_M: float | None = None
UR5E_RTDE_INSERT_MAX_SPIRAL_SPEED_M_S: float | None = None
UR5E_RTDE_INSERT_MAX_SPIRAL_ACCELERATION_M_S2: float | None = None
UR5E_RTDE_INSERT_MAX_AXIAL_FORCE_N: float | None = None
UR5E_RTDE_INSERT_MAX_LATERAL_FORCE_N: float | None = None
UR5E_RTDE_INSERT_MAX_TORQUE_NM: float | None = None
UR5E_RTDE_INSERT_MAX_TOOL_FLANGE_TORQUE_NM: float | None = None
UR5E_RTDE_INSERT_MG_HARD_CAPS: dict[str, float | None] = {}
UR5E_RTDE_INSERT_SG_HARD_CAPS: dict[str, float | None] = {}
UR5E_RTDE_INSERT_LG_HARD_CAPS: dict[str, float | None] = {}
UR5E_RTDE_INSERT_SCP_HARD_CAPS: dict[str, float | None] = {}
UR5E_RTDE_INSERT_MCP_HARD_CAPS: dict[str, float | None] = {}
UR5E_RTDE_INSERT_LCP_HARD_CAPS: dict[str, float | None] = {}
UR5E_RTDE_INSERT_SOFT_FILTER_WINDOW_SEC: float | None = None
UR5E_RTDE_INSERT_SOFT_OVERLOAD_HOLD_SEC: float | None = None
UR5E_RTDE_INSERT_RELIEF_UNLOAD_DWELL_SEC: float | None = None
UR5E_RTDE_INSERT_RELIEF_CLEAR_DWELL_SEC: float | None = None
UR5E_RTDE_INSERT_RELIEF_CLEAR_HYSTERESIS_RATIO: float | None = None
UR5E_RTDE_INSERT_RELIEF_TIMEOUT_SEC: float | None = None
UR5E_RTDE_INSERT_RELIEF_AXIAL_FORCE_RATIO: float | None = None
UR5E_RTDE_INSERT_RELIEF_REVERSE_FORCE_RATIO: float | None = None
UR5E_RTDE_INSERT_RELIEF_RESUME_RAMP_SEC: float | None = None
UR5E_RTDE_INSERT_RELIEF_SEARCH_FORCE_RATIO: float | None = None
UR5E_RTDE_INSERT_RELIEF_SEARCH_SPEED_RATIO: float | None = None
UR5E_RTDE_INSERT_RELIEF_BACKOFF_STEP_M: float | None = None
UR5E_RTDE_INSERT_MAX_RELIEF_RETREAT_M: float | None = None
UR5E_RTDE_INSERT_RELIEF_STATIONARY_SPEED_M_S: float | None = None
UR5E_RTDE_INSERT_RELIEF_STATIONARY_ANGULAR_SPEED_RAD_S: float | None = None
UR5E_RTDE_INSERT_MAX_RELIEF_CYCLES: int | None = None
UR5E_RTDE_INSERT_MAX_TILT_TOLERANCE_RAD: float | None = None
UR5E_RTDE_INSERT_MAX_SEATED_DEPTH_TOLERANCE_M: float | None = None
UR5E_RTDE_INSERT_MAX_SETTLE_TIME_SEC: float | None = None
UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC: float | None = None
UR5E_RTDE_INSERT_MAX_TRAVEL_M: float | None = None
UR5E_RTDE_INSERT_START_POSITION_TOLERANCE_M: float | None = None
UR5E_RTDE_INSERT_START_ORIENTATION_TOLERANCE_RAD: float | None = None
UR5E_RTDE_INSERT_DEMONSTRATION_MAX_DURATION_SEC = 300.0
UR5E_RTDE_INSERT_DEMONSTRATION_BASELINE_SEC = 1.0
UR5E_RTDE_INSERT_DEMONSTRATION_STATIONARY_SPEED_M_S = 0.001
UR5E_RTDE_INSERT_DEMONSTRATION_STATIONARY_ANGULAR_SPEED_RAD_S = 0.02
UR5E_RTDE_INSERT_MG_LEARNED_AXIAL_LIMIT_SCALE = 3.00
UR5E_RTDE_INSERT_MG_DEPTH_AXIAL_LIMIT_SCALE = 3.00
UR5E_RTDE_INSERT_MG_FORCE_COMMAND_SCALE = 4.00
UR5E_RTDE_INSERT_MG_EXECUTION_TIMEOUT_SEC = 60.0

_apply_hardware_arms_config(DEFAULT_CONFIG_FILE)

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
RigidTransform = tuple[Vector3, Quaternion]


class _InsertCanceled(RuntimeError):
    pass


class _InsertSearchExhausted(RuntimeError):
    pass


class _InsertForceLimit(RuntimeError):
    pass


class _InsertSoftOverload(RuntimeError):
    pass


def _bounded_insert_value(
    field_name: str,
    value: Any,
    hard_cap: float | None,
    *,
    allow_zero: bool = False,
) -> float:
    parsed = float(value)
    lower_bound_ok = parsed >= 0.0 if allow_zero else parsed > 0.0
    if not math.isfinite(parsed) or not lower_bound_ok:
        interval = "[0" if allow_zero else "(0"
        raise ValueError(f"{field_name} must be finite and within {interval}, hard cap]")
    if hard_cap is None or not math.isfinite(hard_cap):
        raise RuntimeError(f"{field_name} hard cap is not configured")
    if parsed > hard_cap:
        raise ValueError(f"{field_name}={parsed:.9g} exceeds configured hard cap {hard_cap:.9g}")
    return parsed


def _insert_hard_caps(part_name: str = "") -> dict[str, float | int | None]:
    """Return shared caps plus values for the selected exact part name."""
    caps: dict[str, float | int | None] = {
        "insert_max_contact_speed_m_s": UR5E_RTDE_INSERT_MAX_CONTACT_SPEED_M_S,
        "insert_max_contact_force_delta_n": (UR5E_RTDE_INSERT_MAX_CONTACT_FORCE_DELTA_N),
        "insert_max_engagement_progress_m": (UR5E_RTDE_INSERT_MAX_ENGAGEMENT_PROGRESS_M),
        "insert_max_insertion_force_n": UR5E_RTDE_INSERT_MAX_INSERTION_FORCE_N,
        "insert_max_spiral_radius_m": UR5E_RTDE_INSERT_MAX_SPIRAL_RADIUS_M,
        "insert_max_spiral_pitch_m": UR5E_RTDE_INSERT_MAX_SPIRAL_PITCH_M,
        "insert_max_spiral_speed_m_s": UR5E_RTDE_INSERT_MAX_SPIRAL_SPEED_M_S,
        "insert_max_spiral_acceleration_m_s2": (UR5E_RTDE_INSERT_MAX_SPIRAL_ACCELERATION_M_S2),
        "insert_max_axial_force_n": UR5E_RTDE_INSERT_MAX_AXIAL_FORCE_N,
        "insert_max_lateral_force_n": UR5E_RTDE_INSERT_MAX_LATERAL_FORCE_N,
        "insert_max_torque_nm": UR5E_RTDE_INSERT_MAX_TORQUE_NM,
        "insert_max_tool_flange_torque_nm": (
            UR5E_RTDE_INSERT_MAX_TOOL_FLANGE_TORQUE_NM
        ),
        "insert_soft_filter_window_sec": UR5E_RTDE_INSERT_SOFT_FILTER_WINDOW_SEC,
        "insert_soft_overload_hold_sec": UR5E_RTDE_INSERT_SOFT_OVERLOAD_HOLD_SEC,
        "insert_relief_unload_dwell_sec": UR5E_RTDE_INSERT_RELIEF_UNLOAD_DWELL_SEC,
        "insert_relief_clear_dwell_sec": UR5E_RTDE_INSERT_RELIEF_CLEAR_DWELL_SEC,
        "insert_relief_clear_hysteresis_ratio": (
            UR5E_RTDE_INSERT_RELIEF_CLEAR_HYSTERESIS_RATIO
        ),
        "insert_relief_timeout_sec": UR5E_RTDE_INSERT_RELIEF_TIMEOUT_SEC,
        "insert_relief_axial_force_ratio": UR5E_RTDE_INSERT_RELIEF_AXIAL_FORCE_RATIO,
        "insert_relief_reverse_force_ratio": (
            UR5E_RTDE_INSERT_RELIEF_REVERSE_FORCE_RATIO
        ),
        "insert_relief_resume_ramp_sec": UR5E_RTDE_INSERT_RELIEF_RESUME_RAMP_SEC,
        "insert_relief_search_force_ratio": (
            UR5E_RTDE_INSERT_RELIEF_SEARCH_FORCE_RATIO
        ),
        "insert_relief_search_speed_ratio": (
            UR5E_RTDE_INSERT_RELIEF_SEARCH_SPEED_RATIO
        ),
        "insert_relief_backoff_step_m": UR5E_RTDE_INSERT_RELIEF_BACKOFF_STEP_M,
        "insert_max_relief_retreat_m": UR5E_RTDE_INSERT_MAX_RELIEF_RETREAT_M,
        "insert_relief_stationary_speed_m_s": (
            UR5E_RTDE_INSERT_RELIEF_STATIONARY_SPEED_M_S
        ),
        "insert_relief_stationary_angular_speed_rad_s": (
            UR5E_RTDE_INSERT_RELIEF_STATIONARY_ANGULAR_SPEED_RAD_S
        ),
        "insert_max_relief_cycles": UR5E_RTDE_INSERT_MAX_RELIEF_CYCLES,
        "insert_max_tilt_tolerance_rad": (UR5E_RTDE_INSERT_MAX_TILT_TOLERANCE_RAD),
        "insert_max_seated_depth_tolerance_m": (UR5E_RTDE_INSERT_MAX_SEATED_DEPTH_TOLERANCE_M),
        "insert_max_settle_time_sec": UR5E_RTDE_INSERT_MAX_SETTLE_TIME_SEC,
        "insert_max_timeout_sec": UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC,
        "insert_max_travel_m": UR5E_RTDE_INSERT_MAX_TRAVEL_M,
        "insert_start_position_tolerance_m": (UR5E_RTDE_INSERT_START_POSITION_TOLERANCE_M),
        "insert_start_orientation_tolerance_rad": (
            UR5E_RTDE_INSERT_START_ORIENTATION_TOLERANCE_RAD
        ),
    }
    if part_name == "SG":
        caps.update(UR5E_RTDE_INSERT_SG_HARD_CAPS)
    elif part_name == "MG":
        caps.update(UR5E_RTDE_INSERT_MG_HARD_CAPS)
    elif part_name == "LG":
        caps.update(UR5E_RTDE_INSERT_LG_HARD_CAPS)
    elif part_name == "SCP":
        caps.update(UR5E_RTDE_INSERT_SCP_HARD_CAPS)
    elif part_name == "MCP":
        caps.update(UR5E_RTDE_INSERT_MCP_HARD_CAPS)
    elif part_name == "LCP":
        caps.update(UR5E_RTDE_INSERT_LCP_HARD_CAPS)
    return caps


def _insert_hard_caps_sha256(caps: dict[str, float | int | None]) -> str:
    """Hash the exact finite hard-cap mapping selected for one insert goal."""
    payload = {
        key: float(value)
        for key, value in sorted(caps.items())
        if value is not None
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _insert_axial_soft_limit_scales(part_name: str) -> tuple[float, float]:
    if part_name == "MG":
        return (
            UR5E_RTDE_INSERT_MG_LEARNED_AXIAL_LIMIT_SCALE,
            UR5E_RTDE_INSERT_MG_DEPTH_AXIAL_LIMIT_SCALE,
        )
    return 1.0, 1.0


def _insert_force_command_scale(part_name: str) -> float:
    if part_name == "MG":
        return UR5E_RTDE_INSERT_MG_FORCE_COMMAND_SCALE
    return 1.0


def _insert_depth_completion_enabled(part_name: str) -> bool:
    return part_name == "MG"


def _insert_automatic_withdrawal_enabled(part_name: str) -> bool:
    return part_name != "MG"


def _insert_axis_only_target_enabled(part_name: str) -> bool:
    return part_name == "MG"


def _insert_soft_recovery_enabled(part_name: str) -> bool:
    return part_name != "MG"


def _insert_full_hard_ceiling_enabled(part_name: str) -> bool:
    return part_name == "MG"


def _insert_execution_timeout_sec(part_name: str, requested_timeout_sec: float) -> float:
    if part_name == "MG":
        return max(requested_timeout_sec, UR5E_RTDE_INSERT_MG_EXECUTION_TIMEOUT_SEC)
    return requested_timeout_sec


def _insert_hard_cap_error(part_name: str = "") -> str | None:
    caps = _insert_hard_caps(part_name)
    invalid = [
        name
        for name, value in caps.items()
        if value is None
        or not math.isfinite(value)
        or (
            value < 0.0
            if name == "insert_max_spiral_radius_m"
            else value <= 0.0
        )
    ]
    if invalid:
        return "Insertion hard caps are missing or invalid: " + ", ".join(invalid)
    ratios = (
        "insert_relief_axial_force_ratio",
        "insert_relief_reverse_force_ratio",
        "insert_relief_search_force_ratio",
        "insert_relief_search_speed_ratio",
        "insert_relief_clear_hysteresis_ratio",
    )
    invalid_ratios = [name for name in ratios if float(caps[name]) >= 1.0]
    if invalid_ratios:
        return "Insertion relief ratios must be less than 1: " + ", ".join(
            invalid_ratios
        )
    if int(caps["insert_max_relief_cycles"]) != 3:
        return "insert_max_relief_cycles must equal the protected policy value 3"
    advanced_recovery_fields = (
        "insert_max_contact_search_radius_m",
        "insert_max_disengagement_cycles",
        "insert_search_peck_retreat_m",
        "insert_search_peck_interval_sec",
    )
    configured_advanced_recovery_fields = tuple(
        name for name in advanced_recovery_fields if name in caps
    )
    if configured_advanced_recovery_fields:
        missing_advanced_recovery_fields = tuple(
            name for name in advanced_recovery_fields if name not in caps
        )
        if missing_advanced_recovery_fields:
            return (
                f"{part_name or 'Insertion'} protected advanced recovery policy is "
                "incomplete: "
                + ", ".join(missing_advanced_recovery_fields)
            )
        disengagement_cycles = float(caps["insert_max_disengagement_cycles"])
        if not disengagement_cycles.is_integer():
            return "insert_max_disengagement_cycles must be an integer"
        if float(caps["insert_max_contact_search_radius_m"]) < float(
            caps["insert_max_spiral_radius_m"]
        ):
            return (
                "insert_max_contact_search_radius_m is below "
                "insert_max_spiral_radius_m"
            )
        if float(caps["insert_search_peck_retreat_m"]) >= float(
            caps["insert_max_travel_m"]
        ):
            return "insert_search_peck_retreat_m must be below insert_max_travel_m"
        if float(caps["insert_search_peck_interval_sec"]) >= float(
            caps["insert_max_timeout_sec"]
        ):
            return "insert_search_peck_interval_sec must be below insert_max_timeout_sec"
    if float(caps["insert_relief_backoff_step_m"]) > float(
        caps["insert_max_relief_retreat_m"]
    ):
        return "insert_relief_backoff_step_m exceeds insert_max_relief_retreat_m"
    if float(caps["insert_relief_unload_dwell_sec"]) >= float(
        caps["insert_relief_timeout_sec"]
    ):
        return "insert_relief_unload_dwell_sec must be below insert_relief_timeout_sec"
    if float(caps["insert_relief_clear_dwell_sec"]) >= float(
        caps["insert_relief_timeout_sec"]
    ):
        return "insert_relief_clear_dwell_sec must be below insert_relief_timeout_sec"
    if float(caps["insert_soft_filter_window_sec"]) > float(
        caps["insert_soft_overload_hold_sec"]
    ):
        return "insert_soft_filter_window_sec exceeds insert_soft_overload_hold_sec"
    return None


def _validated_force_depth_profile(
    request: Any,
    *,
    hard_caps: dict[str, float | int | None],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Validate one version-3 force-depth profile against selected hard caps."""
    raw_series = (
        list(request.force_depth_fraction),
        list(request.force_depth_axial_upper_n),
        list(request.force_depth_lateral_upper_n),
        list(request.force_depth_torque_upper_nm),
    )
    if any(len(series) != INSERT_FORCE_DEPTH_PROFILE_POINTS for series in raw_series):
        raise ValueError(
            "force_depth_profile requires exactly "
            f"{INSERT_FORCE_DEPTH_PROFILE_POINTS} synchronized points"
        )
    try:
        fractions, axial, lateral, torque = (
            [float(value) for value in series] for series in raw_series
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("force_depth_profile contains a nonnumeric value") from exc
    if not all(
        math.isfinite(value)
        for series in (fractions, axial, lateral, torque)
        for value in series
    ):
        raise ValueError("force_depth_profile contains a nonfinite value")
    if abs(fractions[0]) > 1e-12 or abs(fractions[-1] - 1.0) > 1e-12:
        raise ValueError("force_depth_fraction must start at 0 and end at 1")
    if any(
        right <= left
        for left, right in zip(fractions, fractions[1:])
    ):
        raise ValueError("force_depth_fraction must increase strictly")
    for field_name, values, cap_name in (
        ("force_depth_axial_upper_n", axial, "insert_max_axial_force_n"),
        ("force_depth_lateral_upper_n", lateral, "insert_max_lateral_force_n"),
        ("force_depth_torque_upper_nm", torque, "insert_max_torque_nm"),
    ):
        hard_cap = float(hard_caps[cap_name] or math.nan)
        if any(value <= 0.0 or value >= hard_cap for value in values):
            raise ValueError(
                f"{field_name} values must be positive and strictly below "
                f"{cap_name}"
            )
    return fractions, axial, lateral, torque


def _force_depth_upper(
    fractions: list[float],
    values: list[float],
    depth_fraction: float,
) -> float:
    """Linearly interpolate one protected force-depth upper envelope."""
    selected = min(1.0, max(0.0, float(depth_fraction)))
    for index in range(1, len(fractions)):
        if selected <= fractions[index]:
            left_fraction = fractions[index - 1]
            right_fraction = fractions[index]
            span = right_fraction - left_fraction
            ratio = (selected - left_fraction) / span
            return values[index - 1] + ratio * (values[index] - values[index - 1])
    return values[-1]


def _vector_dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _vector_norm(value: Vector3) -> float:
    return math.sqrt(_vector_dot(value, value))


def _normalize_vector(value: Vector3) -> Vector3:
    norm = _vector_norm(value)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("insertion_axis_world must contain a nonzero finite vector")
    return tuple(component / norm for component in value)  # type: ignore[return-value]


def _vector_cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _insertion_basis(axis: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    """Return a deterministic right-handed frame whose z-axis is insertion_axis_world."""
    z_axis = _normalize_vector(axis)
    reference = (1.0, 0.0, 0.0) if abs(z_axis[0]) < 0.9 else (0.0, 1.0, 0.0)
    y_axis = _normalize_vector(_vector_cross(z_axis, reference))
    x_axis = _normalize_vector(_vector_cross(y_axis, z_axis))
    return x_axis, y_axis, z_axis


def _quaternion_from_basis(
    x_axis: Vector3,
    y_axis: Vector3,
    z_axis: Vector3,
) -> Quaternion:
    """Convert the column vectors of a rotation matrix to a quaternion."""
    m00, m10, m20 = x_axis
    m01, m11, m21 = y_axis
    m02, m12, m22 = z_axis
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return _normalize_quaternion(
            ((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale)
        )
    if m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return _normalize_quaternion(
            (0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale)
        )
    if m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return _normalize_quaternion(
            ((m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale)
        )
    scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return _normalize_quaternion(
        ((m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale)
    )


def _insertion_pose_metrics(
    actual_world_tool0: RigidTransform,
    expected_start_world_tool0: RigidTransform,
    target_world_tool0: RigidTransform,
    insertion_axis_world: Vector3,
) -> tuple[float, float, float, float]:
    """Return depth, absolute depth error, lateral offset, and tilt error."""
    actual_translation, _actual_rotation = actual_world_tool0
    start_translation, _start_rotation = expected_start_world_tool0
    displacement = tuple(actual_translation[index] - start_translation[index] for index in range(3))
    depth = _vector_dot(displacement, insertion_axis_world)
    target_displacement = tuple(
        target_world_tool0[0][index] - start_translation[index] for index in range(3)
    )
    target_depth = _vector_dot(target_displacement, insertion_axis_world)
    lateral = tuple(displacement[index] - depth * insertion_axis_world[index] for index in range(3))
    _unused_position_error, tilt_error = _pose_errors(
        actual_world_tool0,
        target_world_tool0,
    )
    return depth, abs(target_depth - depth), _vector_norm(lateral), tilt_error


def _insertion_force_metrics(
    actual_tcp_force: list[float],
    force_bias: list[float],
    insertion_axis_base: Vector3,
    tool0_tcp_offset_base: Vector3,
) -> tuple[float, float, float, float, float, list[float]]:
    """Return raw/compressive axial force and active-TCP/tool-flange evidence."""
    corrected = [
        float(value) - float(bias) for value, bias in zip(actual_tcp_force, force_bias, strict=True)
    ]
    force = (corrected[0], corrected[1], corrected[2])
    axial_signed = _vector_dot(force, insertion_axis_base)
    lateral = tuple(force[index] - axial_signed * insertion_axis_base[index] for index in range(3))
    tool_flange_torque = (corrected[3], corrected[4], corrected[5])
    offset_moment = _vector_cross(tool0_tcp_offset_base, force)
    active_tcp_torque = tuple(
        tool_flange_torque[index] - offset_moment[index] for index in range(3)
    )
    return (
        abs(axial_signed),
        max(0.0, -axial_signed),
        _vector_norm(lateral),
        _vector_norm(active_tcp_torque),
        _vector_norm(tool_flange_torque),
        corrected,
    )


def _protected_relief_retreat_m(
    total_relief_backoff_m: float,
    relief_backoff_m: float,
    *,
    relief_backoff_committed: bool,
) -> float:
    """Return protected cumulative retreat without double-counting a committed cycle."""
    if relief_backoff_committed:
        return float(total_relief_backoff_m)
    return float(total_relief_backoff_m) + float(relief_backoff_m)


def _normalize_quaternion(value: Quaternion) -> Quaternion:
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion norm is zero or non-finite")
    return tuple(component / norm for component in value)  # type: ignore[return-value]


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    half_roll = roll / 2.0
    half_pitch = pitch / 2.0
    half_yaw = yaw / 2.0
    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)
    return _normalize_quaternion(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def _quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return _normalize_quaternion(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def _quaternion_conjugate(value: Quaternion) -> Quaternion:
    x, y, z, w = _normalize_quaternion(value)
    return (-x, -y, -z, w)


def _rotate_vector(value: Quaternion, vector: Vector3) -> Vector3:
    x, y, z, w = _normalize_quaternion(value)
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _compose_transform(left: RigidTransform, right: RigidTransform) -> RigidTransform:
    left_translation, left_rotation = left
    right_translation, right_rotation = right
    rotated = _rotate_vector(left_rotation, right_translation)
    return (
        tuple(left_translation[index] + rotated[index] for index in range(3)),  # type: ignore[arg-type]
        _quaternion_multiply(left_rotation, right_rotation),
    )


def _inverse_transform(value: RigidTransform) -> RigidTransform:
    translation, rotation = value
    inverse_rotation = _quaternion_conjugate(rotation)
    inverse_translation = _rotate_vector(
        inverse_rotation,
        (-translation[0], -translation[1], -translation[2]),
    )
    return inverse_translation, inverse_rotation


def _transform_from_message(message: Any) -> RigidTransform:
    transform = getattr(message, "transform", message)
    translation = transform.translation
    rotation = transform.rotation
    return (
        (float(translation.x), float(translation.y), float(translation.z)),
        _normalize_quaternion(
            (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w))
        ),
    )


def _transform_from_pose_stamped(message: PoseStamped) -> RigidTransform:
    position = message.pose.position
    orientation = message.pose.orientation
    return (
        (float(position.x), float(position.y), float(position.z)),
        _normalize_quaternion(
            (
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
        ),
    )


def _quaternion_from_rotvec(rotation_vector: Vector3) -> Quaternion:
    angle = math.sqrt(sum(component * component for component in rotation_vector))
    if angle <= 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    half = angle / 2.0
    scale = math.sin(half) / angle
    return _normalize_quaternion(
        (
            rotation_vector[0] * scale,
            rotation_vector[1] * scale,
            rotation_vector[2] * scale,
            math.cos(half),
        )
    )


def _rotvec_from_quaternion(value: Quaternion) -> Vector3:
    x, y, z, w = _normalize_quaternion(value)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm <= 1e-12:
        return 0.0, 0.0, 0.0
    angle = 2.0 * math.atan2(vector_norm, max(-1.0, min(1.0, w)))
    scale = angle / vector_norm
    return x * scale, y * scale, z * scale


def _transform_from_rtde_pose(values: list[float]) -> RigidTransform:
    if len(values) < 6 or not all(math.isfinite(float(value)) for value in values[:6]):
        raise ValueError("RTDE pose must contain six finite values")
    return (
        (float(values[0]), float(values[1]), float(values[2])),
        _quaternion_from_rotvec((float(values[3]), float(values[4]), float(values[5]))),
    )


def _rtde_pose_from_transform(value: RigidTransform) -> list[float]:
    translation, rotation = value
    rotation_vector = _rotvec_from_quaternion(rotation)
    return [*translation, *rotation_vector]


def _pose_errors(actual: RigidTransform, target: RigidTransform) -> tuple[float, float]:
    actual_translation, actual_rotation = actual
    target_translation, target_rotation = target
    position_error = math.sqrt(
        sum((actual_translation[index] - target_translation[index]) ** 2 for index in range(3))
    )
    dot = abs(sum(a * b for a, b in zip(actual_rotation, target_rotation, strict=True)))
    orientation_error = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
    return position_error, orientation_error


def _configured_cartesian_world_base() -> RigidTransform:
    values = UR5E_RTDE_CARTESIAN_WORLD_BASE
    if values is None:
        detail = UR5E_RTDE_CARTESIAN_WORLD_BASE_CONFIG_ERROR or (
            "ur5e.rtde.cartesian_world_base is missing or invalid"
        )
        raise RuntimeError(
            "protected ur5e.rtde.cartesian_world_base is unavailable: " + detail
        )
    x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad = values
    return (
        (x_m, y_m, z_m),
        _quaternion_from_rpy(roll_rad, pitch_rad, yaw_rad),
    )


def _transform_status_payload(value: RigidTransform | None) -> dict[str, float] | None:
    if value is None:
        return None
    translation, rotation = value
    return {
        "x": float(translation[0]),
        "y": float(translation[1]),
        "z": float(translation[2]),
        "qx": float(rotation[0]),
        "qy": float(rotation[1]),
        "qz": float(rotation[2]),
        "qw": float(rotation[3]),
    }


def _workspace_error(target_world_tool0: RigidTransform) -> str | None:
    translation, _rotation = target_world_tool0
    x, y, z = translation
    bounds = UR5E_RTDE_CARTESIAN_WORKSPACE_BOUNDS
    for axis, value in (("x", x), ("y", y), ("z", z)):
        minimum = float(bounds[f"{axis}_min_m"])
        maximum = float(bounds[f"{axis}_max_m"])
        if not minimum <= value <= maximum:
            return (
                f"world -> tool0 {axis}={value:.6f} m is outside [{minimum:.6f}, {maximum:.6f}] m"
            )
    origin_x, origin_y, _origin_z = UR5E_RTDE_CARTESIAN_REACH_ORIGIN
    xy_radius = math.hypot(x - origin_x, y - origin_y)
    if xy_radius > UR5E_RTDE_CARTESIAN_REACH_RADIUS_M:
        return (
            f"world -> tool0 XY radius {xy_radius:.6f} m from "
            f"configured origin ({origin_x:.6f}, {origin_y:.6f}) exceeds "
            f"{UR5E_RTDE_CARTESIAN_REACH_RADIUS_M:.6f} m"
        )
    return None


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _duration_seconds(duration: Any) -> float:
    return float(getattr(duration, "sec", 0) or 0) + (
        float(getattr(duration, "nanosec", 0) or 0) / 1e9
    )


def _set_duration(duration: Any, seconds: float) -> None:
    value = max(0.0, float(seconds))
    whole = int(value)
    duration.sec = whole
    duration.nanosec = int(round((value - whole) * 1_000_000_000))
    if duration.nanosec >= 1_000_000_000:
        duration.sec += 1
        duration.nanosec -= 1_000_000_000


def _point_seconds(point: Any) -> float:
    return _duration_seconds(point.time_from_start)


def _trajectory_result_timeout_sec(
    requested_final_time_sec: float,
    guarded_final_time_sec: float,
) -> float:
    """Return a server deadline that cannot precede MoveIt's configured allowance."""
    allowed_execution_sec = (
        max(0.0, float(requested_final_time_sec)) * UR5E_RTDE_ALLOWED_EXECUTION_DURATION_SCALING
    )
    guarded_execution_sec = max(0.0, float(guarded_final_time_sec))
    return max(
        5.0,
        max(allowed_execution_sec, guarded_execution_sec) + UR5E_RTDE_RESULT_MARGIN_SEC,
    )


def _positions_by_joint(joint_names: list[str], positions: list[float]) -> dict[str, float]:
    return {str(name): float(value) for name, value in zip(joint_names, positions)}


def _max_named_delta(
    joint_names: list[str],
    target_positions: list[float],
    current_positions: dict[str, float],
) -> tuple[float, str]:
    max_delta = 0.0
    max_joint = ""
    for joint, position in zip(joint_names, target_positions):
        if joint not in current_positions:
            continue
        delta = abs(float(position) - float(current_positions[joint]))
        if delta > max_delta:
            max_delta = delta
            max_joint = str(joint)
    return max_delta, max_joint


def _min_point_spacing(points: list[JointTrajectoryPoint]) -> float:
    if not points:
        return 0.0
    times = [_point_seconds(point) for point in points]
    spacings = [times[0]]
    spacings.extend(max(0.0, current - previous) for previous, current in zip(times, times[1:]))
    return min(spacings) if spacings else 0.0


def _max_segment_velocity(
    joint_names: list[str],
    points: list[JointTrajectoryPoint],
    *,
    current_positions: dict[str, float] | None = None,
) -> tuple[float, str]:
    previous_time = 0.0
    previous_positions: dict[str, float] | None = None
    if current_positions:
        previous_positions = {
            name: float(current_positions[name])
            for name in joint_names
            if name in current_positions
        }

    max_velocity = 0.0
    max_joint = ""
    for point in points:
        current_time = _point_seconds(point)
        positions = _positions_by_joint(joint_names, list(point.positions))
        if previous_positions is not None:
            dt = current_time - previous_time
            for joint in joint_names:
                if joint not in previous_positions or joint not in positions:
                    continue
                if dt <= 0.0:
                    velocity = math.inf
                else:
                    velocity = abs(float(positions[joint]) - float(previous_positions[joint])) / dt
                if velocity > max_velocity:
                    max_velocity = velocity
                    max_joint = str(joint)
        previous_positions = positions
        previous_time = current_time
    return max_velocity, max_joint


def _segment_velocities(
    joint_names: list[str],
    points: list[JointTrajectoryPoint],
    *,
    current_positions: dict[str, float] | None = None,
) -> list[tuple[float, float, dict[str, float]]]:
    previous_time = 0.0
    previous_positions: dict[str, float] | None = None
    if current_positions:
        previous_positions = {
            name: float(current_positions[name])
            for name in joint_names
            if name in current_positions
        }

    segments: list[tuple[float, float, dict[str, float]]] = []
    for point in points:
        current_time = _point_seconds(point)
        positions = _positions_by_joint(joint_names, list(point.positions))
        if previous_positions is not None:
            dt = current_time - previous_time
            velocities: dict[str, float] = {}
            for joint in joint_names:
                if joint not in previous_positions or joint not in positions:
                    continue
                if dt <= 0.0:
                    velocities[joint] = math.inf
                else:
                    velocities[joint] = (
                        float(positions[joint]) - float(previous_positions[joint])
                    ) / dt
            segments.append((current_time, dt, velocities))
        previous_positions = positions
        previous_time = current_time
    return segments


def _max_segment_acceleration(
    joint_names: list[str],
    points: list[JointTrajectoryPoint],
    *,
    current_positions: dict[str, float] | None = None,
) -> tuple[float, str]:
    previous_velocities: dict[str, float] | None = None
    max_acceleration = 0.0
    max_joint = ""
    for _time_sec, dt, velocities in _segment_velocities(
        joint_names,
        points,
        current_positions=current_positions,
    ):
        if previous_velocities is not None:
            for joint in joint_names:
                if joint not in previous_velocities or joint not in velocities:
                    continue
                invalid_segment = (
                    dt <= 0.0
                    or math.isinf(previous_velocities[joint])
                    or math.isinf(velocities[joint])
                )
                if invalid_segment:
                    acceleration = math.inf
                else:
                    acceleration = (
                        abs(float(velocities[joint]) - float(previous_velocities[joint])) / dt
                    )
                if acceleration > max_acceleration:
                    max_acceleration = acceleration
                    max_joint = str(joint)
        previous_velocities = velocities
    return max_acceleration, max_joint


def _max_segment_jerk(
    joint_names: list[str],
    points: list[JointTrajectoryPoint],
    *,
    current_positions: dict[str, float] | None = None,
) -> tuple[float, str]:
    previous_velocities: dict[str, float] | None = None
    previous_accelerations: dict[str, float] | None = None
    max_jerk = 0.0
    max_joint = ""
    for _time_sec, dt, velocities in _segment_velocities(
        joint_names,
        points,
        current_positions=current_positions,
    ):
        accelerations: dict[str, float] = {}
        if previous_velocities is not None:
            for joint in joint_names:
                if joint not in previous_velocities or joint not in velocities:
                    continue
                invalid_segment = (
                    dt <= 0.0
                    or math.isinf(previous_velocities[joint])
                    or math.isinf(velocities[joint])
                )
                if invalid_segment:
                    accelerations[joint] = math.inf
                else:
                    accelerations[joint] = (
                        float(velocities[joint]) - float(previous_velocities[joint])
                    ) / dt
            if previous_accelerations is not None:
                for joint in joint_names:
                    if joint not in previous_accelerations or joint not in accelerations:
                        continue
                    invalid_segment = (
                        dt <= 0.0
                        or math.isinf(previous_accelerations[joint])
                        or math.isinf(accelerations[joint])
                    )
                    if invalid_segment:
                        jerk = math.inf
                    else:
                        jerk = (
                            abs(float(accelerations[joint]) - float(previous_accelerations[joint]))
                            / dt
                        )
                    if jerk > max_jerk:
                        max_jerk = jerk
                        max_joint = str(joint)
            previous_accelerations = accelerations
        previous_velocities = velocities
    return max_jerk, max_joint


def _scale_point_dynamics(point: JointTrajectoryPoint, scale: float) -> None:
    if scale <= 0.0 or abs(scale - 1.0) < 1e-9:
        return
    if list(point.velocities):
        point.velocities = [float(value) / scale for value in point.velocities]
    if list(point.accelerations):
        point.accelerations = [float(value) / (scale * scale) for value in point.accelerations]


def _retime_points(
    points: list[JointTrajectoryPoint],
    *,
    scale: float,
    min_spacing_sec: float,
) -> None:
    previous_time = 0.0
    for point in points:
        target_time = _point_seconds(point) * max(1.0, float(scale))
        if target_time <= previous_time + min_spacing_sec:
            target_time = previous_time + min_spacing_sec
        _set_duration(point.time_from_start, target_time)
        _scale_point_dynamics(point, max(1.0, float(scale)))
        previous_time = target_time


def _make_current_hold_point(
    joint_names: list[str],
    current_positions: dict[str, float],
    time_from_start_sec: float,
) -> JointTrajectoryPoint:
    point = JointTrajectoryPoint()
    point.positions = [float(current_positions[name]) for name in joint_names]
    _set_duration(point.time_from_start, time_from_start_sec)
    return point


def _trajectory_status_base() -> dict[str, Any]:
    return {
        "updated_at": time.time(),
        "state": "checking",
        "message": "",
        "blocked_reason": "",
        "start_delta_rad": None,
        "start_delta_joint": "",
        "first_point_time": None,
        "min_point_spacing": None,
        "max_segment_velocity_rad_s": None,
        "max_segment_velocity_joint": "",
        "max_segment_acceleration_rad_s2": None,
        "max_segment_acceleration_joint": "",
        "max_segment_jerk_rad_s3": None,
        "max_segment_jerk_joint": "",
        "velocity_time_scale": 1.0,
        "acceleration_time_scale": 1.0,
        "jerk_time_scale": 1.0,
        "shoulder_pan_extra_scale_applied": False,
        "time_scale_applied": 1.0,
        "max_joint_velocity_limit_rad_s": UR5E_RTDE_MAX_JOINT_VEL_RAD_S,
        "max_joint_acceleration_limit_rad_s2": UR5E_RTDE_MAX_JOINT_ACCEL_RAD_S2,
        "max_joint_jerk_limit_rad_s3": UR5E_RTDE_MAX_JOINT_JERK_RAD_S3,
        "movej_speed_rad_s": UR5E_RTDE_MOVEJ_SPEED_RAD_S,
        "movej_acceleration_rad_s2": UR5E_RTDE_MOVEJ_ACCEL_RAD_S2,
        "shoulder_pan_extra_scale": UR5E_RTDE_SHOULDER_PAN_EXTRA_SCALE,
        "hardware_runtime_config": HARDWARE_ARMS_CONFIG_FILE,
        "hardware_arms_config": HARDWARE_ARMS_CONFIG_FILE,
    }


def prepare_guarded_trajectory(
    trajectory: Any,
    current_positions: dict[str, float],
    *,
    max_joint_velocity_rad_s: float = UR5E_RTDE_MAX_JOINT_VEL_RAD_S,
    max_joint_acceleration_rad_s2: float = UR5E_RTDE_MAX_JOINT_ACCEL_RAD_S2,
    max_joint_jerk_rad_s3: float = UR5E_RTDE_MAX_JOINT_JERK_RAD_S3,
    start_tolerance_rad: float = UR5E_RTDE_START_TOLERANCE_RAD,
    current_hold_sec: float = UR5E_RTDE_CURRENT_HOLD_SEC,
    min_point_spacing_sec: float = UR5E_RTDE_MIN_POINT_SPACING_SEC,
    shoulder_pan_extra_scale: float = UR5E_RTDE_SHOULDER_PAN_EXTRA_SCALE,
) -> tuple[bool, Any | None, dict[str, Any]]:
    status = _trajectory_status_base()
    status.update(
        max_joint_velocity_limit_rad_s=float(max_joint_velocity_rad_s),
        max_joint_acceleration_limit_rad_s2=float(max_joint_acceleration_rad_s2),
        max_joint_jerk_limit_rad_s3=float(max_joint_jerk_rad_s3),
        shoulder_pan_extra_scale=float(shoulder_pan_extra_scale),
    )
    guarded = copy.deepcopy(trajectory)
    joint_names = [str(name) for name in list(getattr(guarded, "joint_names", []) or [])]
    points = list(getattr(guarded, "points", []) or [])
    status["point_count"] = len(points)
    status["joint_names"] = list(joint_names)

    missing = [joint for joint in ARM_JOINTS if joint not in joint_names]
    if missing:
        status.update(
            state="blocked",
            blocked_reason=f"trajectory missing UR5e arm joints: {', '.join(missing)}",
            message=f"blocked: trajectory missing UR5e arm joints: {', '.join(missing)}",
        )
        return False, None, status
    if not points:
        status.update(
            state="blocked",
            blocked_reason="trajectory has no points",
            message="blocked: trajectory has no points",
        )
        return False, None, status

    current_missing = [joint for joint in joint_names if joint not in current_positions]
    if current_missing:
        status.update(
            state="blocked",
            blocked_reason=f"current /joint_states missing joints: {', '.join(current_missing)}",
            message=f"blocked: current /joint_states missing joints: {', '.join(current_missing)}",
        )
        return False, None, status

    first_point_time = _point_seconds(points[0])
    start_delta, start_joint = _max_named_delta(
        joint_names,
        list(points[0].positions),
        current_positions,
    )
    status["first_point_time"] = first_point_time
    status["start_delta_rad"] = start_delta
    status["start_delta_joint"] = start_joint

    if start_delta > float(start_tolerance_rad):
        status.update(
            state="blocked",
            blocked_reason=(
                f"trajectory start differs from hardware by {start_delta:.4f} rad "
                f"at {start_joint}; tolerance={float(start_tolerance_rad):.4f}"
            ),
            message="blocked: trajectory start differs from hardware",
        )
        return False, None, status

    if first_point_time < float(current_hold_sec):
        offset = float(current_hold_sec) - first_point_time + float(min_point_spacing_sec)
        for point in points:
            _set_duration(point.time_from_start, _point_seconds(point) + offset)
        hold = _make_current_hold_point(joint_names, current_positions, float(current_hold_sec))
        guarded.points = [hold, *points]
        points = list(guarded.points)
        status["inserted_current_hold"] = True
    else:
        status["inserted_current_hold"] = False

    _retime_points(points, scale=1.0, min_spacing_sec=float(min_point_spacing_sec))
    max_velocity, max_velocity_joint = _max_segment_velocity(
        joint_names,
        points,
        current_positions=current_positions,
    )
    velocity_time_scale = 1.0
    if math.isinf(max_velocity):
        velocity_time_scale = 10.0
    elif max_velocity > float(max_joint_velocity_rad_s) > 0.0:
        velocity_time_scale = max_velocity / float(max_joint_velocity_rad_s)
    if velocity_time_scale > 1.0:
        _retime_points(
            points,
            scale=velocity_time_scale,
            min_spacing_sec=float(min_point_spacing_sec),
        )

    max_acceleration, max_acceleration_joint = _max_segment_acceleration(
        joint_names,
        points,
        current_positions=current_positions,
    )
    acceleration_time_scale = 1.0
    if math.isinf(max_acceleration):
        acceleration_time_scale = 10.0
    elif max_acceleration > float(max_joint_acceleration_rad_s2) > 0.0:
        acceleration_time_scale = math.sqrt(max_acceleration / float(max_joint_acceleration_rad_s2))
    if acceleration_time_scale > 1.0:
        _retime_points(
            points,
            scale=acceleration_time_scale,
            min_spacing_sec=float(min_point_spacing_sec),
        )

    max_jerk, max_jerk_joint = _max_segment_jerk(
        joint_names,
        points,
        current_positions=current_positions,
    )
    jerk_time_scale = 1.0
    if math.isinf(max_jerk):
        jerk_time_scale = 10.0
    elif max_jerk > float(max_joint_jerk_rad_s3) > 0.0:
        jerk_time_scale = (max_jerk / float(max_joint_jerk_rad_s3)) ** (1.0 / 3.0)
    if jerk_time_scale > 1.0:
        _retime_points(points, scale=jerk_time_scale, min_spacing_sec=float(min_point_spacing_sec))

    shoulder_pan_extra_scale_applied = False
    if float(shoulder_pan_extra_scale) > 1.0:
        shoulder_pan_limited = (
            (velocity_time_scale > 1.0 and max_velocity_joint == "shoulder_pan_joint")
            or (acceleration_time_scale > 1.0 and max_acceleration_joint == "shoulder_pan_joint")
            or (jerk_time_scale > 1.0 and max_jerk_joint == "shoulder_pan_joint")
        )
        if shoulder_pan_limited:
            _retime_points(
                points,
                scale=float(shoulder_pan_extra_scale),
                min_spacing_sec=float(min_point_spacing_sec),
            )
            shoulder_pan_extra_scale_applied = True

    final_max_velocity, final_max_velocity_joint = _max_segment_velocity(
        joint_names,
        points,
        current_positions=current_positions,
    )
    final_max_acceleration, final_max_acceleration_joint = _max_segment_acceleration(
        joint_names,
        points,
        current_positions=current_positions,
    )
    final_max_jerk, final_max_jerk_joint = _max_segment_jerk(
        joint_names,
        points,
        current_positions=current_positions,
    )
    time_scale = velocity_time_scale * acceleration_time_scale * jerk_time_scale
    if shoulder_pan_extra_scale_applied:
        time_scale *= float(shoulder_pan_extra_scale)
    status.update(
        state="ready",
        message="RTDE trajectory ready",
        blocked_reason="",
        first_point_time=_point_seconds(points[0]) if points else None,
        min_point_spacing=_min_point_spacing(points),
        max_segment_velocity_rad_s=final_max_velocity,
        max_segment_velocity_joint=final_max_velocity_joint or max_velocity_joint,
        max_segment_acceleration_rad_s2=final_max_acceleration,
        max_segment_acceleration_joint=final_max_acceleration_joint or max_acceleration_joint,
        max_segment_jerk_rad_s3=final_max_jerk,
        max_segment_jerk_joint=final_max_jerk_joint or max_jerk_joint,
        velocity_time_scale=velocity_time_scale,
        acceleration_time_scale=acceleration_time_scale,
        jerk_time_scale=jerk_time_scale,
        shoulder_pan_extra_scale_applied=shoulder_pan_extra_scale_applied,
        time_scale_applied=time_scale,
        point_count=len(points),
    )
    return True, guarded, status


def _status_base() -> dict[str, Any]:
    try:
        configured_world_base = _configured_cartesian_world_base()
        configured_world_base_message = (
            "protected ur5e.rtde.cartesian_world_base loaded; live TF has not been validated"
        )
    except RuntimeError as exc:
        configured_world_base = None
        configured_world_base_message = str(exc)
    mg_hard_caps = _insert_hard_caps("MG")
    mg_hard_caps_error = _insert_hard_cap_error("MG") or ""
    exact_part_hard_caps = {
        part_name: _insert_hard_caps(part_name)
        for part_name in INSERT_SUPPORTED_PART_NAMES
    }
    exact_part_hard_caps_error = {
        part_name: _insert_hard_cap_error(part_name) or ""
        for part_name in INSERT_SUPPORTED_PART_NAMES
    }
    exact_part_hard_caps_sha256 = {
        part_name: (
            ""
            if exact_part_hard_caps_error[part_name]
            else _insert_hard_caps_sha256(exact_part_hard_caps[part_name])
        )
        for part_name in INSERT_SUPPORTED_PART_NAMES
    }
    return {
        "updated_at": time.time(),
        "action": ACTION_NAME,
        "action_name": ACTION_NAME,
        "cartesian_action": CARTESIAN_ACTION_NAME,
        "cartesian_action_name": CARTESIAN_ACTION_NAME,
        "insert_action_name": INSERT_ACTION_NAME,
        "insertion_demonstration_action_name": INSERT_DEMONSTRATION_ACTION_NAME,
        "relative_cartesian_action_name": RELATIVE_CARTESIAN_ACTION_NAME,
        "cartesian_jog_service_name": CARTESIAN_JOG_SERVICE_NAME,
        "joint_jog_action_name": JOINT_JOG_ACTION_NAME,
        "joint_action_ready": True,
        "joint_jog_action_ready": MoveUR5eJointJog is not None,
        "cartesian_action_ready": MoveUR5eCartesian is not None,
        "insert_action_ready": MoveUR5eInsert is not None,
        "insertion_demonstration_action_ready": (
            RecordUR5eInsertionDemonstration is not None
        ),
        "insertion_demonstration_active": False,
        "insertion_demonstration_recording_id": "",
        "insertion_demonstration_phase": "",
        "insertion_demonstration_sample_count": 0,
        "insert_supported_part_names": list(INSERT_SUPPORTED_PART_NAMES),
        "insert_MG_hard_caps": mg_hard_caps,
        "insert_MG_hard_caps_error": mg_hard_caps_error,
        "insert_MG_hard_caps_sha256": (
            ""
            if mg_hard_caps_error
            else _insert_hard_caps_sha256(mg_hard_caps)
        ),
        "insert_exact_part_hard_caps": exact_part_hard_caps,
        "insert_exact_part_hard_caps_error": exact_part_hard_caps_error,
        "insert_exact_part_hard_caps_sha256": exact_part_hard_caps_sha256,
        "insert_selected_part_name": "",
        "insert_selected_hard_caps": {},
        "insert_selected_hard_caps_error": "",
        "insert_selected_hard_caps_sha256": "",
        "part_name": "",
        "calibration_id": "",
        "profile_sha256": "",
        "trial_id": "",
        "relative_cartesian_action_ready": MoveUR5eRelativeCartesian is not None,
        "cartesian_jog_service_ready": SetUR5eCartesianJog is not None,
        "cartesian_jog_ready": False,
        "cartesian_function_ready": False,
        "tcp_force_feedback_ready": False,
        "tcp_speed_feedback_ready": False,
        "insert_function_ready": False,
        "insert_readiness_message": (
            _insert_hard_cap_error()
            or "Insertion interface and TCP force/speed feedback validation have not completed"
        ),
        "insert_phase": "",
        "insert_insertion_depth_m": None,
        "insert_depth_error_m": None,
        "insert_lateral_offset_m": None,
        "insert_search_radius_m": None,
        "insert_search_peck_state": "",
        "insert_search_peck_cycle_count": 0,
        "insert_search_peck_retreat_m": 0.0,
        "insert_axial_force_n": None,
        "insert_raw_axial_force_n": None,
        "insert_lateral_force_n": None,
        "insert_torque_nm": None,
        "insert_filtered_axial_force_n": None,
        "insert_filtered_lateral_force_n": None,
        "insert_filtered_torque_nm": None,
        "insert_tool_flange_torque_nm": None,
        "insert_filtered_tool_flange_torque_nm": None,
        "insert_axial_profile_exceeded": False,
        "insert_lateral_profile_exceeded": False,
        "insert_torque_profile_exceeded": False,
        "insert_tared_tcp_force": None,
        "insert_contact_detected": False,
        "insert_engagement_detected": False,
        "insert_seated_detected": False,
        "insert_soft_overload_detected": False,
        "insert_soft_overload_reason": "",
        "insert_soft_overload_duration_sec": 0.0,
        "insert_soft_overload_recovered": False,
        "insert_relief_exhausted": False,
        "insert_relief_cycle_count": 0,
        "insert_relief_elapsed_sec": 0.0,
        "insert_relief_retreat_m": 0.0,
        "insert_relief_load_cleared": False,
        "insert_relief_backoff_m": 0.0,
        "insert_relief_planned_backoff_m": 0.0,
        "insert_total_relief_backoff_m": 0.0,
        "insert_relief_resume_phase": "",
        "insert_relief_force_mode_stop_acknowledged": False,
        "insert_relief_stop_l_command_completed": False,
        "insert_relief_stationary_confirmed": False,
        "insert_relief_force_mode_restart_acknowledged": False,
        "insert_commanded_axial_force_n": 0.0,
        "insert_commanded_lateral_force_x_n": 0.0,
        "insert_commanded_lateral_force_y_n": 0.0,
        "insert_hard_limit_detected": False,
        "insert_hard_limit_reason": "",
        "insert_limit_trigger": "",
        "insert_limit_trigger_value": None,
        "insert_limit_trigger_threshold": None,
        "insert_limit_trigger_actual_tcp_force": None,
        "insert_limit_trigger_tared_tcp_force": None,
        "insert_force_mode_stop_acknowledged": False,
        "insert_servo_stop_acknowledged": False,
        "insert_stop_l_command_completed": False,
        "insert_stationary_confirmed": False,
        "server_trace_id": "",
        "server_trace_path": "",
        "server_trace_sha256": "",
        "server_trace_status": "not_started",
        "server_trace_complete": False,
        "server_trace_sample_count": 0,
        "insert_motion_settled": True,
        "target_insertion_depth_m": None,
        "force_bias_valid": False,
        "force_bias": None,
        "peak_axial_force_n": None,
        "peak_lateral_force_n": None,
        "peak_torque_nm": None,
        "peak_filtered_axial_force_n": None,
        "peak_filtered_lateral_force_n": None,
        "peak_filtered_torque_nm": None,
        "peak_tool_flange_torque_nm": None,
        "actual_tcp_force": None,
        "actual_tcp_speed": None,
        "cartesian_frame_validation_message": "Cartesian frame validation has not completed",
        "cartesian_frame_position_error_m": None,
        "cartesian_frame_orientation_error_rad": None,
        "cartesian_world_base_ready": False,
        "cartesian_world_base_message": configured_world_base_message,
        "cartesian_world_base_expected": _transform_status_payload(
            configured_world_base
        ),
        "cartesian_world_base_observed": None,
        "cartesian_world_base_position_error_m": None,
        "cartesian_world_base_orientation_error_rad": None,
        "state": "checking",
        "message": "",
        "blocked_reason": "",
        "rtde_connected": False,
        "rtde_receive_connected": False,
        "rtde_control_connected": False,
        "rtde_reset_required": False,
        "joint_states_fresh": False,
        "start_delta_rad": None,
        "start_delta_joint": "",
        "first_point_time": None,
        "min_point_spacing": None,
        "max_segment_velocity_rad_s": None,
        "max_segment_velocity_joint": "",
        "max_segment_acceleration_rad_s2": None,
        "max_segment_acceleration_joint": "",
        "max_segment_jerk_rad_s3": None,
        "max_segment_jerk_joint": "",
        "time_scale_applied": 1.0,
        "movej_path_points": 0,
        "trajectory_requested_final_time_sec": None,
        "trajectory_guarded_final_time_sec": None,
        "trajectory_result_timeout_sec": None,
        "trajectory_elapsed_sec": None,
        "allowed_execution_duration_scaling": (UR5E_RTDE_ALLOWED_EXECUTION_DURATION_SCALING),
        "allowed_goal_duration_margin_sec": UR5E_RTDE_RESULT_MARGIN_SEC,
        "joint_goal_tolerance_rad": UR5E_RTDE_GOAL_TOLERANCE_RAD,
        "final_joint_error_rad": None,
        "final_joint_error_joint": "",
        "max_actual_joint_velocity_rad_s": None,
        "max_observed_joint_velocity_rad_s": None,
        "stationary_hold_sec": 0.0,
        "stationary_velocity_limit_rad_s": UR5E_RTDE_STATIONARY_MAX_JOINT_VEL_RAD_S,
        "stationary_hold_required_sec": UR5E_RTDE_STATIONARY_HOLD_SEC,
        "stopped_away_hold_sec": 0.0,
        "stopped_away_hold_required_sec": UR5E_RTDE_STOPPED_AWAY_HOLD_SEC,
        "rtde_feedback_gap_sec": 0.0,
        "rtde_feedback_reconnect_after_sec": UR5E_RTDE_FEEDBACK_RECONNECT_AFTER_SEC,
        "rtde_feedback_reconnect_retry_sec": UR5E_RTDE_FEEDBACK_RECONNECT_RETRY_SEC,
        "rtde_feedback_recovery_timeout_sec": UR5E_RTDE_FEEDBACK_RECOVERY_TIMEOUT_SEC,
        "rtde_feedback_reconnect_count": 0,
        "rtde_feedback_reconnect_error": "",
        "rtde_frequency_hz": UR5E_RTDE_FREQUENCY_HZ,
        "rtde_receive_variables": list(UR5E_RTDE_RECEIVE_VARIABLES),
        "rtde_failure_kind": "",
        "rtde_failure_feedback_gap_sec": None,
        "rtde_last_receive_timestamp": None,
        "rtde_feedback_timestamp_sec": None,
        "rtde_receive_reported_connected_before_reset": None,
        "rtde_control_reported_connected_before_reset": None,
        "rtde_result": "",
        "rtde_command_mode": "",
        "rtde_async_dispatch_elapsed_sec": None,
        "max_joint_velocity_limit_rad_s": UR5E_RTDE_MAX_JOINT_VEL_RAD_S,
        "max_joint_acceleration_limit_rad_s2": UR5E_RTDE_MAX_JOINT_ACCEL_RAD_S2,
        "max_joint_jerk_limit_rad_s3": UR5E_RTDE_MAX_JOINT_JERK_RAD_S3,
        "movej_speed_rad_s": UR5E_RTDE_MOVEJ_SPEED_RAD_S,
        "movej_acceleration_rad_s2": UR5E_RTDE_MOVEJ_ACCEL_RAD_S2,
        "shoulder_pan_extra_scale": UR5E_RTDE_SHOULDER_PAN_EXTRA_SCALE,
        "cartesian_speed_default_m_s": UR5E_RTDE_CARTESIAN_SPEED_M_S,
        "cartesian_speed_limit_m_s": UR5E_RTDE_CARTESIAN_MAX_SPEED_M_S,
        "cartesian_acceleration_limit_m_s2": UR5E_RTDE_CARTESIAN_ACCEL_M_S2,
        "cartesian_position_tolerance_m": UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M,
        "cartesian_orientation_tolerance_rad": (UR5E_RTDE_CARTESIAN_ORIENTATION_TOLERANCE_RAD),
        "hardware_runtime_config": HARDWARE_ARMS_CONFIG_FILE,
        "hardware_arms_config": HARDWARE_ARMS_CONFIG_FILE,
        **_insert_hard_caps(),
    }


def prepare_rtde_trajectory(
    trajectory: Any,
    current_positions: dict[str, float],
) -> tuple[bool, Any | None, dict[str, Any]]:
    ok, guarded, guard_status = prepare_guarded_trajectory(
        trajectory,
        current_positions,
        max_joint_velocity_rad_s=UR5E_RTDE_MAX_JOINT_VEL_RAD_S,
        max_joint_acceleration_rad_s2=UR5E_RTDE_MAX_JOINT_ACCEL_RAD_S2,
        max_joint_jerk_rad_s3=UR5E_RTDE_MAX_JOINT_JERK_RAD_S3,
        start_tolerance_rad=UR5E_RTDE_START_TOLERANCE_RAD,
        current_hold_sec=UR5E_RTDE_CURRENT_HOLD_SEC,
        min_point_spacing_sec=UR5E_RTDE_MIN_POINT_SPACING_SEC,
        shoulder_pan_extra_scale=UR5E_RTDE_SHOULDER_PAN_EXTRA_SCALE,
    )
    status = _status_base()
    for key in (
        "blocked_reason",
        "start_delta_rad",
        "start_delta_joint",
        "first_point_time",
        "min_point_spacing",
        "max_segment_velocity_rad_s",
        "max_segment_velocity_joint",
        "max_segment_acceleration_rad_s2",
        "max_segment_acceleration_joint",
        "max_segment_jerk_rad_s3",
        "max_segment_jerk_joint",
        "time_scale_applied",
        "max_joint_velocity_limit_rad_s",
        "max_joint_acceleration_limit_rad_s2",
        "max_joint_jerk_limit_rad_s3",
        "movej_speed_rad_s",
        "movej_acceleration_rad_s2",
        "shoulder_pan_extra_scale",
        "point_count",
        "joint_names",
    ):
        if key in guard_status:
            status[key] = guard_status.get(key)
    if not ok or guarded is None:
        failure_message = (
            guard_status.get("message")
            or guard_status.get("blocked_reason")
            or "trajectory rejected"
        )
        status.update(
            state="blocked",
            message=str(failure_message),
        )
        return False, None, status
    status.update(state="ready", message="RTDE trajectory ready", blocked_reason="")
    return True, guarded, status


def rtde_movej_path(
    trajectory: Any,
    *,
    speed_rad_s: float | None = None,
    acceleration_rad_s2: float | None = None,
    blend_rad: float = UR5E_RTDE_INTERMEDIATE_BLEND_RAD,
) -> list[list[float]]:
    speed = UR5E_RTDE_MOVEJ_SPEED_RAD_S if speed_rad_s is None else float(speed_rad_s)
    acceleration = (
        UR5E_RTDE_MOVEJ_ACCEL_RAD_S2 if acceleration_rad_s2 is None else float(acceleration_rad_s2)
    )
    joint_names = [str(name) for name in list(getattr(trajectory, "joint_names", []) or [])]
    points = list(getattr(trajectory, "points", []) or [])
    index_by_joint = {name: index for index, name in enumerate(joint_names)}
    path: list[list[float]] = []
    for point_index, point in enumerate(points):
        positions = list(point.positions)
        q = [float(positions[index_by_joint[joint]]) for joint in ARM_JOINTS]
        blend = 0.0 if point_index == len(points) - 1 else max(0.0, float(blend_rad))
        duplicate_delta = (
            max(abs(current - previous) for current, previous in zip(q, path[-1][:6], strict=True))
            if path
            else math.inf
        )
        if duplicate_delta <= 1e-9:
            path[-1][-1] = blend
            continue
        path.append([*q, speed, acceleration, blend])
    return path


class UR5eRTDETrajectoryServer(Node):
    def __init__(  # noqa: PLR0915
        self,
        *,
        robot_ip: str,
        status_file: Path,
        publish_rate_hz: float = 50.0,
        monitor_only: bool = False,
        control_factory: Callable[[str], Any] | None = None,
        receive_factory: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__("ur5e_rtde_trajectory_server")
        self.robot_ip = str(robot_ip or "").strip()
        self.status_file = Path(status_file)
        try:
            self.ros_domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
        except ValueError:
            self.ros_domain_id = -1
        self.monitor_only = bool(monitor_only)
        self.control_factory = control_factory
        self.receive_factory = receive_factory
        self.control = None
        self.receive = None
        self.current_positions: list[float] | None = None
        self.current_positions_monotonic: float | None = None
        self._last_receive_timestamp: float | None = None
        self._last_actual_tcp_force: list[float] | None = None
        self._last_actual_tcp_speed: list[float] | None = None
        self._receive_watch_started_monotonic = time.monotonic()
        self._idle_receive_reconnect_count = 0
        self._receive_lock = threading.Lock()
        self._receive_error = ""
        self._receive_transport_failed = False
        self._rtde_reset_required = False
        self._rtde_reset_reason = ""
        self._rtde_failure_kind = ""
        self._rtde_failure_feedback_gap_sec: float | None = None
        self._rtde_receive_reported_connected_before_reset: bool | None = None
        self._rtde_control_reported_connected_before_reset: bool | None = None
        self._control_error = ""
        self._joint_status_announced = False
        self._status_lock = threading.Lock()
        self._next_status_heartbeat_monotonic = 0.0
        self._active_lock = threading.Lock()
        self._active_goal = None
        self._active_goal_status: dict[str, Any] | None = None
        self._active_motion_kind = ""
        self._latched_terminal_status: dict[str, Any] | None = None
        self._shutdown_requested = False
        self._interfaces_disconnected = False
        self._interface_disconnect_lock = threading.Lock()
        self._cartesian_frame_ready = False
        self._cartesian_jog_ready = False
        self._cartesian_function_ready = False
        self._tcp_force_feedback_ready = False
        self._tcp_speed_feedback_ready = False
        self._insert_function_ready = False
        self._insert_readiness_message = (
            _insert_hard_cap_error()
            or "Insertion interface and TCP force/speed feedback validation have not completed"
        )
        self._insert_force_mode_active = False
        self._insert_force_mode_command: (
            tuple[list[float], list[int], list[float], int, list[float]] | None
        ) = None
        self._insert_servo_active = False
        self._insert_force_mode_stop_acknowledged = False
        self._insert_servo_stop_acknowledged = False
        self._insert_stop_l_command_completed = False
        self._insert_motion_started = False
        self._insertion_demonstration_lock = threading.Lock()
        self._active_insertion_demonstration_goal: Any | None = None
        self._active_insertion_demonstration_status: dict[str, Any] = {}
        self._cartesian_frame_message = "Cartesian frame validation has not completed"
        self._cartesian_frame_position_error_m = math.inf
        self._cartesian_frame_orientation_error_rad = math.inf
        self._cartesian_world_base_ready = False
        self._cartesian_world_base_message = (
            "protected ur5e.rtde.cartesian_world_base has not been validated against live TF"
        )
        try:
            self._cartesian_world_base_expected: RigidTransform | None = (
                _configured_cartesian_world_base()
            )
        except RuntimeError as exc:
            self._cartesian_world_base_expected = None
            self._cartesian_world_base_message = str(exc)
        self._cartesian_world_base_observed: RigidTransform | None = None
        self._cartesian_world_base_position_error_m = math.inf
        self._cartesian_world_base_orientation_error_rad = math.inf
        self._jog_session_token = object()
        self._jog_watchdog_deadline = 0.0
        self._jog_stop_in_progress = False
        self._jog_world_base: RigidTransform | None = None
        self._jog_frame_message = ""
        self._jog_base_velocity_m_s: tuple[float, float, float] | None = None
        self._jog_acceleration_m_s2: float | None = None
        self.terminal_status_file = self.status_file.with_name(
            f"{self.status_file.stem}_last_terminal{self.status_file.suffix}"
        )
        self._joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
        self._timer = self.create_timer(
            1.0 / max(1.0, float(publish_rate_hz)),
            self._publish_joint_state,
        )
        self._action_server = None
        self._cartesian_action_server = None
        self._insert_action_server = None
        self._insertion_demonstration_action_server = None
        self._relative_cartesian_action_server = None
        self._joint_jog_action_server = None
        self._cartesian_jog_service = None
        self._insertion_demonstration_callback_group = ReentrantCallbackGroup()
        if not self.monitor_only:
            self._action_server = ActionServer(
                self,
                FollowJointTrajectory,
                ACTION_NAME,
                execute_callback=self._execute,
                cancel_callback=self._cancel,
            )
            if MoveUR5eCartesian is not None:
                self._cartesian_action_server = ActionServer(
                    self,
                    MoveUR5eCartesian,
                    CARTESIAN_ACTION_NAME,
                    execute_callback=self._execute_cartesian,
                    cancel_callback=self._cancel,
                )
            if MoveUR5eInsert is not None:
                self._insert_action_server = ActionServer(
                    self,
                    MoveUR5eInsert,
                    INSERT_ACTION_NAME,
                    execute_callback=self._execute_insert,
                    cancel_callback=self._cancel,
                )
            if RecordUR5eInsertionDemonstration is not None:
                self._insertion_demonstration_action_server = ActionServer(
                    self,
                    RecordUR5eInsertionDemonstration,
                    INSERT_DEMONSTRATION_ACTION_NAME,
                    execute_callback=self._execute_insertion_demonstration,
                    cancel_callback=self._cancel,
                    callback_group=self._insertion_demonstration_callback_group,
                )
            if MoveUR5eRelativeCartesian is not None:
                self._relative_cartesian_action_server = ActionServer(
                    self,
                    MoveUR5eRelativeCartesian,
                    RELATIVE_CARTESIAN_ACTION_NAME,
                    execute_callback=self._execute_relative_cartesian,
                    cancel_callback=self._cancel,
                )
            if MoveUR5eJointJog is not None:
                self._joint_jog_action_server = ActionServer(
                    self,
                    MoveUR5eJointJog,
                    JOINT_JOG_ACTION_NAME,
                    execute_callback=self._execute_joint_jog,
                    cancel_callback=self._cancel,
                )
            if SetUR5eCartesianJog is not None:
                self._cartesian_jog_service = self.create_service(
                    SetUR5eCartesianJog,
                    CARTESIAN_JOG_SERVICE_NAME,
                    self._set_cartesian_jog,
                )
        status = _status_base()
        if not self.robot_ip:
            status.update(
                state="blocked",
                blocked_reason="--robot-ip is required",
                message="blocked: --robot-ip is required",
            )
            self._write_status(status)
            return
        missing_interfaces = [
            name
            for name, value in (
                ("MoveUR5eCartesian", MoveUR5eCartesian),
                ("MoveUR5eJointJog", MoveUR5eJointJog),
                ("MoveUR5eRelativeCartesian", MoveUR5eRelativeCartesian),
                ("SetUR5eCartesianJog", SetUR5eCartesianJog),
            )
            if value is None
        ]
        if not self.monitor_only and missing_interfaces:
            reason = (
                f"{', '.join(missing_interfaces)} interface is unavailable. Rebuild and source "
                "cais_lab_robotics before starting Hardware Stack."
            )
            status.update(state="blocked", blocked_reason=reason, message=f"blocked: {reason}")
            self._write_status(status)
            return
        self._connect_rtde()

    def _write_status(self, payload: dict[str, Any]) -> None:
        body = dict(payload)
        body["monitor_only"] = self.monitor_only
        if self.monitor_only:
            body["action"] = ""
            body["action_name"] = ""
            body["cartesian_action"] = ""
            body["cartesian_action_name"] = ""
            body["insert_action_name"] = ""
            body["insertion_demonstration_action_name"] = ""
            body["relative_cartesian_action_name"] = ""
            body["cartesian_jog_service_name"] = ""
            body["joint_jog_action_name"] = ""
            body["joint_action_ready"] = False
            body["joint_jog_action_ready"] = False
            body["cartesian_action_ready"] = False
            body["insert_action_ready"] = False
            body["insertion_demonstration_action_ready"] = False
            body["relative_cartesian_action_ready"] = False
            body["cartesian_jog_service_ready"] = False
            body["cartesian_jog_ready"] = False
            body["cartesian_function_ready"] = False
            body["tcp_force_feedback_ready"] = False
            body["tcp_speed_feedback_ready"] = False
            body["insert_function_ready"] = False
            body["insert_readiness_message"] = (
                "Insertion motion is unavailable in read-only calibration monitoring"
            )
        body["ros_domain_id"] = self.ros_domain_id
        body["process_id"] = os.getpid()
        body["rtde_reset_required"] = bool(getattr(self, "_rtde_reset_required", False))
        with self._insertion_demonstration_lock:
            demonstration_status = dict(self._active_insertion_demonstration_status)
        body.update(demonstration_status)
        body["rtde_failure_kind"] = str(getattr(self, "_rtde_failure_kind", "") or "")
        body["rtde_failure_feedback_gap_sec"] = getattr(
            self,
            "_rtde_failure_feedback_gap_sec",
            None,
        )
        body["rtde_last_receive_timestamp"] = getattr(
            self,
            "_last_receive_timestamp",
            None,
        )
        body["rtde_feedback_timestamp_sec"] = getattr(
            self,
            "_last_receive_timestamp",
            None,
        )
        body["rtde_receive_reported_connected_before_reset"] = getattr(
            self,
            "_rtde_receive_reported_connected_before_reset",
            None,
        )
        body["rtde_control_reported_connected_before_reset"] = getattr(
            self,
            "_rtde_control_reported_connected_before_reset",
            None,
        )
        expected_world_base = getattr(
            self,
            "_cartesian_world_base_expected",
            None,
        )
        observed_world_base = getattr(
            self,
            "_cartesian_world_base_observed",
            None,
        )
        world_base_position_error = float(
            getattr(self, "_cartesian_world_base_position_error_m", math.inf)
        )
        world_base_orientation_error = float(
            getattr(self, "_cartesian_world_base_orientation_error_rad", math.inf)
        )
        body["cartesian_world_base_ready"] = bool(
            getattr(self, "_cartesian_world_base_ready", False)
        )
        body["cartesian_world_base_message"] = str(
            getattr(
                self,
                "_cartesian_world_base_message",
                "protected ur5e.rtde.cartesian_world_base has not been validated",
            )
        )
        body["cartesian_world_base_expected"] = _transform_status_payload(
            expected_world_base
        )
        body["cartesian_world_base_observed"] = _transform_status_payload(
            observed_world_base
        )
        body["cartesian_world_base_position_error_m"] = (
            world_base_position_error
            if math.isfinite(world_base_position_error)
            else None
        )
        body["cartesian_world_base_orientation_error_rad"] = (
            world_base_orientation_error
            if math.isfinite(world_base_orientation_error)
            else None
        )
        body["updated_at"] = time.time()
        body["terminal_status_file"] = str(self.terminal_status_file)
        with self._status_lock:
            _atomic_json_write(self.status_file, body)

    def _write_terminal_status(self, payload: dict[str, Any]) -> None:
        """Persist the latest goal outcome separately from restart readiness status."""
        body = dict(payload)
        body["monitor_only"] = self.monitor_only
        body["ros_domain_id"] = self.ros_domain_id
        body["process_id"] = os.getpid()
        body["rtde_reset_required"] = bool(getattr(self, "_rtde_reset_required", False))
        body["rtde_failure_kind"] = str(getattr(self, "_rtde_failure_kind", "") or "")
        body["rtde_failure_feedback_gap_sec"] = getattr(
            self,
            "_rtde_failure_feedback_gap_sec",
            None,
        )
        body["rtde_last_receive_timestamp"] = getattr(
            self,
            "_last_receive_timestamp",
            None,
        )
        body["rtde_feedback_timestamp_sec"] = getattr(
            self,
            "_last_receive_timestamp",
            None,
        )
        expected_world_base = getattr(
            self,
            "_cartesian_world_base_expected",
            None,
        )
        observed_world_base = getattr(
            self,
            "_cartesian_world_base_observed",
            None,
        )
        world_base_position_error = float(
            getattr(self, "_cartesian_world_base_position_error_m", math.inf)
        )
        world_base_orientation_error = float(
            getattr(self, "_cartesian_world_base_orientation_error_rad", math.inf)
        )
        body["cartesian_world_base_ready"] = bool(
            getattr(self, "_cartesian_world_base_ready", False)
        )
        body["cartesian_world_base_message"] = str(
            getattr(self, "_cartesian_world_base_message", "")
        )
        body["cartesian_world_base_expected"] = _transform_status_payload(
            expected_world_base
        )
        body["cartesian_world_base_observed"] = _transform_status_payload(
            observed_world_base
        )
        body["cartesian_world_base_position_error_m"] = (
            world_base_position_error
            if math.isfinite(world_base_position_error)
            else None
        )
        body["cartesian_world_base_orientation_error_rad"] = (
            world_base_orientation_error
            if math.isfinite(world_base_orientation_error)
            else None
        )
        body["updated_at"] = time.time()
        body["terminal_status_file"] = str(self.terminal_status_file)
        with self._status_lock:
            _atomic_json_write(self.terminal_status_file, body)

    def _connect_rtde(self) -> None:
        status = _status_base()
        receive_connected = False
        control_connected = False
        try:
            if self.receive_factory is None:
                import rtde_receive

                self.receive = rtde_receive.RTDEReceiveInterface(
                    self.robot_ip,
                    UR5E_RTDE_FREQUENCY_HZ,
                    list(UR5E_RTDE_RECEIVE_VARIABLES),
                )
            else:
                self.receive = self.receive_factory(self.robot_ip)
            receive_connected = True
            self._receive_error = ""
        except (ImportError, OSError, RuntimeError) as exc:
            self.receive = None
            self._receive_error = f"{type(exc).__name__}: {exc}"
            self._receive_transport_failed = True
            self._rtde_reset_required = True
            self._rtde_reset_reason = f"RTDE receive unavailable: {self._receive_error}"

        if self.monitor_only:
            self.control = None
            self._control_error = "disabled for read-only calibration monitoring"
        else:
            try:
                if self.control_factory is None:
                    import rtde_control

                    self.control = rtde_control.RTDEControlInterface(
                        self.robot_ip,
                        UR5E_RTDE_FREQUENCY_HZ,
                    )
                else:
                    self.control = self.control_factory(self.robot_ip)
                control_connected = True
                self._control_error = ""
            except (ImportError, OSError, RuntimeError) as exc:
                self.control = None
                self._control_error = f"{type(exc).__name__}: {exc}"

        status.update(
            rtde_connected=receive_connected and control_connected,
            rtde_receive_connected=receive_connected,
            rtde_control_connected=control_connected,
            joint_states_fresh=False,
        )
        if receive_connected and control_connected:
            status.update(state="ready", message="UR5e RTDE trajectory server ready")
        elif receive_connected:
            if self.monitor_only:
                status.update(
                    state="monitoring",
                    blocked_reason="",
                    message="UR5e read-only calibration monitoring ready in Local Control",
                )
            else:
                reason = f"RTDE control unavailable: {self._control_error}"
                status.update(
                    state="monitoring",
                    blocked_reason=reason,
                    message=(
                        "UR5e joint-state monitoring ready in Local Control; "
                        "trajectory motion requires Remote Control"
                    ),
                )
        else:
            details = [f"RTDE receive unavailable: {self._receive_error}"]
            if self._control_error and not self.monitor_only:
                details.append(f"RTDE control unavailable: {self._control_error}")
            reason = "; ".join(details)
            status.update(state="blocked", blocked_reason=reason, message=f"blocked: {reason}")
        self._write_status(status)

    def _mark_rtde_reset_required(
        self,
        reason: str,
        *,
        write_status: bool,
        failure_kind: str = "",
        feedback_gap_sec: float | None = None,
    ) -> None:
        """Latch transport recovery until this process is replaced."""
        reset_was_required = bool(getattr(self, "_rtde_reset_required", False))
        self._rtde_reset_required = True
        if not reset_was_required or not str(getattr(self, "_rtde_reset_reason", "")):
            self._rtde_reset_reason = str(reason or "UR5e RTDE reset required")
        if failure_kind and not str(getattr(self, "_rtde_failure_kind", "")):
            self._rtde_failure_kind = str(failure_kind)
        if (
            getattr(self, "_rtde_failure_feedback_gap_sec", None) is None
            and feedback_gap_sec is not None
            and math.isfinite(feedback_gap_sec)
        ):
            self._rtde_failure_feedback_gap_sec = float(feedback_gap_sec)
        if getattr(self, "_rtde_receive_reported_connected_before_reset", None) is None:
            self._rtde_receive_reported_connected_before_reset = self._interface_reported_connected(
                getattr(self, "receive", None)
            )
        if getattr(self, "_rtde_control_reported_connected_before_reset", None) is None:
            self._rtde_control_reported_connected_before_reset = self._interface_reported_connected(
                getattr(self, "control", None)
            )
        self._cartesian_frame_ready = False
        self._cartesian_jog_ready = False
        self._cartesian_function_ready = False
        self._tcp_force_feedback_ready = False
        self._tcp_speed_feedback_ready = False
        self._insert_function_ready = False
        self._insert_readiness_message = self._rtde_reset_reason
        self._cartesian_frame_message = self._rtde_reset_reason
        self._joint_status_announced = False
        self._disconnect_rtde_interfaces()
        if not write_status:
            return
        status = _status_base()
        status.update(
            state="failed",
            blocked_reason=self._rtde_reset_reason,
            message=self._rtde_reset_reason,
            rtde_connected=False,
            rtde_receive_connected=False,
            rtde_control_connected=self.control is not None,
            rtde_reset_required=True,
            joint_states_fresh=False,
        )
        self._write_status(status)

    @staticmethod
    def _interface_reported_connected(interface: Any) -> bool | None:
        """Read a native RTDE connection flag without changing interface state."""
        is_connected = getattr(interface, "isConnected", None)
        if not callable(is_connected):
            return None
        try:
            return bool(is_connected())
        except (OSError, RuntimeError, TypeError, ValueError):
            return None

    def _disconnect_rtde_interfaces(self) -> None:
        """Disconnect each native RTDE interface once after failure or shutdown."""
        disconnect_lock = getattr(self, "_interface_disconnect_lock", None)
        if disconnect_lock is None:
            disconnect_lock = threading.Lock()
            self._interface_disconnect_lock = disconnect_lock
        with disconnect_lock:
            if bool(getattr(self, "_interfaces_disconnected", False)):
                return
            self._interfaces_disconnected = True
            control = self.control
            receive = self.receive
            self.control = None
            self.receive = None
        for interface in (control, receive):
            disconnect = getattr(interface, "disconnect", None)
            if not callable(disconnect):
                continue
            with suppress(OSError, RuntimeError, TypeError, ValueError):
                disconnect()

    def _connect_control_for_goal(self) -> str | None:
        if bool(getattr(self, "_shutdown_requested", False)):
            return "UR5e RTDE server is stopping"
        if bool(getattr(self, "_rtde_reset_required", False)):
            return str(getattr(self, "_rtde_reset_reason", "")) or "UR5e RTDE reset required"
        control = self.control
        if control is None:
            return (
                "UR5e RTDE control unavailable. Set the teach pendant to Remote Control "
                "and use Repair Hardware Stack before commanding motion."
            )
        is_connected = getattr(control, "isConnected", None)
        if not callable(is_connected):
            return None
        try:
            if bool(is_connected()):
                return None
            detail = "RTDEControlInterface reports disconnected"
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
        reason = (
            f"UR5e RTDE control transport failed: {detail}. "
            "Use Repair Hardware Stack to create one fresh RTDE owner."
        )
        self._control_error = detail
        self._mark_rtde_reset_required(
            reason,
            write_status=True,
            failure_kind="control_disconnected",
        )
        return reason

    def _read_actual_q(self) -> list[float] | None:
        if bool(getattr(self, "_rtde_reset_required", False)):
            return None
        error = ""
        sample_fresh = True
        values: list[float] = []
        with self._receive_lock:
            if self.receive is None:
                error = self._receive_error or "RTDE receive connection is unavailable"
            else:
                try:
                    values = [float(value) for value in list(self.receive.getActualQ())]
                    get_timestamp = getattr(self.receive, "getTimestamp", None)
                    if callable(get_timestamp):
                        receive_timestamp = float(get_timestamp())
                        if math.isfinite(receive_timestamp):
                            previous_timestamp = getattr(
                                self,
                                "_last_receive_timestamp",
                                None,
                            )
                            sample_fresh = (
                                previous_timestamp is None
                                or receive_timestamp != previous_timestamp
                            )
                            if sample_fresh:
                                self._last_receive_timestamp = receive_timestamp
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    self._receive_error = error
                    self._receive_transport_failed = True
        if error:
            reason = (
                f"UR5e RTDE feedback transport failed: {error}. "
                "Use Reset UR5e RTDE in Interactive Teleop."
            )
            self._mark_rtde_reset_required(
                reason,
                write_status=True,
                failure_kind="receive_exception",
            )
            return None
        if len(values) < len(ARM_JOINTS):
            return None
        self.current_positions = values[: len(ARM_JOINTS)]
        if sample_fresh:
            self.current_positions_monotonic = time.monotonic()
        if not self._joint_status_announced:
            control_connected = self.control is not None
            status = _status_base()
            status.update(
                state="ready" if control_connected else "monitoring",
                message=(
                    "UR5e RTDE trajectory server ready"
                    if control_connected
                    else (
                        "UR5e read-only calibration monitoring ready in Local Control"
                        if self.monitor_only
                        else (
                            "UR5e joint-state monitoring ready in Local Control; "
                            "trajectory motion requires Remote Control"
                        )
                    )
                ),
                blocked_reason=(
                    ""
                    if control_connected or self.monitor_only
                    else f"RTDE control unavailable: {self._control_error}"
                ),
                rtde_connected=control_connected,
                rtde_receive_connected=True,
                rtde_control_connected=control_connected,
                joint_states_fresh=True,
            )
            self._write_status(status)
            self._joint_status_announced = True
        return self.current_positions

    def _read_actual_qd(self) -> list[float] | None:
        """Return current joint velocities without changing receive readiness."""
        if bool(getattr(self, "_rtde_reset_required", False)):
            return None
        with self._receive_lock:
            receive = self.receive
            get_actual_qd = getattr(receive, "getActualQd", None)
            if not callable(get_actual_qd):
                return None
            try:
                values = [float(value) for value in list(get_actual_qd())]
            except (OSError, RuntimeError, TypeError, ValueError):
                return None
        if len(values) < len(ARM_JOINTS) or not all(math.isfinite(value) for value in values):
            return None
        return values[: len(ARM_JOINTS)]

    def _read_feedback_timestamp(self) -> float | None:
        """Return the controller timestamp used to prove RTDE feedback is advancing."""
        if bool(getattr(self, "_rtde_reset_required", False)):
            return None
        with self._receive_lock:
            receive = self.receive
            get_timestamp = getattr(receive, "getTimestamp", None)
            if not callable(get_timestamp):
                return None
            try:
                value = float(get_timestamp())
            except (OSError, RuntimeError, TypeError, ValueError):
                return None
        return value if math.isfinite(value) else None

    def _current_position_map(self) -> dict[str, float] | None:
        actual = self._read_actual_q()
        if actual is None:
            return None
        return {joint: float(value) for joint, value in zip(ARM_JOINTS, actual)}

    def _joint_states_fresh(self) -> bool:
        if self.current_positions_monotonic is None:
            return False
        return time.monotonic() - self.current_positions_monotonic <= UR5E_RTDE_FEEDBACK_STALE_SEC

    def _stop_cartesian_jog(self, reason: str) -> tuple[bool, str]:
        """Stop one active RTDE jog and release its shared motion ownership."""
        wait_for_existing_stop = False
        with self._active_lock:
            active = self._active_goal is self._jog_session_token
            if self._jog_stop_in_progress:
                wait_for_existing_stop = True
            elif not active:
                self._jog_watchdog_deadline = 0.0
                self._jog_world_base = None
                self._jog_frame_message = ""
                self._jog_base_velocity_m_s = None
                self._jog_acceleration_m_s2 = None
                if bool(getattr(self, "_rtde_reset_required", False)):
                    return False, str(
                        getattr(self, "_rtde_reset_reason", "")
                        or "UR5e RTDE reset required after Cartesian jog"
                    )
                return True, "UR5e Cartesian jog already stopped"
            else:
                self._jog_stop_in_progress = True
        if wait_for_existing_stop:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                with self._active_lock:
                    if not self._jog_stop_in_progress:
                        break
                time.sleep(0.01)
            else:
                return False, "UR5e Cartesian jog stop outcome was not confirmed"
            return self._stop_cartesian_jog(reason)
        try:
            jog_stop = getattr(self.control, "jogStop", None)
            if not callable(jog_stop):
                raise RuntimeError("RTDE control object has no jogStop method")
            stop_result = jog_stop()
            if stop_result is False:
                raise RuntimeError("UR5e RTDE jogStop returned False")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            detail = f"UR5e Cartesian jog stop failed: {type(exc).__name__}: {exc}"
            self._mark_rtde_reset_required(detail, write_status=False)
            return False, detail
        finally:
            with self._active_lock:
                if self._active_goal is self._jog_session_token:
                    self._active_goal = None
                    self._active_goal_status = None
                    self._active_motion_kind = ""
                self._jog_watchdog_deadline = 0.0
                self._jog_stop_in_progress = False
                self._jog_world_base = None
                self._jog_frame_message = ""
                self._jog_base_velocity_m_s = None
                self._jog_acceleration_m_s2 = None
        status = _status_base()
        status.update(
            state="ready",
            message=str(reason or "UR5e Cartesian jog stopped"),
            rtde_connected=True,
            rtde_receive_connected=True,
            rtde_control_connected=True,
            joint_states_fresh=self._joint_states_fresh(),
        )
        self._write_status(status)
        return True, status["message"]

    def _check_jog_watchdog(self) -> None:
        jog_session_token = getattr(self, "_jog_session_token", None)
        watchdog_deadline = float(getattr(self, "_jog_watchdog_deadline", 0.0))
        with self._active_lock:
            expired = bool(
                jog_session_token is not None
                and self._active_goal is jog_session_token
                and watchdog_deadline > 0.0
                and time.monotonic() >= watchdog_deadline
            )
        if expired:
            self._stop_cartesian_jog("UR5e Cartesian jog watchdog stopped motion")

    def _publish_joint_state(self) -> None:  # noqa: C901
        self._check_jog_watchdog()
        previous_sample_at = self.current_positions_monotonic
        actual = self._read_actual_q()
        if actual is None or self.current_positions_monotonic == previous_sample_at:
            now = time.monotonic()
            with self._active_lock:
                active_goal = self._active_goal is not None
            last_fresh_at = self.current_positions_monotonic
            feedback_gap_sec = now - (
                last_fresh_at
                if last_fresh_at is not None
                else self._receive_watch_started_monotonic
            )
            if active_goal:
                return
            if bool(getattr(self, "_rtde_reset_required", False)):
                return
            if feedback_gap_sec >= UR5E_RTDE_FEEDBACK_RECOVERY_TIMEOUT_SEC:
                reason = str(getattr(self, "_rtde_reset_reason", "")) or (
                    "UR5e RTDE feedback stopped advancing for "
                    f"{feedback_gap_sec:.2f} s. Use Reset UR5e RTDE in "
                    "Interactive Teleop."
                )
                self._mark_rtde_reset_required(
                    reason,
                    write_status=True,
                    failure_kind="feedback_timestamp_stalled",
                    feedback_gap_sec=feedback_gap_sec,
                )
            elif feedback_gap_sec >= UR5E_RTDE_FEEDBACK_RECONNECT_AFTER_SEC:
                status = _status_base()
                status.update(
                    state="recovering",
                    message="waiting for transient UR5e RTDE feedback recovery",
                    blocked_reason="",
                    rtde_connected=self.control is not None and self.receive is not None,
                    rtde_receive_connected=self.receive is not None,
                    rtde_control_connected=self.control is not None,
                    joint_states_fresh=False,
                    rtde_feedback_gap_sec=feedback_gap_sec,
                    rtde_feedback_reconnect_count=0,
                    rtde_feedback_reconnect_error="",
                )
                self._write_status(status)
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINTS)
        msg.position = [float(value) for value in actual]
        self._joint_state_pub.publish(msg)
        now = time.monotonic()
        if now < self._next_status_heartbeat_monotonic:
            return
        self._next_status_heartbeat_monotonic = now + 1.0
        with self._active_lock:
            validate_frames = self._active_goal is None
        cartesian_readiness_cached = bool(
            self._cartesian_frame_ready
            and self._cartesian_function_ready
            and self._cartesian_jog_ready
        )
        if (
            validate_frames
            and not cartesian_readiness_cached
            and not self.monitor_only
            and self.control is not None
        ):
            (
                self._cartesian_frame_ready,
                self._cartesian_frame_message,
                self._cartesian_frame_position_error_m,
                self._cartesian_frame_orientation_error_rad,
            ) = self._cartesian_frame_validation()
            function_methods = ("moveL", "isPoseWithinSafetyLimits", "getTCPOffset")
            missing_function_methods = [
                name for name in function_methods if not callable(getattr(self.control, name, None))
            ]
            missing_jog_methods = [
                name
                for name in ("jogStart", "jogStop")
                if not callable(getattr(self.control, name, None))
            ]
            self._cartesian_function_ready = bool(
                self._cartesian_frame_ready and not missing_function_methods
            )
            self._cartesian_jog_ready = bool(
                self._cartesian_function_ready and not missing_jog_methods
            )
            if missing_function_methods:
                self._cartesian_frame_message = (
                    "Cartesian direct interface validation failed: missing RTDE methods "
                    f"{missing_function_methods}"
                )
            elif missing_jog_methods:
                self._cartesian_frame_message = (
                    "Cartesian Smooth Hold validation failed: missing RTDE methods "
                    f"{missing_jog_methods}"
                )
        if (
            validate_frames
            and not self.monitor_only
            and self.control is not None
            and self.receive is not None
        ):
            actual_tcp_force = self._read_actual_tcp_force()
            actual_tcp_speed = self._read_actual_tcp_speed()
            self._tcp_force_feedback_ready = actual_tcp_force is not None
            self._tcp_speed_feedback_ready = actual_tcp_speed is not None
            missing_insert_methods = [
                name
                for name in (
                    "forceMode",
                    "forceModeStop",
                    "getTCPOffset",
                    "isPoseWithinSafetyLimits",
                    "stopL",
                )
                if not callable(getattr(self.control, name, None))
            ]
            cap_error = _insert_hard_cap_error()
            self._insert_function_ready = bool(
                self._cartesian_frame_ready
                and self._cartesian_function_ready
                and self._tcp_force_feedback_ready
                and self._tcp_speed_feedback_ready
                and not missing_insert_methods
                and cap_error is None
                and MoveUR5eInsert is not None
            )
            if cap_error:
                self._insert_readiness_message = cap_error
            elif missing_insert_methods:
                self._insert_readiness_message = (
                    "Insertion interface validation failed: missing RTDE methods "
                    f"{missing_insert_methods}"
                )
            elif not self._tcp_force_feedback_ready:
                self._insert_readiness_message = (
                    "Insertion interface validation failed: actual_TCP_force is unavailable"
                )
            elif not self._tcp_speed_feedback_ready:
                self._insert_readiness_message = (
                    "Insertion interface validation failed: actual_TCP_speed is unavailable"
                )
            elif not self._cartesian_frame_ready or not self._cartesian_function_ready:
                self._insert_readiness_message = self._cartesian_frame_message
            elif MoveUR5eInsert is None:
                self._insert_readiness_message = (
                    "MoveUR5eInsert interface is unavailable; rebuild cais_lab_robotics"
                )
            else:
                self._insert_readiness_message = "UR5e insertion interface ready"
        with self._active_lock:
            control_connected = self.control is not None
            receive_connected = self.receive is not None
            motion_idle = self._active_goal is None
            if bool(getattr(self, "_rtde_reset_required", False)):
                status = _status_base()
                status.update(
                    state="failed",
                    message=str(getattr(self, "_rtde_reset_reason", "")),
                    blocked_reason=str(getattr(self, "_rtde_reset_reason", "")),
                )
                control_connected = False
                receive_connected = False
            elif self._active_goal is not None:
                status = dict(self._active_goal_status or _status_base())
                status.update(
                    state="executing",
                    message=str(
                        status.get("message")
                        or (
                            "executing UR5e RTDE insertion"
                            if self._active_motion_kind == "insert"
                            else (
                                "executing UR5e RTDE Cartesian moveL"
                                if self._active_motion_kind == "cartesian"
                                else "executing UR5e RTDE moveJ path"
                            )
                        )
                    ),
                    blocked_reason="",
                )
            elif self._latched_terminal_status is not None:
                status = dict(self._latched_terminal_status)
            else:
                status = _status_base()
                status.update(
                    state="ready" if control_connected else "monitoring",
                    message=(
                        "UR5e RTDE trajectory server ready"
                        if control_connected
                        else (
                            "UR5e read-only calibration monitoring ready in Local Control"
                            if self.monitor_only
                            else (
                                "UR5e joint-state monitoring ready in Local Control; "
                                "trajectory motion requires Remote Control"
                            )
                        )
                    ),
                    blocked_reason="",
                )
            status.update(
                rtde_connected=control_connected and receive_connected,
                rtde_receive_connected=receive_connected,
                rtde_control_connected=control_connected,
                joint_states_fresh=self._joint_states_fresh(),
                actual_positions_rad=[float(value) for value in actual],
                cartesian_jog_ready=bool(
                    self._cartesian_jog_ready
                    and control_connected
                    and receive_connected
                    and motion_idle
                    and not self._rtde_reset_required
                ),
                cartesian_function_ready=bool(
                    self._cartesian_function_ready
                    and control_connected
                    and receive_connected
                    and motion_idle
                    and not self._rtde_reset_required
                ),
                tcp_force_feedback_ready=bool(
                    self._tcp_force_feedback_ready
                    and receive_connected
                    and not self._rtde_reset_required
                ),
                tcp_speed_feedback_ready=bool(
                    self._tcp_speed_feedback_ready
                    and receive_connected
                    and not self._rtde_reset_required
                ),
                insert_function_ready=bool(
                    self._insert_function_ready
                    and control_connected
                    and receive_connected
                    and motion_idle
                    and not self._rtde_reset_required
                ),
                insert_readiness_message=self._insert_readiness_message,
                actual_tcp_force=(
                    list(getattr(self, "_last_actual_tcp_force", None))
                    if getattr(self, "_last_actual_tcp_force", None) is not None
                    else None
                ),
                actual_tcp_speed=(
                    list(getattr(self, "_last_actual_tcp_speed", None))
                    if getattr(self, "_last_actual_tcp_speed", None) is not None
                    else None
                ),
                cartesian_frame_validation_message=self._cartesian_frame_message,
                cartesian_frame_position_error_m=(
                    self._cartesian_frame_position_error_m
                    if math.isfinite(self._cartesian_frame_position_error_m)
                    else None
                ),
                cartesian_frame_orientation_error_rad=(
                    self._cartesian_frame_orientation_error_rad
                    if math.isfinite(self._cartesian_frame_orientation_error_rad)
                    else None
                ),
            )
            self._write_status(status)

    def _write_active_goal_status(
        self,
        goal_handle: Any,
        payload: dict[str, Any],
    ) -> None:
        with self._active_lock:
            if self._active_goal is not goal_handle:
                return
            self._active_goal_status = dict(payload)
            self._write_status(payload)

    def _finish_active_goal_status(
        self,
        goal_handle: Any,
        payload: dict[str, Any],
        *,
        latch_status: bool = False,
    ) -> None:
        with self._active_lock:
            if self._active_goal is goal_handle:
                self._insert_motion_started = False
                self._active_goal = None
                self._active_goal_status = None
                self._active_motion_kind = ""
            if latch_status:
                self._latched_terminal_status = dict(payload)
            self._write_status(payload)
            self._write_terminal_status(payload)

    def _cancel(self, _goal_handle: Any) -> CancelResponse:
        return CancelResponse.ACCEPT

    @staticmethod
    def _result(error_code: int, error_string: str) -> FollowJointTrajectory.Result:
        result = FollowJointTrajectory.Result()
        result.error_code = int(error_code)
        result.error_string = str(error_string or "")
        return result

    def _stop_motion(self) -> bool:  # noqa: C901, PLR0912
        if self.control is None:
            return False
        with self._active_lock:
            motion_kind = self._active_motion_kind
        if motion_kind == "cartesian_jog":
            jog_stop = getattr(self.control, "jogStop", None)
            if callable(jog_stop):
                try:
                    jog_stop()
                    return True
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass
        if motion_kind == "insert":
            return self._stop_insert_motion()
        method_names = (
            ("stopL", "stopJ", "servoStop", "stopScript")
            if motion_kind in {"cartesian", "relative_cartesian"}
            else ("stopJ", "stopL", "servoStop", "stopScript")
        )
        for method_name in method_names:
            method = getattr(self.control, method_name, None)
            if method is None:
                continue
            try:
                if method_name in {"stopJ", "stopL"}:
                    method(UR5E_RTDE_STOP_ACCEL_RAD_S2)
                else:
                    method()
                return True
            except TypeError:
                try:
                    method()
                    return True
                except Exception:
                    continue
            except Exception:
                continue
        return False

    def _stop_insert_motion(self) -> bool:
        """Stop every insertion command and require affirmative acknowledgements."""
        if self.control is None:
            self._insert_force_mode_stop_acknowledged = False
            self._insert_servo_stop_acknowledged = False
            self._insert_stop_l_command_completed = False
            return False
        stop_ok = True
        force_mode_was_active = bool(
            getattr(self, "_insert_force_mode_active", False)
        )
        force_mode_ok = not force_mode_was_active
        if force_mode_was_active:
            force_mode_stop = getattr(self.control, "forceModeStop", None)
            try:
                force_mode_ok = bool(callable(force_mode_stop) and force_mode_stop())
            except (OSError, RuntimeError, TypeError, ValueError):
                force_mode_ok = False
            stop_ok = stop_ok and force_mode_ok
            if force_mode_ok:
                self._insert_force_mode_active = False
                self._insert_force_mode_command = None
        self._insert_force_mode_stop_acknowledged = force_mode_ok
        servo_was_active = bool(getattr(self, "_insert_servo_active", False))
        servo_ok = not servo_was_active
        if servo_was_active:
            servo_stop = getattr(self.control, "servoStop", None)
            try:
                servo_ok = bool(callable(servo_stop) and servo_stop())
            except (OSError, RuntimeError, TypeError, ValueError):
                servo_ok = False
            stop_ok = stop_ok and servo_ok
            if servo_ok:
                self._insert_servo_active = False
        self._insert_servo_stop_acknowledged = servo_ok
        stop_l = getattr(self.control, "stopL", None)
        try:
            stop_l_result = (
                stop_l(UR5E_RTDE_STOP_ACCEL_RAD_S2) if callable(stop_l) else False
            )
            # ur_rtde versions returning None still accepted and executed stopL.
            stop_l_ok = stop_l_result is not False
        except TypeError:
            try:
                stop_l_result = stop_l() if callable(stop_l) else False
                stop_l_ok = stop_l_result is not False
            except (OSError, RuntimeError, TypeError, ValueError):
                stop_l_ok = False
        except (OSError, RuntimeError, ValueError):
            stop_l_ok = False
        self._insert_stop_l_command_completed = stop_l_ok
        return stop_ok and stop_l_ok

    def _confirm_stationary_after_stop(
        self,
        timeout_sec: float = 2.0,
        *,
        linear_speed_limit_m_s: float | None = None,
        angular_speed_limit_rad_s: float | None = None,
    ) -> bool:
        """Require fresh joint and TCP stationary evidence after a stop command."""
        linear_limit = float(
            linear_speed_limit_m_s
            if linear_speed_limit_m_s is not None
            else UR5E_RTDE_INSERT_DEMONSTRATION_STATIONARY_SPEED_M_S
        )
        angular_limit = float(
            angular_speed_limit_rad_s
            if angular_speed_limit_rad_s is not None
            else UR5E_RTDE_INSERT_DEMONSTRATION_STATIONARY_ANGULAR_SPEED_RAD_S
        )
        if (
            not math.isfinite(linear_limit)
            or linear_limit <= 0.0
            or not math.isfinite(angular_limit)
            or angular_limit <= 0.0
        ):
            return False
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        stationary_since: float | None = None
        last_feedback_timestamp: float | None = None
        while time.monotonic() < deadline:
            feedback_timestamp = self._read_feedback_timestamp()
            actual_qd = self._read_actual_qd()
            actual_tcp_speed = self._read_actual_tcp_speed()
            now = time.monotonic()
            feedback_advanced = bool(
                feedback_timestamp is not None
                and (
                    last_feedback_timestamp is None
                    or feedback_timestamp > last_feedback_timestamp
                )
            )
            if feedback_timestamp is not None:
                last_feedback_timestamp = feedback_timestamp
            joint_stationary = bool(
                actual_qd is not None
                and len(actual_qd) == 6
                and all(math.isfinite(float(value)) for value in actual_qd)
                and max(abs(float(value)) for value in actual_qd)
                <= UR5E_RTDE_STATIONARY_MAX_JOINT_VEL_RAD_S
            )
            tcp_stationary = bool(
                actual_tcp_speed is not None
                and len(actual_tcp_speed) == 6
                and all(math.isfinite(float(value)) for value in actual_tcp_speed)
                and _vector_norm(tuple(float(value) for value in actual_tcp_speed[:3]))
                <= linear_limit
                and _vector_norm(tuple(float(value) for value in actual_tcp_speed[3:6]))
                <= angular_limit
            )
            if feedback_advanced and joint_stationary and tcp_stationary:
                stationary_since = stationary_since or now
                if now - stationary_since >= UR5E_RTDE_STATIONARY_HOLD_SEC:
                    return True
            else:
                stationary_since = None
            time.sleep(0.02)
        return False

    def _ensure_control_program_for_goal(self) -> str | None:
        """Ensure the RTDE control script is running before dispatching a goal."""
        control = self.control
        is_program_running = getattr(control, "isProgramRunning", None)
        if not callable(is_program_running):
            return None
        try:
            if bool(is_program_running()):
                return None
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return f"UR5e RTDE control program readiness failed: {type(exc).__name__}: {exc}"

        reupload_script = getattr(control, "reuploadScript", None)
        if not callable(reupload_script):
            return "UR5e RTDE control program is not running and cannot be reuploaded"
        try:
            reuploaded = bool(reupload_script())
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return f"UR5e RTDE control program reupload failed: {type(exc).__name__}: {exc}"
        if not reuploaded:
            return "UR5e RTDE control program reupload returned False"

        deadline = time.monotonic() + UR5E_RTDE_CONTROL_PROGRAM_START_TIMEOUT_SEC
        while time.monotonic() < deadline:
            try:
                if bool(is_program_running()):
                    return None
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                return (
                    "UR5e RTDE control program readiness failed after reupload: "
                    f"{type(exc).__name__}: {exc}"
                )
            time.sleep(0.02)
        return (
            "UR5e RTDE control program did not start within "
            f"{UR5E_RTDE_CONTROL_PROGRAM_START_TIMEOUT_SEC:.2f} s after reupload"
        )

    def _async_operation_status(self) -> dict[str, Any]:
        """Read serializable RTDE asynchronous-operation progress diagnostics."""
        status: dict[str, Any] = {
            "rtde_async_operation_supported": False,
            "rtde_async_operation_running": None,
            "rtde_async_operation_progress": None,
            "rtde_async_operation_id": None,
            "rtde_async_operation_change_count": None,
            "rtde_async_operation_value": None,
        }
        control = self.control
        get_extended = getattr(control, "getAsyncOperationProgressEx", None)
        if callable(get_extended):
            try:
                operation = get_extended()
                status.update(
                    rtde_async_operation_supported=True,
                    rtde_async_operation_running=bool(operation.isAsyncOperationRunning()),
                    rtde_async_operation_progress=int(operation.progress()),
                    rtde_async_operation_id=int(operation.operationId()),
                    rtde_async_operation_change_count=int(operation.changeCount()),
                    rtde_async_operation_value=int(operation.value()),
                )
                return status
            except (OSError, RuntimeError, TypeError, ValueError, AttributeError):
                pass

        get_legacy = getattr(control, "getAsyncOperationProgress", None)
        if callable(get_legacy):
            try:
                value = int(get_legacy())
            except (OSError, RuntimeError, TypeError, ValueError):
                return status
            status.update(
                rtde_async_operation_supported=True,
                rtde_async_operation_running=value >= 0,
                rtde_async_operation_progress=value if value >= 0 else None,
                rtde_async_operation_value=value,
            )
        return status

    def _clear_active_goal(self, goal_handle: Any) -> None:
        with self._active_lock:
            if self._active_goal is goal_handle:
                self._insert_motion_started = False
                self._active_goal = None
                self._active_goal_status = None
                self._active_motion_kind = ""

    def _execute_movej_path(self, path: list[list[float]]) -> tuple[str, str]:
        if self.control is None:
            raise RuntimeError("RTDE control connection is not available")
        movej = getattr(self.control, "moveJ", None)
        if movej is None:
            raise RuntimeError("RTDE control object has no moveJ method")
        type_errors: list[str] = []
        try:
            result = movej(path, True)
            return repr(result), "asynchronous_positional"
        except TypeError as exc:
            type_errors.append(f"positional_async={exc}")
        try:
            result = movej(path, asynchronous=True)
            return repr(result), "asynchronous_keyword"
        except TypeError as exc:
            type_errors.append(f"keyword_async={exc}")
        result = movej(path)
        suffix = "; ".join(type_errors)
        return f"{repr(result)}; blocking_fallback=True; {suffix}", "blocking_fallback"

    def _execute_movej_target(
        self,
        target: list[float],
        *,
        speed_rad_s: float,
        acceleration_rad_s2: float,
    ) -> bool:
        """Dispatch one asynchronous RTDE moveJ target with explicit limits."""
        control = self.control
        movej = getattr(control, "moveJ", None)
        if not callable(movej):
            raise RuntimeError("RTDE control object has no moveJ method")
        try:
            return bool(movej(target, speed_rad_s, acceleration_rad_s2, True))
        except TypeError:
            return bool(
                movej(
                    target,
                    speed=speed_rad_s,
                    acceleration=acceleration_rad_s2,
                    asynchronous=True,
                )
            )

    @staticmethod
    def _joint_jog_result(
        error_code: int,
        error_string: str,
        *,
        final_joint_error_rad: float = math.inf,
        state_uncertain: bool = False,
    ) -> Any:
        if MoveUR5eJointJog is None:
            return None
        result = MoveUR5eJointJog.Result()
        result.error_code = int(error_code)
        result.error_string = str(error_string or "")
        result.final_joint_error_rad = float(final_joint_error_rad)
        result.state_uncertain = bool(state_uncertain)
        return result

    @staticmethod
    def _publish_joint_jog_feedback(goal_handle: Any, joint_error_rad: float) -> None:
        if MoveUR5eJointJog is None:
            return
        feedback = MoveUR5eJointJog.Feedback()
        feedback.joint_error_rad = float(joint_error_rad)
        goal_handle.publish_feedback(feedback)

    def _execute_joint_jog(self, goal_handle: Any) -> Any:  # noqa: C901, PLR0912
        """Execute one guarded Interactive Teleop UR5e joint jog."""
        with self._active_lock:
            if self._shutdown_requested:
                goal_handle.abort()
                return self._joint_jog_result(-1, "UR5e RTDE server is stopping")
            if self._rtde_reset_required:
                goal_handle.abort()
                return self._joint_jog_result(
                    -1,
                    self._rtde_reset_reason or "UR5e RTDE reset required",
                    state_uncertain=True,
                )
            if self._latched_terminal_status is not None:
                reason = str(
                    self._latched_terminal_status.get("blocked_reason")
                    or self._latched_terminal_status.get("message")
                    or "UR5e RTDE trajectory server requires repair"
                )
                goal_handle.abort()
                return self._joint_jog_result(-1, reason, state_uncertain=True)
            if self._active_goal is not None:
                goal_handle.abort()
                return self._joint_jog_result(-1, "UR5e RTDE motion already executing")
            self._active_goal = goal_handle
            self._active_goal_status = None
            self._active_motion_kind = "joint_jog"

        status = _status_base()
        status.update(
            state="checking",
            message="checking guarded UR5e RTDE joint jog",
            motion_kind="joint_jog",
        )
        motion_attempted = False
        final_error = math.inf
        try:
            request = goal_handle.request
            joint = int(request.joint)
            delta_rad = float(request.delta_rad)
            speed_rad_s = float(request.speed_rad_s)
            acceleration_rad_s2 = float(request.acceleration_rad_s2)
            if joint < 1 or joint > len(ARM_JOINTS):
                raise ValueError(f"joint must be within 1..{len(ARM_JOINTS)}")
            if not math.isfinite(delta_rad) or not (
                0.0 < abs(delta_rad) <= UR5E_RTDE_JOINT_JOG_MAX_DELTA_RAD
            ):
                raise ValueError(
                    "joint delta must be finite and within "
                    f"(0, {math.degrees(UR5E_RTDE_JOINT_JOG_MAX_DELTA_RAD):.1f}] deg"
                )
            if not math.isfinite(speed_rad_s) or not (
                0.0 < speed_rad_s <= UR5E_RTDE_MAX_JOINT_VEL_RAD_S
            ):
                raise ValueError("joint speed is outside the configured limit")
            if not math.isfinite(acceleration_rad_s2) or not (
                0.0 < acceleration_rad_s2 <= UR5E_RTDE_MAX_JOINT_ACCEL_RAD_S2
            ):
                raise ValueError("joint acceleration is outside the configured limit")

            control_error = self._connect_control_for_goal()
            if control_error:
                raise RuntimeError(control_error)
            actual = self._read_actual_q()
            if actual is None or not self._joint_states_fresh():
                raise RuntimeError("UR5e RTDE feedback stale or missing")
            program_error = self._ensure_control_program_for_goal()
            if program_error:
                raise RuntimeError(program_error)

            target = [float(value) for value in actual]
            target[joint - 1] += delta_rad
            within_limits = getattr(self.control, "isJointsWithinSafetyLimits", None)
            if not callable(within_limits):
                raise RuntimeError("RTDE control object has no isJointsWithinSafetyLimits method")
            if not bool(within_limits(target)):
                raise ValueError(f"UR controller rejected J{joint} target as outside safety limits")

            initial = [float(value) for value in actual]
            initial_target_error = abs(target[joint - 1] - initial[joint - 1])
            timeout_sec = max(
                4.0,
                initial_target_error / speed_rad_s + UR5E_RTDE_RESULT_MARGIN_SEC,
            )
            execution_started = time.monotonic()
            status.update(
                state="executing",
                message=f"executing guarded UR5e RTDE J{joint} jog",
                blocked_reason="",
                rtde_connected=True,
                rtde_receive_connected=True,
                rtde_control_connected=True,
                joint_states_fresh=True,
                joint_jog_joint=joint,
                joint_jog_delta_rad=delta_rad,
                joint_jog_speed_rad_s=speed_rad_s,
                joint_jog_acceleration_rad_s2=acceleration_rad_s2,
                initial_positions_rad=initial,
                final_target_positions_rad=target,
                trajectory_result_timeout_sec=timeout_sec,
            )
            self._write_active_goal_status(goal_handle, status)
            motion_attempted = True
            if not self._execute_movej_target(
                target,
                speed_rad_s=speed_rad_s,
                acceleration_rad_s2=acceleration_rad_s2,
            ):
                raise RuntimeError("UR5e RTDE moveJ returned False")

            stationary_since: float | None = None
            previous_actual = list(initial)
            previous_actual_at = execution_started
            deadline = execution_started + timeout_sec
            while rclpy.ok() and time.monotonic() < deadline:
                if goal_handle.is_cancel_requested:
                    self._stop_motion()
                    stop_deadline = time.monotonic() + 2.0
                    stop_stationary_since: float | None = None
                    stop_confirmed = False
                    while time.monotonic() < stop_deadline:
                        actual_qd = self._read_actual_qd()
                        if actual_qd is None:
                            stop_stationary_since = None
                        elif max(abs(float(value)) for value in actual_qd) <= (
                            UR5E_RTDE_STATIONARY_MAX_JOINT_VEL_RAD_S
                        ):
                            stop_stationary_since = stop_stationary_since or time.monotonic()
                            if (
                                time.monotonic() - stop_stationary_since
                                >= UR5E_RTDE_STATIONARY_HOLD_SEC
                            ):
                                stop_confirmed = True
                                break
                        else:
                            stop_stationary_since = None
                        time.sleep(0.02)
                    if not stop_confirmed:
                        self._mark_rtde_reset_required(
                            "UR5e joint jog cancellation did not confirm stationary motion",
                            write_status=False,
                        )
                    goal_handle.canceled()
                    status.update(
                        state="canceled",
                        message=(
                            "UR5e joint jog canceled"
                            if stop_confirmed
                            else "UR5e joint jog canceled without stationary confirmation"
                        ),
                        blocked_reason=(
                            ""
                            if stop_confirmed
                            else "UR5e joint jog cancellation did not confirm stationary motion"
                        ),
                        rtde_reset_required=not stop_confirmed,
                    )
                    self._finish_active_goal_status(
                        goal_handle,
                        status,
                        latch_status=not stop_confirmed,
                    )
                    return self._joint_jog_result(
                        -1,
                        "canceled",
                        final_joint_error_rad=final_error,
                        state_uncertain=not stop_confirmed,
                    )
                actual = self._read_actual_q()
                actual_at = time.monotonic()
                if actual is None:
                    raise RuntimeError("UR5e RTDE joint jog feedback is unavailable")
                final_error = abs(target[joint - 1] - actual[joint - 1])
                self._publish_joint_jog_feedback(goal_handle, final_error)
                actual_qd = self._read_actual_qd()
                if actual_qd is not None:
                    max_velocity = max(abs(float(value)) for value in actual_qd)
                else:
                    elapsed = actual_at - previous_actual_at
                    max_velocity = (
                        max(
                            abs(current - previous) / elapsed
                            for current, previous in zip(
                                actual,
                                previous_actual,
                                strict=True,
                            )
                        )
                        if elapsed > 1e-6
                        else math.inf
                    )
                target_reached = final_error <= UR5E_RTDE_JOINT_JOG_TOLERANCE_RAD
                stationary = max_velocity <= UR5E_RTDE_STATIONARY_MAX_JOINT_VEL_RAD_S
                if target_reached and stationary:
                    stationary_since = stationary_since or actual_at
                else:
                    stationary_since = None
                status.update(
                    actual_positions_rad=[float(value) for value in actual],
                    final_joint_error_rad=final_error,
                    max_actual_joint_velocity_rad_s=max_velocity,
                    stationary_hold_sec=(
                        actual_at - stationary_since if stationary_since is not None else 0.0
                    ),
                    trajectory_elapsed_sec=actual_at - execution_started,
                )
                with self._active_lock:
                    if self._active_goal is goal_handle:
                        self._active_goal_status = dict(status)
                if (
                    stationary_since is not None
                    and actual_at - stationary_since >= UR5E_RTDE_STATIONARY_HOLD_SEC
                ):
                    status.update(
                        state="succeeded",
                        message=f"UR5e J{joint} jog reached its target",
                    )
                    goal_handle.succeed()
                    self._finish_active_goal_status(goal_handle, status)
                    return self._joint_jog_result(
                        0,
                        "",
                        final_joint_error_rad=final_error,
                    )
                previous_actual = list(actual)
                previous_actual_at = actual_at
                time.sleep(0.02)

            self._stop_motion()
            reason = "UR5e RTDE joint jog result timeout"
            status.update(state="failed", message=reason, blocked_reason=reason)
            goal_handle.abort()
            self._finish_active_goal_status(goal_handle, status, latch_status=True)
            return self._joint_jog_result(
                -1,
                reason,
                final_joint_error_rad=final_error,
                state_uncertain=True,
            )
        except ValueError as exc:
            reason = f"UR5e RTDE joint jog rejected: {exc}"
            status.update(state="blocked", message=reason, blocked_reason=reason)
            goal_handle.abort()
            self._finish_active_goal_status(goal_handle, status)
            return self._joint_jog_result(
                -1,
                reason,
                final_joint_error_rad=final_error,
            )
        except (LookupError, RuntimeError, TransformException) as exc:
            reason = f"UR5e RTDE joint jog failed: {type(exc).__name__}: {exc}"
            if motion_attempted:
                self._stop_motion()
                self._mark_rtde_reset_required(reason, write_status=False)
            status.update(
                state="failed",
                message=reason,
                blocked_reason=reason,
                rtde_reset_required=motion_attempted,
            )
            goal_handle.abort()
            self._finish_active_goal_status(
                goal_handle,
                status,
                latch_status=motion_attempted,
            )
            return self._joint_jog_result(
                -4,
                reason,
                final_joint_error_rad=final_error,
                state_uncertain=motion_attempted,
            )
        finally:
            self._clear_active_goal(goal_handle)

    def _insert_result(  # noqa: PLR0913
        self,
        error_code: int,
        error_string: str,
        *,
        trial_id: str = "",
        hard_caps_sha256: str = "",
        state_uncertain: bool = False,
        motion_settled: bool = True,
        final_world_tool0: RigidTransform | None = None,
        final_insertion_depth_m: float = math.inf,
        final_depth_error_m: float = math.inf,
        final_lateral_offset_m: float = math.inf,
        final_tilt_error_rad: float = math.inf,
        final_search_radius_m: float = 0.0,
        peak_axial_force_n: float = 0.0,
        peak_lateral_force_n: float = 0.0,
        peak_torque_nm: float = 0.0,
        peak_filtered_axial_force_n: float = 0.0,
        peak_filtered_lateral_force_n: float = 0.0,
        peak_filtered_torque_nm: float = 0.0,
        peak_tool_flange_torque_nm: float = 0.0,
        contact_detected: bool = False,
        engagement_detected: bool = False,
        seated_detected: bool = False,
        force_bias: list[float] | None = None,
        final_phase: str = "",
        soft_overload_detected: bool = False,
        soft_overload_recovered: bool = False,
        relief_exhausted: bool = False,
        relief_cycle_count: int = 0,
        last_soft_overload_reason: str = "",
        relief_load_cleared: bool = False,
        relief_backoff_m: float = 0.0,
        relief_planned_backoff_m: float = 0.0,
        total_relief_backoff_m: float = 0.0,
        relief_resume_phase: str = "",
        relief_force_mode_stop_acknowledged: bool = False,
        relief_stop_l_command_completed: bool = False,
        relief_stationary_confirmed: bool = False,
        relief_force_mode_restart_acknowledged: bool = False,
        hard_limit_detected: bool = False,
        hard_limit_reason: str = "",
        limit_trigger: str = "",
        limit_trigger_value: float = 0.0,
        limit_trigger_threshold: float = 0.0,
        limit_trigger_actual_tcp_force: list[float] | None = None,
        limit_trigger_tared_tcp_force: list[float] | None = None,
        force_mode_stop_acknowledged: bool = False,
        servo_stop_acknowledged: bool = False,
        stop_l_command_completed: bool = False,
        stationary_confirmed: bool = False,
        server_trace_id: str = "",
        server_trace_path: str = "",
        server_trace_sha256: str = "",
        server_trace_status: str = "not_started",
        server_trace_complete: bool = False,
        server_trace_sample_count: int = 0,
        tactile_center_world_tool0: RigidTransform | None = None,
        tactile_center_depth_m: float = 0.0,
        tactile_center_confidence: float = 0.0,
        tactile_center_evidence_sha256: str = "",
        scheduled_search_radius_m: float = 0.0,
        explored_search_radius_m: float = 0.0,
        explored_search_angle_rad: float = 0.0,
        disengagement_cycle_count: int = 0,
        last_disengagement_reason: str = "",
        disengagement_withdrawal_m: float = 0.0,
        disengagement_contact_cleared: bool = False,
        disengagement_force_mode_stop_acknowledged: bool = False,
        recenter_position_error_m: float = math.inf,
        recenter_command_acknowledged: bool = False,
        disengagement_stationary_confirmed: bool = False,
        retare_baseline_consistent: bool = False,
    ) -> Any:
        if MoveUR5eInsert is None:
            return None
        result = MoveUR5eInsert.Result()
        result.error_code = int(error_code)
        result.error_string = str(error_string or "")
        result.trial_id = str(trial_id or "")
        result.hard_caps_sha256 = str(hard_caps_sha256 or "")
        result.state_uncertain = bool(state_uncertain)
        result.motion_settled = bool(motion_settled)
        result.final_tool0_pose_valid = final_world_tool0 is not None
        result.final_tool0_pose = (
            self._pose_stamped_from_transform(final_world_tool0)
            if final_world_tool0 is not None
            else PoseStamped()
        )
        result.final_insertion_depth_m = float(final_insertion_depth_m)
        result.final_depth_error_m = float(final_depth_error_m)
        result.final_lateral_offset_m = float(final_lateral_offset_m)
        result.final_tilt_error_rad = float(final_tilt_error_rad)
        result.final_search_radius_m = float(final_search_radius_m)
        result.peak_axial_force_n = float(peak_axial_force_n)
        result.peak_lateral_force_n = float(peak_lateral_force_n)
        result.peak_torque_nm = float(peak_torque_nm)
        result.peak_filtered_axial_force_n = float(peak_filtered_axial_force_n)
        result.peak_filtered_lateral_force_n = float(peak_filtered_lateral_force_n)
        result.peak_filtered_torque_nm = float(peak_filtered_torque_nm)
        result.peak_tool_flange_torque_nm = float(peak_tool_flange_torque_nm)
        result.contact_detected = bool(contact_detected)
        result.engagement_detected = bool(engagement_detected)
        result.seated_detected = bool(seated_detected)
        result.force_bias_valid = bool(
            force_bias is not None
            and len(force_bias) == 6
            and all(math.isfinite(float(value)) for value in force_bias)
        )
        result.force_bias = (
            [float(value) for value in force_bias]
            if result.force_bias_valid
            else [0.0] * 6
        )
        result.final_phase = str(final_phase or "")
        result.soft_overload_detected = bool(soft_overload_detected)
        result.soft_overload_recovered = bool(soft_overload_recovered)
        result.relief_exhausted = bool(relief_exhausted)
        result.relief_cycle_count = int(relief_cycle_count)
        result.last_soft_overload_reason = str(last_soft_overload_reason or "")
        result.relief_load_cleared = bool(relief_load_cleared)
        result.relief_backoff_m = float(relief_backoff_m)
        result.relief_planned_backoff_m = float(relief_planned_backoff_m)
        result.total_relief_backoff_m = float(total_relief_backoff_m)
        result.relief_resume_phase = str(relief_resume_phase or "")
        result.relief_force_mode_stop_acknowledged = bool(
            relief_force_mode_stop_acknowledged
        )
        result.relief_stop_l_command_completed = bool(
            relief_stop_l_command_completed
        )
        result.relief_stationary_confirmed = bool(relief_stationary_confirmed)
        result.relief_force_mode_restart_acknowledged = bool(
            relief_force_mode_restart_acknowledged
        )
        result.hard_limit_detected = bool(hard_limit_detected)
        result.hard_limit_reason = str(hard_limit_reason or "")
        result.limit_trigger = str(limit_trigger or "")
        result.limit_trigger_value = float(limit_trigger_value)
        result.limit_trigger_threshold = float(limit_trigger_threshold)
        result.limit_trigger_actual_tcp_force = [
            float(value)
            for value in (limit_trigger_actual_tcp_force or [0.0] * 6)
        ]
        result.limit_trigger_tared_tcp_force = [
            float(value)
            for value in (limit_trigger_tared_tcp_force or [0.0] * 6)
        ]
        result.force_mode_stop_acknowledged = bool(force_mode_stop_acknowledged)
        result.servo_stop_acknowledged = bool(servo_stop_acknowledged)
        result.stop_l_command_completed = bool(stop_l_command_completed)
        result.stationary_confirmed = bool(stationary_confirmed)
        result.server_trace_id = str(server_trace_id or "")
        result.server_trace_path = str(server_trace_path or "")
        result.server_trace_sha256 = str(server_trace_sha256 or "")
        result.server_trace_status = str(server_trace_status or "not_started")
        result.server_trace_complete = bool(server_trace_complete)
        result.server_trace_sample_count = int(server_trace_sample_count)
        result.tactile_center_valid = tactile_center_world_tool0 is not None
        result.tactile_center_tool0_pose = (
            self._pose_stamped_from_transform(tactile_center_world_tool0)
            if tactile_center_world_tool0 is not None
            else PoseStamped()
        )
        result.tactile_center_depth_m = float(tactile_center_depth_m)
        result.tactile_center_confidence = float(tactile_center_confidence)
        result.tactile_center_evidence_sha256 = str(
            tactile_center_evidence_sha256 or ""
        )
        result.scheduled_search_radius_m = float(scheduled_search_radius_m)
        result.explored_search_radius_m = float(explored_search_radius_m)
        result.explored_search_angle_rad = float(explored_search_angle_rad)
        result.disengagement_cycle_count = int(disengagement_cycle_count)
        result.last_disengagement_reason = str(last_disengagement_reason or "")
        result.disengagement_withdrawal_m = float(disengagement_withdrawal_m)
        result.disengagement_contact_cleared = bool(
            disengagement_contact_cleared
        )
        result.disengagement_force_mode_stop_acknowledged = bool(
            disengagement_force_mode_stop_acknowledged
        )
        result.recenter_position_error_m = float(recenter_position_error_m)
        result.recenter_command_acknowledged = bool(
            recenter_command_acknowledged
        )
        result.disengagement_stationary_confirmed = bool(
            disengagement_stationary_confirmed
        )
        result.retare_baseline_consistent = bool(retare_baseline_consistent)
        return result

    @staticmethod
    def _insertion_demonstration_result(
        error_code: int,
        error_string: str,
        *,
        state_uncertain: bool,
        motion_settled: bool,
        recording_id: str,
        trace_path: str = "",
        trace_sha256: str = "",
        sample_count: int = 0,
        started_at: float = 0.0,
        finished_at: float = 0.0,
        force_bias: list[float] | None = None,
        baseline_force_span_n: float = 0.0,
        baseline_torque_span_nm: float = 0.0,
    ) -> Any:
        result = RecordUR5eInsertionDemonstration.Result()
        result.error_code = int(error_code)
        result.error_string = str(error_string or "")
        result.state_uncertain = bool(state_uncertain)
        result.motion_settled = bool(motion_settled)
        result.recording_id = str(recording_id or "")
        result.trace_path = str(trace_path or "")
        result.trace_sha256 = str(trace_sha256 or "")
        result.sample_count = int(sample_count)
        result.started_at = float(started_at)
        result.finished_at = float(finished_at)
        result.baseline_valid = force_bias is not None
        result.force_bias = [float(value) for value in (force_bias or [0.0] * 6)]
        result.baseline_force_span_n = float(baseline_force_span_n)
        result.baseline_torque_span_nm = float(baseline_torque_span_nm)
        return result

    def _publish_insertion_demonstration_feedback(
        self,
        goal_handle: Any,
        *,
        phase: str,
        sample_count: int,
        elapsed_sec: float,
        actual_world_tool0: RigidTransform,
        actual_tcp_force: list[float],
        actual_tcp_speed: list[float],
        force_bias: list[float] | None,
    ) -> None:
        feedback = RecordUR5eInsertionDemonstration.Feedback()
        feedback.phase = str(phase)
        feedback.sample_count = int(sample_count)
        feedback.elapsed_sec = float(elapsed_sec)
        feedback.actual_tool0_pose = self._pose_stamped_from_transform(
            actual_world_tool0
        )
        feedback.actual_tcp_force = [float(value) for value in actual_tcp_force]
        feedback.actual_tcp_speed = [float(value) for value in actual_tcp_speed]
        feedback.baseline_valid = force_bias is not None
        feedback.force_bias = [float(value) for value in (force_bias or [0.0] * 6)]
        goal_handle.publish_feedback(feedback)

    def _execute_insertion_demonstration(self, goal_handle: Any) -> Any:  # noqa: C901, PLR0912, PLR0915
        """Record synchronized UR5e feedback while another action owns jog motion."""
        request = goal_handle.request
        recording_id = str(getattr(request, "recording_id", "") or "")
        part_name = str(getattr(request, "part_name", "") or "")
        destination_location = str(
            getattr(request, "destination_location", "") or ""
        )
        context_sha256 = str(getattr(request, "context_sha256", "") or "")
        started_at = time.time()

        def terminal(
            code: int,
            message: str,
            *,
            state_uncertain: bool = False,
            trace_path: str = "",
            trace_sha256: str = "",
            sample_count: int = 0,
            force_bias: list[float] | None = None,
            baseline_force_span_n: float = 0.0,
            baseline_torque_span_nm: float = 0.0,
        ) -> Any:
            motion_settled = self._confirm_stationary_after_stop(timeout_sec=0.5)
            return self._insertion_demonstration_result(
                code,
                message,
                state_uncertain=state_uncertain or not motion_settled,
                motion_settled=motion_settled,
                recording_id=recording_id,
                trace_path=trace_path,
                trace_sha256=trace_sha256,
                sample_count=sample_count,
                started_at=started_at,
                finished_at=time.time(),
                force_bias=force_bias,
                baseline_force_span_n=baseline_force_span_n,
                baseline_torque_span_nm=baseline_torque_span_nm,
            )

        valid_recording_id = bool(
            recording_id
            and len(recording_id) <= 128
            and all(character.isalnum() or character in {"-", "_"} for character in recording_id)
        )
        try:
            valid_context_hash = len(context_sha256) == 64 and int(
                context_sha256, 16
            ) >= 0
        except ValueError:
            valid_context_hash = False
        if (
            not valid_recording_id
            or part_name not in INSERT_SUPPORTED_PART_NAMES
            or destination_location != "assembly_board-v1"
            or not valid_context_hash
        ):
            goal_handle.abort()
            return terminal(-2, "insertion demonstration identity is invalid")
        try:
            expected_start = _transform_from_pose_stamped(
                request.expected_start_tool0_pose
            )
            maximum_duration_sec = float(request.max_duration_sec)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            goal_handle.abort()
            return terminal(-2, f"insertion demonstration goal is invalid: {exc}")
        if (
            str(request.expected_start_tool0_pose.header.frame_id or "") != "world"
            or not math.isfinite(maximum_duration_sec)
            or maximum_duration_sec <= 0.0
            or maximum_duration_sec
            > UR5E_RTDE_INSERT_DEMONSTRATION_MAX_DURATION_SEC
        ):
            goal_handle.abort()
            return terminal(-2, "insertion demonstration duration or start frame is invalid")
        if self.monitor_only or self.receive is None or self.control is None:
            goal_handle.abort()
            return terminal(-1, "UR5e trajectory control and receive feedback are required")
        if bool(getattr(self, "_rtde_reset_required", False)):
            goal_handle.abort()
            return terminal(-1, str(getattr(self, "_rtde_reset_reason", "") or "RTDE reset required"))
        try:
            (
                world_base,
                frame_message,
                frame_position_error,
                frame_orientation_error,
            ) = self._validated_cartesian_world_base()
        except (RuntimeError, ValueError) as exc:
            reason = str(exc)
            status = _status_base()
            status.update(
                state="blocked",
                message=reason,
                blocked_reason=reason,
                insertion_demonstration_active=False,
            )
            self._write_status(status)
            goal_handle.abort()
            return terminal(-1, reason)
        with self._insertion_demonstration_lock:
            if self._active_insertion_demonstration_goal is not None:
                goal_handle.abort()
                return terminal(-1, "another insertion demonstration is already active")
            self._active_insertion_demonstration_goal = goal_handle
            self._active_insertion_demonstration_status = {
                "insertion_demonstration_active": True,
                "insertion_demonstration_recording_id": recording_id,
                "insertion_demonstration_phase": "recording_baseline",
                "insertion_demonstration_sample_count": 0,
                "cartesian_frame_validation_message": frame_message,
                "cartesian_frame_position_error_m": frame_position_error,
                "cartesian_frame_orientation_error_rad": frame_orientation_error,
            }

        trace_path = INSERT_DEMONSTRATION_TRACE_ROOT / f"{recording_id}.jsonl"
        trace_tmp = trace_path.with_name(f".{trace_path.name}.{os.getpid()}.tmp")
        sample_count = 0
        force_bias: list[float] | None = None
        baseline_force_span_n = 0.0
        baseline_torque_span_nm = 0.0
        baseline_samples: deque[tuple[float, list[float]]] = deque()
        last_feedback_timestamp: float | None = None
        last_publish_at = 0.0
        trace_error = ""
        phase = "recording_baseline"
        try:
            INSERT_DEMONSTRATION_TRACE_ROOT.mkdir(parents=True, exist_ok=True)
            tool0_tcp = self._active_tcp_offset()
            actual_base_tcp = self._read_actual_tcp_transform()
            if actual_base_tcp is None:
                raise RuntimeError("actual TCP pose is unavailable")
            actual_world_tool0 = self._world_tool0_from_actual_tcp(
                actual_base_tcp,
                world_base=world_base,
                tool0_tcp=tool0_tcp,
            )
            start_position_error, start_orientation_error = _pose_errors(
                actual_world_tool0,
                expected_start,
            )
            if (
                start_position_error > UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M
                or start_orientation_error
                > UR5E_RTDE_CARTESIAN_ORIENTATION_TOLERANCE_RAD
            ):
                raise ValueError(
                    "current world -> tool0 does not match the frozen pre-insertion pose"
                )
            execution_started = time.monotonic()
            deadline = execution_started + maximum_duration_sec
            with trace_tmp.open("w", encoding="utf-8") as trace_file:
                while rclpy.ok() and time.monotonic() < deadline:
                    if goal_handle.is_cancel_requested:
                        break
                    feedback_timestamp = self._read_feedback_timestamp()
                    if (
                        feedback_timestamp is None
                        or (
                            last_feedback_timestamp is not None
                            and feedback_timestamp <= last_feedback_timestamp
                        )
                    ):
                        time.sleep(max(1.0 / UR5E_RTDE_FREQUENCY_HZ, 0.002))
                        continue
                    last_feedback_timestamp = feedback_timestamp
                    actual_q = self._read_actual_q()
                    actual_base_tcp = self._read_actual_tcp_transform()
                    actual_tcp_force = self._read_actual_tcp_force()
                    actual_tcp_speed = self._read_actual_tcp_speed()
                    if any(
                        value is None
                        for value in (
                            actual_q,
                            actual_base_tcp,
                            actual_tcp_force,
                            actual_tcp_speed,
                        )
                    ):
                        raise RuntimeError("complete advancing RTDE feedback is unavailable")
                    actual_world_tool0 = self._world_tool0_from_actual_tcp(
                        actual_base_tcp,
                        world_base=world_base,
                        tool0_tcp=tool0_tcp,
                    )
                    elapsed_sec = time.monotonic() - execution_started
                    linear_speed_m_s = math.sqrt(
                        sum(float(value) ** 2 for value in actual_tcp_speed[:3])
                    )
                    angular_speed_rad_s = math.sqrt(
                        sum(float(value) ** 2 for value in actual_tcp_speed[3:])
                    )
                    now_monotonic = time.monotonic()
                    if force_bias is None:
                        if (
                            linear_speed_m_s
                            > UR5E_RTDE_INSERT_DEMONSTRATION_STATIONARY_SPEED_M_S
                            or angular_speed_rad_s
                            > UR5E_RTDE_INSERT_DEMONSTRATION_STATIONARY_ANGULAR_SPEED_RAD_S
                        ):
                            baseline_samples.clear()
                        else:
                            baseline_samples.append(
                                (now_monotonic, list(actual_tcp_force))
                            )
                            while (
                                baseline_samples
                                and now_monotonic - baseline_samples[0][0]
                                > UR5E_RTDE_INSERT_DEMONSTRATION_BASELINE_SEC
                            ):
                                baseline_samples.popleft()
                            if (
                                len(baseline_samples) >= 5
                                and baseline_samples[-1][0] - baseline_samples[0][0]
                                >= UR5E_RTDE_INSERT_DEMONSTRATION_BASELINE_SEC * 0.90
                            ):
                                values = [sample[1] for sample in baseline_samples]
                                force_bias = [
                                    sum(sample[index] for sample in values) / len(values)
                                    for index in range(6)
                                ]
                                baseline_force_span_n = _vector_norm(
                                    tuple(
                                        max(sample[index] for sample in values)
                                        - min(sample[index] for sample in values)
                                        for index in range(3)
                                    )
                                )
                                baseline_torque_span_nm = _vector_norm(
                                    tuple(
                                        max(sample[index] for sample in values)
                                        - min(sample[index] for sample in values)
                                        for index in range(3, 6)
                                    )
                                )
                                phase = "recording_insertion"
                    translation, rotation = actual_world_tool0
                    with self._active_lock:
                        active_motion_kind = str(self._active_motion_kind or "")
                        active_motion_status = dict(self._active_goal_status or {})
                    sample = {
                        "sample_index": sample_count,
                        "recorded_at": time.time(),
                        "elapsed_sec": elapsed_sec,
                        "rtde_timestamp_sec": float(feedback_timestamp),
                        "phase": phase,
                        "actual_q": [float(value) for value in actual_q],
                        "actual_base_tcp_pose": _rtde_pose_from_transform(actual_base_tcp),
                        "world_base_pose": {
                            "x": float(world_base[0][0]),
                            "y": float(world_base[0][1]),
                            "z": float(world_base[0][2]),
                            "qx": float(world_base[1][0]),
                            "qy": float(world_base[1][1]),
                            "qz": float(world_base[1][2]),
                            "qw": float(world_base[1][3]),
                        },
                        "tool0_tcp_pose": {
                            "x": float(tool0_tcp[0][0]),
                            "y": float(tool0_tcp[0][1]),
                            "z": float(tool0_tcp[0][2]),
                            "qx": float(tool0_tcp[1][0]),
                            "qy": float(tool0_tcp[1][1]),
                            "qz": float(tool0_tcp[1][2]),
                            "qw": float(tool0_tcp[1][3]),
                        },
                        "world_tool0_pose": {
                            "x": float(translation[0]),
                            "y": float(translation[1]),
                            "z": float(translation[2]),
                            "qx": float(rotation[0]),
                            "qy": float(rotation[1]),
                            "qz": float(rotation[2]),
                            "qw": float(rotation[3]),
                        },
                        "actual_tcp_force": [float(value) for value in actual_tcp_force],
                        "actual_tcp_speed": [float(value) for value in actual_tcp_speed],
                        "force_bias_valid": force_bias is not None,
                        "force_bias": list(force_bias) if force_bias is not None else None,
                        "active_motion_kind": active_motion_kind,
                        "active_motion_message": str(
                            active_motion_status.get("message") or ""
                        ),
                        "active_motion_status": active_motion_status,
                    }
                    trace_file.write(
                        json.dumps(sample, allow_nan=False, separators=(",", ":"))
                        + "\n"
                    )
                    sample_count += 1
                    if sample_count % 25 == 0:
                        trace_file.flush()
                    if now_monotonic - last_publish_at >= 0.10:
                        self._publish_insertion_demonstration_feedback(
                            goal_handle,
                            phase=phase,
                            sample_count=sample_count,
                            elapsed_sec=elapsed_sec,
                            actual_world_tool0=actual_world_tool0,
                            actual_tcp_force=actual_tcp_force,
                            actual_tcp_speed=actual_tcp_speed,
                            force_bias=force_bias,
                        )
                        last_publish_at = now_monotonic
                    with self._insertion_demonstration_lock:
                        self._active_insertion_demonstration_status = {
                            "insertion_demonstration_active": True,
                            "insertion_demonstration_recording_id": recording_id,
                            "insertion_demonstration_phase": phase,
                            "insertion_demonstration_sample_count": sample_count,
                        }
                    time.sleep(max(1.0 / UR5E_RTDE_FREQUENCY_HZ, 0.002))
                trace_file.flush()
                os.fsync(trace_file.fileno())
            if not goal_handle.is_cancel_requested:
                trace_error = (
                    "insertion demonstration exceeded its protected maximum duration"
                )
            trace_tmp.replace(trace_path)
            trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            if trace_error:
                goal_handle.abort()
                return terminal(
                    -3,
                    trace_error,
                    state_uncertain=True,
                    trace_path=str(trace_path),
                    trace_sha256=trace_sha256,
                    sample_count=sample_count,
                    force_bias=force_bias,
                    baseline_force_span_n=baseline_force_span_n,
                    baseline_torque_span_nm=baseline_torque_span_nm,
                )
            goal_handle.canceled()
            return terminal(
                0,
                "insertion demonstration recording stopped",
                trace_path=str(trace_path),
                trace_sha256=trace_sha256,
                sample_count=sample_count,
                force_bias=force_bias,
                baseline_force_span_n=baseline_force_span_n,
                baseline_torque_span_nm=baseline_torque_span_nm,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            with suppress(OSError):
                if trace_tmp.is_file():
                    trace_tmp.replace(trace_path)
            trace_sha256 = ""
            if trace_path.is_file():
                with suppress(OSError):
                    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            goal_handle.abort()
            return terminal(
                -3,
                f"insertion demonstration recording failed: {type(exc).__name__}: {exc}",
                state_uncertain=False,
                trace_path=str(trace_path) if trace_path.is_file() else "",
                trace_sha256=trace_sha256,
                sample_count=sample_count,
                force_bias=force_bias,
                baseline_force_span_n=baseline_force_span_n,
                baseline_torque_span_nm=baseline_torque_span_nm,
            )
        finally:
            with self._insertion_demonstration_lock:
                if self._active_insertion_demonstration_goal is goal_handle:
                    self._active_insertion_demonstration_goal = None
                    self._active_insertion_demonstration_status = {}

    def _publish_insert_feedback(  # noqa: PLR0913
        self,
        goal_handle: Any,
        *,
        phase: str,
        actual_world_tool0: RigidTransform,
        insertion_depth_m: float,
        depth_error_m: float,
        lateral_offset_m: float,
        search_radius_m: float,
        axial_force_n: float,
        raw_axial_force_n: float = 0.0,
        lateral_force_n: float,
        torque_nm: float,
        contact_detected: bool,
        engagement_detected: bool,
        seated_detected: bool,
        force_bias_valid: bool,
        force_bias: list[float],
        actual_tcp_force: list[float],
        actual_tcp_speed: list[float],
        trial_id: str = "",
        filtered_axial_force_n: float = 0.0,
        filtered_lateral_force_n: float = 0.0,
        filtered_torque_nm: float = 0.0,
        tool_flange_torque_nm: float = 0.0,
        filtered_tool_flange_torque_nm: float = 0.0,
        current_force_depth_fraction: float = 0.0,
        force_depth_axial_upper_n: float = 0.0,
        force_depth_lateral_upper_n: float = 0.0,
        force_depth_torque_upper_nm: float = 0.0,
        axial_profile_exceeded: bool = False,
        axial_progress_stalled: bool = False,
        tared_tcp_force: list[float] | None = None,
        soft_overload_detected: bool = False,
        soft_overload_reason: str = "",
        soft_overload_duration_sec: float = 0.0,
        relief_cycle_count: int = 0,
        relief_elapsed_sec: float = 0.0,
        relief_retreat_m: float = 0.0,
        relief_load_cleared: bool = False,
        relief_backoff_m: float = 0.0,
        relief_planned_backoff_m: float = 0.0,
        total_relief_backoff_m: float = 0.0,
        relief_resume_phase: str = "",
        commanded_axial_force_n: float = 0.0,
        commanded_lateral_force_x_n: float = 0.0,
        commanded_lateral_force_y_n: float = 0.0,
        hard_limit_detected: bool = False,
        hard_limit_reason: str = "",
        limit_trigger: str = "",
        limit_trigger_value: float = 0.0,
        limit_trigger_threshold: float = 0.0,
        limit_trigger_actual_tcp_force: list[float] | None = None,
        limit_trigger_tared_tcp_force: list[float] | None = None,
        tactile_center_world_tool0: RigidTransform | None = None,
        tactile_center_depth_m: float = 0.0,
        tactile_center_confidence: float = 0.0,
        tactile_center_evidence_sha256: str = "",
        scheduled_search_radius_m: float = 0.0,
        explored_search_radius_m: float = 0.0,
        explored_search_angle_rad: float = 0.0,
        disengagement_cycle_count: int = 0,
        last_disengagement_reason: str = "",
        disengagement_withdrawal_m: float = 0.0,
        disengagement_contact_cleared: bool = False,
        disengagement_force_mode_stop_acknowledged: bool = False,
        recenter_position_error_m: float = math.inf,
        recenter_command_acknowledged: bool = False,
        disengagement_stationary_confirmed: bool = False,
        retare_baseline_consistent: bool = False,
    ) -> None:
        if MoveUR5eInsert is None:
            return
        feedback = MoveUR5eInsert.Feedback()
        feedback.phase = str(phase)
        feedback.trial_id = str(trial_id or "")
        feedback.actual_tool0_pose = self._pose_stamped_from_transform(actual_world_tool0)
        feedback.insertion_depth_m = float(insertion_depth_m)
        feedback.depth_error_m = float(depth_error_m)
        feedback.lateral_offset_m = float(lateral_offset_m)
        feedback.search_radius_m = float(search_radius_m)
        feedback.axial_force_n = float(axial_force_n)
        feedback.raw_axial_force_n = float(raw_axial_force_n)
        feedback.lateral_force_n = float(lateral_force_n)
        feedback.torque_nm = float(torque_nm)
        feedback.filtered_axial_force_n = float(filtered_axial_force_n)
        feedback.filtered_lateral_force_n = float(filtered_lateral_force_n)
        feedback.filtered_torque_nm = float(filtered_torque_nm)
        feedback.tool_flange_torque_nm = float(tool_flange_torque_nm)
        feedback.filtered_tool_flange_torque_nm = float(
            filtered_tool_flange_torque_nm
        )
        feedback.current_force_depth_fraction = float(
            current_force_depth_fraction
        )
        feedback.force_depth_axial_upper_n = float(force_depth_axial_upper_n)
        feedback.force_depth_lateral_upper_n = float(
            force_depth_lateral_upper_n
        )
        feedback.force_depth_torque_upper_nm = float(
            force_depth_torque_upper_nm
        )
        feedback.axial_profile_exceeded = bool(axial_profile_exceeded)
        feedback.axial_progress_stalled = bool(axial_progress_stalled)
        feedback.contact_detected = bool(contact_detected)
        feedback.engagement_detected = bool(engagement_detected)
        feedback.seated_detected = bool(seated_detected)
        feedback.force_bias_valid = bool(force_bias_valid)
        feedback.force_bias = [float(value) for value in force_bias]
        feedback.actual_tcp_force = [float(value) for value in actual_tcp_force]
        feedback.tared_tcp_force = [
            float(value) for value in (tared_tcp_force or [0.0] * 6)
        ]
        feedback.actual_tcp_speed = [float(value) for value in actual_tcp_speed]
        feedback.soft_overload_detected = bool(soft_overload_detected)
        feedback.soft_overload_reason = str(soft_overload_reason or "")
        feedback.soft_overload_duration_sec = float(soft_overload_duration_sec)
        feedback.relief_cycle_count = int(relief_cycle_count)
        feedback.relief_elapsed_sec = float(relief_elapsed_sec)
        feedback.relief_retreat_m = float(relief_retreat_m)
        feedback.relief_load_cleared = bool(relief_load_cleared)
        feedback.relief_backoff_m = float(relief_backoff_m)
        feedback.relief_planned_backoff_m = float(relief_planned_backoff_m)
        feedback.total_relief_backoff_m = float(total_relief_backoff_m)
        feedback.relief_resume_phase = str(relief_resume_phase or "")
        feedback.commanded_axial_force_n = float(commanded_axial_force_n)
        feedback.commanded_lateral_force_x_n = float(commanded_lateral_force_x_n)
        feedback.commanded_lateral_force_y_n = float(commanded_lateral_force_y_n)
        feedback.tactile_center_valid = tactile_center_world_tool0 is not None
        feedback.tactile_center_tool0_pose = (
            self._pose_stamped_from_transform(tactile_center_world_tool0)
            if tactile_center_world_tool0 is not None
            else PoseStamped()
        )
        feedback.tactile_center_depth_m = float(tactile_center_depth_m)
        feedback.tactile_center_confidence = float(tactile_center_confidence)
        feedback.tactile_center_evidence_sha256 = str(
            tactile_center_evidence_sha256 or ""
        )
        feedback.scheduled_search_radius_m = float(scheduled_search_radius_m)
        feedback.explored_search_radius_m = float(explored_search_radius_m)
        feedback.explored_search_angle_rad = float(explored_search_angle_rad)
        feedback.disengagement_cycle_count = int(disengagement_cycle_count)
        feedback.last_disengagement_reason = str(last_disengagement_reason or "")
        feedback.disengagement_withdrawal_m = float(disengagement_withdrawal_m)
        feedback.disengagement_contact_cleared = bool(
            disengagement_contact_cleared
        )
        feedback.disengagement_force_mode_stop_acknowledged = bool(
            disengagement_force_mode_stop_acknowledged
        )
        feedback.recenter_position_error_m = float(recenter_position_error_m)
        feedback.recenter_command_acknowledged = bool(
            recenter_command_acknowledged
        )
        feedback.disengagement_stationary_confirmed = bool(
            disengagement_stationary_confirmed
        )
        feedback.retare_baseline_consistent = bool(retare_baseline_consistent)
        feedback.hard_limit_detected = bool(hard_limit_detected)
        feedback.hard_limit_reason = str(hard_limit_reason or "")
        feedback.limit_trigger = str(limit_trigger or "")
        feedback.limit_trigger_value = float(limit_trigger_value)
        feedback.limit_trigger_threshold = float(limit_trigger_threshold)
        feedback.limit_trigger_actual_tcp_force = [
            float(value)
            for value in (limit_trigger_actual_tcp_force or [0.0] * 6)
        ]
        feedback.limit_trigger_tared_tcp_force = [
            float(value)
            for value in (limit_trigger_tared_tcp_force or [0.0] * 6)
        ]
        goal_handle.publish_feedback(feedback)

    def _start_insert_force_mode(
        self,
        *,
        actual_base_tcp: RigidTransform,
        insertion_axis_base: Vector3,
        insertion_force_n: float,
        contact_speed_m_s: float,
        spiral_speed_m_s: float,
        tilt_tolerance_rad: float,
    ) -> None:
        """Start translationally compliant insertion from a stationary TCP."""
        force_mode = getattr(self.control, "forceMode", None)
        if not callable(force_mode):
            raise RuntimeError("RTDE control object has no forceMode method")
        x_axis, y_axis, z_axis = _insertion_basis(insertion_axis_base)
        task_frame: RigidTransform = (
            actual_base_tcp[0],
            _quaternion_from_basis(x_axis, y_axis, z_axis),
        )
        limits = [
            spiral_speed_m_s,
            spiral_speed_m_s,
            contact_speed_m_s,
            tilt_tolerance_rad,
            tilt_tolerance_rad,
            tilt_tolerance_rad,
        ]
        self._insert_force_mode_command = (
            _rtde_pose_from_transform(task_frame),
            [0, 0, 1, 0, 0, 0],
            [0.0, 0.0, insertion_force_n, 0.0, 0.0, 0.0],
            2,
            limits,
        )
        self._insert_force_mode_active = True
        accepted = force_mode(*self._insert_force_mode_command)
        if not bool(accepted):
            raise RuntimeError("UR5e RTDE forceMode returned False")

    def _refresh_insert_force_mode(
        self,
        *,
        lateral_force_x_n: float = 0.0,
        lateral_force_y_n: float = 0.0,
        axial_force_n: float | None = None,
        lateral_compliant: bool | None = None,
    ) -> None:
        """Refresh force mode with bounded lateral search forces in its task frame."""
        command = self._insert_force_mode_command
        force_mode = getattr(self.control, "forceMode", None)
        if command is None or not callable(force_mode):
            raise RuntimeError("UR5e RTDE insertion force mode is not active")
        task_frame, selection, wrench, force_type, limits = command
        refreshed_selection = list(selection)
        if lateral_compliant is not None:
            refreshed_selection[0] = int(lateral_compliant)
            refreshed_selection[1] = int(lateral_compliant)
        refreshed_wrench = list(wrench)
        refreshed_wrench[0] = float(lateral_force_x_n)
        refreshed_wrench[1] = float(lateral_force_y_n)
        if axial_force_n is not None:
            refreshed_wrench[2] = float(axial_force_n)
        refreshed_command = (
            task_frame,
            refreshed_selection,
            refreshed_wrench,
            force_type,
            limits,
        )
        self._insert_force_mode_command = refreshed_command
        if not bool(force_mode(*refreshed_command)):
            raise RuntimeError("UR5e RTDE forceMode refresh returned False")

    def _execute_insert_servo_pose(
        self,
        target_base_tcp: RigidTransform,
        *,
        speed_m_s: float,
        acceleration_m_s2: float,
        cycle_sec: float,
    ) -> None:
        servo_l = getattr(self.control, "servoL", None)
        if not callable(servo_l):
            raise RuntimeError("RTDE control object has no servoL method")
        self._insert_servo_active = True
        accepted = servo_l(
            _rtde_pose_from_transform(target_base_tcp),
            speed_m_s,
            acceleration_m_s2,
            cycle_sec,
            0.05,
            300.0,
        )
        if not bool(accepted):
            raise RuntimeError("UR5e RTDE servoL returned False")

    def _stop_insert_servo(self) -> bool:
        if not bool(getattr(self, "_insert_servo_active", False)):
            return True
        servo_stop = getattr(self.control, "servoStop", None)
        if not callable(servo_stop):
            return False
        try:
            stopped = bool(servo_stop())
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        if stopped:
            self._insert_servo_active = False
        return stopped

    def _execute_insert(  # noqa: C901, PLR0912, PLR0915 - guarded insertion lifecycle.
        self,
        goal_handle: Any,
    ) -> Any:
        """Execute one force-guarded insertion with bounded pre-engagement relief."""
        with self._active_lock:
            if bool(getattr(self, "_shutdown_requested", False)):
                goal_handle.abort()
                return self._insert_result(-1, "UR5e RTDE server is stopping")
            if bool(getattr(self, "_rtde_reset_required", False)):
                reason = str(getattr(self, "_rtde_reset_reason", "")) or (
                    "UR5e RTDE reset required"
                )
                goal_handle.abort()
                return self._insert_result(-1, reason, state_uncertain=True)
            if self._latched_terminal_status is not None:
                reason = str(
                    self._latched_terminal_status.get("blocked_reason")
                    or self._latched_terminal_status.get("message")
                    or "UR5e RTDE server requires repair"
                )
                goal_handle.abort()
                return self._insert_result(-1, reason, state_uncertain=True)
            cap_error = _insert_hard_cap_error()
            if cap_error:
                goal_handle.abort()
                return self._insert_result(-1, cap_error)
            if self._active_goal is not None:
                goal_handle.abort()
                return self._insert_result(-1, "UR5e RTDE motion already executing")
            self._active_goal = goal_handle
            self._active_goal_status = None
            self._active_motion_kind = "insert"
            self._insert_motion_started = False
            self._insert_force_mode_stop_acknowledged = False
            self._insert_servo_stop_acknowledged = False
            self._insert_stop_l_command_completed = False

        status = _status_base()
        status.update(
            state="checking",
            message="checking guarded UR5e RTDE insertion",
            motion_kind="insert",
            insert_phase="checking",
        )
        self._write_active_goal_status(goal_handle, status)
        motion_attempted = False
        expected_start_world_tool0: RigidTransform | None = None
        target_world_tool0: RigidTransform | None = None
        world_base: RigidTransform | None = None
        tool0_tcp: RigidTransform | None = None
        insertion_axis_world: Vector3 | None = None
        insertion_axis_base: Vector3 | None = None
        force_bias = [0.0] * 6
        force_bias_valid = False
        max_axial_force_n = math.inf
        max_lateral_force_n = math.inf
        max_torque_nm = math.inf
        max_travel_m = math.inf
        final_world_tool0: RigidTransform | None = None
        final_insertion_depth_m = math.inf
        final_depth_error_m = math.inf
        final_lateral_offset_m = math.inf
        final_tilt_error_rad = math.inf
        final_search_radius_m = 0.0
        peak_axial_force_n = 0.0
        peak_lateral_force_n = 0.0
        peak_torque_nm = 0.0
        peak_filtered_axial_force_n = 0.0
        peak_filtered_lateral_force_n = 0.0
        peak_filtered_torque_nm = 0.0
        peak_tool_flange_torque_nm = 0.0
        contact_detected = False
        engagement_detected = False
        seated_detected = False
        trial_id = ""
        hard_caps_sha256 = ""
        selected_hard_caps: dict[str, float | int | None] = {}
        advanced_recovery_enabled = False
        force_depth_fraction: list[float] = []
        force_depth_axial_upper_n: list[float] = []
        force_depth_lateral_upper_n: list[float] = []
        force_depth_torque_upper_nm: list[float] = []
        current_force_depth_fraction = 0.0
        current_force_depth_axial_upper_n = 0.0
        current_force_depth_lateral_upper_n = 0.0
        current_force_depth_torque_upper_nm = 0.0
        axial_profile_exceeded = False
        lateral_profile_exceeded = False
        torque_profile_exceeded = False
        guarded_axial_force_ceiling_n = 0.0
        guarded_lateral_force_ceiling_n = 0.0
        guarded_torque_ceiling_nm = 0.0
        guarded_axial_force_exceeded = False
        guarded_lateral_force_exceeded = False
        guarded_torque_exceeded = False
        axial_progress_stalled = False
        final_phase = "checking"
        soft_overload_detected = False
        soft_overload_recovered = False
        relief_exhausted = False
        relief_cycle_count = 0
        last_soft_overload_reason = ""
        hard_limit_detected = False
        hard_limit_reason = ""
        limit_trigger = ""
        limit_trigger_value = 0.0
        limit_trigger_threshold = 0.0
        limit_trigger_actual_tcp_force = [0.0] * 6
        limit_trigger_tared_tcp_force = [0.0] * 6
        last_sample: dict[str, Any] = {}
        filtered_axial_force_n = 0.0
        filtered_lateral_force_n = 0.0
        filtered_torque_nm = 0.0
        filtered_tool_flange_torque_nm = 0.0
        soft_overload_duration_sec = 0.0
        relief_elapsed_sec = 0.0
        relief_retreat_m = 0.0
        relief_resume_phase = ""
        commanded_axial_force_n = 0.0
        commanded_lateral_force_x_n = 0.0
        commanded_lateral_force_y_n = 0.0
        relief_load_cleared = False
        relief_backoff_m = 0.0
        relief_planned_backoff_m = 0.0
        total_relief_backoff_m = 0.0
        force_mode_stop_acknowledged = False
        servo_stop_acknowledged = False
        stop_l_command_completed = False
        stationary_confirmed = False
        relief_force_mode_stop_acknowledged = False
        relief_stop_l_command_completed = False
        relief_stationary_confirmed = False
        relief_force_mode_restart_acknowledged = False
        server_trace_id = ""
        server_trace_path = ""
        server_trace_sha256 = ""
        server_trace_status = "not_started"
        server_trace_complete = False
        server_trace_sample_count = 0
        server_trace_tmp_path: Path | None = None
        server_trace_file: Any | None = None
        last_insert_feedback_timestamp: float | None = None
        tactile_center_world_tool0: RigidTransform | None = None
        tactile_center_depth_m = 0.0
        tactile_center_confidence = 0.0
        tactile_center_evidence_sha256 = ""
        scheduled_search_radius_m = 0.0
        explored_search_radius_m = 0.0
        explored_search_angle_rad = 0.0
        disengagement_cycle_count = 0
        last_disengagement_reason = ""
        disengagement_withdrawal_m = 0.0
        disengagement_contact_cleared = False
        disengagement_force_mode_stop_acknowledged = False
        recenter_position_error_m = math.inf
        recenter_command_acknowledged = False
        disengagement_stationary_confirmed = False
        retare_baseline_consistent = False
        search_peck_state = ""
        search_peck_cycle_count = 0
        search_peck_retreat_m = 0.0

        feedback_advance_grace_sec = min(
            float(UR5E_RTDE_FEEDBACK_RECONNECT_AFTER_SEC),
            max(0.05, 5.0 / float(UR5E_RTDE_FREQUENCY_HZ)),
        )
        feedback_advance_poll_sec = max(
            0.002,
            min(1.0 / float(UR5E_RTDE_FREQUENCY_HZ), 0.01),
        )

        def require_fresh_insert_feedback() -> float:
            nonlocal last_insert_feedback_timestamp
            receive_timestamp = self._read_feedback_timestamp()
            if receive_timestamp is None:
                raise RuntimeError("UR5e insertion RTDE timestamp is unavailable")
            if last_insert_feedback_timestamp is None:
                last_insert_feedback_timestamp = receive_timestamp
                return receive_timestamp
            if receive_timestamp < last_insert_feedback_timestamp:
                raise RuntimeError(
                    "UR5e insertion RTDE feedback timestamp moved backwards"
                )
            deadline = time.monotonic() + feedback_advance_grace_sec
            while receive_timestamp == last_insert_feedback_timestamp:
                if goal_handle.is_cancel_requested:
                    raise _InsertCanceled(
                        "canceled while waiting for advancing UR5e RTDE feedback"
                    )
                if not rclpy.ok():
                    raise RuntimeError(
                        "ROS shutdown interrupted UR5e insertion feedback"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "UR5e insertion RTDE feedback timestamp stopped advancing "
                        f"for {feedback_advance_grace_sec:.3f} s"
                    )
                time.sleep(feedback_advance_poll_sec)
                receive_timestamp = self._read_feedback_timestamp()
                if receive_timestamp is None:
                    raise RuntimeError(
                        "UR5e insertion RTDE timestamp is unavailable"
                    )
                if receive_timestamp < last_insert_feedback_timestamp:
                    raise RuntimeError(
                        "UR5e insertion RTDE feedback timestamp moved backwards"
                    )
            last_insert_feedback_timestamp = receive_timestamp
            return receive_timestamp

        def result(
            error_code: int,
            error_string: str,
            state_uncertain: bool,
            *,
            motion_settled: bool = True,
        ) -> Any:
            return self._insert_result(
                error_code,
                error_string,
                state_uncertain=state_uncertain,
                motion_settled=motion_settled,
                final_world_tool0=final_world_tool0,
                final_insertion_depth_m=final_insertion_depth_m,
                final_depth_error_m=final_depth_error_m,
                final_lateral_offset_m=final_lateral_offset_m,
                final_tilt_error_rad=final_tilt_error_rad,
                final_search_radius_m=final_search_radius_m,
                peak_axial_force_n=peak_axial_force_n,
                peak_lateral_force_n=peak_lateral_force_n,
                peak_torque_nm=peak_torque_nm,
                peak_filtered_axial_force_n=peak_filtered_axial_force_n,
                peak_filtered_lateral_force_n=peak_filtered_lateral_force_n,
                peak_filtered_torque_nm=peak_filtered_torque_nm,
                peak_tool_flange_torque_nm=peak_tool_flange_torque_nm,
                contact_detected=contact_detected,
                engagement_detected=engagement_detected,
                seated_detected=seated_detected,
                force_bias=force_bias if force_bias_valid else None,
                trial_id=trial_id,
                hard_caps_sha256=hard_caps_sha256,
                final_phase=final_phase,
                soft_overload_detected=soft_overload_detected,
                soft_overload_recovered=soft_overload_recovered,
                relief_exhausted=relief_exhausted,
                relief_cycle_count=relief_cycle_count,
                last_soft_overload_reason=last_soft_overload_reason,
                relief_load_cleared=relief_load_cleared,
                relief_backoff_m=relief_backoff_m,
                relief_planned_backoff_m=relief_planned_backoff_m,
                total_relief_backoff_m=total_relief_backoff_m,
                relief_resume_phase=relief_resume_phase,
                relief_force_mode_stop_acknowledged=(
                    relief_force_mode_stop_acknowledged
                ),
                relief_stop_l_command_completed=relief_stop_l_command_completed,
                relief_stationary_confirmed=relief_stationary_confirmed,
                relief_force_mode_restart_acknowledged=(
                    relief_force_mode_restart_acknowledged
                ),
                hard_limit_detected=hard_limit_detected,
                hard_limit_reason=hard_limit_reason,
                limit_trigger=limit_trigger,
                limit_trigger_value=limit_trigger_value,
                limit_trigger_threshold=limit_trigger_threshold,
                limit_trigger_actual_tcp_force=limit_trigger_actual_tcp_force,
                limit_trigger_tared_tcp_force=limit_trigger_tared_tcp_force,
                force_mode_stop_acknowledged=force_mode_stop_acknowledged,
                servo_stop_acknowledged=servo_stop_acknowledged,
                stop_l_command_completed=stop_l_command_completed,
                stationary_confirmed=stationary_confirmed,
                server_trace_id=server_trace_id,
                server_trace_path=server_trace_path,
                server_trace_sha256=server_trace_sha256,
                server_trace_status=server_trace_status,
                server_trace_complete=server_trace_complete,
                server_trace_sample_count=server_trace_sample_count,
                tactile_center_world_tool0=tactile_center_world_tool0,
                tactile_center_depth_m=tactile_center_depth_m,
                tactile_center_confidence=tactile_center_confidence,
                tactile_center_evidence_sha256=(
                    tactile_center_evidence_sha256
                ),
                scheduled_search_radius_m=scheduled_search_radius_m,
                explored_search_radius_m=explored_search_radius_m,
                explored_search_angle_rad=explored_search_angle_rad,
                disengagement_cycle_count=disengagement_cycle_count,
                last_disengagement_reason=last_disengagement_reason,
                disengagement_withdrawal_m=disengagement_withdrawal_m,
                disengagement_contact_cleared=(
                    disengagement_contact_cleared
                ),
                disengagement_force_mode_stop_acknowledged=(
                    disengagement_force_mode_stop_acknowledged
                ),
                recenter_position_error_m=recenter_position_error_m,
                recenter_command_acknowledged=recenter_command_acknowledged,
                disengagement_stationary_confirmed=(
                    disengagement_stationary_confirmed
                ),
                retare_baseline_consistent=retare_baseline_consistent,
            )

        def capture_final_pose() -> None:
            nonlocal final_world_tool0
            nonlocal final_insertion_depth_m
            nonlocal final_depth_error_m
            nonlocal final_lateral_offset_m
            nonlocal final_tilt_error_rad
            if (
                world_base is None
                or tool0_tcp is None
                or expected_start_world_tool0 is None
                or target_world_tool0 is None
                or insertion_axis_world is None
            ):
                return
            actual_base_tcp = self._read_actual_tcp_transform()
            if actual_base_tcp is None:
                return
            actual_world_tool0 = self._world_tool0_from_actual_tcp(
                actual_base_tcp,
                world_base=world_base,
                tool0_tcp=tool0_tcp,
            )
            (
                final_insertion_depth_m,
                final_depth_error_m,
                final_lateral_offset_m,
                final_tilt_error_rad,
            ) = _insertion_pose_metrics(
                actual_world_tool0,
                expected_start_world_tool0,
                target_world_tool0,
                insertion_axis_world,
            )
            final_world_tool0 = actual_world_tool0

        def stop_and_confirm() -> bool:
            nonlocal force_mode_stop_acknowledged
            nonlocal servo_stop_acknowledged
            nonlocal stop_l_command_completed
            nonlocal stationary_confirmed
            stop_commanded = self._stop_motion()
            force_mode_stop_acknowledged = bool(
                getattr(self, "_insert_force_mode_stop_acknowledged", False)
            )
            servo_stop_acknowledged = bool(
                getattr(self, "_insert_servo_stop_acknowledged", False)
            )
            stop_l_command_completed = bool(
                getattr(self, "_insert_stop_l_command_completed", False)
            )
            stationary_confirmed = self._confirm_stationary_after_stop(
                linear_speed_limit_m_s=relief_stationary_speed_m_s,
                angular_speed_limit_rad_s=(
                    relief_stationary_angular_speed_rad_s
                ),
            )
            capture_final_pose()
            return bool(stop_commanded and stationary_confirmed)

        def start_server_trace() -> None:
            nonlocal server_trace_id
            nonlocal server_trace_path
            nonlocal server_trace_status
            nonlocal server_trace_tmp_path
            nonlocal server_trace_file
            if trial_id:
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", trial_id) is None:
                    raise ValueError(
                        "trial_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
                    )
                server_trace_id = trial_id
            else:
                server_trace_id = f"automatic-{int(time.time() * 1_000_000_000)}-{os.getpid()}"
            root = INSERT_TRIAL_TRACE_ROOT.resolve()
            trace_directory = (root / server_trace_id).resolve()
            if trace_directory.parent != root:
                raise ValueError("server insertion trace path escaped its protected root")
            trace_directory.mkdir(parents=True, exist_ok=True)
            path = trace_directory / "trace.jsonl"
            if path.exists():
                raise ValueError(
                    f"server insertion trace already exists for trial_id {server_trace_id}"
                )
            server_trace_path = str(path)
            server_trace_tmp_path = trace_directory / (
                f".trace.jsonl.{os.getpid()}.{int(time.time() * 1_000_000_000)}.tmp"
            )
            server_trace_file = server_trace_tmp_path.open("x", encoding="utf-8")
            server_trace_status = "recording"
            append_server_trace(
                "header",
                {
                    "part_name": part_name,
                    "calibration_id": calibration_id,
                    "profile_sha256": profile_sha256,
                    "hard_caps_sha256": hard_caps_sha256,
                },
            )

        def append_server_trace(kind: str, payload: dict[str, Any]) -> None:
            nonlocal server_trace_sample_count
            if server_trace_file is None:
                return
            record = {
                "kind": str(kind),
                "recorded_at": time.time(),
                "trial_id": trial_id,
                "server_trace_id": server_trace_id,
                **payload,
            }
            server_trace_file.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            server_trace_file.flush()
            if kind == "sample":
                server_trace_sample_count += 1

        def finalize_server_trace(
            *,
            terminal_state: str,
            error_code: int,
            error_string: str,
        ) -> None:
            nonlocal server_trace_file
            nonlocal server_trace_sha256
            nonlocal server_trace_status
            nonlocal server_trace_complete
            if server_trace_file is None or server_trace_tmp_path is None:
                return
            try:
                append_server_trace(
                    "terminal",
                    {
                        "terminal_state": terminal_state,
                        "error_code": int(error_code),
                        "error_string": str(error_string or ""),
                        **terminal_evidence(),
                    },
                )
                os.fsync(server_trace_file.fileno())
                server_trace_file.close()
                server_trace_file = None
                final_path = Path(server_trace_path)
                server_trace_tmp_path.replace(final_path)
                server_trace_sha256 = hashlib.sha256(final_path.read_bytes()).hexdigest()
                server_trace_status = "complete"
                server_trace_complete = True
            except (OSError, TypeError, ValueError):
                server_trace_status = "failed"
                server_trace_complete = False
                with suppress(OSError):
                    if server_trace_file is not None:
                        server_trace_file.close()
                server_trace_file = None

        def terminal_evidence() -> dict[str, Any]:
            return {
                "trial_id": trial_id,
                "hard_caps_sha256": hard_caps_sha256,
                "insert_phase": final_phase,
                "insert_current_force_depth_fraction": (
                    current_force_depth_fraction
                ),
                "insert_force_depth_axial_upper_n": (
                    current_force_depth_axial_upper_n
                ),
                "insert_force_depth_lateral_upper_n": (
                    current_force_depth_lateral_upper_n
                ),
                "insert_force_depth_torque_upper_nm": (
                    current_force_depth_torque_upper_nm
                ),
                "insert_axial_profile_exceeded": axial_profile_exceeded,
                "insert_lateral_profile_exceeded": lateral_profile_exceeded,
                "insert_torque_profile_exceeded": torque_profile_exceeded,
                "insert_axial_progress_stalled": axial_progress_stalled,
                "insert_filtered_axial_force_n": filtered_axial_force_n,
                "insert_filtered_lateral_force_n": filtered_lateral_force_n,
                "insert_filtered_torque_nm": filtered_torque_nm,
                "insert_filtered_tool_flange_torque_nm": (
                    filtered_tool_flange_torque_nm
                ),
                "insert_soft_overload_detected": soft_overload_detected,
                "insert_soft_overload_reason": last_soft_overload_reason,
                "insert_soft_overload_duration_sec": soft_overload_duration_sec,
                "insert_soft_overload_recovered": soft_overload_recovered,
                "insert_relief_exhausted": relief_exhausted,
                "insert_relief_cycle_count": relief_cycle_count,
                "insert_relief_elapsed_sec": relief_elapsed_sec,
                "insert_relief_retreat_m": relief_retreat_m,
                "insert_relief_load_cleared": relief_load_cleared,
                "insert_relief_backoff_m": relief_backoff_m,
                "insert_relief_planned_backoff_m": relief_planned_backoff_m,
                "insert_total_relief_backoff_m": total_relief_backoff_m,
                "insert_relief_resume_phase": relief_resume_phase,
                "insert_relief_force_mode_stop_acknowledged": (
                    relief_force_mode_stop_acknowledged
                ),
                "insert_relief_stop_l_command_completed": (
                    relief_stop_l_command_completed
                ),
                "insert_relief_stationary_confirmed": relief_stationary_confirmed,
                "insert_relief_force_mode_restart_acknowledged": (
                    relief_force_mode_restart_acknowledged
                ),
                "insert_commanded_axial_force_n": commanded_axial_force_n,
                "insert_commanded_lateral_force_x_n": (
                    commanded_lateral_force_x_n
                ),
                "insert_commanded_lateral_force_y_n": (
                    commanded_lateral_force_y_n
                ),
                "insert_hard_limit_detected": hard_limit_detected,
                "insert_hard_limit_reason": hard_limit_reason,
                "insert_limit_trigger": limit_trigger,
                "insert_limit_trigger_value": limit_trigger_value,
                "insert_limit_trigger_threshold": limit_trigger_threshold,
                "insert_limit_trigger_actual_tcp_force": list(
                    limit_trigger_actual_tcp_force
                ),
                "insert_limit_trigger_tared_tcp_force": list(
                    limit_trigger_tared_tcp_force
                ),
                "insert_force_mode_stop_acknowledged": (
                    force_mode_stop_acknowledged
                ),
                "insert_servo_stop_acknowledged": servo_stop_acknowledged,
                "insert_stop_l_command_completed": stop_l_command_completed,
                "insert_stationary_confirmed": stationary_confirmed,
                "peak_filtered_axial_force_n": peak_filtered_axial_force_n,
                "peak_filtered_lateral_force_n": peak_filtered_lateral_force_n,
                "peak_filtered_torque_nm": peak_filtered_torque_nm,
                "peak_tool_flange_torque_nm": peak_tool_flange_torque_nm,
                "server_trace_id": server_trace_id,
                "server_trace_path": server_trace_path,
                "server_trace_sha256": server_trace_sha256,
                "server_trace_status": server_trace_status,
                "server_trace_complete": server_trace_complete,
                "server_trace_sample_count": server_trace_sample_count,
                "insert_tactile_center_valid": (
                    tactile_center_world_tool0 is not None
                ),
                "insert_tactile_center_tool0_pose": (
                    _transform_status_payload(tactile_center_world_tool0)
                ),
                "insert_tactile_center_depth_m": tactile_center_depth_m,
                "insert_tactile_center_confidence": tactile_center_confidence,
                "insert_tactile_center_evidence_sha256": (
                    tactile_center_evidence_sha256
                ),
                "insert_scheduled_search_radius_m": scheduled_search_radius_m,
                "insert_explored_search_radius_m": explored_search_radius_m,
                "insert_explored_search_angle_rad": explored_search_angle_rad,
                "insert_disengagement_cycle_count": disengagement_cycle_count,
                "insert_last_disengagement_reason": last_disengagement_reason,
                "insert_disengagement_withdrawal_m": (
                    disengagement_withdrawal_m
                ),
                "insert_disengagement_contact_cleared": (
                    disengagement_contact_cleared
                ),
                "insert_disengagement_force_mode_stop_acknowledged": (
                    disengagement_force_mode_stop_acknowledged
                ),
                "insert_recenter_position_error_m": recenter_position_error_m,
                "insert_recenter_command_acknowledged": (
                    recenter_command_acknowledged
                ),
                "insert_disengagement_stationary_confirmed": (
                    disengagement_stationary_confirmed
                ),
                "insert_retare_baseline_consistent": (
                    retare_baseline_consistent
                ),
            }

        def publish_last_sample(phase: str) -> None:
            if not last_sample:
                return
            self._publish_insert_feedback(
                goal_handle,
                phase=phase,
                trial_id=trial_id,
                actual_world_tool0=last_sample["actual_world_tool0"],
                insertion_depth_m=float(last_sample["insertion_depth_m"]),
                depth_error_m=float(last_sample["depth_error_m"]),
                lateral_offset_m=float(last_sample["lateral_offset_m"]),
                search_radius_m=final_search_radius_m,
                axial_force_n=float(last_sample["axial_force_n"]),
                raw_axial_force_n=float(last_sample["raw_axial_force_n"]),
                lateral_force_n=float(last_sample["lateral_force_n"]),
                torque_nm=float(last_sample["torque_nm"]),
                filtered_axial_force_n=filtered_axial_force_n,
                filtered_lateral_force_n=filtered_lateral_force_n,
                filtered_torque_nm=filtered_torque_nm,
                tool_flange_torque_nm=float(
                    last_sample["tool_flange_torque_nm"]
                ),
                filtered_tool_flange_torque_nm=filtered_tool_flange_torque_nm,
                current_force_depth_fraction=current_force_depth_fraction,
                force_depth_axial_upper_n=current_force_depth_axial_upper_n,
                force_depth_lateral_upper_n=current_force_depth_lateral_upper_n,
                force_depth_torque_upper_nm=current_force_depth_torque_upper_nm,
                axial_profile_exceeded=axial_profile_exceeded,
                axial_progress_stalled=axial_progress_stalled,
                contact_detected=contact_detected,
                engagement_detected=engagement_detected,
                seated_detected=seated_detected,
                force_bias_valid=force_bias_valid,
                force_bias=force_bias,
                actual_tcp_force=last_sample["actual_tcp_force"],
                tared_tcp_force=last_sample["tared_tcp_force"],
                actual_tcp_speed=last_sample["actual_tcp_speed"],
                soft_overload_detected=soft_overload_detected,
                soft_overload_reason=last_soft_overload_reason,
                soft_overload_duration_sec=soft_overload_duration_sec,
                relief_cycle_count=relief_cycle_count,
                relief_elapsed_sec=relief_elapsed_sec,
                relief_retreat_m=relief_retreat_m,
                relief_load_cleared=relief_load_cleared,
                relief_backoff_m=relief_backoff_m,
                relief_planned_backoff_m=relief_planned_backoff_m,
                total_relief_backoff_m=total_relief_backoff_m,
                relief_resume_phase=relief_resume_phase,
                commanded_axial_force_n=commanded_axial_force_n,
                commanded_lateral_force_x_n=commanded_lateral_force_x_n,
                commanded_lateral_force_y_n=commanded_lateral_force_y_n,
                hard_limit_detected=hard_limit_detected,
                hard_limit_reason=hard_limit_reason,
                limit_trigger=limit_trigger,
                limit_trigger_value=limit_trigger_value,
                limit_trigger_threshold=limit_trigger_threshold,
                limit_trigger_actual_tcp_force=limit_trigger_actual_tcp_force,
                limit_trigger_tared_tcp_force=limit_trigger_tared_tcp_force,
                tactile_center_world_tool0=tactile_center_world_tool0,
                tactile_center_depth_m=tactile_center_depth_m,
                tactile_center_confidence=tactile_center_confidence,
                tactile_center_evidence_sha256=(
                    tactile_center_evidence_sha256
                ),
                scheduled_search_radius_m=scheduled_search_radius_m,
                explored_search_radius_m=explored_search_radius_m,
                explored_search_angle_rad=explored_search_angle_rad,
                disengagement_cycle_count=disengagement_cycle_count,
                last_disengagement_reason=last_disengagement_reason,
                disengagement_withdrawal_m=disengagement_withdrawal_m,
                disengagement_contact_cleared=(
                    disengagement_contact_cleared
                ),
                disengagement_force_mode_stop_acknowledged=(
                    disengagement_force_mode_stop_acknowledged
                ),
                recenter_position_error_m=recenter_position_error_m,
                recenter_command_acknowledged=recenter_command_acknowledged,
                disengagement_stationary_confirmed=(
                    disengagement_stationary_confirmed
                ),
                retare_baseline_consistent=retare_baseline_consistent,
            )

        def raise_hard_limit(
            trigger: str,
            value: float,
            threshold: float,
            reason: str,
            *,
            phase: str,
        ) -> None:
            nonlocal hard_limit_detected
            nonlocal hard_limit_reason
            nonlocal limit_trigger
            nonlocal limit_trigger_value
            nonlocal limit_trigger_threshold
            nonlocal limit_trigger_actual_tcp_force
            nonlocal limit_trigger_tared_tcp_force
            hard_limit_detected = True
            hard_limit_reason = reason
            limit_trigger = trigger
            limit_trigger_value = float(value)
            limit_trigger_threshold = float(threshold)
            limit_trigger_actual_tcp_force = list(
                last_sample.get("actual_tcp_force", [0.0] * 6)
            )
            limit_trigger_tared_tcp_force = list(
                last_sample.get("tared_tcp_force", [0.0] * 6)
            )
            append_server_trace(
                "hard_trigger",
                {
                    **last_sample,
                    "hard_limit_detected": True,
                    "hard_limit_reason": reason,
                    "limit_trigger": trigger,
                    "limit_trigger_value": float(value),
                    "limit_trigger_threshold": float(threshold),
                },
            )
            status.update(
                insert_hard_limit_detected=True,
                insert_hard_limit_reason=reason,
                insert_limit_trigger=trigger,
                insert_limit_trigger_value=float(value),
                insert_limit_trigger_threshold=float(threshold),
                insert_limit_trigger_actual_tcp_force=list(
                    limit_trigger_actual_tcp_force
                ),
                insert_limit_trigger_tared_tcp_force=list(
                    limit_trigger_tared_tcp_force
                ),
            )
            self._write_active_goal_status(goal_handle, status)
            publish_last_sample(phase)
            raise _InsertForceLimit(reason)

        def sample(phase: str) -> dict[str, Any]:  # noqa: C901, PLR0915
            nonlocal final_world_tool0
            nonlocal final_insertion_depth_m
            nonlocal final_depth_error_m
            nonlocal final_lateral_offset_m
            nonlocal final_tilt_error_rad
            nonlocal peak_axial_force_n
            nonlocal peak_lateral_force_n
            nonlocal peak_torque_nm
            nonlocal peak_tool_flange_torque_nm
            nonlocal last_sample
            nonlocal current_force_depth_fraction
            nonlocal current_force_depth_axial_upper_n
            nonlocal current_force_depth_lateral_upper_n
            nonlocal current_force_depth_torque_upper_nm
            nonlocal axial_profile_exceeded
            nonlocal lateral_profile_exceeded
            nonlocal torque_profile_exceeded
            nonlocal axial_progress_stalled
            nonlocal progress_reference_depth_m
            nonlocal progress_reference_at
            if (
                world_base is None
                or tool0_tcp is None
                or expected_start_world_tool0 is None
                or target_world_tool0 is None
                or insertion_axis_world is None
                or insertion_axis_base is None
            ):
                raise RuntimeError("insertion feedback transforms are unavailable")
            require_fresh_insert_feedback()
            actual_base_tcp = self._read_actual_tcp_transform()
            actual_tcp_force = self._read_actual_tcp_force()
            actual_tcp_speed = self._read_actual_tcp_speed()
            if actual_base_tcp is None:
                raise RuntimeError("UR5e insertion pose is unavailable")
            if actual_tcp_force is None:
                raise RuntimeError("actual_TCP_force is unavailable during insertion")
            if actual_tcp_speed is None:
                raise RuntimeError("actual_TCP_speed is unavailable during insertion")
            actual_world_tool0 = self._world_tool0_from_actual_tcp(
                actual_base_tcp,
                world_base=world_base,
                tool0_tcp=tool0_tcp,
            )
            (
                insertion_depth_m,
                depth_error_m,
                lateral_offset_m,
                tilt_error_rad,
            ) = _insertion_pose_metrics(
                actual_world_tool0,
                expected_start_world_tool0,
                target_world_tool0,
                insertion_axis_world,
            )
            actual_base_tool0 = _compose_transform(
                actual_base_tcp,
                _inverse_transform(tool0_tcp),
            )
            tool0_tcp_offset_base = _rotate_vector(
                actual_base_tool0[1],
                tool0_tcp[0],
            )
            (
                raw_axial_force_n,
                axial_force_n,
                lateral_force_n,
                torque_nm,
                tool_flange_torque_nm,
                tared_tcp_force,
            ) = _insertion_force_metrics(
                actual_tcp_force,
                force_bias,
                insertion_axis_base,
                tool0_tcp_offset_base,
            )
            axial_tcp_speed_m_s = _vector_dot(
                tuple(actual_tcp_speed[:3]),
                insertion_axis_base,
            )
            peak_axial_force_n = max(peak_axial_force_n, raw_axial_force_n)
            peak_lateral_force_n = max(peak_lateral_force_n, lateral_force_n)
            peak_torque_nm = max(peak_torque_nm, torque_nm)
            peak_tool_flange_torque_nm = max(
                peak_tool_flange_torque_nm,
                tool_flange_torque_nm,
            )
            final_world_tool0 = actual_world_tool0
            final_insertion_depth_m = insertion_depth_m
            final_depth_error_m = depth_error_m
            final_lateral_offset_m = lateral_offset_m
            final_tilt_error_rad = tilt_error_rad
            travel_m, _unused_orientation_error = _pose_errors(
                actual_world_tool0,
                expected_start_world_tool0,
            )
            last_sample = {
                "phase": phase,
                "actual_world_tool0": actual_world_tool0,
                "insertion_depth_m": insertion_depth_m,
                "depth_error_m": depth_error_m,
                "lateral_offset_m": lateral_offset_m,
                "tilt_error_rad": tilt_error_rad,
                "axial_force_n": axial_force_n,
                "raw_axial_force_n": raw_axial_force_n,
                "lateral_force_n": lateral_force_n,
                "torque_nm": torque_nm,
                "tool_flange_torque_nm": tool_flange_torque_nm,
                "tared_tcp_force": list(tared_tcp_force),
                "actual_tcp_force": list(actual_tcp_force),
                "actual_tcp_speed": list(actual_tcp_speed),
                "axial_tcp_speed_m_s": axial_tcp_speed_m_s,
                "linear_speed_m_s": _vector_norm(tuple(actual_tcp_speed[:3])),
                "angular_speed_rad_s": _vector_norm(tuple(actual_tcp_speed[3:6])),
            }
            trace_relief_retreat_m = relief_retreat_m
            if (
                phase in {"relieving", "backing_off", "resuming"}
                and relief_started_at is not None
                and not relief_backoff_committed
            ):
                trace_relief_retreat_m = max(
                    trace_relief_retreat_m,
                    relief_entry_depth_m - insertion_depth_m,
                )
            trace_relief_backoff_m = max(
                relief_backoff_m,
                trace_relief_retreat_m,
            )
            append_server_trace(
                "sample",
                {
                    **last_sample,
                    "filtered_axial_force_n": filtered_axial_force_n,
                    "filtered_lateral_force_n": filtered_lateral_force_n,
                    "filtered_torque_nm": filtered_torque_nm,
                    "filtered_tool_flange_torque_nm": (
                        filtered_tool_flange_torque_nm
                    ),
                    "current_force_depth_fraction": (
                        current_force_depth_fraction
                    ),
                    "force_depth_axial_upper_n": (
                        current_force_depth_axial_upper_n
                    ),
                    "force_depth_lateral_upper_n": (
                        current_force_depth_lateral_upper_n
                    ),
                    "force_depth_torque_upper_nm": (
                        current_force_depth_torque_upper_nm
                    ),
                    "axial_profile_exceeded": axial_profile_exceeded,
                    "lateral_profile_exceeded": lateral_profile_exceeded,
                    "torque_profile_exceeded": torque_profile_exceeded,
                    "guarded_axial_force_ceiling_n": (
                        guarded_axial_force_ceiling_n
                    ),
                    "guarded_lateral_force_ceiling_n": (
                        guarded_lateral_force_ceiling_n
                    ),
                    "guarded_torque_ceiling_nm": guarded_torque_ceiling_nm,
                    "guarded_axial_force_exceeded": (
                        guarded_axial_force_exceeded
                    ),
                    "guarded_lateral_force_exceeded": (
                        guarded_lateral_force_exceeded
                    ),
                    "guarded_torque_exceeded": guarded_torque_exceeded,
                    "axial_progress_stalled": axial_progress_stalled,
                    "force_bias_valid": force_bias_valid,
                    "force_bias": list(force_bias),
                    "contact_detected": contact_detected,
                    "engagement_detected": engagement_detected,
                    "seated_detected": seated_detected,
                    "soft_overload_detected": soft_overload_detected,
                    "soft_overload_reason": last_soft_overload_reason,
                    "relief_cycle_count": relief_cycle_count,
                    "relief_elapsed_sec": relief_elapsed_sec,
                    "relief_retreat_m": trace_relief_retreat_m,
                    "relief_backoff_m": trace_relief_backoff_m,
                    "relief_planned_backoff_m": relief_planned_backoff_m,
                    "total_relief_backoff_m": total_relief_backoff_m,
                    "commanded_axial_force_n": commanded_axial_force_n,
                    "commanded_lateral_force_x_n": commanded_lateral_force_x_n,
                    "commanded_lateral_force_y_n": commanded_lateral_force_y_n,
                    "search_peck_state": search_peck_state,
                    "search_peck_cycle_count": search_peck_cycle_count,
                    "search_peck_retreat_m": search_peck_retreat_m,
                },
            )

            hard_axial_force_n = float(
                selected_hard_caps.get("insert_max_axial_force_n") or math.nan
            )
            hard_lateral_force_n = float(
                selected_hard_caps.get("insert_max_lateral_force_n") or math.nan
            )
            hard_torque_nm = float(
                selected_hard_caps.get("insert_max_torque_nm") or math.nan
            )
            hard_tool_flange_torque_nm = float(
                selected_hard_caps.get("insert_max_tool_flange_torque_nm")
                or math.nan
            )
            workspace_error = _workspace_error(actual_world_tool0)
            if workspace_error:
                raise_hard_limit(
                    "workspace_pose",
                    1.0,
                    0.0,
                    f"actual insertion pose left protected workspace: {workspace_error}",
                    phase=phase,
                )
            if raw_axial_force_n > hard_axial_force_n:
                raise_hard_limit(
                    "axial_force_n",
                    raw_axial_force_n,
                    hard_axial_force_n,
                    f"absolute axial force {raw_axial_force_n:.3f} N exceeded hard ceiling "
                    f"{hard_axial_force_n:.3f} N",
                    phase=phase,
                )
            if lateral_force_n > hard_lateral_force_n:
                raise_hard_limit(
                    "lateral_force_n",
                    lateral_force_n,
                    hard_lateral_force_n,
                    f"lateral force {lateral_force_n:.3f} N exceeded hard ceiling "
                    f"{hard_lateral_force_n:.3f} N",
                    phase=phase,
                )
            if torque_nm > hard_torque_nm:
                raise_hard_limit(
                    "active_tcp_torque_nm",
                    torque_nm,
                    hard_torque_nm,
                    f"active-TCP torque {torque_nm:.3f} Nm exceeded hard ceiling "
                    f"{hard_torque_nm:.3f} Nm",
                    phase=phase,
                )
            if tool_flange_torque_nm > hard_tool_flange_torque_nm:
                raise_hard_limit(
                    "tool_flange_torque_nm",
                    tool_flange_torque_nm,
                    hard_tool_flange_torque_nm,
                    f"tool-flange torque {tool_flange_torque_nm:.3f} Nm exceeded hard "
                    f"ceiling {hard_tool_flange_torque_nm:.3f} Nm",
                    phase=phase,
                )
            if travel_m > max_travel_m:
                raise_hard_limit(
                    "insertion_travel_m",
                    travel_m,
                    max_travel_m,
                    f"insertion travel {travel_m:.6f} m exceeded {max_travel_m:.6f} m",
                    phase=phase,
                )
            if insertion_depth_m < -start_position_tolerance_m:
                raise_hard_limit(
                    "reverse_insertion_travel_m",
                    -insertion_depth_m,
                    start_position_tolerance_m,
                    "insertion moved opposite insertion_axis_world by "
                    f"{-insertion_depth_m:.6f} m, exceeding the accepted start "
                    f"position tolerance {start_position_tolerance_m:.6f} m",
                    phase=phase,
                )
            if insertion_depth_m > target_depth_m + seated_depth_tolerance_m:
                raise_hard_limit(
                    "insertion_depth_m",
                    insertion_depth_m,
                    target_depth_m + seated_depth_tolerance_m,
                    f"insertion depth {insertion_depth_m:.6f} m exceeded target depth "
                    f"{target_depth_m:.6f} m plus seated tolerance",
                    phase=phase,
                )
            protected_spiral_radius_m = (
                float(UR5E_RTDE_INSERT_MAX_SPIRAL_RADIUS_M or math.nan)
                if spiral_radius_m > 0.0
                else 0.0
            )
            mg_recovery_phase = phase in {
                "relieving",
                "backing_off",
                "resuming",
                "cocked",
                "disengaging",
                "recentering",
                "retaring",
                "retrying",
            }
            lateral_travel_limit_m = (
                max_contact_search_radius_m
                + (start_position_tolerance_m if mg_recovery_phase else 0.0)
                if advanced_recovery_enabled
                else protected_spiral_radius_m + start_position_tolerance_m
            )
            if lateral_offset_m > lateral_travel_limit_m:
                raise_hard_limit(
                    "lateral_offset_m",
                    lateral_offset_m,
                    lateral_travel_limit_m,
                    f"lateral insertion offset {lateral_offset_m:.6f} m exceeded "
                    f"{lateral_travel_limit_m:.6f} m",
                    phase=phase,
                )
            if tilt_error_rad > tilt_tolerance_rad:
                raise_hard_limit(
                    "tilt_error_rad",
                    tilt_error_rad,
                    tilt_tolerance_rad,
                    f"insertion tilt {tilt_error_rad:.6f} rad exceeded "
                    f"{tilt_tolerance_rad:.6f} rad",
                    phase=phase,
                )
            status.update(
                state="executing",
                message=f"executing UR5e RTDE insertion: {phase}",
                blocked_reason="",
                trial_id=trial_id,
                insert_phase=phase,
                insert_insertion_depth_m=insertion_depth_m,
                insert_depth_error_m=depth_error_m,
                insert_lateral_offset_m=lateral_offset_m,
                insert_search_radius_m=final_search_radius_m,
                insert_search_peck_state=search_peck_state,
                insert_search_peck_cycle_count=search_peck_cycle_count,
                insert_search_peck_retreat_m=search_peck_retreat_m,
                insert_axial_force_n=axial_force_n,
                insert_raw_axial_force_n=raw_axial_force_n,
                insert_lateral_force_n=lateral_force_n,
                insert_torque_nm=torque_nm,
                insert_filtered_axial_force_n=filtered_axial_force_n,
                insert_filtered_lateral_force_n=filtered_lateral_force_n,
                insert_filtered_torque_nm=filtered_torque_nm,
                insert_tool_flange_torque_nm=tool_flange_torque_nm,
                insert_filtered_tool_flange_torque_nm=(
                    filtered_tool_flange_torque_nm
                ),
                insert_current_force_depth_fraction=(
                    current_force_depth_fraction
                ),
                insert_force_depth_axial_upper_n=(
                    current_force_depth_axial_upper_n
                ),
                insert_force_depth_lateral_upper_n=(
                    current_force_depth_lateral_upper_n
                ),
                insert_force_depth_torque_upper_nm=(
                    current_force_depth_torque_upper_nm
                ),
                insert_axial_profile_exceeded=axial_profile_exceeded,
                insert_lateral_profile_exceeded=lateral_profile_exceeded,
                insert_torque_profile_exceeded=torque_profile_exceeded,
                insert_axial_progress_stalled=axial_progress_stalled,
                insert_tared_tcp_force=list(tared_tcp_force),
                insert_contact_detected=contact_detected,
                insert_engagement_detected=engagement_detected,
                insert_seated_detected=seated_detected,
                insert_tactile_center_valid=(
                    tactile_center_world_tool0 is not None
                ),
                insert_tactile_center_tool0_pose=(
                    _transform_status_payload(tactile_center_world_tool0)
                ),
                insert_tactile_center_depth_m=tactile_center_depth_m,
                insert_tactile_center_confidence=tactile_center_confidence,
                insert_tactile_center_evidence_sha256=(
                    tactile_center_evidence_sha256
                ),
                insert_scheduled_search_radius_m=scheduled_search_radius_m,
                insert_explored_search_radius_m=explored_search_radius_m,
                insert_explored_search_angle_rad=explored_search_angle_rad,
                insert_disengagement_cycle_count=disengagement_cycle_count,
                insert_last_disengagement_reason=last_disengagement_reason,
                insert_disengagement_withdrawal_m=disengagement_withdrawal_m,
                insert_disengagement_contact_cleared=(
                    disengagement_contact_cleared
                ),
                insert_disengagement_force_mode_stop_acknowledged=(
                    disengagement_force_mode_stop_acknowledged
                ),
                insert_recenter_position_error_m=recenter_position_error_m,
                insert_recenter_command_acknowledged=(
                    recenter_command_acknowledged
                ),
                insert_disengagement_stationary_confirmed=(
                    disengagement_stationary_confirmed
                ),
                insert_retare_baseline_consistent=retare_baseline_consistent,
                insert_soft_overload_detected=soft_overload_detected,
                insert_soft_overload_reason=last_soft_overload_reason,
                insert_soft_overload_duration_sec=soft_overload_duration_sec,
                insert_soft_overload_recovered=soft_overload_recovered,
                insert_relief_exhausted=relief_exhausted,
                insert_relief_cycle_count=relief_cycle_count,
                insert_relief_elapsed_sec=relief_elapsed_sec,
                insert_relief_retreat_m=relief_retreat_m,
                insert_relief_load_cleared=relief_load_cleared,
                insert_relief_backoff_m=relief_backoff_m,
                insert_relief_planned_backoff_m=relief_planned_backoff_m,
                insert_total_relief_backoff_m=total_relief_backoff_m,
                insert_relief_resume_phase=relief_resume_phase,
                insert_relief_force_mode_stop_acknowledged=(
                    relief_force_mode_stop_acknowledged
                ),
                insert_relief_stop_l_command_completed=(
                    relief_stop_l_command_completed
                ),
                insert_relief_stationary_confirmed=relief_stationary_confirmed,
                insert_relief_force_mode_restart_acknowledged=(
                    relief_force_mode_restart_acknowledged
                ),
                insert_commanded_axial_force_n=commanded_axial_force_n,
                insert_commanded_lateral_force_x_n=commanded_lateral_force_x_n,
                insert_commanded_lateral_force_y_n=commanded_lateral_force_y_n,
                server_trace_id=server_trace_id,
                server_trace_path=server_trace_path,
                server_trace_status=server_trace_status,
                server_trace_complete=False,
                server_trace_sample_count=server_trace_sample_count,
                actual_tcp_force=list(actual_tcp_force),
                actual_tcp_speed=list(actual_tcp_speed),
                peak_axial_force_n=peak_axial_force_n,
                peak_lateral_force_n=peak_lateral_force_n,
                peak_torque_nm=peak_torque_nm,
                peak_tool_flange_torque_nm=peak_tool_flange_torque_nm,
            )
            self._write_active_goal_status(goal_handle, status)
            self._publish_insert_feedback(
                goal_handle,
                phase=phase,
                trial_id=trial_id,
                actual_world_tool0=actual_world_tool0,
                insertion_depth_m=insertion_depth_m,
                depth_error_m=depth_error_m,
                lateral_offset_m=lateral_offset_m,
                search_radius_m=final_search_radius_m,
                axial_force_n=axial_force_n,
                raw_axial_force_n=raw_axial_force_n,
                lateral_force_n=lateral_force_n,
                torque_nm=torque_nm,
                filtered_axial_force_n=filtered_axial_force_n,
                filtered_lateral_force_n=filtered_lateral_force_n,
                filtered_torque_nm=filtered_torque_nm,
                tool_flange_torque_nm=tool_flange_torque_nm,
                filtered_tool_flange_torque_nm=filtered_tool_flange_torque_nm,
                current_force_depth_fraction=current_force_depth_fraction,
                force_depth_axial_upper_n=current_force_depth_axial_upper_n,
                force_depth_lateral_upper_n=current_force_depth_lateral_upper_n,
                force_depth_torque_upper_nm=current_force_depth_torque_upper_nm,
                axial_profile_exceeded=axial_profile_exceeded,
                axial_progress_stalled=axial_progress_stalled,
                contact_detected=contact_detected,
                engagement_detected=engagement_detected,
                seated_detected=seated_detected,
                force_bias_valid=force_bias_valid,
                force_bias=force_bias,
                actual_tcp_force=actual_tcp_force,
                tared_tcp_force=tared_tcp_force,
                actual_tcp_speed=actual_tcp_speed,
                soft_overload_detected=soft_overload_detected,
                soft_overload_reason=last_soft_overload_reason,
                soft_overload_duration_sec=soft_overload_duration_sec,
                relief_cycle_count=relief_cycle_count,
                relief_elapsed_sec=relief_elapsed_sec,
                relief_retreat_m=relief_retreat_m,
                relief_load_cleared=relief_load_cleared,
                relief_backoff_m=relief_backoff_m,
                relief_planned_backoff_m=relief_planned_backoff_m,
                total_relief_backoff_m=total_relief_backoff_m,
                relief_resume_phase=relief_resume_phase,
                commanded_axial_force_n=commanded_axial_force_n,
                commanded_lateral_force_x_n=commanded_lateral_force_x_n,
                commanded_lateral_force_y_n=commanded_lateral_force_y_n,
                hard_limit_detected=hard_limit_detected,
                hard_limit_reason=hard_limit_reason,
                limit_trigger=limit_trigger,
                limit_trigger_value=limit_trigger_value,
                limit_trigger_threshold=limit_trigger_threshold,
                limit_trigger_actual_tcp_force=limit_trigger_actual_tcp_force,
                limit_trigger_tared_tcp_force=limit_trigger_tared_tcp_force,
                tactile_center_world_tool0=tactile_center_world_tool0,
                tactile_center_depth_m=tactile_center_depth_m,
                tactile_center_confidence=tactile_center_confidence,
                tactile_center_evidence_sha256=(
                    tactile_center_evidence_sha256
                ),
                scheduled_search_radius_m=scheduled_search_radius_m,
                explored_search_radius_m=explored_search_radius_m,
                explored_search_angle_rad=explored_search_angle_rad,
                disengagement_cycle_count=disengagement_cycle_count,
                last_disengagement_reason=last_disengagement_reason,
                disengagement_withdrawal_m=disengagement_withdrawal_m,
                disengagement_contact_cleared=(
                    disengagement_contact_cleared
                ),
                disengagement_force_mode_stop_acknowledged=(
                    disengagement_force_mode_stop_acknowledged
                ),
                recenter_position_error_m=recenter_position_error_m,
                recenter_command_acknowledged=recenter_command_acknowledged,
                disengagement_stationary_confirmed=(
                    disengagement_stationary_confirmed
                ),
                retare_baseline_consistent=retare_baseline_consistent,
            )
            return last_sample

        try:
            request = goal_handle.request
            part_name = str(request.part_name or "")
            calibration_id = str(request.calibration_id or "")
            profile_sha256 = str(request.profile_sha256 or "")
            trial_id = str(getattr(request, "trial_id", "") or "")
            if trial_id != trial_id.strip():
                raise ValueError("trial_id must not contain surrounding whitespace")
            if part_name not in INSERT_SUPPORTED_PART_NAMES:
                raise ValueError(
                    "part_name must be one of the exact supported tokens "
                    f"{list(INSERT_SUPPORTED_PART_NAMES)}"
                )
            selected_hard_caps = _insert_hard_caps(part_name)
            selected_cap_error = _insert_hard_cap_error(part_name)
            if selected_cap_error:
                raise ValueError(selected_cap_error)
            automatic_withdrawal_enabled = _insert_automatic_withdrawal_enabled(
                part_name
            )
            advanced_recovery_enabled = bool(
                automatic_withdrawal_enabled
                and all(
                    field_name in selected_hard_caps
                    for field_name in (
                        "insert_max_contact_search_radius_m",
                        "insert_max_disengagement_cycles",
                        "insert_search_peck_retreat_m",
                        "insert_search_peck_interval_sec",
                    )
                )
            )
            hard_caps_sha256 = _insert_hard_caps_sha256(selected_hard_caps)
            requested_hard_caps_sha256 = str(
                getattr(request, "hard_caps_sha256", "") or ""
            )
            if requested_hard_caps_sha256 != hard_caps_sha256:
                raise ValueError(
                    "hard_caps_sha256 does not match the server-selected exact-part caps"
                )
            status.update(
                insert_selected_part_name=part_name,
                insert_selected_hard_caps=selected_hard_caps,
                insert_selected_hard_caps_error="",
                insert_selected_hard_caps_sha256=hard_caps_sha256,
            )
            self._write_active_goal_status(goal_handle, status)
            if not calibration_id or calibration_id != calibration_id.strip():
                raise ValueError("calibration_id is required without surrounding whitespace")
            if len(profile_sha256) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in profile_sha256
            ):
                raise ValueError("profile_sha256 must contain exactly 64 hexadecimal characters")
            start_server_trace()

            (
                force_depth_fraction,
                force_depth_axial_upper_n,
                force_depth_lateral_upper_n,
                force_depth_torque_upper_nm,
            ) = _validated_force_depth_profile(
                request,
                hard_caps=selected_hard_caps,
            )

            contact_speed_m_s = _bounded_insert_value(
                "contact_speed_m_s",
                request.contact_speed_m_s,
                UR5E_RTDE_INSERT_MAX_CONTACT_SPEED_M_S,
            )
            contact_force_delta_n = _bounded_insert_value(
                "contact_force_delta_n",
                request.contact_force_delta_n,
                UR5E_RTDE_INSERT_MAX_CONTACT_FORCE_DELTA_N,
            )
            engagement_progress_m = _bounded_insert_value(
                "engagement_progress_m",
                request.engagement_progress_m,
                UR5E_RTDE_INSERT_MAX_ENGAGEMENT_PROGRESS_M,
            )
            insertion_force_n = _bounded_insert_value(
                "insertion_force_n",
                request.insertion_force_n,
                float(selected_hard_caps["insert_max_insertion_force_n"]),
            )
            spiral_radius_m = _bounded_insert_value(
                "spiral_radius_m",
                request.spiral_radius_m,
                UR5E_RTDE_INSERT_MAX_SPIRAL_RADIUS_M,
                allow_zero=True,
            )
            spiral_pitch_m = _bounded_insert_value(
                "spiral_pitch_m",
                request.spiral_pitch_m,
                UR5E_RTDE_INSERT_MAX_SPIRAL_PITCH_M,
            )
            spiral_speed_m_s = _bounded_insert_value(
                "spiral_speed_m_s",
                request.spiral_speed_m_s,
                UR5E_RTDE_INSERT_MAX_SPIRAL_SPEED_M_S,
            )
            spiral_acceleration_m_s2 = _bounded_insert_value(
                "spiral_acceleration_m_s2",
                request.spiral_acceleration_m_s2,
                UR5E_RTDE_INSERT_MAX_SPIRAL_ACCELERATION_M_S2,
            )
            max_axial_force_n = _bounded_insert_value(
                "max_axial_force_n",
                request.max_axial_force_n,
                float(selected_hard_caps["insert_max_axial_force_n"]),
            )
            max_lateral_force_n = _bounded_insert_value(
                "max_lateral_force_n",
                request.max_lateral_force_n,
                float(selected_hard_caps["insert_max_lateral_force_n"]),
            )
            max_torque_nm = _bounded_insert_value(
                "max_torque_nm",
                request.max_torque_nm,
                float(selected_hard_caps["insert_max_torque_nm"]),
            )
            learned_axial_limit_scale, _depth_axial_limit_scale = (
                _insert_axial_soft_limit_scales(part_name)
            )
            insertion_force_n = min(
                insertion_force_n * _insert_force_command_scale(part_name),
                float(selected_hard_caps["insert_max_insertion_force_n"]),
                max_axial_force_n * learned_axial_limit_scale,
            )
            baseline_force_uncertainty_n = _bounded_insert_value(
                "baseline_force_uncertainty_n",
                request.baseline_force_uncertainty_n,
                float(selected_hard_caps["insert_max_lateral_force_n"]),
            )
            baseline_torque_uncertainty_nm = _bounded_insert_value(
                "baseline_torque_uncertainty_nm",
                request.baseline_torque_uncertainty_nm,
                float(selected_hard_caps["insert_max_tool_flange_torque_nm"]),
            )
            tilt_tolerance_rad = _bounded_insert_value(
                "tilt_tolerance_rad",
                request.tilt_tolerance_rad,
                UR5E_RTDE_INSERT_MAX_TILT_TOLERANCE_RAD,
            )
            seated_depth_tolerance_m = _bounded_insert_value(
                "seated_depth_tolerance_m",
                request.seated_depth_tolerance_m,
                UR5E_RTDE_INSERT_MAX_SEATED_DEPTH_TOLERANCE_M,
            )
            settle_time_sec = _bounded_insert_value(
                "settle_time_sec",
                request.settle_time_sec,
                UR5E_RTDE_INSERT_MAX_SETTLE_TIME_SEC,
            )
            timeout_sec = _bounded_insert_value(
                "timeout_sec",
                request.timeout_sec,
                UR5E_RTDE_INSERT_MAX_TIMEOUT_SEC,
            )
            max_travel_m = float(UR5E_RTDE_INSERT_MAX_TRAVEL_M or math.nan)
            soft_filter_window_sec = float(
                UR5E_RTDE_INSERT_SOFT_FILTER_WINDOW_SEC or math.nan
            )
            soft_overload_hold_sec = float(
                UR5E_RTDE_INSERT_SOFT_OVERLOAD_HOLD_SEC or math.nan
            )
            relief_unload_dwell_sec = float(
                UR5E_RTDE_INSERT_RELIEF_UNLOAD_DWELL_SEC or math.nan
            )
            relief_clear_dwell_sec = float(
                UR5E_RTDE_INSERT_RELIEF_CLEAR_DWELL_SEC or math.nan
            )
            relief_clear_hysteresis_ratio = float(
                UR5E_RTDE_INSERT_RELIEF_CLEAR_HYSTERESIS_RATIO or math.nan
            )
            relief_timeout_sec = float(UR5E_RTDE_INSERT_RELIEF_TIMEOUT_SEC or math.nan)
            relief_axial_force_ratio = float(
                UR5E_RTDE_INSERT_RELIEF_AXIAL_FORCE_RATIO or math.nan
            )
            relief_reverse_force_ratio = float(
                UR5E_RTDE_INSERT_RELIEF_REVERSE_FORCE_RATIO or math.nan
            )
            relief_resume_ramp_sec = float(
                UR5E_RTDE_INSERT_RELIEF_RESUME_RAMP_SEC or math.nan
            )
            relief_search_force_ratio = float(
                UR5E_RTDE_INSERT_RELIEF_SEARCH_FORCE_RATIO or math.nan
            )
            relief_search_speed_ratio = float(
                UR5E_RTDE_INSERT_RELIEF_SEARCH_SPEED_RATIO or math.nan
            )
            relief_backoff_step_m = float(
                UR5E_RTDE_INSERT_RELIEF_BACKOFF_STEP_M or math.nan
            )
            max_relief_retreat_m = float(
                selected_hard_caps.get("insert_max_relief_retreat_m")
                or math.nan
            )
            relief_stationary_speed_m_s = float(
                UR5E_RTDE_INSERT_RELIEF_STATIONARY_SPEED_M_S or math.nan
            )
            relief_stationary_angular_speed_rad_s = float(
                UR5E_RTDE_INSERT_RELIEF_STATIONARY_ANGULAR_SPEED_RAD_S
                or math.nan
            )
            max_relief_cycles = int(UR5E_RTDE_INSERT_MAX_RELIEF_CYCLES or 0)
            if not automatic_withdrawal_enabled:
                max_relief_cycles = 0
            max_contact_search_radius_m = float(
                selected_hard_caps.get("insert_max_contact_search_radius_m")
                or spiral_radius_m
            )
            max_disengagement_cycles = int(
                selected_hard_caps.get("insert_max_disengagement_cycles") or 0
            )
            search_peck_retreat_limit_m = float(
                selected_hard_caps.get("insert_search_peck_retreat_m") or 0.0
            )
            search_peck_interval_sec = float(
                selected_hard_caps.get("insert_search_peck_interval_sec")
                or math.inf
            )
            start_position_tolerance_m = float(
                UR5E_RTDE_INSERT_START_POSITION_TOLERANCE_M or math.nan
            )
            start_orientation_tolerance_rad = float(
                UR5E_RTDE_INSERT_START_ORIENTATION_TOLERANCE_RAD or math.nan
            )
            if contact_force_delta_n > max_axial_force_n:
                raise ValueError("contact_force_delta_n exceeds max_axial_force_n")
            if insertion_force_n > max_axial_force_n * learned_axial_limit_scale:
                raise ValueError("insertion_force_n exceeds max_axial_force_n")
            if contact_force_delta_n > insertion_force_n:
                raise ValueError(
                    "contact_force_delta_n exceeds insertion_force_n, so stable "
                    "bottom-contact evidence cannot be established"
                )
            for field_name, recipe_limit, hard_ceiling in (
                (
                    "max_axial_force_n",
                    max_axial_force_n,
                    selected_hard_caps["insert_max_axial_force_n"],
                ),
                (
                    "max_lateral_force_n",
                    max_lateral_force_n,
                    selected_hard_caps["insert_max_lateral_force_n"],
                ),
                (
                    "max_torque_nm",
                    max_torque_nm,
                    selected_hard_caps["insert_max_torque_nm"],
                ),
            ):
                if recipe_limit >= float(hard_ceiling or math.nan):
                    raise ValueError(
                        f"{field_name} must be strictly below its independent hard ceiling"
                    )
            guarded_uncertainty_scale = (
                0.0 if _insert_full_hard_ceiling_enabled(part_name) else 1.0
            )
            guarded_axial_force_ceiling_n = float(
                selected_hard_caps["insert_max_axial_force_n"]
            ) - baseline_force_uncertainty_n * guarded_uncertainty_scale
            guarded_lateral_force_ceiling_n = float(
                selected_hard_caps["insert_max_lateral_force_n"]
            ) - baseline_force_uncertainty_n * guarded_uncertainty_scale
            guarded_torque_ceiling_nm = float(
                selected_hard_caps["insert_max_torque_nm"]
            ) - baseline_torque_uncertainty_nm * guarded_uncertainty_scale
            for field_name, recipe_limit, guarded_ceiling in (
                (
                    "max_axial_force_n",
                    max_axial_force_n,
                    guarded_axial_force_ceiling_n,
                ),
                (
                    "max_lateral_force_n",
                    max_lateral_force_n,
                    guarded_lateral_force_ceiling_n,
                ),
                (
                    "max_torque_nm",
                    max_torque_nm,
                    guarded_torque_ceiling_nm,
                ),
            ):
                if recipe_limit >= guarded_ceiling:
                    raise ValueError(
                        f"{field_name} does not leave its measured baseline uncertainty "
                        "reserve below the independent hard ceiling"
                    )

            control_error = self._connect_control_for_goal()
            if control_error:
                raise RuntimeError(control_error)
            required_methods = (
                "forceMode",
                "forceModeStop",
                "getTCPOffset",
                "isPoseWithinSafetyLimits",
                "stopL",
            )
            missing_methods = [
                name for name in required_methods if not callable(getattr(self.control, name, None))
            ]
            if missing_methods:
                raise RuntimeError(f"missing RTDE insertion methods {missing_methods}")
            if not self._joint_states_fresh() or self._read_actual_q() is None:
                raise RuntimeError("UR5e RTDE feedback stale or missing")
            program_error = self._ensure_control_program_for_goal()
            if program_error:
                raise RuntimeError(program_error)

            for field_name, pose_message in (
                ("expected_start_tool0_pose", request.expected_start_tool0_pose),
                ("target_tool0_pose", request.target_tool0_pose),
            ):
                frame_id = str(pose_message.header.frame_id or "").strip()
                if frame_id != "world":
                    raise ValueError(
                        f"{field_name} requires frame_id=world, received {frame_id or '(empty)'}"
                    )
            expected_start_world_tool0 = _transform_from_pose_stamped(
                request.expected_start_tool0_pose
            )
            target_world_tool0 = _transform_from_pose_stamped(request.target_tool0_pose)
            for pose_name, pose_value in (
                ("expected_start_tool0_pose", expected_start_world_tool0),
                ("target_tool0_pose", target_world_tool0),
            ):
                workspace_error = _workspace_error(pose_value)
                if workspace_error:
                    raise ValueError(f"{pose_name}: {workspace_error}")

            insertion_axis_world = _normalize_vector(
                (
                    float(request.insertion_axis_world.x),
                    float(request.insertion_axis_world.y),
                    float(request.insertion_axis_world.z),
                )
            )
            target_delta = tuple(
                target_world_tool0[0][index] - expected_start_world_tool0[0][index]
                for index in range(3)
            )
            target_depth_m = _vector_dot(target_delta, insertion_axis_world)
            if _insert_axis_only_target_enabled(part_name):
                target_delta = tuple(
                    target_depth_m * insertion_axis_world[index]
                    for index in range(3)
                )
                target_world_tool0 = (
                    tuple(
                        expected_start_world_tool0[0][index] + target_delta[index]
                        for index in range(3)
                    ),
                    expected_start_world_tool0[1],
                )
            target_lateral = tuple(
                target_delta[index] - target_depth_m * insertion_axis_world[index]
                for index in range(3)
            )
            target_travel_m = _vector_norm(target_delta)
            if target_depth_m <= 0.0:
                raise ValueError(
                    "target_tool0_pose must lie in the positive insertion_axis_world direction"
                )
            if target_travel_m > max_travel_m:
                raise ValueError(
                    f"target insertion travel {target_travel_m:.6f} m exceeds {max_travel_m:.6f} m"
                )
            target_lateral_limit_m = (
                max_contact_search_radius_m
                if advanced_recovery_enabled
                else start_position_tolerance_m
            )
            if _vector_norm(target_lateral) > target_lateral_limit_m:
                raise ValueError(
                    "target_tool0_pose is outside the protected lateral start "
                    f"boundary of {target_lateral_limit_m:.6f} m"
                )
            _unused_position_error, requested_orientation_change = _pose_errors(
                expected_start_world_tool0,
                target_world_tool0,
            )
            if requested_orientation_change > start_orientation_tolerance_rad:
                raise ValueError(
                    "MoveUR5eInsert does not perform orientation search; start and target "
                    "orientation differ by "
                    f"{requested_orientation_change:.6f} rad"
                )
            if engagement_progress_m > target_depth_m + seated_depth_tolerance_m:
                raise ValueError("engagement_progress_m exceeds the available insertion depth")

            (
                world_base,
                frame_message,
                frame_position_error,
                frame_orientation_error,
            ) = self._validated_cartesian_world_base()
            tool0_tcp = self._active_tcp_offset()
            insertion_axis_base = _normalize_vector(
                _rotate_vector(_quaternion_conjugate(world_base[1]), insertion_axis_world)
            )
            expected_base_tool0 = _compose_transform(
                _inverse_transform(world_base),
                expected_start_world_tool0,
            )
            expected_base_tcp = _compose_transform(expected_base_tool0, tool0_tcp)
            target_base_tool0 = _compose_transform(
                _inverse_transform(world_base),
                target_world_tool0,
            )
            target_base_tcp = _compose_transform(target_base_tool0, tool0_tcp)
            within_safety_limits = self.control.isPoseWithinSafetyLimits
            if not bool(within_safety_limits(_rtde_pose_from_transform(expected_base_tcp))):
                raise ValueError(
                    "UR controller rejected expected_start_tool0_pose as outside safety limits"
                )
            if not bool(within_safety_limits(_rtde_pose_from_transform(target_base_tcp))):
                raise ValueError(
                    "UR controller rejected target_tool0_pose as outside safety limits"
                )
            actual_base_tcp = self._read_actual_tcp_transform()
            if actual_base_tcp is None:
                raise RuntimeError("UR5e actual TCP pose is unavailable")
            actual_tcp_speed = self._read_actual_tcp_speed()
            if actual_tcp_speed is None:
                raise RuntimeError("UR5e actual_TCP_speed is unavailable")
            actual_world_tool0 = self._world_tool0_from_actual_tcp(
                actual_base_tcp,
                world_base=world_base,
                tool0_tcp=tool0_tcp,
            )
            start_position_error, start_orientation_error = _pose_errors(
                actual_world_tool0,
                expected_start_world_tool0,
            )
            if start_position_error > start_position_tolerance_m:
                raise ValueError(
                    f"actual start position differs by {start_position_error:.6f} m; "
                    f"limit is {start_position_tolerance_m:.6f} m"
                )
            if start_orientation_error > start_orientation_tolerance_rad:
                raise ValueError(
                    f"actual start orientation differs by {start_orientation_error:.6f} rad; "
                    f"limit is {start_orientation_tolerance_rad:.6f} rad"
                )

            status.update(
                state="checking",
                message="confirming stationary hold before zeroing software TCP force bias",
                blocked_reason="",
                insert_phase="zeroing_force",
                part_name=part_name,
                calibration_id=calibration_id,
                profile_sha256=profile_sha256,
                hard_caps_sha256=hard_caps_sha256,
                trial_id=trial_id,
                server_trace_id=server_trace_id,
                server_trace_path=server_trace_path,
                server_trace_status=server_trace_status,
                server_trace_complete=False,
                server_trace_sample_count=server_trace_sample_count,
                target_insertion_depth_m=target_depth_m,
                cartesian_frame_validation_message=frame_message,
                cartesian_frame_position_error_m=frame_position_error,
                cartesian_frame_orientation_error_rad=frame_orientation_error,
                tcp_force_feedback_ready=True,
                insert_function_ready=False,
                insert_readiness_message="Insertion goal owns the motion slot",
            )
            self._write_active_goal_status(goal_handle, status)
            if not self._confirm_stationary_after_stop(
                linear_speed_limit_m_s=relief_stationary_speed_m_s,
                angular_speed_limit_rad_s=(
                    relief_stationary_angular_speed_rad_s
                ),
            ):
                raise RuntimeError(
                    "UR5e did not establish a stationary joint-velocity hold before "
                    "zeroing the insertion force bias"
                )
            if goal_handle.is_cancel_requested:
                raise _InsertCanceled("canceled before motion")
            (
                checking_depth_m,
                checking_depth_error_m,
                checking_lateral_offset_m,
                _checking_tilt_error_rad,
            ) = _insertion_pose_metrics(
                actual_world_tool0,
                expected_start_world_tool0,
                target_world_tool0,
                insertion_axis_world,
            )
            force_samples: list[list[float]] = []
            for _sample_index in range(5):
                if goal_handle.is_cancel_requested:
                    raise _InsertCanceled("canceled before motion")
                if not rclpy.ok():
                    raise RuntimeError("ROS shutdown interrupted UR5e insertion before motion")
                require_fresh_insert_feedback()
                force_sample = self._read_actual_tcp_force()
                if force_sample is None:
                    raise RuntimeError("actual_TCP_force is unavailable while zeroing bias")
                force_samples.append(force_sample)
                if _sample_index == 0:
                    self._publish_insert_feedback(
                        goal_handle,
                        phase="checking",
                        trial_id=trial_id,
                        actual_world_tool0=actual_world_tool0,
                        insertion_depth_m=checking_depth_m,
                        depth_error_m=checking_depth_error_m,
                        lateral_offset_m=checking_lateral_offset_m,
                        search_radius_m=0.0,
                        axial_force_n=0.0,
                        lateral_force_n=0.0,
                        torque_nm=0.0,
                        contact_detected=False,
                        engagement_detected=False,
                        seated_detected=False,
                        force_bias_valid=False,
                        force_bias=[0.0] * 6,
                        actual_tcp_force=force_sample,
                        actual_tcp_speed=actual_tcp_speed,
                    )
                time.sleep(0.02)
            force_baseline_span_n = _vector_norm(
                tuple(
                    max(sample[index] for sample in force_samples)
                    - min(sample[index] for sample in force_samples)
                    for index in range(3)
                )
            )
            torque_baseline_span_nm = _vector_norm(
                tuple(
                    max(sample[index] for sample in force_samples)
                    - min(sample[index] for sample in force_samples)
                    for index in range(3, 6)
                )
            )
            if force_baseline_span_n >= contact_force_delta_n:
                raise RuntimeError(
                    "stationary TCP force baseline varied by "
                    f"{force_baseline_span_n:.3f} N; contact threshold is "
                    f"{contact_force_delta_n:.3f} N"
                )
            if torque_baseline_span_nm >= max_torque_nm * 0.50:
                raise RuntimeError(
                    "stationary TCP torque baseline varied by "
                    f"{torque_baseline_span_nm:.3f} Nm"
                )
            force_bias = [
                sum(sample[index] for sample in force_samples) / len(force_samples)
                for index in range(6)
            ]
            force_bias_valid = True
            baseline_base_tool0 = _compose_transform(
                actual_base_tcp,
                _inverse_transform(tool0_tcp),
            )
            baseline_tool0_tcp_offset_base = _rotate_vector(
                baseline_base_tool0[1],
                tool0_tcp[0],
            )
            (
                zeroed_raw_axial_force_n,
                zeroed_axial_force_n,
                zeroed_lateral_force_n,
                zeroed_torque_nm,
                zeroed_tool_flange_torque_nm,
                zeroed_tared_tcp_force,
            ) = _insertion_force_metrics(
                force_samples[-1],
                force_bias,
                insertion_axis_base,
                baseline_tool0_tcp_offset_base,
            )
            self._publish_insert_feedback(
                goal_handle,
                phase="zeroing_force",
                trial_id=trial_id,
                actual_world_tool0=actual_world_tool0,
                insertion_depth_m=checking_depth_m,
                depth_error_m=checking_depth_error_m,
                lateral_offset_m=checking_lateral_offset_m,
                search_radius_m=0.0,
                axial_force_n=zeroed_axial_force_n,
                raw_axial_force_n=zeroed_raw_axial_force_n,
                lateral_force_n=zeroed_lateral_force_n,
                torque_nm=zeroed_torque_nm,
                tool_flange_torque_nm=zeroed_tool_flange_torque_nm,
                contact_detected=False,
                engagement_detected=False,
                seated_detected=False,
                force_bias_valid=True,
                force_bias=force_bias,
                actual_tcp_force=force_samples[-1],
                tared_tcp_force=zeroed_tared_tcp_force,
                actual_tcp_speed=actual_tcp_speed,
            )
            execution_deadline = time.monotonic() + _insert_execution_timeout_sec(
                part_name,
                timeout_sec,
            )
            control_cycle_sec = max(1.0 / UR5E_RTDE_FREQUENCY_HZ, 0.02)
            engagement_hold_sec = max(0.10, min(settle_time_sec, 0.25))
            stall_hold_sec = max(0.10, min(settle_time_sec, 0.50))
            seated_hold_sec = max(0.10, settle_time_sec)
            force_filter_window_sec = soft_filter_window_sec
            contact_hold_sec = max(0.06, min(engagement_hold_sec, 0.10))
            progress_epsilon_m = max(
                1e-5,
                min(seated_depth_tolerance_m, engagement_progress_m) * 0.25,
            )
            rebound_tolerance_m = max(
                progress_epsilon_m,
                min(seated_depth_tolerance_m, engagement_progress_m * 0.5),
            )
            near_zero_axial_speed_m_s = max(
                1e-5,
                min(
                    contact_speed_m_s * 0.10,
                    seated_depth_tolerance_m / seated_hold_sec,
                ),
            )
            bottom_force_variation_n = max(0.5, contact_force_delta_n)
            status.update(
                state="executing",
                message="executing direct compliant UR5e insertion",
                insert_phase="seating",
                force_bias_valid=True,
                force_bias=list(force_bias),
            )
            self._write_active_goal_status(goal_handle, status)

            if bool(getattr(self, "_shutdown_requested", False)) or not rclpy.ok():
                raise RuntimeError("UR5e RTDE server stopped before insertion motion")
            dispatch_ready, dispatch_message, _dispatch_position, _dispatch_orientation = (
                self._cartesian_frame_validation()
            )
            if not dispatch_ready:
                raise ValueError(dispatch_message)
            motion_attempted = True
            status["insert_motion_settled"] = False
            with self._active_lock:
                if self._active_goal is goal_handle:
                    self._insert_motion_started = True
            actual_base_tcp = self._read_actual_tcp_transform()
            if actual_base_tcp is None:
                raise RuntimeError("UR5e actual TCP pose is unavailable before force mode")
            self._start_insert_force_mode(
                actual_base_tcp=actual_base_tcp,
                insertion_axis_base=insertion_axis_base,
                insertion_force_n=insertion_force_n,
                contact_speed_m_s=contact_speed_m_s,
                spiral_speed_m_s=spiral_speed_m_s,
                tilt_tolerance_rad=tilt_tolerance_rad,
            )

            phase = "seating"
            final_phase = phase
            deepest_depth_m = 0.0
            filtered_force_samples: deque[
                tuple[float, float, float, float, float]
            ] = deque()
            contact_candidate_since: float | None = None
            contact_candidate_depth_m = 0.0
            contact_reference_depth_m: float | None = None
            progress_reference_depth_m = 0.0
            progress_reference_at = time.monotonic()
            search_engagement_reference_depth_m: float | None = None
            engagement_candidate_since: float | None = None
            engagement_candidate_interruption_since: float | None = None
            engagement_candidate_peak_depth_m = 0.0
            engagement_depth_m = 0.0
            seated_candidate_since: float | None = None
            seated_force_min_n = math.inf
            seated_force_max_n = 0.0
            spiral_started_at: float | None = None
            spiral_theta = 0.0
            spiral_scale = spiral_pitch_m / (2.0 * math.pi)
            lateral_search_force_n = min(
                insertion_force_n,
                max_lateral_force_n * 0.50,
            )
            soft_overload_candidate_since: float | None = None
            relief_started_at: float | None = None
            relief_entry_depth_m = 0.0
            relief_clear_since: float | None = None
            relief_backoff_complete = False
            relief_backoff_committed = False
            relief_committed_backoff_m = 0.0
            resume_started_at: float | None = None
            frozen_search_resume_pending = False
            initial_force_bias = list(force_bias)
            spiral_phase_offset_rad = 0.0
            expanded_search_boundary_theta: float | None = None
            expanded_search_stage_radii_m = [
                radius_m
                for radius_m in (0.003, 0.005)
                if radius_m < max_contact_search_radius_m
            ]
            expanded_search_stage_radii_m.append(max_contact_search_radius_m)
            expanded_search_stage_index = 0
            tactile_candidate_since: float | None = None
            tactile_candidate_peak_depth_m = 0.0
            tactile_candidate_load_score = math.inf
            tactile_candidate_best_world_tool0: RigidTransform | None = None
            tactile_candidate_best_lateral_force_n = 0.0
            tactile_candidate_best_torque_nm = 0.0
            disengagement_started_at: float | None = None
            disengagement_deadline: float | None = None
            disengagement_clear_since: float | None = None
            disengagement_entry_depth_m = 0.0
            disengagement_progress_reference_depth_m = 0.0
            disengagement_progress_reference_at: float | None = None
            retry_started_at: float | None = None
            search_peck_state = "idle"
            search_peck_started_at: float | None = None
            search_peck_entry_depth_m = 0.0
            search_peck_target_retreat_m = 0.0
            search_peck_retreat_m = 0.0
            search_peck_next_at = math.inf
            search_origin_world_tool0: RigidTransform | None = None
            search_guard_entry_theta = 0.0
            search_escape_candidate_since: float | None = None

            def update_tactile_center_candidate(
                *,
                sampled_at: float,
                actual_world_tool0: RigidTransform,
                insertion_depth_m: float,
                filtered_lateral_force_n: float,
                filtered_torque_nm: float,
                profile_load_ok: bool,
            ) -> None:
                """Retain the deepest stable, low-load tactile pin-entry evidence."""
                nonlocal tactile_candidate_since
                nonlocal tactile_candidate_peak_depth_m
                nonlocal tactile_candidate_load_score
                nonlocal tactile_candidate_best_world_tool0
                nonlocal tactile_candidate_best_lateral_force_n
                nonlocal tactile_candidate_best_torque_nm
                nonlocal tactile_center_world_tool0
                nonlocal tactile_center_depth_m
                nonlocal tactile_center_confidence
                nonlocal tactile_center_evidence_sha256
                if (
                    not contact_detected
                    or not profile_load_ok
                    or insertion_depth_m < progress_epsilon_m
                ):
                    tactile_candidate_since = None
                    tactile_candidate_peak_depth_m = 0.0
                    tactile_candidate_load_score = math.inf
                    tactile_candidate_best_world_tool0 = None
                    return
                if insertion_depth_m < (
                    tactile_candidate_peak_depth_m - rebound_tolerance_m
                ):
                    tactile_candidate_since = None
                    tactile_candidate_peak_depth_m = 0.0
                    tactile_candidate_load_score = math.inf
                    tactile_candidate_best_world_tool0 = None
                    return
                load_score = (
                    filtered_lateral_force_n
                    / max(current_force_depth_lateral_upper_n, 1e-9)
                    + filtered_torque_nm
                    / max(current_force_depth_torque_upper_nm, 1e-9)
                )
                if tactile_candidate_since is None:
                    tactile_candidate_since = sampled_at
                    tactile_candidate_peak_depth_m = insertion_depth_m
                    tactile_candidate_load_score = load_score
                    tactile_candidate_best_world_tool0 = actual_world_tool0
                    tactile_candidate_best_lateral_force_n = (
                        filtered_lateral_force_n
                    )
                    tactile_candidate_best_torque_nm = filtered_torque_nm
                else:
                    better_depth = insertion_depth_m > (
                        tactile_candidate_peak_depth_m + progress_epsilon_m
                    )
                    tied_depth = abs(
                        insertion_depth_m - tactile_candidate_peak_depth_m
                    ) <= progress_epsilon_m
                    if better_depth or (
                        tied_depth and load_score < tactile_candidate_load_score
                    ):
                        tactile_candidate_peak_depth_m = insertion_depth_m
                        tactile_candidate_load_score = load_score
                        tactile_candidate_best_world_tool0 = actual_world_tool0
                        tactile_candidate_best_lateral_force_n = (
                            filtered_lateral_force_n
                        )
                        tactile_candidate_best_torque_nm = filtered_torque_nm
                    if sampled_at - tactile_candidate_since < engagement_hold_sec:
                        return
                if insertion_depth_m + rebound_tolerance_m < (
                    tactile_candidate_peak_depth_m
                ):
                    return
                if tactile_candidate_best_world_tool0 is None:
                    return
                tactile_center_world_tool0 = tactile_candidate_best_world_tool0
                tactile_center_depth_m = tactile_candidate_peak_depth_m
                tactile_center_confidence = min(
                    1.0,
                    max(
                        0.0,
                        tactile_candidate_peak_depth_m
                        / max(target_depth_m, 1e-9),
                    ),
                ) * min(
                    1.0,
                    max(0.0, 1.0 - tactile_candidate_load_score / 2.0),
                )
                evidence = {
                    "trial_id": trial_id,
                    "depth_m": float(tactile_candidate_peak_depth_m),
                    "pose": _transform_status_payload(
                        tactile_candidate_best_world_tool0
                    ),
                    "filtered_lateral_force_n": float(
                        tactile_candidate_best_lateral_force_n
                    ),
                    "filtered_torque_nm": float(
                        tactile_candidate_best_torque_nm
                    ),
                    "confidence": float(tactile_center_confidence),
                }
                tactile_center_evidence_sha256 = hashlib.sha256(
                    json.dumps(
                        evidence,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()

            def begin_disengagement(reason: str, *, insertion_depth_m: float) -> None:
                """Start full-clearance recovery authorized by exact-part policy values."""
                nonlocal phase
                nonlocal final_phase
                nonlocal disengagement_cycle_count
                nonlocal last_disengagement_reason
                nonlocal disengagement_started_at
                nonlocal disengagement_deadline
                nonlocal disengagement_clear_since
                nonlocal disengagement_entry_depth_m
                nonlocal disengagement_progress_reference_depth_m
                nonlocal disengagement_progress_reference_at
                nonlocal disengagement_withdrawal_m
                nonlocal disengagement_contact_cleared
                nonlocal disengagement_force_mode_stop_acknowledged
                nonlocal recenter_command_acknowledged
                nonlocal disengagement_stationary_confirmed
                nonlocal retare_baseline_consistent
                nonlocal search_peck_state
                nonlocal search_peck_started_at
                nonlocal search_peck_retreat_m
                nonlocal search_peck_next_at
                nonlocal engagement_candidate_since
                nonlocal engagement_candidate_interruption_since
                nonlocal engagement_candidate_peak_depth_m
                if not advanced_recovery_enabled or max_disengagement_cycles <= 0:
                    raise _InsertSearchExhausted(reason)
                if disengagement_cycle_count >= max_disengagement_cycles:
                    raise _InsertSearchExhausted(
                        f"{reason}; protected disengagement cycle limit "
                        f"{max_disengagement_cycles} exhausted"
                    )
                disengagement_cycle_count += 1
                last_disengagement_reason = str(reason)
                disengagement_started_at = time.monotonic()
                disengagement_timeout_sec = max(
                    relief_timeout_sec,
                    target_depth_m / contact_speed_m_s + relief_timeout_sec,
                )
                disengagement_deadline = min(
                    execution_deadline,
                    disengagement_started_at + disengagement_timeout_sec,
                )
                disengagement_clear_since = None
                disengagement_entry_depth_m = max(0.0, insertion_depth_m)
                disengagement_progress_reference_depth_m = 0.0
                disengagement_progress_reference_at = None
                disengagement_withdrawal_m = 0.0
                disengagement_contact_cleared = False
                disengagement_force_mode_stop_acknowledged = False
                recenter_command_acknowledged = False
                disengagement_stationary_confirmed = False
                retare_baseline_consistent = False
                search_peck_state = "idle"
                search_peck_started_at = None
                search_peck_retreat_m = 0.0
                search_peck_next_at = math.inf
                engagement_candidate_since = None
                engagement_candidate_interruption_since = None
                engagement_candidate_peak_depth_m = 0.0
                phase = "cocked"
                final_phase = phase
                append_server_trace(
                    "cocked",
                    {
                        **last_sample,
                        "disengagement_cycle_count": disengagement_cycle_count,
                        "reason": last_disengagement_reason,
                        "tactile_center_valid": (
                            tactile_center_world_tool0 is not None
                        ),
                        "tactile_center_evidence_sha256": (
                            tactile_center_evidence_sha256
                        ),
                    },
                )

            def require_disengagement_deadline(sampled_at: float) -> None:
                """Fail when full protected disengagement exceeds its motion budget."""
                nonlocal last_disengagement_reason
                if (
                    disengagement_started_at is None
                    or disengagement_deadline is None
                    or sampled_at < disengagement_deadline
                ):
                    return
                disengagement_elapsed_sec = max(
                    0.0,
                    sampled_at - disengagement_started_at,
                )
                disengagement_trigger_reason = last_disengagement_reason
                last_disengagement_reason = (
                    f"{disengagement_trigger_reason}; {part_name} disengagement "
                    "timed out after "
                    f"{disengagement_elapsed_sec:.3f} s before contact cleared"
                )
                raise _InsertSearchExhausted(last_disengagement_reason)

            def require_disengagement_progress(
                *,
                sampled_at: float,
                insertion_depth_m: float,
            ) -> None:
                """Require bounded outward progress once reverse withdrawal starts."""
                nonlocal disengagement_progress_reference_depth_m
                nonlocal disengagement_progress_reference_at
                nonlocal last_disengagement_reason
                if disengagement_progress_reference_at is None:
                    disengagement_progress_reference_depth_m = insertion_depth_m
                    disengagement_progress_reference_at = sampled_at
                    return
                if insertion_depth_m <= (
                    disengagement_progress_reference_depth_m - progress_epsilon_m
                ):
                    disengagement_progress_reference_depth_m = insertion_depth_m
                    disengagement_progress_reference_at = sampled_at
                    return
                no_progress_elapsed_sec = max(
                    0.0,
                    sampled_at - disengagement_progress_reference_at,
                )
                if no_progress_elapsed_sec < relief_timeout_sec:
                    return
                disengagement_trigger_reason = last_disengagement_reason
                last_disengagement_reason = (
                    f"{disengagement_trigger_reason}; {part_name} disengagement made no "
                    "protected withdrawal progress "
                    f"for {no_progress_elapsed_sec:.3f} s before contact cleared"
                )
                raise _InsertSearchExhausted(last_disengagement_reason)

            def complete_disengagement_recenter() -> None:  # noqa: C901, PLR0912, PLR0915
                """Retract to the exact start, recenter while clear, and retry."""
                nonlocal phase
                nonlocal final_phase
                nonlocal force_bias
                nonlocal force_bias_valid
                nonlocal contact_detected
                nonlocal engagement_detected
                nonlocal contact_candidate_since
                nonlocal contact_candidate_depth_m
                nonlocal contact_reference_depth_m
                nonlocal search_engagement_reference_depth_m
                nonlocal engagement_candidate_since
                nonlocal engagement_candidate_interruption_since
                nonlocal engagement_candidate_peak_depth_m
                nonlocal engagement_depth_m
                nonlocal progress_reference_depth_m
                nonlocal progress_reference_at
                nonlocal filtered_force_samples
                nonlocal soft_overload_candidate_since
                nonlocal soft_overload_duration_sec
                nonlocal recenter_position_error_m
                nonlocal recenter_command_acknowledged
                nonlocal disengagement_stationary_confirmed
                nonlocal disengagement_contact_cleared
                nonlocal disengagement_force_mode_stop_acknowledged
                nonlocal retare_baseline_consistent
                nonlocal retry_started_at
                nonlocal spiral_theta
                nonlocal spiral_phase_offset_rad
                nonlocal expanded_search_boundary_theta
                nonlocal expanded_search_stage_index
                nonlocal disengagement_withdrawal_m
                nonlocal disengagement_progress_reference_depth_m
                nonlocal disengagement_progress_reference_at
                nonlocal search_origin_world_tool0
                nonlocal search_guard_entry_theta
                nonlocal search_escape_candidate_since
                if (
                    world_base is None
                    or tool0_tcp is None
                    or expected_start_world_tool0 is None
                    or insertion_axis_world is None
                    or insertion_axis_base is None
                ):
                    raise RuntimeError(
                        f"{part_name} disengagement recenter geometry is unavailable"
                    )
                if not stop_and_confirm():
                    raise RuntimeError(
                        f"{part_name} disengagement did not receive stop and stationary "
                        "acknowledgements"
                    )
                disengagement_force_mode_stop_acknowledged = (
                    force_mode_stop_acknowledged
                )
                disengagement_stationary_confirmed = stationary_confirmed
                current_world_tool0 = final_world_tool0
                current_translation = current_world_tool0[0]
                current_delta = tuple(
                    current_translation[index]
                    - expected_start_world_tool0[0][index]
                    for index in range(3)
                )
                current_axial_depth_m = _vector_dot(
                    current_delta,
                    insertion_axis_world,
                )
                exact_start_depth_translation = tuple(
                    current_translation[index]
                    - current_axial_depth_m * insertion_axis_world[index]
                    for index in range(3)
                )
                exact_start_depth_world_tool0 = (
                    exact_start_depth_translation,
                    expected_start_world_tool0[1],
                )
                workspace_error = _workspace_error(
                    exact_start_depth_world_tool0
                )
                if workspace_error:
                    raise RuntimeError(
                        f"{part_name} exact pre-insertion withdrawal target is outside the "
                        "workspace: " + workspace_error
                    )
                exact_start_depth_base_tool0 = _compose_transform(
                    _inverse_transform(world_base),
                    exact_start_depth_world_tool0,
                )
                exact_start_depth_base_tcp = _compose_transform(
                    exact_start_depth_base_tool0,
                    tool0_tcp,
                )
                if not bool(
                    self.control.isPoseWithinSafetyLimits(
                        _rtde_pose_from_transform(exact_start_depth_base_tcp)
                    )
                ):
                    raise RuntimeError(
                        f"UR controller rejected the {part_name} exact pre-insertion "
                        "withdrawal target"
                    )
                phase = "disengaging"
                final_phase = phase
                status.update(
                    insert_phase=phase,
                    message=(
                        f"{part_name} load-clear dwell confirmed — returning to the exact "
                        "retained pre-insertion depth"
                    ),
                )
                self._write_active_goal_status(goal_handle, status)
                disengagement_progress_reference_depth_m = max(
                    0.0,
                    current_axial_depth_m,
                )
                disengagement_progress_reference_at = time.monotonic()
                withdrawal_deadline = min(
                    execution_deadline,
                    disengagement_deadline
                    if disengagement_deadline is not None
                    else execution_deadline,
                )
                while rclpy.ok() and time.monotonic() < withdrawal_deadline:
                    require_disengagement_deadline(time.monotonic())
                    if goal_handle.is_cancel_requested:
                        raise _InsertCanceled(
                            f"canceled while withdrawing {part_name}"
                        )
                    self._execute_insert_servo_pose(
                        exact_start_depth_base_tcp,
                        speed_m_s=contact_speed_m_s,
                        acceleration_m_s2=spiral_acceleration_m_s2,
                        cycle_sec=control_cycle_sec,
                    )
                    withdrawal_sample = sample("disengaging")
                    actual_world_tool0 = withdrawal_sample["actual_world_tool0"]
                    withdrawal_position_error_m, withdrawal_orientation_error_rad = (
                        _pose_errors(
                            actual_world_tool0,
                            exact_start_depth_world_tool0,
                        )
                    )
                    disengagement_withdrawal_m = max(
                        disengagement_withdrawal_m,
                        disengagement_entry_depth_m
                        - float(withdrawal_sample["insertion_depth_m"]),
                    )
                    withdrawal_sampled_at = time.monotonic()
                    require_disengagement_progress(
                        sampled_at=withdrawal_sampled_at,
                        insertion_depth_m=float(
                            withdrawal_sample["insertion_depth_m"]
                        ),
                    )
                    status.update(
                        insert_phase="disengaging",
                        insert_disengagement_withdrawal_m=(
                            disengagement_withdrawal_m
                        ),
                        insert_disengagement_contact_cleared=False,
                    )
                    self._write_active_goal_status(goal_handle, status)
                    if (
                        withdrawal_position_error_m
                        <= min(
                            UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M,
                            progress_epsilon_m,
                        )
                        and withdrawal_orientation_error_rad
                        <= start_orientation_tolerance_rad
                    ):
                        break
                    time.sleep(control_cycle_sec)
                else:
                    require_disengagement_deadline(time.monotonic())
                    raise _InsertSearchExhausted(
                        f"{part_name} exact pre-insertion withdrawal timed out before "
                        "contact cleared"
                    )
                if not self._stop_insert_servo():
                    raise RuntimeError(
                        f"{part_name} exact pre-insertion withdrawal servoStop was not "
                        "acknowledged"
                    )
                disengagement_stationary_confirmed = (
                    self._confirm_stationary_after_stop(
                        linear_speed_limit_m_s=relief_stationary_speed_m_s,
                        angular_speed_limit_rad_s=(
                            relief_stationary_angular_speed_rad_s
                        ),
                    )
                )
                if not disengagement_stationary_confirmed:
                    raise RuntimeError(
                        f"{part_name} exact pre-insertion withdrawal did not establish "
                        "stationary feedback"
                    )
                phase = "recentering"
                final_phase = phase
                recenter_world_tool0 = expected_start_world_tool0
                if tactile_center_world_tool0 is not None:
                    candidate_translation = tuple(
                        tactile_center_world_tool0[0][index]
                        - tactile_center_depth_m * insertion_axis_world[index]
                        for index in range(3)
                    )
                    candidate_lateral = tuple(
                        candidate_translation[index]
                        - expected_start_world_tool0[0][index]
                        for index in range(3)
                    )
                    candidate_axial = _vector_dot(
                        candidate_lateral,
                        insertion_axis_world,
                    )
                    candidate_translation = tuple(
                        candidate_translation[index]
                        - candidate_axial * insertion_axis_world[index]
                        for index in range(3)
                    )
                    candidate_offset = _vector_norm(
                        tuple(
                            candidate_translation[index]
                            - expected_start_world_tool0[0][index]
                            for index in range(3)
                        )
                    )
                    if candidate_offset <= max_contact_search_radius_m:
                        recenter_world_tool0 = (
                            candidate_translation,
                            expected_start_world_tool0[1],
                        )
                workspace_error = _workspace_error(recenter_world_tool0)
                if workspace_error:
                    raise RuntimeError(
                        f"{part_name} tactile recenter target is outside the workspace: "
                        + workspace_error
                    )
                recenter_base_tool0 = _compose_transform(
                    _inverse_transform(world_base),
                    recenter_world_tool0,
                )
                recenter_base_tcp = _compose_transform(
                    recenter_base_tool0,
                    tool0_tcp,
                )
                if not bool(
                    self.control.isPoseWithinSafetyLimits(
                        _rtde_pose_from_transform(recenter_base_tcp)
                    )
                ):
                    raise RuntimeError(
                        f"UR controller rejected the {part_name} tactile recenter target"
                    )
                recenter_deadline = min(
                    execution_deadline,
                    time.monotonic() + max(1.0, target_depth_m / contact_speed_m_s),
                )
                recenter_command_acknowledged = True
                while rclpy.ok() and time.monotonic() < recenter_deadline:
                    if goal_handle.is_cancel_requested:
                        raise _InsertCanceled(
                            f"canceled while recentering {part_name}"
                        )
                    self._execute_insert_servo_pose(
                        recenter_base_tcp,
                        speed_m_s=contact_speed_m_s,
                        acceleration_m_s2=spiral_acceleration_m_s2,
                        cycle_sec=control_cycle_sec,
                    )
                    recenter_sample = sample("recentering")
                    actual_world_tool0 = recenter_sample["actual_world_tool0"]
                    recenter_position_error_m, recenter_orientation_error_rad = (
                        _pose_errors(actual_world_tool0, recenter_world_tool0)
                    )
                    status.update(
                        insert_phase="recentering",
                        insert_recenter_position_error_m=(
                            recenter_position_error_m
                        ),
                        insert_recenter_command_acknowledged=True,
                    )
                    self._write_active_goal_status(goal_handle, status)
                    if (
                        recenter_position_error_m
                        <= UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M
                        and recenter_orientation_error_rad
                        <= start_orientation_tolerance_rad
                    ):
                        break
                    time.sleep(control_cycle_sec)
                else:
                    raise RuntimeError(f"{part_name} tactile recenter timed out")
                if not self._stop_insert_servo():
                    raise RuntimeError(
                        f"{part_name} tactile recenter servoStop was not acknowledged"
                    )
                disengagement_stationary_confirmed = (
                    self._confirm_stationary_after_stop(
                        linear_speed_limit_m_s=relief_stationary_speed_m_s,
                        angular_speed_limit_rad_s=(
                            relief_stationary_angular_speed_rad_s
                        ),
                    )
                )
                if not disengagement_stationary_confirmed:
                    raise RuntimeError(
                        f"{part_name} tactile recenter did not establish stationary feedback"
                    )
                phase = "retaring"
                final_phase = phase
                status.update(
                    insert_phase=phase,
                    message=f"retaring {part_name} after tactile recenter",
                )
                self._write_active_goal_status(goal_handle, status)
                retare_samples: list[list[float]] = []
                for _sample_index in range(5):
                    retare_sample = sample("retaring")
                    actual_tcp_force = list(retare_sample["actual_tcp_force"])
                    actual_tcp_speed = list(retare_sample["actual_tcp_speed"])
                    if (
                        _vector_norm(tuple(actual_tcp_speed[:3]))
                        > relief_stationary_speed_m_s
                        or _vector_norm(tuple(actual_tcp_speed[3:]))
                        > relief_stationary_angular_speed_rad_s
                    ):
                        raise RuntimeError(
                            f"{part_name} moved while retaring after disengagement"
                        )
                    retare_samples.append(list(actual_tcp_force))
                    time.sleep(control_cycle_sec)
                retared_bias = [
                    sum(sample[index] for sample in retare_samples)
                    / len(retare_samples)
                    for index in range(6)
                ]
                retare_force_span_n = _vector_norm(
                    tuple(
                        max(sample[index] for sample in retare_samples)
                        - min(sample[index] for sample in retare_samples)
                        for index in range(3)
                    )
                )
                retare_torque_span_nm = _vector_norm(
                    tuple(
                        max(sample[index] for sample in retare_samples)
                        - min(sample[index] for sample in retare_samples)
                        for index in range(3, 6)
                    )
                )
                force_bias_delta = tuple(
                    retared_bias[index] - initial_force_bias[index]
                    for index in range(3)
                )
                signed_axial_force_bias_delta_n = _vector_dot(
                    force_bias_delta,
                    insertion_axis_base,
                )
                axial_force_bias_delta_n = abs(signed_axial_force_bias_delta_n)
                lateral_force_bias_delta = tuple(
                    force_bias_delta[index]
                    - signed_axial_force_bias_delta_n * insertion_axis_base[index]
                    for index in range(3)
                )
                lateral_force_bias_delta_n = _vector_norm(
                    lateral_force_bias_delta
                )
                force_bias_delta_n = _vector_norm(force_bias_delta)
                torque_bias_delta_nm = _vector_norm(
                    tuple(
                        retared_bias[index] - initial_force_bias[index]
                        for index in range(3, 6)
                    )
                )
                retare_samples_stable = bool(
                    retare_force_span_n <= baseline_force_uncertainty_n
                    and retare_torque_span_nm <= baseline_torque_uncertainty_nm
                )
                retare_contact_free = bool(
                    axial_force_bias_delta_n
                    <= contact_force_delta_n * relief_clear_hysteresis_ratio
                    and lateral_force_bias_delta_n
                    <= force_depth_lateral_upper_n[0]
                    * relief_clear_hysteresis_ratio
                    and torque_bias_delta_nm
                    <= force_depth_torque_upper_nm[0]
                    * relief_clear_hysteresis_ratio
                )
                retare_baseline_consistent = bool(
                    retare_samples_stable and retare_contact_free
                )
                append_server_trace(
                    "retare",
                    {
                        "disengagement_cycle_count": disengagement_cycle_count,
                        "force_bias_delta_n": force_bias_delta_n,
                        "axial_force_bias_delta_n": (
                            axial_force_bias_delta_n
                        ),
                        "lateral_force_bias_delta_n": (
                            lateral_force_bias_delta_n
                        ),
                        "torque_bias_delta_nm": torque_bias_delta_nm,
                        "retare_force_span_n": retare_force_span_n,
                        "retare_torque_span_nm": retare_torque_span_nm,
                        "retare_samples_stable": retare_samples_stable,
                        "retare_contact_free": retare_contact_free,
                        "retare_baseline_consistent": (
                            retare_baseline_consistent
                        ),
                    },
                )
                if not retare_baseline_consistent:
                    raise _InsertSearchExhausted(
                        f"Automatic retry stopped — {part_name} may have shifted in RG2 "
                        "or remained in contact after full withdrawal"
                    )
                disengagement_contact_cleared = True
                status.update(insert_disengagement_contact_cleared=True)
                self._write_active_goal_status(goal_handle, status)
                force_bias = retared_bias
                force_bias_valid = True
                contact_detected = False
                engagement_detected = False
                contact_candidate_since = None
                contact_candidate_depth_m = 0.0
                contact_reference_depth_m = None
                search_engagement_reference_depth_m = None
                engagement_candidate_since = None
                engagement_candidate_interruption_since = None
                engagement_candidate_peak_depth_m = 0.0
                engagement_depth_m = 0.0
                progress_reference_depth_m = 0.0
                progress_reference_at = time.monotonic()
                filtered_force_samples.clear()
                soft_overload_candidate_since = None
                soft_overload_duration_sec = 0.0
                spiral_phase_offset_rad = (
                    disengagement_cycle_count
                    * 2.0
                    * math.pi
                    / max_disengagement_cycles
                )
                if tactile_center_world_tool0 is not None:
                    spiral_theta = 0.0
                    expanded_search_boundary_theta = None
                    expanded_search_stage_index = 0
                search_origin_world_tool0 = None
                search_guard_entry_theta = spiral_theta
                search_escape_candidate_since = None
                restart_base_tcp = self._read_actual_tcp_transform()
                if restart_base_tcp is None:
                    raise RuntimeError(
                        f"actual TCP pose is unavailable while retrying {part_name}"
                    )
                self._start_insert_force_mode(
                    actual_base_tcp=restart_base_tcp,
                    insertion_axis_base=insertion_axis_base,
                    insertion_force_n=(
                        insertion_force_n * relief_axial_force_ratio
                    ),
                    contact_speed_m_s=contact_speed_m_s,
                    spiral_speed_m_s=(
                        spiral_speed_m_s * relief_search_speed_ratio
                    ),
                    tilt_tolerance_rad=tilt_tolerance_rad,
                )
                phase = "retrying"
                final_phase = phase
                retry_started_at = time.monotonic()

            def restart_force_mode_after_backoff() -> None:
                nonlocal relief_backoff_m
                nonlocal relief_backoff_committed
                nonlocal relief_committed_backoff_m
                nonlocal relief_force_mode_stop_acknowledged
                nonlocal relief_stop_l_command_completed
                nonlocal relief_stationary_confirmed
                nonlocal relief_force_mode_restart_acknowledged
                nonlocal relief_retreat_m
                nonlocal total_relief_backoff_m
                if not stop_and_confirm():
                    raise RuntimeError(
                        "relief backoff did not receive stop and stationary acknowledgements"
                    )
                relief_force_mode_stop_acknowledged = force_mode_stop_acknowledged
                relief_stop_l_command_completed = stop_l_command_completed
                relief_stationary_confirmed = stationary_confirmed
                settled_relief_retreat_m = max(
                    0.0,
                    relief_entry_depth_m - final_insertion_depth_m,
                )
                relief_retreat_m = max(relief_retreat_m, settled_relief_retreat_m)
                relief_backoff_m = max(relief_backoff_m, relief_retreat_m)
                if relief_backoff_committed:
                    total_relief_backoff_m += max(
                        0.0,
                        relief_backoff_m - relief_committed_backoff_m,
                    )
                else:
                    total_relief_backoff_m += relief_backoff_m
                    relief_backoff_committed = True
                relief_committed_backoff_m = relief_backoff_m
                if total_relief_backoff_m > max_relief_retreat_m + 1e-12:
                    raise _InsertSoftOverload(
                        "stationary cumulative relief retreat "
                        f"{total_relief_backoff_m:.6f} m exceeded protected maximum "
                        f"{max_relief_retreat_m:.6f} m"
                    )
                append_server_trace(
                    "relief_backoff_complete",
                    {
                        **last_sample,
                        "relief_cycle_count": relief_cycle_count,
                        "relief_backoff_m": relief_backoff_m,
                        "relief_planned_backoff_m": relief_planned_backoff_m,
                        "total_relief_backoff_m": total_relief_backoff_m,
                        "relief_load_cleared": relief_load_cleared,
                        "relief_stationary_confirmed": relief_stationary_confirmed,
                    },
                )
                dispatch_ready, dispatch_message, _position_error, _orientation_error = (
                    self._cartesian_frame_validation()
                )
                if not dispatch_ready:
                    raise RuntimeError(dispatch_message)
                restart_base_tcp = self._read_actual_tcp_transform()
                if restart_base_tcp is None:
                    raise RuntimeError(
                        "actual TCP pose is unavailable while restarting relief force mode"
                    )
                self._start_insert_force_mode(
                    actual_base_tcp=restart_base_tcp,
                    insertion_axis_base=insertion_axis_base,
                    insertion_force_n=(
                        insertion_force_n * relief_axial_force_ratio
                    ),
                    contact_speed_m_s=contact_speed_m_s,
                    spiral_speed_m_s=(
                        spiral_speed_m_s
                        * relief_search_speed_ratio**relief_cycle_count
                    ),
                    tilt_tolerance_rad=tilt_tolerance_rad,
                )
                relief_force_mode_restart_acknowledged = True
                append_server_trace(
                    "relief_restart",
                    {
                        "relief_cycle_count": relief_cycle_count,
                        "relief_backoff_m": relief_backoff_m,
                        "relief_planned_backoff_m": relief_planned_backoff_m,
                        "total_relief_backoff_m": total_relief_backoff_m,
                        "relief_force_mode_stop_acknowledged": (
                            relief_force_mode_stop_acknowledged
                        ),
                        "relief_stop_l_command_completed": (
                            relief_stop_l_command_completed
                        ),
                        "relief_stationary_confirmed": relief_stationary_confirmed,
                        "relief_force_mode_restart_acknowledged": True,
                    },
                )

            def begin_relief(
                *,
                sampled_at: float,
                insertion_depth_m: float,
                trigger_name: str,
                trigger_value: float,
                trigger_threshold: float,
                reason: str,
            ) -> None:
                nonlocal phase
                nonlocal final_phase
                nonlocal relief_cycle_count
                nonlocal relief_exhausted
                nonlocal relief_started_at
                nonlocal relief_entry_depth_m
                nonlocal relief_clear_since
                nonlocal relief_backoff_complete
                nonlocal relief_backoff_committed
                nonlocal relief_committed_backoff_m
                nonlocal resume_started_at
                nonlocal relief_resume_phase
                nonlocal limit_trigger
                nonlocal limit_trigger_value
                nonlocal limit_trigger_threshold
                nonlocal limit_trigger_actual_tcp_force
                nonlocal limit_trigger_tared_tcp_force
                nonlocal relief_load_cleared
                nonlocal relief_backoff_m
                nonlocal relief_planned_backoff_m
                nonlocal relief_force_mode_stop_acknowledged
                nonlocal relief_stop_l_command_completed
                nonlocal relief_stationary_confirmed
                nonlocal relief_force_mode_restart_acknowledged
                nonlocal frozen_search_resume_pending
                nonlocal search_peck_state
                nonlocal search_peck_started_at
                nonlocal search_peck_retreat_m
                nonlocal search_peck_next_at
                nonlocal engagement_candidate_since
                nonlocal engagement_candidate_interruption_since
                nonlocal engagement_candidate_peak_depth_m
                engagement_candidate_since = None
                engagement_candidate_interruption_since = None
                engagement_candidate_peak_depth_m = 0.0
                limit_trigger = f"soft_{trigger_name}"
                limit_trigger_value = trigger_value
                limit_trigger_threshold = trigger_threshold
                limit_trigger_actual_tcp_force = list(
                    last_sample.get("actual_tcp_force", [0.0] * 6)
                )
                limit_trigger_tared_tcp_force = list(
                    last_sample.get("tared_tcp_force", [0.0] * 6)
                )
                if engagement_detected or phase == "settling":
                    seated_axial_stop_candidate = bool(
                        depth_error_m <= seated_depth_tolerance_m
                        and force_window_ready
                        and filtered_axial_force_n >= contact_force_delta_n
                        and filtered_axial_force_n
                        <= guarded_axial_force_ceiling_n
                        and filtered_lateral_force_n <= max_lateral_force_n
                        and filtered_torque_nm <= max_torque_nm
                        and tilt_error_rad <= tilt_tolerance_rad
                        and abs(axial_tcp_speed_m_s)
                        <= near_zero_axial_speed_m_s
                        and deepest_depth_m - insertion_depth_m
                        <= rebound_tolerance_m
                    )
                    if seated_axial_stop_candidate:
                        phase = "settling"
                        final_phase = phase
                        return
                    relief_exhausted = True
                    begin_disengagement(
                        f"{reason}; {part_name} cocking persisted after engagement",
                        insertion_depth_m=insertion_depth_m,
                    )
                    return
                if not contact_detected or phase not in {
                    "seating",
                    "searching",
                    "expanded_searching",
                    "resuming",
                }:
                    relief_exhausted = True
                    raise _InsertSoftOverload(
                        f"{reason}; relief requires pre-engagement contact"
                    )
                if relief_cycle_count >= max_relief_cycles:
                    relief_exhausted = True
                    begin_disengagement(
                        f"{reason}; micro-relief cycle limit exhausted",
                        insertion_depth_m=insertion_depth_m,
                    )
                    return
                relief_resume_phase = (
                    phase
                    if phase in {"seating", "searching", "expanded_searching"}
                    else relief_resume_phase
                )
                frozen_search_resume_pending = relief_resume_phase in {
                    "searching",
                    "expanded_searching",
                }
                relief_cycle_count += 1
                relief_started_at = sampled_at
                relief_entry_depth_m = insertion_depth_m
                relief_clear_since = None
                relief_backoff_complete = False
                relief_backoff_committed = False
                relief_committed_backoff_m = 0.0
                resume_started_at = None
                relief_load_cleared = False
                relief_backoff_m = 0.0
                relief_planned_backoff_m = 0.0
                relief_force_mode_stop_acknowledged = False
                relief_stop_l_command_completed = False
                relief_stationary_confirmed = False
                relief_force_mode_restart_acknowledged = False
                search_peck_state = "idle"
                search_peck_started_at = None
                search_peck_retreat_m = 0.0
                search_peck_next_at = math.inf
                phase = "relieving"
                final_phase = phase

            while rclpy.ok() and time.monotonic() < execution_deadline:
                if goal_handle.is_cancel_requested:
                    raise _InsertCanceled("canceled")
                now = time.monotonic()
                lateral_force_x_n = 0.0
                lateral_force_y_n = 0.0
                axial_force_command_n = insertion_force_n
                if phase in {"searching", "expanded_searching"}:
                    if spiral_started_at is None:
                        spiral_started_at = now
                    search_elapsed_sec = max(0.0, now - spiral_started_at)
                    ramp_time_sec = max(
                        control_cycle_sec,
                        spiral_speed_m_s / spiral_acceleration_m_s2,
                    )
                    ramp_fraction = min(1.0, search_elapsed_sec / ramp_time_sec)
                    protected_search_speed_m_s = (
                        float(selected_hard_caps["insert_max_spiral_speed_m_s"])
                        if phase == "expanded_searching"
                        else spiral_speed_m_s
                    )
                    search_speed_m_s = (
                        protected_search_speed_m_s
                        * relief_search_speed_ratio
                        ** min(
                            1,
                            relief_cycle_count + disengagement_cycle_count,
                        )
                        * ramp_fraction
                    )
                    path_scale = math.sqrt(
                        spiral_scale * spiral_scale
                        + final_search_radius_m * final_search_radius_m
                    )
                    if search_peck_state != "descending":
                        spiral_theta += (
                            search_speed_m_s
                            * control_cycle_sec
                            / max(path_scale, 1e-12)
                        )
                    stopping_margin_m = max(
                        search_speed_m_s * control_cycle_sec * 2.0,
                        1e-6,
                    )
                    expanded_stage_radius_m = expanded_search_stage_radii_m[
                        expanded_search_stage_index
                    ]
                    expanded_radius_limit_m = max(
                        spiral_radius_m,
                        expanded_stage_radius_m - stopping_margin_m,
                    )
                    scheduled_radius_unbounded_m = spiral_scale * spiral_theta
                    if (
                        phase == "searching"
                        and advanced_recovery_enabled
                        and scheduled_radius_unbounded_m >= spiral_radius_m
                    ):
                        phase = "expanded_searching"
                        final_phase = phase
                    search_radius_limit_m = (
                        expanded_radius_limit_m
                        if phase == "expanded_searching"
                        else spiral_radius_m
                    )
                    final_search_radius_m = min(
                        search_radius_limit_m,
                        scheduled_radius_unbounded_m,
                    )
                    scheduled_search_radius_m = final_search_radius_m
                    explored_search_radius_m = max(
                        explored_search_radius_m,
                        final_search_radius_m,
                    )
                    explored_search_angle_rad = max(
                        explored_search_angle_rad,
                        spiral_theta,
                    )
                    if (
                        phase == "expanded_searching"
                        and search_peck_state == "idle"
                        and final_search_radius_m
                        >= expanded_radius_limit_m - 1e-12
                    ):
                        if expanded_search_boundary_theta is None:
                            expanded_search_boundary_theta = spiral_theta
                        elif (
                            spiral_theta - expanded_search_boundary_theta
                            >= 2.0 * math.pi
                        ):
                            if expanded_search_stage_index + 1 < len(
                                expanded_search_stage_radii_m
                            ):
                                expanded_search_stage_index += 1
                                expanded_search_boundary_theta = None
                            else:
                                raise _InsertSearchExhausted(
                                    "No pin entry found within "
                                    f"{max_contact_search_radius_m * 1000.0:.0f} mm"
                                )
                    radial_ratio = min(
                        1.0,
                        spiral_scale / max(final_search_radius_m, spiral_scale),
                    )
                    force_scale = (
                        lateral_search_force_n
                        * relief_search_force_ratio
                        ** min(
                            1,
                            relief_cycle_count + disengagement_cycle_count,
                        )
                        * ramp_fraction
                        / math.sqrt(1.0 + radial_ratio * radial_ratio)
                    )
                    tangential_force_n = force_scale
                    radial_force_n = force_scale * radial_ratio
                    search_angle_rad = spiral_theta + spiral_phase_offset_rad
                    lateral_force_x_n = (
                        radial_force_n * math.cos(search_angle_rad)
                        - tangential_force_n * math.sin(search_angle_rad)
                    )
                    lateral_force_y_n = (
                        radial_force_n * math.sin(search_angle_rad)
                        + tangential_force_n * math.cos(search_angle_rad)
                    )
                    if search_peck_state == "unloading":
                        axial_force_command_n = (
                            -insertion_force_n * relief_reverse_force_ratio
                        )
                    elif search_peck_state == "descending":
                        lateral_force_x_n = 0.0
                        lateral_force_y_n = 0.0
                        axial_force_command_n = (
                            insertion_force_n * relief_axial_force_ratio
                        )
                    elif phase == "expanded_searching":
                        axial_force_command_n = (
                            insertion_force_n * relief_axial_force_ratio
                        )
                elif phase == "cocked":
                    axial_force_command_n = 0.0
                elif phase == "disengaging":
                    axial_force_command_n = (
                        -insertion_force_n * relief_reverse_force_ratio
                    )
                elif phase == "retrying":
                    if retry_started_at is None:
                        retry_started_at = now
                    retry_fraction = min(
                        1.0,
                        max(0.0, now - retry_started_at)
                        / relief_resume_ramp_sec,
                    )
                    reduced_force_n = (
                        insertion_force_n * relief_axial_force_ratio
                    )
                    axial_force_command_n = reduced_force_n + (
                        insertion_force_n - reduced_force_n
                    ) * retry_fraction
                elif phase == "settling":
                    axial_force_command_n = (
                        insertion_force_n * relief_axial_force_ratio
                    )
                elif phase == "relieving":
                    axial_force_command_n = insertion_force_n * relief_axial_force_ratio
                elif phase == "backing_off":
                    protected_total_retreat_m = _protected_relief_retreat_m(
                        total_relief_backoff_m,
                        relief_backoff_m,
                        relief_backoff_committed=relief_backoff_committed,
                    )
                    if (
                        not relief_backoff_complete
                        and protected_total_retreat_m
                        >= max_relief_retreat_m - 1e-12
                    ):
                        relief_exhausted = True
                        begin_disengagement(
                            "protected cumulative relief retreat reached "
                            f"{max_relief_retreat_m:.6f} m before another reverse command",
                            insertion_depth_m=final_insertion_depth_m,
                        )
                        continue
                    if (
                        not relief_backoff_complete
                        and final_insertion_depth_m
                        <= start_position_tolerance_m
                    ):
                        relief_exhausted = True
                        begin_disengagement(
                            "relief cannot command another reverse sample within "
                            "the protected pre-insertion margin "
                            f"{start_position_tolerance_m:.6f} m",
                            insertion_depth_m=final_insertion_depth_m,
                        )
                        continue
                    remaining_planned_backoff_m = max(
                        0.0,
                        relief_planned_backoff_m - relief_backoff_m,
                    )
                    remaining_total_retreat_m = max(
                        0.0,
                        max_relief_retreat_m - protected_total_retreat_m,
                    )
                    reverse_speed_m_s = max(
                        relief_stationary_speed_m_s,
                        max(
                            0.0,
                            -float(last_sample.get("axial_tcp_speed_m_s", 0.0)),
                        ),
                    )
                    reverse_stop_margin_m = (
                        reverse_speed_m_s * control_cycle_sec * 2.0
                    )
                    if remaining_total_retreat_m <= reverse_stop_margin_m:
                        relief_exhausted = True
                        begin_disengagement(
                            "remaining cumulative relief retreat "
                            f"{remaining_total_retreat_m:.6f} m cannot accommodate "
                            "another reverse control sample",
                            insertion_depth_m=final_insertion_depth_m,
                        )
                        continue
                    if (
                        relief_backoff_m > 0.0
                        and remaining_planned_backoff_m <= reverse_stop_margin_m
                    ):
                        relief_backoff_complete = True
                        restart_force_mode_after_backoff()
                        phase = "resuming"
                        final_phase = phase
                        resume_started_at = now
                        soft_overload_candidate_since = None
                        soft_overload_duration_sec = 0.0
                        filtered_force_samples.clear()
                        continue
                    axial_force_command_n = (
                        -insertion_force_n * relief_reverse_force_ratio
                    )
                elif phase == "resuming":
                    if resume_started_at is None:
                        resume_started_at = now
                    resume_fraction = min(
                        1.0,
                        max(0.0, now - resume_started_at) / relief_resume_ramp_sec,
                    )
                    relief_axial_force_n = insertion_force_n * relief_axial_force_ratio
                    axial_force_command_n = relief_axial_force_n + (
                        insertion_force_n - relief_axial_force_n
                    ) * resume_fraction
                commanded_axial_force_n = axial_force_command_n
                lateral_compliant = bool(
                    phase in {"searching", "expanded_searching"}
                    and search_peck_state == "idle"
                )
                if not lateral_compliant:
                    lateral_force_x_n = 0.0
                    lateral_force_y_n = 0.0
                commanded_lateral_force_x_n = lateral_force_x_n
                commanded_lateral_force_y_n = lateral_force_y_n
                self._refresh_insert_force_mode(
                    lateral_force_x_n=lateral_force_x_n,
                    lateral_force_y_n=lateral_force_y_n,
                    axial_force_n=axial_force_command_n,
                    lateral_compliant=lateral_compliant,
                )
                current_sample = sample(phase)
                insertion_depth_m = float(current_sample["insertion_depth_m"])
                depth_error_m = float(current_sample["depth_error_m"])
                tilt_error_rad = float(current_sample["tilt_error_rad"])
                axial_force_n = float(current_sample["axial_force_n"])
                lateral_force_n = float(current_sample["lateral_force_n"])
                torque_nm = float(current_sample["torque_nm"])
                tool_flange_torque_nm = float(
                    current_sample["tool_flange_torque_nm"]
                )
                axial_tcp_speed_m_s = float(current_sample["axial_tcp_speed_m_s"])
                sampled_at = time.monotonic()
                search_lateral_offset_m = 0.0
                if search_origin_world_tool0 is not None:
                    search_displacement = tuple(
                        current_sample["actual_world_tool0"][0][index]
                        - search_origin_world_tool0[0][index]
                        for index in range(3)
                    )
                    search_axial_displacement_m = _vector_dot(
                        search_displacement,
                        insertion_axis_world,
                    )
                    search_lateral_offset_m = _vector_norm(
                        tuple(
                            search_displacement[index]
                            - search_axial_displacement_m
                            * insertion_axis_world[index]
                            for index in range(3)
                        )
                    )
                deepest_depth_m = max(deepest_depth_m, insertion_depth_m)
                filtered_force_samples.append(
                    (
                        sampled_at,
                        axial_force_n,
                        lateral_force_n,
                        torque_nm,
                        tool_flange_torque_nm,
                    )
                )
                while (
                    filtered_force_samples
                    and sampled_at - filtered_force_samples[0][0]
                    > force_filter_window_sec
                ):
                    filtered_force_samples.popleft()
                force_window_span_sec = (
                    filtered_force_samples[-1][0] - filtered_force_samples[0][0]
                    if len(filtered_force_samples) >= 2
                    else 0.0
                )
                force_window_ready = bool(
                    len(filtered_force_samples) >= 3
                    and force_window_span_sec
                    >= min(0.04, force_filter_window_sec * 0.50)
                )

                def median_force(index: int) -> float:
                    values = sorted(sample_value[index] for sample_value in filtered_force_samples)
                    midpoint = len(values) // 2
                    if len(values) % 2:
                        return values[midpoint]
                    return (values[midpoint - 1] + values[midpoint]) * 0.50

                filtered_axial_force_n = median_force(1)
                filtered_lateral_force_n = median_force(2)
                filtered_torque_nm = median_force(3)
                filtered_tool_flange_torque_nm = median_force(4)
                peak_filtered_axial_force_n = max(
                    peak_filtered_axial_force_n,
                    filtered_axial_force_n,
                )
                peak_filtered_lateral_force_n = max(
                    peak_filtered_lateral_force_n,
                    filtered_lateral_force_n,
                )
                peak_filtered_torque_nm = max(
                    peak_filtered_torque_nm,
                    filtered_torque_nm,
                )
                status.update(
                    insert_filtered_axial_force_n=filtered_axial_force_n,
                    insert_filtered_lateral_force_n=filtered_lateral_force_n,
                    insert_filtered_torque_nm=filtered_torque_nm,
                    insert_filtered_tool_flange_torque_nm=(
                        filtered_tool_flange_torque_nm
                    ),
                )
                profile_start_depth_m = (
                    contact_reference_depth_m
                    if contact_reference_depth_m is not None
                    else insertion_depth_m
                )
                current_force_depth_fraction = min(
                    1.0,
                    max(
                        0.0,
                        (insertion_depth_m - profile_start_depth_m)
                        / max(target_depth_m - profile_start_depth_m, 1e-12),
                    ),
                )
                current_force_depth_axial_upper_n = _force_depth_upper(
                    force_depth_fraction,
                    force_depth_axial_upper_n,
                    current_force_depth_fraction,
                )
                current_force_depth_lateral_upper_n = _force_depth_upper(
                    force_depth_fraction,
                    force_depth_lateral_upper_n,
                    current_force_depth_fraction,
                )
                current_force_depth_torque_upper_nm = _force_depth_upper(
                    force_depth_fraction,
                    force_depth_torque_upper_nm,
                    current_force_depth_fraction,
                )
                (
                    learned_axial_limit_scale,
                    depth_axial_limit_scale,
                ) = _insert_axial_soft_limit_scales(part_name)
                effective_max_axial_force_n = min(
                    max_axial_force_n * learned_axial_limit_scale,
                    guarded_axial_force_ceiling_n,
                )
                effective_force_depth_axial_upper_n = min(
                    current_force_depth_axial_upper_n * depth_axial_limit_scale,
                    guarded_axial_force_ceiling_n,
                )
                axial_progress_advancing = bool(
                    insertion_depth_m
                    >= progress_reference_depth_m + progress_epsilon_m
                )
                if axial_progress_advancing:
                    progress_reference_depth_m = insertion_depth_m
                    progress_reference_at = sampled_at
                axial_progress_stalled = bool(
                    contact_detected
                    and depth_error_m > seated_depth_tolerance_m
                    and sampled_at - progress_reference_at >= stall_hold_sec
                )
                axial_profile_exceeded = bool(
                    force_window_ready
                    and contact_detected
                    and filtered_axial_force_n
                    > effective_force_depth_axial_upper_n
                )
                lateral_profile_exceeded = bool(
                    force_window_ready
                    and contact_detected
                    and filtered_lateral_force_n
                    > current_force_depth_lateral_upper_n
                )
                torque_profile_exceeded = bool(
                    force_window_ready
                    and contact_detected
                    and filtered_torque_nm
                    > current_force_depth_torque_upper_nm
                )
                learned_axial_force_exceeded = bool(
                    force_window_ready
                    and contact_detected
                    and filtered_axial_force_n > effective_max_axial_force_n
                )
                learned_lateral_force_exceeded = bool(
                    force_window_ready
                    and contact_detected
                    and filtered_lateral_force_n > max_lateral_force_n
                )
                learned_torque_exceeded = bool(
                    force_window_ready
                    and contact_detected
                    and filtered_torque_nm > max_torque_nm
                )
                guarded_axial_force_exceeded = bool(
                    force_window_ready
                    and contact_detected
                    and filtered_axial_force_n
                    > guarded_axial_force_ceiling_n
                )
                guarded_lateral_force_exceeded = bool(
                    force_window_ready
                    and contact_detected
                    and filtered_lateral_force_n
                    > guarded_lateral_force_ceiling_n
                )
                guarded_torque_exceeded = bool(
                    force_window_ready
                    and contact_detected
                    and filtered_torque_nm > guarded_torque_ceiling_nm
                )
                force_depth_profile_in_band = bool(
                    not axial_profile_exceeded
                    and not lateral_profile_exceeded
                    and not torque_profile_exceeded
                )
                learned_soft_limits_in_band = bool(
                    not learned_axial_force_exceeded
                    and not learned_lateral_force_exceeded
                    and not learned_torque_exceeded
                )
                guarded_soft_limits_in_band = bool(
                    not guarded_axial_force_exceeded
                    and not guarded_lateral_force_exceeded
                    and not guarded_torque_exceeded
                )
                engagement_profile_load = bool(
                    force_window_ready
                    and guarded_soft_limits_in_band
                    and (
                        (
                            learned_soft_limits_in_band
                            and force_depth_profile_in_band
                        )
                        or axial_progress_advancing
                    )
                )
                if (
                    force_window_ready
                    and filtered_axial_force_n >= contact_force_delta_n
                ):
                    if contact_candidate_since is None:
                        contact_candidate_since = sampled_at
                        contact_candidate_depth_m = insertion_depth_m
                    elif sampled_at - contact_candidate_since >= contact_hold_sec:
                        contact_detected = True
                        if contact_reference_depth_m is None:
                            contact_reference_depth_m = contact_candidate_depth_m
                elif not contact_detected:
                    contact_candidate_since = None
                    contact_candidate_depth_m = 0.0

                update_tactile_center_candidate(
                    sampled_at=sampled_at,
                    actual_world_tool0=current_sample["actual_world_tool0"],
                    insertion_depth_m=insertion_depth_m,
                    filtered_lateral_force_n=filtered_lateral_force_n,
                    filtered_torque_nm=filtered_torque_nm,
                    profile_load_ok=(
                        force_window_ready
                        and learned_soft_limits_in_band
                        and force_depth_profile_in_band
                    ),
                )
                if (
                    engagement_detected
                    and depth_error_m <= seated_depth_tolerance_m
                    and filtered_axial_force_n <= effective_max_axial_force_n
                    and filtered_lateral_force_n <= max_lateral_force_n
                    and filtered_torque_nm <= max_torque_nm
                    and phase
                    not in {
                        "cocked",
                        "disengaging",
                        "recentering",
                        "retaring",
                        "retrying",
                    }
                ):
                    phase = "settling"
                    final_phase = phase
                    search_peck_state = "idle"
                    search_peck_started_at = None
                    search_peck_next_at = math.inf
                search_escaped_scheduled_spiral = bool(
                    advanced_recovery_enabled
                    and phase in {"searching", "expanded_searching"}
                    and contact_detected
                    and axial_progress_stalled
                    and search_origin_world_tool0 is not None
                    and search_lateral_offset_m
                    > scheduled_search_radius_m + start_position_tolerance_m
                )
                if search_escaped_scheduled_spiral:
                    if search_escape_candidate_since is None:
                        search_escape_candidate_since = sampled_at
                    search_completed_full_turn = bool(
                        spiral_theta - search_guard_entry_theta >= 2.0 * math.pi
                    )
                    if (
                        search_completed_full_turn
                        and sampled_at - search_escape_candidate_since
                        >= stall_hold_sec
                    ):
                        begin_disengagement(
                            f"{part_name} search displacement escaped the scheduled tactile spiral "
                            "while insertion progress was stalled",
                            insertion_depth_m=insertion_depth_m,
                        )
                        time.sleep(control_cycle_sec)
                        continue
                else:
                    search_escape_candidate_since = None

                if (
                    advanced_recovery_enabled
                    and phase in {"searching", "expanded_searching"}
                    and not engagement_detected
                ):
                    if search_peck_state == "idle":
                        if (
                            contact_detected
                            and axial_progress_stalled
                            and sampled_at >= search_peck_next_at
                        ):
                            available_retreat_m = max(0.0, insertion_depth_m)
                            search_peck_target_retreat_m = min(
                                search_peck_retreat_limit_m,
                                available_retreat_m,
                            )
                            if search_peck_target_retreat_m > progress_epsilon_m:
                                search_peck_state = "unloading"
                                search_peck_started_at = sampled_at
                                search_peck_entry_depth_m = insertion_depth_m
                                search_peck_retreat_m = 0.0
                                search_peck_cycle_count += 1
                                soft_overload_candidate_since = None
                                soft_overload_duration_sec = 0.0
                                append_server_trace(
                                    "search_peck_unloading",
                                    {
                                        **last_sample,
                                        "search_peck_cycle_count": (
                                            search_peck_cycle_count
                                        ),
                                        "search_peck_entry_depth_m": (
                                            search_peck_entry_depth_m
                                        ),
                                        "search_peck_target_retreat_m": (
                                            search_peck_target_retreat_m
                                        ),
                                    },
                                )
                            else:
                                search_peck_next_at = (
                                    sampled_at + search_peck_interval_sec
                                )
                    elif search_peck_state == "unloading":
                        search_peck_retreat_m = max(
                            search_peck_retreat_m,
                            search_peck_entry_depth_m - insertion_depth_m,
                        )
                        reverse_speed_m_s = max(0.0, -axial_tcp_speed_m_s)
                        reverse_stop_margin_m = max(
                            progress_epsilon_m,
                            reverse_speed_m_s * control_cycle_sec * 2.0,
                        )
                        remaining_retreat_m = max(
                            0.0,
                            search_peck_target_retreat_m
                            - search_peck_retreat_m,
                        )
                        search_peck_elapsed_sec = (
                            sampled_at - search_peck_started_at
                            if search_peck_started_at is not None
                            else 0.0
                        )
                        if (
                            remaining_retreat_m <= reverse_stop_margin_m
                            or search_peck_elapsed_sec >= relief_timeout_sec
                        ):
                            search_peck_state = "descending"
                            search_peck_started_at = sampled_at
                            filtered_force_samples.clear()
                            append_server_trace(
                                "search_peck_descending",
                                {
                                    **last_sample,
                                    "search_peck_cycle_count": (
                                        search_peck_cycle_count
                                    ),
                                    "search_peck_retreat_m": (
                                        search_peck_retreat_m
                                    ),
                                },
                            )
                    elif search_peck_state == "descending":
                        search_peck_elapsed_sec = (
                            sampled_at - search_peck_started_at
                            if search_peck_started_at is not None
                            else 0.0
                        )
                        progressed_below_peck_entry = insertion_depth_m >= (
                            search_peck_entry_depth_m + progress_epsilon_m
                        )
                        recontacted_without_progress = bool(
                            insertion_depth_m
                            >= search_peck_entry_depth_m - progress_epsilon_m
                            and abs(axial_tcp_speed_m_s)
                            <= near_zero_axial_speed_m_s
                            and force_window_ready
                            and filtered_axial_force_n >= contact_force_delta_n
                        )
                        if progressed_below_peck_entry:
                            search_peck_next_at = math.inf
                        elif (
                            recontacted_without_progress
                            or search_peck_elapsed_sec >= relief_timeout_sec
                        ):
                            append_server_trace(
                                "search_peck_complete",
                                {
                                    **last_sample,
                                    "search_peck_cycle_count": (
                                        search_peck_cycle_count
                                    ),
                                    "search_peck_retreat_m": (
                                        search_peck_retreat_m
                                    ),
                                    "recontacted_without_progress": (
                                        recontacted_without_progress
                                    ),
                                },
                            )
                            search_peck_state = "idle"
                            search_peck_started_at = None
                            search_peck_next_at = (
                                sampled_at + search_peck_interval_sec
                            )
                            filtered_force_samples.clear()

                observed_soft_band_evidence: list[
                    tuple[str, float, float, str]
                ] = []
                soft_overload_evidence: list[tuple[str, float, float, str]] = []
                soft_recovery_enabled = _insert_soft_recovery_enabled(part_name)
                if force_window_ready and search_peck_state == "idle":
                    if learned_axial_force_exceeded:
                        observed_soft_band_evidence.append(
                            (
                                "axial_force_n",
                                filtered_axial_force_n,
                                effective_max_axial_force_n,
                                "filtered axial force "
                                f"{filtered_axial_force_n:.3f} N exceeded overall learned "
                                f"limit {effective_max_axial_force_n:.3f} N",
                            )
                        )
                    if learned_lateral_force_exceeded:
                        observed_soft_band_evidence.append(
                            (
                                "lateral_force_n",
                                filtered_lateral_force_n,
                                max_lateral_force_n,
                                "filtered lateral force "
                                f"{filtered_lateral_force_n:.3f} N exceeded overall learned "
                                f"limit {max_lateral_force_n:.3f} N",
                            )
                        )
                    if learned_torque_exceeded:
                        observed_soft_band_evidence.append(
                            (
                                "active_tcp_torque_nm",
                                filtered_torque_nm,
                                max_torque_nm,
                                "filtered active-TCP torque "
                                f"{filtered_torque_nm:.3f} Nm exceeded overall learned "
                                f"limit {max_torque_nm:.3f} Nm",
                            )
                        )
                    if guarded_axial_force_exceeded:
                        soft_overload_evidence.append(
                            (
                                "axial_force_n",
                                filtered_axial_force_n,
                                guarded_axial_force_ceiling_n,
                                "filtered axial force "
                                f"{filtered_axial_force_n:.3f} N exceeded guarded ceiling "
                                f"{guarded_axial_force_ceiling_n:.3f} N below the hard cap",
                            )
                        )
                    elif (
                        soft_recovery_enabled
                        and axial_progress_stalled
                        and learned_axial_force_exceeded
                    ):
                        soft_overload_evidence.append(
                            (
                                "axial_force_n",
                                filtered_axial_force_n,
                                effective_max_axial_force_n,
                                "filtered axial force "
                                f"{filtered_axial_force_n:.3f} N exceeded overall learned "
                                f"limit {effective_max_axial_force_n:.3f} N while insertion "
                                "progress "
                                "was stalled",
                            )
                        )
                    elif (
                        soft_recovery_enabled
                        and axial_progress_stalled
                        and axial_profile_exceeded
                    ):
                        soft_overload_evidence.append(
                            (
                                "axial_force_n",
                                filtered_axial_force_n,
                                effective_force_depth_axial_upper_n,
                                "filtered axial force "
                                f"{filtered_axial_force_n:.3f} N exceeded depth-profile "
                                f"limit {effective_force_depth_axial_upper_n:.3f} N while "
                                "insertion progress was stalled",
                            )
                        )
                    if guarded_lateral_force_exceeded:
                        soft_overload_evidence.append(
                            (
                                "lateral_force_n",
                                filtered_lateral_force_n,
                                guarded_lateral_force_ceiling_n,
                                "filtered lateral force "
                                f"{filtered_lateral_force_n:.3f} N exceeded guarded ceiling "
                                f"{guarded_lateral_force_ceiling_n:.3f} N below the hard cap",
                            )
                        )
                    elif (
                        soft_recovery_enabled
                        and axial_progress_stalled
                        and learned_lateral_force_exceeded
                    ):
                        soft_overload_evidence.append(
                            (
                                "lateral_force_n",
                                filtered_lateral_force_n,
                                max_lateral_force_n,
                                "filtered lateral force "
                                f"{filtered_lateral_force_n:.3f} N exceeded overall learned "
                                f"limit {max_lateral_force_n:.3f} N while insertion progress "
                                "was stalled",
                            )
                        )
                    elif (
                        soft_recovery_enabled
                        and axial_progress_stalled
                        and lateral_profile_exceeded
                    ):
                        soft_overload_evidence.append(
                            (
                                "lateral_force_n",
                                filtered_lateral_force_n,
                                current_force_depth_lateral_upper_n,
                                "filtered lateral force "
                                f"{filtered_lateral_force_n:.3f} N exceeded depth-profile "
                                f"limit {current_force_depth_lateral_upper_n:.3f} N while "
                                "insertion progress was stalled",
                            )
                        )
                    if guarded_torque_exceeded:
                        soft_overload_evidence.append(
                            (
                                "active_tcp_torque_nm",
                                filtered_torque_nm,
                                guarded_torque_ceiling_nm,
                                "filtered active-TCP torque "
                                f"{filtered_torque_nm:.3f} Nm exceeded guarded ceiling "
                                f"{guarded_torque_ceiling_nm:.3f} Nm below the hard cap",
                            )
                        )
                    elif (
                        soft_recovery_enabled
                        and axial_progress_stalled
                        and learned_torque_exceeded
                    ):
                        soft_overload_evidence.append(
                            (
                                "active_tcp_torque_nm",
                                filtered_torque_nm,
                                max_torque_nm,
                                "filtered active-TCP torque "
                                f"{filtered_torque_nm:.3f} Nm exceeded overall learned "
                                f"limit {max_torque_nm:.3f} Nm while insertion progress "
                                "was stalled",
                            )
                        )
                    elif (
                        soft_recovery_enabled
                        and axial_progress_stalled
                        and torque_profile_exceeded
                    ):
                        soft_overload_evidence.append(
                            (
                                "active_tcp_torque_nm",
                                filtered_torque_nm,
                                current_force_depth_torque_upper_nm,
                                "filtered active-TCP torque "
                                f"{filtered_torque_nm:.3f} Nm exceeded depth-profile "
                                f"limit {current_force_depth_torque_upper_nm:.3f} Nm while "
                                "insertion progress was stalled",
                            )
                        )
                if observed_soft_band_evidence:
                    soft_overload_detected = True
                    last_soft_overload_reason = "; ".join(
                        evidence[3] for evidence in observed_soft_band_evidence
                    )
                if soft_overload_evidence:
                    soft_overload_detected = True
                    trigger_name, trigger_value, trigger_threshold, reason = (
                        soft_overload_evidence[0]
                    )
                    last_soft_overload_reason = "; ".join(
                        evidence[3] for evidence in soft_overload_evidence
                    )
                    if soft_overload_candidate_since is None:
                        soft_overload_candidate_since = sampled_at
                    soft_overload_duration_sec = max(
                        0.0,
                        sampled_at - soft_overload_candidate_since,
                    )
                    if (
                        phase
                        not in {
                            "relieving",
                            "backing_off",
                            "cocked",
                            "disengaging",
                            "recentering",
                            "retaring",
                            "retrying",
                        }
                        and soft_overload_duration_sec >= soft_overload_hold_sec
                    ):
                        begin_relief(
                            sampled_at=sampled_at,
                            insertion_depth_m=insertion_depth_m,
                            trigger_name=trigger_name,
                            trigger_value=trigger_value,
                            trigger_threshold=trigger_threshold,
                            reason=reason,
                        )
                else:
                    soft_overload_candidate_since = None
                    soft_overload_duration_sec = 0.0

                if phase in {"cocked", "disengaging", "retrying"}:
                    if disengagement_started_at is None:
                        disengagement_started_at = sampled_at
                    if disengagement_deadline is None:
                        disengagement_timeout_sec = max(
                            relief_timeout_sec,
                            target_depth_m / contact_speed_m_s
                            + relief_timeout_sec,
                        )
                        disengagement_deadline = min(
                            execution_deadline,
                            disengagement_started_at
                            + disengagement_timeout_sec,
                        )
                    if phase in {"cocked", "disengaging"}:
                        require_disengagement_deadline(sampled_at)
                    if phase == "cocked":
                        if (
                            sampled_at - disengagement_started_at
                            >= relief_unload_dwell_sec
                        ):
                            phase = "disengaging"
                            final_phase = phase
                            filtered_force_samples.clear()
                            disengagement_clear_since = None
                            disengagement_progress_reference_depth_m = (
                                insertion_depth_m
                            )
                            disengagement_progress_reference_at = sampled_at
                    elif phase == "disengaging":
                        disengagement_withdrawal_m = max(
                            disengagement_withdrawal_m,
                            disengagement_entry_depth_m - insertion_depth_m,
                        )
                        require_disengagement_progress(
                            sampled_at=sampled_at,
                            insertion_depth_m=insertion_depth_m,
                        )
                        disengagement_load_clear = bool(
                            force_window_ready
                            and filtered_axial_force_n
                            <= contact_force_delta_n
                            * relief_clear_hysteresis_ratio
                            and filtered_lateral_force_n
                            <= current_force_depth_lateral_upper_n
                            * relief_clear_hysteresis_ratio
                            and filtered_torque_nm
                            <= current_force_depth_torque_upper_nm
                            * relief_clear_hysteresis_ratio
                        )
                        if disengagement_load_clear:
                            if disengagement_clear_since is None:
                                disengagement_clear_since = sampled_at
                            elif (
                                sampled_at - disengagement_clear_since
                                >= relief_clear_dwell_sec
                            ):
                                complete_disengagement_recenter()
                        else:
                            disengagement_clear_since = None
                    elif (
                        retry_started_at is not None
                        and sampled_at - retry_started_at
                        >= relief_resume_ramp_sec
                    ):
                        phase = "seating"
                        final_phase = phase
                        retry_started_at = None
                    status.update(
                        insert_phase=phase,
                        insert_disengagement_cycle_count=(
                            disengagement_cycle_count
                        ),
                        insert_last_disengagement_reason=(
                            last_disengagement_reason
                        ),
                        insert_disengagement_withdrawal_m=(
                            disengagement_withdrawal_m
                        ),
                        insert_disengagement_contact_cleared=(
                            disengagement_contact_cleared
                        ),
                        insert_recenter_position_error_m=(
                            recenter_position_error_m
                        ),
                        insert_retare_baseline_consistent=(
                            retare_baseline_consistent
                        ),
                    )
                    self._write_active_goal_status(goal_handle, status)
                    time.sleep(control_cycle_sec)
                    continue

                stationary_for_relief = bool(
                    float(current_sample["linear_speed_m_s"])
                    <= relief_stationary_speed_m_s
                    and float(current_sample["angular_speed_rad_s"])
                    <= relief_stationary_angular_speed_rad_s
                )
                soft_load_clear = bool(
                    force_window_ready
                    and filtered_axial_force_n
                    <= effective_force_depth_axial_upper_n
                    * relief_clear_hysteresis_ratio
                    and filtered_lateral_force_n
                    <= current_force_depth_lateral_upper_n
                    * relief_clear_hysteresis_ratio
                    and filtered_torque_nm
                    <= current_force_depth_torque_upper_nm
                    * relief_clear_hysteresis_ratio
                )
                if phase in {"relieving", "backing_off", "resuming"}:
                    relief_load_cleared = soft_load_clear
                    if relief_started_at is None:
                        raise RuntimeError("insertion relief start time is unavailable")
                    relief_elapsed_sec = max(0.0, sampled_at - relief_started_at)
                    relief_retreat_m = max(
                        0.0,
                        relief_entry_depth_m - insertion_depth_m,
                    )
                    relief_backoff_m = max(relief_backoff_m, relief_retreat_m)
                    if (
                        relief_backoff_committed
                        and relief_backoff_m > relief_committed_backoff_m
                    ):
                        total_relief_backoff_m += (
                            relief_backoff_m - relief_committed_backoff_m
                        )
                        relief_committed_backoff_m = relief_backoff_m
                    protected_total_retreat_m = _protected_relief_retreat_m(
                        total_relief_backoff_m,
                        relief_backoff_m,
                        relief_backoff_committed=relief_backoff_committed,
                    )
                    if protected_total_retreat_m > max_relief_retreat_m + 1e-12:
                        relief_exhausted = True
                        begin_disengagement(
                            "cumulative relief retreat "
                            f"{protected_total_retreat_m:.6f} m exceeded protected maximum "
                            f"{max_relief_retreat_m:.6f} m",
                            insertion_depth_m=insertion_depth_m,
                        )
                        continue
                    if relief_retreat_m > max_relief_retreat_m + 1e-12:
                        relief_exhausted = True
                        begin_disengagement(
                            "relief retreat "
                            f"{relief_retreat_m:.6f} m exceeded protected maximum "
                            f"{max_relief_retreat_m:.6f} m",
                            insertion_depth_m=insertion_depth_m,
                        )
                        continue
                    if relief_elapsed_sec > relief_timeout_sec:
                        relief_exhausted = True
                        begin_disengagement(
                            f"relief cycle {relief_cycle_count} exceeded "
                            f"{relief_timeout_sec:.3f} s",
                            insertion_depth_m=insertion_depth_m,
                        )
                        continue
                    if phase == "relieving":
                        if soft_load_clear and stationary_for_relief:
                            if relief_clear_since is None:
                                relief_clear_since = sampled_at
                            if (
                                relief_elapsed_sec >= relief_unload_dwell_sec
                                and sampled_at - relief_clear_since
                                >= relief_clear_dwell_sec
                            ):
                                phase = "resuming"
                                resume_started_at = sampled_at
                                soft_overload_candidate_since = None
                                soft_overload_duration_sec = 0.0
                                filtered_force_samples.clear()
                        elif relief_elapsed_sec >= relief_unload_dwell_sec:
                            remaining_relief_retreat_m = (
                                max_relief_retreat_m - total_relief_backoff_m
                            )
                            relief_planned_backoff_m = min(
                                relief_backoff_step_m,
                                remaining_relief_retreat_m,
                            )
                            minimum_depth_for_backoff_m = (
                                relief_planned_backoff_m
                                + start_position_tolerance_m
                            )
                            if insertion_depth_m <= minimum_depth_for_backoff_m:
                                relief_exhausted = True
                                begin_disengagement(
                                    "available positive insertion depth "
                                    f"{insertion_depth_m:.6f} m is below protected "
                                    "backoff command "
                                    f"{relief_planned_backoff_m:.6f} m plus "
                                    "pre-insertion margin "
                                    f"{start_position_tolerance_m:.6f} m",
                                    insertion_depth_m=insertion_depth_m,
                                )
                                continue
                            phase = "backing_off"
                            relief_clear_since = None
                    elif phase == "backing_off":
                        if soft_load_clear:
                            if relief_clear_since is None:
                                relief_clear_since = sampled_at
                        else:
                            relief_clear_since = None
                        relief_backoff_complete = bool(
                            relief_retreat_m + 1e-12
                            >= relief_planned_backoff_m
                            or (
                                relief_clear_since is not None
                                and sampled_at - relief_clear_since
                                >= relief_clear_dwell_sec
                            )
                        )
                        if relief_backoff_complete:
                            restart_force_mode_after_backoff()
                            phase = "resuming"
                            resume_started_at = sampled_at
                            soft_overload_candidate_since = None
                            soft_overload_duration_sec = 0.0
                            filtered_force_samples.clear()
                    elif phase == "resuming":
                        if resume_started_at is None:
                            resume_started_at = sampled_at
                        if sampled_at - resume_started_at >= relief_resume_ramp_sec:
                            phase = (
                                relief_resume_phase
                                if frozen_search_resume_pending
                                else "seating"
                            )
                            if phase in {"searching", "expanded_searching"}:
                                search_peck_next_at = (
                                    sampled_at + search_peck_interval_sec
                                    if advanced_recovery_enabled
                                    else math.inf
                                )
                            final_phase = phase
                            frozen_search_resume_pending = False
                            soft_overload_recovered = True
                            soft_overload_candidate_since = None
                            soft_overload_duration_sec = 0.0
                            filtered_force_samples.clear()
                            progress_reference_depth_m = insertion_depth_m
                            progress_reference_at = sampled_at
                            relief_started_at = None
                            relief_clear_since = None
                            relief_backoff_complete = False
                            relief_backoff_committed = False
                            relief_committed_backoff_m = 0.0
                            resume_started_at = None
                    final_phase = phase
                    if phase in {"relieving", "backing_off", "resuming"}:
                        time.sleep(control_cycle_sec)
                        continue

                if engagement_detected and (
                    insertion_depth_m < engagement_depth_m - rebound_tolerance_m
                ):
                    begin_disengagement(
                        "detected engagement rebounded before seating",
                        insertion_depth_m=insertion_depth_m,
                    )
                    time.sleep(control_cycle_sec)
                    continue

                engagement_reference_depth_m = (
                    search_engagement_reference_depth_m
                    if search_engagement_reference_depth_m is not None
                    else (
                        contact_reference_depth_m
                        if contact_reference_depth_m is not None
                        and depth_error_m > seated_depth_tolerance_m
                        else 0.0
                    )
                )
                engagement_progress_from_reference_m = (
                    insertion_depth_m - engagement_reference_depth_m
                )
                if not engagement_detected:
                    engagement_candidate_expected_band_load = bool(
                        engagement_candidate_since is not None
                        and contact_detected
                        and engagement_progress_from_reference_m
                        >= engagement_progress_m
                        and guarded_soft_limits_in_band
                        and not axial_progress_stalled
                        and insertion_depth_m
                        >= engagement_candidate_peak_depth_m
                        - rebound_tolerance_m
                    )
                    if (
                        contact_detected
                        and (
                            engagement_profile_load
                            or engagement_candidate_expected_band_load
                        )
                        and engagement_progress_from_reference_m
                        >= engagement_progress_m
                    ):
                        if (
                            engagement_candidate_since is not None
                            and engagement_candidate_interruption_since is not None
                        ):
                            engagement_candidate_since += max(
                                0.0,
                                sampled_at
                                - engagement_candidate_interruption_since,
                            )
                        engagement_candidate_interruption_since = None
                        if engagement_candidate_since is None:
                            engagement_candidate_since = sampled_at
                            engagement_candidate_peak_depth_m = insertion_depth_m
                        elif (
                            insertion_depth_m
                            < engagement_candidate_peak_depth_m - rebound_tolerance_m
                        ):
                            engagement_candidate_since = None
                            engagement_candidate_interruption_since = None
                            engagement_candidate_peak_depth_m = 0.0
                        else:
                            engagement_candidate_peak_depth_m = max(
                                engagement_candidate_peak_depth_m,
                                insertion_depth_m,
                            )
                            if (
                                sampled_at - engagement_candidate_since
                                >= engagement_hold_sec
                            ):
                                engagement_detected = True
                                engagement_depth_m = insertion_depth_m
                                search_peck_state = "idle"
                                search_peck_started_at = None
                                search_peck_next_at = math.inf
                                phase = "seating"
                    elif (
                        engagement_candidate_since is not None
                        and contact_detected
                        and engagement_progress_from_reference_m
                        >= engagement_progress_m
                        and not engagement_profile_load
                        and insertion_depth_m
                        > engagement_candidate_peak_depth_m
                    ):
                        # Preserve advancing evidence across only the same brief
                        # interval that an overall learned soft-limit exceedance
                        # is allowed before relief becomes mandatory.
                        if engagement_candidate_interruption_since is None:
                            engagement_candidate_interruption_since = sampled_at
                        engagement_candidate_peak_depth_m = insertion_depth_m
                        if (
                            sampled_at - engagement_candidate_interruption_since
                            > soft_overload_hold_sec
                        ):
                            engagement_candidate_since = None
                            engagement_candidate_interruption_since = None
                            engagement_candidate_peak_depth_m = 0.0
                    else:
                        engagement_candidate_since = None
                        engagement_candidate_interruption_since = None
                        engagement_candidate_peak_depth_m = 0.0

                if insertion_depth_m >= (
                    progress_reference_depth_m + progress_epsilon_m
                ):
                    progress_reference_depth_m = insertion_depth_m
                    progress_reference_at = sampled_at
                elif (
                    not engagement_detected
                    and phase not in {"searching", "expanded_searching"}
                    and contact_detected
                    and depth_error_m > seated_depth_tolerance_m
                    and sampled_at - progress_reference_at >= stall_hold_sec
                ):
                    if spiral_radius_m <= 0.0:
                        raise _InsertSearchExhausted(
                            "direct insertion stalled after contact without engagement"
                        )
                    phase = "searching"
                    spiral_started_at = sampled_at
                    search_origin_world_tool0 = current_sample[
                        "actual_world_tool0"
                    ]
                    search_guard_entry_theta = spiral_theta
                    search_escape_candidate_since = None
                    search_peck_state = "idle"
                    search_peck_started_at = None
                    search_peck_retreat_m = 0.0
                    search_peck_next_at = (
                        sampled_at + search_peck_interval_sec
                        if advanced_recovery_enabled
                        else math.inf
                    )
                    if not frozen_search_resume_pending:
                        search_engagement_reference_depth_m = max(
                            0.0,
                            contact_reference_depth_m
                            if contact_reference_depth_m is not None
                            else insertion_depth_m,
                        )
                    frozen_search_resume_pending = False
                    engagement_candidate_since = None
                    engagement_candidate_interruption_since = None
                    engagement_candidate_peak_depth_m = 0.0

                reached_target_depth = depth_error_m <= seated_depth_tolerance_m
                stable_bottom_contact = bool(
                    force_window_ready
                    and filtered_axial_force_n >= contact_force_delta_n
                )
                profile_load_at_depth = bool(
                    tilt_error_rad <= tilt_tolerance_rad
                    and (
                        (
                            learned_soft_limits_in_band
                            and force_depth_profile_in_band
                        )
                        or (
                            guarded_soft_limits_in_band
                            and filtered_axial_force_n
                            >= contact_force_delta_n
                            and filtered_lateral_force_n
                            <= max_lateral_force_n
                            and filtered_torque_nm <= max_torque_nm
                        )
                    )
                )
                stationary_at_depth = (
                    abs(axial_tcp_speed_m_s) <= near_zero_axial_speed_m_s
                )
                no_seated_rebound = bool(
                    deepest_depth_m - insertion_depth_m <= rebound_tolerance_m
                )
                seated_evidence = bool(
                    engagement_detected
                    and reached_target_depth
                    and stable_bottom_contact
                    and profile_load_at_depth
                    and stationary_at_depth
                    and no_seated_rebound
                )
                if (
                    _insert_depth_completion_enabled(part_name)
                    and engagement_detected
                    and reached_target_depth
                    and guarded_soft_limits_in_band
                    and tilt_error_rad <= tilt_tolerance_rad
                ):
                    phase = "settling"
                    seated_detected = True
                    sample("settling")
                    break
                if seated_evidence:
                    phase = "settling"
                    if seated_candidate_since is None:
                        seated_candidate_since = sampled_at
                        seated_force_min_n = filtered_axial_force_n
                        seated_force_max_n = filtered_axial_force_n
                    else:
                        seated_force_min_n = min(
                            seated_force_min_n,
                            filtered_axial_force_n,
                        )
                        seated_force_max_n = max(
                            seated_force_max_n,
                            filtered_axial_force_n,
                        )
                        stable_force = (
                            seated_force_max_n - seated_force_min_n
                            <= bottom_force_variation_n
                        )
                        if (
                            stable_force
                            and sampled_at - seated_candidate_since >= seated_hold_sec
                        ):
                            seated_detected = True
                            sample("settling")
                            break
                else:
                    seated_candidate_since = None
                    seated_force_min_n = math.inf
                    seated_force_max_n = 0.0

                final_phase = phase
                if (
                    phase == "searching"
                    and final_search_radius_m >= spiral_radius_m
                    and not engagement_detected
                ):
                    raise _InsertSearchExhausted(
                        "spiral search reached spiral_radius_m without engagement"
                    )
                time.sleep(control_cycle_sec)
            else:
                if not rclpy.ok():
                    raise RuntimeError("ROS shutdown interrupted UR5e insertion")
                if phase in {"cocked", "disengaging"}:
                    disengagement_timed_out_at = time.monotonic()
                    disengagement_elapsed_sec = max(
                        0.0,
                        disengagement_timed_out_at
                        - float(
                            disengagement_started_at
                            if disengagement_started_at is not None
                            else disengagement_timed_out_at
                        ),
                    )
                    last_disengagement_reason = (
                        f"{part_name} disengagement timed out after "
                        f"{disengagement_elapsed_sec:.3f} s before contact cleared"
                    )
                    raise _InsertSearchExhausted(
                        last_disengagement_reason
                    )
                if not engagement_detected:
                    raise _InsertSearchExhausted("insertion timed out without engagement")
                raise _InsertSearchExhausted(
                    "insertion timed out without stable seating evidence"
                )

            if not engagement_detected or not seated_detected:
                raise _InsertSearchExhausted(
                    "insertion ended without engagement and stable seating evidence"
                )
            if not stop_and_confirm():
                raise RuntimeError("insertion completion did not confirm stationary motion")
            capture_final_pose()
            if (
                final_depth_error_m > seated_depth_tolerance_m
                or final_tilt_error_rad > tilt_tolerance_rad
                or deepest_depth_m - final_insertion_depth_m > rebound_tolerance_m
            ):
                raise RuntimeError("final inserted pose is outside seated tolerance")
            final_phase = "settling"
            finalize_server_trace(
                terminal_state="succeeded",
                error_code=0,
                error_string="",
            )
            status.update(**terminal_evidence())
            status.update(
                state="succeeded",
                message=(
                    "UR5e insertion engagement and stable seating confirmed with "
                    "stationary hold"
                ),
                blocked_reason="",
                insert_phase="settling",
                insert_insertion_depth_m=final_insertion_depth_m,
                insert_depth_error_m=final_depth_error_m,
                insert_lateral_offset_m=final_lateral_offset_m,
                insert_search_radius_m=final_search_radius_m,
                insert_contact_detected=contact_detected,
                insert_engagement_detected=engagement_detected,
                insert_seated_detected=seated_detected,
                insert_motion_settled=True,
            )
            goal_handle.succeed()
            self._finish_active_goal_status(goal_handle, status)
            return result(0, "", False)
        except _InsertCanceled as exc:
            stop_confirmed = not motion_attempted or stop_and_confirm()
            reason = str(exc) or "canceled"
            finalize_server_trace(
                terminal_state="canceled",
                error_code=-3,
                error_string=reason,
            )
            status.update(**terminal_evidence())
            status.update(
                state="canceled",
                message=(
                    "UR5e insertion canceled"
                    if stop_confirmed
                    else "UR5e insertion canceled without stationary confirmation"
                ),
                blocked_reason=(
                    ""
                    if stop_confirmed
                    else "UR5e insertion cancellation did not confirm stationary motion"
                ),
                rtde_reset_required=not stop_confirmed,
                insert_motion_settled=stop_confirmed,
            )
            if not stop_confirmed:
                self._mark_rtde_reset_required(
                    status["blocked_reason"],
                    write_status=False,
                    failure_kind="insert_cancel_stop_unconfirmed",
                )
            goal_handle.canceled()
            self._finish_active_goal_status(
                goal_handle,
                status,
                latch_status=not stop_confirmed,
            )
            return result(
                -3,
                reason,
                not stop_confirmed,
                motion_settled=stop_confirmed,
            )
        except _InsertSearchExhausted as exc:
            stop_confirmed = not motion_attempted or stop_and_confirm()
            reason = f"UR5e insertion search exhausted: {exc}"
            finalize_server_trace(
                terminal_state="failed",
                error_code=-5,
                error_string=reason,
            )
            status.update(**terminal_evidence())
            status.update(
                state="failed",
                message=reason,
                blocked_reason=reason,
                rtde_reset_required=not stop_confirmed,
                insert_motion_settled=stop_confirmed,
            )
            if not stop_confirmed:
                self._mark_rtde_reset_required(
                    "UR5e insertion search stop did not confirm stationary motion",
                    write_status=False,
                    failure_kind="insert_search_stop_unconfirmed",
                )
            goal_handle.abort()
            self._finish_active_goal_status(
                goal_handle,
                status,
                latch_status=not stop_confirmed,
            )
            return result(
                -5,
                reason,
                not stop_confirmed,
                motion_settled=stop_confirmed,
            )
        except _InsertSoftOverload as exc:
            stop_confirmed = not motion_attempted or stop_and_confirm()
            reason = f"UR5e insertion soft overload relief failed: {exc}"
            relief_exhausted = True
            finalize_server_trace(
                terminal_state="failed",
                error_code=-7,
                error_string=reason,
            )
            status.update(**terminal_evidence())
            publish_last_sample(final_phase)
            status.update(
                state="failed",
                message=reason,
                blocked_reason=reason,
                rtde_reset_required=not stop_confirmed,
                insert_motion_settled=stop_confirmed,
            )
            if not stop_confirmed:
                self._mark_rtde_reset_required(
                    "UR5e insertion overload relief stop did not confirm stationary motion",
                    write_status=False,
                    failure_kind="insert_relief_stop_unconfirmed",
                )
            goal_handle.abort()
            self._finish_active_goal_status(
                goal_handle,
                status,
                latch_status=not stop_confirmed,
            )
            return result(
                -7,
                reason,
                not stop_confirmed,
                motion_settled=stop_confirmed,
            )
        except _InsertForceLimit as exc:
            stop_confirmed = not motion_attempted or stop_and_confirm()
            reason = f"UR5e insertion force/torque/travel limit: {exc}"
            finalize_server_trace(
                terminal_state="failed",
                error_code=-6,
                error_string=reason,
            )
            status.update(**terminal_evidence())
            publish_last_sample(final_phase)
            status.update(
                state="failed",
                message=reason,
                blocked_reason=reason,
                rtde_reset_required=not stop_confirmed,
                insert_motion_settled=stop_confirmed,
            )
            if not stop_confirmed:
                self._mark_rtde_reset_required(
                    "UR5e insertion safety stop did not confirm stationary motion",
                    write_status=False,
                    failure_kind="insert_safety_stop_unconfirmed",
                )
            goal_handle.abort()
            self._finish_active_goal_status(
                goal_handle,
                status,
                latch_status=not stop_confirmed,
            )
            return result(
                -6,
                reason,
                not stop_confirmed,
                motion_settled=stop_confirmed,
            )
        except ValueError as exc:
            stop_confirmed = not motion_attempted or stop_and_confirm()
            reason = f"UR5e RTDE insertion target rejected: {exc}"
            finalize_server_trace(
                terminal_state="blocked",
                error_code=-2,
                error_string=reason,
            )
            status.update(**terminal_evidence())
            status.update(
                state="blocked",
                message=reason,
                blocked_reason=reason,
                rtde_reset_required=motion_attempted and not stop_confirmed,
                insert_motion_settled=stop_confirmed,
            )
            goal_handle.abort()
            self._finish_active_goal_status(
                goal_handle,
                status,
                latch_status=motion_attempted and not stop_confirmed,
            )
            return result(
                -2,
                reason,
                motion_attempted and not stop_confirmed,
                motion_settled=stop_confirmed,
            )
        except Exception as exc:  # noqa: BLE001 - unknown RTDE acceptance is safety-critical.
            stop_confirmed = not motion_attempted or stop_and_confirm()
            state_uncertain = bool(motion_attempted)
            reason = f"UR5e RTDE insertion failed: {type(exc).__name__}: {exc}"
            generic_error_code = -4 if motion_attempted else -1
            finalize_server_trace(
                terminal_state="failed",
                error_code=generic_error_code,
                error_string=reason,
            )
            status.update(**terminal_evidence())
            if motion_attempted:
                self._mark_rtde_reset_required(
                    reason,
                    write_status=False,
                    failure_kind=(
                        "insert_execution_unknown" if stop_confirmed else "insert_stop_unconfirmed"
                    ),
                )
            reset_required = bool(
                motion_attempted or getattr(self, "_rtde_reset_required", False)
            )
            status.update(
                state="failed",
                message=reason,
                blocked_reason=reason,
                rtde_reset_required=reset_required,
                insert_motion_settled=stop_confirmed,
            )
            goal_handle.abort()
            self._finish_active_goal_status(
                goal_handle,
                status,
                latch_status=reset_required,
            )
            return result(
                generic_error_code,
                reason,
                state_uncertain,
                motion_settled=stop_confirmed,
            )
        finally:
            if bool(getattr(self, "_insert_force_mode_active", False)) or bool(
                getattr(self, "_insert_servo_active", False)
            ):
                self._stop_insert_motion()
                self._confirm_stationary_after_stop()
            self._clear_active_goal(goal_handle)

    @staticmethod
    def _cartesian_result(
        error_code: int,
        error_string: str,
        *,
        position_error_m: float = math.inf,
        orientation_error_rad: float = math.inf,
    ) -> Any:
        if MoveUR5eCartesian is None:
            return None
        result = MoveUR5eCartesian.Result()
        result.error_code = int(error_code)
        result.error_string = str(error_string or "")
        result.final_position_error_m = float(position_error_m)
        result.final_orientation_error_rad = float(orientation_error_rad)
        return result

    @staticmethod
    def _relative_cartesian_result(
        error_code: int,
        error_string: str,
        *,
        translation_error_m: float = math.inf,
        orientation_drift_rad: float = math.inf,
    ) -> Any:
        if MoveUR5eRelativeCartesian is None:
            return None
        result = MoveUR5eRelativeCartesian.Result()
        result.error_code = int(error_code)
        result.error_string = str(error_string or "")
        result.final_translation_error_m = float(translation_error_m)
        result.final_orientation_drift_rad = float(orientation_drift_rad)
        return result

    def _lookup_rigid_transform(self, target_frame: str, source_frame: str) -> RigidTransform:
        message = self._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
            timeout=Duration(seconds=0.5),
        )
        return _transform_from_message(message)

    def _active_tcp_offset(self) -> RigidTransform:
        control = self.control
        get_tcp_offset = getattr(control, "getTCPOffset", None)
        if not callable(get_tcp_offset):
            raise RuntimeError("RTDE control object has no getTCPOffset method")
        return _transform_from_rtde_pose([float(value) for value in get_tcp_offset()])

    def _resolve_cartesian_target(
        self,
        target_message: PoseStamped,
    ) -> tuple[RigidTransform, RigidTransform, RigidTransform, RigidTransform]:
        frame_id = str(target_message.header.frame_id or "").strip()
        if frame_id != "world":
            raise ValueError(
                f"MoveUR5eCartesian requires frame_id=world, received {frame_id or '(empty)'}"
            )
        target_world_tool0 = _transform_from_pose_stamped(target_message)
        workspace_error = _workspace_error(target_world_tool0)
        if workspace_error:
            raise ValueError(workspace_error)
        world_base, _frame_message, _position_error, _orientation_error = (
            self._validated_cartesian_world_base()
        )
        tool0_tcp = self._active_tcp_offset()
        base_tool0 = _compose_transform(_inverse_transform(world_base), target_world_tool0)
        target_base_tcp = _compose_transform(base_tool0, tool0_tcp)
        return target_world_tool0, target_base_tcp, world_base, tool0_tcp

    def _read_actual_tcp_pose(self) -> list[float] | None:
        """Return the controller base -> active TCP pose without rewriting its rotvec."""
        if bool(getattr(self, "_rtde_reset_required", False)):
            return None
        with self._receive_lock:
            get_actual_tcp_pose = getattr(self.receive, "getActualTCPPose", None)
            if not callable(get_actual_tcp_pose):
                return None
            try:
                values = [float(value) for value in list(get_actual_tcp_pose())]
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                reason = (
                    "UR5e RTDE Cartesian feedback transport failed: "
                    f"{type(exc).__name__}: {exc}. Use Repair Hardware Stack."
                )
                self._receive_error = f"{type(exc).__name__}: {exc}"
                self._receive_transport_failed = True
                with self._active_lock:
                    defer_reset = bool(
                        self._active_motion_kind == "insert"
                        and getattr(self, "_insert_motion_started", False)
                    )
                if not defer_reset:
                    self._mark_rtde_reset_required(
                        reason,
                        write_status=True,
                        failure_kind="cartesian_receive_exception",
                    )
                return None
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            return None
        return values

    def _read_actual_tcp_force(self) -> list[float] | None:
        """Return the six finite base-frame TCP wrench values used by insertion."""
        if bool(getattr(self, "_rtde_reset_required", False)):
            return None
        with self._receive_lock:
            get_actual_tcp_force = getattr(self.receive, "getActualTCPForce", None)
            if not callable(get_actual_tcp_force):
                return None
            try:
                values = [float(value) for value in list(get_actual_tcp_force())]
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                reason = (
                    "UR5e RTDE TCP force feedback transport failed: "
                    f"{type(exc).__name__}: {exc}. Use Repair Hardware Stack."
                )
                self._receive_error = f"{type(exc).__name__}: {exc}"
                self._receive_transport_failed = True
                with self._active_lock:
                    defer_reset = bool(
                        self._active_motion_kind == "insert"
                        and getattr(self, "_insert_motion_started", False)
                    )
                if not defer_reset:
                    self._mark_rtde_reset_required(
                        reason,
                        write_status=True,
                        failure_kind="tcp_force_receive_exception",
                    )
                return None
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            return None
        self._last_actual_tcp_force = list(values)
        return values

    def _read_actual_tcp_speed(self) -> list[float] | None:
        """Return the six finite base-frame active-TCP speed values."""
        if bool(getattr(self, "_rtde_reset_required", False)):
            return None
        with self._receive_lock:
            get_actual_tcp_speed = getattr(self.receive, "getActualTCPSpeed", None)
            if not callable(get_actual_tcp_speed):
                return None
            try:
                values = [float(value) for value in list(get_actual_tcp_speed())]
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                reason = (
                    "UR5e RTDE TCP speed feedback transport failed: "
                    f"{type(exc).__name__}: {exc}. Use Repair Hardware Stack."
                )
                self._receive_error = f"{type(exc).__name__}: {exc}"
                self._receive_transport_failed = True
                with self._active_lock:
                    defer_reset = bool(
                        self._active_motion_kind == "insert"
                        and getattr(self, "_insert_motion_started", False)
                    )
                if not defer_reset:
                    self._mark_rtde_reset_required(
                        reason,
                        write_status=True,
                        failure_kind="tcp_speed_receive_exception",
                    )
                return None
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            return None
        self._last_actual_tcp_speed = list(values)
        return values

    def _read_actual_tcp_transform(self) -> RigidTransform | None:
        values = self._read_actual_tcp_pose()
        if values is None:
            return None
        try:
            return _transform_from_rtde_pose(values)
        except ValueError:
            return None

    def _world_tool0_from_actual_tcp(
        self,
        actual_base_tcp: RigidTransform,
        *,
        world_base: RigidTransform,
        tool0_tcp: RigidTransform,
    ) -> RigidTransform:
        base_tool0 = _compose_transform(actual_base_tcp, _inverse_transform(tool0_tcp))
        return _compose_transform(world_base, base_tool0)

    def _cartesian_frame_validation_with_world_base(
        self,
    ) -> tuple[bool, str, float, float, RigidTransform | None]:
        """Validate one live world -> base sample and reuse it for frame evidence."""
        try:
            expected_world_base = _configured_cartesian_world_base()
        except RuntimeError as exc:
            self._cartesian_world_base_ready = False
            self._cartesian_world_base_message = str(exc)
            self._cartesian_world_base_expected = None
            self._cartesian_world_base_observed = None
            self._cartesian_world_base_position_error_m = math.inf
            self._cartesian_world_base_orientation_error_rad = math.inf
            return (
                False,
                f"Cartesian frame validation failed: {exc}",
                math.inf,
                math.inf,
                None,
            )
        self._cartesian_world_base_expected = expected_world_base
        try:
            observed_world_base = self._lookup_rigid_transform("world", "base")
        except (
            AttributeError,
            LookupError,
            OSError,
            RuntimeError,
            TransformException,
            TypeError,
            ValueError,
        ) as exc:
            self._cartesian_world_base_ready = False
            self._cartesian_world_base_message = (
                "Cartesian world -> base mount validation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self._cartesian_world_base_observed = None
            self._cartesian_world_base_position_error_m = math.inf
            self._cartesian_world_base_orientation_error_rad = math.inf
            return (
                False,
                self._cartesian_world_base_message,
                math.inf,
                math.inf,
                None,
            )
        self._cartesian_world_base_observed = observed_world_base
        mount_position_error, mount_orientation_error = _pose_errors(
            observed_world_base,
            expected_world_base,
        )
        self._cartesian_world_base_position_error_m = mount_position_error
        self._cartesian_world_base_orientation_error_rad = mount_orientation_error
        if mount_position_error > UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M:
            self._cartesian_world_base_ready = False
            self._cartesian_world_base_message = (
                "Cartesian world -> base mount validation failed: observed translation "
                "differs from protected ur5e.rtde.cartesian_world_base by "
                f"{mount_position_error:.6f} m; limit is "
                f"{UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M:.6f} m"
            )
            return (
                False,
                self._cartesian_world_base_message,
                math.inf,
                math.inf,
                observed_world_base,
            )
        if mount_orientation_error > UR5E_RTDE_CARTESIAN_ORIENTATION_TOLERANCE_RAD:
            self._cartesian_world_base_ready = False
            self._cartesian_world_base_message = (
                "Cartesian world -> base mount validation failed: observed orientation "
                "differs from protected ur5e.rtde.cartesian_world_base by "
                f"{mount_orientation_error:.6f} rad; limit is "
                f"{UR5E_RTDE_CARTESIAN_ORIENTATION_TOLERANCE_RAD:.6f} rad"
            )
            return (
                False,
                self._cartesian_world_base_message,
                math.inf,
                math.inf,
                observed_world_base,
            )
        self._cartesian_world_base_ready = True
        self._cartesian_world_base_message = (
            "live world -> base matches protected ur5e.rtde.cartesian_world_base"
        )
        try:
            tf_world_tool0 = self._lookup_rigid_transform("world", "tool0")
            tool0_tcp = self._active_tcp_offset()
            actual_base_tcp = self._read_actual_tcp_transform()
            if actual_base_tcp is None:
                raise RuntimeError("UR5e actual TCP pose is unavailable")
            reconstructed = self._world_tool0_from_actual_tcp(
                actual_base_tcp,
                world_base=observed_world_base,
                tool0_tcp=tool0_tcp,
            )
            position_error, orientation_error = _pose_errors(
                reconstructed,
                tf_world_tool0,
            )
        except (
            AttributeError,
            LookupError,
            OSError,
            RuntimeError,
            TransformException,
            TypeError,
            ValueError,
        ) as exc:
            return (
                False,
                f"Cartesian frame validation failed: {type(exc).__name__}: {exc}",
                math.inf,
                math.inf,
                observed_world_base,
            )
        if position_error > UR5E_RTDE_CARTESIAN_FRAME_POSITION_TOLERANCE_M:
            return (
                False,
                "Cartesian frame validation failed: reconstructed world -> tool0 "
                f"position differs from TF by {position_error:.6f} m",
                position_error,
                orientation_error,
                observed_world_base,
            )
        if orientation_error > UR5E_RTDE_CARTESIAN_FRAME_ORIENTATION_TOLERANCE_RAD:
            return (
                False,
                "Cartesian frame validation failed: reconstructed world -> tool0 "
                f"orientation differs from TF by {orientation_error:.6f} rad",
                position_error,
                orientation_error,
                observed_world_base,
            )
        return (
            True,
            "UR5e Cartesian frame validation ready",
            position_error,
            orientation_error,
            observed_world_base,
        )

    def _cartesian_frame_validation(
        self,
    ) -> tuple[bool, str, float, float]:
        """Validate TF against live RTDE TCP and the protected robot mount."""
        ready, message, position_error, orientation_error, _world_base = (
            self._cartesian_frame_validation_with_world_base()
        )
        self._cartesian_frame_ready = ready
        self._cartesian_frame_message = message
        self._cartesian_frame_position_error_m = position_error
        self._cartesian_frame_orientation_error_rad = orientation_error
        return ready, message, position_error, orientation_error

    def _validated_cartesian_world_base(
        self,
    ) -> tuple[RigidTransform, str, float, float]:
        """Return the exact live world -> base sample accepted by frame validation."""
        ready, message, position_error, orientation_error, world_base = (
            self._cartesian_frame_validation_with_world_base()
        )
        self._cartesian_frame_ready = ready
        self._cartesian_frame_message = message
        self._cartesian_frame_position_error_m = position_error
        self._cartesian_frame_orientation_error_rad = orientation_error
        if not ready or world_base is None:
            raise ValueError(message)
        return world_base, message, position_error, orientation_error

    def _pose_stamped_from_transform(self, value: RigidTransform) -> PoseStamped:
        translation, rotation = value
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.pose.position.x = float(translation[0])
        message.pose.position.y = float(translation[1])
        message.pose.position.z = float(translation[2])
        message.pose.orientation.x = float(rotation[0])
        message.pose.orientation.y = float(rotation[1])
        message.pose.orientation.z = float(rotation[2])
        message.pose.orientation.w = float(rotation[3])
        return message

    def _publish_cartesian_feedback(
        self,
        goal_handle: Any,
        actual_world_tool0: RigidTransform,
        *,
        position_error_m: float,
        orientation_error_rad: float,
    ) -> None:
        if MoveUR5eCartesian is None:
            return
        feedback = MoveUR5eCartesian.Feedback()
        feedback.actual_tool0_pose = self._pose_stamped_from_transform(actual_world_tool0)
        feedback.position_error_m = float(position_error_m)
        feedback.orientation_error_rad = float(orientation_error_rad)
        goal_handle.publish_feedback(feedback)

    def _execute_movel(
        self,
        target_base_tcp: RigidTransform,
        *,
        speed_m_s: float,
        acceleration_m_s2: float,
    ) -> bool:
        return self._execute_movel_pose(
            _rtde_pose_from_transform(target_base_tcp),
            speed_m_s=speed_m_s,
            acceleration_m_s2=acceleration_m_s2,
        )

    def _execute_movel_pose(
        self,
        target_base_tcp_pose: list[float],
        *,
        speed_m_s: float,
        acceleration_m_s2: float,
    ) -> bool:
        """Execute one exact RTDE pose, preserving a supplied rotation vector."""
        move_l = getattr(self.control, "moveL", None)
        if not callable(move_l):
            raise RuntimeError("RTDE control object has no moveL method")
        pose = [float(value) for value in target_base_tcp_pose]
        try:
            return bool(move_l(pose, speed_m_s, acceleration_m_s2, True))
        except TypeError:
            return bool(
                move_l(
                    pose,
                    speed=speed_m_s,
                    acceleration=acceleration_m_s2,
                    asynchronous=True,
                )
            )

    def _relative_cartesian_feedback(
        self,
        goal_handle: Any,
        translation_error_m: float,
        orientation_drift_rad: float,
    ) -> None:
        if MoveUR5eRelativeCartesian is None:
            return
        feedback = MoveUR5eRelativeCartesian.Feedback()
        feedback.translation_error_m = float(translation_error_m)
        feedback.orientation_drift_rad = float(orientation_drift_rad)
        goal_handle.publish_feedback(feedback)

    def _execute_relative_cartesian(self, goal_handle: Any) -> Any:
        """Execute one translation-only world-frame Step jog through RTDE moveL."""
        with self._active_lock:
            if self._shutdown_requested:
                goal_handle.abort()
                return self._relative_cartesian_result(-1, "UR5e RTDE server is stopping")
            if self._rtde_reset_required:
                goal_handle.abort()
                return self._relative_cartesian_result(
                    -1,
                    self._rtde_reset_reason or "UR5e RTDE reset required",
                )
            if self._active_goal is not None:
                goal_handle.abort()
                return self._relative_cartesian_result(
                    -1,
                    "UR5e RTDE motion already executing",
                )
            self._active_goal = goal_handle
            self._active_goal_status = None
            self._active_motion_kind = "relative_cartesian"

        status = _status_base()
        status.update(
            state="checking",
            message="checking translation-only UR5e RTDE Cartesian Step",
            motion_kind="relative_cartesian",
        )
        translation_error = math.inf
        orientation_drift = math.inf
        try:
            control_error = self._connect_control_for_goal()
            if control_error:
                raise RuntimeError(control_error)
            if not self._joint_states_fresh() or self._read_actual_q() is None:
                raise RuntimeError("UR5e RTDE feedback stale or missing")
            program_error = self._ensure_control_program_for_goal()
            if program_error:
                raise RuntimeError(program_error)
            (
                world_base,
                frame_message,
                frame_position_error,
                frame_orientation_error,
            ) = self._validated_cartesian_world_base()

            request = goal_handle.request
            world_delta = (
                float(request.world_translation_m.x),
                float(request.world_translation_m.y),
                float(request.world_translation_m.z),
            )
            if not all(math.isfinite(value) for value in world_delta):
                raise ValueError("world translation contains non-finite values")
            distance = math.sqrt(sum(value * value for value in world_delta))
            if not 0.0 < distance <= UR5E_RTDE_CARTESIAN_JOG_MAX_STEP_M:
                raise ValueError(
                    f"world translation magnitude {distance:.6f} m must be within "
                    f"(0, {UR5E_RTDE_CARTESIAN_JOG_MAX_STEP_M:.6f}] m"
                )
            speed = float(request.speed_m_s or UR5E_RTDE_CARTESIAN_SPEED_M_S)
            acceleration = float(request.acceleration_m_s2 or UR5E_RTDE_CARTESIAN_ACCEL_M_S2)
            if not math.isfinite(speed) or not 0.0 < speed <= UR5E_RTDE_CARTESIAN_MAX_SPEED_M_S:
                raise ValueError("relative Cartesian speed is outside the configured limit")
            if (
                not math.isfinite(acceleration)
                or not 0.0 < acceleration <= UR5E_RTDE_CARTESIAN_ACCEL_M_S2
            ):
                raise ValueError("relative Cartesian acceleration is outside the configured limit")

            base_delta = _rotate_vector(
                _quaternion_conjugate(world_base[1]),
                world_delta,
            )
            start_pose = self._read_actual_tcp_pose()
            if start_pose is None:
                raise RuntimeError("UR5e actual TCP pose is unavailable")
            target_pose = [
                start_pose[0] + base_delta[0],
                start_pose[1] + base_delta[1],
                start_pose[2] + base_delta[2],
                *start_pose[3:6],
            ]
            tool0_tcp = self._active_tcp_offset()
            actual_world_tool0 = self._world_tool0_from_actual_tcp(
                _transform_from_rtde_pose(start_pose),
                world_base=world_base,
                tool0_tcp=tool0_tcp,
            )
            target_world_tool0 = (
                tuple(actual_world_tool0[0][index] + world_delta[index] for index in range(3)),
                actual_world_tool0[1],
            )
            workspace_error = _workspace_error(target_world_tool0)
            if workspace_error:
                raise ValueError(workspace_error)
            within_safety_limits = getattr(self.control, "isPoseWithinSafetyLimits", None)
            if not callable(within_safety_limits):
                raise RuntimeError("RTDE control object has no isPoseWithinSafetyLimits method")
            if not bool(within_safety_limits(target_pose)):
                raise ValueError(
                    "UR controller rejected the relative Cartesian target as outside safety limits"
                )

            start_transform = _transform_from_rtde_pose(start_pose)
            target_transform = _transform_from_rtde_pose(target_pose)
            translation_error = distance
            orientation_drift = 0.0
            timeout_sec = max(4.0, distance / speed + UR5E_RTDE_RESULT_MARGIN_SEC)
            status.update(
                state="executing",
                message="executing translation-only UR5e RTDE Cartesian Step",
                blocked_reason="",
                rtde_connected=True,
                rtde_receive_connected=True,
                rtde_control_connected=True,
                joint_states_fresh=True,
                target_world_translation_m=list(world_delta),
                target_base_translation_m=list(base_delta),
                start_base_tcp=start_pose,
                target_base_tcp=target_pose,
                speed_m_s=speed,
                acceleration_m_s2=acceleration,
                cartesian_frame_validation_message=frame_message,
                cartesian_frame_position_error_m=frame_position_error,
                cartesian_frame_orientation_error_rad=frame_orientation_error,
            )
            self._write_active_goal_status(goal_handle, status)
            dispatch_ready, dispatch_message, _dispatch_position, _dispatch_orientation = (
                self._cartesian_frame_validation()
            )
            if not dispatch_ready:
                raise ValueError(dispatch_message)
            if not self._execute_movel_pose(
                target_pose,
                speed_m_s=speed,
                acceleration_m_s2=acceleration,
            ):
                raise RuntimeError("UR5e RTDE moveL returned False")

            deadline = time.monotonic() + timeout_sec
            stationary_since: float | None = None
            while rclpy.ok() and time.monotonic() < deadline:
                if goal_handle.is_cancel_requested:
                    self._stop_motion()
                    goal_handle.canceled()
                    status.update(state="canceled", message="UR5e Cartesian Step canceled")
                    self._finish_active_goal_status(goal_handle, status)
                    return self._relative_cartesian_result(-3, "canceled")
                actual_pose = self._read_actual_tcp_pose()
                if actual_pose is None:
                    time.sleep(0.02)
                    continue
                actual_transform = _transform_from_rtde_pose(actual_pose)
                translation_error, _target_orientation_error = _pose_errors(
                    actual_transform,
                    target_transform,
                )
                _unused_position_error, orientation_drift = _pose_errors(
                    actual_transform,
                    start_transform,
                )
                self._relative_cartesian_feedback(
                    goal_handle,
                    translation_error,
                    orientation_drift,
                )
                if orientation_drift > UR5E_RTDE_CARTESIAN_JOG_ORIENTATION_DRIFT_RAD:
                    self._stop_motion()
                    raise ValueError(
                        "translation-only UR5e Cartesian Step changed orientation by "
                        f"{orientation_drift:.6f} rad"
                    )
                velocities = self._read_actual_qd()
                stationary = bool(
                    velocities is not None
                    and max(abs(value) for value in velocities)
                    <= UR5E_RTDE_STATIONARY_MAX_JOINT_VEL_RAD_S
                )
                reached = translation_error <= UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M
                now = time.monotonic()
                stationary_since = stationary_since or now if reached and stationary else None
                status.update(
                    cartesian_translation_error_m=translation_error,
                    cartesian_orientation_drift_rad=orientation_drift,
                )
                self._write_active_goal_status(goal_handle, status)
                if (
                    stationary_since is not None
                    and now - stationary_since >= UR5E_RTDE_STATIONARY_HOLD_SEC
                ):
                    status.update(
                        state="succeeded",
                        message="translation-only UR5e Cartesian Step completed",
                    )
                    goal_handle.succeed()
                    self._finish_active_goal_status(goal_handle, status)
                    return self._relative_cartesian_result(
                        0,
                        "",
                        translation_error_m=translation_error,
                        orientation_drift_rad=orientation_drift,
                    )
                time.sleep(0.02)
            self._stop_motion()
            raise RuntimeError("translation-only UR5e Cartesian Step timed out")
        except ValueError as exc:
            reason = f"UR5e RTDE relative Cartesian target rejected: {exc}"
            status.update(state="blocked", message=reason, blocked_reason=reason)
            goal_handle.abort()
            self._finish_active_goal_status(goal_handle, status, latch_status=True)
            return self._relative_cartesian_result(
                -2,
                reason,
                translation_error_m=translation_error,
                orientation_drift_rad=orientation_drift,
            )
        except Exception as exc:
            reason = f"UR5e RTDE relative Cartesian failed: {type(exc).__name__}: {exc}"
            self._stop_motion()
            transport_text = str(exc).lower()
            if isinstance(exc, OSError) or (
                isinstance(exc, RuntimeError)
                and any(
                    marker in transport_text
                    for marker in (
                        "broken pipe",
                        "connection",
                        "disconnected",
                        "receive transport",
                        "send failed",
                        "socket",
                    )
                )
            ):
                self._mark_rtde_reset_required(reason, write_status=False)
            status.update(state="failed", message=reason, blocked_reason=reason)
            goal_handle.abort()
            self._finish_active_goal_status(goal_handle, status, latch_status=True)
            return self._relative_cartesian_result(
                -4,
                reason,
                translation_error_m=translation_error,
                orientation_drift_rad=orientation_drift,
            )
        finally:
            self._clear_active_goal(goal_handle)

    def _set_cartesian_jog(  # noqa: C901, PLR0915 - guarded jog refresh lifecycle.
        self,
        request: Any,
        response: Any,
    ) -> Any:
        """Start, refresh, or stop one watchdog-guarded translation-only RTDE jog."""
        if bool(request.stop):
            response.accepted, response.message = self._stop_cartesian_jog(
                "UR5e Cartesian jog stopped"
            )
            return response
        try:
            if self._shutdown_requested:
                raise RuntimeError("UR5e RTDE server is stopping")
            control_error = self._connect_control_for_goal()
            if control_error:
                raise RuntimeError(control_error)
            if not self._joint_states_fresh():
                raise RuntimeError("UR5e RTDE feedback stale or missing")
            with self._active_lock:
                jog_session_active = self._active_goal is self._jog_session_token
                cached_world_base = getattr(self, "_jog_world_base", None)
                cached_frame_message = str(
                    getattr(self, "_jog_frame_message", "") or ""
                )
                ready_world_base = getattr(
                    self,
                    "_cartesian_world_base_observed",
                    None,
                )
                cached_cartesian_frame_ready = bool(
                    getattr(self, "_cartesian_frame_ready", False)
                    and getattr(self, "_cartesian_world_base_ready", False)
                    and getattr(self, "_cartesian_jog_ready", False)
                    and ready_world_base is not None
                )
                ready_frame_message = str(
                    getattr(self, "_cartesian_frame_message", "") or ""
                )
            if jog_session_active and cached_world_base is not None:
                world_base = cached_world_base
                frame_message = cached_frame_message
            elif cached_cartesian_frame_ready:
                world_base = ready_world_base
                frame_message = ready_frame_message
            else:
                raise RuntimeError(
                    "UR5e Cartesian frame readiness is unavailable; "
                    "Repair Hardware Stack before Smooth Hold"
                )
            world_velocity_m_s = (
                float(request.world_linear_velocity_m_s.x),
                float(request.world_linear_velocity_m_s.y),
                float(request.world_linear_velocity_m_s.z),
            )
            if not all(math.isfinite(value) for value in world_velocity_m_s):
                raise ValueError("world Cartesian jog velocity contains non-finite values")
            nonzero_axes = sum(abs(value) > 1e-9 for value in world_velocity_m_s)
            if nonzero_axes != 1:
                raise ValueError("UR5e Cartesian jog requires exactly one world axis")
            speed_m_s = max(abs(value) for value in world_velocity_m_s)
            if speed_m_s > UR5E_RTDE_CARTESIAN_MAX_SPEED_M_S:
                raise ValueError("UR5e Cartesian jog velocity exceeds the configured limit")
            acceleration = float(request.acceleration_m_s2 or UR5E_RTDE_CARTESIAN_ACCEL_M_S2)
            if (
                not math.isfinite(acceleration)
                or not 0.0 < acceleration <= UR5E_RTDE_CARTESIAN_ACCEL_M_S2
            ):
                raise ValueError("UR5e Cartesian jog acceleration is outside the configured limit")
            watchdog_sec = min(
                UR5E_RTDE_CARTESIAN_JOG_MAX_WATCHDOG_SEC,
                max(UR5E_RTDE_CARTESIAN_JOG_MIN_WATCHDOG_SEC, float(request.watchdog_sec)),
            )
            base_velocity_m_s = _rotate_vector(
                _quaternion_conjugate(world_base[1]),
                world_velocity_m_s,
            )
            actual_pose = self._read_actual_tcp_pose()
            if actual_pose is None:
                raise RuntimeError("UR5e actual TCP pose is unavailable")
            lookahead_pose = [
                actual_pose[index] + base_velocity_m_s[index] * watchdog_sec for index in range(3)
            ] + actual_pose[3:6]
            within_safety_limits = getattr(self.control, "isPoseWithinSafetyLimits", None)
            if not callable(within_safety_limits) or not bool(within_safety_limits(lookahead_pose)):
                raise ValueError("UR controller rejected the Cartesian jog lookahead pose")
            with self._active_lock:
                if self._active_goal not in (None, self._jog_session_token):
                    raise RuntimeError("UR5e RTDE motion already executing")
                unchanged_jog = bool(
                    self._active_goal is self._jog_session_token
                    and self._jog_base_velocity_m_s is not None
                    and all(
                        math.isclose(current, previous, abs_tol=1e-12)
                        for current, previous in zip(
                            base_velocity_m_s,
                            self._jog_base_velocity_m_s,
                            strict=True,
                        )
                    )
                    and self._jog_acceleration_m_s2 is not None
                    and math.isclose(
                        acceleration,
                        self._jog_acceleration_m_s2,
                        abs_tol=1e-12,
                    )
                )
                self._active_goal = self._jog_session_token
                self._active_motion_kind = "cartesian_jog"
                self._jog_watchdog_deadline = time.monotonic() + watchdog_sec
                self._jog_world_base = world_base
                self._jog_frame_message = frame_message
            if unchanged_jog:
                response.accepted = True
                response.message = "UR5e Cartesian Smooth Hold active"
                return response
            jog_start = getattr(self.control, "jogStart", None)
            if not callable(jog_start):
                raise RuntimeError("RTDE control object has no jogStart method")
            feature_base = int(
                getattr(
                    self.control,
                    "FEATURE_BASE",
                    getattr(type(self.control), "FEATURE_BASE", 0),
                )
            )
            # ur_rtde jogStart translation values are mm/s. The ROS service and
            # every other Cartesian calculation in this server use metres/second.
            speeds = [
                base_velocity_m_s[0] * 1000.0,
                base_velocity_m_s[1] * 1000.0,
                base_velocity_m_s[2] * 1000.0,
                0.0,
                0.0,
                0.0,
            ]
            if not bool(jog_start(speeds, feature_base, acceleration)):
                raise RuntimeError("UR5e RTDE jogStart returned False")
            with self._active_lock:
                self._jog_base_velocity_m_s = tuple(base_velocity_m_s)
                self._jog_acceleration_m_s2 = acceleration
            status = _status_base()
            status.update(
                state="executing",
                message="executing translation-only UR5e Cartesian Smooth Hold",
                motion_kind="cartesian_jog",
                world_linear_velocity_m_s=list(world_velocity_m_s),
                base_linear_velocity_mm_s=speeds[:3],
                cartesian_frame_validation_message=frame_message,
                rtde_connected=True,
                rtde_receive_connected=True,
                rtde_control_connected=True,
                joint_states_fresh=True,
            )
            with self._active_lock:
                self._active_goal_status = dict(status)
            self._write_status(status)
            response.accepted = True
            response.message = "UR5e Cartesian Smooth Hold active"
            return response
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            reason = f"UR5e Cartesian Smooth Hold rejected: {type(exc).__name__}: {exc}"
            with self._active_lock:
                active = self._active_goal is self._jog_session_token
            if active:
                self._stop_cartesian_jog(reason)
            else:
                status = _status_base()
                status.update(
                    state="blocked",
                    message=reason,
                    blocked_reason=reason,
                )
                self._write_status(status)
            response.accepted = False
            response.message = reason
            return response

    def _execute_cartesian(  # noqa: C901, PLR0912, PLR0915 - guarded hardware lifecycle.
        self,
        goal_handle: Any,
    ) -> Any:
        with self._active_lock:
            if bool(getattr(self, "_shutdown_requested", False)):
                reason = "UR5e RTDE server is stopping"
                goal_handle.abort()
                return self._cartesian_result(-1, reason)
            if bool(getattr(self, "_rtde_reset_required", False)):
                reason = str(getattr(self, "_rtde_reset_reason", "")) or "UR5e RTDE reset required"
                goal_handle.abort()
                return self._cartesian_result(-1, reason)
            if self._latched_terminal_status is not None:
                reason = str(
                    self._latched_terminal_status.get("blocked_reason")
                    or self._latched_terminal_status.get("message")
                    or "UR5e RTDE server requires repair"
                )
                goal_handle.abort()
                return self._cartesian_result(-1, reason)
            if self._active_goal is not None:
                reason = "UR5e RTDE motion already executing"
                goal_handle.abort()
                return self._cartesian_result(-1, reason)
            self._active_goal = goal_handle
            self._active_goal_status = None
            self._active_motion_kind = "cartesian"

        status = _status_base()
        status.update(
            state="checking",
            message="checking guarded UR5e RTDE Cartesian target",
            motion_kind="cartesian",
        )
        position_error = math.inf
        orientation_error = math.inf
        try:
            control_error = self._connect_control_for_goal()
            if control_error:
                raise RuntimeError(control_error)
            if not self._joint_states_fresh() or self._read_actual_q() is None:
                raise RuntimeError("UR5e RTDE feedback stale or missing")
            program_error = self._ensure_control_program_for_goal()
            if program_error:
                raise RuntimeError(program_error)

            request = goal_handle.request
            speed = float(request.speed_m_s or UR5E_RTDE_CARTESIAN_SPEED_M_S)
            acceleration = float(request.acceleration_m_s2 or UR5E_RTDE_CARTESIAN_ACCEL_M_S2)
            if not math.isfinite(speed) or not 0.0 < speed <= UR5E_RTDE_CARTESIAN_MAX_SPEED_M_S:
                raise ValueError(
                    f"Cartesian speed {speed!r} must be within "
                    f"(0, {UR5E_RTDE_CARTESIAN_MAX_SPEED_M_S:.6f}] m/s"
                )
            if (
                not math.isfinite(acceleration)
                or not 0.0 < acceleration <= UR5E_RTDE_CARTESIAN_ACCEL_M_S2
            ):
                raise ValueError(
                    f"Cartesian acceleration {acceleration!r} must be within "
                    f"(0, {UR5E_RTDE_CARTESIAN_ACCEL_M_S2:.6f}] m/s^2"
                )

            (
                target_world_tool0,
                target_base_tcp,
                world_base,
                tool0_tcp,
            ) = self._resolve_cartesian_target(request.target_tool0_pose)
            frame_ready = bool(self._cartesian_frame_ready)
            frame_message = str(self._cartesian_frame_message)
            frame_position_error = float(self._cartesian_frame_position_error_m)
            frame_orientation_error = float(
                self._cartesian_frame_orientation_error_rad
            )
            if not frame_ready:
                raise ValueError(frame_message)
            target_rtde_pose = _rtde_pose_from_transform(target_base_tcp)
            within_safety_limits = getattr(self.control, "isPoseWithinSafetyLimits", None)
            if not callable(within_safety_limits):
                raise RuntimeError("RTDE control object has no isPoseWithinSafetyLimits method")
            if not bool(within_safety_limits(target_rtde_pose)):
                raise ValueError(
                    "UR controller rejected the Cartesian target as outside safety limits"
                )

            actual_base_tcp = self._read_actual_tcp_transform()
            if actual_base_tcp is None:
                raise RuntimeError("UR5e actual TCP pose is unavailable")
            actual_world_tool0 = self._world_tool0_from_actual_tcp(
                actual_base_tcp,
                world_base=world_base,
                tool0_tcp=tool0_tcp,
            )
            position_error, orientation_error = _pose_errors(
                actual_world_tool0,
                target_world_tool0,
            )
            distance = position_error
            timeout_sec = max(8.0, distance / speed + UR5E_RTDE_RESULT_MARGIN_SEC)
            status.update(
                state="executing",
                message="executing UR5e RTDE Cartesian moveL",
                blocked_reason="",
                rtde_connected=True,
                rtde_receive_connected=True,
                rtde_control_connected=True,
                joint_states_fresh=True,
                cartesian_speed_m_s=speed,
                cartesian_acceleration_m_s2=acceleration,
                cartesian_result_timeout_sec=timeout_sec,
                cartesian_position_error_m=position_error,
                cartesian_orientation_error_rad=orientation_error,
                target_world_tool0=list(target_world_tool0[0]),
                target_base_tcp=target_rtde_pose,
                cartesian_frame_validation_message=frame_message,
                cartesian_frame_position_error_m=frame_position_error,
                cartesian_frame_orientation_error_rad=frame_orientation_error,
            )
            self._write_active_goal_status(goal_handle, status)

            motion_required = bool(
                position_error > UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M
                or orientation_error > UR5E_RTDE_CARTESIAN_ORIENTATION_TOLERANCE_RAD
            )
            if motion_required:
                dispatch_ready, dispatch_message, _dispatch_position, _dispatch_orientation = (
                    self._cartesian_frame_validation()
                )
                if not dispatch_ready:
                    raise ValueError(dispatch_message)
                if not self._execute_movel(
                    target_base_tcp,
                    speed_m_s=speed,
                    acceleration_m_s2=acceleration,
                ):
                    raise RuntimeError("UR5e RTDE moveL returned False")

            deadline = time.monotonic() + timeout_sec
            stationary_since: float | None = None
            while rclpy.ok() and time.monotonic() < deadline:
                if goal_handle.is_cancel_requested:
                    self._stop_motion()
                    status.update(state="canceled", message="UR5e RTDE Cartesian move canceled")
                    goal_handle.canceled()
                    self._finish_active_goal_status(goal_handle, status)
                    return self._cartesian_result(
                        -3,
                        "canceled",
                        position_error_m=position_error,
                        orientation_error_rad=orientation_error,
                    )
                if bool(getattr(self, "_rtde_reset_required", False)):
                    raise RuntimeError(
                        str(getattr(self, "_rtde_reset_reason", "")) or "UR5e RTDE reset required"
                    )
                actual_base_tcp = self._read_actual_tcp_transform()
                if actual_base_tcp is None:
                    time.sleep(0.02)
                    continue
                actual_world_tool0 = self._world_tool0_from_actual_tcp(
                    actual_base_tcp,
                    world_base=world_base,
                    tool0_tcp=tool0_tcp,
                )
                position_error, orientation_error = _pose_errors(
                    actual_world_tool0,
                    target_world_tool0,
                )
                self._publish_cartesian_feedback(
                    goal_handle,
                    actual_world_tool0,
                    position_error_m=position_error,
                    orientation_error_rad=orientation_error,
                )
                velocities = self._read_actual_qd()
                stationary = bool(
                    velocities is not None
                    and max(abs(value) for value in velocities)
                    <= UR5E_RTDE_STATIONARY_MAX_JOINT_VEL_RAD_S
                )
                reached = bool(
                    position_error <= UR5E_RTDE_CARTESIAN_POSITION_TOLERANCE_M
                    and orientation_error <= UR5E_RTDE_CARTESIAN_ORIENTATION_TOLERANCE_RAD
                )
                now = time.monotonic()
                stationary_since = stationary_since or now if reached and stationary else None
                status.update(
                    cartesian_position_error_m=position_error,
                    cartesian_orientation_error_rad=orientation_error,
                    stationary_hold_sec=(now - stationary_since if stationary_since else 0.0),
                )
                self._write_active_goal_status(goal_handle, status)
                if (
                    stationary_since is not None
                    and now - stationary_since >= UR5E_RTDE_STATIONARY_HOLD_SEC
                ):
                    status.update(
                        state="succeeded",
                        message=(
                            "UR5e RTDE Cartesian target reached and completed stationary hold"
                        ),
                    )
                    goal_handle.succeed()
                    self._finish_active_goal_status(goal_handle, status)
                    return self._cartesian_result(
                        0,
                        "",
                        position_error_m=position_error,
                        orientation_error_rad=orientation_error,
                    )
                time.sleep(0.02)

            self._stop_motion()
            reason = "UR5e RTDE Cartesian result timeout"
            status.update(state="failed", message=reason, blocked_reason=reason)
            goal_handle.abort()
            self._finish_active_goal_status(goal_handle, status, latch_status=True)
            return self._cartesian_result(
                -3,
                reason,
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
            )
        except ValueError as exc:
            reason = f"UR5e RTDE Cartesian target rejected: {exc}"
            status.update(state="blocked", message=reason, blocked_reason=reason)
            goal_handle.abort()
            self._finish_active_goal_status(goal_handle, status)
            return self._cartesian_result(
                -2,
                reason,
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
            )
        except Exception as exc:
            reason = f"UR5e RTDE Cartesian failed: {type(exc).__name__}: {exc}"
            self._stop_motion()
            if isinstance(exc, (OSError, RuntimeError)):
                self._mark_rtde_reset_required(reason, write_status=False)
            status.update(
                state="failed",
                message=reason,
                blocked_reason=reason,
                rtde_reset_required=bool(getattr(self, "_rtde_reset_required", False)),
            )
            goal_handle.abort()
            self._finish_active_goal_status(
                goal_handle,
                status,
                latch_status=bool(getattr(self, "_rtde_reset_required", False)),
            )
            return self._cartesian_result(
                -4,
                reason,
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
            )
        finally:
            self._clear_active_goal(goal_handle)

    def _execute(  # noqa: C901, PLR0912, PLR0915 - one guarded hardware status lifecycle.
        self,
        goal_handle: Any,
    ) -> FollowJointTrajectory.Result:
        with self._active_lock:
            if bool(getattr(self, "_rtde_reset_required", False)):
                reason = str(getattr(self, "_rtde_reset_reason", "")) or "UR5e RTDE reset required"
                status = _status_base()
                status.update(
                    state="failed",
                    blocked_reason=reason,
                    message=reason,
                )
                self._write_status(status)
                goal_handle.abort()
                return self._result(-1, reason)
            if self._latched_terminal_status is not None:
                reason = str(
                    self._latched_terminal_status.get("blocked_reason")
                    or self._latched_terminal_status.get("message")
                    or "UR5e RTDE trajectory server requires repair"
                )
                self._write_status(self._latched_terminal_status)
                goal_handle.abort()
                return self._result(-1, reason)
            if self._active_goal is not None:
                reason = "UR5e RTDE trajectory already executing"
                status = _status_base()
                status.update(
                    state="blocked",
                    blocked_reason=reason,
                    message=f"blocked: {reason}",
                    rtde_connected=self.control is not None and self.receive is not None,
                    rtde_receive_connected=self.receive is not None,
                    rtde_control_connected=self.control is not None,
                    joint_states_fresh=self._joint_states_fresh(),
                )
                self._write_status(status)
                goal_handle.abort()
                return self._result(-1, reason)
            self._active_goal = goal_handle
            self._active_goal_status = None
            self._active_motion_kind = "joint"
        status = _status_base()
        control_error = self._connect_control_for_goal()
        if control_error:
            status.update(
                state="blocked",
                blocked_reason=control_error,
                message=f"blocked: {control_error}",
                rtde_connected=False,
                rtde_receive_connected=self.receive is not None,
                rtde_control_connected=False,
                joint_states_fresh=self._joint_states_fresh(),
            )
            goal_handle.abort()
            self._finish_active_goal_status(goal_handle, status)
            return self._result(-1, control_error)
        control_program_error = self._ensure_control_program_for_goal()
        if control_program_error:
            status.update(
                state="blocked",
                blocked_reason=control_program_error,
                message=f"blocked: {control_program_error}",
                rtde_connected=False,
                rtde_receive_connected=self.receive is not None,
                rtde_control_connected=False,
                joint_states_fresh=self._joint_states_fresh(),
            )
            goal_handle.abort()
            self._finish_active_goal_status(goal_handle, status)
            return self._result(-1, control_program_error)
        status["rtde_connected"] = self.control is not None and self.receive is not None
        status["rtde_receive_connected"] = self.receive is not None
        status["rtde_control_connected"] = self.control is not None
        current_positions = self._current_position_map()
        status["joint_states_fresh"] = self._joint_states_fresh()
        if current_positions is None or not status["joint_states_fresh"]:
            reason = "UR5e RTDE feedback stale or missing"
            status.update(state="blocked", blocked_reason=reason, message=f"blocked: {reason}")
            goal_handle.abort()
            self._finish_active_goal_status(goal_handle, status)
            return self._result(-1, reason)

        requested_trajectory = goal_handle.request.trajectory
        requested_final_time = max(
            (_point_seconds(point) for point in list(requested_trajectory.points)),
            default=0.0,
        )
        ok, guarded_trajectory, status = prepare_rtde_trajectory(
            requested_trajectory,
            current_positions,
        )
        status["rtde_connected"] = self.control is not None and self.receive is not None
        status["rtde_receive_connected"] = self.receive is not None
        status["rtde_control_connected"] = self.control is not None
        status["joint_states_fresh"] = self._joint_states_fresh()
        if not ok or guarded_trajectory is None:
            goal_handle.abort()
            self._finish_active_goal_status(goal_handle, status)
            return self._result(-1, str(status.get("blocked_reason") or "RTDE trajectory rejected"))

        guarded_point_count = len(list(guarded_trajectory.points))
        path = rtde_movej_path(guarded_trajectory)
        status["movej_path_points"] = len(path)
        status["movej_duplicate_points_removed"] = max(0, guarded_point_count - len(path))
        status.update(state="executing", message="executing UR5e RTDE moveJ path")
        final_q = [float(value) for value in path[-1][: len(ARM_JOINTS)]]
        initial_q = [float(current_positions[joint]) for joint in ARM_JOINTS]
        initial_target_error, _initial_target_joint = _max_named_delta(
            ARM_JOINTS,
            final_q,
            dict(zip(ARM_JOINTS, initial_q, strict=True)),
        )
        final_time = max(
            (_point_seconds(point) for point in list(guarded_trajectory.points)),
            default=0.0,
        )
        result_timeout_sec = _trajectory_result_timeout_sec(requested_final_time, final_time)
        execution_started = time.monotonic()
        deadline = execution_started + result_timeout_sec
        status.update(
            trajectory_requested_final_time_sec=requested_final_time,
            trajectory_guarded_final_time_sec=final_time,
            trajectory_result_timeout_sec=result_timeout_sec,
            trajectory_elapsed_sec=0.0,
            control_program_start_timeout_sec=UR5E_RTDE_CONTROL_PROGRAM_START_TIMEOUT_SEC,
            motion_start_timeout_sec=UR5E_RTDE_MOTION_START_TIMEOUT_SEC,
            motion_start_delta_rad=UR5E_RTDE_MOTION_START_DELTA_RAD,
            motion_required=initial_target_error > UR5E_RTDE_GOAL_TOLERANCE_RAD,
            motion_started=initial_target_error <= UR5E_RTDE_GOAL_TOLERANCE_RAD,
            motion_start_elapsed_sec=(
                0.0 if initial_target_error <= UR5E_RTDE_GOAL_TOLERANCE_RAD else None
            ),
            initial_positions_rad=list(initial_q),
            final_target_positions_rad=list(final_q),
            actual_positions_rad=list(initial_q),
        )
        self._write_active_goal_status(goal_handle, status)
        stationary_since: float | None = None
        stopped_away_since: float | None = None
        previous_actual: list[float] | None = None
        previous_actual_at: float | None = None
        max_observed_joint_velocity: float | None = None
        motion_started = initial_target_error <= UR5E_RTDE_GOAL_TOLERANCE_RAD
        initial_feedback_timestamp = self._read_feedback_timestamp()
        status["rtde_feedback_timestamp_initial_sec"] = initial_feedback_timestamp
        status["rtde_feedback_timestamp_sec"] = initial_feedback_timestamp
        status["rtde_feedback_timestamp_advanced"] = False
        last_feedback_timestamp = initial_feedback_timestamp
        last_feedback_advance_at = execution_started
        try:
            dispatch_started = time.monotonic()
            rtde_result, rtde_command_mode = self._execute_movej_path(path)
            status["rtde_result"] = rtde_result
            status["rtde_command_mode"] = rtde_command_mode
            status["rtde_async_dispatch_elapsed_sec"] = time.monotonic() - dispatch_started
            self._write_active_goal_status(goal_handle, status)
            if str(rtde_result).strip() == "False":
                reason = "UR5e RTDE moveJ returned False"
                status.update(state="failed", message=reason, blocked_reason=reason)
                goal_handle.abort()
                self._finish_active_goal_status(goal_handle, status)
                return self._result(-4, reason)
            while rclpy.ok() and time.monotonic() < deadline:
                if goal_handle.is_cancel_requested:
                    self._stop_motion()
                    status.update(state="canceled", message="UR5e RTDE trajectory canceled")
                    goal_handle.canceled()
                    self._finish_active_goal_status(goal_handle, status)
                    return self._result(-1, "canceled")
                actual = self._read_actual_q()
                actual_at = time.monotonic()
                if bool(getattr(self, "_rtde_reset_required", False)):
                    reason = (
                        str(getattr(self, "_rtde_reset_reason", ""))
                        or "UR5e RTDE feedback transport failed. Physical state is "
                        "uncertain; use Reset UR5e RTDE in Interactive Teleop."
                    )
                    self._stop_motion()
                    status.update(
                        state="failed",
                        message=reason,
                        blocked_reason=reason,
                        rtde_connected=False,
                        rtde_receive_connected=False,
                        rtde_reset_required=True,
                        joint_states_fresh=False,
                        trajectory_elapsed_sec=actual_at - execution_started,
                    )
                    goal_handle.abort()
                    self._finish_active_goal_status(
                        goal_handle,
                        status,
                        latch_status=True,
                    )
                    return self._result(-1, reason)
                feedback_timestamp = self._read_feedback_timestamp()
                status["rtde_feedback_timestamp_sec"] = feedback_timestamp
                feedback_advanced = feedback_timestamp is not None and (
                    last_feedback_timestamp is None or feedback_timestamp > last_feedback_timestamp
                )
                if feedback_advanced:
                    last_feedback_timestamp = feedback_timestamp
                    last_feedback_advance_at = actual_at
                    status["rtde_feedback_timestamp_advanced"] = True
                    status["rtde_feedback_gap_sec"] = 0.0
                    status["rtde_feedback_reconnect_error"] = ""
                else:
                    feedback_gap_sec = actual_at - last_feedback_advance_at
                    status["rtde_feedback_gap_sec"] = feedback_gap_sec
                    if feedback_gap_sec >= UR5E_RTDE_FEEDBACK_RECOVERY_TIMEOUT_SEC:
                        reason = (
                            "UR5e RTDE trajectory feedback stopped advancing and did not "
                            "recover within "
                            f"{UR5E_RTDE_FEEDBACK_RECOVERY_TIMEOUT_SEC:.2f} s. "
                            "Physical state is uncertain; use Reset UR5e RTDE in "
                            "Interactive Teleop."
                        )
                        self._stop_motion()
                        self._mark_rtde_reset_required(reason, write_status=False)
                        status.update(
                            state="failed",
                            message=reason,
                            blocked_reason=reason,
                            rtde_connected=False,
                            rtde_receive_connected=False,
                            rtde_reset_required=True,
                            joint_states_fresh=False,
                            trajectory_elapsed_sec=actual_at - execution_started,
                        )
                        goal_handle.abort()
                        self._finish_active_goal_status(
                            goal_handle,
                            status,
                            latch_status=True,
                        )
                        return self._result(-1, reason)
                    time.sleep(0.02)
                    continue

                if actual is not None:
                    status["actual_positions_rad"] = [float(value) for value in actual]
                    max_delta, max_joint = _max_named_delta(
                        ARM_JOINTS,
                        final_q,
                        dict(zip(ARM_JOINTS, actual, strict=True)),
                    )
                    status["final_joint_error_rad"] = max_delta
                    status["final_joint_error_joint"] = max_joint
                    actual_qd = self._read_actual_qd()
                    if actual_qd is not None:
                        max_actual_joint_velocity = max(abs(value) for value in actual_qd)
                    elif previous_actual is not None and previous_actual_at is not None:
                        sample_sec = actual_at - previous_actual_at
                        max_actual_joint_velocity = (
                            max(
                                abs(current - previous) / sample_sec
                                for current, previous in zip(
                                    actual,
                                    previous_actual,
                                    strict=True,
                                )
                            )
                            if sample_sec > 0.0
                            else None
                        )
                    else:
                        max_actual_joint_velocity = None
                    status["max_actual_joint_velocity_rad_s"] = max_actual_joint_velocity
                    if max_actual_joint_velocity is not None:
                        max_observed_joint_velocity = max(
                            max_observed_joint_velocity or 0.0,
                            max_actual_joint_velocity,
                        )
                    status["max_observed_joint_velocity_rad_s"] = max_observed_joint_velocity
                    status["trajectory_elapsed_sec"] = actual_at - execution_started

                    start_delta = max(
                        abs(current - initial)
                        for current, initial in zip(actual, initial_q, strict=True)
                    )
                    status["motion_start_observed_delta_rad"] = start_delta
                    if not motion_started and (
                        start_delta >= UR5E_RTDE_MOTION_START_DELTA_RAD
                        or (
                            max_actual_joint_velocity is not None
                            and max_actual_joint_velocity > UR5E_RTDE_STATIONARY_MAX_JOINT_VEL_RAD_S
                        )
                    ):
                        motion_started = True
                        status["motion_started"] = True
                        status["motion_start_elapsed_sec"] = actual_at - execution_started

                    target_reached = max_delta <= UR5E_RTDE_GOAL_TOLERANCE_RAD
                    stationary = (
                        max_actual_joint_velocity is not None
                        and max_actual_joint_velocity <= UR5E_RTDE_STATIONARY_MAX_JOINT_VEL_RAD_S
                    )
                    if target_reached and stationary:
                        if stationary_since is None:
                            stationary_since = actual_at
                        status["stationary_hold_sec"] = actual_at - stationary_since
                    else:
                        stationary_since = None
                        status["stationary_hold_sec"] = 0.0

                    if motion_started and stationary and not target_reached:
                        if stopped_away_since is None:
                            stopped_away_since = actual_at
                        status["stopped_away_hold_sec"] = actual_at - stopped_away_since
                    else:
                        stopped_away_since = None
                        status["stopped_away_hold_sec"] = 0.0

                    with self._active_lock:
                        if self._active_goal is goal_handle:
                            self._active_goal_status = dict(status)

                    if (
                        stationary_since is not None
                        and actual_at - stationary_since >= UR5E_RTDE_STATIONARY_HOLD_SEC
                    ):
                        status.update(
                            state="succeeded",
                            message=(
                                "UR5e RTDE trajectory reached final joint target and "
                                "completed stationary hold"
                            ),
                            trajectory_elapsed_sec=actual_at - execution_started,
                        )
                        goal_handle.succeed()
                        self._finish_active_goal_status(goal_handle, status)
                        return self._result(0, "")

                    if (
                        not motion_started
                        and not target_reached
                        and actual_at - execution_started >= UR5E_RTDE_MOTION_START_TIMEOUT_SEC
                    ):
                        if initial_feedback_timestamp is not None and not status.get(
                            "rtde_feedback_timestamp_advanced"
                        ):
                            reason = (
                                "UR5e RTDE trajectory feedback did not advance after moveJ "
                                "was accepted"
                            )
                        else:
                            reason = "UR5e RTDE trajectory did not start after moveJ was accepted"
                        self._stop_motion()
                        status.update(
                            state="failed",
                            message=reason,
                            blocked_reason=reason,
                            trajectory_elapsed_sec=actual_at - execution_started,
                        )
                        goal_handle.abort()
                        self._finish_active_goal_status(
                            goal_handle,
                            status,
                            latch_status=True,
                        )
                        return self._result(-1, reason)

                    if (
                        stopped_away_since is not None
                        and actual_at - stopped_away_since >= UR5E_RTDE_STOPPED_AWAY_HOLD_SEC
                    ):
                        reason = "UR5e RTDE trajectory ended before reaching the final joint target"
                        status.update(
                            state="failed",
                            message=reason,
                            blocked_reason=reason,
                            trajectory_elapsed_sec=actual_at - execution_started,
                        )
                        goal_handle.abort()
                        self._finish_active_goal_status(
                            goal_handle,
                            status,
                            latch_status=True,
                        )
                        return self._result(-1, reason)
                    previous_actual = list(actual)
                    previous_actual_at = actual_at
                time.sleep(0.02)
            reason = "UR5e RTDE trajectory result timeout"
            self._stop_motion()
            status.update(
                state="failed",
                message=reason,
                blocked_reason=reason,
                trajectory_elapsed_sec=time.monotonic() - execution_started,
            )
            goal_handle.abort()
            self._finish_active_goal_status(goal_handle, status, latch_status=True)
            return self._result(-1, reason)
        except Exception as exc:
            reason = f"UR5e RTDE trajectory failed: {type(exc).__name__}: {exc}"
            self._stop_motion()
            self._mark_rtde_reset_required(reason, write_status=False)
            status["rtde_connected"] = False
            status["rtde_control_connected"] = False
            status["rtde_receive_connected"] = self.receive is not None
            status.update(
                state="failed",
                message=reason,
                blocked_reason=reason,
                rtde_reset_required=True,
            )
            goal_handle.abort()
            self._finish_active_goal_status(goal_handle, status, latch_status=True)
            return self._result(-4, reason)
        finally:
            self._clear_active_goal(goal_handle)

    def destroy_node(self) -> bool:
        self._shutdown_requested = True
        with self._active_lock:
            active_goal = self._active_goal
        if active_goal is not None:
            self._stop_motion()
        self._disconnect_rtde_interfaces()
        return super().destroy_node()


def build_parser() -> argparse.ArgumentParser:
    config = _load_hardware_arms_config(DEFAULT_CONFIG_FILE)
    default_status_file = _nested(
        config,
        ("digital_twin", "status_paths", "ur5e_rtde_trajectory"),
        str(DEFAULT_STATUS_FILE),
    )
    parser = argparse.ArgumentParser(description="UR5e RTDE FollowJointTrajectory action server")
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--status-file", default=str(default_status_file))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument("--publish-rate-hz", type=float, default=50.0)
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="publish read-only UR5e joint feedback without RTDE control or an action server",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_hardware_arms_config(Path(args.config))
    rclpy.init()
    node = UR5eRTDETrajectoryServer(
        robot_ip=str(args.robot_ip),
        status_file=Path(args.status_file),
        publish_rate_hz=float(args.publish_rate_hz),
        monitor_only=bool(args.monitor_only),
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    failed = False
    try:
        executor.spin()
    except Exception as exc:
        failed = True
        status = _status_base()
        status.update(
            state="failed",
            blocked_reason=f"UR5e RTDE trajectory server exited: {type(exc).__name__}: {exc}",
            message=f"UR5e RTDE trajectory server exited: {type(exc).__name__}: {exc}",
            rtde_connected=False,
            rtde_receive_connected=False,
            rtde_control_connected=False,
            joint_states_fresh=False,
        )
        node._write_status(status)
        raise
    finally:
        if not failed and bool(getattr(node, "_rtde_reset_required", False)):
            reason = str(getattr(node, "_rtde_reset_reason", "")) or ("UR5e RTDE reset required")
            status = _status_base()
            status.update(
                state="failed",
                blocked_reason=reason,
                message=reason,
                rtde_connected=False,
                rtde_receive_connected=False,
                rtde_control_connected=False,
                joint_states_fresh=False,
            )
            node._write_status(status)
        elif not failed:
            status = _status_base()
            status.update(
                state="stopped",
                blocked_reason="UR5e RTDE trajectory server stopped",
                message="UR5e RTDE trajectory server stopped",
                rtde_connected=False,
                rtde_receive_connected=False,
                rtde_control_connected=False,
                joint_states_fresh=False,
            )
            node._write_status(status)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
