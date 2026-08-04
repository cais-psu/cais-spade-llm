#!/usr/bin/env python3.10
"""UR5e RTDE-backed FollowJointTrajectory action server."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import rclpy
import yaml
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
ACTION_NAME = "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
DEFAULT_STATUS_FILE = Path("/tmp") / "cais_ur5e_rtde_trajectory_status.json"
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


def _apply_hardware_arms_config(config_path: Path) -> None:
    """Load UR5e RTDE runtime limits from xarm6_ur5e_hardware_runtime.yaml."""
    global ACTION_NAME
    global UR5E_RTDE_MAX_JOINT_VEL_RAD_S
    global UR5E_RTDE_MAX_JOINT_ACCEL_RAD_S2
    global UR5E_RTDE_MAX_JOINT_JERK_RAD_S3
    global UR5E_RTDE_SHOULDER_PAN_EXTRA_SCALE
    global UR5E_RTDE_MOVEJ_SPEED_RAD_S
    global UR5E_RTDE_MOVEJ_ACCEL_RAD_S2
    global HARDWARE_ARMS_CONFIG_FILE

    HARDWARE_ARMS_CONFIG_FILE = str(Path(config_path).expanduser())
    config = _load_hardware_arms_config(config_path)
    ACTION_NAME = _str(
        config,
        ("ur5e", "hardware_trajectory_action"),
        ACTION_NAME,
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

UR5E_RTDE_CURRENT_HOLD_SEC = 0.25
UR5E_RTDE_MIN_POINT_SPACING_SEC = 0.10
UR5E_RTDE_MAX_JOINT_VEL_RAD_S = 0.324
UR5E_RTDE_MAX_JOINT_ACCEL_RAD_S2 = 0.486
UR5E_RTDE_MAX_JOINT_JERK_RAD_S3 = 2.025
UR5E_RTDE_SHOULDER_PAN_EXTRA_SCALE = 1.0
UR5E_RTDE_START_TOLERANCE_RAD = 0.15
UR5E_RTDE_GOAL_TOLERANCE_RAD = 0.025
UR5E_RTDE_MOVEJ_SPEED_RAD_S = 0.486
UR5E_RTDE_MOVEJ_ACCEL_RAD_S2 = 0.81
UR5E_RTDE_INTERMEDIATE_BLEND_RAD = 0.005
UR5E_RTDE_STOP_ACCEL_RAD_S2 = 0.50
UR5E_RTDE_FEEDBACK_STALE_SEC = 2.0
UR5E_RTDE_RESULT_MARGIN_SEC = 12.0

_apply_hardware_arms_config(DEFAULT_CONFIG_FILE)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _duration_seconds(duration: Any) -> float:
    return float(getattr(duration, "sec", 0) or 0) + float(getattr(duration, "nanosec", 0) or 0) / 1e9


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
        previous_positions = {name: float(current_positions[name]) for name in joint_names if name in current_positions}

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
        previous_positions = {name: float(current_positions[name]) for name in joint_names if name in current_positions}

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
                    velocities[joint] = (float(positions[joint]) - float(previous_positions[joint])) / dt
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
                if dt <= 0.0 or math.isinf(previous_velocities[joint]) or math.isinf(velocities[joint]):
                    acceleration = math.inf
                else:
                    acceleration = abs(float(velocities[joint]) - float(previous_velocities[joint])) / dt
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
                if dt <= 0.0 or math.isinf(previous_velocities[joint]) or math.isinf(velocities[joint]):
                    accelerations[joint] = math.inf
                else:
                    accelerations[joint] = (float(velocities[joint]) - float(previous_velocities[joint])) / dt
            if previous_accelerations is not None:
                for joint in joint_names:
                    if joint not in previous_accelerations or joint not in accelerations:
                        continue
                    if dt <= 0.0 or math.isinf(previous_accelerations[joint]) or math.isinf(accelerations[joint]):
                        jerk = math.inf
                    else:
                        jerk = abs(float(accelerations[joint]) - float(previous_accelerations[joint])) / dt
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
    start_delta, start_joint = _max_named_delta(joint_names, list(points[0].positions), current_positions)
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
        _retime_points(points, scale=velocity_time_scale, min_spacing_sec=float(min_point_spacing_sec))

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
        _retime_points(points, scale=acceleration_time_scale, min_spacing_sec=float(min_point_spacing_sec))

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
    return {
        "updated_at": time.time(),
        "action": ACTION_NAME,
        "action_name": ACTION_NAME,
        "state": "checking",
        "message": "",
        "blocked_reason": "",
        "rtde_connected": False,
        "rtde_receive_connected": False,
        "rtde_control_connected": False,
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
        "final_joint_error_rad": None,
        "final_joint_error_joint": "",
        "rtde_result": "",
        "rtde_command_mode": "",
        "rtde_async_dispatch_elapsed_sec": None,
        "max_joint_velocity_limit_rad_s": UR5E_RTDE_MAX_JOINT_VEL_RAD_S,
        "max_joint_acceleration_limit_rad_s2": UR5E_RTDE_MAX_JOINT_ACCEL_RAD_S2,
        "max_joint_jerk_limit_rad_s3": UR5E_RTDE_MAX_JOINT_JERK_RAD_S3,
        "movej_speed_rad_s": UR5E_RTDE_MOVEJ_SPEED_RAD_S,
        "movej_acceleration_rad_s2": UR5E_RTDE_MOVEJ_ACCEL_RAD_S2,
        "shoulder_pan_extra_scale": UR5E_RTDE_SHOULDER_PAN_EXTRA_SCALE,
        "hardware_runtime_config": HARDWARE_ARMS_CONFIG_FILE,
        "hardware_arms_config": HARDWARE_ARMS_CONFIG_FILE,
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
        status.update(
            state="blocked",
            message=str(guard_status.get("message") or guard_status.get("blocked_reason") or "trajectory rejected"),
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
        UR5E_RTDE_MOVEJ_ACCEL_RAD_S2
        if acceleration_rad_s2 is None
        else float(acceleration_rad_s2)
    )
    joint_names = [str(name) for name in list(getattr(trajectory, "joint_names", []) or [])]
    points = list(getattr(trajectory, "points", []) or [])
    index_by_joint = {name: index for index, name in enumerate(joint_names)}
    path: list[list[float]] = []
    for point_index, point in enumerate(points):
        positions = list(point.positions)
        q = [float(positions[index_by_joint[joint]]) for joint in ARM_JOINTS]
        blend = 0.0 if point_index == len(points) - 1 else max(0.0, float(blend_rad))
        path.append([*q, speed, acceleration, blend])
    return path


class UR5eRTDETrajectoryServer(Node):
    def __init__(
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
        self._receive_lock = threading.Lock()
        self._next_receive_connect_monotonic = 0.0
        self._receive_error = ""
        self._control_error = ""
        self._joint_status_announced = False
        self._next_status_heartbeat_monotonic = 0.0
        self._active_lock = threading.Lock()
        self._active_goal = None
        self._joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._timer = self.create_timer(1.0 / max(1.0, float(publish_rate_hz)), self._publish_joint_state)
        self._action_server = None
        if not self.monitor_only:
            self._action_server = ActionServer(
                self,
                FollowJointTrajectory,
                ACTION_NAME,
                execute_callback=self._execute,
                cancel_callback=self._cancel,
            )
        status = _status_base()
        if not self.robot_ip:
            status.update(state="blocked", blocked_reason="--robot-ip is required", message="blocked: --robot-ip is required")
            self._write_status(status)
            return
        self._connect_rtde()

    def _write_status(self, payload: dict[str, Any]) -> None:
        body = dict(payload)
        body["monitor_only"] = self.monitor_only
        if self.monitor_only:
            body["action"] = ""
            body["action_name"] = ""
        body["ros_domain_id"] = self.ros_domain_id
        body["process_id"] = os.getpid()
        body["updated_at"] = time.time()
        _atomic_json_write(self.status_file, body)

    def _connect_rtde(self) -> None:
        status = _status_base()
        receive_connected = False
        control_connected = False
        try:
            if self.receive_factory is None:
                import rtde_receive

                self.receive_factory = rtde_receive.RTDEReceiveInterface
            self.receive = self.receive_factory(self.robot_ip)
            receive_connected = True
            self._receive_error = ""
        except (ImportError, OSError, RuntimeError) as exc:
            self.receive = None
            self._receive_error = f"{type(exc).__name__}: {exc}"
            self._next_receive_connect_monotonic = time.monotonic() + 1.0

        if self.monitor_only:
            self.control = None
            self._control_error = "disabled for read-only calibration monitoring"
        else:
            try:
                if self.control_factory is None:
                    import rtde_control

                    self.control_factory = rtde_control.RTDEControlInterface
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

    def _reconnect_receive_locked(self) -> bool:
        if self.receive_factory is None:
            try:
                import rtde_receive
            except ImportError as exc:
                self._receive_error = f"{type(exc).__name__}: {exc}"
                self._next_receive_connect_monotonic = time.monotonic() + 1.0
                return False
            self.receive_factory = rtde_receive.RTDEReceiveInterface
        try:
            self.receive = self.receive_factory(self.robot_ip)
        except (OSError, RuntimeError) as exc:
            self.receive = None
            self._receive_error = f"{type(exc).__name__}: {exc}"
            self._next_receive_connect_monotonic = time.monotonic() + 1.0
            return False
        self._receive_error = ""
        self._next_receive_connect_monotonic = 0.0
        return True

    def _connect_control_for_goal(self) -> str | None:
        if self.control is not None:
            is_connected = getattr(self.control, "isConnected", None)
            if is_connected is None:
                return None
            try:
                if bool(is_connected()):
                    return None
            except RuntimeError:
                pass
            disconnect = getattr(self.control, "disconnect", None)
            if disconnect is not None:
                with suppress(RuntimeError):
                    disconnect()
            self.control = None
        try:
            if self.control_factory is None:
                import rtde_control

                self.control_factory = rtde_control.RTDEControlInterface
            self.control = self.control_factory(self.robot_ip)
        except (ImportError, OSError, RuntimeError) as exc:
            self.control = None
            self._control_error = f"{type(exc).__name__}: {exc}"
            return (
                "UR5e RTDE control unavailable. Set the teach pendant to Remote Control "
                f"before commanding motion: {self._control_error}"
            )
        self._control_error = ""
        return None

    def _read_actual_q(self) -> list[float] | None:
        error = ""
        recovered = False
        with self._receive_lock:
            if self.receive is None:
                if time.monotonic() < self._next_receive_connect_monotonic:
                    return None
                if not self._reconnect_receive_locked():
                    error = self._receive_error
                else:
                    recovered = True
            if self.receive is not None:
                try:
                    values = [float(value) for value in list(self.receive.getActualQ())]
                except (OSError, RuntimeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    disconnect = getattr(self.receive, "disconnect", None)
                    if disconnect is not None:
                        with suppress(RuntimeError):
                            disconnect()
                    self.receive = None
                    self._receive_error = error
                    self._next_receive_connect_monotonic = time.monotonic() + 1.0
        if error:
            self._joint_status_announced = False
            status = _status_base()
            status.update(
                state="blocked",
                blocked_reason=f"RTDE getActualQ failed: {error}",
                message=f"blocked: RTDE getActualQ failed: {error}",
                rtde_connected=False,
                rtde_receive_connected=False,
                rtde_control_connected=self.control is not None,
            )
            self._write_status(status)
            return None
        if len(values) < len(ARM_JOINTS):
            return None
        self.current_positions = values[: len(ARM_JOINTS)]
        self.current_positions_monotonic = time.monotonic()
        if recovered or not self._joint_status_announced:
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

    def _current_position_map(self) -> dict[str, float] | None:
        actual = self._read_actual_q()
        if actual is None:
            return None
        return {joint: float(value) for joint, value in zip(ARM_JOINTS, actual)}

    def _joint_states_fresh(self) -> bool:
        if self.current_positions_monotonic is None:
            return False
        return time.monotonic() - self.current_positions_monotonic <= UR5E_RTDE_FEEDBACK_STALE_SEC

    def _publish_joint_state(self) -> None:
        actual = self._read_actual_q()
        if actual is None:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINTS)
        msg.position = [float(value) for value in actual]
        self._joint_state_pub.publish(msg)
        now = time.monotonic()
        if now < self._next_status_heartbeat_monotonic:
            return
        with self._active_lock:
            if self._active_goal is not None:
                return
        self._next_status_heartbeat_monotonic = now + 1.0
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
            blocked_reason="",
            rtde_connected=control_connected,
            rtde_receive_connected=True,
            rtde_control_connected=control_connected,
            joint_states_fresh=self._joint_states_fresh(),
        )
        self._write_status(status)

    def _cancel(self, _goal_handle: Any) -> CancelResponse:
        return CancelResponse.ACCEPT

    @staticmethod
    def _result(error_code: int, error_string: str) -> FollowJointTrajectory.Result:
        result = FollowJointTrajectory.Result()
        result.error_code = int(error_code)
        result.error_string = str(error_string or "")
        return result

    def _stop_motion(self) -> None:
        if self.control is None:
            return
        for method_name in ("stopJ", "stopScript", "servoStop"):
            method = getattr(self.control, method_name, None)
            if method is None:
                continue
            try:
                if method_name == "stopJ":
                    method(UR5E_RTDE_STOP_ACCEL_RAD_S2)
                else:
                    method()
                return
            except TypeError:
                try:
                    method()
                    return
                except Exception:
                    continue
            except Exception:
                continue

    def _clear_active_goal(self, goal_handle: Any) -> None:
        with self._active_lock:
            if self._active_goal is goal_handle:
                self._active_goal = None

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

    def _execute(  # noqa: PLR0915 - guarded hardware execution keeps one status lifecycle.
        self,
        goal_handle: Any,
    ) -> FollowJointTrajectory.Result:
        with self._active_lock:
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
            self._write_status(status)
            goal_handle.abort()
            self._clear_active_goal(goal_handle)
            return self._result(-1, control_error)
        status["rtde_connected"] = self.control is not None and self.receive is not None
        status["rtde_receive_connected"] = self.receive is not None
        status["rtde_control_connected"] = self.control is not None
        current_positions = self._current_position_map()
        status["joint_states_fresh"] = self._joint_states_fresh()
        if current_positions is None or not status["joint_states_fresh"]:
            reason = "UR5e RTDE feedback stale or missing"
            status.update(state="blocked", blocked_reason=reason, message=f"blocked: {reason}")
            self._write_status(status)
            goal_handle.abort()
            self._clear_active_goal(goal_handle)
            return self._result(-1, reason)

        ok, guarded_trajectory, status = prepare_rtde_trajectory(
            goal_handle.request.trajectory,
            current_positions,
        )
        status["rtde_connected"] = self.control is not None and self.receive is not None
        status["rtde_receive_connected"] = self.receive is not None
        status["rtde_control_connected"] = self.control is not None
        status["joint_states_fresh"] = self._joint_states_fresh()
        if not ok or guarded_trajectory is None:
            self._write_status(status)
            goal_handle.abort()
            self._clear_active_goal(goal_handle)
            return self._result(-1, str(status.get("blocked_reason") or "RTDE trajectory rejected"))

        path = rtde_movej_path(guarded_trajectory)
        status["movej_path_points"] = len(path)
        status.update(state="executing", message="executing UR5e RTDE moveJ path")
        self._write_status(status)
        final_q = [float(value) for value in path[-1][: len(ARM_JOINTS)]]
        final_time = max((_point_seconds(point) for point in list(guarded_trajectory.points)), default=0.0)
        deadline = time.monotonic() + max(5.0, final_time + UR5E_RTDE_RESULT_MARGIN_SEC)

        try:
            dispatch_started = time.monotonic()
            rtde_result, rtde_command_mode = self._execute_movej_path(path)
            status["rtde_result"] = rtde_result
            status["rtde_command_mode"] = rtde_command_mode
            status["rtde_async_dispatch_elapsed_sec"] = time.monotonic() - dispatch_started
            self._write_status(status)
            if str(rtde_result).strip() == "False":
                reason = "UR5e RTDE moveJ returned False"
                status.update(state="failed", message=reason, blocked_reason=reason)
                self._write_status(status)
                goal_handle.abort()
                self._clear_active_goal(goal_handle)
                return self._result(-4, reason)
            while rclpy.ok() and time.monotonic() < deadline:
                if goal_handle.is_cancel_requested:
                    self._stop_motion()
                    status.update(state="canceled", message="UR5e RTDE trajectory canceled")
                    self._write_status(status)
                    goal_handle.canceled()
                    self._clear_active_goal(goal_handle)
                    return self._result(-1, "canceled")
                actual = self._read_actual_q()
                if actual is not None:
                    max_delta, max_joint = _max_named_delta(ARM_JOINTS, final_q, dict(zip(ARM_JOINTS, actual)))
                    status["final_joint_error_rad"] = max_delta
                    status["final_joint_error_joint"] = max_joint
                    if max_delta <= UR5E_RTDE_GOAL_TOLERANCE_RAD:
                        status.update(state="succeeded", message="UR5e RTDE trajectory reached final joint target")
                        self._write_status(status)
                        goal_handle.succeed()
                        self._clear_active_goal(goal_handle)
                        return self._result(0, "")
                time.sleep(0.02)
            reason = "UR5e RTDE trajectory result timeout"
            self._stop_motion()
            status.update(state="failed", message=reason, blocked_reason=reason)
            self._write_status(status)
            goal_handle.abort()
            self._clear_active_goal(goal_handle)
            return self._result(-1, reason)
        except Exception as exc:
            reason = f"UR5e RTDE trajectory failed: {type(exc).__name__}: {exc}"
            self._stop_motion()
            disconnect = getattr(self.control, "disconnect", None)
            if disconnect is not None:
                with suppress(RuntimeError):
                    disconnect()
            self.control = None
            status["rtde_connected"] = False
            status["rtde_control_connected"] = False
            status["rtde_receive_connected"] = self.receive is not None
            status.update(state="failed", message=reason, blocked_reason=reason)
            self._write_status(status)
            goal_handle.abort()
            return self._result(-4, reason)
        finally:
            self._clear_active_goal(goal_handle)

    def destroy_node(self) -> bool:
        try:
            if self.receive is not None and hasattr(self.receive, "disconnect"):
                self.receive.disconnect()
        except Exception:
            pass
        try:
            if self.control is not None and hasattr(self.control, "disconnect"):
                self.control.disconnect()
        except Exception:
            pass
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
        if not failed:
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
