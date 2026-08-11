#!/usr/bin/env python3.10
"""Synched gazebo + hardware digital twin helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Any

import yaml

HARDWARE_ARMS_CONFIG_FILE = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "hardware_runtime"
    / "xarm6_ur5e_hardware_runtime.yaml"
)


def _load_hardware_arms_config() -> dict[str, Any]:
    with HARDWARE_ARMS_CONFIG_FILE.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return dict(loaded) if isinstance(loaded, dict) else {}


def _nested(config: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _list(config: dict[str, Any], keys: tuple[str, ...], default: list[Any]) -> list[Any]:
    value = _nested(config, keys, default)
    return list(value) if isinstance(value, list) else list(default)


def _str(config: dict[str, Any], keys: tuple[str, ...], default: str) -> str:
    value = str(_nested(config, keys, default) or "").strip()
    return value or str(default)


def _float(config: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    try:
        return float(_nested(config, keys, default))
    except (TypeError, ValueError):
        return float(default)


HARDWARE_ARMS_CONFIG = _load_hardware_arms_config()

ROBOTS: dict[str, dict[str, Any]] = {
    "xarm6": {
        "prefix": "xarm6_",
        "gazebo_joints": [
            "xarm6_joint1",
            "xarm6_joint2",
            "xarm6_joint3",
            "xarm6_joint4",
            "xarm6_joint5",
            "xarm6_joint6",
        ],
        "hardware_joint_candidates": [
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                "xarm6_joint1",
                "xarm6_joint2",
                "xarm6_joint3",
                "xarm6_joint4",
                "xarm6_joint5",
                "xarm6_joint6",
            ],
        ],
        "hardware_joint_state_topics": _list(
            HARDWARE_ARMS_CONFIG,
            ("xarm6", "joint_state_topics"),
            [
                "/joint_states",
                "/xarm/joint_states",
                "/xarm6/joint_states",
                "/xarm6/xarm/joint_states",
                "/xarm6/xarm_gripper/joint_states",
            ],
        ),
        "trajectory_topics": _list(
            HARDWARE_ARMS_CONFIG,
            ("xarm6", "trajectory_topics"),
            [
                "/xarm6/xarm6_traj_controller/joint_trajectory",
                "/xarm6_traj_controller/joint_trajectory",
                "/xarm_traj_controller/joint_trajectory",
                "/xarm6_xarm6_traj_controller/joint_trajectory",
            ],
        ),
        "hardware_trajectory_action": _str(
            HARDWARE_ARMS_CONFIG,
            ("xarm6", "hardware_trajectory_action"),
            "/xarm6/xarm6_traj_controller/follow_joint_trajectory",
        ),
        # Controllers spawned by the passive/mirror gazebo launch (prefix is applied
        # twice: launch prefix 'xarm6_' + controller name 'xarm6_traj_controller').
        "gazebo_trajectory_topics": _list(
            HARDWARE_ARMS_CONFIG,
            ("xarm6", "gazebo_trajectory_topics"),
            [
                "/xarm6_xarm6_traj_controller/joint_trajectory",
                "/xarm6_traj_controller/joint_trajectory",
            ],
        ),
        # Optional 1-DOF gripper mirror. The xarm gripper trajectory controller drives
        # the single 'drive_joint'; the finger joints follow via mimic. The hardware
        # position is resolved with the same prefix/endswith lookup as the arm joints.
        "gripper": {
            "gazebo_joint": "xarm6_drive_joint",
            "gazebo_trajectory_topics": _list(
                HARDWARE_ARMS_CONFIG,
                ("xarm6", "gripper", "gazebo_trajectory_topics"),
                ["/xarm6_xarm_gripper_traj_controller/joint_trajectory"],
            ),
            "hardware_service": _str(
                HARDWARE_ARMS_CONFIG,
                ("xarm6", "gripper", "hardware_service"),
                "set_gripper_position",
            ),
            "hardware_action": _str(
                HARDWARE_ARMS_CONFIG,
                ("xarm6", "gripper", "hardware_action"),
                "/xarm6/xarm_gripper/gripper_action",
            ),
            "open_position": _float(HARDWARE_ARMS_CONFIG, ("xarm6", "gripper", "open_position"), 0.0),
            "close_position": _float(HARDWARE_ARMS_CONFIG, ("xarm6", "gripper", "close_position"), 0.85),
            "open_pulse": _float(HARDWARE_ARMS_CONFIG, ("xarm6", "gripper", "open_pulse"), 850.0),
            "close_pulse": _float(HARDWARE_ARMS_CONFIG, ("xarm6", "gripper", "close_pulse"), 0.0),
        },
    },
    "ur5e": {
        "prefix": "ur5e_",
        "gazebo_joints": [
            "ur5e_shoulder_pan_joint",
            "ur5e_shoulder_lift_joint",
            "ur5e_elbow_joint",
            "ur5e_wrist_1_joint",
            "ur5e_wrist_2_joint",
            "ur5e_wrist_3_joint",
        ],
        "hardware_joint_candidates": [
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            [
                "ur5e_shoulder_pan_joint",
                "ur5e_shoulder_lift_joint",
                "ur5e_elbow_joint",
                "ur5e_wrist_1_joint",
                "ur5e_wrist_2_joint",
                "ur5e_wrist_3_joint",
            ],
        ],
        "hardware_joint_state_topics": _list(
            HARDWARE_ARMS_CONFIG,
            ("ur5e", "joint_state_topics"),
            ["/joint_states"],
        ),
        "trajectory_topics": _list(HARDWARE_ARMS_CONFIG, ("ur5e", "trajectory_topics"), []),
        "hardware_trajectory_action": _str(
            HARDWARE_ARMS_CONFIG,
            ("ur5e", "hardware_trajectory_action"),
            "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory",
        ),
        # Controller spawned by the passive/mirror gazebo launch.
        "gazebo_trajectory_topics": _list(
            HARDWARE_ARMS_CONFIG,
            ("ur5e", "gazebo_trajectory_topics"),
            ["/ur5e_joint_trajectory_controller/joint_trajectory"],
        ),
        "gripper": {
            "gazebo_joint": "ur5e_rg2_finger_width",
            "gazebo_trajectory_topics": _list(
                HARDWARE_ARMS_CONFIG,
                ("ur5e", "gripper", "gazebo_trajectory_topics"),
                ["/ur5e_rg2_gripper_traj_controller/joint_trajectory"],
            ),
            "hardware_action": _str(
                HARDWARE_ARMS_CONFIG,
                ("ur5e", "gripper", "action"),
                "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory",
            ),
        },
    },
}

_XARM6_RELAYED_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))

# Re-target period for streamed mirror trajectory points (seconds). Small enough to
# track hardware closely, large enough to give the JTC a smooth interpolation window.
MIRROR_POINT_TIME_SEC = _float(HARDWARE_ARMS_CONFIG, ("dual_robots", "mirror", "point_time_sec"), 0.1)
MIRROR_HARDWARE_HEARTBEAT_SEC = _float(
    HARDWARE_ARMS_CONFIG,
    ("dual_robots", "mirror", "hardware_heartbeat_sec"),
    1.0,
)
MIRROR_HARDWARE_STALE_SEC = _float(
    HARDWARE_ARMS_CONFIG,
    ("dual_robots", "mirror", "hardware_stale_sec"),
    3.0,
)
MIRROR_GAZEBO_JOINT_STATE_STALE_SEC = _float(
    HARDWARE_ARMS_CONFIG,
    ("dual_robots", "mirror", "gazebo_joint_state_stale_sec"),
    2.0,
)
MIRROR_GAZEBO_CONVERGENCE_TOLERANCE_RAD = _float(
    HARDWARE_ARMS_CONFIG,
    ("dual_robots", "mirror", "gazebo_convergence_tolerance_rad"),
    0.01,
)
MIRROR_GAZEBO_CONVERGENCE_TIMEOUT_SEC = _float(
    HARDWARE_ARMS_CONFIG,
    ("dual_robots", "mirror", "gazebo_convergence_timeout_sec"),
    2.0,
)
MIRROR_GAZEBO_REPUBLISH_PERIOD_SEC = _float(
    HARDWARE_ARMS_CONFIG,
    ("dual_robots", "mirror", "gazebo_republish_period_sec"),
    0.25,
)
MIRROR_MIN_PUBLISH_PERIOD_SEC = 0.0
MIRROR_MIN_JOINT_DELTA_RAD = 0.0
UR5E_MIRROR_POINT_TIME_SEC = _float(
    HARDWARE_ARMS_CONFIG,
    ("dual_robots", "mirror", "ur5e_point_time_sec"),
    0.12,
)
UR5E_MIRROR_MIN_PUBLISH_PERIOD_SEC = _float(
    HARDWARE_ARMS_CONFIG,
    ("dual_robots", "mirror", "ur5e_min_publish_period_sec"),
    0.05,
)
UR5E_MIRROR_MIN_JOINT_DELTA_RAD = _float(
    HARDWARE_ARMS_CONFIG,
    ("dual_robots", "mirror", "ur5e_min_joint_delta_rad"),
    0.0010,
)
NO_MATCHING_JOINT_STATE_REPORT_SEC = 5.0
HARDWARE_SNAPSHOT_TIMEOUT_SEC = 20.0

# Replay safety: the approach from the robot's current pose to the first recorded
# waypoint is time-scaled so no joint exceeds this speed (deg/s). This replaces a
# hard first-waypoint distance block, which made authored (far) poses un-replayable.
REPLAY_SPEED_SCALE = 1.0
MAX_REPLAY_JOINT_VEL_DEG_S = 25.0 * REPLAY_SPEED_SCALE
UR5E_REPLAY_MAX_JOINT_VEL_DEG_S = 10.0 * REPLAY_SPEED_SCALE
MOVE_GROUP_REPLAY_VELOCITY_SCALING = 0.25 * REPLAY_SPEED_SCALE
MOVE_GROUP_REPLAY_ACCELERATION_SCALING = 0.25
DEFAULT_REPLAY_WAYPOINT_DURATION_SEC = 2.0 / REPLAY_SPEED_SCALE
INITIALIZE_GAZEBO_TOLERANCE_RAD = 0.02
INITIALIZE_GAZEBO_ATTEMPTS = 5
HARDWARE_TRAJECTORY_START_DELAY_SEC = 0.2
PAIRED_HARDWARE_TRAJECTORY_START_DELAY_SEC = 1.0
HARDWARE_TRAJECTORY_CURRENT_POINT_SEC = 0.0
UR5E_HARDWARE_TRAJECTORY_CURRENT_POINT_SEC = 0.25
TEACH_REPLAY_OBSERVED_COMPLETION_TOLERANCE_RAD = 0.02
TEACH_REPLAY_OBSERVED_COMPLETION_TIMEOUT_SEC = 3.0
TEACH_REPLAY_PREPARED_START_DRIFT_TOLERANCE_RAD = 0.05
TEACH_REPLAY_PREPARED_START_DRIFT_TIMEOUT_SEC = 3.0
XARM6_TEACH_REPLAY_FINAL_HOLD_SEC = 0.5
XARM6_TEACH_REPLAY_GOAL_TIME_TOLERANCE_SEC = 2.0
XARM6_TEACH_REPLAY_RESULT_TIMEOUT_MARGIN_SEC = 12.0
MOVE_GROUP_ACTION_NAME = "/move_action"
UR5E_HARDWARE_MOVE_GROUP = "ur_manipulator"
MOVE_GROUP_JOINT_TOLERANCE_RAD = 0.001
MOVE_GROUP_ALLOWED_PLANNING_TIME_SEC = 5.0
MOVE_GROUP_PLAN_TIMEOUT_SEC = 15.0
MOVE_GROUP_GOAL_ACCEPTANCE_TIMEOUT_SEC = 20.0
MOVE_GROUP_GOAL_ACCEPTANCE_RETRY_COUNT = 1
MOVE_GROUP_GOAL_ACCEPTANCE_RETRY_DELAY_SEC = 0.5
UR5E_TEACH_REPLAY_TIME_SCALE = 1.5
UR5E_TEACH_REPLAY_FINAL_HOLD_SEC = 0.5
UR5E_TEACH_REPLAY_MIN_POINT_STEP_SEC = 0.02
UR5E_TEACH_REPLAY_RESULT_TIMEOUT_MARGIN_SEC = 20.0
UR5E_TEACH_REPLAY_GOAL_TIME_TOLERANCE_SEC = 2.0
UR5E_FINAL_ERROR_SNAPSHOT_TIMEOUT_SEC = 2.0
PREPARED_REPLAY_VERSION = 7


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.setdefault("updated_at", time.time())
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _stable_json_hash(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _recording_hash(recording: dict[str, Any]) -> str:
    return _stable_json_hash(recording)


def _prepared_replay_metadata(args: argparse.Namespace, recording: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": PREPARED_REPLAY_VERSION,
        "recording_hash": _recording_hash(recording),
        "replay_target": str(args.replay_target or "hardware"),
        "gazebo_domain_id": int(args.gazebo_domain_id),
        "hardware_domain_id": int(args.hardware_domain_id),
        "waypoint_count": len(list(recording.get("waypoints") or [])),
        "robot": str(recording.get("robot") or ""),
        "recording_type": str(recording.get("recording_type") or ""),
    }


def _prepared_replay_validation_error(
    args: argparse.Namespace,
    recording: dict[str, Any],
    prepared: dict[str, Any],
) -> str:
    metadata = dict(prepared.get("metadata") or {})
    expected = _prepared_replay_metadata(args, recording)
    for key, value in expected.items():
        if metadata.get(key) != value:
            return f"prepared replay stale: {key} changed."
    if not isinstance(prepared.get("plans"), dict):
        return "prepared replay stale: plans missing."
    if set(dict(prepared.get("plans") or {}).keys()) != {"xarm6", "ur5e"}:
        return "prepared replay stale: paired plans missing."
    if str(args.replay_target or "hardware") in ("hardware", "both"):
        ur5e_plan = dict(dict(prepared.get("plans") or {}).get("ur5e") or {})
        if not ur5e_plan.get("hardware_points"):
            return "prepared replay stale: ur5e hardware trajectory missing."
    return ""


def _prepared_start_drift_error(
    args: argparse.Namespace,
    prepared: dict[str, Any],
) -> str:
    plans = dict(prepared.get("plans") or {})
    for robot, plan_raw in plans.items():
        robot_key = str(robot)
        plan = dict(plan_raw or {})
        joint_names = [str(name) for name in list(plan.get("hardware_names") or [])]
        start_positions = [float(value) for value in list(plan.get("hardware_positions") or [])]
        if not joint_names or not start_positions:
            continue
        snapshot_result = _read_snapshot(
            int(args.hardware_domain_id),
            robot_key,
            "hardware",
            TEACH_REPLAY_PREPARED_START_DRIFT_TIMEOUT_SEC,
        )
        if not snapshot_result.get("success"):
            return f"prepared replay stale: {robot_key} hardware start pose unavailable: {snapshot_result.get('message') or 'snapshot failed'}"
        snapshot = dict(snapshot_result.get("snapshot") or {})
        observed_positions, missing = _positions_for_joint_names(snapshot, robot_key, joint_names)
        if missing:
            return f"prepared replay stale: {robot_key} hardware start pose missing joints: {', '.join(missing)}"
        if len(observed_positions) != len(start_positions):
            return f"prepared replay stale: {robot_key} hardware start pose joint count changed."
        max_delta_rad = max(
            (_angular_delta(actual, expected) for actual, expected in zip(observed_positions, start_positions)),
            default=0.0,
        )
        if max_delta_rad > TEACH_REPLAY_PREPARED_START_DRIFT_TOLERANCE_RAD:
            return (
                f"prepared replay stale: {robot_key} hardware moved "
                f"{math.degrees(max_delta_rad):.3f} deg from prepared start pose "
                f"(ceiling {math.degrees(TEACH_REPLAY_PREPARED_START_DRIFT_TOLERANCE_RAD):.3f} deg)."
            )
    return ""


def _refresh_prepared_hardware_start_from_snapshot(
    args: argparse.Namespace,
    prepared: dict[str, Any],
) -> str:
    plans = dict(prepared.get("plans") or {})
    for robot, plan_raw in plans.items():
        robot_key = str(robot)
        plan = dict(plan_raw or {})
        joint_names = [str(name) for name in list(plan.get("hardware_names") or [])]
        if not joint_names:
            continue
        snapshot_result = _read_snapshot(
            int(args.hardware_domain_id),
            robot_key,
            "hardware",
            TEACH_REPLAY_PREPARED_START_DRIFT_TIMEOUT_SEC,
        )
        if not snapshot_result.get("success"):
            return f"prepared replay stale: {robot_key} hardware start pose unavailable: {snapshot_result.get('message') or 'snapshot failed'}"
        snapshot = dict(snapshot_result.get("snapshot") or {})
        observed_positions, missing = _positions_for_joint_names(snapshot, robot_key, joint_names)
        if missing:
            return f"prepared replay stale: {robot_key} hardware start pose missing joints: {', '.join(missing)}"
        hardware_points = [dict(point) for point in list(plan.get("hardware_points") or [])]
        if hardware_points:
            first_point = dict(hardware_points[0])
            first_point["positions"] = [float(value) for value in observed_positions]
            hardware_points[0] = first_point
            plan["hardware_points"] = hardware_points
        plan["hardware_positions"] = [float(value) for value in observed_positions]
        plans[robot_key] = plan
    prepared["plans"] = plans
    return ""


def _write_status(path: Path, **payload: Any) -> None:
    _atomic_json_write(path, dict(payload))


def _write_replay_status(args: argparse.Namespace, *, state: str, message: str, last_error: str = "") -> None:
    status_file = str(getattr(args, "status_file", "") or "").strip()
    if not status_file:
        return
    direction_file = Path(str(getattr(args, "direction_file", "") or ""))
    direction = _direction(direction_file) if str(direction_file) else "gazebo -> hardware"
    _write_status(
        Path(status_file),
        target=str(getattr(args, "target", "") or ""),
        state=state,
        direction=direction,
        message=message,
        last_error=last_error,
    )


def _direction(direction_file: Path) -> str:
    value = str(_read_json(direction_file).get("direction") or "hardware -> gazebo").strip()
    if value not in {"hardware -> gazebo", "gazebo -> hardware"}:
        return "hardware -> gazebo"
    return value


def _strip_prefix(robot: str, joint_name: str) -> str:
    prefix = str(ROBOTS[robot]["prefix"])
    name = str(joint_name)
    return name[len(prefix):] if name.startswith(prefix) else name


def _joint_lookup(snapshot: dict[str, float], robot: str, gazebo_joint: str) -> tuple[str | None, float | None]:
    candidates = [gazebo_joint, _strip_prefix(robot, gazebo_joint)]
    for candidate in candidates:
        if candidate in snapshot:
            return candidate, float(snapshot[candidate])
    suffix = _strip_prefix(robot, gazebo_joint)
    for name, value in snapshot.items():
        if str(name).endswith(suffix):
            return str(name), float(value)
    return None, None


def _resolve_gazebo_positions(
    snapshot: dict[str, float],
    robot: str,
) -> tuple[list[str], list[float], list[str]]:
    gazebo_joints = list(ROBOTS[robot]["gazebo_joints"])
    positions: list[float] = []
    missing: list[str] = []
    for gazebo_joint in gazebo_joints:
        _source_name, value = _joint_lookup(snapshot, robot, gazebo_joint)
        if value is None:
            missing.append(gazebo_joint)
        else:
            positions.append(float(value))
    if missing:
        return gazebo_joints, [], missing
    return gazebo_joints, positions, []


def _gazebo_joint_match_count(snapshot: dict[str, float], robot: str) -> int:
    count = 0
    for gazebo_joint in ROBOTS[robot]["gazebo_joints"]:
        _source_name, value = _joint_lookup(snapshot, robot, gazebo_joint)
        if value is not None:
            count += 1
    return count


def _hardware_joint_match_count(snapshot: dict[str, float], robot: str) -> int:
    best_count = 0
    for candidate_group in ROBOTS[robot]["hardware_joint_candidates"]:
        best_count = max(best_count, sum(1 for name in candidate_group if name in snapshot))
    best_count = max(best_count, _gazebo_joint_match_count(snapshot, robot))
    return best_count


def _joint_match_count(snapshot: dict[str, float], robot: str, source: str) -> int:
    if source == "gazebo":
        return _gazebo_joint_match_count(snapshot, robot)
    return _hardware_joint_match_count(snapshot, robot)


def _joint_state_topics(robot: str, source: str) -> list[str]:
    if source == "hardware":
        topics = ROBOTS[robot].get("hardware_joint_state_topics") or ["/joint_states"]
        return [str(topic) for topic in topics if str(topic or "").strip()]
    return ["/joint_states"]


def _mirror_point_time_sec(robot: str) -> float:
    return (
        UR5E_MIRROR_POINT_TIME_SEC
        if str(robot or "").strip().lower() == "ur5e"
        else MIRROR_POINT_TIME_SEC
    )


def _mirror_min_publish_period_sec(robot: str) -> float:
    return (
        UR5E_MIRROR_MIN_PUBLISH_PERIOD_SEC
        if str(robot or "").strip().lower() == "ur5e"
        else MIRROR_MIN_PUBLISH_PERIOD_SEC
    )


def _mirror_min_joint_delta_rad(robot: str) -> float:
    return (
        UR5E_MIRROR_MIN_JOINT_DELTA_RAD
        if str(robot or "").strip().lower() == "ur5e"
        else MIRROR_MIN_JOINT_DELTA_RAD
    )


def _max_position_delta_rad(
    previous_positions: list[float] | None,
    positions: list[float],
) -> float:
    if not previous_positions:
        return math.inf
    deltas = [
        abs(float(current) - float(previous))
        for previous, current in zip(previous_positions, positions)
    ]
    return max(deltas) if deltas else 0.0


def _should_publish_mirror_update(
    robot: str,
    *,
    positions: list[float],
    last_positions: list[float] | None,
    last_publish_ts: float,
    now: float,
) -> tuple[bool, str]:
    max_delta = _max_position_delta_rad(last_positions, positions)
    min_delta = max(0.0, _mirror_min_joint_delta_rad(robot))
    if max_delta < min_delta:
        return False, "below_delta"

    min_period = max(0.0, _mirror_min_publish_period_sec(robot))
    if last_publish_ts > 0.0 and now - float(last_publish_ts) < min_period:
        return False, "rate_limited"

    return True, ""


def _should_enqueue_hardware_update(
    robot: str,
    *,
    positions: list[float],
    last_positions: list[float] | None,
    gripper_position: float | None,
    last_gripper_position: float | None,
    last_enqueue_ts: float,
    now: float,
) -> bool:
    min_period = max(0.0, _mirror_min_publish_period_sec(robot))
    if last_enqueue_ts > 0.0 and now - float(last_enqueue_ts) < min_period:
        return False

    heartbeat_sec = max(0.0, MIRROR_HARDWARE_HEARTBEAT_SEC)
    if (
        last_enqueue_ts > 0.0
        and heartbeat_sec > 0.0
        and now - float(last_enqueue_ts) >= heartbeat_sec
    ):
        return True

    max_joint_delta = _max_position_delta_rad(last_positions, positions)
    if gripper_position is None and last_gripper_position is None:
        gripper_delta = 0.0
    elif gripper_position is None or last_gripper_position is None:
        gripper_delta = math.inf
    else:
        gripper_delta = abs(float(gripper_position) - float(last_gripper_position))
    min_delta = max(0.0, _mirror_min_joint_delta_rad(robot))
    return max_joint_delta >= min_delta or gripper_delta >= min_delta


def _hardware_mirror_status_without_update(
    *,
    last_hardware_update_ts: float,
    last_published_positions: list[float] | None,
    hardware_diagnostic_message: str,
    now: float,
) -> tuple[str, str, str, float | None]:
    if hardware_diagnostic_message:
        return "waiting", hardware_diagnostic_message, hardware_diagnostic_message, None
    if last_hardware_update_ts <= 0.0 or last_published_positions is None:
        return "waiting", "waiting for hardware /joint_states.", "", None

    hardware_joint_state_age_sec = max(0.0, now - float(last_hardware_update_ts))
    if hardware_joint_state_age_sec <= max(0.0, MIRROR_HARDWARE_STALE_SEC):
        return (
            "mirroring",
            "hardware -> gazebo active; hardware pose unchanged; "
            f"hardware_joint_state_age_sec={hardware_joint_state_age_sec:.2f}.",
            "",
            hardware_joint_state_age_sec,
        )

    message = (
        "hardware /joint_states is stale; "
        f"hardware_joint_state_age_sec={hardware_joint_state_age_sec:.2f}; "
        f"hardware_stale_sec={max(0.0, MIRROR_HARDWARE_STALE_SEC):.2f}."
    )
    return "waiting", message, message, hardware_joint_state_age_sec


def _gazebo_target_status(
    robot: str,
    *,
    target_positions: list[float] | None,
    gazebo_joint_snapshot: dict[str, float],
    gazebo_joint_state_received_at: float,
    target_changed_at: float,
    now: float,
) -> dict[str, Any]:
    """Report whether Gazebo has actually reached the hardware joint target."""
    status: dict[str, Any] = {
        "gazebo_converged": False,
        "gazebo_joint_state_age_sec": None,
        "gazebo_max_joint_error_rad": None,
        "gazebo_max_joint_error_joint": "",
        "gazebo_convergence_tolerance_rad": max(
            0.0,
            MIRROR_GAZEBO_CONVERGENCE_TOLERANCE_RAD,
        ),
        "gazebo_convergence_timeout_sec": max(
            0.0,
            MIRROR_GAZEBO_CONVERGENCE_TIMEOUT_SEC,
        ),
        "gazebo_republish_period_sec": max(
            0.0,
            MIRROR_GAZEBO_REPUBLISH_PERIOD_SEC,
        ),
    }
    if target_positions is None:
        status.update(
            state="waiting",
            message="waiting for the first hardware joint target.",
            last_error="",
        )
        return status

    if gazebo_joint_state_received_at <= 0.0:
        message = "waiting for Gazebo /joint_states before confirming hardware -> gazebo mirror."
        status.update(state="waiting", message=message, last_error=message)
        return status

    gazebo_joint_state_age_sec = max(0.0, now - gazebo_joint_state_received_at)
    status["gazebo_joint_state_age_sec"] = gazebo_joint_state_age_sec
    if gazebo_joint_state_age_sec > max(0.0, MIRROR_GAZEBO_JOINT_STATE_STALE_SEC):
        message = (
            "Gazebo /joint_states is stale; "
            f"gazebo_joint_state_age_sec={gazebo_joint_state_age_sec:.2f}; "
            "hardware -> gazebo convergence cannot be confirmed."
        )
        status.update(state="waiting", message=message, last_error=message)
        return status

    gazebo_joint_names, gazebo_positions, missing = _resolve_gazebo_positions(
        gazebo_joint_snapshot,
        robot,
    )
    if missing:
        message = "Gazebo /joint_states missing required joints: " + ", ".join(missing)
        status.update(state="waiting", message=message, last_error=message)
        return status

    errors = [
        _angular_delta(observed, target)
        for observed, target in zip(gazebo_positions, target_positions, strict=True)
    ]
    max_error = max(errors, default=0.0)
    max_error_index = errors.index(max_error) if errors else 0
    max_error_joint = gazebo_joint_names[max_error_index] if gazebo_joint_names else ""
    status["gazebo_max_joint_error_rad"] = max_error
    status["gazebo_max_joint_error_joint"] = max_error_joint
    if max_error <= max(0.0, MIRROR_GAZEBO_CONVERGENCE_TOLERANCE_RAD):
        status.update(
            state="mirroring",
            message=(
                "hardware -> gazebo active; Gazebo reached the hardware joint target; "
                f"gazebo_max_joint_error_rad={max_error:.4f}."
            ),
            last_error="",
            gazebo_converged=True,
        )
        return status

    convergence_elapsed_sec = max(0.0, now - target_changed_at)
    status["gazebo_convergence_elapsed_sec"] = convergence_elapsed_sec
    if convergence_elapsed_sec <= max(0.0, MIRROR_GAZEBO_CONVERGENCE_TIMEOUT_SEC):
        status.update(
            state="mirroring",
            message=(
                "hardware -> gazebo active; Gazebo is following the hardware joint target; "
                f"gazebo_max_joint_error_rad={max_error:.4f}."
            ),
            last_error="",
        )
        return status

    message = (
        "Gazebo did not reach the hardware joint target; "
        f"gazebo_max_joint_error_rad={max_error:.4f}; "
        f"gazebo_max_joint_error_joint={max_error_joint}."
    )
    status.update(state="waiting", message=message, last_error=message)
    return status


def _resolve_gripper(snapshot: dict[str, float], robot: str) -> tuple[str | None, float | None]:
    """Resolve the gazebo gripper joint name and its hardware position, if configured."""
    gripper = ROBOTS[robot].get("gripper")
    if not gripper:
        return None, None
    gazebo_joint = str(gripper.get("gazebo_joint") or "").strip()
    if not gazebo_joint:
        return None, None
    _source_name, value = _joint_lookup(snapshot, robot, gazebo_joint)
    if value is None:
        return None, None
    return gazebo_joint, float(value)


def _resolve_hardware_positions(
    snapshot: dict[str, float],
    robot: str,
) -> tuple[list[str], list[float], list[str]]:
    for candidate_group in ROBOTS[robot]["hardware_joint_candidates"]:
        if all(name in snapshot for name in candidate_group):
            return list(candidate_group), [float(snapshot[name]) for name in candidate_group], []

    joint_names: list[str] = []
    positions: list[float] = []
    missing: list[str] = []
    for gazebo_joint in ROBOTS[robot]["gazebo_joints"]:
        source_name, value = _joint_lookup(snapshot, robot, gazebo_joint)
        if source_name is None or value is None:
            missing.append(_strip_prefix(robot, gazebo_joint))
        else:
            joint_names.append(source_name)
            positions.append(float(value))
    return joint_names, positions, missing


def _angular_delta(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def _positions_for_joint_names(
    snapshot: dict[str, float],
    robot: str,
    joint_names: list[str],
) -> tuple[list[float], list[str]]:
    positions: list[float] = []
    missing: list[str] = []
    for joint_name in joint_names:
        name = str(joint_name)
        if name in snapshot:
            positions.append(float(snapshot[name]))
        else:
            missing.append(name)
    if not missing:
        return positions, []

    resolved_names, resolved_positions, resolved_missing = _resolve_hardware_positions(
        snapshot,
        robot,
    )
    if resolved_missing:
        return [], missing
    normalized_resolved = [_strip_prefix(robot, name) for name in resolved_names]
    normalized_requested = [_strip_prefix(robot, name) for name in joint_names]
    if normalized_resolved == normalized_requested:
        return list(resolved_positions), []
    return [], missing


def _observed_completion_from_snapshot(
    snapshot: dict[str, float],
    robot: str,
    joint_names: list[str],
    target_positions: list[float],
    tolerance_rad: float,
) -> dict[str, Any]:
    positions, missing = _positions_for_joint_names(snapshot, robot, joint_names)
    if missing:
        return {
            "success": False,
            "message": f"missing observed hardware joints: {', '.join(missing)}",
            "missing": missing,
            "seen_names": sorted(snapshot),
        }
    if len(positions) != len(target_positions):
        return {
            "success": False,
            "message": "observed hardware joint count does not match target.",
        }
    deltas = [
        _angular_delta(actual, target)
        for actual, target in zip(positions, target_positions)
    ]
    max_delta_rad = max(deltas) if deltas else 0.0
    max_delta_deg = math.degrees(max_delta_rad)
    success = max_delta_rad <= tolerance_rad
    return {
        "success": success,
        "message": (
            f"target reached by hardware /joint_states; final_joint_error_deg={max_delta_deg:.3f}"
            if success
            else (
                "target not reached by hardware /joint_states; "
                f"final_joint_error_deg={max_delta_deg:.3f}; "
                f"tolerance_deg={math.degrees(tolerance_rad):.3f}"
            )
        ),
        "final_joint_error_rad": max_delta_rad,
        "final_joint_error_deg": max_delta_deg,
        "tolerance_rad": tolerance_rad,
        "tolerance_deg": math.degrees(tolerance_rad),
    }


def _observed_completion_worker(
    domain_id: int,
    robot: str,
    joint_names: list[str],
    target_positions: list[float],
    tolerance_rad: float,
    timeout_sec: float,
    result_queue: mp.Queue,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    class ObservedCompletionNode(Node):
        def __init__(self) -> None:
            super().__init__(f"digital_twin_{robot}_observed_completion")
            self.result: dict[str, Any] | None = None
            self.best_result: dict[str, Any] | None = None
            self._subs = [
                self.create_subscription(JointState, topic, self._on_joint_state, 10)
                for topic in _joint_state_topics(robot, "hardware")
            ]

        def _on_joint_state(self, msg: Any) -> None:
            snapshot = _snapshot_from_joint_state_msg(msg)
            result = _observed_completion_from_snapshot(
                snapshot,
                robot,
                joint_names,
                target_positions,
                tolerance_rad,
            )
            if "final_joint_error_rad" in result:
                if (
                    self.best_result is None
                    or float(result["final_joint_error_rad"])
                    < float(self.best_result.get("final_joint_error_rad") or math.inf)
                ):
                    self.best_result = dict(result)
            if result.get("success"):
                self.result = dict(result)

    node = None
    try:
        node = ObservedCompletionNode()
        deadline = time.time() + max(0.1, float(timeout_sec))
        while rclpy.ok() and time.time() < deadline and node.result is None:
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.result is not None:
            result_queue.put(node.result)
            return
        if node.best_result is not None:
            result_queue.put(node.best_result)
            return
        result_queue.put(
            {
                "success": False,
                "message": "observed hardware completion timed out before matching joints arrived.",
            }
        )
    except Exception as exc:
        result_queue.put({"success": False, "message": f"observed hardware completion failed: {exc}"})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _wait_observed_completion(
    domain_id: int,
    robot: str,
    joint_names: list[str],
    target_positions: list[float],
    *,
    tolerance_rad: float = TEACH_REPLAY_OBSERVED_COMPLETION_TOLERANCE_RAD,
    timeout_sec: float = TEACH_REPLAY_OBSERVED_COMPLETION_TIMEOUT_SEC,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_observed_completion_worker,
        args=(
            domain_id,
            robot,
            list(joint_names),
            [float(value) for value in target_positions],
            float(tolerance_rad),
            float(timeout_sec),
            result_queue,
        ),
        daemon=True,
    )
    proc.start()
    proc.join(max(0.5, float(timeout_sec)) + 1.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {"success": False, "message": "observed hardware completion worker timed out."}
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {"success": False, "message": "observed hardware completion returned no data."}


def _apply_observed_completion_fallback(
    result: dict[str, Any],
    *,
    domain_id: int,
    robot: str,
    joint_names: list[str],
    target_positions: list[float],
) -> dict[str, Any]:
    action_result = dict(result)
    wrapped = dict(result)
    wrapped["action_result"] = action_result
    if result.get("success"):
        wrapped["observed_completion"] = {
            "success": True,
            "message": "action succeeded; observed completion fallback not required.",
        }
        return wrapped
    if result.get("skip_observed_completion"):
        wrapped["observed_completion"] = {
            "success": False,
            "message": "skipped because action did not enter execution.",
        }
        return wrapped

    observed = _wait_observed_completion(
        domain_id,
        robot,
        joint_names,
        target_positions,
    )
    wrapped["observed_completion"] = observed
    observed_message = str(observed.get("message") or "")
    wrapped["message"] = (
        f"{str(result.get('message') or '').rstrip('. ')}; "
        f"observed_completion: {observed_message}."
    )
    if observed.get("success"):
        wrapped["success"] = True
    return wrapped


def _init_ros_domain(domain_id: int):
    os.environ["ROS_DOMAIN_ID"] = str(int(domain_id))
    import rclpy

    rclpy.init()
    return rclpy


def _snapshot_from_joint_state_msg(msg: Any) -> dict[str, float]:
    return {str(name): float(pos) for name, pos in zip(msg.name, msg.position)}


def _pose_from_transform_msg(msg: Any) -> dict[str, Any]:
    """Return a serializable pose from a geometry transform message."""
    transform = msg.transform
    translation = transform.translation
    rotation = transform.rotation
    return {
        "frame_id": str(getattr(msg.header, "frame_id", "") or "world"),
        "child_frame_id": str(getattr(msg, "child_frame_id", "") or "tool0"),
        "x": float(translation.x),
        "y": float(translation.y),
        "z": float(translation.z),
        "qx": float(rotation.x),
        "qy": float(rotation.y),
        "qz": float(rotation.z),
        "qw": float(rotation.w),
    }


def _hardware_update_from_snapshot(
    snapshot: dict[str, float],
    robot: str,
    *,
    remembered_gripper_joint: str | None = None,
    remembered_gripper_position: float | None = None,
) -> tuple[dict[str, Any], str | None, float | None]:
    gripper_joint, gripper_position = _resolve_gripper(snapshot, robot)
    if gripper_joint is None:
        gripper_joint = remembered_gripper_joint
        gripper_position = remembered_gripper_position

    _hardware_names, positions, missing = _resolve_hardware_positions(snapshot, robot)
    if missing:
        matched_count = _hardware_joint_match_count(snapshot, robot)
        if matched_count == 0:
            return (
                {
                    "diagnostic": "no_matching_hardware_joints",
                    "seen_names": sorted(snapshot),
                    "source_stamp": time.time(),
                },
                gripper_joint,
                gripper_position,
            )
        return (
            {
                "diagnostic": "missing_hardware_joints",
                "missing": list(missing),
                "seen_names": sorted(snapshot),
                "source_stamp": time.time(),
            },
            gripper_joint,
            gripper_position,
        )

    item: dict[str, Any] = {
        "joint_names": list(ROBOTS[robot]["gazebo_joints"]),
        "positions": positions,
        "source_stamp": time.time(),
    }
    if gripper_joint is not None and gripper_position is not None:
        item["gripper_joint"] = gripper_joint
        item["gripper_position"] = float(gripper_position)
    return item, gripper_joint, gripper_position


def _joint_state_snapshot_worker(  # noqa: C901 - joint and optional TF diagnostics share one node.
    domain_id: int,
    robot: str,
    source: str,
    timeout_sec: float,
    result_queue: mp.Queue,
    include_world_tool_pose: bool = False,
) -> None:
    rclpy = None
    node = None
    try:
        rclpy = _init_ros_domain(domain_id)
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        if include_world_tool_pose:
            from rclpy.duration import Duration
            from rclpy.time import Time
            from tf2_ros import Buffer, TransformException, TransformListener

        class SnapshotNode(Node):
            def __init__(self) -> None:
                super().__init__(f"digital_twin_{source}_snapshot")
                self.snapshot: dict[str, float] | None = None
                self.accumulated_snapshot: dict[str, float] = {}
                self.missing: list[str] = []
                self.seen_names: list[str] = []
                self.unmatched_seen_names: list[str] = []
                self.pose: dict[str, Any] | None = None
                self.pose_error = ""
                self.pose_child_frame = "link_eef" if robot == "xarm6" else "tool0"
                self.world_base_pose: dict[str, Any] | None = None
                self.world_base_pose_error = ""
                self.world_base_child_frame = (
                    "link_base" if robot == "xarm6" else "base_link"
                )
                self.tf_buffer = Buffer() if include_world_tool_pose else None
                self.tf_listener = (
                    TransformListener(self.tf_buffer, self, spin_thread=False)
                    if self.tf_buffer is not None
                    else None
                )
                for topic in _joint_state_topics(robot, source):
                    self.create_subscription(JointState, topic, self._cb, 10)

            def _cb(self, msg: JointState) -> None:
                snapshot = _snapshot_from_joint_state_msg(msg)
                matched_count = _joint_match_count(snapshot, robot, source)
                if matched_count == 0:
                    seen = set(self.unmatched_seen_names)
                    seen.update(snapshot)
                    self.unmatched_seen_names = sorted(seen)
                    return

                self.accumulated_snapshot.update(snapshot)
                if source == "gazebo":
                    _names, _positions, missing = _resolve_gazebo_positions(
                        self.accumulated_snapshot,
                        robot,
                    )
                else:
                    _names, _positions, missing = _resolve_hardware_positions(
                        self.accumulated_snapshot,
                        robot,
                    )
                if not missing:
                    self.snapshot = dict(self.accumulated_snapshot)
                    return

                self.missing = list(missing)
                self.seen_names = sorted(self.accumulated_snapshot)

            def read_world_tool_pose(self) -> None:
                if self.tf_buffer is None:
                    return
                if self.pose is None:
                    try:
                        message = self.tf_buffer.lookup_transform(
                            "world",
                            self.pose_child_frame,
                            Time(),
                            timeout=Duration(seconds=0.1),
                        )
                    except TransformException as exc:
                        self.pose_error = str(exc)
                    else:
                        self.pose = _pose_from_transform_msg(message)
                if self.world_base_pose is None:
                    try:
                        message = self.tf_buffer.lookup_transform(
                            "world",
                            self.world_base_child_frame,
                            Time(),
                            timeout=Duration(seconds=0.1),
                        )
                    except TransformException as exc:
                        self.world_base_pose_error = str(exc)
                    else:
                        self.world_base_pose = _pose_from_transform_msg(message)

        node = SnapshotNode()
        deadline = time.time() + max(0.5, float(timeout_sec))
        while rclpy.ok() and time.time() < deadline and (
            node.snapshot is None
            or (
                include_world_tool_pose
                and (node.pose is None or node.world_base_pose is None)
            )
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
            if include_world_tool_pose:
                node.read_world_tool_pose()
        if node.snapshot is None:
            missing = getattr(node, "missing", [])
            seen_names = getattr(node, "seen_names", [])
            if missing:
                message = (
                    f"{source} /joint_states missing required joints."
                    f" Missing {source} joints for {robot}: {', '.join(missing)}."
                    f" Seen joints: {', '.join(seen_names)}."
                )
            elif getattr(node, "unmatched_seen_names", []):
                message = (
                    f"{source} /joint_states has no {robot} arm joints yet."
                    f" Seen joints: {', '.join(node.unmatched_seen_names)}."
                )
            else:
                message = f"{source} /joint_states has no {robot} arm joints yet."
            result_queue.put(
                {
                    "success": False,
                    "message": message,
                }
            )
        elif include_world_tool_pose and node.pose is None:
            detail = f": {node.pose_error}" if node.pose_error else ""
            result_queue.put(
                {
                    "success": False,
                    "message": (
                        f"TF world -> {node.pose_child_frame} is unavailable{detail}"
                    ),
                    "world_tool0_ready": False,
                }
            )
        elif include_world_tool_pose and node.world_base_pose is None:
            detail = (
                f": {node.world_base_pose_error}"
                if node.world_base_pose_error
                else ""
            )
            result_queue.put(
                {
                    "success": False,
                    "message": (
                        f"TF world -> {node.world_base_child_frame} is unavailable"
                        f"{detail}"
                    ),
                    "world_tool0_ready": False,
                }
            )
        else:
            result: dict[str, Any] = {
                "success": True,
                "snapshot": node.snapshot,
                "received_at": time.time(),
            }
            if include_world_tool_pose:
                result["pose"] = dict(node.pose or {})
                result["world_base_pose"] = dict(node.world_base_pose or {})
                result["world_tool0_ready"] = True
            result_queue.put(result)
    except Exception as exc:
        topics = ", ".join(_joint_state_topics(robot, source))
        result_queue.put(
            {
                "success": False,
                "message": (
                    f"{source} /joint_states snapshot failed for {robot}: {exc}; "
                    f"ROS_DOMAIN_ID={int(domain_id)}; topics={topics}"
                ),
            }
        )
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy is not None:
            rclpy.shutdown()


def _xarm6_relayed_positions(
    names: list[str],
    positions: list[float],
) -> tuple[list[float] | None, str]:
    """Validate one root /joint_states message for the physical xArm6 chain."""
    observed = {
        str(name): positions[index]
        for index, name in enumerate(names)
        if index < len(positions)
    }
    missing = [name for name in _XARM6_RELAYED_JOINT_NAMES if name not in observed]
    if missing:
        return None, (
            "/joint_states did not provide all six xArm6 joints; "
            f"missing={missing}; seen={sorted(observed)}"
        )
    resolved: list[float] = []
    non_finite: list[str] = []
    for name in _XARM6_RELAYED_JOINT_NAMES:
        try:
            value = float(observed[name])
        except (TypeError, ValueError):
            non_finite.append(name)
            continue
        if not math.isfinite(value):
            non_finite.append(name)
            continue
        resolved.append(value)
    if non_finite:
        return None, (
            "/joint_states provided non-finite xArm6 positions; "
            f"joints={non_finite}"
        )
    return resolved, ""


def _xarm6_tf_readiness(  # noqa: C901 - exact relay and three TF diagnostics share one node.
    domain_id: int,
    timeout_sec: float,
) -> dict[str, Any]:
    """Wait for the exact root relay input and the complete xArm6 TF chain."""
    rclpy = None
    node = None
    timeout = max(1.0, float(timeout_sec))
    try:
        rclpy = _init_ros_domain(domain_id)
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.time import Time
        from sensor_msgs.msg import JointState
        from tf2_ros import Buffer, TransformException, TransformListener

        class XArm6TfReadinessNode(Node):
            def __init__(self) -> None:
                super().__init__("digital_twin_xarm6_tf_readiness")
                self.relay_ready = False
                self.relay_problem = (
                    "/joint_states has not provided a new message containing "
                    "joint1..joint6"
                )
                self.tf_buffer = Buffer()
                self.tf_listener = TransformListener(
                    self.tf_buffer,
                    self,
                    spin_thread=False,
                )
                self.transform_ready = {
                    "world -> link_base": False,
                    "link_base -> link_eef": False,
                    "world -> link_eef": False,
                }
                self.transform_problem = {
                    name: "transform has not been received"
                    for name in self.transform_ready
                }
                self.create_subscription(
                    JointState,
                    "/joint_states",
                    self._joint_state,
                    20,
                )

            def _joint_state(self, message: JointState) -> None:
                positions, problem = _xarm6_relayed_positions(
                    list(message.name),
                    list(message.position),
                )
                self.relay_problem = problem
                if positions is not None:
                    self.relay_ready = True

            def read_transforms(self) -> None:
                transforms = (
                    ("world -> link_base", "world", "link_base"),
                    ("link_base -> link_eef", "link_base", "link_eef"),
                    ("world -> link_eef", "world", "link_eef"),
                )
                for label, target_frame, source_frame in transforms:
                    if self.transform_ready[label]:
                        continue
                    try:
                        self.tf_buffer.lookup_transform(
                            target_frame,
                            source_frame,
                            Time(),
                            timeout=Duration(seconds=0.05),
                        )
                    except TransformException as exc:
                        self.transform_problem[label] = str(exc)
                        continue
                    self.transform_ready[label] = True
                    self.transform_problem[label] = ""

            def ready(self) -> bool:
                return self.relay_ready and all(self.transform_ready.values())

        node = XArm6TfReadinessNode()
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline and not node.ready():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.read_transforms()
        if node.ready():
            return {
                "success": True,
                "message": (
                    "xArm6 relayed /joint_states and TF world -> link_eef are ready"
                ),
            }
        if not node.relay_ready:
            return {
                "success": False,
                "message": f"xArm6 relayed /joint_states is not ready: {node.relay_problem}",
            }
        for label in (
            "world -> link_base",
            "link_base -> link_eef",
            "world -> link_eef",
        ):
            if not node.transform_ready[label]:
                detail = node.transform_problem[label]
                return {
                    "success": False,
                    "message": f"TF {label} is unavailable: {detail}",
                }
        return {
            "success": False,
            "message": f"xArm6 TF readiness did not complete within {timeout:.1f}s",
        }
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "success": False,
            "message": f"xArm6 relayed /joint_states and TF readiness failed: {exc}",
        }
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy is not None:
            rclpy.shutdown()


def _hardware_joint_state_worker(domain_id: int, robot: str, updates: mp.Queue) -> None:
    rclpy = _init_ros_domain(domain_id)
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    def _put_latest(item: dict[str, Any]) -> None:
        try:
            updates.put_nowait(item)
        except queue.Full:
            try:
                updates.get_nowait()
            except queue.Empty:
                pass
            try:
                updates.put_nowait(item)
            except queue.Full:
                pass

    class HardwareNode(Node):
        def __init__(self) -> None:
            super().__init__("digital_twin_hardware_joint_state")
            for topic in _joint_state_topics(robot, "hardware"):
                self.create_subscription(JointState, topic, self._cb, 10)
            # The gripper may be published in a separate /joint_states message from the
            # arm; remember the latest value so it can ride along with each arm update.
            self._gripper_joint: str | None = None
            self._gripper_position: float | None = None
            self._last_missing_report_ts = 0.0
            self._last_no_match_report_ts = 0.0
            self._started_at = time.time()
            self._seen_arm_match = False
            self._last_unmatched_seen_names: list[str] = []
            self._last_enqueued_positions: list[float] | None = None
            self._last_enqueued_gripper_position: float | None = None
            self._last_enqueued_at = 0.0

        def _cb(self, msg: JointState) -> None:
            snapshot = _snapshot_from_joint_state_msg(msg)
            item, self._gripper_joint, self._gripper_position = _hardware_update_from_snapshot(
                snapshot,
                robot,
                remembered_gripper_joint=self._gripper_joint,
                remembered_gripper_position=self._gripper_position,
            )
            diagnostic = str(item.get("diagnostic") or "")
            if diagnostic:
                now = time.time()
                if diagnostic == "no_matching_hardware_joints":
                    self._last_unmatched_seen_names = list(item.get("seen_names") or [])
                    if (
                        not self._seen_arm_match
                        and now - self._started_at > NO_MATCHING_JOINT_STATE_REPORT_SEC
                        and now - self._last_no_match_report_ts > NO_MATCHING_JOINT_STATE_REPORT_SEC
                    ):
                        _put_latest(
                            {
                                "diagnostic": "no_matching_hardware_joints",
                                "seen_names": list(self._last_unmatched_seen_names),
                                "source_stamp": now,
                            }
                        )
                        self._last_no_match_report_ts = now
                    return
                self._seen_arm_match = True
                if now - self._last_missing_report_ts > 1.0:
                    _put_latest(item)
                    self._last_missing_report_ts = now
                return
            self._seen_arm_match = True
            positions = [float(value) for value in list(item.get("positions") or [])]
            now = time.time()
            gripper_value = item.get("gripper_position")
            if not _should_enqueue_hardware_update(
                robot,
                positions=positions,
                last_positions=self._last_enqueued_positions,
                gripper_position=(
                    float(gripper_value) if gripper_value is not None else None
                ),
                last_gripper_position=self._last_enqueued_gripper_position,
                last_enqueue_ts=self._last_enqueued_at,
                now=now,
            ):
                return
            _put_latest(item)
            self._last_enqueued_positions = positions
            self._last_enqueued_gripper_position = (
                float(gripper_value) if gripper_value is not None else None
            )
            self._last_enqueued_at = now

    node = HardwareNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _gazebo_mirror_worker(  # noqa: C901, PLR0912, PLR0915 - one mirror status lifecycle.
    domain_id: int,
    robot: str,
    target: str,
    status_file: Path,
    direction_file: Path,
    updates: mp.Queue,
) -> None:
    """Stream hardware joint poses into the passive gazebo trajectory controller.

    Instead of teleporting joints with /gazebo/set_model_configuration (which fights
    physics and the gazebo_ros2_control plugin), publish a one-point JointTrajectory
    to the gazebo arm controller each time a new hardware snapshot arrives. The
    controller actively holds the streamed pose, so the model tracks hardware smoothly.
    """
    rclpy = _init_ros_domain(domain_id)
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    gripper_cfg = ROBOTS[robot].get("gripper") or {}
    gripper_topics = list(gripper_cfg.get("gazebo_trajectory_topics") or [])

    class GazeboMirrorNode(Node):
        def __init__(self) -> None:
            super().__init__("digital_twin_gazebo_mirror")
            self._mirror_publishers = [
                self.create_publisher(JointTrajectory, topic, 10)
                for topic in ROBOTS[robot]["gazebo_trajectory_topics"]
            ]
            self._gripper_publishers = [
                self.create_publisher(JointTrajectory, topic, 10)
                for topic in gripper_topics
            ]
            self._gazebo_joint_snapshot: dict[str, float] = {}
            self._gazebo_joint_state_received_at = 0.0
            self._gazebo_joint_state_subscription = self.create_subscription(
                JointState,
                "/joint_states",
                self._on_joint_state,
                10,
            )

        def _on_joint_state(self, message: Any) -> None:
            positions = {
                str(name): float(position)
                for name, position in zip(message.name, message.position, strict=True)
            }
            if positions:
                self._gazebo_joint_snapshot.update(positions)
                self._gazebo_joint_state_received_at = time.time()

    node = GazeboMirrorNode()

    def _connected_publishers() -> list[Any]:
        return [pub for pub in node._mirror_publishers if pub.get_subscription_count() > 0]

    mirror_point_time_sec = _mirror_point_time_sec(robot)
    mirror_min_publish_period_sec = _mirror_min_publish_period_sec(robot)
    mirror_min_joint_delta_rad = _mirror_min_joint_delta_rad(robot)

    def _make_point_traj(joint_names: list[str], positions: list[float]) -> Any:
        traj = JointTrajectory()
        traj.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        whole = int(mirror_point_time_sec)
        point.time_from_start.sec = whole
        point.time_from_start.nanosec = int((mirror_point_time_sec - whole) * 1_000_000_000)
        traj.points = [point]
        return traj

    try:
        deadline = time.time() + 30.0
        connected = _connected_publishers()
        while rclpy.ok() and time.time() < deadline and not connected:
            rclpy.spin_once(node, timeout_sec=0.1)
            connected = _connected_publishers()
        if not connected:
            topics = ", ".join(ROBOTS[robot]["gazebo_trajectory_topics"])
            _write_status(
                status_file,
                target=target,
                state="error",
                direction=_direction(direction_file),
                message=f"gazebo trajectory controller not connected for {robot}: {topics}",
                last_error=f"gazebo trajectory controller not connected for {robot}: {topics}",
            )
            return

        latest: dict[str, Any] | None = None
        last_published_positions: list[float] | None = None
        last_publish_ts = 0.0
        last_status_ts = 0.0
        last_skip_reason = "none"
        last_mirror_max_joint_delta_rad = math.inf
        last_hardware_update_ts = 0.0
        hardware_diagnostic_message = ""
        last_published_joint_names: list[str] = []
        target_changed_at = 0.0
        while rclpy.ok():
            direction = _direction(direction_file)
            if direction != "hardware -> gazebo":
                now = time.time()
                if now - last_status_ts > 1.0:
                    _write_status(
                        status_file,
                        target=target,
                        state="paused",
                        direction=direction,
                        message="direction is gazebo -> hardware; live hardware -> gazebo mirror is paused.",
                    )
                    last_status_ts = now
                # Drop buffered snapshots so we don't replay a stale pose on resume.
                try:
                    while True:
                        updates.get_nowait()
                except queue.Empty:
                    pass
                latest = None
                time.sleep(0.1)
                continue

            try:
                latest = updates.get(timeout=0.25)
                while True:
                    latest = updates.get_nowait()
            except queue.Empty:
                pass

            rclpy.spin_once(node, timeout_sec=0.0)

            if latest is None:
                now = time.time()
                state, message, last_error, hardware_joint_state_age_sec = (
                    _hardware_mirror_status_without_update(
                        last_hardware_update_ts=last_hardware_update_ts,
                        last_published_positions=last_published_positions,
                        hardware_diagnostic_message=hardware_diagnostic_message,
                        now=now,
                    )
                )
                gazebo_status = _gazebo_target_status(
                    robot,
                    target_positions=last_published_positions,
                    gazebo_joint_snapshot=node._gazebo_joint_snapshot,
                    gazebo_joint_state_received_at=node._gazebo_joint_state_received_at,
                    target_changed_at=target_changed_at,
                    now=now,
                )
                if state == "mirroring" and last_published_positions is not None:
                    state = str(gazebo_status["state"])
                    message = str(gazebo_status["message"])
                    last_error = str(gazebo_status["last_error"])
                status_fields = dict(gazebo_status)
                status_fields.update(
                    state=state,
                    message=message,
                    last_error=last_error,
                )

                should_republish = (
                    state in {"mirroring", "waiting"}
                    and last_published_positions is not None
                    and last_published_joint_names
                    and not bool(gazebo_status["gazebo_converged"])
                    and now - last_publish_ts
                    >= max(0.0, MIRROR_GAZEBO_REPUBLISH_PERIOD_SEC)
                    and hardware_joint_state_age_sec is not None
                    and hardware_joint_state_age_sec
                    <= max(0.0, MIRROR_HARDWARE_STALE_SEC)
                )
                if should_republish:
                    connected = _connected_publishers() or node._mirror_publishers
                    arm_traj = _make_point_traj(
                        last_published_joint_names,
                        last_published_positions,
                    )
                    for publisher in connected:
                        publisher.publish(arm_traj)
                    last_publish_ts = now
                if now - last_status_ts > 1.0:
                    _write_status(
                        status_file,
                        target=target,
                        direction=direction,
                        hardware_joint_state_age_sec=hardware_joint_state_age_sec,
                        hardware_heartbeat_sec=max(0.0, MIRROR_HARDWARE_HEARTBEAT_SEC),
                        hardware_stale_sec=max(0.0, MIRROR_HARDWARE_STALE_SEC),
                        **status_fields,
                    )
                    last_status_ts = now
                continue

            if latest.get("diagnostic") == "missing_hardware_joints":
                missing = ", ".join(str(name) for name in (latest.get("missing") or []))
                seen_names = ", ".join(str(name) for name in (latest.get("seen_names") or []))
                message = f"hardware /joint_states missing required {robot} joints: {missing}"
                if seen_names:
                    message += f". Seen joints: {seen_names}"
                hardware_diagnostic_message = message
                _write_status(
                    status_file,
                    target=target,
                    state="waiting",
                    direction=direction,
                    message=message,
                    last_error=message,
                )
                last_status_ts = time.time()
                latest = None
                continue

            if latest.get("diagnostic") == "no_matching_hardware_joints":
                seen_names = ", ".join(str(name) for name in (latest.get("seen_names") or []))
                message = f"hardware /joint_states has no {robot} arm joints yet"
                if seen_names:
                    message += f". Seen joints: {seen_names}"
                hardware_diagnostic_message = message
                _write_status(
                    status_file,
                    target=target,
                    state="waiting",
                    direction=direction,
                    message=message,
                    last_error=message,
                )
                last_status_ts = time.time()
                latest = None
                continue

            connected = _connected_publishers() or node._mirror_publishers
            latest_positions = [float(value) for value in list(latest["positions"])]
            now = time.time()
            last_hardware_update_ts = max(
                last_hardware_update_ts,
                float(latest.get("source_stamp") or now),
            )
            hardware_diagnostic_message = ""
            mirror_max_joint_delta_rad = _max_position_delta_rad(
                last_published_positions,
                latest_positions,
            )
            target_changed = mirror_max_joint_delta_rad >= max(
                0.0,
                mirror_min_joint_delta_rad,
            )
            should_publish, skip_reason = _should_publish_mirror_update(
                robot,
                positions=latest_positions,
                last_positions=last_published_positions,
                last_publish_ts=last_publish_ts,
                now=now,
            )
            gazebo_status = _gazebo_target_status(
                robot,
                target_positions=last_published_positions,
                gazebo_joint_snapshot=node._gazebo_joint_snapshot,
                gazebo_joint_state_received_at=node._gazebo_joint_state_received_at,
                target_changed_at=target_changed_at,
                now=now,
            )
            if (
                not should_publish
                and not bool(gazebo_status["gazebo_converged"])
                and now - last_publish_ts
                >= max(0.0, MIRROR_GAZEBO_REPUBLISH_PERIOD_SEC)
            ):
                should_publish = True
                skip_reason = "gazebo_not_converged"
            if not should_publish:
                last_skip_reason = skip_reason or "skipped"
                status_now = time.time()
                if status_now - last_status_ts > 0.5:
                    latency_ms = (status_now - float(latest.get("source_stamp") or status_now)) * 1000.0
                    _write_status(
                        status_file,
                        target=target,
                        direction=direction,
                        latency_ms=latency_ms,
                        **gazebo_status,
                    )
                    last_status_ts = status_now
                if skip_reason == "below_delta":
                    latest = None
                rclpy.spin_once(node, timeout_sec=0.0)
                continue

            arm_traj = _make_point_traj(latest["joint_names"], latest_positions)
            for publisher in connected:
                publisher.publish(arm_traj)
            last_published_positions = list(latest_positions)
            last_published_joint_names = [str(name) for name in latest["joint_names"]]
            last_publish_ts = now
            if target_changed or target_changed_at <= 0.0:
                target_changed_at = now
            last_mirror_max_joint_delta_rad = mirror_max_joint_delta_rad
            last_skip_reason = "none"

            gripper_joint = latest.get("gripper_joint")
            gripper_position = latest.get("gripper_position")
            if node._gripper_publishers and gripper_joint is not None and gripper_position is not None:
                gripper_traj = _make_point_traj([gripper_joint], [float(gripper_position)])
                for publisher in node._gripper_publishers:
                    publisher.publish(gripper_traj)

            rclpy.spin_once(node, timeout_sec=0.0)

            now = time.time()
            latency_ms = (now - float(latest.get("source_stamp") or now)) * 1000.0
            if now - last_status_ts > 0.5:
                gazebo_status = _gazebo_target_status(
                    robot,
                    target_positions=last_published_positions,
                    gazebo_joint_snapshot=node._gazebo_joint_snapshot,
                    gazebo_joint_state_received_at=node._gazebo_joint_state_received_at,
                    target_changed_at=target_changed_at,
                    now=now,
                )
                _write_status(
                    status_file,
                    target=target,
                    direction=direction,
                    latency_ms=latency_ms,
                    mirror_point_time_sec=mirror_point_time_sec,
                    mirror_min_publish_period_sec=mirror_min_publish_period_sec,
                    mirror_min_joint_delta_rad=mirror_min_joint_delta_rad,
                    mirror_max_joint_delta_rad=last_mirror_max_joint_delta_rad,
                    last_skip_reason=last_skip_reason,
                    **gazebo_status,
                )
                last_status_ts = now
            latest = None
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _publish_trajectory_worker(
    domain_id: int,
    topics: list[str],
    joint_names: list[str],
    points: list[dict[str, Any]],
    result_queue: mp.Queue,
) -> None:
    """Publish a (possibly multi-point) JointTrajectory to the first connected topic.

    ``points`` is a list of ``{"positions": [...], "time": <sec from start>}`` dicts, so this
    handles both the one-shot single-pose apply and multi-waypoint recorded replays.
    """
    rclpy = _init_ros_domain(domain_id)
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class TrajectoryNode(Node):
        def __init__(self) -> None:
            super().__init__("digital_twin_trajectory_publish")
            # NB: do not name this `publishers` — rclpy Node has a read-only `publishers`
            # property and assigning it raises AttributeError (crashes the worker).
            self._pubs = [
                self.create_publisher(JointTrajectory, topic, 10)
                for topic in topics
            ]

    node = None
    try:
        node = TrajectoryNode()
        deadline = time.time() + 3.0
        connected: list[Any] = []
        while time.time() < deadline and not connected:
            rclpy.spin_once(node, timeout_sec=0.1)
            connected = [pub for pub in node._pubs if pub.get_subscription_count() > 0]
        if not connected:
            result_queue.put({"success": False, "message": "trajectory controller not connected."})
            return

        traj = JointTrajectory()
        traj.joint_names = list(joint_names)
        traj.points = []
        for entry in points:
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in entry["positions"]]
            if "velocities" in entry:
                point.velocities = [float(v) for v in entry["velocities"]]
            if "accelerations" in entry:
                point.accelerations = [float(v) for v in entry["accelerations"]]
            duration_sec = float(entry.get("time") or 0.0)
            whole = int(duration_sec)
            point.time_from_start.sec = whole
            point.time_from_start.nanosec = int((duration_sec - whole) * 1_000_000_000)
            traj.points.append(point)
        for _ in range(3):
            for publisher in connected:
                publisher.publish(traj)
            rclpy.spin_once(node, timeout_sec=0.1)
        result_queue.put({"success": True, "message": "trajectory published."})
    except Exception as exc:
        result_queue.put({"success": False, "message": str(exc)})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _publish_trajectory(
    domain_id: int,
    topics: list[str],
    joint_names: list[str],
    points: list[dict[str, Any]],
    join_timeout_sec: float = 8.0,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_publish_trajectory_worker,
        args=(domain_id, list(topics), list(joint_names), list(points), result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(join_timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {"success": False, "message": "trajectory publish timed out."}
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {"success": False, "message": "trajectory publish returned no data."}


def _goal_status_label(status: int | None) -> str:
    labels = {
        0: "unknown",
        1: "accepted",
        2: "executing",
        3: "canceling",
        4: "succeeded",
        5: "canceled",
        6: "aborted",
    }
    if status is None:
        return "unknown"
    return labels.get(int(status), f"status {status}")


def _wait_follow_joint_trajectory_action_worker(
    domain_id: int,
    action_name: str,
    timeout_sec: float,
    result_queue: mp.Queue,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from rclpy.node import Node

    node = None
    try:
        node = Node("digital_twin_follow_joint_trajectory_preflight")
        client = ActionClient(node, FollowJointTrajectory, action_name)
        if not client.wait_for_server(timeout_sec=max(0.5, float(timeout_sec))):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action server unavailable.",
                }
            )
            return
        result_queue.put(
            {
                "success": True,
                "message": f"{action_name}: action server available.",
            }
        )
    except Exception as exc:
        result_queue.put({"success": False, "message": f"{action_name}: {exc}"})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _wait_follow_joint_trajectory_action(
    domain_id: int,
    action_name: str,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_wait_follow_joint_trajectory_action_worker,
        args=(domain_id, action_name, timeout_sec, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(max(1.0, float(timeout_sec)) + 2.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {"success": False, "message": f"{action_name}: action server wait timed out."}
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {"success": False, "message": f"{action_name}: action server wait returned no data."}


def _follow_joint_trajectory_time_summary(points: list[dict[str, Any]]) -> dict[str, float | None]:
    times = [max(0.0, float(dict(point).get("time") or 0.0)) for point in points]
    spacings = [current - previous for previous, current in zip(times, times[1:])]
    return {
        "first_point_time": times[0] if times else None,
        "second_point_time": times[1] if len(times) > 1 else None,
        "final_point_time": times[-1] if times else None,
        "min_point_spacing": min(spacings) if spacings else None,
    }


def _format_optional_time(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _controller_name_from_follow_joint_trajectory_action(action_name: str) -> str:
    parts = [part for part in str(action_name or "").strip().split("/") if part]
    if len(parts) >= 2 and parts[-1] == "follow_joint_trajectory":
        return parts[-2]
    return ""


def _follow_joint_trajectory_metadata(
    *,
    domain_id: int,
    action_name: str,
    joint_names: list[str],
    points: list[dict[str, Any]],
    start_delay_sec: float,
    goal_time_tolerance_sec: float,
    header_stamp: bool,
    controller_state: str = "",
) -> dict[str, Any]:
    timing = _follow_joint_trajectory_time_summary(points)
    metadata: dict[str, Any] = {
        "ros_domain_id": int(domain_id),
        "action_name": str(action_name),
        "joint_names": [str(name) for name in joint_names],
        "points": len(points),
        "header_stamp": bool(header_stamp),
        "start_delay_sec": float(start_delay_sec),
        "goal_time_tolerance_sec": float(goal_time_tolerance_sec),
        **timing,
    }
    if controller_state:
        metadata["controller_state"] = str(controller_state)
    return metadata


def _follow_joint_trajectory_diagnostic_detail(
    *,
    domain_id: int,
    action_name: str,
    joint_names: list[str],
    points: list[dict[str, Any]],
    start_delay_sec: float,
    goal_time_tolerance_sec: float,
    header_stamp: bool,
    controller_state: str = "",
) -> str:
    metadata = _follow_joint_trajectory_metadata(
        domain_id=domain_id,
        action_name=action_name,
        joint_names=joint_names,
        points=points,
        start_delay_sec=start_delay_sec,
        goal_time_tolerance_sec=goal_time_tolerance_sec,
        header_stamp=header_stamp,
        controller_state=controller_state,
    )
    parts = [
        f"ROS_DOMAIN_ID={metadata['ros_domain_id']}",
        f"action={metadata['action_name']}",
        f"joints={','.join(str(name) for name in metadata['joint_names'])}",
        f"points={metadata['points']}",
        f"first_point_time={_format_optional_time(metadata.get('first_point_time'))}",
        f"second_point_time={_format_optional_time(metadata.get('second_point_time'))}",
        f"final_point_time={_format_optional_time(metadata.get('final_point_time'))}",
        f"min_point_spacing={_format_optional_time(metadata.get('min_point_spacing'))}",
        f"header_stamp={bool(metadata['header_stamp'])}",
        f"start_delay_sec={float(metadata['start_delay_sec']):.3f}",
        f"goal_time_tolerance_sec={float(metadata['goal_time_tolerance_sec']):.3f}",
    ]
    if metadata.get("controller_state"):
        parts.append(f"controller_state={metadata['controller_state']}")
    return "; ".join(parts)


def _send_follow_joint_trajectory_worker(
    domain_id: int,
    action_name: str,
    joint_names: list[str],
    points: list[dict[str, Any]],
    result_timeout_sec: float,
    result_queue: mp.Queue,
    start_delay_sec: float,
    goal_time_tolerance_sec: float,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectoryPoint

    def _wait_future(node: Any, future: Any, timeout_sec: float) -> bool:
        deadline = time.time() + max(0.5, float(timeout_sec))
        while rclpy.ok() and time.time() < deadline and not future.done():
            rclpy.spin_once(node, timeout_sec=0.1)
        return bool(future.done())

    node = None
    try:
        node = Node("digital_twin_follow_joint_trajectory_send")
        client = ActionClient(node, FollowJointTrajectory, action_name)
        if not client.wait_for_server(timeout_sec=5.0):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action server unavailable.",
                }
            )
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(joint_names)
        if float(goal_time_tolerance_sec) > 0.0:
            tolerance = max(0.0, float(goal_time_tolerance_sec))
            whole_tolerance = int(tolerance)
            goal.goal_time_tolerance.sec = whole_tolerance
            goal.goal_time_tolerance.nanosec = int(
                (tolerance - whole_tolerance) * 1_000_000_000
            )
        if float(start_delay_sec) > 0.0:
            stamp = node.get_clock().now().to_msg()
            delay = max(0.0, float(start_delay_sec))
            whole_delay = int(delay)
            nano_delay = int((delay - whole_delay) * 1_000_000_000)
            total_nanosec = int(stamp.nanosec) + nano_delay
            stamp.sec = int(stamp.sec) + whole_delay + total_nanosec // 1_000_000_000
            stamp.nanosec = total_nanosec % 1_000_000_000
            goal.trajectory.header.stamp = stamp
        goal.trajectory.points = []
        for entry in points:
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in entry["positions"]]
            if "velocities" in entry:
                point.velocities = [float(v) for v in entry["velocities"]]
            if "accelerations" in entry:
                point.accelerations = [float(v) for v in entry["accelerations"]]
            duration_sec = float(entry.get("time") or 0.0)
            whole = int(duration_sec)
            point.time_from_start.sec = whole
            point.time_from_start.nanosec = int((duration_sec - whole) * 1_000_000_000)
            goal.trajectory.points.append(point)

        send_future = client.send_goal_async(goal)
        if not _wait_future(node, send_future, 8.0):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action goal acceptance timed out.",
                }
            )
            return

        goal_handle = send_future.result()
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            detail = _follow_joint_trajectory_diagnostic_detail(
                domain_id=domain_id,
                action_name=action_name,
                joint_names=joint_names,
                points=points,
                start_delay_sec=start_delay_sec,
                goal_time_tolerance_sec=goal_time_tolerance_sec,
                header_stamp=float(start_delay_sec) > 0.0,
            )
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action goal rejected; {detail}.",
                    "skip_observed_completion": True,
                    **_follow_joint_trajectory_metadata(
                        domain_id=domain_id,
                        action_name=action_name,
                        joint_names=joint_names,
                        points=points,
                        start_delay_sec=start_delay_sec,
                        goal_time_tolerance_sec=goal_time_tolerance_sec,
                        header_stamp=float(start_delay_sec) > 0.0,
                    ),
                }
            )
            return

        result_future = goal_handle.get_result_async()
        if not _wait_future(node, result_future, result_timeout_sec):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action accepted; result timed out.",
                }
            )
            return

        result_response = result_future.result()
        status = getattr(result_response, "status", None)
        label = _goal_status_label(status)
        action_result = getattr(result_response, "result", None)
        error_code = getattr(action_result, "error_code", None)
        error_string = str(getattr(action_result, "error_string", "") or "")
        status_int = int(status) if status is not None else None
        success = status_int == 4 and (error_code is None or int(error_code) == 0)
        message = f"{action_name}: action accepted; action {label}"
        if error_code is not None:
            message += f"; error_code={int(error_code)}"
        if error_string:
            message += f"; {error_string}"
        result_queue.put(
            {
                "success": success,
                "message": message + ".",
                "status": status_int,
                "status_label": label,
                "error_code": int(error_code) if error_code is not None else None,
                **_follow_joint_trajectory_metadata(
                    domain_id=domain_id,
                    action_name=action_name,
                    joint_names=joint_names,
                    points=points,
                    start_delay_sec=start_delay_sec,
                    goal_time_tolerance_sec=goal_time_tolerance_sec,
                    header_stamp=float(start_delay_sec) > 0.0,
                ),
            }
        )
    except Exception as exc:
        result_queue.put({"success": False, "message": f"{action_name}: {exc}"})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _publish_follow_joint_trajectory_action(
    domain_id: int,
    action_name: str,
    joint_names: list[str],
    points: list[dict[str, Any]],
    join_timeout_sec: float = 8.0,
    start_delay_sec: float = 0.0,
    goal_time_tolerance_sec: float = 0.0,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_send_follow_joint_trajectory_worker,
        args=(
            domain_id,
            action_name,
            list(joint_names),
            list(points),
            join_timeout_sec,
            result_queue,
            start_delay_sec,
            goal_time_tolerance_sec,
        ),
        daemon=True,
    )
    proc.start()
    proc.join(max(1.0, float(join_timeout_sec)) + 15.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {"success": False, "message": f"{action_name}: action worker timed out."}
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {"success": False, "message": f"{action_name}: action worker returned no data."}


def _send_paired_follow_joint_trajectory_worker(
    domain_id: int,
    arms: list[dict[str, Any]],
    result_timeout_sec: float,
    result_queue: mp.Queue,
    start_delay_sec: float,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectoryPoint

    def _wait_future(node: Any, future: Any, wait_timeout_sec: float) -> bool:
        deadline = time.time() + max(0.5, float(wait_timeout_sec))
        while rclpy.ok() and time.time() < deadline and not future.done():
            rclpy.spin_once(node, timeout_sec=0.1)
        return bool(future.done())

    def _wait_futures(node: Any, futures: list[Any], wait_timeout_sec: float) -> bool:
        deadline = time.time() + max(0.5, float(wait_timeout_sec))
        while rclpy.ok() and time.time() < deadline:
            if all(future.done() for future in futures):
                return True
            rclpy.spin_once(node, timeout_sec=0.1)
        return all(future.done() for future in futures)

    def _controller_state(node: Any, controller_name: str) -> str:
        name = str(controller_name or "").strip()
        if not name:
            return ""
        try:
            from controller_manager_msgs.srv import ListControllers

            client = node.create_client(ListControllers, "/controller_manager/list_controllers")
            if not client.wait_for_service(timeout_sec=1.0):
                return "unknown(service_unavailable)"
            future = client.call_async(ListControllers.Request())
            if not _wait_future(node, future, 1.5):
                return "unknown(service_timeout)"
            response = future.result()
            for controller in list(getattr(response, "controller", []) or []):
                if str(getattr(controller, "name", "") or "") == name:
                    return str(getattr(controller, "state", "") or "unknown")
            return "unknown(not_listed)"
        except Exception as exc:
            return f"unknown({exc})"

    def _metadata(arm: dict[str, Any], *, controller_state: str = "") -> dict[str, Any]:
        return _follow_joint_trajectory_metadata(
            domain_id=domain_id,
            action_name=str(arm.get("action_name") or ""),
            joint_names=[str(name) for name in list(arm.get("joint_names") or [])],
            points=[dict(point) for point in list(arm.get("points") or [])],
            start_delay_sec=start_delay_sec,
            goal_time_tolerance_sec=float(arm.get("goal_time_tolerance_sec") or 0.0),
            header_stamp=True,
            controller_state=controller_state,
        )

    def _detail(arm: dict[str, Any], *, controller_state: str = "") -> str:
        return _follow_joint_trajectory_diagnostic_detail(
            domain_id=domain_id,
            action_name=str(arm.get("action_name") or ""),
            joint_names=[str(name) for name in list(arm.get("joint_names") or [])],
            points=[dict(point) for point in list(arm.get("points") or [])],
            start_delay_sec=start_delay_sec,
            goal_time_tolerance_sec=float(arm.get("goal_time_tolerance_sec") or 0.0),
            header_stamp=True,
            controller_state=controller_state,
        )

    def _result(
        arm: dict[str, Any],
        *,
        success: bool,
        message: str,
        controller_state: str = "",
        skip_observed_completion: bool = False,
        canceled: bool = False,
        status: int | None = None,
        status_label: str = "",
        error_code: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "success": bool(success),
            "message": message,
            "status": status,
            "status_label": status_label,
            "error_code": error_code,
            **_metadata(arm, controller_state=controller_state),
        }
        if skip_observed_completion:
            payload["skip_observed_completion"] = True
        if canceled:
            payload["canceled"] = True
        return payload

    def _make_goal(arm: dict[str, Any], stamp_sec: int, stamp_nanosec: int) -> Any:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [str(name) for name in list(arm.get("joint_names") or [])]
        tolerance = max(0.0, float(arm.get("goal_time_tolerance_sec") or 0.0))
        if tolerance > 0.0:
            whole_tolerance = int(tolerance)
            goal.goal_time_tolerance.sec = whole_tolerance
            goal.goal_time_tolerance.nanosec = int(
                (tolerance - whole_tolerance) * 1_000_000_000
            )
        goal.trajectory.header.stamp.sec = int(stamp_sec)
        goal.trajectory.header.stamp.nanosec = int(stamp_nanosec)
        for entry in [dict(point) for point in list(arm.get("points") or [])]:
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in entry["positions"]]
            if "velocities" in entry:
                point.velocities = [float(v) for v in entry["velocities"]]
            if "accelerations" in entry:
                point.accelerations = [float(v) for v in entry["accelerations"]]
            duration_sec = float(entry.get("time") or 0.0)
            whole = int(duration_sec)
            point.time_from_start.sec = whole
            point.time_from_start.nanosec = int((duration_sec - whole) * 1_000_000_000)
            goal.trajectory.points.append(point)
        return goal

    node = None
    try:
        node = Node("digital_twin_paired_follow_joint_trajectory_send")
        arms_by_key = {
            str(arm.get("key") or arm.get("robot") or arm.get("action_name")): dict(arm)
            for arm in arms
        }
        clients: dict[str, Any] = {}
        for key, arm in arms_by_key.items():
            action_name = str(arm.get("action_name") or "")
            client = ActionClient(node, FollowJointTrajectory, action_name)
            clients[key] = client
            if not client.wait_for_server(timeout_sec=5.0):
                results = {
                    item_key: _result(
                        item_arm,
                        success=False,
                        message=(
                            f"{str(item_arm.get('action_name') or '')}: not sent because paired arm action server "
                            f"was unavailable; {_detail(item_arm)}."
                            if item_key != key
                            else f"{action_name}: action server unavailable; {_detail(item_arm)}."
                        ),
                        skip_observed_completion=True,
                    )
                    for item_key, item_arm in arms_by_key.items()
                }
                result_queue.put(results)
                return

        stamp = node.get_clock().now().to_msg()
        delay = max(0.0, float(start_delay_sec))
        whole_delay = int(delay)
        nano_delay = int((delay - whole_delay) * 1_000_000_000)
        total_nanosec = int(stamp.nanosec) + nano_delay
        stamp_sec = int(stamp.sec) + whole_delay + total_nanosec // 1_000_000_000
        stamp_nanosec = total_nanosec % 1_000_000_000

        send_futures = {
            key: clients[key].send_goal_async(_make_goal(arm, stamp_sec, stamp_nanosec))
            for key, arm in arms_by_key.items()
        }
        accepted_handles: dict[str, Any] = {}
        failed_results: dict[str, dict[str, Any]] = {}
        _wait_futures(node, list(send_futures.values()), 8.0)
        for key, future in send_futures.items():
            arm = arms_by_key[key]
            action_name = str(arm.get("action_name") or "")
            controller_name = str(
                arm.get("controller_name")
                or _controller_name_from_follow_joint_trajectory_action(action_name)
            )
            controller_state = _controller_state(node, controller_name) if key == "ur5e" else ""
            if not future.done():
                failed_results[key] = _result(
                    arm,
                    success=False,
                    message=(
                        f"{action_name}: action goal acceptance timed out; "
                        f"{_detail(arm, controller_state=controller_state)}."
                    ),
                    controller_state=controller_state,
                    skip_observed_completion=True,
                )
                continue
            goal_handle = future.result()
            if goal_handle is None or not getattr(goal_handle, "accepted", False):
                failed_results[key] = _result(
                    arm,
                    success=False,
                    message=(
                        f"{action_name}: action goal rejected; "
                        f"{_detail(arm, controller_state=controller_state)}."
                    ),
                    controller_state=controller_state,
                    skip_observed_completion=True,
                )
                continue
            accepted_handles[key] = goal_handle

        if failed_results:
            results = dict(failed_results)
            for key, goal_handle in accepted_handles.items():
                arm = arms_by_key[key]
                action_name = str(arm.get("action_name") or "")
                try:
                    cancel_future = goal_handle.cancel_goal_async()
                    _wait_future(node, cancel_future, 2.0)
                except Exception:
                    pass
                results[key] = _result(
                    arm,
                    success=False,
                    message=(
                        f"{action_name}: action accepted; canceled before execution because paired arm goal "
                        f"was rejected; {_detail(arm)}."
                    ),
                    skip_observed_completion=True,
                    canceled=True,
                )
            result_queue.put(results)
            return

        result_futures = {
            key: goal_handle.get_result_async()
            for key, goal_handle in accepted_handles.items()
        }
        _wait_futures(node, list(result_futures.values()), result_timeout_sec)
        results: dict[str, dict[str, Any]] = {}
        for key, future in result_futures.items():
            arm = arms_by_key[key]
            action_name = str(arm.get("action_name") or "")
            if not future.done():
                results[key] = _result(
                    arm,
                    success=False,
                    message=f"{action_name}: action accepted; result timed out; {_detail(arm)}.",
                )
                continue
            result_response = future.result()
            status = getattr(result_response, "status", None)
            label = _goal_status_label(status)
            action_result = getattr(result_response, "result", None)
            error_code = getattr(action_result, "error_code", None)
            error_string = str(getattr(action_result, "error_string", "") or "")
            status_int = int(status) if status is not None else None
            success = status_int == 4 and (error_code is None or int(error_code) == 0)
            message = f"{action_name}: action accepted; action {label}"
            if error_code is not None:
                message += f"; error_code={int(error_code)}"
            if error_string:
                message += f"; {error_string}"
            results[key] = _result(
                arm,
                success=success,
                message=message + ".",
                status=status_int,
                status_label=label,
                error_code=int(error_code) if error_code is not None else None,
            )
        result_queue.put(results)
    except Exception as exc:
        result_queue.put(
            {
                str(arm.get("key") or arm.get("robot") or arm.get("action_name")): {
                    "success": False,
                    "message": f"{str(arm.get('action_name') or '')}: {exc}",
                    "skip_observed_completion": True,
                }
                for arm in arms
            }
        )
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _publish_paired_follow_joint_trajectory_actions(
    domain_id: int,
    arms: list[dict[str, Any]],
    join_timeout_sec: float = 8.0,
    start_delay_sec: float = PAIRED_HARDWARE_TRAJECTORY_START_DELAY_SEC,
) -> dict[str, dict[str, Any]]:
    arm_items = [dict(arm) for arm in arms]
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_send_paired_follow_joint_trajectory_worker,
        args=(
            domain_id,
            arm_items,
            join_timeout_sec,
            result_queue,
            start_delay_sec,
        ),
        daemon=True,
    )
    proc.start()
    proc.join(max(1.0, float(join_timeout_sec)) + 15.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {
            str(arm.get("key") or arm.get("robot") or arm.get("action_name")): {
                "success": False,
                "message": f"{str(arm.get('action_name') or '')}: paired action worker timed out.",
                "skip_observed_completion": True,
            }
            for arm in arm_items
        }
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        result = {}
    if isinstance(result, dict) and result:
        return {
            str(key): dict(value)
            for key, value in result.items()
            if isinstance(value, dict)
        }
    return {
        str(arm.get("key") or arm.get("robot") or arm.get("action_name")): {
            "success": False,
            "message": f"{str(arm.get('action_name') or '')}: paired action worker returned no data.",
            "skip_observed_completion": True,
        }
        for arm in arm_items
    }


def _wait_execute_trajectory_action_worker(
    domain_id: int,
    timeout_sec: float,
    result_queue: mp.Queue,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from moveit_msgs.action import ExecuteTrajectory
    from rclpy.action import ActionClient
    from rclpy.node import Node

    node = None
    action_name = "/execute_trajectory"
    try:
        node = Node("digital_twin_execute_trajectory_preflight")
        client = ActionClient(node, ExecuteTrajectory, action_name)
        if not client.wait_for_server(timeout_sec=max(0.5, float(timeout_sec))):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action server unavailable.",
                }
            )
            return
        result_queue.put(
            {
                "success": True,
                "message": f"{action_name}: action server available.",
            }
        )
    except Exception as exc:
        result_queue.put({"success": False, "message": f"{action_name}: {exc}"})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _wait_execute_trajectory_action(
    domain_id: int,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_wait_execute_trajectory_action_worker,
        args=(domain_id, timeout_sec, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(max(1.0, float(timeout_sec)) + 2.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {"success": False, "message": "/execute_trajectory: action server wait timed out."}
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {"success": False, "message": "/execute_trajectory: action server wait returned no data."}


def _wait_move_group_action_worker(
    domain_id: int,
    timeout_sec: float,
    result_queue: mp.Queue,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from moveit_msgs.action import MoveGroup
    from rclpy.action import ActionClient
    from rclpy.node import Node

    node = None
    action_name = MOVE_GROUP_ACTION_NAME
    try:
        node = Node("digital_twin_move_group_preflight")
        client = ActionClient(node, MoveGroup, action_name)
        if not client.wait_for_server(timeout_sec=max(0.5, float(timeout_sec))):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action server unavailable.",
                }
            )
            return
        result_queue.put(
            {
                "success": True,
                "message": f"{action_name}: action server available.",
            }
        )
    except Exception as exc:
        result_queue.put({"success": False, "message": f"{action_name}: {exc}"})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _wait_move_group_action(
    domain_id: int,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_wait_move_group_action_worker,
        args=(domain_id, timeout_sec, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(max(1.0, float(timeout_sec)) + 2.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {"success": False, "message": f"{MOVE_GROUP_ACTION_NAME}: action server wait timed out."}
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {"success": False, "message": f"{MOVE_GROUP_ACTION_NAME}: action server wait returned no data."}


def _send_execute_trajectory_worker(
    domain_id: int,
    joint_names: list[str],
    points: list[dict[str, Any]],
    result_timeout_sec: float,
    result_queue: mp.Queue,
    start_delay_sec: float,
    use_header_stamp: bool,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from moveit_msgs.action import ExecuteTrajectory
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectoryPoint

    def _wait_future(node: Any, future: Any, timeout_sec: float) -> bool:
        deadline = time.time() + max(0.5, float(timeout_sec))
        while rclpy.ok() and time.time() < deadline and not future.done():
            rclpy.spin_once(node, timeout_sec=0.1)
        return bool(future.done())

    action_name = "/execute_trajectory"
    node = None
    try:
        node = Node("digital_twin_execute_trajectory_send")
        client = ActionClient(node, ExecuteTrajectory, action_name)
        if not client.wait_for_server(timeout_sec=5.0):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action server unavailable.",
                }
            )
            return

        goal = ExecuteTrajectory.Goal()
        goal.trajectory.joint_trajectory.joint_names = list(joint_names)
        if bool(use_header_stamp) and float(start_delay_sec) > 0.0:
            stamp = node.get_clock().now().to_msg()
            delay = max(0.0, float(start_delay_sec))
            whole_delay = int(delay)
            nano_delay = int((delay - whole_delay) * 1_000_000_000)
            total_nanosec = int(stamp.nanosec) + nano_delay
            stamp.sec = int(stamp.sec) + whole_delay + total_nanosec // 1_000_000_000
            stamp.nanosec = total_nanosec % 1_000_000_000
            goal.trajectory.joint_trajectory.header.stamp = stamp
        goal.trajectory.joint_trajectory.points = []
        for entry in points:
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in entry["positions"]]
            if "velocities" in entry:
                point.velocities = [float(v) for v in entry["velocities"]]
            if "accelerations" in entry:
                point.accelerations = [float(v) for v in entry["accelerations"]]
            duration_sec = float(entry.get("time") or 0.0)
            whole = int(duration_sec)
            point.time_from_start.sec = whole
            point.time_from_start.nanosec = int((duration_sec - whole) * 1_000_000_000)
            goal.trajectory.joint_trajectory.points.append(point)

        send_future = client.send_goal_async(goal)
        if not _wait_future(node, send_future, 8.0):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action goal acceptance timed out.",
                }
            )
            return

        goal_handle = send_future.result()
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action goal rejected.",
                }
            )
            return

        result_future = goal_handle.get_result_async()
        if not _wait_future(node, result_future, result_timeout_sec):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action accepted; result timed out.",
                }
            )
            return

        result_response = result_future.result()
        status = getattr(result_response, "status", None)
        label = _goal_status_label(status)
        action_result = getattr(result_response, "result", None)
        error_code = getattr(action_result, "error_code", None)
        error_val = getattr(error_code, "val", error_code)
        status_int = int(status) if status is not None else None
        success = status_int == 4 and (error_val is None or int(error_val) == 1)
        message = f"{action_name}: action accepted; action {label}"
        if error_val is not None:
            message += f"; error_code={int(error_val)}"
        result_queue.put(
            {
                "success": success,
                "message": message + ".",
                "status": status_int,
                "status_label": label,
                "error_code": int(error_val) if error_val is not None else None,
            }
        )
    except Exception as exc:
        result_queue.put({"success": False, "message": f"{action_name}: {exc}"})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _publish_execute_trajectory_action(
    domain_id: int,
    joint_names: list[str],
    points: list[dict[str, Any]],
    join_timeout_sec: float = 8.0,
    start_delay_sec: float = 0.0,
    use_header_stamp: bool = False,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_send_execute_trajectory_worker,
        args=(
            domain_id,
            list(joint_names),
            list(points),
            join_timeout_sec,
            result_queue,
            start_delay_sec,
            use_header_stamp,
        ),
        daemon=True,
    )
    proc.start()
    proc.join(max(1.0, float(join_timeout_sec)) + 15.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {"success": False, "message": "/execute_trajectory: action worker timed out."}
    try:
        result = dict(result_queue.get_nowait())
        result.update(
            {
                "header_stamp": bool(use_header_stamp),
                "start_delay_sec": float(start_delay_sec),
                **_follow_joint_trajectory_metadata(
                    "/execute_trajectory",
                    joint_names,
                    points,
                    header_stamp=bool(use_header_stamp),
                    start_delay_sec=float(start_delay_sec),
                ),
            }
        )
        return result
    except queue.Empty:
        return {"success": False, "message": "/execute_trajectory: action worker returned no data."}


def _serialize_joint_trajectory(trajectory: Any) -> dict[str, Any]:
    joint_trajectory = getattr(trajectory, "joint_trajectory", trajectory)
    points: list[dict[str, Any]] = []
    for point in list(getattr(joint_trajectory, "points", []) or []):
        duration = getattr(point, "time_from_start", None)
        seconds = 0.0
        if duration is not None:
            seconds = float(getattr(duration, "sec", 0)) + float(getattr(duration, "nanosec", 0)) / 1_000_000_000.0
        item: dict[str, Any] = {
            "positions": [float(v) for v in list(getattr(point, "positions", []) or [])],
            "time": seconds,
        }
        velocities = list(getattr(point, "velocities", []) or [])
        accelerations = list(getattr(point, "accelerations", []) or [])
        if velocities:
            item["velocities"] = [float(v) for v in velocities]
        if accelerations:
            item["accelerations"] = [float(v) for v in accelerations]
        points.append(item)
    return {
        "joint_names": [str(name) for name in list(getattr(joint_trajectory, "joint_names", []) or [])],
        "points": points,
    }


def _positions_match(
    left: list[float],
    right: list[float],
    tolerance_rad: float = MOVE_GROUP_JOINT_TOLERANCE_RAD,
) -> bool:
    if len(left) != len(right):
        return False
    return all(_angular_delta(a, b) <= tolerance_rad for a, b in zip(left, right))


def _remap_trajectory_point_values(
    point: dict[str, Any],
    key: str,
    source_joint_names: list[str],
    target_joint_names: list[str],
) -> list[float]:
    values = [float(value) for value in list(point.get(key) or [])]
    if not values:
        return []
    if len(values) != len(source_joint_names):
        raise ValueError(f"{key} length does not match planned trajectory joints")
    index_by_name = {name: index for index, name in enumerate(source_joint_names)}
    return [values[index_by_name[name]] for name in target_joint_names]


def _stitch_ur5e_move_group_plan_results(
    joint_names: list[str],
    plan_results: list[dict[str, Any]],
) -> dict[str, Any]:
    target_joint_names = [str(name) for name in list(joint_names or [])]
    if not target_joint_names:
        return {
            "success": False,
            "message": "UR5e MoveIt planned trajectory has no hardware joint names.",
        }
    if not plan_results:
        return {
            "success": False,
            "message": "UR5e MoveIt planned trajectory has no planned segments.",
        }

    stitched_points: list[dict[str, Any]] = []
    last_positions: list[float] | None = None
    last_time = -UR5E_TEACH_REPLAY_MIN_POINT_STEP_SEC
    has_velocities = False
    has_accelerations = False
    segment_count = 0

    for segment_index, result in enumerate(plan_results, start=1):
        if not bool(result.get("success")):
            return {
                "success": False,
                "message": (
                    f"UR5e MoveIt planned trajectory segment {segment_index} failed: "
                    f"{str(result.get('message') or 'planning failed')}"
                ),
            }
        trajectory = dict(result.get("trajectory") or {})
        source_joint_names = [str(name) for name in list(trajectory.get("joint_names") or [])]
        raw_points = [dict(point) for point in list(trajectory.get("points") or [])]
        if not source_joint_names or not raw_points:
            return {
                "success": False,
                "message": f"UR5e MoveIt planned trajectory segment {segment_index} has no joint names or points.",
            }
        if set(source_joint_names) != set(target_joint_names):
            return {
                "success": False,
                "message": (
                    f"UR5e MoveIt planned trajectory segment {segment_index} joint names "
                    "do not match hardware joint names."
                ),
                "planned_joint_names": source_joint_names,
                "expected_joint_names": target_joint_names,
            }

        segment_count += 1
        base_time = max(0.0, float(raw_points[0].get("time") or 0.0))
        segment_start_time = max(0.0, last_time)
        for point_index, point in enumerate(raw_points):
            try:
                positions = _remap_trajectory_point_values(
                    point,
                    "positions",
                    source_joint_names,
                    target_joint_names,
                )
                velocities = _remap_trajectory_point_values(
                    point,
                    "velocities",
                    source_joint_names,
                    target_joint_names,
                )
                accelerations = _remap_trajectory_point_values(
                    point,
                    "accelerations",
                    source_joint_names,
                    target_joint_names,
                )
            except ValueError as exc:
                return {
                    "success": False,
                    "message": f"UR5e MoveIt planned trajectory segment {segment_index}: {exc}.",
                }
            if len(positions) != len(target_joint_names):
                return {
                    "success": False,
                    "message": (
                        f"UR5e MoveIt planned trajectory segment {segment_index} point "
                        "length does not match hardware joint names."
                    ),
                }
            if (
                stitched_points
                and point_index == 0
                and last_positions is not None
                and _positions_match(positions, last_positions)
            ):
                continue

            raw_time = max(0.0, float(point.get("time") or 0.0))
            relative_time = max(0.0, raw_time - base_time)
            scaled_time = segment_start_time + relative_time * UR5E_TEACH_REPLAY_TIME_SCALE
            if stitched_points and scaled_time <= last_time:
                scaled_time = last_time + UR5E_TEACH_REPLAY_MIN_POINT_STEP_SEC

            item: dict[str, Any] = {
                "positions": positions,
                "time": scaled_time,
            }
            if velocities:
                has_velocities = True
                item["velocities"] = [
                    float(value) / UR5E_TEACH_REPLAY_TIME_SCALE
                    for value in velocities
                ]
            if accelerations:
                has_accelerations = True
                scale_sq = UR5E_TEACH_REPLAY_TIME_SCALE * UR5E_TEACH_REPLAY_TIME_SCALE
                item["accelerations"] = [float(value) / scale_sq for value in accelerations]
            stitched_points.append(item)
            last_positions = positions
            last_time = scaled_time

    if not stitched_points:
        return {
            "success": False,
            "message": "UR5e MoveIt planned trajectory stitching produced no points.",
        }

    final_point = dict(stitched_points[-1])
    final_positions = [float(value) for value in list(final_point.get("positions") or [])]
    if has_velocities:
        final_point["velocities"] = [0.0] * len(target_joint_names)
    if has_accelerations:
        final_point["accelerations"] = [0.0] * len(target_joint_names)
    stitched_points[-1] = final_point

    hold_point: dict[str, Any] = {
        "positions": list(final_positions),
        "time": float(stitched_points[-1]["time"]) + UR5E_TEACH_REPLAY_FINAL_HOLD_SEC,
    }
    if has_velocities:
        hold_point["velocities"] = [0.0] * len(target_joint_names)
    if has_accelerations:
        hold_point["accelerations"] = [0.0] * len(target_joint_names)
    stitched_points.append(hold_point)

    return {
        "success": True,
        "joint_names": target_joint_names,
        "points": stitched_points,
        "segments": segment_count,
        "final_hold_sec": UR5E_TEACH_REPLAY_FINAL_HOLD_SEC,
        "time_scale": UR5E_TEACH_REPLAY_TIME_SCALE,
        "has_velocities": has_velocities,
        "has_accelerations": has_accelerations,
        "final_time": float(stitched_points[-1]["time"]),
    }


def _prepare_ur5e_teach_replay_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    joint_names = [str(name) for name in list(trajectory.get("joint_names") or [])]
    raw_points = [dict(point) for point in list(trajectory.get("points") or [])]
    if not joint_names or not raw_points:
        return {
            "success": False,
            "message": "UR5e planned trajectory has no joint names or points.",
        }

    has_velocities = any("velocities" in point for point in raw_points)
    has_accelerations = any("accelerations" in point for point in raw_points)
    prepared_points: list[dict[str, Any]] = []
    last_time = -UR5E_TEACH_REPLAY_MIN_POINT_STEP_SEC
    original_final_time = 0.0
    for point in raw_points:
        positions = [float(value) for value in list(point.get("positions") or [])]
        if len(positions) != len(joint_names):
            return {
                "success": False,
                "message": "UR5e planned trajectory point length does not match joint names.",
            }
        original_time = max(0.0, float(point.get("time") or 0.0))
        original_final_time = max(original_final_time, original_time)
        scaled_time = original_time * UR5E_TEACH_REPLAY_TIME_SCALE
        if prepared_points and scaled_time <= last_time:
            scaled_time = last_time + UR5E_TEACH_REPLAY_MIN_POINT_STEP_SEC
        item: dict[str, Any] = {
            "positions": positions,
            "time": scaled_time,
        }
        velocities = list(point.get("velocities") or [])
        accelerations = list(point.get("accelerations") or [])
        if velocities:
            item["velocities"] = [
                float(value) / UR5E_TEACH_REPLAY_TIME_SCALE
                for value in velocities
            ]
        if accelerations:
            scale_sq = UR5E_TEACH_REPLAY_TIME_SCALE * UR5E_TEACH_REPLAY_TIME_SCALE
            item["accelerations"] = [float(value) / scale_sq for value in accelerations]
        prepared_points.append(item)
        last_time = scaled_time

    final_point = dict(prepared_points[-1])
    final_positions = [float(value) for value in list(final_point.get("positions") or [])]
    if has_velocities:
        final_point["velocities"] = [0.0] * len(joint_names)
    if has_accelerations:
        final_point["accelerations"] = [0.0] * len(joint_names)
    prepared_points[-1] = final_point

    hold_point: dict[str, Any] = {
        "positions": list(final_positions),
        "time": float(prepared_points[-1]["time"]) + UR5E_TEACH_REPLAY_FINAL_HOLD_SEC,
    }
    if has_velocities:
        hold_point["velocities"] = [0.0] * len(joint_names)
    if has_accelerations:
        hold_point["accelerations"] = [0.0] * len(joint_names)
    prepared_points.append(hold_point)

    return {
        "success": True,
        "joint_names": joint_names,
        "points": prepared_points,
        "original_final_time": original_final_time,
        "scaled_final_time": float(prepared_points[-2]["time"]),
        "final_hold_sec": UR5E_TEACH_REPLAY_FINAL_HOLD_SEC,
        "time_scale": UR5E_TEACH_REPLAY_TIME_SCALE,
        "has_velocities": has_velocities,
        "has_accelerations": has_accelerations,
    }


def _ur5e_final_joint_error_detail(
    domain_id: int,
    joint_names: list[str],
    target_positions: list[float],
) -> str:
    snapshot_result = _read_snapshot(
        domain_id,
        "ur5e",
        "hardware",
        UR5E_FINAL_ERROR_SNAPSHOT_TIMEOUT_SEC,
    )
    if not snapshot_result.get("success"):
        return f"final_joint_error_deg=unavailable ({snapshot_result.get('message') or 'snapshot failed'})"
    hardware_names, hardware_positions, missing = _resolve_hardware_positions(
        dict(snapshot_result.get("snapshot") or {}),
        "ur5e",
    )
    if missing:
        return f"final_joint_error_deg=unavailable (missing joints: {', '.join(str(name) for name in missing)})"
    current_by_name = {
        str(name): float(position)
        for name, position in zip(hardware_names, hardware_positions)
    }
    target_by_name = {
        str(name): float(position)
        for name, position in zip(joint_names, target_positions)
    }
    deltas = [
        math.degrees(_angular_delta(current_by_name[name], target_by_name[name]))
        for name in joint_names
        if name in current_by_name and name in target_by_name
    ]
    if not deltas:
        return "final_joint_error_deg=unavailable (no matching target joints)"
    return f"final_joint_error_deg={max(deltas):.3f}"


def _plan_move_group_joint_goal_worker(
    domain_id: int,
    group_name: str,
    joint_names: list[str],
    start_positions: list[float],
    target_positions: list[float],
    waypoint_index: int,
    timeout_sec: float,
    plan_only: bool,
    acceptance_timeout_sec: float,
    result_queue: mp.Queue,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import Constraints, JointConstraint
    from rclpy.action import ActionClient
    from rclpy.node import Node

    def _wait_future(node: Any, future: Any, wait_timeout_sec: float) -> bool:
        deadline = time.time() + max(0.5, float(wait_timeout_sec))
        while rclpy.ok() and time.time() < deadline and not future.done():
            rclpy.spin_once(node, timeout_sec=0.1)
        return bool(future.done())

    action_name = MOVE_GROUP_ACTION_NAME
    node = None
    try:
        node = Node(
            "digital_twin_move_group_plan"
            if plan_only
            else "digital_twin_move_group_plan_and_execute"
        )
        client = ActionClient(node, MoveGroup, action_name)
        if not client.wait_for_server(timeout_sec=max(0.5, min(8.0, float(timeout_sec)))):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action server unavailable.",
                    "action_name": action_name,
                    "group_name": group_name,
                    "waypoint_index": int(waypoint_index),
                    "joint_names": list(joint_names),
                }
            )
            return

        goal = MoveGroup.Goal()
        goal.request.group_name = str(group_name)
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = MOVE_GROUP_ALLOWED_PLANNING_TIME_SEC
        goal.request.max_velocity_scaling_factor = MOVE_GROUP_REPLAY_VELOCITY_SCALING
        goal.request.max_acceleration_scaling_factor = MOVE_GROUP_REPLAY_ACCELERATION_SCALING
        goal.request.start_state.joint_state.name = list(joint_names)
        goal.request.start_state.joint_state.position = [float(v) for v in start_positions]
        goal.request.start_state.is_diff = False

        constraints = Constraints()
        constraints.name = f"{group_name}_waypoint_{int(waypoint_index)}"
        for joint_name, position in zip(joint_names, target_positions):
            constraint = JointConstraint()
            constraint.joint_name = str(joint_name)
            constraint.position = float(position)
            constraint.tolerance_above = MOVE_GROUP_JOINT_TOLERANCE_RAD
            constraint.tolerance_below = MOVE_GROUP_JOINT_TOLERANCE_RAD
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = bool(plan_only)
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        send_future = client.send_goal_async(goal)
        acceptance_timeout = max(8.0, float(acceptance_timeout_sec))
        if not _wait_future(node, send_future, acceptance_timeout):
            result_queue.put(
                {
                    "success": False,
                    "message": (
                        f"{action_name}: action goal acceptance timed out; "
                        f"acceptance_timeout_sec={acceptance_timeout:.3f}."
                    ),
                    "action_name": action_name,
                    "group_name": group_name,
                    "waypoint_index": int(waypoint_index),
                    "joint_names": list(joint_names),
                    "acceptance_timeout_sec": acceptance_timeout,
                }
            )
            return

        goal_handle = send_future.result()
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action goal rejected.",
                    "action_name": action_name,
                    "group_name": group_name,
                    "waypoint_index": int(waypoint_index),
                    "joint_names": list(joint_names),
                }
            )
            return

        result_future = goal_handle.get_result_async()
        if not _wait_future(node, result_future, timeout_sec):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action accepted; planning result timed out.",
                    "action_name": action_name,
                    "group_name": group_name,
                    "waypoint_index": int(waypoint_index),
                    "joint_names": list(joint_names),
                }
            )
            return

        result_response = result_future.result()
        status = getattr(result_response, "status", None)
        label = _goal_status_label(status)
        action_result = getattr(result_response, "result", None)
        error_code = getattr(action_result, "error_code", None)
        error_val = getattr(error_code, "val", error_code)
        trajectory = _serialize_joint_trajectory(getattr(action_result, "planned_trajectory", None))
        planning_time = float(getattr(action_result, "planning_time", 0.0) or 0.0)
        status_int = int(status) if status is not None else None
        if plan_only:
            success = (
                status_int == 4
                and error_val is not None
                and int(error_val) == 1
                and bool(trajectory["joint_names"])
                and bool(trajectory["points"])
            )
        else:
            success = status_int == 4 and error_val is not None and int(error_val) == 1
        mode_label = "planned" if plan_only else "plan_and_execute"
        message = (
            f"{action_name}: {mode_label}; action accepted; action {label}; "
            f"moveit_error_code={int(error_val) if error_val is not None else 'unknown'}; "
            f"group={group_name}; waypoint={int(waypoint_index)}; "
            f"planning_time={planning_time:.3f}; points={len(trajectory['points'])}."
        )
        result_queue.put(
            {
                "success": success,
                "message": message,
                "status": status_int,
                "status_label": label,
                "error_code": int(error_val) if error_val is not None else None,
                "mode": mode_label,
                "planning_time": planning_time,
                "trajectory": trajectory,
                "action_name": action_name,
                "group_name": group_name,
                "waypoint_index": int(waypoint_index),
                "joint_names": list(joint_names),
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "success": False,
                "message": f"{action_name}: {exc}",
                "action_name": action_name,
                "group_name": group_name,
                "waypoint_index": int(waypoint_index),
                "joint_names": list(joint_names),
            }
        )
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _plan_move_group_joint_goal(
    domain_id: int,
    group_name: str,
    joint_names: list[str],
    start_positions: list[float],
    target_positions: list[float],
    *,
    waypoint_index: int,
    timeout_sec: float = MOVE_GROUP_PLAN_TIMEOUT_SEC,
    acceptance_timeout_sec: float = MOVE_GROUP_GOAL_ACCEPTANCE_TIMEOUT_SEC,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_plan_move_group_joint_goal_worker,
        args=(
            domain_id,
            str(group_name),
            list(joint_names),
            [float(v) for v in start_positions],
            [float(v) for v in target_positions],
            int(waypoint_index),
            float(timeout_sec),
            True,
            float(acceptance_timeout_sec),
            result_queue,
        ),
        daemon=True,
    )
    proc.start()
    proc.join(
        max(1.0, float(timeout_sec))
        + max(8.0, float(acceptance_timeout_sec))
        + 10.0
    )
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {
            "success": False,
            "message": f"{MOVE_GROUP_ACTION_NAME}: planning worker timed out.",
            "action_name": MOVE_GROUP_ACTION_NAME,
            "group_name": str(group_name),
            "waypoint_index": int(waypoint_index),
            "joint_names": list(joint_names),
        }
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {
            "success": False,
            "message": f"{MOVE_GROUP_ACTION_NAME}: planning worker returned no data.",
            "action_name": MOVE_GROUP_ACTION_NAME,
            "group_name": str(group_name),
            "waypoint_index": int(waypoint_index),
            "joint_names": list(joint_names),
        }


def _execute_move_group_joint_goal(
    domain_id: int,
    group_name: str,
    joint_names: list[str],
    start_positions: list[float],
    target_positions: list[float],
    *,
    waypoint_index: int,
    timeout_sec: float,
    acceptance_timeout_sec: float = MOVE_GROUP_GOAL_ACCEPTANCE_TIMEOUT_SEC,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_plan_move_group_joint_goal_worker,
        args=(
            domain_id,
            str(group_name),
            list(joint_names),
            [float(v) for v in start_positions],
            [float(v) for v in target_positions],
            int(waypoint_index),
            float(timeout_sec),
            False,
            float(acceptance_timeout_sec),
            result_queue,
        ),
        daemon=True,
    )
    proc.start()
    proc.join(
        max(1.0, float(timeout_sec))
        + max(8.0, float(acceptance_timeout_sec))
        + 10.0
    )
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {
            "success": False,
            "message": f"{MOVE_GROUP_ACTION_NAME}: plan_and_execute worker timed out.",
            "action_name": MOVE_GROUP_ACTION_NAME,
            "group_name": str(group_name),
            "waypoint_index": int(waypoint_index),
            "joint_names": list(joint_names),
            "mode": "plan_and_execute",
        }
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {
            "success": False,
            "message": f"{MOVE_GROUP_ACTION_NAME}: plan_and_execute worker returned no data.",
            "action_name": MOVE_GROUP_ACTION_NAME,
            "group_name": str(group_name),
            "waypoint_index": int(waypoint_index),
            "joint_names": list(joint_names),
            "mode": "plan_and_execute",
        }


def _xarm_gripper_service_candidates(suffix: str) -> list[str]:
    return [
        f"/xarm6/xarm/{suffix}",
        f"/xarm6/{suffix}",
        f"/xarm/{suffix}",
        f"/{suffix}",
    ]


def _xarm_gripper_joint_to_pulse(joint_position: float) -> int:
    gripper = ROBOTS["xarm6"].get("gripper") or {}
    open_pos = float(gripper.get("open_position", 0.0))
    close_pos = float(gripper.get("close_position", 0.85))
    open_pulse = float(gripper.get("open_pulse", 850.0))
    close_pulse = float(gripper.get("close_pulse", 0.0))
    denom = open_pos - close_pos
    if abs(denom) < 1e-9:
        return int(round(close_pulse))
    ratio = (float(joint_position) - close_pos) / denom
    ratio = min(max(ratio, 0.0), 1.0)
    return int(round(close_pulse + (open_pulse - close_pulse) * ratio))


def _wait_xarm_gripper_service_worker(
    domain_id: int,
    timeout_sec: float,
    result_queue: mp.Queue,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from rclpy.node import Node
    from xarm_msgs.srv import GripperMove

    node = None
    try:
        node = Node("digital_twin_xarm_gripper_preflight")
        candidates = list(_xarm_gripper_service_candidates("set_gripper_position"))
        try:
            for service_name, _types in node.get_service_names_and_types():
                if service_name.endswith("/set_gripper_position") and service_name not in candidates:
                    candidates.append(service_name)
        except Exception:
            pass
        clients = [(name, node.create_client(GripperMove, name)) for name in candidates]
        deadline = time.time() + max(0.5, float(timeout_sec))
        while rclpy.ok() and time.time() < deadline:
            for name, client in clients:
                if client.service_is_ready():
                    result_queue.put(
                        {
                            "success": True,
                            "message": f"{name}: service available.",
                            "service": name,
                        }
                    )
                    return
            rclpy.spin_once(node, timeout_sec=0.1)
            for name, client in clients:
                if client.wait_for_service(timeout_sec=0.0):
                    result_queue.put(
                        {
                            "success": True,
                            "message": f"{name}: service available.",
                            "service": name,
                        }
                    )
                    return
        result_queue.put(
            {
                "success": False,
                "message": "xarm6 set_gripper_position service unavailable.",
            }
        )
    except Exception as exc:
        result_queue.put({"success": False, "message": f"xarm6 gripper preflight: {exc}"})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _wait_xarm_gripper_service(domain_id: int, timeout_sec: float = 5.0) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_wait_xarm_gripper_service_worker,
        args=(domain_id, timeout_sec, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(max(1.0, float(timeout_sec)) + 2.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {"success": False, "message": "xarm6 set_gripper_position service wait timed out."}
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {"success": False, "message": "xarm6 set_gripper_position service wait returned no data."}


def _wait_gripper_command_action_worker(
    domain_id: int,
    action_name: str,
    timeout_sec: float,
    result_queue: mp.Queue,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from control_msgs.action import GripperCommand
    from rclpy.action import ActionClient
    from rclpy.node import Node

    node = None
    try:
        node = Node("digital_twin_gripper_command_preflight")
        client = ActionClient(node, GripperCommand, action_name)
        if not client.wait_for_server(timeout_sec=max(0.5, float(timeout_sec))):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action server unavailable.",
                }
            )
            return
        result_queue.put(
            {
                "success": True,
                "message": f"{action_name}: action server available.",
                "action": action_name,
            }
        )
    except Exception as exc:
        result_queue.put({"success": False, "message": f"{action_name}: {exc}"})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _wait_gripper_command_action(
    domain_id: int,
    action_name: str,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_wait_gripper_command_action_worker,
        args=(domain_id, action_name, timeout_sec, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(max(1.0, float(timeout_sec)) + 2.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {"success": False, "message": f"{action_name}: action server wait timed out."}
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {"success": False, "message": f"{action_name}: action server wait returned no data."}


def _wait_xarm_gripper_endpoint(domain_id: int, timeout_sec: float = 5.0) -> dict[str, Any]:
    service_result = _wait_xarm_gripper_service(domain_id, timeout_sec=timeout_sec)
    if service_result.get("success"):
        out = dict(service_result)
        out["method"] = "service"
        return out
    action_name = str(dict(ROBOTS["xarm6"].get("gripper") or {}).get("hardware_action") or "").strip()
    action_result = _wait_gripper_command_action(domain_id, action_name, timeout_sec=timeout_sec)
    if action_result.get("success"):
        out = dict(action_result)
        out["method"] = "action"
        out["fallback_error"] = str(service_result.get("message") or "")
        return out
    return {
        "success": False,
        "message": (
            f"xarm6 gripper unavailable: {str(service_result.get('message') or '')}; "
            f"{str(action_result.get('message') or '')}"
        ).strip("; "),
    }


def _publish_xarm_gripper_service_sequence_worker(
    domain_id: int,
    points: list[dict[str, Any]],
    result_timeout_sec: float,
    result_queue: mp.Queue,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from rclpy.node import Node
    from xarm_msgs.srv import GripperMove, SetFloat32, SetInt16

    def _candidate_service_names(node: Any, suffix: str) -> list[str]:
        names = list(_xarm_gripper_service_candidates(suffix))
        try:
            for service_name, _types in node.get_service_names_and_types():
                if service_name.endswith(f"/{suffix}") and service_name not in names:
                    names.append(service_name)
        except Exception:
            pass
        return names

    def _service_client(node: Any, srv_type: Any, suffix: str, timeout_sec: float) -> tuple[str, Any] | tuple[None, None]:
        clients = [(name, node.create_client(srv_type, name)) for name in _candidate_service_names(node, suffix)]
        deadline = time.time() + max(0.1, float(timeout_sec))
        while rclpy.ok() and time.time() < deadline:
            for name, client in clients:
                if client.service_is_ready():
                    return name, client
            rclpy.spin_once(node, timeout_sec=0.05)
            for name, client in clients:
                if client.wait_for_service(timeout_sec=0.0):
                    return name, client
        return None, None

    def _call(node: Any, client: Any, request: Any, timeout_sec: float) -> tuple[Any, str]:
        try:
            future = client.call_async(request)
        except Exception as exc:
            return None, str(exc)
        deadline = time.time() + max(0.1, float(timeout_sec))
        while rclpy.ok() and time.time() < deadline and not future.done():
            rclpy.spin_once(node, timeout_sec=0.05)
        if not future.done():
            return None, "timeout"
        response = future.result()
        if response is None:
            return None, "service failed"
        return response, ""

    node = None
    try:
        node = Node("digital_twin_xarm_gripper_replay")
        service_name, gripper_client = _service_client(node, GripperMove, "set_gripper_position", 5.0)
        if gripper_client is None:
            result_queue.put(
                {
                    "success": False,
                    "message": "xarm6 set_gripper_position service unavailable.",
                }
            )
            return

        for suffix, value in (("set_gripper_enable", 1), ("set_gripper_mode", 0)):
            _name, client = _service_client(node, SetInt16, suffix, 0.3)
            if client is None:
                continue
            req = SetInt16.Request()
            req.data = int(value)
            _call(node, client, req, timeout_sec=1.0)

        _name, speed_client = _service_client(node, SetFloat32, "set_gripper_speed", 0.3)
        if speed_client is not None:
            req = SetFloat32.Request()
            req.data = 2000.0
            _call(node, speed_client, req, timeout_sec=1.0)

        started_at = time.time()
        sent = 0
        for entry in points:
            target_time = float(entry.get("time") or 0.0)
            while rclpy.ok() and time.time() - started_at < target_time:
                rclpy.spin_once(node, timeout_sec=0.05)
            positions = list(entry.get("positions") or [])
            if not positions:
                continue
            pulse = _xarm_gripper_joint_to_pulse(float(positions[0]))
            req = GripperMove.Request()
            req.pos = float(pulse)
            req.wait = False
            req.timeout = 2.0
            response, error = _call(node, gripper_client, req, timeout_sec=2.5)
            if error:
                result_queue.put(
                    {
                        "success": False,
                        "message": f"{service_name}: {error}",
                    }
                )
                return
            ret = int(getattr(response, "ret", -1))
            msg = str(getattr(response, "message", "") or "").strip()
            if ret != 0:
                detail = f"ret={ret}"
                if msg:
                    detail += f" {msg}"
                result_queue.put(
                    {
                        "success": False,
                        "message": f"{service_name}: {detail}",
                    }
                )
                return
            sent += 1

        result_queue.put(
            {
                "success": True,
                "message": f"{service_name}: gripper replay sent {sent} commands.",
                "commands": sent,
            }
        )
    except Exception as exc:
        result_queue.put({"success": False, "message": f"xarm6 gripper replay: {exc}"})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _publish_xarm_gripper_service_sequence(
    domain_id: int,
    points: list[dict[str, Any]],
    join_timeout_sec: float = 8.0,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_publish_xarm_gripper_service_sequence_worker,
        args=(domain_id, list(points), join_timeout_sec, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(max(1.0, float(join_timeout_sec)) + 4.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {"success": False, "message": "xarm6 gripper replay timed out."}
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {"success": False, "message": "xarm6 gripper replay returned no data."}


def _publish_gripper_command_action_sequence_worker(
    domain_id: int,
    action_name: str,
    points: list[dict[str, Any]],
    result_timeout_sec: float,
    result_queue: mp.Queue,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from control_msgs.action import GripperCommand
    from rclpy.action import ActionClient
    from rclpy.node import Node

    def _wait_future(node: Any, future: Any, timeout_sec: float) -> bool:
        deadline = time.time() + max(0.5, float(timeout_sec))
        while rclpy.ok() and time.time() < deadline and not future.done():
            rclpy.spin_once(node, timeout_sec=0.05)
        return bool(future.done())

    node = None
    try:
        node = Node("digital_twin_gripper_command_replay")
        client = ActionClient(node, GripperCommand, action_name)
        if not client.wait_for_server(timeout_sec=5.0):
            result_queue.put(
                {
                    "success": False,
                    "message": f"{action_name}: action server unavailable.",
                }
            )
            return

        started_at = time.time()
        sent = 0
        for entry in points:
            target_time = float(entry.get("time") or 0.0)
            while rclpy.ok() and time.time() - started_at < target_time:
                rclpy.spin_once(node, timeout_sec=0.05)
            positions = list(entry.get("positions") or [])
            if not positions:
                continue
            goal = GripperCommand.Goal()
            goal.command.position = float(positions[0])
            goal.command.max_effort = 0.0
            send_future = client.send_goal_async(goal)
            if not _wait_future(node, send_future, 2.0):
                result_queue.put(
                    {
                        "success": False,
                        "message": f"{action_name}: goal acceptance timed out.",
                    }
                )
                return
            goal_handle = send_future.result()
            if goal_handle is None or not getattr(goal_handle, "accepted", False):
                result_queue.put(
                    {
                        "success": False,
                        "message": f"{action_name}: goal rejected.",
                    }
                )
                return
            result_future = goal_handle.get_result_async()
            _wait_future(node, result_future, min(0.5, max(0.1, float(result_timeout_sec))))
            sent += 1

        result_queue.put(
            {
                "success": True,
                "message": f"{action_name}: gripper action replay accepted {sent} goals.",
                "commands": sent,
            }
        )
    except Exception as exc:
        result_queue.put({"success": False, "message": f"{action_name}: {exc}"})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _publish_gripper_command_action_sequence(
    domain_id: int,
    action_name: str,
    points: list[dict[str, Any]],
    join_timeout_sec: float = 8.0,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_publish_gripper_command_action_sequence_worker,
        args=(domain_id, action_name, list(points), join_timeout_sec, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(max(1.0, float(join_timeout_sec)) + 4.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {"success": False, "message": f"{action_name}: gripper action replay timed out."}
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {"success": False, "message": f"{action_name}: gripper action replay returned no data."}


def _publish_xarm_gripper_sequence(
    domain_id: int,
    points: list[dict[str, Any]],
    join_timeout_sec: float = 8.0,
) -> dict[str, Any]:
    service_result = _publish_xarm_gripper_service_sequence(
        domain_id,
        points,
        join_timeout_sec=join_timeout_sec,
    )
    if service_result.get("success"):
        return service_result
    action_name = str(dict(ROBOTS["xarm6"].get("gripper") or {}).get("hardware_action") or "").strip()
    action_result = _publish_gripper_command_action_sequence(
        domain_id,
        action_name,
        points,
        join_timeout_sec=join_timeout_sec,
    )
    if action_result.get("success"):
        out = dict(action_result)
        out["fallback_error"] = str(service_result.get("message") or "")
        return out
    return {
        "success": False,
        "message": (
            f"{str(service_result.get('message') or '')}; "
            f"{str(action_result.get('message') or '')}"
        ).strip("; "),
    }


def _set_gazebo_model_configuration_worker(
    domain_id: int,
    model_name: str,
    joint_names: list[str],
    joint_positions: list[float],
    result_queue: mp.Queue,
) -> None:
    rclpy = _init_ros_domain(domain_id)
    from gazebo_msgs.srv import SetModelConfiguration
    from rclpy.node import Node

    node = None
    try:
        node = Node("digital_twin_set_model_configuration")
        client = node.create_client(SetModelConfiguration, "/gazebo/set_model_configuration")
        if not client.wait_for_service(timeout_sec=6.0):
            result_queue.put(
                {
                    "success": False,
                    "message": "/gazebo/set_model_configuration not available.",
                }
            )
            return

        req = SetModelConfiguration.Request()
        req.model_name = str(model_name or "dual_robot")
        req.urdf_param_name = ""
        req.joint_names = list(joint_names)
        req.joint_positions = [float(value) for value in joint_positions]
        future = client.call_async(req)
        deadline = time.time() + 8.0
        while rclpy.ok() and time.time() < deadline and not future.done():
            rclpy.spin_once(node, timeout_sec=0.1)
        if not future.done():
            result_queue.put(
                {
                    "success": False,
                    "message": "/gazebo/set_model_configuration timed out.",
                }
            )
            return

        response = future.result()
        success = bool(getattr(response, "success", False))
        status_message = str(getattr(response, "status_message", "") or "")
        result_queue.put(
            {
                "success": success,
                "message": status_message or (
                    "gazebo model configuration set."
                    if success
                    else "gazebo model configuration failed."
                ),
            }
        )
    except Exception as exc:
        result_queue.put({"success": False, "message": str(exc)})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def _set_gazebo_model_configuration(
    domain_id: int,
    model_name: str,
    joint_names: list[str],
    joint_positions: list[float],
    join_timeout_sec: float = 10.0,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_set_gazebo_model_configuration_worker,
        args=(domain_id, model_name, list(joint_names), list(joint_positions), result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(join_timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return {"success": False, "message": "gazebo model configuration timed out."}
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {"success": False, "message": "gazebo model configuration returned no data."}


def _read_snapshot(
    domain_id: int,
    robot: str,
    source: str,
    timeout_sec: float,
    *,
    include_world_tool_pose: bool = False,
) -> dict[str, Any]:
    attempts = 2
    for attempt in range(attempts):
        result_queue: mp.Queue = mp.Queue(maxsize=1)
        proc = mp.Process(
            target=_joint_state_snapshot_worker,
            args=(
                domain_id,
                robot,
                source,
                timeout_sec,
                result_queue,
                include_world_tool_pose,
            ),
            daemon=True,
        )
        try:
            proc.start()
            proc.join(timeout_sec + 2.0)
        except Exception as exc:
            topics = ", ".join(_joint_state_topics(robot, source))
            return {
                "success": False,
                "message": (
                    f"{source} /joint_states snapshot process failed for {robot}: {exc}; "
                    f"ROS_DOMAIN_ID={int(domain_id)}; topics={topics}"
                ),
            }
        if proc.is_alive():
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                result = None
            proc.terminate()
            proc.join(timeout=1.0)
            if result is not None:
                return result
            return {"success": False, "message": f"{source} /joint_states timed out."}
        try:
            return result_queue.get(timeout=0.5)
        except queue.Empty:
            if attempt + 1 < attempts:
                continue
            topics = ", ".join(_joint_state_topics(robot, source))
            return {
                "success": False,
                "message": (
                    f"{source} /joint_states returned no data for {robot} after {attempts} attempts; "
                    f"ROS_DOMAIN_ID={int(domain_id)}; topics={topics}"
                ),
            }
    return {"success": False, "message": f"{source} /joint_states returned no data."}


def _publish_hardware_trajectory(
    domain_id: int,
    robot: str,
    joint_names: list[str],
    positions: list[float],
) -> dict[str, Any]:
    if str(robot or "").strip().lower() == "ur5e":
        action_name = str(ROBOTS[robot].get("hardware_trajectory_action") or "").strip()
        return _publish_follow_joint_trajectory_action(
            domain_id,
            action_name,
            joint_names,
            [{"positions": positions, "time": 2.0}],
            join_timeout_sec=8.0,
            goal_time_tolerance_sec=2.0,
        )
    return _publish_trajectory(
        domain_id,
        ROBOTS[robot]["trajectory_topics"],
        joint_names,
        [{"positions": positions, "time": 2.0}],
    )


def run_mirror(args: argparse.Namespace) -> int:
    status_file = Path(args.status_file)
    direction_file = Path(args.direction_file)
    _write_status(
        status_file,
        target=args.target,
        state="starting",
        direction=_direction(direction_file),
        message="starting digital twin sync.",
    )
    updates: mp.Queue = mp.Queue(maxsize=4)
    hardware_proc = mp.Process(
        target=_hardware_joint_state_worker,
        args=(int(args.hardware_domain_id), args.robot, updates),
        daemon=True,
    )
    gazebo_proc = mp.Process(
        target=_gazebo_mirror_worker,
        args=(
            int(args.gazebo_domain_id),
            args.robot,
            args.target,
            status_file,
            direction_file,
            updates,
        ),
        daemon=True,
    )
    hardware_proc.start()
    gazebo_proc.start()

    stopping = False

    def _stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while not stopping:
            if not hardware_proc.is_alive():
                _write_status(
                    status_file,
                    target=args.target,
                    state="error",
                    direction=_direction(direction_file),
                    message="hardware joint-state worker exited.",
                    last_error="hardware joint-state worker exited.",
                )
                return 2
            if not gazebo_proc.is_alive():
                _write_status(
                    status_file,
                    target=args.target,
                    state="error",
                    direction=_direction(direction_file),
                    message="gazebo configuration worker exited.",
                    last_error="gazebo configuration worker exited.",
                )
                return 2
            time.sleep(0.25)
    finally:
        for proc in (hardware_proc, gazebo_proc):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        _write_status(
            status_file,
            target=args.target,
            state="stopped",
            direction=_direction(direction_file),
            message="digital twin sync stopped.",
        )
    return 0


def run_apply_gazebo_to_hardware(args: argparse.Namespace) -> int:
    status_file = Path(args.status_file)
    direction_file = Path(args.direction_file)
    if _direction(direction_file) != "gazebo -> hardware":
        result = {
            "success": False,
            "message": "direction must be gazebo -> hardware.",
        }
        print(json.dumps(result))
        return 3

    _write_status(
        status_file,
        target=args.target,
        state="checking",
        direction="gazebo -> hardware",
        message="checking gazebo and hardware joint states.",
    )
    gazebo_result = _read_snapshot(int(args.gazebo_domain_id), args.robot, "gazebo", 5.0)
    if not gazebo_result.get("success"):
        result = {"success": False, "message": str(gazebo_result.get("message") or "missing gazebo state")}
        _write_status(status_file, target=args.target, state="blocked", direction="gazebo -> hardware", **result)
        print(json.dumps(result))
        return 4
    hardware_result = _read_snapshot(int(args.hardware_domain_id), args.robot, "hardware", 5.0)
    if not hardware_result.get("success"):
        result = {"success": False, "message": str(hardware_result.get("message") or "missing hardware state")}
        _write_status(status_file, target=args.target, state="blocked", direction="gazebo -> hardware", **result)
        print(json.dumps(result))
        return 5

    gazebo_snapshot = dict(gazebo_result.get("snapshot") or {})
    hardware_snapshot = dict(hardware_result.get("snapshot") or {})
    _gazebo_names, gazebo_positions, gazebo_missing = _resolve_gazebo_positions(gazebo_snapshot, args.robot)
    hardware_names, hardware_positions, hardware_missing = _resolve_hardware_positions(hardware_snapshot, args.robot)
    if gazebo_missing or hardware_missing:
        missing = ", ".join(gazebo_missing + hardware_missing)
        result = {"success": False, "message": f"missing joints: {missing}"}
        _write_status(status_file, target=args.target, state="blocked", direction="gazebo -> hardware", **result)
        print(json.dumps(result))
        return 6

    deltas = [_angular_delta(gz, hw) for gz, hw in zip(gazebo_positions, hardware_positions)]
    max_delta = max(deltas) if deltas else 0.0
    max_delta_deg = math.degrees(max_delta)
    guard_deg = float(args.max_joint_delta_deg)
    if max_delta_deg > guard_deg:
        result = {
            "success": False,
            "message": f"blocked: max joint delta {max_delta_deg:.2f} deg exceeds {guard_deg:.2f} deg.",
            "max_joint_delta_deg": max_delta_deg,
        }
        _write_status(status_file, target=args.target, state="blocked", direction="gazebo -> hardware", **result)
        print(json.dumps(result))
        return 7

    publish_result = _publish_hardware_trajectory(
        int(args.hardware_domain_id),
        args.robot,
        hardware_names,
        gazebo_positions,
    )
    result = {
        "success": bool(publish_result.get("success")),
        "message": str(publish_result.get("message") or ""),
        "max_joint_delta_deg": max_delta_deg,
    }
    _write_status(
        status_file,
        target=args.target,
        state="applied" if result["success"] else "blocked",
        direction="gazebo -> hardware",
        **result,
    )
    print(json.dumps(result))
    return 0 if result["success"] else 8


def run_initialize_gazebo_from_hardware(args: argparse.Namespace) -> int:
    """Copy one robot's current hardware arm/gripper pose into passive Gazebo once."""
    if args.robot not in ROBOTS:
        print(json.dumps({"success": False, "message": f"unsupported robot: {args.robot}"}))
        return 3

    status_file = Path(args.status_file) if str(args.status_file or "").strip() else None

    def write_status(**payload: Any) -> None:
        if status_file is not None:
            _write_status(status_file, target=args.target, **payload)

    write_status(
        state="initializing",
        direction="hardware -> gazebo",
        message="initializing gazebo pose from hardware.",
        last_error="",
    )

    hardware_result = _read_snapshot(
        int(args.hardware_domain_id),
        args.robot,
        "hardware",
        HARDWARE_SNAPSHOT_TIMEOUT_SEC,
    )
    if not hardware_result.get("success"):
        result = {
            "success": False,
            "message": str(hardware_result.get("message") or "missing hardware state"),
        }
        write_status(
            state="blocked",
            direction="hardware -> gazebo",
            message=result["message"],
            last_error=result["message"],
        )
        print(json.dumps(result))
        return 4

    hardware_snapshot = dict(hardware_result.get("snapshot") or {})
    _hardware_names, hardware_positions, hardware_missing = _resolve_hardware_positions(
        hardware_snapshot,
        args.robot,
    )
    if hardware_missing:
        result = {
            "success": False,
            "message": f"missing hardware joints: {', '.join(hardware_missing)}",
        }
        write_status(
            state="blocked",
            direction="hardware -> gazebo",
            message=result["message"],
            last_error=result["message"],
        )
        print(json.dumps(result))
        return 5

    gripper_result: dict[str, Any] | None = None
    gripper_joint, gripper_position = _resolve_gripper(hardware_snapshot, args.robot)
    gripper_cfg = ROBOTS[args.robot].get("gripper") or {}
    gazebo_gripper_joint = str(gripper_cfg.get("gazebo_joint") or gripper_joint or "")
    gripper_topics = list(gripper_cfg.get("gazebo_trajectory_topics") or [])
    tolerance = float(getattr(args, "init_tolerance_rad", INITIALIZE_GAZEBO_TOLERANCE_RAD))
    attempts = max(1, int(getattr(args, "init_attempts", INITIALIZE_GAZEBO_ATTEMPTS)))

    success = False
    message = ""
    max_delta_rad: float | None = None
    max_delta_joint = ""
    configuration_result: dict[str, Any] | None = None
    arm_result: dict[str, Any] | None = None
    model_name = str(args.model_name or "dual_robot")
    gazebo_joint_names = list(ROBOTS[args.robot]["gazebo_joints"])
    gazebo_joint_positions = list(hardware_positions)
    if gazebo_gripper_joint and gripper_position is not None:
        gazebo_joint_names.append(gazebo_gripper_joint)
        gazebo_joint_positions.append(float(gripper_position))
    for attempt in range(1, attempts + 1):
        configuration_result = _set_gazebo_model_configuration(
            int(args.gazebo_domain_id),
            model_name,
            gazebo_joint_names,
            gazebo_joint_positions,
        )
        arm_result = _publish_trajectory(
            int(args.gazebo_domain_id),
            ROBOTS[args.robot]["gazebo_trajectory_topics"],
            list(ROBOTS[args.robot]["gazebo_joints"]),
            [{"positions": hardware_positions, "time": 0.75}],
            join_timeout_sec=8.0,
        )
        if not arm_result.get("success"):
            configuration_message = (
                str(configuration_result.get("message") or "")
                if configuration_result is not None
                else ""
            )
            message = (
                f"{configuration_message}; "
                f"{str(arm_result.get('message') or 'trajectory publish failed.')}"
            ).strip("; ")
            continue

        if gripper_joint is not None and gripper_position is not None and gripper_topics:
            gripper_result = _publish_trajectory(
                int(args.gazebo_domain_id),
                gripper_topics,
                [gazebo_gripper_joint],
                [{"positions": [float(gripper_position)], "time": 0.75}],
                join_timeout_sec=5.0,
            )

        time.sleep(1.0)
        gazebo_result = _read_snapshot(int(args.gazebo_domain_id), args.robot, "gazebo", 3.0)
        if not gazebo_result.get("success"):
            message = str(gazebo_result.get("message") or "missing gazebo state after publish.")
            continue

        gazebo_snapshot = dict(gazebo_result.get("snapshot") or {})
        _gazebo_names, gazebo_positions, gazebo_missing = _resolve_gazebo_positions(
            gazebo_snapshot,
            args.robot,
        )
        if gazebo_missing:
            message = f"gazebo missing joints after publish: {', '.join(gazebo_missing)}"
            continue

        deltas = [_angular_delta(gazebo, hardware) for gazebo, hardware in zip(gazebo_positions, hardware_positions)]
        max_delta_rad = max(deltas) if deltas else 0.0
        if deltas:
            max_delta_index = max(range(len(deltas)), key=lambda i: deltas[i])
            max_delta_joint = str(ROBOTS[args.robot]["gazebo_joints"][max_delta_index])
        if max_delta_rad <= tolerance:
            success = True
            message = f"gazebo pose initialized from hardware on attempt {attempt}."
            break
        message = (
            f"gazebo pose still differs from hardware by {max_delta_rad:.4f} rad "
            f"on {max_delta_joint or 'unknown joint'} after attempt {attempt}."
        )

    if gripper_result is not None:
        gripper_message = str(gripper_result.get("message") or "")
        if not gripper_result.get("success"):
            message = f"{message} gripper: {gripper_message}".strip()
        elif success:
            message = f"{message} gripper initialized.".strip()

    result = {
        "success": bool(success),
        "message": message or ("gazebo pose initialized from hardware." if success else "gazebo initialization failed."),
        "robot": args.robot,
        "positions": list(hardware_positions),
        "joint_names": list(ROBOTS[args.robot]["gazebo_joints"]),
        "gripper_position": gripper_position,
        "max_joint_delta_rad": max_delta_rad,
        "max_joint_delta_joint": max_delta_joint,
        "attempts": attempts,
        "init_tolerance_rad": tolerance,
        "set_model_configuration": configuration_result,
        "arm_trajectory_publish": arm_result,
    }
    write_status(
        state="initialized" if success else "blocked",
        direction="hardware -> gazebo",
        message=result["message"],
        last_error="" if success else result["message"],
    )
    print(json.dumps(result))
    return 0 if success else 6


def run_snapshot(args: argparse.Namespace) -> int:
    """Read one joint snapshot and print it as JSON (used by 'Capture Waypoint')."""
    source = str(args.source or "gazebo")
    domain_id = int(args.gazebo_domain_id) if source == "gazebo" else int(args.hardware_domain_id)
    snap = _read_snapshot(
        domain_id,
        args.robot,
        source,
        5.0,
        include_world_tool_pose=bool(args.include_world_tool_pose),
    )
    if not snap.get("success"):
        print(json.dumps({"success": False, "message": str(snap.get("message") or "no snapshot")}))
        return 4
    snapshot = dict(snap.get("snapshot") or {})
    if source == "gazebo":
        joint_names, positions, missing = _resolve_gazebo_positions(snapshot, args.robot)
    else:
        joint_names, positions, missing = _resolve_hardware_positions(snapshot, args.robot)
    if missing:
        print(json.dumps({"success": False, "message": f"missing joints: {', '.join(missing)}"}))
        return 5
    gripper_joint, gripper_position = _resolve_gripper(snapshot, args.robot)
    payload = {
        "success": True,
        "source": source,
        "joint_names": joint_names,
        "positions": positions,
        "gripper_joint": gripper_joint,
        "gripper_position": gripper_position,
    }
    if args.include_world_tool_pose:
        payload["pose"] = dict(snap.get("pose") or {})
        payload["world_base_pose"] = dict(snap.get("world_base_pose") or {})
        payload["world_tool0_ready"] = bool(snap.get("world_tool0_ready"))
    print(json.dumps(payload))
    return 0


def run_xarm6_tf_readiness(args: argparse.Namespace) -> int:
    """Report startup readiness for the exact root xArm6 TF input path."""
    result = _xarm6_tf_readiness(
        int(args.hardware_domain_id),
        float(args.tf_readiness_timeout_sec),
    )
    print(json.dumps(result))
    return 0 if result.get("success") else 4


def _paired_replay_robot_plan(
    args: argparse.Namespace,
    recording: dict[str, Any],
    robot: str,
    *,
    need_hardware: bool,
    step: float,
) -> dict[str, Any]:
    robot_meta = dict(dict(recording.get("robots") or {}).get(robot) or {})
    robot_waypoints: list[dict[str, Any]] = []
    for waypoint in list(recording.get("waypoints") or []):
        robot_body = dict(dict(waypoint.get("robots") or {}).get(robot) or {})
        positions = [float(v) for v in (robot_body.get("positions") or [])]
        if not positions:
            return {"success": False, "message": f"{robot} recording has an empty waypoint."}
        item: dict[str, Any] = {"positions": positions}
        if robot_body.get("gripper") is not None:
            item["gripper"] = float(robot_body.get("gripper"))
        robot_waypoints.append(item)

    hardware_names: list[str] = []
    hardware_positions: list[float] = []
    max_delta_deg = 0.0
    max_delta_joint = ""
    approach_time = step
    if need_hardware:
        hardware_result = _read_snapshot(
            int(args.hardware_domain_id),
            robot,
            "hardware",
            HARDWARE_SNAPSHOT_TIMEOUT_SEC,
        )
        if not hardware_result.get("success"):
            return {
                "success": False,
                "message": f"{robot}: {str(hardware_result.get('message') or 'missing hardware state')}",
            }
        hardware_snapshot = dict(hardware_result.get("snapshot") or {})
        hardware_names, hardware_positions, hardware_missing = _resolve_hardware_positions(
            hardware_snapshot,
            robot,
        )
        if hardware_missing:
            return {
                "success": False,
                "message": f"{robot}: missing hardware joints: {', '.join(hardware_missing)}",
            }

        first = [float(v) for v in robot_waypoints[0]["positions"]]
        deltas = [_angular_delta(a, b) for a, b in zip(first, hardware_positions)]
        max_delta_deg = math.degrees(max(deltas)) if deltas else 0.0
        if deltas:
            max_delta_index = max(range(len(deltas)), key=lambda i: deltas[i])
            max_delta_joint = (
                str(hardware_names[max_delta_index])
                if max_delta_index < len(hardware_names)
                else ""
            )
        ceiling_deg = float(args.max_joint_delta_deg)
        if max_delta_deg > ceiling_deg:
            return {
                "success": False,
                "message": (
                    f"{robot}: blocked: first waypoint is {max_delta_deg:.2f} deg from hardware "
                    f"on {max_delta_joint or 'unknown joint'} (ceiling {ceiling_deg:.2f})."
                ),
                "max_joint_delta_deg": max_delta_deg,
                "max_joint_delta_joint": max_delta_joint,
            }
        max_vel = max(1.0, float(getattr(args, "max_joint_vel_deg_s", MAX_REPLAY_JOINT_VEL_DEG_S)))
        if robot == "ur5e":
            max_vel = min(max_vel, UR5E_REPLAY_MAX_JOINT_VEL_DEG_S)
        approach_time = max(step, max_delta_deg / max_vel)

    return {
        "success": True,
        "robot": robot,
        "joint_names": list(robot_meta.get("joint_names") or ROBOTS[robot]["gazebo_joints"]),
        "gripper_joint": str(
            robot_meta.get("gripper_joint")
            or dict(ROBOTS[robot].get("gripper") or {}).get("gazebo_joint")
            or ""
        ),
        "hardware_names": hardware_names,
        "hardware_positions": hardware_positions,
        "waypoints": robot_waypoints,
        "approach_time": approach_time,
        "max_joint_delta_deg": max_delta_deg,
        "max_joint_delta_joint": max_delta_joint,
    }


def _append_final_hold_point(
    points: list[dict[str, Any]],
    hold_sec: float,
) -> list[dict[str, Any]]:
    prepared = [dict(point) for point in points]
    if not prepared or float(hold_sec) <= 0.0:
        return prepared
    final_point = dict(prepared[-1])
    final_time = float(final_point.get("time") or 0.0)
    hold_point = {
        "positions": [float(value) for value in list(final_point.get("positions") or [])],
        "time": final_time + float(hold_sec),
    }
    if "velocities" in final_point:
        hold_point["velocities"] = [0.0 for _ in list(final_point.get("positions") or [])]
    if "accelerations" in final_point:
        hold_point["accelerations"] = [0.0 for _ in list(final_point.get("positions") or [])]
    prepared.append(hold_point)
    return prepared


def _build_paired_replay_preparation(args: argparse.Namespace, recording: dict[str, Any]) -> dict[str, Any]:
    """Build paired replay data and preflight hardware endpoints without commanding motion."""
    waypoints = list(recording.get("waypoints") or [])
    if not waypoints:
        return {"success": False, "message": "recording has no waypoints."}

    step = max(0.2, float(args.waypoint_duration_sec))
    replay_target = str(args.replay_target or "hardware")
    need_gazebo = replay_target in ("gazebo", "both")
    need_hardware = replay_target in ("hardware", "both")
    robots = [
        robot
        for robot in ("xarm6", "ur5e")
        if robot in dict(recording.get("robots") or {})
    ]
    if set(robots) != {"xarm6", "ur5e"}:
        return {"success": False, "message": "paired recording must contain xarm6 and ur5e."}

    plans: dict[str, dict[str, Any]] = {}
    for robot in robots:
        _write_replay_status(
            args,
            state="preparing",
            message=f"preflighting {robot} replay.",
        )
        plan = _paired_replay_robot_plan(
            args,
            recording,
            robot,
            need_hardware=need_hardware,
            step=step,
        )
        if not plan.get("success"):
            return plan
        plans[robot] = plan

    if need_hardware:
        xarm6_action_name = str(ROBOTS["xarm6"].get("hardware_trajectory_action") or "").strip()
        xarm6_action_result = _wait_follow_joint_trajectory_action(
            int(args.hardware_domain_id),
            xarm6_action_name,
            timeout_sec=8.0,
        )
        if not xarm6_action_result.get("success"):
            return {
                "success": False,
                "message": f"xarm6: {str(xarm6_action_result.get('message') or f'{xarm6_action_name} unavailable')}",
            }
        ur5e_action_name = str(ROBOTS["ur5e"].get("hardware_trajectory_action") or "").strip()
        ur5e_action_result = _wait_follow_joint_trajectory_action(
            int(args.hardware_domain_id),
            ur5e_action_name,
            timeout_sec=8.0,
        )
        if not ur5e_action_result.get("success"):
            return {
                "success": False,
                "message": f"ur5e: {str(ur5e_action_result.get('message') or f'{ur5e_action_name} unavailable')}",
            }
        for robot in robots:
            plan = plans[robot]
            gripper_points = [
                waypoint for waypoint in list(plan.get("waypoints") or [])
                if "gripper" in dict(waypoint)
            ]
            if len(gripper_points) == len(list(plan.get("waypoints") or [])):
                if robot == "xarm6":
                    endpoint_result = _wait_xarm_gripper_endpoint(
                        int(args.hardware_domain_id),
                        timeout_sec=8.0,
                    )
                    if not endpoint_result.get("success"):
                        return {
                            "success": False,
                            "message": f"{robot}: {str(endpoint_result.get('message') or 'xarm6 gripper endpoint unavailable')}",
                        }
                elif robot == "ur5e":
                    gripper_action = str(dict(ROBOTS[robot].get("gripper") or {}).get("hardware_action") or "").strip()
                    if gripper_action:
                        gripper_action_result = _wait_follow_joint_trajectory_action(
                            int(args.hardware_domain_id),
                            gripper_action,
                            timeout_sec=8.0,
                        )
                        if not gripper_action_result.get("success"):
                            return {
                                "success": False,
                                "message": f"{robot}: {str(gripper_action_result.get('message') or 'RG2 action server unavailable')}",
                            }

    approach_time = max(float(plan.get("approach_time") or step) for plan in plans.values())
    timeout = approach_time + step * len(waypoints) + 8.0
    for plan in plans.values():
        plan["points"] = [
            {"positions": list(waypoint["positions"]), "time": approach_time + i * step}
            for i, waypoint in enumerate(plan["waypoints"])
        ]
        hardware_current_point_sec = (
            UR5E_HARDWARE_TRAJECTORY_CURRENT_POINT_SEC
            if plan.get("robot") == "ur5e"
            else HARDWARE_TRAJECTORY_CURRENT_POINT_SEC
        )
        hardware_time_offset = hardware_current_point_sec - HARDWARE_TRAJECTORY_CURRENT_POINT_SEC
        plan["hardware_points"] = (
            [
                {
                    "positions": list(plan.get("hardware_positions") or []),
                    "time": hardware_current_point_sec,
                },
                *[
                    {
                        "positions": list(point["positions"]),
                        "time": float(point.get("time") or 0.0) + hardware_time_offset,
                    }
                    for point in list(plan["points"])
                ],
            ]
            if need_hardware
            else []
        )
        if need_hardware:
            if plan.get("robot") == "xarm6":
                plan["hardware_points"] = _append_final_hold_point(
                    list(plan["hardware_points"]),
                    XARM6_TEACH_REPLAY_FINAL_HOLD_SEC,
                )
                plan["hardware_final_hold_sec"] = XARM6_TEACH_REPLAY_FINAL_HOLD_SEC
                plan["hardware_goal_time_tolerance_sec"] = XARM6_TEACH_REPLAY_GOAL_TIME_TOLERANCE_SEC
            elif plan.get("robot") == "ur5e":
                plan["hardware_points"] = _append_final_hold_point(
                    list(plan["hardware_points"]),
                    UR5E_TEACH_REPLAY_FINAL_HOLD_SEC,
                )
                plan["hardware_final_hold_sec"] = UR5E_TEACH_REPLAY_FINAL_HOLD_SEC
                plan["hardware_goal_time_tolerance_sec"] = UR5E_TEACH_REPLAY_GOAL_TIME_TOLERANCE_SEC
        gripper_waypoints = [
            waypoint for waypoint in list(plan.get("waypoints") or [])
            if "gripper" in dict(waypoint)
        ]
        first_gripper_time = min(0.5, max(0.1, approach_time))
        plan["gripper_points"] = (
            [
                {
                    "positions": [float(waypoint["gripper"])],
                    "time": first_gripper_time if i == 0 else approach_time + i * step,
                }
                for i, waypoint in enumerate(plan["waypoints"])
            ]
            if len(gripper_waypoints) == len(list(plan.get("waypoints") or []))
            else []
        )

    if need_hardware:
        ur5e_plan = plans["ur5e"]
        ur5e_action_name = str(ROBOTS["ur5e"].get("hardware_trajectory_action") or "").strip()
        ur5e_points = [dict(point) for point in list(ur5e_plan.get("hardware_points") or [])]
        ur5e_plan["hardware_plan_result"] = {
            "success": True,
            "message": (
                f"{ur5e_action_name}: ur5e direct trajectory preflight ready; "
                f"waypoints={len(waypoints)}; points={len(ur5e_points)}."
            ),
            "action_name": ur5e_action_name,
            "mode": "follow_joint_trajectory",
            "joint_names": [str(name) for name in list(ur5e_plan.get("hardware_names") or [])],
            "waypoints": len(waypoints),
            "points": len(ur5e_points),
        }

    return {
        "success": True,
        "message": f"prepared paired replay to {replay_target}.",
        "metadata": _prepared_replay_metadata(args, recording),
        "plans": plans,
        "robots": robots,
        "waypoints": len(waypoints),
        "step": step,
        "replay_target": replay_target,
        "need_gazebo": need_gazebo,
        "need_hardware": need_hardware,
        "approach_time": approach_time,
        "timeout": timeout,
    }


def _preflight_ur5e_move_group_replay(
    args: argparse.Namespace,
    plan: dict[str, Any],
    *,
    timeout_sec: float,
) -> dict[str, Any]:
    def _move_group_acceptance_timed_out(result: dict[str, Any]) -> bool:
        message = str(result.get("message") or "").lower()
        return (
            str(result.get("action_name") or MOVE_GROUP_ACTION_NAME) == MOVE_GROUP_ACTION_NAME
            and "action goal acceptance timed out" in message
        )

    action_result = _wait_move_group_action(int(args.hardware_domain_id), timeout_sec=8.0)
    if not action_result.get("success"):
        return {
            "success": False,
            "message": (
                f"{MOVE_GROUP_ACTION_NAME}: ur5e MoveIt preflight failed: "
                f"{str(action_result.get('message') or 'action server unavailable')}"
            ),
            "action_name": MOVE_GROUP_ACTION_NAME,
            "group_name": UR5E_HARDWARE_MOVE_GROUP,
            "mode": "plan_only",
            "move_group_results": [dict(action_result)],
        }
    execute_action_result = _wait_execute_trajectory_action(
        int(args.hardware_domain_id),
        timeout_sec=8.0,
    )
    if not execute_action_result.get("success"):
        return {
            "success": False,
            "message": (
                "/execute_trajectory: ur5e MoveIt preflight failed: "
                f"{str(execute_action_result.get('message') or 'action server unavailable')}"
            ),
            "action_name": "/execute_trajectory",
            "group_name": UR5E_HARDWARE_MOVE_GROUP,
            "mode": "execute_trajectory",
            "move_group_results": [dict(action_result)],
            "execute_trajectory_result": dict(execute_action_result),
        }

    joint_names = [str(name) for name in list(plan.get("hardware_names") or [])]
    start_positions = [float(value) for value in list(plan.get("hardware_positions") or [])]
    waypoints = [dict(waypoint) for waypoint in list(plan.get("waypoints") or [])]
    if not joint_names or not start_positions or len(joint_names) != len(start_positions):
        return {
            "success": False,
            "message": f"{MOVE_GROUP_ACTION_NAME}: ur5e MoveIt preflight failed: missing ur5e hardware start state.",
            "action_name": MOVE_GROUP_ACTION_NAME,
            "group_name": UR5E_HARDWARE_MOVE_GROUP,
            "mode": "plan_only",
            "joint_names": list(joint_names),
            "move_group_results": [],
        }
    if not waypoints:
        return {
            "success": False,
            "message": f"{MOVE_GROUP_ACTION_NAME}: ur5e MoveIt preflight failed: no ur5e waypoints.",
            "action_name": MOVE_GROUP_ACTION_NAME,
            "group_name": UR5E_HARDWARE_MOVE_GROUP,
            "mode": "plan_only",
            "joint_names": list(joint_names),
            "move_group_results": [],
        }

    plan_results: list[dict[str, Any]] = []
    plan_attempt_results: list[dict[str, Any]] = []
    moveit_timeout = max(float(timeout_sec), MOVE_GROUP_PLAN_TIMEOUT_SEC)
    acceptance_timeout = max(
        MOVE_GROUP_GOAL_ACCEPTANCE_TIMEOUT_SEC,
        min(max(moveit_timeout, MOVE_GROUP_PLAN_TIMEOUT_SEC), 30.0),
    )
    max_attempts = 1 + max(0, int(MOVE_GROUP_GOAL_ACCEPTANCE_RETRY_COUNT))
    for index, waypoint in enumerate(waypoints, start=1):
        target_positions = [float(value) for value in list(waypoint.get("positions") or [])]
        if len(target_positions) != len(joint_names):
            result = {
                "success": False,
                "message": (
                    f"{MOVE_GROUP_ACTION_NAME}: ur5e MoveIt preflight failed waypoint "
                    f"{index}/{len(waypoints)}: target position count {len(target_positions)} "
                    f"does not match joint count {len(joint_names)}."
                ),
                "action_name": MOVE_GROUP_ACTION_NAME,
                "group_name": UR5E_HARDWARE_MOVE_GROUP,
                "mode": "plan_only",
                "waypoint_index": int(index),
                "joint_names": list(joint_names),
            }
        else:
            result = {}
            for attempt in range(1, max_attempts + 1):
                result = _plan_move_group_joint_goal(
                    int(args.hardware_domain_id),
                    UR5E_HARDWARE_MOVE_GROUP,
                    joint_names,
                    start_positions,
                    target_positions,
                    waypoint_index=index,
                    timeout_sec=moveit_timeout,
                    acceptance_timeout_sec=acceptance_timeout,
                )
                result = dict(result)
                result["attempt"] = int(attempt)
                result["attempts"] = int(max_attempts)
                result.setdefault("acceptance_timeout_sec", float(acceptance_timeout))
                plan_attempt_results.append(dict(result))
                if bool(result.get("success")):
                    break
                if attempt >= max_attempts or not _move_group_acceptance_timed_out(result):
                    break
                time.sleep(max(0.0, MOVE_GROUP_GOAL_ACCEPTANCE_RETRY_DELAY_SEC))
        if bool(result.get("success")):
            plan_results.append(dict(result))
        if not bool(result.get("success")):
            return {
                "success": False,
                "message": (
                    f"{MOVE_GROUP_ACTION_NAME}: ur5e MoveIt preflight failed waypoint "
                    f"{index}/{len(waypoints)}: {str(result.get('message') or 'planning failed')}"
                ),
                "action_name": MOVE_GROUP_ACTION_NAME,
                "group_name": UR5E_HARDWARE_MOVE_GROUP,
                "mode": "plan_only",
                "failed_waypoint_index": int(index),
                "joint_names": list(joint_names),
                "move_group_results": plan_results,
                "move_group_attempt_results": plan_attempt_results,
                "acceptance_timeout_sec": float(acceptance_timeout),
            }
        start_positions = list(target_positions)

    stitched = _stitch_ur5e_move_group_plan_results(joint_names, plan_results)
    if not bool(stitched.get("success")):
        return {
            "success": False,
            "message": (
                f"/execute_trajectory: ur5e MoveIt preflight failed while stitching: "
                f"{str(stitched.get('message') or 'trajectory stitching failed')}"
            ),
            "action_name": "/execute_trajectory",
            "group_name": UR5E_HARDWARE_MOVE_GROUP,
            "mode": "execute_trajectory",
            "joint_names": list(joint_names),
            "move_group_results": plan_results,
            "move_group_attempt_results": plan_attempt_results,
            "stitch_result": dict(stitched),
        }

    return {
        "success": True,
        "message": (
            f"{MOVE_GROUP_ACTION_NAME}: ur5e MoveIt preflight planned; "
            f"/execute_trajectory ready; group={UR5E_HARDWARE_MOVE_GROUP}; "
            f"waypoints={len(waypoints)}; points={len(list(stitched.get('points') or []))}."
        ),
        "action_name": MOVE_GROUP_ACTION_NAME,
        "execute_action_name": "/execute_trajectory",
        "group_name": UR5E_HARDWARE_MOVE_GROUP,
        "mode": "plan_only",
        "joint_names": list(joint_names),
        "waypoints": len(waypoints),
        "move_group_results": plan_results,
        "move_group_attempt_results": plan_attempt_results,
        "acceptance_timeout_sec": float(acceptance_timeout),
        "execute_trajectory_joint_names": list(stitched.get("joint_names") or []),
        "execute_trajectory_points": [dict(point) for point in list(stitched.get("points") or [])],
        "execute_trajectory_final_time": float(stitched.get("final_time") or 0.0),
        "execute_trajectory_segments": int(stitched.get("segments") or 0),
        "final_hold_sec": float(stitched.get("final_hold_sec") or 0.0),
        "time_scale": float(stitched.get("time_scale") or 1.0),
        "execute_trajectory_result": dict(execute_action_result),
    }


def run_paired_replay(args: argparse.Namespace, recording: dict[str, Any]) -> int:
    """Replay paired xArm6 + UR5e waypoints on one shared timeline."""
    replay_target = str(args.replay_target or "hardware")
    _write_replay_status(
        args,
        state="replaying",
        message=f"preparing paired replay to {replay_target}.",
    )
    prepared: dict[str, Any] = {}
    using_prepared_file = False
    prepared_file_raw = str(getattr(args, "prepared_file", "") or "").strip()
    if prepared_file_raw:
        prepared_file = Path(prepared_file_raw)
        prepared_candidate = _read_json(prepared_file)
        if prepared_candidate:
            stale_reason = _prepared_replay_validation_error(args, recording, prepared_candidate)
            if stale_reason:
                _write_replay_status(
                    args,
                    state="replaying",
                    message=f"{stale_reason} Preparing replay synchronously.",
                )
            else:
                drift_reason = (
                    _prepared_start_drift_error(args, prepared_candidate)
                    if replay_target in ("hardware", "both")
                    else ""
                )
                if drift_reason:
                    _write_replay_status(
                        args,
                        state="replaying",
                        message=f"{drift_reason} Re-preparing because hardware moved.",
                    )
                else:
                    refresh_reason = (
                        _refresh_prepared_hardware_start_from_snapshot(args, prepared_candidate)
                        if replay_target in ("hardware", "both")
                        else ""
                    )
                    if refresh_reason:
                        _write_replay_status(
                            args,
                            state="replaying",
                            message=f"{refresh_reason} Re-preparing because hardware moved.",
                        )
                    else:
                        prepared = prepared_candidate
                        using_prepared_file = True

    if not prepared:
        _write_replay_status(
            args,
            state="preparing",
            message="preparing paired replay before hardware execution.",
        )
        prepared = _build_paired_replay_preparation(args, recording)
        if not prepared.get("success"):
            message = str(prepared.get("message") or "replay preflight failed.")
            _write_replay_status(args, state="blocked", message=message, last_error=message)
            print(json.dumps(prepared))
            return 6

    plans = {
        str(robot): dict(plan)
        for robot, plan in dict(prepared.get("plans") or {}).items()
    }
    robots = [str(robot) for robot in list(prepared.get("robots") or [])]
    waypoints = list(recording.get("waypoints") or [])
    need_gazebo = bool(prepared.get("need_gazebo"))
    need_hardware = bool(prepared.get("need_hardware"))
    approach_time = float(prepared.get("approach_time") or 0.0)
    timeout = float(prepared.get("timeout") or 8.0)

    if need_hardware and using_prepared_file:
        xarm6_action_name = str(ROBOTS["xarm6"].get("hardware_trajectory_action") or "").strip()
        xarm6_action_result = _wait_follow_joint_trajectory_action(
            int(args.hardware_domain_id),
            xarm6_action_name,
            timeout_sec=3.0,
        )
        if not xarm6_action_result.get("success"):
            message = f"xarm6: {str(xarm6_action_result.get('message') or f'{xarm6_action_name} unavailable')}"
            _write_replay_status(args, state="blocked", message=message, last_error=message)
            print(json.dumps({"success": False, "message": message}))
            return 6
        ur5e_action_name = str(ROBOTS["ur5e"].get("hardware_trajectory_action") or "").strip()
        ur5e_action_result = _wait_follow_joint_trajectory_action(
            int(args.hardware_domain_id),
            ur5e_action_name,
            timeout_sec=3.0,
        )
        if not ur5e_action_result.get("success"):
            message = f"ur5e: {str(ur5e_action_result.get('message') or f'{ur5e_action_name} unavailable')}"
            _write_replay_status(args, state="blocked", message=message, last_error=message)
            print(json.dumps({"success": False, "message": message}))
            return 6
        for robot in robots:
            gripper_points = list(dict(plans.get(robot) or {}).get("gripper_points") or [])
            if not gripper_points:
                continue
            if robot == "xarm6":
                endpoint_result = _wait_xarm_gripper_endpoint(int(args.hardware_domain_id), timeout_sec=3.0)
                if not endpoint_result.get("success"):
                    message = f"{robot}: {str(endpoint_result.get('message') or 'xarm6 gripper endpoint unavailable')}"
                    _write_replay_status(args, state="blocked", message=message, last_error=message)
                    print(json.dumps({"success": False, "message": message}))
                    return 6
            elif robot == "ur5e":
                gripper_action = str(dict(ROBOTS[robot].get("gripper") or {}).get("hardware_action") or "").strip()
                if gripper_action:
                    gripper_action_result = _wait_follow_joint_trajectory_action(
                        int(args.hardware_domain_id),
                        gripper_action,
                        timeout_sec=3.0,
                    )
                    if not gripper_action_result.get("success"):
                        message = f"{robot}: {str(gripper_action_result.get('message') or 'RG2 action server unavailable')}"
                        _write_replay_status(args, state="blocked", message=message, last_error=message)
                        print(json.dumps({"success": False, "message": message}))
                        return 6

    results: dict[str, dict[str, Any]] = {}
    result_lock = threading.Lock()
    threads: list[threading.Thread] = []
    expected_results: list[str] = []
    _write_replay_status(
        args,
        state="replaying",
        message=(
            "starting hardware execution from prepared replay."
            if using_prepared_file and need_hardware
            else f"publishing paired replay to {replay_target}."
        ),
    )

    def _set_result(key: str, value: dict[str, Any]) -> None:
        with result_lock:
            results[key] = value

    def _publish_for(robot: str, side: str) -> None:
        plan = plans[robot]
        if side == "gazebo":
            _set_result(f"{robot}/gazebo", _publish_trajectory(
                int(args.gazebo_domain_id),
                ROBOTS[robot]["gazebo_trajectory_topics"],
                list(plan["joint_names"]),
                list(plan["points"]),
                join_timeout_sec=timeout,
            ))
            return

    def _decorate_hardware_arm_result(
        result: dict[str, Any],
        *,
        robot: str,
        action_name: str,
        joint_names: list[str],
        points: list[dict[str, Any]],
    ) -> dict[str, Any]:
        plan = plans[robot]
        result = dict(result)
        detail = (
            f"{robot} arm commanded; "
            f"action={action_name}; "
            f"joints={','.join(str(name) for name in joint_names)}; "
            f"max_joint_delta_deg={float(plan.get('max_joint_delta_deg') or 0.0):.3f}; "
            f"max_joint_delta_joint={str(plan.get('max_joint_delta_joint') or '')}; "
            f"approach_time={float(approach_time):.3f}; "
            f"points={len(points)}"
        )
        if "zero_velocities" in result:
            detail += f"; zero_velocities={bool(result.get('zero_velocities'))}"
        if "header_stamp" in result:
            detail += f"; header_stamp={bool(result.get('header_stamp'))}"
        if "start_delay_sec" in result:
            detail += f"; start_delay_sec={float(result.get('start_delay_sec') or 0.0):.3f}"
        if "first_point_time" in result:
            detail += f"; first_point_time={_format_optional_time(result.get('first_point_time'))}"
        if "second_point_time" in result:
            detail += f"; second_point_time={_format_optional_time(result.get('second_point_time'))}"
        if "final_point_time" in result:
            detail += f"; final_point_time={_format_optional_time(result.get('final_point_time'))}"
        if "min_point_spacing" in result:
            detail += f"; min_point_spacing={_format_optional_time(result.get('min_point_spacing'))}"
        if "controller_state" in result:
            detail += f"; controller_state={str(result.get('controller_state') or '')}"
        if "canceled" in result:
            detail += f"; canceled={bool(result.get('canceled'))}"
        if "goal_time_tolerance_sec" in result:
            detail += f"; goal_time_tolerance_sec={float(result.get('goal_time_tolerance_sec') or 0.0):.3f}"
        if "final_hold_sec" in result:
            detail += f"; final_hold_sec={float(result.get('final_hold_sec') or 0.0):.3f}"
        if "observed_completion" in result:
            observed = dict(result.get("observed_completion") or {})
            detail += f"; observed_completion={bool(observed.get('success'))}"
        result["message"] = f"{str(result.get('message') or '').rstrip('. ')} ({detail})."
        result["action_name"] = action_name
        result["joint_names"] = list(joint_names)
        result["max_joint_delta_deg"] = float(plan.get("max_joint_delta_deg") or 0.0)
        result["max_joint_delta_joint"] = str(plan.get("max_joint_delta_joint") or "")
        result["approach_time"] = float(approach_time)
        result["points"] = len(points)
        return result

    def _publish_xarm_hardware_arm() -> None:
        plan = plans["xarm6"]
        action_name = str(ROBOTS["xarm6"].get("hardware_trajectory_action") or "").strip()
        joint_names = list(plan["hardware_names"])
        points = list(plan["hardware_points"])
        target_positions = [float(value) for value in list(points[-1].get("positions") or [])] if points else []
        final_time = max((float(point.get("time") or 0.0) for point in points), default=0.0)
        xarm_timeout = max(
            timeout,
            final_time + XARM6_TEACH_REPLAY_RESULT_TIMEOUT_MARGIN_SEC,
        )
        result = _publish_follow_joint_trajectory_action(
            int(args.hardware_domain_id),
            action_name,
            joint_names,
            points,
            join_timeout_sec=xarm_timeout,
            start_delay_sec=HARDWARE_TRAJECTORY_START_DELAY_SEC,
            goal_time_tolerance_sec=XARM6_TEACH_REPLAY_GOAL_TIME_TOLERANCE_SEC,
        )
        result = _apply_observed_completion_fallback(
            result,
            domain_id=int(args.hardware_domain_id),
            robot="xarm6",
            joint_names=joint_names,
            target_positions=target_positions,
        )
        result["goal_time_tolerance_sec"] = XARM6_TEACH_REPLAY_GOAL_TIME_TOLERANCE_SEC
        result["final_hold_sec"] = float(plan.get("hardware_final_hold_sec") or 0.0)
        _set_result(
            "xarm6/hardware_arm",
            _decorate_hardware_arm_result(
                result,
                robot="xarm6",
                action_name=action_name,
                joint_names=joint_names,
                points=points,
            ),
        )

    def _publish_ur5e_hardware_arm() -> None:
        plan = plans["ur5e"]
        action_name = str(ROBOTS["ur5e"].get("hardware_trajectory_action") or "").strip()
        joint_names = [str(name) for name in list(plan.get("hardware_names") or [])]
        points = [dict(point) for point in list(plan.get("hardware_points") or [])]
        _set_result("ur5e/hardware_plan", dict(plan.get("hardware_plan_result") or {}))
        target_positions = [float(value) for value in list(points[-1].get("positions") or [])] if points else []
        final_time = max((float(point.get("time") or 0.0) for point in points), default=0.0)
        if not joint_names or not points:
            result = {
                "success": False,
                "message": f"{action_name}: no UR5e hardware trajectory points.",
                "mode": "follow_joint_trajectory",
                "action_name": action_name,
            }
        else:
            result = _publish_follow_joint_trajectory_action(
                int(args.hardware_domain_id),
                action_name,
                joint_names,
                points,
                join_timeout_sec=max(
                    timeout,
                    final_time + UR5E_TEACH_REPLAY_RESULT_TIMEOUT_MARGIN_SEC,
                ),
                start_delay_sec=HARDWARE_TRAJECTORY_START_DELAY_SEC,
                goal_time_tolerance_sec=UR5E_TEACH_REPLAY_GOAL_TIME_TOLERANCE_SEC,
            )
            result = _apply_observed_completion_fallback(
                result,
                domain_id=int(args.hardware_domain_id),
                robot="ur5e",
                joint_names=joint_names,
                target_positions=target_positions,
            )
            result["mode"] = "follow_joint_trajectory"
            result["action_name"] = action_name
            result["goal_time_tolerance_sec"] = UR5E_TEACH_REPLAY_GOAL_TIME_TOLERANCE_SEC
            result["final_hold_sec"] = float(plan.get("hardware_final_hold_sec") or UR5E_TEACH_REPLAY_FINAL_HOLD_SEC)
        _set_result(
            "ur5e/hardware_arm",
            _decorate_hardware_arm_result(
                result,
                robot="ur5e",
                action_name=action_name,
                joint_names=joint_names,
                points=points,
            ),
        )

    def _publish_paired_hardware_arms() -> None:
        arm_threads = [
            threading.Thread(target=_publish_xarm_hardware_arm, daemon=True),
            threading.Thread(target=_publish_ur5e_hardware_arm, daemon=True),
        ]
        for arm_thread in arm_threads:
            arm_thread.start()
        arm_join_timeout = timeout + max(
            XARM6_TEACH_REPLAY_RESULT_TIMEOUT_MARGIN_SEC,
            UR5E_TEACH_REPLAY_RESULT_TIMEOUT_MARGIN_SEC,
        ) + 20.0
        for arm_thread in arm_threads:
            arm_thread.join(arm_join_timeout)
        with result_lock:
            decorated_results = {
                "xarm6": dict(results.get("xarm6/hardware_arm") or {}),
                "ur5e": dict(results.get("ur5e/hardware_arm") or {}),
            }

        ur5e_plan = plans["ur5e"]
        if ur5e_plan.get("gripper_points"):
            if bool(decorated_results.get("ur5e", {}).get("success")):
                _publish_gripper_for("ur5e", "hardware")
            else:
                _set_result(
                    "ur5e/hardware_gripper",
                    {
                        "success": True,
                        "message": "skipped because ur5e hardware arm failed.",
                    },
                )

    def _publish_gripper_for(robot: str, side: str) -> None:
        plan = plans[robot]
        gripper_points = list(plan.get("gripper_points") or [])
        gripper_joint = str(plan.get("gripper_joint") or "").strip()
        if not gripper_points or not gripper_joint:
            _set_result(f"{robot}/{side}_gripper", {"success": True, "message": "no gripper waypoints."})
            return
        gripper_cfg = dict(ROBOTS[robot].get("gripper") or {})
        if side == "gazebo":
            _set_result(f"{robot}/gazebo_gripper", _publish_trajectory(
                int(args.gazebo_domain_id),
                list(gripper_cfg.get("gazebo_trajectory_topics") or []),
                [gripper_joint],
                gripper_points,
                join_timeout_sec=timeout,
            ))
            return
        if robot == "xarm6":
            _set_result(f"{robot}/hardware_gripper", _publish_xarm_gripper_sequence(
                int(args.hardware_domain_id),
                gripper_points,
                join_timeout_sec=timeout,
            ))
            return
        _set_result(f"{robot}/hardware_gripper", _publish_follow_joint_trajectory_action(
            int(args.hardware_domain_id),
            str(gripper_cfg.get("hardware_action") or ""),
            [gripper_joint],
            gripper_points,
            join_timeout_sec=timeout,
        ))

    thread_launch_blocked = False
    if need_hardware:
        ur5e_preflight = dict(plans["ur5e"].get("hardware_plan_result") or {})
        plans["ur5e"]["hardware_plan_result"] = dict(ur5e_preflight)
        _set_result("ur5e/hardware_plan", dict(ur5e_preflight))
        if not bool(ur5e_preflight.get("success")) or not list(plans["ur5e"].get("hardware_points") or []):
            thread_launch_blocked = True
            preflight_message = str(
                ur5e_preflight.get("message")
                or "ur5e direct trajectory preflight failed: missing hardware trajectory points"
            )
            _set_result(
                "xarm6/hardware_arm",
                {
                    "success": False,
                    "message": f"not sent because ur5e direct trajectory preflight failed: {preflight_message}",
                    "skip_observed_completion": True,
                },
            )
            _set_result(
                "ur5e/hardware_arm",
                {
                    "success": False,
                    "message": f"not sent because ur5e direct trajectory preflight failed: {preflight_message}",
                    "action_name": str(ROBOTS["ur5e"].get("hardware_trajectory_action") or ""),
                    "mode": "follow_joint_trajectory",
                    "skip_observed_completion": True,
                },
            )
            if plans["xarm6"].get("gripper_points"):
                _set_result(
                    "xarm6/hardware_gripper",
                    {
                        "success": False,
                        "message": f"not sent because ur5e direct trajectory preflight failed: {preflight_message}",
                    },
                )
            if plans["ur5e"].get("gripper_points"):
                _set_result(
                    "ur5e/hardware_gripper",
                    {
                        "success": True,
                        "message": "skipped because ur5e hardware arm failed.",
                    },
                )

    if not thread_launch_blocked:
        for robot in robots:
            if need_gazebo:
                expected_results.append(f"{robot}/gazebo")
                threads.append(threading.Thread(target=_publish_for, args=(robot, "gazebo"), daemon=True))
                if plans[robot].get("gripper_points"):
                    expected_results.append(f"{robot}/gazebo_gripper")
                    threads.append(threading.Thread(target=_publish_gripper_for, args=(robot, "gazebo"), daemon=True))
            if need_hardware and plans[robot].get("gripper_points"):
                expected_results.append(f"{robot}/hardware_gripper")
                if robot == "xarm6":
                    threads.append(threading.Thread(target=_publish_gripper_for, args=(robot, "hardware"), daemon=True))
        if need_hardware:
            expected_results.append("xarm6/hardware_arm")
            expected_results.append("ur5e/hardware_plan")
            expected_results.append("ur5e/hardware_arm")
            _set_result("ur5e/hardware_plan", dict(plans["ur5e"].get("hardware_plan_result") or {}))
            threads.append(threading.Thread(target=_publish_paired_hardware_arms, daemon=True))
        for thread in threads:
            thread.start()
        thread_join_timeout = timeout + 5.0
        if need_hardware:
            xarm_points = list(dict(plans.get("xarm6") or {}).get("hardware_points") or [])
            xarm_final_time = max((float(point.get("time") or 0.0) for point in xarm_points), default=0.0)
            thread_join_timeout = max(
                thread_join_timeout,
                xarm_final_time + XARM6_TEACH_REPLAY_RESULT_TIMEOUT_MARGIN_SEC + 20.0,
            )
            ur5e_points = list(
                dict(plans.get("ur5e") or {}).get("hardware_points")
                or dict(plans.get("ur5e") or {}).get("points")
                or []
            )
            ur5e_final_time = max((float(point.get("time") or 0.0) for point in ur5e_points), default=0.0)
            thread_join_timeout = max(
                thread_join_timeout,
                ur5e_final_time + UR5E_TEACH_REPLAY_RESULT_TIMEOUT_MARGIN_SEC + 20.0,
            )
        for thread in threads:
            thread.join(thread_join_timeout)

        with result_lock:
            for key in expected_results:
                if key not in results:
                    results[key] = {
                        "success": False,
                        "message": "replay thread did not finish.",
                    }

    ok = bool(results) and all(bool(result.get("success")) for result in results.values())
    message = "; ".join(f"{name}: {result.get('message') or ''}" for name, result in sorted(results.items()))
    out: dict[str, Any] = {
        "success": ok,
        "message": message,
        "waypoints": len(waypoints),
        "robots": robots,
        "used_prepared_file": using_prepared_file,
        "approach_time": approach_time,
        "waypoint_duration_sec": float(prepared.get("step") or 0.0),
        "timeout": timeout,
    }
    if need_hardware:
        max_delta_robot = max(
            plans.values(),
            key=lambda plan: float(plan.get("max_joint_delta_deg") or 0.0),
        )
        out["max_joint_delta_deg"] = float(max_delta_robot.get("max_joint_delta_deg") or 0.0)
        out["max_joint_delta_joint"] = str(max_delta_robot.get("max_joint_delta_joint") or "")
    _write_replay_status(
        args,
        state="replayed" if ok else "blocked",
        message=message,
        last_error="" if ok else message,
    )
    print(json.dumps(out))
    return 0 if ok else 8


def run_prepare_replay(args: argparse.Namespace) -> int:
    """Prepare paired replay data without commanding gazebo or hardware motion."""
    try:
        with Path(args.recording_file).open("r", encoding="utf-8") as f:
            recording = json.load(f)
    except Exception as exc:
        print(json.dumps({"success": False, "message": f"cannot read recording: {exc}"}))
        return 4

    if not str(getattr(args, "prepared_file", "") or "").strip():
        print(json.dumps({"success": False, "message": "--prepared-file is required for prepare-replay."}))
        return 4

    if str(recording.get("robot") or "").strip().lower() != "dual robots" and str(
        recording.get("recording_type") or ""
    ).strip() != "paired_dual_robots":
        print(json.dumps({"success": False, "message": "prepare-replay only supports paired_dual_robots."}))
        return 4

    _write_replay_status(
        args,
        state="preparing",
        message=f"preparing paired replay to {str(args.replay_target or 'hardware')}.",
    )
    prepared = _build_paired_replay_preparation(args, recording)
    if not prepared.get("success"):
        message = str(prepared.get("message") or "prepare-replay failed.")
        _write_replay_status(args, state="blocked", message=message, last_error=message)
        print(json.dumps(prepared))
        return 6

    prepared_path = Path(str(args.prepared_file))
    _atomic_json_write(prepared_path, prepared)
    out = {
        "success": True,
        "message": str(prepared.get("message") or "prepared replay."),
        "prepared_file": str(prepared_path),
        "waypoints": int(prepared.get("waypoints") or 0),
        "robots": list(prepared.get("robots") or []),
    }
    _write_replay_status(
        args,
        state="prepared",
        message=out["message"],
        last_error="",
    )
    print(json.dumps(out))
    return 0


def run_replay(args: argparse.Namespace) -> int:
    """Replay recorded waypoints in gazebo (preview), on hardware, or on both in sync ('both')."""
    try:
        with Path(args.recording_file).open("r", encoding="utf-8") as f:
            recording = json.load(f)
    except Exception as exc:
        print(json.dumps({"success": False, "message": f"cannot read recording: {exc}"}))
        return 4

    if str(recording.get("robot") or "").strip().lower() == "dual robots" or str(
        recording.get("recording_type") or ""
    ).strip() == "paired_dual_robots":
        return run_paired_replay(args, recording)

    waypoints = list(recording.get("waypoints") or [])
    if not waypoints:
        print(json.dumps({"success": False, "message": "recording has no waypoints."}))
        return 5

    step = max(0.2, float(args.waypoint_duration_sec))
    replay_target = str(args.replay_target or "hardware")
    need_gazebo = replay_target in ("gazebo", "both")
    need_hardware = replay_target in ("hardware", "both")
    hardware_uses_moveit = need_hardware and str(args.robot) == "ur5e"

    recorded_joint_names = list(recording.get("joint_names") or ROBOTS[args.robot]["gazebo_joints"])

    # Hardware path: resolve hardware joint names now and size the approach to the first
    # waypoint by a safe joint speed (not a hard distance block). Checked BEFORE publishing
    # to either side, so gazebo never moves when hardware would be unsafe.
    hardware_names: list[str] = []
    max_delta_deg = 0.0
    max_delta_joint = ""
    approach_time = step
    if need_hardware:
        hardware_result = _read_snapshot(
            int(args.hardware_domain_id),
            args.robot,
            "hardware",
            HARDWARE_SNAPSHOT_TIMEOUT_SEC,
        )
        if not hardware_result.get("success"):
            print(json.dumps({"success": False, "message": str(hardware_result.get("message") or "missing hardware state")}))
            return 6
        hardware_snapshot = dict(hardware_result.get("snapshot") or {})
        hardware_names, hardware_positions, hardware_missing = _resolve_hardware_positions(hardware_snapshot, args.robot)
        if hardware_missing:
            print(json.dumps({"success": False, "message": f"missing hardware joints: {', '.join(hardware_missing)}"}))
            return 6

        first = [float(v) for v in waypoints[0]["positions"]]
        deltas = [_angular_delta(a, b) for a, b in zip(first, hardware_positions)]
        max_delta_deg = math.degrees(max(deltas)) if deltas else 0.0
        if deltas:
            max_delta_index = max(range(len(deltas)), key=lambda i: deltas[i])
            max_delta_joint = (
                str(hardware_names[max_delta_index])
                if max_delta_index < len(hardware_names)
                else ""
            )

        # Absolute sanity ceiling only: a near-180 deg single-joint delta usually means an
        # encoder/wrap problem, not a deliberate authored move.
        ceiling_deg = float(args.max_joint_delta_deg)
        if max_delta_deg > ceiling_deg:
            print(
                json.dumps(
                    {
                        "success": False,
                        "message": (
                            f"blocked: first waypoint is {max_delta_deg:.2f} deg from hardware "
                            f"on {max_delta_joint or 'unknown joint'} (ceiling {ceiling_deg:.2f})."
                        ),
                        "max_joint_delta_deg": max_delta_deg,
                        "max_joint_delta_joint": max_delta_joint,
                    }
                )
            )
            return 7

        max_vel = max(1.0, float(getattr(args, "max_joint_vel_deg_s", MAX_REPLAY_JOINT_VEL_DEG_S)))
        approach_time = max(step, max_delta_deg / max_vel)

    # First point lands at approach_time (speed-limited); the rest follow at `step` spacing.
    # Same point times drive gazebo and hardware so they stay in sync in 'both'.
    points = [
        {"positions": list(wp["positions"]), "time": approach_time + i * step}
        for i, wp in enumerate(waypoints)
    ]
    timeout = approach_time + step * len(points) + 8.0
    if hardware_uses_moveit:
        action_result = _wait_move_group_action(int(args.hardware_domain_id), timeout_sec=8.0)
        if not action_result.get("success"):
            print(
                json.dumps(
                    {
                        "success": False,
                        "message": (
                            "/move_action unavailable for ur5e plan_and_execute replay: "
                            f"{str(action_result.get('message') or 'action server unavailable')}"
                        ),
                        "waypoints": len(points),
                        "saved_steps": int(recording.get("saved_steps") or len(points)),
                        "approach_time": approach_time,
                        "max_joint_delta_deg": max_delta_deg,
                        "max_joint_delta_joint": max_delta_joint,
                    }
                )
            )
            return 6

    # Publish to the required domains concurrently so sim and hardware move in sync.
    results: dict[str, dict[str, Any]] = {}

    def _do_gazebo() -> None:
        results["gazebo"] = _publish_trajectory(
            int(args.gazebo_domain_id),
            ROBOTS[args.robot]["gazebo_trajectory_topics"],
            recorded_joint_names,
            points,
            join_timeout_sec=timeout,
        )

    def _do_hardware() -> None:
        if hardware_uses_moveit:
            start_positions = [float(value) for value in hardware_positions]
            move_results: list[dict[str, Any]] = []
            failed_result: dict[str, Any] | None = None
            failed_index: int | None = None
            moveit_timeout = max(timeout, MOVE_GROUP_PLAN_TIMEOUT_SEC)
            for index, waypoint in enumerate(waypoints, start=1):
                target_positions = [float(value) for value in list(waypoint.get("positions") or [])]
                result = _execute_move_group_joint_goal(
                    int(args.hardware_domain_id),
                    UR5E_HARDWARE_MOVE_GROUP,
                    hardware_names,
                    start_positions,
                    target_positions,
                    waypoint_index=index,
                    timeout_sec=moveit_timeout,
                )
                move_results.append(dict(result))
                if not bool(result.get("success")):
                    failed_result = dict(result)
                    failed_index = index
                    break
                start_positions = list(target_positions)
            if failed_result is None:
                results["hardware"] = {
                    "success": True,
                    "message": (
                        "ur5e/hardware_arm used MoveIt plan_and_execute; "
                        f"waypoints={len(waypoints)}; group={UR5E_HARDWARE_MOVE_GROUP}."
                    ),
                    "mode": "plan_and_execute",
                    "action_name": MOVE_GROUP_ACTION_NAME,
                    "group_name": UR5E_HARDWARE_MOVE_GROUP,
                    "move_group_results": move_results,
                }
            else:
                results["hardware"] = {
                    "success": False,
                    "message": (
                        "ur5e/hardware_arm used MoveIt plan_and_execute; "
                        f"failed waypoint {failed_index}/{len(waypoints)}: "
                        f"{str(failed_result.get('message') or 'MoveIt execution failed')}"
                    ),
                    "mode": "plan_and_execute",
                    "action_name": MOVE_GROUP_ACTION_NAME,
                    "group_name": UR5E_HARDWARE_MOVE_GROUP,
                    "failed_waypoint_index": failed_index,
                    "move_group_results": move_results,
                }
            return
        results["hardware"] = _publish_trajectory(
            int(args.hardware_domain_id),
            ROBOTS[args.robot]["trajectory_topics"],
            hardware_names,
            points,
            join_timeout_sec=timeout,
        )

    threads: list[threading.Thread] = []
    if need_gazebo:
        threads.append(threading.Thread(target=_do_gazebo, daemon=True))
    if need_hardware:
        threads.append(threading.Thread(target=_do_hardware, daemon=True))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 5.0)

    ok = bool(results) and all(bool(r.get("success")) for r in results.values())
    message = "; ".join(f"{name}: {r.get('message') or ''}" for name, r in results.items())
    out: dict[str, Any] = {
        "success": ok,
        "message": message,
        "waypoints": len(points),
        "saved_steps": int(recording.get("saved_steps") or len(points)),
        "approach_time": approach_time,
        "waypoint_duration_sec": step,
        "timeout": timeout,
    }
    if need_hardware:
        out["max_joint_delta_deg"] = max_delta_deg
        out["max_joint_delta_joint"] = max_delta_joint
    print(json.dumps(out))
    return 0 if ok else 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synched gazebo + hardware digital twin helper")
    parser.add_argument(
        "--mode",
        choices=[
            "mirror",
            "apply-gazebo-to-hardware",
            "initialize-gazebo-from-hardware",
            "snapshot",
            "xarm6-tf-readiness",
            "prepare-replay",
            "replay",
        ],
        required=True,
    )
    parser.add_argument("--target", default="")
    parser.add_argument("--robot", choices=[*sorted(ROBOTS), "dual robots"], required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--gazebo-domain-id", type=int, required=True)
    parser.add_argument("--hardware-domain-id", type=int, required=True)
    parser.add_argument("--status-file", default="")
    parser.add_argument("--direction-file", default="")
    parser.add_argument("--max-joint-delta-deg", type=float, default=10.0)
    # snapshot mode
    parser.add_argument("--source", choices=["gazebo", "hardware"], default="gazebo")
    parser.add_argument("--include-world-tool-pose", action="store_true")
    parser.add_argument("--tf-readiness-timeout-sec", type=float, default=20.0)
    # initialize-gazebo-from-hardware mode
    parser.add_argument("--init-tolerance-rad", type=float, default=INITIALIZE_GAZEBO_TOLERANCE_RAD)
    parser.add_argument("--init-attempts", type=int, default=INITIALIZE_GAZEBO_ATTEMPTS)
    # replay mode
    parser.add_argument("--recording-file", default="")
    parser.add_argument("--prepared-file", default="")
    parser.add_argument("--replay-target", choices=["hardware", "gazebo", "both"], default="hardware")
    parser.add_argument("--waypoint-duration-sec", type=float, default=DEFAULT_REPLAY_WAYPOINT_DURATION_SEC)
    parser.add_argument("--max-joint-vel-deg-s", type=float, default=MAX_REPLAY_JOINT_VEL_DEG_S)
    parser.add_argument("--ur5e-hardware-trajectory-action", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ur5e_hardware_trajectory_action = str(args.ur5e_hardware_trajectory_action or "").strip()
    if ur5e_hardware_trajectory_action:
        ROBOTS["ur5e"]["hardware_trajectory_action"] = ur5e_hardware_trajectory_action
    if args.mode == "mirror":
        return run_mirror(args)
    if args.mode == "apply-gazebo-to-hardware":
        return run_apply_gazebo_to_hardware(args)
    if args.mode == "initialize-gazebo-from-hardware":
        return run_initialize_gazebo_from_hardware(args)
    if args.mode == "snapshot":
        return run_snapshot(args)
    if args.mode == "xarm6-tf-readiness":
        return run_xarm6_tf_readiness(args)
    if args.mode == "prepare-replay":
        return run_prepare_replay(args)
    if args.mode == "replay":
        return run_replay(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
