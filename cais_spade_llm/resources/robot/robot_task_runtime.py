"""Execution runtime for registry-backed robot tasks."""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from math import acos, isclose, isfinite, sqrt
from pathlib import Path
from typing import Any

from .gazebo_pick_place_controller import (
    _PHYSICAL_XARM6_ASSEMBLY_SLOT_INSERT_ERROR,
    derive_move_insert_timeout_sec,
)
from .robot_task_model import (
    RobotTaskDefinition,
    RobotTaskEffect,
    RobotTaskStep,
    _evaluate_guard,
    _resolve_value,
)
from .robot_task_registry import robot_task_registry

_TAUGHT_FUNCTIONS_ROOT = Path(__file__).resolve().parent / "taught_functions"
_CARTESIAN_POSE_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")
_CARTESIAN_POSITION_FIELDS = ("x", "y", "z")
_POSITION_SOURCES = frozenset({"computed", "captured", "manual", "captured_relative"})
_ASSEMBLY_BOARD_V1_ARUCO_MAX_AGE_SEC = 8.0
_OPERATOR_CONFIRMED_HELD_PART_HANDOFF_MAX_AGE_SEC = 8.0
_MOVE_INSERT_FAILURE_RESULT_FIELDS = (
    "success",
    "message",
    "terminal",
    "goal_status",
    "state_uncertain",
    "motion_settled",
    "error_code",
    "trial_id",
    "final_phase",
    "final_tool0_pose_valid",
    "final_insertion_depth_m",
    "max_insertion_depth_m",
    "final_depth_error_m",
    "final_lateral_offset_m",
    "final_tilt_error_rad",
    "final_search_radius_m",
    "peak_axial_force_n",
    "peak_lateral_force_n",
    "peak_torque_nm",
    "peak_filtered_axial_force_n",
    "peak_filtered_lateral_force_n",
    "peak_filtered_torque_nm",
    "peak_tool_flange_torque_nm",
    "contact_detected",
    "engagement_detected",
    "seated_detected",
    "force_bias_valid",
    "soft_overload_detected",
    "soft_overload_recovered",
    "relief_exhausted",
    "relief_cycle_count",
    "last_soft_overload_reason",
    "hard_limit_detected",
    "hard_limit_reason",
    "limit_trigger",
    "limit_trigger_value",
    "limit_trigger_threshold",
    "relief_load_cleared",
    "relief_backoff_m",
    "relief_planned_backoff_m",
    "total_relief_backoff_m",
    "relief_resume_phase",
    "force_mode_stop_acknowledged",
    "servo_stop_acknowledged",
    "stop_l_command_completed",
    "stationary_confirmed",
    "server_trace_id",
    "server_trace_path",
    "server_trace_sha256",
    "server_trace_status",
    "server_trace_complete",
    "server_trace_sample_count",
    "disengagement_cycle_count",
    "last_disengagement_reason",
    "disengagement_withdrawal_m",
    "disengagement_contact_cleared",
    "disengagement_force_mode_stop_acknowledged",
    "recenter_position_error_m",
    "recenter_command_acknowledged",
    "disengagement_stationary_confirmed",
    "retare_baseline_consistent",
    "profile_sha256",
    "hard_caps_sha256",
)
_MOVE_INSERT_FAILURE_RESULT_VECTOR_FIELDS = (
    "force_bias",
    "limit_trigger_actual_tcp_force",
    "limit_trigger_tared_tcp_force",
)
_MOVE_INSERT_TRANSLATIONAL_PARTS = ("SG", "MG", "LG", "SCP", "MCP", "LCP")
_MOVE_INSERT_MODEL_MAP = {
    "SG": "gear_small",
    "MG": "gear_medium",
    "LG": "gear_large",
    "SCP": "circ_pin_small",
    "MCP": "circ_pin_medium",
    "LCP": "circ_pin_large",
}
_MOVE_INSERT_DIAGNOSTIC_TRACE_MAX_SAMPLES = 100_000
_MANUAL_FUNCTION_EXECUTION_AUTHORITY = object()
_HARDWARE_JOINT_NAMES = {
    "ur5e": (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ),
    "xarm6": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
}


def _normalize_pick_targets(raw: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw or {})
    tx = float(payload.get("tx", 0.0) or 0.0)
    ty = float(payload.get("ty", 0.0) or 0.0)
    tz = float(payload.get("tz", 0.0) or 0.0)
    pick_z = float(payload.get("pick_z", 0.0) or 0.0)
    travel_z = float(payload.get("travel_z", 1.2) or 1.2)
    origin_pose = dict(payload.get("origin_pose") or {})
    origin_pose.setdefault("x", tx)
    origin_pose.setdefault("y", ty)
    origin_pose.setdefault("z", tz)
    if not any(field in origin_pose for field in ("qx", "qy", "qz", "qw")):
        origin_pose.update({"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0})
    payload["origin_pose"] = origin_pose
    payload.setdefault(
        "origin_pose_provenance",
        {
            "frame_id": "world",
            "part_name": str(payload.get("part_name") or ""),
            "model_name": str(payload.get("model_name") or ""),
            "source": str(payload.get("target_pose_source") or "computed"),
            "orientation_source": "upright_axial_part_assumption",
        },
    )
    payload.setdefault("approach_pose", {"x": tx, "y": ty, "z": travel_z})
    payload.setdefault("target_pose", {"x": tx, "y": ty, "z": pick_z})
    return payload


def _normalize_place_targets(
    raw: dict[str, Any], *, runtime_state: dict[str, Any]
) -> dict[str, Any]:
    payload = dict(raw or {})
    slot_x = float(payload.get("slot_x", 0.0) or 0.0)
    slot_y = float(payload.get("slot_y", 0.0) or 0.0)
    place_z = float(payload.get("place_z", 0.0) or 0.0)
    travel_z = float(dict(runtime_state.get("_task_ctx") or {}).get("travel_z", 1.2) or 1.2)
    payload.setdefault("approach_pose", {"x": slot_x, "y": slot_y, "z": travel_z})
    payload.setdefault("target_pose", {"x": slot_x, "y": slot_y, "z": place_z})
    payload.setdefault("pre_insert_pose", deepcopy(payload["target_pose"]))
    payload.setdefault("insert_pose", deepcopy(payload["target_pose"]))
    return payload


def _normalized_se3_pose(
    raw_pose: Any,
    *,
    label: str,
) -> tuple[dict[str, float], str]:
    if not isinstance(raw_pose, dict):
        return {}, f"{label} is missing"
    pose: dict[str, float] = {}
    for field in _CARTESIAN_POSE_FIELDS:
        try:
            value = float(raw_pose[field])
        except (KeyError, TypeError, ValueError):
            return {}, f"{label}.{field} is missing or invalid"
        if not isfinite(value):
            return {}, f"{label}.{field} is not finite"
        pose[field] = value
    quaternion_norm = sqrt(sum(pose[field] ** 2 for field in ("qx", "qy", "qz", "qw")))
    if quaternion_norm <= 1e-12:
        return {}, f"{label} quaternion is zero"
    for field in ("qx", "qy", "qz", "qw"):
        pose[field] /= quaternion_norm
    return pose, ""


def _sanitized_move_insert_result(  # noqa: C901, PLR0912 - diagnostics are explicit.
    raw_result: Any,
) -> dict[str, Any]:
    raw = dict(raw_result) if isinstance(raw_result, dict) else {}
    sanitized = {
        field: deepcopy(raw[field])
        for field in _MOVE_INSERT_FAILURE_RESULT_FIELDS
        if field in raw
        and isinstance(raw[field], (bool, int, float, str))
        and not (
            isinstance(raw[field], float) and not isfinite(raw[field])
        )
    }
    for pose_field in ("absolute_position", "final_tool0_pose"):
        pose, pose_error = _normalized_se3_pose(
            raw.get(pose_field),
            label=f"move_insert {pose_field}",
        )
        if not pose_error:
            sanitized[pose_field] = pose
    for vector_field in _MOVE_INSERT_FAILURE_RESULT_VECTOR_FIELDS:
        raw_vector = raw.get(vector_field)
        if not isinstance(raw_vector, (list, tuple)) or len(raw_vector) != 6:
            continue
        try:
            vector = [float(value) for value in raw_vector]
        except (TypeError, ValueError, OverflowError):
            continue
        if all(isfinite(value) for value in vector):
            sanitized[vector_field] = vector
    raw_trace = raw.get("feedback_trace")
    if isinstance(raw_trace, list):
        observed_depths: list[float] = []
        for raw_sample in raw_trace:
            if not isinstance(raw_sample, dict):
                continue
            try:
                insertion_depth_m = float(raw_sample.get("insertion_depth_m"))
            except (TypeError, ValueError, OverflowError):
                continue
            if isfinite(insertion_depth_m):
                observed_depths.append(max(0.0, insertion_depth_m))
        if observed_depths:
            sanitized["max_insertion_depth_m"] = max(observed_depths)
    server_trace_sha256 = str(raw.get("server_trace_sha256") or "")
    try:
        server_trace_sha256_valid = bool(
            len(server_trace_sha256) == 64
            and int(server_trace_sha256, 16) >= 0
        )
    except ValueError:
        server_trace_sha256_valid = False
    server_trace_id = str(raw.get("server_trace_id") or "")
    server_trace_path = Path(str(raw.get("server_trace_path") or ""))
    try:
        server_trace_sample_count = int(raw.get("server_trace_sample_count") or 0)
    except (TypeError, ValueError, OverflowError):
        server_trace_sample_count = 0
    authoritative_server_trace = bool(
        raw.get("server_trace_complete") is True
        and str(raw.get("server_trace_status") or "") == "complete"
        and server_trace_id
        and str(raw.get("trial_id") or "") == server_trace_id
        and server_trace_path.name == "trace.jsonl"
        and server_trace_path.parent.name == server_trace_id
        and server_trace_sha256_valid
        and server_trace_sample_count > 0
    )
    if authoritative_server_trace and isinstance(raw_trace, list):
        sanitized["feedback_trace"] = []
    elif isinstance(raw_trace, list):
        sanitized_trace: list[dict[str, Any]] = []
        for raw_sample in raw_trace[:_MOVE_INSERT_DIAGNOSTIC_TRACE_MAX_SAMPLES]:
            if not isinstance(raw_sample, dict):
                continue
            sample: dict[str, Any] = {}
            for field, value in raw_sample.items():
                if isinstance(value, (bool, int, str)) or (
                    isinstance(value, float) and isfinite(value)
                ):
                    sample[str(field)] = value
                elif isinstance(value, list):
                    finite_values: list[float] = []
                    for item in value:
                        if isinstance(item, bool):
                            finite_values = []
                            break
                        try:
                            numeric = float(item)
                        except (TypeError, ValueError, OverflowError):
                            finite_values = []
                            break
                        if not isfinite(numeric):
                            finite_values = []
                            break
                        finite_values.append(numeric)
                    if finite_values:
                        sample[str(field)] = finite_values
                elif isinstance(value, dict):
                    finite_mapping: dict[str, float] = {}
                    for nested_field, nested_value in value.items():
                        if isinstance(nested_value, bool):
                            finite_mapping = {}
                            break
                        try:
                            numeric = float(nested_value)
                        except (TypeError, ValueError, OverflowError):
                            finite_mapping = {}
                            break
                        if not isfinite(numeric):
                            finite_mapping = {}
                            break
                        finite_mapping[str(nested_field)] = numeric
                    if finite_mapping:
                        sample[str(field)] = finite_mapping
            if sample:
                sanitized_trace.append(sample)
        sanitized["feedback_trace"] = sanitized_trace
    return sanitized


def _move_insert_result_sha256(result: dict[str, Any]) -> str:
    """Return a stable identity for one settled move_insert result."""
    canonical = json.dumps(
        result,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _quaternion_multiply(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:
    lx, ly, lz, lw = (left[field] for field in ("qx", "qy", "qz", "qw"))
    rx, ry, rz, rw = (right[field] for field in ("qx", "qy", "qz", "qw"))
    result = {
        "qx": lw * rx + lx * rw + ly * rz - lz * ry,
        "qy": lw * ry - lx * rz + ly * rw + lz * rx,
        "qz": lw * rz + lx * ry - ly * rx + lz * rw,
        "qw": lw * rw - lx * rx - ly * ry - lz * rz,
    }
    norm = sqrt(sum(value**2 for value in result.values()))
    return {field: value / norm for field, value in result.items()}


def _rotate_translation(
    quaternion: dict[str, float],
    translation: dict[str, float],
) -> dict[str, float]:
    qx, qy, qz, qw = (quaternion[field] for field in ("qx", "qy", "qz", "qw"))
    vx, vy, vz = (translation[field] for field in _CARTESIAN_POSITION_FIELDS)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return {
        "x": vx + qw * tx + qy * tz - qz * ty,
        "y": vy + qw * ty + qz * tx - qx * tz,
        "z": vz + qw * tz + qx * ty - qy * tx,
    }


def _compose_se3(
    parent_pose: dict[str, float],
    relative_pose: dict[str, float],
) -> dict[str, float]:
    rotated_translation = _rotate_translation(parent_pose, relative_pose)
    orientation = _quaternion_multiply(parent_pose, relative_pose)
    return {
        field: parent_pose[field] + rotated_translation[field]
        for field in _CARTESIAN_POSITION_FIELDS
    } | orientation


def _inverse_se3(pose: dict[str, float]) -> dict[str, float]:
    inverse_orientation = {
        "qx": -pose["qx"],
        "qy": -pose["qy"],
        "qz": -pose["qz"],
        "qw": pose["qw"],
    }
    inverse_translation = _rotate_translation(
        inverse_orientation,
        {field: -pose[field] for field in _CARTESIAN_POSITION_FIELDS},
    )
    return {**inverse_translation, **inverse_orientation}


def _held_part_handoff_from_context(
    task_context: dict[str, Any],
    *,
    part_name: str,
) -> tuple[dict[str, Any], str]:
    if not part_name or part_name != part_name.strip():
        return {}, "pick_grasp requires an exact non-empty part identifier"
    if task_context.get("part_name") != part_name:
        return {}, "pick_grasp part identity does not match pick_approach"
    world_tool0, tool_error = _normalized_se3_pose(
        dict(task_context.get("resolved_cartesian_positions") or {}).get("descend"),
        label="pick_approach.descend world -> tool0 pose",
    )
    if tool_error:
        return {}, tool_error
    world_part, part_error = _normalized_se3_pose(
        task_context.get("origin_pose"),
        label="pick_approach detected world -> held part origin pose",
    )
    if part_error:
        return {}, part_error
    provenance = task_context.get("origin_pose_provenance")
    if not isinstance(provenance, dict):
        return {}, "pick_approach detected part pose provenance is missing"
    if provenance.get("frame_id") != "world":
        return {}, "pick_approach detected part pose provenance frame_id must be world"
    if provenance.get("part_name") != part_name:
        return {}, "pick_approach detected part pose provenance part_name does not match"
    model_name = task_context.get("model_name")
    if provenance.get("model_name") not in (None, "", model_name):
        return {}, "pick_approach detected part pose provenance model_name does not match"
    tool0_to_held_part = _compose_se3(_inverse_se3(world_tool0), world_part)
    return {
        "part_name": part_name,
        "model_name": model_name,
        "origin_resource_location": task_context.get("origin_resource_location"),
        "frame_id": "world",
        "tool_frame": "tool0",
        "part_frame": "held_part_origin",
        "source": "pick_grasp",
        "world_tool0_pose_at_grasp": world_tool0,
        "world_held_part_pose_at_grasp": world_part,
        "tool0_to_held_part": tool0_to_held_part,
        "origin_pose_provenance": deepcopy(provenance),
    }, ""


def _physical_place_approach_held_part_handoff_error(
    task_context: dict[str, Any],
    *,
    part_name: Any,
) -> str:
    """Validate the immutable held-part SE(3) handoff before placement motion."""
    if (
        not isinstance(part_name, str)
        or not part_name
        or part_name != part_name.strip()
    ):
        return "physical place_approach requires an exact non-empty part_name"
    if task_context.get("part_name") != part_name:
        return "physical place_approach task context part_name does not match"

    raw_handoff = task_context.get("held_part_handoff")
    if not isinstance(raw_handoff, dict):
        return "physical place_approach held_part_handoff is missing"
    exact_identity = {
        "part_name": part_name,
        "frame_id": "world",
        "tool_frame": "tool0",
        "part_frame": "held_part_origin",
    }
    mismatched_identity = [
        field
        for field, expected in exact_identity.items()
        if raw_handoff.get(field) != expected
    ]
    if mismatched_identity:
        return (
            "physical place_approach held_part_handoff identity is invalid: "
            f"{mismatched_identity}"
        )

    model_name = task_context.get("model_name")
    if (
        not isinstance(model_name, str)
        or not model_name
        or model_name != model_name.strip()
        or raw_handoff.get("model_name") != model_name
    ):
        return "physical place_approach held_part_handoff model_name does not match"
    origin_resource_location = task_context.get("origin_resource_location")
    if (
        not isinstance(origin_resource_location, str)
        or not origin_resource_location
        or origin_resource_location != origin_resource_location.strip()
        or raw_handoff.get("origin_resource_location") != origin_resource_location
    ):
        return (
            "physical place_approach held_part_handoff origin_resource_location "
            "does not match"
        )
    source = raw_handoff.get("source")
    if not isinstance(source, str) or not source or source != source.strip():
        return "physical place_approach held_part_handoff source is invalid"

    world_tool0, tool_error = _normalized_se3_pose(
        raw_handoff.get("world_tool0_pose_at_grasp"),
        label="physical place_approach held_part_handoff.world_tool0_pose_at_grasp",
    )
    if tool_error:
        return tool_error
    world_part, part_error = _normalized_se3_pose(
        raw_handoff.get("world_held_part_pose_at_grasp"),
        label=(
            "physical place_approach "
            "held_part_handoff.world_held_part_pose_at_grasp"
        ),
    )
    if part_error:
        return part_error
    tool0_to_part, transform_error = _normalized_se3_pose(
        raw_handoff.get("tool0_to_held_part"),
        label="physical place_approach held_part_handoff.tool0_to_held_part",
    )
    if transform_error:
        return transform_error

    recomposed_part = _compose_se3(world_tool0, tool0_to_part)
    translation_error_m = sqrt(
        sum(
            (recomposed_part[field] - world_part[field]) ** 2
            for field in _CARTESIAN_POSITION_FIELDS
        )
    )
    quaternion_alignment = abs(
        sum(
            recomposed_part[field] * world_part[field]
            for field in ("qx", "qy", "qz", "qw")
        )
    )
    rotation_error_rad = 2.0 * acos(
        max(-1.0, min(1.0, quaternion_alignment))
    )
    if translation_error_m > 1e-6 or rotation_error_rad > 1e-4:
        return (
            "physical place_approach held_part_handoff does not reconstruct its "
            "frozen pick poses"
        )

    provenance = raw_handoff.get("origin_pose_provenance")
    if not isinstance(provenance, dict):
        return "physical place_approach held_part_handoff provenance is missing"
    if (
        provenance.get("frame_id") != "world"
        or provenance.get("part_name") != part_name
        or provenance.get("model_name") not in (None, "", model_name)
    ):
        return "physical place_approach held_part_handoff provenance is invalid"
    return ""


def _validated_operator_confirmed_held_part_handoff(  # noqa: C901, PLR0912
    raw_handoff: Any,
    *,
    part_name: str = "MG",
) -> tuple[dict[str, Any], str]:
    """Validate an exact-part handoff from the confirmed pick recording."""
    model_name = _MOVE_INSERT_MODEL_MAP.get(part_name)
    if model_name is None:
        return {}, (
            "operator-confirmed held-part handoff requires exact part_name "
            "'SG', 'MG', 'LG', 'SCP', 'MCP', or 'LCP'"
        )
    if not isinstance(raw_handoff, dict):
        return {}, "operator-confirmed held-part handoff is missing"
    required_fields = {
        "robot",
        "destination_location",
        "part_name",
        "model_name",
        "origin_resource_location",
        "frame_id",
        "tool_frame",
        "part_frame",
        "source",
        "orientation_source",
        "captured_at",
        "current_world_tool0_pose",
        "current_tf_stamp_sec",
        "pick_approach_recording_path",
        "pick_approach_recording_sha256",
        "pick_approach_recorded_at",
        "world_tool0_pose_at_grasp",
        "world_held_part_pose_at_grasp",
        "tool0_to_held_part",
        "origin_pose_provenance",
    }
    missing_fields = sorted(required_fields - set(raw_handoff))
    unknown_fields = sorted(set(raw_handoff) - required_fields)
    if missing_fields or unknown_fields:
        details: list[str] = []
        if missing_fields:
            details.append(f"missing fields: {missing_fields}")
        if unknown_fields:
            details.append(f"unknown fields: {unknown_fields}")
        return {}, "operator-confirmed held-part handoff has " + "; ".join(details)

    exact_identity = {
        "robot": "ur5e",
        "destination_location": "assembly_board-v1",
        "part_name": part_name,
        "model_name": model_name,
        "frame_id": "world",
        "tool_frame": "tool0",
        "part_frame": "held_part_origin",
        "source": "operator_confirmed_pick_approach_recording",
        "orientation_source": "realsense_roboflow_identity",
    }
    mismatched_identity = [
        field
        for field, expected in exact_identity.items()
        if raw_handoff.get(field) != expected
    ]
    if mismatched_identity:
        return {}, (
            "operator-confirmed held-part handoff identity is invalid: "
            f"{mismatched_identity}"
        )
    origin_resource_location = raw_handoff.get("origin_resource_location")
    if (
        not isinstance(origin_resource_location, str)
        or not origin_resource_location
        or origin_resource_location != origin_resource_location.strip()
    ):
        return {}, "operator-confirmed held-part origin_resource_location is invalid"
    recording_path = raw_handoff.get("pick_approach_recording_path")
    if (
        not isinstance(recording_path, str)
        or not recording_path
        or recording_path != recording_path.strip()
    ):
        return {}, "operator-confirmed pick_approach recording path is invalid"
    recording_sha256 = raw_handoff.get("pick_approach_recording_sha256")
    if not isinstance(recording_sha256, str) or len(recording_sha256) != 64:
        return {}, "operator-confirmed pick_approach recording SHA-256 is invalid"
    try:
        int(recording_sha256, 16)
    except ValueError:
        return {}, "operator-confirmed pick_approach recording SHA-256 is invalid"
    try:
        numeric_evidence_fields = (
            "captured_at",
            "current_tf_stamp_sec",
            "pick_approach_recorded_at",
        )
        if any(
            isinstance(raw_handoff[field], bool)
            for field in numeric_evidence_fields
        ):
            raise TypeError
        captured_at = float(raw_handoff["captured_at"])
        current_tf_stamp_sec = float(raw_handoff["current_tf_stamp_sec"])
        recorded_at = float(raw_handoff["pick_approach_recorded_at"])
    except (TypeError, ValueError):
        return {}, "operator-confirmed held-part timestamps are invalid"
    age_sec = time.time() - captured_at
    if (
        not isfinite(captured_at)
        or captured_at <= 0.0
        or age_sec < -1.0
        or age_sec > _OPERATOR_CONFIRMED_HELD_PART_HANDOFF_MAX_AGE_SEC
    ):
        return {}, "operator-confirmed held-part current TF evidence is stale or invalid"
    if not isclose(captured_at, current_tf_stamp_sec, rel_tol=0.0, abs_tol=1e-6):
        return {}, "operator-confirmed held-part current TF timestamps do not match"
    if not isfinite(recorded_at) or recorded_at <= 0.0:
        return {}, "operator-confirmed pick_approach recorded_at is invalid"

    for pose_field in (
        "current_world_tool0_pose",
        "world_tool0_pose_at_grasp",
        "world_held_part_pose_at_grasp",
        "tool0_to_held_part",
    ):
        raw_pose = raw_handoff.get(pose_field)
        if (
            not isinstance(raw_pose, dict)
            or set(raw_pose) != set(_CARTESIAN_POSE_FIELDS)
            or any(
                isinstance(raw_pose[field], bool)
                for field in _CARTESIAN_POSE_FIELDS
            )
        ):
            return {}, f"operator-confirmed held-part {pose_field} is invalid"

    current_world_tool0, current_tool_error = _normalized_se3_pose(
        raw_handoff.get("current_world_tool0_pose"),
        label="operator-confirmed current world -> tool0 pose",
    )
    if current_tool_error:
        return {}, current_tool_error

    world_tool0, tool_error = _normalized_se3_pose(
        raw_handoff.get("world_tool0_pose_at_grasp"),
        label="operator-confirmed pick_approach.descend world -> tool0 pose",
    )
    if tool_error:
        return {}, tool_error
    world_part, part_error = _normalized_se3_pose(
        raw_handoff.get("world_held_part_pose_at_grasp"),
        label="operator-confirmed pick reference world -> held part origin pose",
    )
    if part_error:
        return {}, part_error
    supplied_tool0_to_part, transform_error = _normalized_se3_pose(
        raw_handoff.get("tool0_to_held_part"),
        label="operator-confirmed tool0 -> held part transform",
    )
    if transform_error:
        return {}, transform_error
    derived_tool0_to_part = _compose_se3(_inverse_se3(world_tool0), world_part)
    translation_error_m = sqrt(
        sum(
            (
                supplied_tool0_to_part[field]
                - derived_tool0_to_part[field]
            )
            ** 2
            for field in _CARTESIAN_POSITION_FIELDS
        )
    )
    quaternion_dot = abs(
        sum(
            supplied_tool0_to_part[field] * derived_tool0_to_part[field]
            for field in ("qx", "qy", "qz", "qw")
        )
    )
    rotation_error_rad = 2.0 * acos(max(-1.0, min(1.0, quaternion_dot)))
    if translation_error_m > 1e-6 or rotation_error_rad > 1e-4:
        return {}, (
            "operator-confirmed tool0 -> held part transform does not recompose "
            "the confirmed pick poses"
        )

    provenance = raw_handoff.get("origin_pose_provenance")
    required_provenance = {
        "frame_id",
        "part_name",
        "model_name",
        "source",
        "orientation_source",
        "captured_at",
        "pick_approach_recording_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        return {}, "operator-confirmed held-part origin pose provenance is invalid"
    if (
        provenance.get("frame_id") != "world"
        or provenance.get("part_name") != part_name
        or provenance.get("model_name") != model_name
        or provenance.get("source")
        != "operator_confirmed_pick_approach_recording"
        or provenance.get("orientation_source") != "realsense_roboflow_identity"
        or provenance.get("pick_approach_recording_sha256") != recording_sha256
    ):
        return {}, "operator-confirmed held-part origin pose provenance is invalid"
    try:
        provenance_captured_at = float(provenance["captured_at"])
    except (TypeError, ValueError):
        return {}, "operator-confirmed held-part origin pose captured_at is invalid"
    if not isfinite(provenance_captured_at) or provenance_captured_at <= 0.0:
        return {}, "operator-confirmed held-part origin pose captured_at is invalid"

    return {
        **{field: deepcopy(raw_handoff[field]) for field in exact_identity},
        "origin_resource_location": origin_resource_location,
        "captured_at": captured_at,
        "current_world_tool0_pose": current_world_tool0,
        "current_tf_stamp_sec": current_tf_stamp_sec,
        "pick_approach_recording_path": recording_path,
        "pick_approach_recording_sha256": recording_sha256,
        "pick_approach_recorded_at": recorded_at,
        "world_tool0_pose_at_grasp": world_tool0,
        "world_held_part_pose_at_grasp": world_part,
        "tool0_to_held_part": supplied_tool0_to_part,
        "origin_pose_provenance": {
            **deepcopy(provenance),
            "captured_at": provenance_captured_at,
        },
    }, ""


def _assembly_board_v1_aruco_payload(
    raw_payload: Any,
    *,
    robot: str,
    destination_location: str,
    require_fresh: bool,
) -> tuple[dict[str, Any], str]:
    if not isinstance(raw_payload, dict):
        return {}, "assembly_board-v1 ArUco localization payload is missing"
    payload = dict(raw_payload)
    if str(payload.get("destination_location") or "").strip() != destination_location:
        return {}, "assembly_board-v1 ArUco destination_location does not match the task"
    camera_role = str(payload.get("camera_role") or payload.get("robot") or "").strip().lower()
    if not robot or camera_role != robot:
        return {}, (
            "assembly_board-v1 ArUco camera_role does not match the executing robot: "
            f"expected {robot or '<unknown>'!r}, found {camera_role or '<empty>'!r}"
        )
    if str(payload.get("frame_id") or "").strip() != "world":
        return {}, "assembly_board-v1 ArUco frame_id must be world"
    generation = payload.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        return {}, "assembly_board-v1 ArUco generation must be a positive integer"
    calibration_id = str(payload.get("calibration_id") or "").strip()
    if not calibration_id:
        return {}, "assembly_board-v1 ArUco calibration_id is missing"
    try:
        captured_at = float(payload["captured_at"])
    except (KeyError, TypeError, ValueError):
        return {}, "assembly_board-v1 ArUco captured_at is missing or invalid"
    if not isfinite(captured_at) or captured_at <= 0.0:
        return {}, "assembly_board-v1 ArUco captured_at is invalid"
    if require_fresh:
        age_sec = time.time() - captured_at
        if age_sec < -1.0 or age_sec > _ASSEMBLY_BOARD_V1_ARUCO_MAX_AGE_SEC:
            return {}, f"assembly_board-v1 ArUco pose is stale (age={age_sec:.2f}s)"
    pose, pose_error = _normalized_se3_pose(
        payload.get("pose"),
        label="assembly_board-v1 ArUco pose",
    )
    if pose_error:
        return {}, pose_error
    return {
        "destination_location": destination_location,
        "camera_role": camera_role,
        "generation": deepcopy(generation),
        "calibration_id": calibration_id,
        "captured_at": captured_at,
        "frame_id": "world",
        "pose": pose,
    }, ""


def _build_runtime_state(agent: Any) -> dict[str, Any]:
    return {
        "_held_part": deepcopy(getattr(agent, "_held_part", None)),
        "_current_state": deepcopy(getattr(agent, "_current_state", "")),
        "_position": deepcopy(getattr(agent, "_position", {})),
        "_gripper_state": deepcopy(getattr(agent, "_gripper_state", "")),
        "_recovery_pose_ref": deepcopy(getattr(agent, "_recovery_pose_ref", None)),
        "_task_ctx": deepcopy(getattr(agent, "_task_ctx", {})),
    }


def _commit_runtime_state(agent: Any, runtime_state: dict[str, Any]) -> None:
    for field in (
        "_held_part",
        "_current_state",
        "_position",
        "_gripper_state",
        "_recovery_pose_ref",
        "_task_ctx",
    ):
        setattr(agent, field, deepcopy(runtime_state.get(field)))


def _primitive_payload_from_result(
    *,
    step: RobotTaskStep,
    params: dict[str, Any],
    primitive_result: dict[str, Any],
    runtime_state: dict[str, Any],
) -> Any:
    payload: Any = None
    if step.store_as == "pick_targets":
        payload = _normalize_pick_targets(
            {
                key: value
                for key, value in dict(primitive_result or {}).items()
                if key not in {"success", "message"}
            }
        )
    elif step.store_as == "place_targets":
        payload = _normalize_place_targets(
            {
                key: value
                for key, value in dict(primitive_result or {}).items()
                if key not in {"success", "message"}
            },
            runtime_state=runtime_state,
        )
    elif step.store_as == "assembly_board_v1_aruco":
        raw_data = primitive_result.get("data")
        payload = deepcopy(
            raw_data
            if isinstance(raw_data, dict)
            else {
                key: value
                for key, value in dict(primitive_result or {}).items()
                if key not in {"success", "message"}
            }
        )
    elif isinstance(primitive_result.get("data"), (dict, list)):
        payload = deepcopy(primitive_result.get("data"))
    elif isinstance(primitive_result.get("observation"), dict):
        payload = deepcopy(dict(primitive_result.get("observation") or {}))

    if step.op in {"move_cartesian", "move_insert"}:
        payload = dict(payload or {})
        absolute_position, primitive_pose_error = _normalized_se3_pose(
            primitive_result.get("absolute_position"),
            label=f"controller result pose for {step.id}",
        )
        if primitive_pose_error and step.op == "move_cartesian":
            absolute_position = {
                "x": float(params.get("x", 0.0) or 0.0),
                "y": float(params.get("y", 0.0) or 0.0),
                "z": float(params.get("z", 0.0) or 0.0),
            }
            orientation_fields = ("qx", "qy", "qz", "qw")
            if all(params.get(field) is not None for field in orientation_fields):
                full_pose, full_pose_error = _normalized_se3_pose(
                    {
                        **absolute_position,
                        **{field: params[field] for field in orientation_fields},
                    },
                    label=f"resolved pose for {step.id}",
                )
                if not full_pose_error:
                    absolute_position = full_pose
        if primitive_pose_error and step.op == "move_insert":
            payload["pose_error"] = primitive_pose_error
        else:
            payload["absolute_position"] = absolute_position
    elif step.op == "move_relative":
        position = dict(runtime_state.get("_position") or {})
        payload = dict(payload or {})
        payload["absolute_position"] = {
            "x": float(position.get("x", 0.0) or 0.0) + float(params.get("dx", 0.0) or 0.0),
            "y": float(position.get("y", 0.0) or 0.0) + float(params.get("dy", 0.0) or 0.0),
            "z": float(position.get("z", 0.0) or 0.0) + float(params.get("dz", 0.0) or 0.0),
        }
    return payload


def _robot_name(agent: Any) -> str:
    scope_name = getattr(agent, "_robot_scope_name", None)
    if callable(scope_name):
        try:
            token = str(scope_name() or "").strip().lower()
        except (AttributeError, TypeError, ValueError):
            token = ""
        if token:
            return token.split("@", 1)[0]
    token = (
        str(getattr(agent, "agent_name", "") or getattr(agent, "name", "") or "").strip().lower()
    )
    return token.split("@", 1)[0]


def _cartesian_position_steps(task: RobotTaskDefinition) -> tuple[RobotTaskStep, ...]:
    return tuple(step for step in task.program.steps if step.op == "move_cartesian")


def _recorded_joints_error(
    *,
    agent: Any,
    robot: str,
    task: RobotTaskDefinition,
    step: RobotTaskStep,
    waypoint: dict[str, Any],
) -> str:
    joint_names = waypoint.get("joint_names")
    joint_positions = waypoint.get("joint_positions")
    if (
        not isinstance(joint_names, list)
        or not isinstance(joint_positions, list)
        or not joint_names
        or len(joint_names) != len(joint_positions)
        or any(not str(name or "").strip() for name in joint_names)
        or len({str(name).strip() for name in joint_names}) != len(joint_names)
    ):
        return f"physical position joints are incomplete for {task.name}.{step.id}"
    try:
        finite_joint_positions = [float(value) for value in joint_positions]
    except (TypeError, ValueError):
        return f"physical position joints are invalid for {task.name}.{step.id}"
    if not all(isfinite(value) for value in finite_joint_positions):
        return f"physical position joints are not finite for {task.name}.{step.id}"
    controller = getattr(agent, "_controller", None)
    controller_config = dict(
        getattr(agent, "controller_config", {})
        or getattr(controller, "controller_config", {})
        or {}
    )
    expected_joint_names = tuple(
        str(name).strip()
        for name in list(
            getattr(controller, "arm_joint_names", [])
            or controller_config.get("arm_joint_names")
            or []
        )
        if str(name or "").strip()
    ) or _HARDWARE_JOINT_NAMES.get(robot)
    recorded_joint_names = tuple(str(name).strip() for name in joint_names)
    if expected_joint_names and set(recorded_joint_names) != set(expected_joint_names):
        return (
            f"physical position joints do not match the configured {robot} hardware "
            f"set for {task.name}.{step.id}"
        )
    return ""


def _configured_recording_frames(
    agent: Any,
) -> tuple[str, str, str]:
    """Return configured world, end-effector, and TCP frames for one robot."""
    controller = getattr(agent, "_controller", None)
    controller_config = dict(
        getattr(agent, "controller_config", {})
        or getattr(controller, "controller_config", {})
        or {}
    )
    move_group = dict(controller_config.get("move_group") or {})
    frame_id = str(
        getattr(controller, "frame_id", "")
        or move_group.get("frame_id")
    ).strip()
    ee_link = str(
        getattr(controller, "ee_link", "")
        or move_group.get("ee_link")
    ).strip()
    tcp_link = str(
        getattr(controller, "tcp_link", "")
        or move_group.get("tcp_link")
    ).strip()
    return frame_id, ee_link, tcp_link


def _recorded_cartesian_pose(  # noqa: C901 - correction validation is intentionally explicit.
    *,
    agent: Any,
    robot: str,
    task: RobotTaskDefinition,
    step: RobotTaskStep,
    recorded_step: dict[str, Any],
) -> tuple[dict[str, float], str]:
    primitive = str(recorded_step.get("primitive") or "").strip()
    if primitive != step.op or primitive != "move_cartesian":
        return {}, (
            f"physical position primitive mismatch for {task.name}.{step.id}: "
            f"expected move_cartesian, found {primitive or '<empty>'}"
        )

    waypoint = recorded_step.get("waypoint")
    if not isinstance(waypoint, dict):
        return {}, f"physical position waypoint is missing for {task.name}.{step.id}"
    if str(recorded_step.get("capture_source") or "").strip().lower() != "hardware":
        return {}, f"physical position capture_source must be hardware for {task.name}.{step.id}"
    if str(waypoint.get("source") or "").strip().lower() != "hardware":
        return {}, f"physical position source must be hardware for {task.name}.{step.id}"
    joint_error = _recorded_joints_error(
        agent=agent,
        robot=robot,
        task=task,
        step=step,
        waypoint=waypoint,
    )
    if joint_error:
        return {}, joint_error
    pose = waypoint.get("pose")
    if not isinstance(pose, dict):
        return {}, f"physical position pose is missing for {task.name}.{step.id}"
    expected_frame, expected_child_frame, _tcp_link = _configured_recording_frames(agent)
    if not expected_frame:
        return {}, f"configured frame_id is missing for {robot}"
    if str(pose.get("frame_id") or "").strip() != expected_frame:
        return {}, (
            f"physical position frame must be {expected_frame} for {task.name}.{step.id}"
        )
    if not expected_child_frame:
        return {}, f"configured ee_link is missing for {robot}"
    if str(pose.get("child_frame_id") or "").strip() != expected_child_frame:
        return {}, (
            f"physical position child frame must be {expected_child_frame} for "
            f"{task.name}.{step.id}"
        )

    recorded_params = recorded_step.get("params")
    if not isinstance(recorded_params, dict):
        return {}, f"physical position params are missing for {task.name}.{step.id}"

    result: dict[str, float] = {}
    for field_name in _CARTESIAN_POSE_FIELDS:
        try:
            pose_value = float(pose[field_name])
            param_value = float(recorded_params[field_name])
        except (KeyError, TypeError, ValueError):
            return {}, (
                f"physical position {field_name} is missing or invalid for {task.name}.{step.id}"
            )
        if not isfinite(pose_value) or not isfinite(param_value):
            return {}, f"physical position {field_name} is not finite for {task.name}.{step.id}"
        if abs(pose_value - param_value) > 1e-9:
            return {}, (
                f"physical position {field_name} differs between params and waypoint for "
                f"{task.name}.{step.id}"
            )
        result[field_name] = pose_value

    quaternion_norm_squared = sum(result[name] ** 2 for name in ("qx", "qy", "qz", "qw"))
    if quaternion_norm_squared <= 1e-12:
        return {}, f"physical position quaternion is zero for {task.name}.{step.id}"
    return result, ""


def _recorded_relative_position(  # noqa: C901, PLR0912 - explicit persisted-field validation.
    *,
    robot: str,
    task: RobotTaskDefinition,
    step: RobotTaskStep,
    recorded_step: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any], str]:
    raw_relative = recorded_step.get("relative_position_m")
    if not isinstance(raw_relative, dict):
        return {}, {}, f"relative_position_m is missing for {task.name}.{step.id}"
    relative_position_m: dict[str, float] = {}
    for field in _CARTESIAN_POSITION_FIELDS:
        try:
            value = float(raw_relative[field])
        except (KeyError, TypeError, ValueError):
            return {}, {}, (
                f"relative_position_m.{field} is missing or invalid for "
                f"{task.name}.{step.id}"
            )
        if not isfinite(value):
            return {}, {}, (
                f"relative_position_m.{field} is not finite for {task.name}.{step.id}"
            )
        relative_position_m[field] = value

    raw_reference = recorded_step.get("relative_reference")
    if not isinstance(raw_reference, dict):
        return {}, {}, f"relative_reference is missing for {task.name}.{step.id}"
    expected_kind = "detected_part" if task.name == "pick_approach" else "destination_target"
    kind = str(raw_reference.get("kind") or "").strip()
    frame_id = str(raw_reference.get("frame_id") or "").strip()
    name = str(raw_reference.get("name") or "").strip()
    source = str(raw_reference.get("source") or "").strip()
    if kind != expected_kind:
        return {}, {}, (
            f"relative_reference.kind must be {expected_kind} for {task.name}.{step.id}"
        )
    if frame_id != "world":
        return {}, {}, (
            f"relative_reference.frame_id must be world for {task.name}.{step.id}"
        )
    if not name:
        return {}, {}, f"relative_reference.name is missing for {task.name}.{step.id}"
    if not source:
        return {}, {}, f"relative_reference.source is missing for {task.name}.{step.id}"
    raw_reference_position = raw_reference.get("position_m")
    if not isinstance(raw_reference_position, dict):
        return {}, {}, (
            f"relative_reference.position_m is missing for {task.name}.{step.id}"
        )
    reference_position_m: dict[str, float] = {}
    for field in _CARTESIAN_POSITION_FIELDS:
        try:
            value = float(raw_reference_position[field])
        except (KeyError, TypeError, ValueError):
            return {}, {}, (
                f"relative_reference.position_m.{field} is missing or invalid for "
                f"{task.name}.{step.id}"
            )
        if not isfinite(value):
            return {}, {}, (
                f"relative_reference.position_m.{field} is not finite for "
                f"{task.name}.{step.id}"
            )
        reference_position_m[field] = value
    try:
        captured_at = float(raw_reference["captured_at"])
    except (KeyError, TypeError, ValueError):
        return {}, {}, (
            f"relative_reference.captured_at is missing or invalid for {task.name}.{step.id}"
        )
    if not isfinite(captured_at) or captured_at <= 0.0:
        return {}, {}, (
            f"relative_reference.captured_at is invalid for {task.name}.{step.id}"
        )
    reference = {
        "kind": kind,
        "frame_id": frame_id,
        "name": name,
        "position_m": reference_position_m,
        "source": source,
        "captured_at": captured_at,
    }
    if raw_reference.get("camera_role") is not None:
        camera_role = str(raw_reference.get("camera_role") or "").strip().lower()
        if camera_role != robot:
            return {}, {}, (
                "relative_reference.camera_role does not match the executing robot for "
                f"{task.name}.{step.id}"
            )
        generation = raw_reference.get("generation")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
        ):
            return {}, {}, (
                "relative_reference.generation must be a positive integer for "
                f"{task.name}.{step.id}"
            )
        reference.update({"camera_role": camera_role, "generation": deepcopy(generation)})
        if task.name == "place_approach" and name == "assembly_board-v1":
            calibration_id = raw_reference.get("calibration_id")
            if (
                not isinstance(calibration_id, str)
                or not calibration_id
                or calibration_id != calibration_id.strip()
            ):
                return {}, {}, (
                    "relative_reference.calibration_id is missing or invalid for "
                    f"{task.name}.{step.id}"
                )
            reference["calibration_id"] = calibration_id
            if raw_reference.get("pose") is None:
                return {}, {}, (
                    f"relative_reference.pose is missing for {task.name}.{step.id}"
                )
        if raw_reference.get("pose") is not None:
            reference_pose, reference_pose_error = _normalized_se3_pose(
                raw_reference.get("pose"),
                label=f"relative_reference.pose for {task.name}.{step.id}",
            )
            if reference_pose_error:
                return {}, {}, reference_pose_error
            if any(
                abs(reference_pose[field] - reference_position_m[field]) > 1e-9
                for field in _CARTESIAN_POSITION_FIELDS
            ):
                return {}, {}, (
                    "relative_reference.pose does not match position_m for "
                    f"{task.name}.{step.id}"
                )
            reference["pose"] = reference_pose
    return relative_position_m, reference, ""


def _recorded_relative_pose(
    *,
    task: RobotTaskDefinition,
    step: RobotTaskStep,
    recorded_step: dict[str, Any],
    reference_pose: dict[str, float],
    captured_pose: dict[str, float],
) -> tuple[dict[str, float], str]:
    relative_pose, error = _normalized_se3_pose(
        recorded_step.get("relative_pose"),
        label=f"relative_pose for {task.name}.{step.id}",
    )
    if error:
        return {}, error
    reconstructed_pose = _compose_se3(reference_pose, relative_pose)
    normalized_captured_pose, captured_pose_error = _normalized_se3_pose(
        captured_pose,
        label=f"captured pose for {task.name}.{step.id}",
    )
    if captured_pose_error:
        return {}, captured_pose_error
    if any(
        abs(reconstructed_pose[field] - normalized_captured_pose[field]) > 1e-6
        for field in _CARTESIAN_POSITION_FIELDS
    ):
        return {}, (
            f"relative_pose translation does not reconstruct the captured pose for "
            f"{task.name}.{step.id}"
        )
    quaternion_alignment = abs(
        sum(
            reconstructed_pose[field] * normalized_captured_pose[field]
            for field in ("qx", "qy", "qz", "qw")
        )
    )
    if 1.0 - quaternion_alignment > 1e-6:
        return {}, (
            f"relative_pose orientation does not reconstruct the captured pose for "
            f"{task.name}.{step.id}"
        )
    return relative_pose, ""


def _recorded_cartesian_override(  # noqa: C901, PLR0912 - explicit legacy and relative-source gates.
    *,
    agent: Any,
    robot: str,
    task: RobotTaskDefinition,
    step: RobotTaskStep,
    recorded_step: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    pose, error = _recorded_cartesian_pose(
        agent=agent,
        robot=robot,
        task=task,
        step=step,
        recorded_step=recorded_step,
    )
    if error:
        return {}, error

    raw_sources = recorded_step.get("position_sources")
    if raw_sources is None:
        position_sources = {field: "captured" for field in _CARTESIAN_POSITION_FIELDS}
    elif not isinstance(raw_sources, dict):
        return {}, f"physical position_sources is invalid for {task.name}.{step.id}"
    else:
        position_sources = {
            field: str(raw_sources.get(field) or "").strip()
            for field in _CARTESIAN_POSITION_FIELDS
        }
    for field, source in position_sources.items():
        if source not in _POSITION_SOURCES:
            return {}, (
                f"physical position source for {task.name}.{step.id}.{field} must be "
                "computed, captured, manual, or captured_relative"
            )

    relative_fields = {
        field for field, source in position_sources.items() if source == "captured_relative"
    }
    if relative_fields != set(_CARTESIAN_POSITION_FIELDS):
        return {}, (
            f"captured_relative must be selected for x, y, and z for "
            f"{task.name}.{step.id}"
        )
    relative_position_m: dict[str, float] = {}
    relative_pose: dict[str, float] = {}
    relative_reference: dict[str, Any] = {}
    computed_position_m: dict[str, float] = {}
    computed_pose: dict[str, float] = {}
    computed_source = ""
    computed_at: float | None = None
    relative_position_m, relative_reference, relative_error = (
        _recorded_relative_position(
            robot=robot,
            task=task,
            step=step,
            recorded_step=recorded_step,
        )
    )
    if relative_error:
        return {}, relative_error
    raw_computed_position = recorded_step.get("computed_position_m")
    if not isinstance(raw_computed_position, dict):
        return {}, f"computed_position_m is invalid for {task.name}.{step.id}"
    for field in _CARTESIAN_POSITION_FIELDS:
        try:
            value = float(raw_computed_position[field])
        except (KeyError, TypeError, ValueError):
            return {}, (
                f"computed_position_m.{field} is missing or invalid for "
                f"{task.name}.{step.id}"
            )
        if not isfinite(value):
            return {}, (
                f"computed_position_m.{field} is not finite for {task.name}.{step.id}"
            )
        computed_position_m[field] = value
    computed_source = str(recorded_step.get("computed_source") or "").strip()
    try:
        computed_at = float(recorded_step["computed_at"])
    except (KeyError, TypeError, ValueError):
        return {}, f"computed_at is missing or invalid for {task.name}.{step.id}"
    if not computed_source:
        return {}, f"computed_source is missing for {task.name}.{step.id}"
    if not isfinite(computed_at) or computed_at <= 0.0:
        return {}, f"computed_at is invalid for {task.name}.{step.id}"
    raw_computed_pose = recorded_step.get("computed_pose")
    if raw_computed_pose is None and task.name == "place_approach":
        return {}, (
            f"Unsafe legacy {task.name}.{step.id} correction: computed_pose is "
            "missing; use Locate & Accept Board, Capture Pose, and Save/Replace "
            "Pose again before physical Run place_approach."
        )
    if raw_computed_pose is None:
        computed_pose = {
            **computed_position_m,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
    else:
        computed_pose, computed_pose_error = _normalized_se3_pose(
            raw_computed_pose,
            label=f"computed_pose for {task.name}.{step.id}",
        )
        if computed_pose_error:
            return {}, computed_pose_error
        if any(
            abs(computed_pose[field] - computed_position_m[field]) > 1e-9
            for field in _CARTESIAN_POSITION_FIELDS
        ):
            return {}, (
                f"computed_pose XYZ does not match computed_position_m for "
                f"{task.name}.{step.id}"
            )
    relative_pose, relative_pose_error = _recorded_relative_pose(
        task=task,
        step=step,
        recorded_step=recorded_step,
        reference_pose=computed_pose,
        captured_pose=pose,
    )
    if relative_pose_error:
        return {}, relative_pose_error

    raw_manual = recorded_step.get("manual_position_m", {})
    if not isinstance(raw_manual, dict):
        return {}, f"manual_position_m is invalid for {task.name}.{step.id}"
    manual_position_m: dict[str, float] = {}
    resolved_values: dict[str, float] = {
        field: float(pose[field]) for field in ("qx", "qy", "qz", "qw")
    }
    for field, source in position_sources.items():
        if source in {"computed", "captured_relative"}:
            continue
        if source == "captured":
            resolved_values[field] = float(pose[field])
            continue
        try:
            value = float(raw_manual[field])
        except (KeyError, TypeError, ValueError):
            return {}, f"manual position is missing for {task.name}.{step.id}.{field}"
        if not isfinite(value):
            return {}, f"manual position is not finite for {task.name}.{step.id}.{field}"
        manual_position_m[field] = value
        resolved_values[field] = value

    return {
        "values": resolved_values,
        "position_sources": position_sources,
        "manual_position_m": manual_position_m,
        "relative_position_m": relative_position_m,
        "relative_pose": relative_pose,
        "relative_reference": relative_reference,
        "computed_position_m": computed_position_m,
        "computed_pose": computed_pose,
        "computed_source": computed_source,
        "computed_at": computed_at,
        "captured_pose": pose,
    }, ""


def _load_physical_cartesian_overrides(  # noqa: C901, PLR0912
    *,
    agent: Any,
    task: RobotTaskDefinition,
    args: dict[str, Any],
    allow_missing: bool = False,
) -> tuple[dict[str, dict[str, Any]], Path | None, str]:
    cartesian_steps = _cartesian_position_steps(task)
    if not cartesian_steps:
        return {}, None, ""

    robot = _robot_name(agent)
    if not robot:
        return (
            {},
            None,
            "physical position recording requires a configured robot name",
        )
    path = _TAUGHT_FUNCTIONS_ROOT / task.name / "default__hardware.json"
    try:
        with path.open("r", encoding="utf-8") as recording_file:
            payload = json.load(recording_file)
    except FileNotFoundError:
        return {}, path, ""
    except (OSError, json.JSONDecodeError) as exc:
        return {}, path, f"could not load physical position file {path}: {exc}"
    if not isinstance(payload, dict):
        return {}, path, f"physical position file is not a JSON object: {path}"

    if str(payload.get("function_name") or "").strip() != task.name:
        return {}, path, f"physical position function mismatch in {path}"
    if str(payload.get("capture_source") or "").strip().lower() != "hardware":
        return {}, path, f"physical position capture_source must be hardware in {path}"

    raw_robots = payload.get("robots")
    if not isinstance(raw_robots, dict):
        return {}, path, f"physical position robots must be an object in {path}"
    raw_robot_entry = raw_robots.get(robot)
    if raw_robot_entry is None:
        return {}, path, ""
    if not isinstance(raw_robot_entry, dict):
        return {}, path, f"physical position robot entry must be an object for {robot} in {path}"
    robot_entry = dict(raw_robot_entry)
    raw_steps = robot_entry.get("steps")
    if not isinstance(raw_steps, list):
        return {}, path, f"physical position steps must be a list in {path}"
    recorded_by_id: dict[str, dict[str, Any]] = {}
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            return {}, path, f"physical position step must be an object in {path}"
        invalid_reason = str(raw_step.get("invalid_reason") or "").strip()
        if task.name == "place_approach" and invalid_reason:
            step_id = str(raw_step.get("step_name") or "<unknown>").strip()
            return {}, path, f"{task.name}.{step_id}: {invalid_reason}"
        if raw_step.get("confirmed") is not True:
            continue
        step_id = str(raw_step.get("step_name") or "").strip()
        if not step_id:
            return {}, path, f"physical position step_name is empty in {path}"
        if step_id in recorded_by_id:
            return {}, path, f"duplicate physical position step_name {step_id} in {path}"
        recorded_by_id[step_id] = raw_step
    if not recorded_by_id:
        return {}, path, ""

    configured_frame, configured_ee_link, configured_tcp_link = (
        _configured_recording_frames(agent)
    )
    if not configured_frame or not configured_ee_link or not configured_tcp_link:
        return {}, path, (
            f"configured frame_id, ee_link, and tcp_link are required for {robot}"
        )
    if str(robot_entry.get("frame_id") or "").strip() != configured_frame:
        return {}, path, f"physical position frame_id does not match configured {robot}"
    if str(robot_entry.get("ee_link") or "").strip() != configured_ee_link:
        return {}, path, f"physical position ee_link does not match configured {robot}"
    if str(robot_entry.get("tcp_link") or "").strip() != configured_tcp_link:
        return {}, path, f"physical position tcp_link does not match configured {robot}"

    overrides: dict[str, dict[str, Any]] = {}
    for step in cartesian_steps:
        recorded_step = recorded_by_id.get(step.id)
        if recorded_step is None:
            continue
        override, error = _recorded_cartesian_override(
            agent=agent,
            robot=robot,
            task=task,
            step=step,
            recorded_step=recorded_step,
        )
        if error:
            return {}, path, error
        overrides[step.id] = override
    return overrides, path, ""


def _apply_place_approach_recording_qualification(
    targets: dict[str, Any],
    recording_path: Path | None,
) -> tuple[dict[str, Any], str]:
    """Downgrade a stale qualified recording to the supervised-trial profile."""
    move_insert_profile = targets.get("move_insert_profile")
    qualification = (
        dict(move_insert_profile.get("qualification") or {})
        if isinstance(move_insert_profile, dict)
        else {}
    )
    if not qualification:
        return targets, ""
    expected_recording_sha256 = str(
        qualification.get("place_approach_recording_sha256") or ""
    )
    try:
        if recording_path is not None and recording_path.is_file():
            current_recording_sha256 = hashlib.sha256(
                recording_path.read_bytes()
            ).hexdigest()
        else:
            current_recording_sha256 = hashlib.sha256(
                b"place_approach/default__hardware.json:missing"
            ).hexdigest()
    except OSError as exc:
        return targets, f"Could not verify the qualified place_approach recording: {exc}"
    if current_recording_sha256 == expected_recording_sha256:
        return targets, ""
    downgraded = deepcopy(targets)
    downgraded_profile = deepcopy(move_insert_profile)
    downgraded_profile["qualification"] = {}
    downgraded["move_insert_profile"] = downgraded_profile
    downgraded["move_insert_mode"] = "force_limited_trial"
    downgraded["move_insert_qualification_error"] = (
        "place_approach recording changed after supervised move_insert qualification"
    )
    return downgraded, ""


def _validated_move_insert_boundary(  # noqa: C901, PLR0912 - safety evidence is explicit.
    *,
    start_pose: Any,
    targets: dict[str, Any],
    raw_hard_caps: Any,
) -> tuple[dict[str, Any], str]:
    """Validate the exact resolved insertion boundary against RTDE hard caps."""
    raw_insert_pose = targets.get("insert_pose")
    raw_learned_pre_insert_pose = targets.get("pre_insert_pose")
    for label, raw_pose in (
        ("place_approach resolved descend pose", start_pose),
        ("place_approach learned pre_insert_pose", raw_learned_pre_insert_pose),
        ("place_approach exact insert_pose", raw_insert_pose),
    ):
        if not isinstance(raw_pose, dict) or any(
            isinstance(raw_pose.get(field), bool)
            for field in _CARTESIAN_POSE_FIELDS
        ):
            return {}, f"{label} is invalid"
    normalized_start, start_error = _normalized_se3_pose(
        start_pose,
        label="place_approach resolved descend pose",
    )
    normalized_learned_pre_insert, learned_pre_insert_error = _normalized_se3_pose(
        raw_learned_pre_insert_pose,
        label="place_approach learned pre_insert_pose",
    )
    normalized_insert, insert_error = _normalized_se3_pose(
        raw_insert_pose,
        label="place_approach exact insert_pose",
    )
    if start_error or learned_pre_insert_error or insert_error:
        return {}, start_error or learned_pre_insert_error or insert_error

    raw_profile = targets.get("move_insert_profile")
    if not isinstance(raw_profile, dict):
        return {}, "place_approach frozen move_insert_profile is missing"
    demonstration_recipe = raw_profile.get("demonstration_recipe")
    boundary_part_name = (
        str(demonstration_recipe.get("part_name") or "")
        if isinstance(demonstration_recipe, dict)
        else str(raw_profile.get("part_name") or "")
    )

    raw_axis = targets.get("insertion_axis_world")
    if not isinstance(raw_axis, dict) or any(
        isinstance(raw_axis.get(field), bool)
        for field in _CARTESIAN_POSITION_FIELDS
    ):
        return {}, "place_approach frozen insertion_axis_world is invalid"
    try:
        axis = {
            field: float(raw_axis[field])
            for field in _CARTESIAN_POSITION_FIELDS
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return {}, "place_approach frozen insertion_axis_world is invalid"
    axis_norm = sqrt(sum(value**2 for value in axis.values()))
    if not isfinite(axis_norm) or axis_norm <= 1e-12:
        return {}, "place_approach frozen insertion_axis_world is invalid"
    axis = {field: value / axis_norm for field, value in axis.items()}

    required_caps = (
        "insert_max_travel_m",
        "insert_start_position_tolerance_m",
        "insert_start_orientation_tolerance_rad",
        "insert_max_timeout_sec",
        *(
            ("insert_max_contact_search_radius_m",)
            if boundary_part_name in _MOVE_INSERT_TRANSLATIONAL_PARTS
            else ()
        ),
    )
    if not isinstance(raw_hard_caps, dict):
        return {}, "place_approach verified live move_insert hard caps are missing"
    hard_caps: dict[str, float] = {}
    for cap_name, raw_value in raw_hard_caps.items():
        if not isinstance(cap_name, str) or not cap_name:
            return {}, "place_approach move_insert hard caps are invalid"
        if isinstance(raw_value, bool):
            return {}, f"place_approach move_insert hard cap {cap_name} is invalid"
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            return {}, f"place_approach move_insert hard cap {cap_name} is invalid"
        if not isfinite(value) or value <= 0.0:
            return {}, f"place_approach move_insert hard cap {cap_name} is invalid"
        hard_caps[cap_name] = value
    for cap_name in required_caps:
        if cap_name not in hard_caps:
            return {}, f"place_approach move_insert hard cap {cap_name} is invalid"
    canonical_hard_caps = json.dumps(
        {name: value for name, value in sorted(hard_caps.items())},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    hard_caps_sha256 = hashlib.sha256(
        canonical_hard_caps.encode("utf-8")
    ).hexdigest()

    delta = {
        field: normalized_insert[field] - normalized_start[field]
        for field in _CARTESIAN_POSITION_FIELDS
    }
    insertion_depth_m = sum(
        delta[field] * axis[field] for field in _CARTESIAN_POSITION_FIELDS
    )
    insertion_travel_m = sqrt(sum(value**2 for value in delta.values()))
    lateral_error_m = sqrt(
        sum(
            (delta[field] - insertion_depth_m * axis[field]) ** 2
            for field in _CARTESIAN_POSITION_FIELDS
        )
    )
    if not isfinite(insertion_depth_m) or insertion_depth_m <= 0.0:
        return {}, (
            "place_approach exact insert_pose must have positive progress along "
            "insertion_axis_world"
        )
    if insertion_travel_m > hard_caps["insert_max_travel_m"]:
        return {}, (
            f"place_approach resolved insertion travel {insertion_travel_m:.9g} m "
            "exceeds the verified live insert_max_travel_m hard cap"
        )
    lateral_limit_name = (
        "insert_max_contact_search_radius_m"
        if boundary_part_name in _MOVE_INSERT_TRANSLATIONAL_PARTS
        else "insert_start_position_tolerance_m"
    )
    if lateral_error_m > hard_caps[lateral_limit_name]:
        return {}, (
            "place_approach exact insert_pose is outside the verified live "
            f"{lateral_limit_name} of the resolved descend"
        )

    if boundary_part_name in _MOVE_INSERT_TRANSLATIONAL_PARTS:
        tool_axis = {"x": 0.0, "y": 0.0, "z": 1.0}
        start_tool_axis = _rotate_translation(normalized_start, tool_axis)
        insert_tool_axis = _rotate_translation(normalized_insert, tool_axis)
        tool_axis_alignment = sum(
            start_tool_axis[field] * insert_tool_axis[field]
            for field in _CARTESIAN_POSITION_FIELDS
        )
        orientation_error_rad = acos(
            max(-1.0, min(1.0, tool_axis_alignment))
        )
    else:
        quaternion_alignment = abs(
            sum(
                normalized_start[field] * normalized_insert[field]
                for field in ("qx", "qy", "qz", "qw")
            )
        )
        orientation_error_rad = 2.0 * acos(
            max(-1.0, min(1.0, quaternion_alignment))
        )
    if orientation_error_rad > hard_caps["insert_start_orientation_tolerance_rad"]:
        return {}, (
            "place_approach resolved descend and exact insert_pose orientations "
            "exceed the verified live insert_start_orientation_tolerance_rad"
        )

    learned_start_position_error_m = sqrt(
        sum(
            (
                normalized_start[field]
                - normalized_learned_pre_insert[field]
            )
            ** 2
            for field in _CARTESIAN_POSITION_FIELDS
        )
    )
    learned_start_quaternion_alignment = abs(
        sum(
            normalized_start[field] * normalized_learned_pre_insert[field]
            for field in ("qx", "qy", "qz", "qw")
        )
    )
    learned_start_orientation_error_rad = 2.0 * acos(
        max(-1.0, min(1.0, learned_start_quaternion_alignment))
    )
    for field in (
        "pre_insert_offset_m",
        "engagement_progress_m",
        "seated_depth_tolerance_m",
    ):
        raw_value = raw_profile.get(field)
        if isinstance(raw_value, bool):
            return {}, f"place_approach move_insert_profile.{field} is invalid"
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            return {}, f"place_approach move_insert_profile.{field} is invalid"
        if not isfinite(value) or value <= 0.0:
            return {}, f"place_approach move_insert_profile.{field} is invalid"
    execution_insert = deepcopy(normalized_insert)
    if boundary_part_name in _MOVE_INSERT_TRANSLATIONAL_PARTS:
        demonstrated_insertion_depth_m = float(
            raw_profile["pre_insert_offset_m"]
        )
        resolved_lateral_delta = {
            field: delta[field] - insertion_depth_m * axis[field]
            for field in _CARTESIAN_POSITION_FIELDS
        }
        execution_insert.update(
            {
                field: normalized_start[field]
                + resolved_lateral_delta[field]
                + demonstrated_insertion_depth_m * axis[field]
                for field in _CARTESIAN_POSITION_FIELDS
            }
        )
        execution_insert.update(
            {
                field: normalized_start[field]
                for field in ("qx", "qy", "qz", "qw")
            }
        )
        execution_delta = {
            field: execution_insert[field] - normalized_start[field]
            for field in _CARTESIAN_POSITION_FIELDS
        }
        insertion_depth_m = sum(
            execution_delta[field] * axis[field]
            for field in _CARTESIAN_POSITION_FIELDS
        )
        insertion_travel_m = sqrt(
            sum(value**2 for value in execution_delta.values())
        )
        lateral_error_m = sqrt(
            sum(
                (
                    execution_delta[field]
                    - insertion_depth_m * axis[field]
                )
                ** 2
                for field in _CARTESIAN_POSITION_FIELDS
            )
        )
        if insertion_travel_m > hard_caps["insert_max_travel_m"]:
            return {}, (
                "place_approach demonstrated insertion travel "
                f"{insertion_travel_m:.9g} m exceeds the verified live "
                "insert_max_travel_m hard cap"
            )
    if float(raw_profile["engagement_progress_m"]) > (
        insertion_depth_m + float(raw_profile["seated_depth_tolerance_m"])
    ):
        return {}, (
            "place_approach move_insert engagement_progress_m exceeds the "
            "available resolved insertion depth"
        )
    derived_timeout_sec, timeout_error = derive_move_insert_timeout_sec(
        normalized_start,
        execution_insert,
        axis,
        raw_profile,
        part_name=boundary_part_name,
        insert_max_timeout_sec=hard_caps["insert_max_timeout_sec"],
    )
    if timeout_error:
        return {}, timeout_error
    if derived_timeout_sec > hard_caps["insert_max_timeout_sec"]:
        return {}, (
            f"place_approach derived move_insert timeout {derived_timeout_sec:.9g} s "
            "exceeds the verified live insert_max_timeout_sec hard cap"
        )
    return {
        "pre_insert_pose": normalized_start,
        "insert_pose": execution_insert,
        "insertion_axis_world": axis,
        "move_insert_timeout_sec": derived_timeout_sec,
        "move_insert_hard_caps": hard_caps,
        "move_insert_hard_caps_sha256": hard_caps_sha256,
        "insertion_depth_m": insertion_depth_m,
        "insertion_travel_m": insertion_travel_m,
        "lateral_error_m": lateral_error_m,
        "orientation_error_rad": orientation_error_rad,
        "learned_start_position_error_m": learned_start_position_error_m,
        "learned_start_orientation_error_rad": (
            learned_start_orientation_error_rad
        ),
    }, ""


def _apply_resolved_cartesian_state(  # noqa: C901, PLR0912 - explicit pose gates.
    *,
    task: RobotTaskDefinition,
    physical_overrides: dict[str, dict[str, Any]],
    computed_positions: dict[str, dict[str, float]],
    computed_reference: dict[str, Any],
    computed_at: float,
    resolved_positions: dict[str, dict[str, float]],
    runtime_state: dict[str, Any],
    step_outputs: dict[str, Any],
) -> str:
    cartesian_steps = _cartesian_position_steps(task)
    if not cartesian_steps:
        return ""
    above_pose = resolved_positions.get(cartesian_steps[0].id)
    target_pose = resolved_positions.get(cartesian_steps[-1].id)
    task_context = dict(runtime_state.get("_task_ctx") or {})
    if above_pose is not None:
        task_context["travel_z"] = float(above_pose["z"])
    if target_pose is not None:
        runtime_state["_position"] = {
            "x": float(target_pose["x"]),
            "y": float(target_pose["y"]),
            "z": float(target_pose["z"]),
        }
        if task.name == "pick_approach":
            task_context["pick_z"] = float(target_pose["z"])
    if physical_overrides:
        task_context["cartesian_position_sources"] = {
            step_id: dict(override.get("position_sources") or {})
            for step_id, override in physical_overrides.items()
        }
    if computed_positions:
        task_context["computed_cartesian_positions"] = deepcopy(computed_positions)
        task_context["computed_cartesian_reference"] = deepcopy(computed_reference)
        task_context["computed_cartesian_at"] = float(computed_at)
    if resolved_positions:
        task_context["resolved_cartesian_positions"] = deepcopy(resolved_positions)
    if task.name == "place_approach":
        targets = step_outputs.get("place_targets")
        if isinstance(targets, dict) and target_pose is not None:
            resolved_pre_insert, resolved_pre_insert_error = _normalized_se3_pose(
                target_pose,
                label="resolved place_approach pre_insert_pose",
            )
            move_insert_mode = str(targets.get("move_insert_mode") or "")
            if resolved_pre_insert_error and move_insert_mode:
                return resolved_pre_insert_error
            if move_insert_mode in {"force_limited", "force_limited_trial"}:
                boundary_error = str(
                    targets.get("move_insert_boundary_error") or ""
                ).strip()
                boundary: dict[str, Any] = {}
                if not boundary_error:
                    boundary, boundary_error = _validated_move_insert_boundary(
                        start_pose=resolved_pre_insert,
                        targets=targets,
                        raw_hard_caps=targets.get("move_insert_hard_caps"),
                    )
                task_context["pre_insert_pose"] = deepcopy(resolved_pre_insert)
                if boundary_error:
                    exact_insert, exact_insert_error = _normalized_se3_pose(
                        targets.get("insert_pose"),
                        label="computed place_approach insert_pose",
                    )
                    if not exact_insert_error:
                        task_context["insert_pose"] = exact_insert
                    task_context["move_insert_boundary_ready"] = False
                    task_context["move_insert_boundary_error"] = boundary_error
                    task_context.pop("move_insert_hard_caps", None)
                    task_context.pop("move_insert_hard_caps_sha256", None)
                    task_context.pop("move_insert_boundary_metrics", None)
                else:
                    task_context.update(
                        {
                            "pre_insert_pose": deepcopy(boundary["pre_insert_pose"]),
                            "insert_pose": deepcopy(boundary["insert_pose"]),
                            "insertion_axis_world": deepcopy(
                                boundary["insertion_axis_world"]
                            ),
                            "move_insert_timeout_sec": float(
                                boundary["move_insert_timeout_sec"]
                            ),
                            "move_insert_hard_caps": deepcopy(
                                boundary["move_insert_hard_caps"]
                            ),
                            "move_insert_hard_caps_sha256": str(
                                boundary["move_insert_hard_caps_sha256"]
                            ),
                            "move_insert_boundary_metrics": {
                                field: float(boundary[field])
                                for field in (
                                    "insertion_depth_m",
                                    "insertion_travel_m",
                                    "lateral_error_m",
                                    "orientation_error_rad",
                                    "learned_start_position_error_m",
                                    "learned_start_orientation_error_rad",
                                )
                            },
                            "move_insert_boundary_ready": True,
                            "move_insert_boundary_error": "",
                        }
                    )
            elif move_insert_mode == "simulation_direct":
                exact_insert, exact_insert_error = _normalized_se3_pose(
                    targets.get("insert_pose"),
                    label="computed place_approach insert_pose",
                )
                if exact_insert_error:
                    return exact_insert_error
                task_context["pre_insert_pose"] = deepcopy(resolved_pre_insert)
                task_context["insert_pose"] = exact_insert
            else:
                for stale_field in (
                    "move_insert_profile",
                    "move_insert_profile_sha256",
                    "move_insert_mode",
                    "move_insert_timeout_sec",
                    "pre_insert_pose",
                    "insert_pose",
                    "insertion_axis_world",
                    "move_insert_hard_caps",
                    "move_insert_hard_caps_sha256",
                    "move_insert_boundary_ready",
                    "move_insert_boundary_error",
                    "move_insert_boundary_metrics",
                    "move_insert_qualification_error",
                ):
                    task_context.pop(stale_field, None)
        localization = step_outputs.get("assembly_board_v1_aruco")
        if isinstance(localization, dict) and localization:
            task_context["assembly_board_v1_aruco"] = deepcopy(localization)
            task_context["assembly_board_v1_aruco_generation"] = deepcopy(
                localization.get("generation")
            )
    runtime_state["_task_ctx"] = task_context
    return ""


def _computed_cartesian_state(
    *,
    task: RobotTaskDefinition,
    step_outputs: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Return raw computed Cartesian poses and their exact world reference."""
    output_name = "pick_targets" if task.name == "pick_approach" else "place_targets"
    targets = step_outputs.get(output_name)
    cartesian_steps = _cartesian_position_steps(task)
    if not isinstance(targets, dict) or not cartesian_steps:
        return {}, {}
    positions: dict[str, dict[str, float]] = {}
    for step, pose_key in zip(
        cartesian_steps[:2],
        ("approach_pose", "target_pose"),
        strict=False,
    ):
        raw_pose = targets.get(pose_key)
        if not isinstance(raw_pose, dict):
            continue
        try:
            pose = {
                field: float(raw_pose[field]) for field in _CARTESIAN_POSITION_FIELDS
            }
        except (KeyError, TypeError, ValueError):
            continue
        if not all(isfinite(value) for value in pose.values()):
            continue
        for field in ("qx", "qy", "qz", "qw"):
            try:
                value = float(raw_pose[field])
            except (KeyError, TypeError, ValueError):
                continue
            if isfinite(value):
                pose[field] = value
        positions[step.id] = pose

    if task.name == "pick_approach":
        raw_reference = dict(targets.get("origin_pose") or {})
        reference = {
            "kind": "detected_part",
            "frame_id": str(targets.get("frame_id") or "world").strip(),
            "name": str(targets.get("part_name") or "").strip(),
            "source": str(targets.get("target_pose_source") or "live_detection").strip(),
            "captured_at": targets.get("captured_at"),
        }
    else:
        localization = dict(step_outputs.get("assembly_board_v1_aruco") or {})
        if localization:
            raw_reference = dict(localization.get("pose") or {})
            reference = {
                "kind": "destination_target",
                "frame_id": str(localization.get("frame_id") or "").strip(),
                "name": str(localization.get("destination_location") or "").strip(),
                "source": "assembly_board-v1_aruco",
                "captured_at": localization.get("captured_at"),
                "camera_role": str(localization.get("camera_role") or "").strip(),
                "generation": deepcopy(localization.get("generation")),
                "calibration_id": str(
                    localization.get("calibration_id") or ""
                ).strip(),
                "pose": deepcopy(raw_reference),
            }
        else:
            raw_reference = dict(targets.get("target_pose") or {})
            reference = {
                "kind": "destination_target",
                "frame_id": "world",
                "name": str(targets.get("destination_location") or "").strip(),
                "source": "computed_destination",
                "captured_at": time.time(),
            }
    try:
        reference["position_m"] = {
            field: float(raw_reference[field]) for field in _CARTESIAN_POSITION_FIELDS
        }
        reference["captured_at"] = float(reference["captured_at"])
    except (KeyError, TypeError, ValueError):
        return positions, {}
    if (
        reference["frame_id"] != "world"
        or not reference["name"]
        or not reference["source"]
        or not all(isfinite(value) for value in reference["position_m"].values())
        or not isfinite(reference["captured_at"])
    ):
        return positions, {}
    return positions, reference


def _apply_cartesian_overrides_to_targets(  # noqa: C901, PLR0912, PLR0915 - explicit reference and safety gates.
    *,
    task: RobotTaskDefinition,
    physical_overrides: dict[str, dict[str, Any]],
    step_outputs: dict[str, Any],
    agent: Any | None = None,
) -> str:
    if not physical_overrides:
        return ""
    output_name = "pick_targets" if task.name == "pick_approach" else "place_targets"
    targets = step_outputs.get(output_name)
    if not isinstance(targets, dict):
        return ""
    cartesian_steps = _cartesian_position_steps(task)
    if not cartesian_steps:
        return ""
    pose_keys = ("approach_pose", "target_pose")
    current_reference_pose: dict[str, float] = {}
    current_reference_calibration_id = ""
    if task.name == "pick_approach":
        current_reference = dict(targets.get("origin_pose") or {})
        current_reference_name = str(targets.get("part_name") or "").strip()
        if str(targets.get("frame_id") or "").strip() != "world":
            return "pick_approach current detected part reference frame must be world"
        try:
            reference_captured_at = float(targets["captured_at"])
        except (KeyError, TypeError, ValueError):
            return "pick_approach current detected part reference has no capture timestamp"
        reference_age_sec = time.time() - reference_captured_at
        if (
            not isfinite(reference_captured_at)
            or reference_age_sec < -1.0
            or reference_age_sec > 8.0
        ):
            return (
                "pick_approach current detected part reference is stale "
                f"(age={reference_age_sec:.2f}s)"
            )
    else:
        current_reference_name = str(targets.get("destination_location") or "").strip()
        if current_reference_name == "assembly_board-v1":
            localization, localization_error = _assembly_board_v1_aruco_payload(
                step_outputs.get("assembly_board_v1_aruco"),
                robot=_robot_name(agent) if agent is not None else "",
                destination_location=current_reference_name,
                require_fresh=True,
            )
            if localization_error:
                return localization_error
            current_reference_pose = dict(localization["pose"])
            current_reference_calibration_id = str(
                localization.get("calibration_id") or ""
            )
            current_reference = {
                field: current_reference_pose[field]
                for field in _CARTESIAN_POSITION_FIELDS
            }
        else:
            current_reference = dict(targets.get("target_pose") or {})
    if not current_reference_name:
        return f"{task.name} current relative reference name is unavailable"
    try:
        current_reference = {
            field: float(current_reference[field]) for field in _CARTESIAN_POSITION_FIELDS
        }
    except (KeyError, TypeError, ValueError):
        return f"{task.name} current relative reference position is incomplete"
    if not all(isfinite(value) for value in current_reference.values()):
        return f"{task.name} current relative reference position is not finite"

    for index, step in enumerate(cartesian_steps[:2]):
        override = physical_overrides.get(step.id)
        if not override:
            continue
        pose_key = pose_keys[index]
        pose = dict(targets.get(pose_key) or {})
        relative_pose = dict(override.get("relative_pose") or {})
        if task.name == "place_approach" and current_reference_name == "assembly_board-v1":
            recorded_reference = dict(override.get("relative_reference") or {})
            if recorded_reference.get("name") != current_reference_name:
                return (
                    f"{task.name}.{step.id} recorded destination reference does not "
                    "match assembly_board-v1"
                )
            if recorded_reference.get("source") != "assembly_board-v1_aruco":
                return (
                    f"{task.name}.{step.id} recorded destination reference source "
                    "must be assembly_board-v1_aruco"
                )
            if (
                recorded_reference.get("calibration_id")
                != current_reference_calibration_id
            ):
                return (
                    f"{task.name}.{step.id} recorded assembly_board-v1 calibration "
                    "identity does not match the current accepted board calibration"
                )
            recorded_reference_pose = dict(recorded_reference.get("pose") or {})
            saved_computed_pose = dict(override.get("computed_pose") or {})
            try:
                board_delta = _compose_se3(
                    current_reference_pose,
                    _inverse_se3(recorded_reference_pose),
                )
                base_pose = _compose_se3(board_delta, saved_computed_pose)
            except (KeyError, TypeError, ValueError):
                return (
                    f"{task.name}.{step.id} recorded board or computed pose is incomplete"
                )
            relative_base = {
                field: float(base_pose[field])
                for field in _CARTESIAN_POSITION_FIELDS
            }
        else:
            try:
                relative_base = {
                    field: float(pose[field])
                    for field in _CARTESIAN_POSITION_FIELDS
                }
                base_pose = {
                    **relative_base,
                    "qx": float(pose.get("qx", 0.0)),
                    "qy": float(pose.get("qy", 0.0)),
                    "qz": float(pose.get("qz", 0.0)),
                    "qw": float(pose.get("qw", 1.0)),
                }
            except (KeyError, TypeError, ValueError):
                return f"{task.name}.{step.id} computed target pose is incomplete"
        pose = _compose_se3(base_pose, relative_pose)
        override["resolved_reference_position_m"] = deepcopy(current_reference)
        override["resolved_computed_position_m"] = deepcopy(relative_base)
        override["resolved_computed_pose"] = deepcopy(base_pose)
        override["resolved_position_m"] = {
            field: float(pose[field]) for field in _CARTESIAN_POSITION_FIELDS
        }
        pose_error = _resolved_cartesian_pose_error(
            agent=agent,
            task=task,
            step=step,
            pose=pose,
        )
        if pose_error:
            return pose_error
        targets[pose_key] = pose

    if (
        task.name in {"pick_approach", "place_approach"}
        and _robot_name(agent) == "ur5e"
        and (
            task.name == "pick_approach"
            or current_reference_name == "assembly_board-v1"
        )
    ):
        descend_pose = dict(targets.get("target_pose") or {})
        try:
            descend_qx = float(descend_pose["qx"])
            descend_qy = float(descend_pose["qy"])
        except (KeyError, TypeError, ValueError):
            return f"{task.name}.descend orientation is incomplete"
        vertical_xy_norm = sqrt(descend_qx**2 + descend_qy**2)
        if not isfinite(vertical_xy_norm) or vertical_xy_norm <= 1e-12:
            return f"{task.name}.descend cannot resolve a vertical tool orientation"
        vertical_orientation = {
            "qx": descend_qx / vertical_xy_norm,
            "qy": descend_qy / vertical_xy_norm,
            "qz": 0.0,
            "qw": 0.0,
        }
        for pose_key in ("approach_pose", "target_pose"):
            pose = dict(targets.get(pose_key) or {})
            if pose:
                pose.update(vertical_orientation)
                targets[pose_key] = pose

        if descend_pose and task.name == "place_approach":
            pre_insert_pose = dict(targets.get("pre_insert_pose") or {})
            if pre_insert_pose:
                pre_insert_pose.update(
                    {
                        "x": float(descend_pose["x"]),
                        "y": float(descend_pose["y"]),
                        **vertical_orientation,
                    }
                )
                targets["pre_insert_pose"] = pre_insert_pose

            insert_pose = dict(targets.get("insert_pose") or {})
            if insert_pose:
                insert_pose.update(
                    {
                        "x": float(descend_pose["x"]),
                        "y": float(descend_pose["y"]),
                        **vertical_orientation,
                    }
                )
                targets["insert_pose"] = insert_pose
            targets["insertion_axis_world"] = {
                "x": 0.0,
                "y": 0.0,
                "z": -1.0,
            }

    approach_pose = dict(targets.get("approach_pose") or {})
    target_pose = dict(targets.get("target_pose") or {})
    try:
        approach_z = float(approach_pose["z"])
        target_z = float(target_pose["z"])
    except (KeyError, TypeError, ValueError):
        return f"{task.name} resolved Cartesian positions are incomplete"
    if not isfinite(approach_z) or not isfinite(target_z):
        return f"{task.name} resolved Cartesian Z contains a non-finite value"
    if approach_z <= target_z + 1e-6:
        return (
            f"{task.name} resolved move-above Z must remain above the descend Z "
            "so the later lift is positive"
        )
    if task.name == "pick_approach":
        targets["travel_z"] = approach_z
        targets["pick_z"] = target_z
        return _validate_resolved_mg_pick_target(targets)
    return ""


def _resolved_cartesian_pose_error(
    *,
    agent: Any | None,
    task: RobotTaskDefinition,
    step: RobotTaskStep,
    pose: dict[str, Any],
) -> str:
    try:
        position = {field: float(pose[field]) for field in _CARTESIAN_POSITION_FIELDS}
    except (KeyError, TypeError, ValueError):
        return f"{task.name}.{step.id} resolved Cartesian position is incomplete"
    if not all(isfinite(value) for value in position.values()):
        return f"{task.name}.{step.id} resolved Cartesian position is not finite"
    if agent is None:
        return ""

    workspace_check = getattr(agent, "_is_pose_in_workspace", None)
    if not callable(workspace_check):
        return f"{task.name}.{step.id} workspace validation is unavailable"
    try:
        workspace_ready, workspace_reason = workspace_check(position)
    except (AttributeError, TypeError, ValueError) as exc:
        return f"{task.name}.{step.id} workspace validation failed: {exc}"
    if not workspace_ready:
        return f"{task.name}.{step.id} {workspace_reason}"

    capabilities = dict(getattr(agent, "static_capabilities", {}) or {})
    gripper_reach = capabilities.get("gripper_reach")
    if not isinstance(gripper_reach, dict) or not gripper_reach:
        return f"{task.name}.{step.id} gripper_reach capability data is unavailable"
    if str(gripper_reach.get("frame") or "world").strip() != "world":
        return f"{task.name}.{step.id} gripper_reach frame must be world"
    origin_pose = dict(gripper_reach.get("origin_pose") or {})
    try:
        origin_x = float(origin_pose["x"])
        origin_y = float(origin_pose["y"])
        max_xy_radius_m = float(gripper_reach["max_xy_radius_m"])
        tolerance_m = float(gripper_reach.get("tolerance_m") or 0.0)
        z_min_m = float(gripper_reach["z_min_m"])
        z_max_m = float(gripper_reach["z_max_m"])
    except (KeyError, TypeError, ValueError):
        return f"{task.name}.{step.id} gripper_reach capability data is incomplete"
    reach_values = (
        origin_x,
        origin_y,
        max_xy_radius_m,
        tolerance_m,
        z_min_m,
        z_max_m,
    )
    if not all(isfinite(value) for value in reach_values):
        return f"{task.name}.{step.id} gripper_reach capability data is not finite"
    xy_radius_m = (
        (position["x"] - origin_x) ** 2 + (position["y"] - origin_y) ** 2
    ) ** 0.5
    if xy_radius_m > max_xy_radius_m + tolerance_m:
        return (
            f"{task.name}.{step.id} resolved pose is outside gripper_reach: "
            f"xy_radius={xy_radius_m:.4f} m, max={max_xy_radius_m:.4f} m"
        )
    if position["z"] < z_min_m - tolerance_m or position["z"] > z_max_m + tolerance_m:
        return (
            f"{task.name}.{step.id} resolved pose is outside gripper_reach: "
            f"z={position['z']:.4f} m, range=[{z_min_m:.4f}, {z_max_m:.4f}] m"
        )
    return ""


def _validate_resolved_mg_pick_target(targets: dict[str, Any]) -> str:
    if str(targets.get("part_name") or "").strip() != "MG":
        return ""
    required_fields = (
        "table_surface_z_m",
        "tcp_offset_z",
        "pick_tool0_z_adjustment_m",
        "tooth_height_m",
        "part_height",
        "tooth_clearance_m",
        "minimum_hub_overlap_m",
        "open_inner_pad_lower_z_from_tcp_m",
        "closed_inner_pad_lower_z_from_tcp_m",
        "closed_inner_pad_upper_z_from_tcp_m",
    )
    try:
        values = {field: float(targets[field]) for field in required_fields}
        resolved_tool0_z = float(dict(targets["target_pose"])["z"])
    except (KeyError, TypeError, ValueError):
        return "physical MG taught target is missing STL safety diagnostics"
    if not all(isfinite(value) for value in (*values.values(), resolved_tool0_z)):
        return "physical MG taught target contains non-finite STL safety diagnostics"

    resolved_pick_tcp_z = (
        resolved_tool0_z
        - values["pick_tool0_z_adjustment_m"]
        + values["tcp_offset_z"]
    )
    tcp_offset_from_table_m = resolved_pick_tcp_z - values["table_surface_z_m"]
    lowest_endpoint = min(
        values["open_inner_pad_lower_z_from_tcp_m"],
        values["closed_inner_pad_lower_z_from_tcp_m"],
    )
    finger_tooth_clearance_m = (
        tcp_offset_from_table_m + lowest_endpoint - values["tooth_height_m"]
    )
    closed_pad_lower_m = (
        tcp_offset_from_table_m + values["closed_inner_pad_lower_z_from_tcp_m"]
    )
    closed_pad_upper_m = (
        tcp_offset_from_table_m + values["closed_inner_pad_upper_z_from_tcp_m"]
    )
    finger_hub_overlap_m = max(
        0.0,
        min(closed_pad_upper_m, values["part_height"])
        - max(closed_pad_lower_m, values["tooth_height_m"]),
    )
    targets.update(
        {
            "pick_tcp_z": resolved_pick_tcp_z,
            "pick_tcp_z_offset_from_table_m": tcp_offset_from_table_m,
            "finger_tooth_clearance_m": finger_tooth_clearance_m,
            "finger_hub_overlap_m": finger_hub_overlap_m,
            "resolved_taught_tool0_z_m": resolved_tool0_z,
        }
    )
    if finger_tooth_clearance_m + 1e-9 < values["tooth_clearance_m"]:
        return (
            "physical MG taught Z would contact the teeth: "
            f"clearance={finger_tooth_clearance_m * 1000.0:.2f} mm, "
            f"required={values['tooth_clearance_m'] * 1000.0:.2f} mm"
        )
    return ""


async def _execute_task_step(  # noqa: C901 - task execution gates stay explicit.
    *,
    agent: Any,
    task: RobotTaskDefinition,
    step: RobotTaskStep,
    args: dict[str, Any],
    runtime_state: dict[str, Any],
    step_outputs: dict[str, Any],
    physical_overrides: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for guard in step.when:
        if not _evaluate_guard(
            guard,
            agent=agent,
            args=args,
            runtime_state=runtime_state,
            step_outputs=step_outputs,
        ):
            return {"success": True, "skipped": True}
    if (
        task.name == "place_approach"
        and step.id == "localize_assembly_board_v1"
        and str(args.get("destination_location") or "").strip()
        != "assembly_board-v1"
    ):
        return {"success": True, "skipped": True}

    params = _resolve_value(
        step.params, args=args, runtime_state=runtime_state, step_outputs=step_outputs
    )
    if not isinstance(params, dict):
        params = {}
    if (
        str(getattr(agent, "execution_mode", "dry_run") or "").strip().lower()
        == "physical"
        and task.name in {"pick_approach", "place_approach"}
        and step.op == "move_to_named_pose"
    ):
        pose_name = str(params.get("pose_name") or "").strip()
        named_positions = getattr(agent, "named_positions", {})
        if not isinstance(named_positions, dict) or pose_name not in named_positions:
            return {
                "success": True,
                "skipped": True,
                "payload": {
                    "pose_name": pose_name,
                    "staging_source": "computed_cartesian_target",
                },
            }
    if (
        step.physical_position_required or step.op == "move_cartesian"
    ) and step.id in physical_overrides:
        override = physical_overrides[step.id]
        board_relative_place_pose = bool(
            task.name == "place_approach"
            and str(args.get("destination_location") or "").strip()
            == "assembly_board-v1"
            and override.get("relative_pose")
        )
        if not board_relative_place_pose:
            params.update(dict(override.get("values") or {}))
    if step.op == "move_cartesian" and all(
        params.get(field) is None for field in ("qx", "qy", "qz", "qw")
    ):
        for field in ("qx", "qy", "qz", "qw"):
            params.pop(field, None)

    if str(step.executor or "primitive").strip() != "primitive":
        return {
            "success": False,
            "raw": {"message": f"unsupported non-primitive step executor '{step.executor}'"},
        }

    if getattr(agent, "execution_mode", "dry_run") == "dry_run":
        raw_payload = _resolve_value(
            step.dry_run_output,
            args=args,
            runtime_state=runtime_state,
            step_outputs=step_outputs,
        )
        primitive_result = {"success": True}
        if isinstance(raw_payload, dict):
            primitive_result.update(raw_payload)
        payload = _primitive_payload_from_result(
            step=step,
            params=params,
            primitive_result=primitive_result,
            runtime_state=runtime_state,
        )
        return {"success": True, "payload": payload, "raw": primitive_result}

    result = await agent._execute_primitive(
        step.op,
        {key: value for key, value in params.items() if not str(key).startswith("_")},
    )
    payload = None
    if result.get("success"):
        payload = _primitive_payload_from_result(
            step=step,
            params=params,
            primitive_result=result,
            runtime_state=runtime_state,
        )
        if step.op == "move_insert" and isinstance(payload, dict) and payload.get(
            "pose_error"
        ):
            return {
                "success": False,
                "raw": {"message": str(payload["pose_error"])},
                "payload": None,
            }
        if step.store_as == "assembly_board_v1_aruco":
            payload, localization_error = _assembly_board_v1_aruco_payload(
                payload,
                robot=_robot_name(agent),
                destination_location=str(params.get("destination_location") or "").strip(),
                require_fresh=True,
            )
            if localization_error:
                return {
                    "success": False,
                    "raw": {"message": localization_error},
                    "payload": None,
                }
    return {"success": bool(result.get("success")), "raw": result, "payload": payload}


def _apply_effect(
    effect: RobotTaskEffect,
    *,
    args: dict[str, Any],
    runtime_state: dict[str, Any],
    step_outputs: dict[str, Any],
) -> None:
    value = _resolve_value(
        effect.value, args=args, runtime_state=runtime_state, step_outputs=step_outputs
    )

    if effect.target == "task_ctx":
        if effect.action == "clear":
            runtime_state["_task_ctx"] = {}
            return
        if effect.action == "merge":
            current = dict(runtime_state.get("_task_ctx") or {})
            incoming = dict(value or {})
            if effect.skip_empty_values:
                incoming = {key: item for key, item in incoming.items() if item not in (None, "")}
            current.update(deepcopy(incoming))
            runtime_state["_task_ctx"] = current
            return
        runtime_state["_task_ctx"] = deepcopy(dict(value or {}))
        return

    if effect.action == "clear":
        runtime_state[f"_{effect.target}"] = None if effect.target != "task_ctx" else {}
        return
    runtime_state[f"_{effect.target}"] = deepcopy(value)


def _physical_place_insert_board_lock_error(  # noqa: C901, PLR0912 - irreversible gate.
    *,
    agent: Any,
    args: dict[str, Any],
    runtime_state: dict[str, Any],
    require_seated_pose: bool = False,
) -> str:
    task_context = dict(runtime_state.get("_task_ctx") or {})
    handoff_error = _physical_place_approach_held_part_handoff_error(
        task_context,
        part_name=args.get("part_name"),
    )
    if handoff_error:
        return handoff_error
    move_insert_mode = str(task_context.get("move_insert_mode") or "")
    if move_insert_mode in {"force_limited", "force_limited_trial"}:
        boundary_error = str(
            task_context.get("move_insert_boundary_error") or ""
        ).strip()
        if boundary_error:
            return f"place_insert move_insert boundary is not ready: {boundary_error}"
        if task_context.get("move_insert_boundary_ready") is not True:
            return (
                "place_insert move_insert boundary is not ready; run place_approach "
                "with verified live RTDE insertion hard caps"
            )
    move_insert_enabled = task_context.get("move_insert_mode") == "force_limited"
    profile: dict[str, Any] = {}
    profile_hash = ""
    if move_insert_enabled:
        part_name = args.get("part_name")
        if (
            not isinstance(part_name, str)
            or not part_name
            or part_name != part_name.strip()
            or runtime_state.get("_held_part") != part_name
            or task_context.get("part_name") != part_name
        ):
            return "place_insert exact held-part identity changed after place_approach"
        raw_profile = task_context.get("move_insert_profile")
        if not isinstance(raw_profile, dict) or raw_profile.get("part_name") != part_name:
            return "place_insert frozen move_insert profile part identity is invalid"
        profile = raw_profile
        profile_hash = task_context.get("move_insert_profile_sha256")
        if (
            not isinstance(profile_hash, str)
            or len(profile_hash) != 64
            or profile.get("profile_sha256") != profile_hash
        ):
            return "place_insert frozen move_insert profile_sha256 is invalid"
        try:
            int(profile_hash, 16)
        except ValueError:
            return "place_insert frozen move_insert profile_sha256 is invalid"
        raw_hard_caps = task_context.get("move_insert_hard_caps")
        if not isinstance(raw_hard_caps, dict) or not raw_hard_caps:
            return "place_insert frozen move_insert hard caps are invalid"
        try:
            canonical_hard_caps = json.dumps(
                {
                    name: float(value)
                    for name, value in sorted(raw_hard_caps.items())
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError, OverflowError):
            return "place_insert frozen move_insert hard caps are invalid"
        hard_caps_sha256 = hashlib.sha256(
            canonical_hard_caps.encode("utf-8")
        ).hexdigest()
        task_context["move_insert_hard_caps_sha256"] = hard_caps_sha256
        runtime_state["_task_ctx"] = task_context
        shared_calibration_id = profile.get("shared_calibration_id")
        override_calibration_id = profile.get("override_calibration_id")
        effective_calibration_id = profile.get("calibration_id")
        if (
            not isinstance(shared_calibration_id, str)
            or not shared_calibration_id
            or shared_calibration_id != shared_calibration_id.strip()
            or (
                override_calibration_id is not None
                and (
                    not isinstance(override_calibration_id, str)
                    or not override_calibration_id
                    or override_calibration_id != override_calibration_id.strip()
                )
            )
            or effective_calibration_id
            != (override_calibration_id or shared_calibration_id)
        ):
            return "place_insert frozen move_insert calibration identities are invalid"
        _start_pose, start_error = _normalized_se3_pose(
            dict(task_context.get("resolved_cartesian_positions") or {}).get("descend"),
            label="place_insert frozen pre_insert pose",
        )
        _target_pose, target_error = _normalized_se3_pose(
            task_context.get("insert_pose"),
            label="place_insert frozen insert pose",
        )
        if start_error or target_error:
            return start_error or target_error
    localization, localization_error = _assembly_board_v1_aruco_payload(
        task_context.get("assembly_board_v1_aruco"),
        robot=_robot_name(agent),
        destination_location=str(args.get("destination_location") or "").strip(),
        require_fresh=False,
    )
    if localization_error:
        return (
            "place_insert requires the frozen assembly_board-v1 ArUco generation from "
            f"place_approach: {localization_error}"
        )
    if task_context.get("assembly_board_v1_aruco_generation") != localization.get(
        "generation"
    ):
        return "place_insert assembly_board-v1 ArUco generation lock changed"

    controller = getattr(agent, "_controller", None)
    acceptance_reader = getattr(controller, "_assembly_board_v1_aruco_acceptance", None)
    if not callable(acceptance_reader):
        return "place_insert cannot verify the current assembly_board-v1 acceptance"
    current_acceptance, acceptance_error = acceptance_reader()
    if acceptance_error:
        return (
            "place_insert cannot verify the current assembly_board-v1 acceptance: "
            f"{acceptance_error}"
        )
    if current_acceptance.get("accepted_generation") != localization.get("generation"):
        return (
            "place_insert assembly_board-v1 accepted generation changed after "
            "place_approach"
        )
    if str(current_acceptance.get("calibration_id") or "").strip() != localization.get(
        "calibration_id"
    ):
        return (
            "place_insert assembly_board-v1 calibration identity changed after "
            "place_approach"
        )
    if not require_seated_pose or not move_insert_enabled:
        return ""

    move_insert_result = task_context.get("move_insert_result")
    if not isinstance(move_insert_result, dict):
        return "place_insert successful move_insert result is missing before release"
    if (
        move_insert_result.get("success") is not True
        or move_insert_result.get("state_uncertain") is True
        or move_insert_result.get("final_tool0_pose_valid") is not True
        or move_insert_result.get("engagement_detected") is not True
        or move_insert_result.get("seated_detected") is not True
        or move_insert_result.get("profile_sha256") != profile_hash
        or move_insert_result.get("hard_caps_sha256") != hard_caps_sha256
    ):
        return (
            "place_insert move_insert result lacks confirmed engagement, seating, "
            "identity, or final-pose validity"
        )
    final_pose, final_pose_error = _normalized_se3_pose(
        dict(task_context.get("resolved_cartesian_positions") or {}).get("move_insert"),
        label="place_insert successful move_insert final pose",
    )
    if final_pose_error:
        return final_pose_error
    controller = getattr(agent, "_controller", None)
    current_pose_reader = getattr(controller, "_get_ee_pose", None)
    if not callable(current_pose_reader):
        return "place_insert cannot read a fresh world -> tool0 pose before release"
    try:
        current_message = current_pose_reader()
        current_pose, current_pose_error = _normalized_se3_pose(
            {
                "x": current_message.position.x,
                "y": current_message.position.y,
                "z": current_message.position.z,
                "qx": current_message.orientation.x,
                "qy": current_message.orientation.y,
                "qz": current_message.orientation.z,
                "qw": current_message.orientation.w,
            },
            label="place_insert fresh world -> tool0 pose",
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return f"place_insert fresh world -> tool0 pose is unavailable: {exc}"
    if current_pose_error:
        return current_pose_error
    try:
        translation_tolerance = float(profile["seated_depth_tolerance_m"])
        rotation_tolerance = float(profile["tilt_tolerance_rad"])
    except (KeyError, TypeError, ValueError):
        return "place_insert seated pose tolerances are invalid"
    translation_error = sqrt(
        sum(
            (current_pose[field] - final_pose[field]) ** 2
            for field in _CARTESIAN_POSITION_FIELDS
        )
    )
    quaternion_dot = abs(
        sum(
            current_pose[field] * final_pose[field]
            for field in ("qx", "qy", "qz", "qw")
        )
    )
    rotation_error = 2.0 * acos(max(-1.0, min(1.0, quaternion_dot)))
    if translation_error > translation_tolerance or rotation_error > rotation_tolerance:
        return (
            "place_insert fresh world -> tool0 pose moved outside seated tolerances: "
            f"translation={translation_error:.6f} m, rotation={rotation_error:.6f} rad"
        )
    return ""


async def execute_robot_task(  # noqa: C901, PLR0912, PLR0915
    agent: Any,
    task_name: str,
    manual_function_execution_authority: object | None = None,
    operator_confirmed_held_part: bool = False,
    operator_confirmed_held_part_handoff: dict[str, Any] | None = None,
    /,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute one exact registry-backed task against the selected robot mode.

    Args:
        agent: RobotAgent-compatible runtime owner.
        task_name: Exact registered robot function name.
        manual_function_execution_authority: Internal identity capability for the
            Control-page manual Function Execution path.
        operator_confirmed_held_part: Whether the operator confirms the exact selected
            ``part_name`` is physically held for one independent ``place_approach``
            commissioning run. This positional value is accepted only with the
            internal manual execution authority.
        operator_confirmed_held_part_handoff: Bridge-validated complete held-part
            handoff derived from the confirmed ``pick_approach.descend`` recording.
        **kwargs: Arguments declared by the selected task definition.

    Returns:
        Task completion, block, or failure payload.
    """
    task = robot_task_registry().get(str(task_name or "").strip())
    if task is None:
        return {"status": "failed", "content": f"unknown robot task '{task_name}'"}

    manual_function_execution = (
        manual_function_execution_authority is _MANUAL_FUNCTION_EXECUTION_AUTHORITY
    )
    args = deepcopy(dict(kwargs or {}))
    runtime_state = _build_runtime_state(agent)
    step_outputs: dict[str, Any] = {}
    task_context = dict(runtime_state.get("_task_ctx") or {})
    if not isinstance(operator_confirmed_held_part, bool):
        return {
            "status": "blocked",
            "content": "operator_confirmed_held_part must be a boolean.",
        }
    if (
        operator_confirmed_held_part_handoff is not None
        and not operator_confirmed_held_part
    ):
        return {
            "status": "blocked",
            "content": (
                "operator_confirmed_held_part_handoff requires "
                "operator_confirmed_held_part=true."
            ),
        }
    if operator_confirmed_held_part and (
        not manual_function_execution or task.name != "place_approach"
    ):
        return {
            "status": "blocked",
            "content": (
                "operator_confirmed_held_part is available only for manual "
                "place_approach commissioning."
            ),
        }
    requested_part_name = args.get("part_name")
    if operator_confirmed_held_part and not (
        str(getattr(agent, "execution_mode", "") or "") == "physical"
        and _robot_name(agent) == "ur5e"
        and args.get("destination_location") == "assembly_board-v1"
        and requested_part_name in _MOVE_INSERT_TRANSLATIONAL_PARTS
    ):
        return {
            "status": "blocked",
            "content": (
                "operator_confirmed_held_part is available only for physical ur5e "
                "place_approach at assembly_board-v1 with exact part_name 'SG', "
                "'MG', 'LG', 'SCP', 'MCP', or 'LCP'."
            ),
        }
    validated_operator_handoff: dict[str, Any] = {}
    if operator_confirmed_held_part:
        validated_operator_handoff, operator_handoff_error = (
            _validated_operator_confirmed_held_part_handoff(
                operator_confirmed_held_part_handoff,
                part_name=requested_part_name,
            )
        )
        if operator_handoff_error:
            return {
                "status": "blocked",
                "content": operator_handoff_error,
            }
    if operator_confirmed_held_part and (
        not isinstance(requested_part_name, str)
        or not requested_part_name
        or requested_part_name != requested_part_name.strip()
    ):
        return {
            "status": "blocked",
            "content": (
                "operator-confirmed place_approach requires one exact non-empty "
                "part_name."
            ),
        }
    independent_place_approach = bool(
        manual_function_execution
        and task.name == "place_approach"
        and runtime_state.get("_held_part") in (None, "")
        and task_context.get("part_name") in (None, "")
        and task_context.get("origin_resource_location") in (None, "")
    )
    operator_held_place_approach = bool(
        independent_place_approach
        and operator_confirmed_held_part
        and validated_operator_handoff
    )
    if operator_confirmed_held_part and not operator_held_place_approach:
        return {
            "status": "blocked",
            "content": (
                "operator_confirmed_held_part cannot replace active pick_grasp "
                "custody or context; clear or complete the active pick sequence "
                "first."
            ),
        }
    if operator_held_place_approach:
        runtime_state["_held_part"] = validated_operator_handoff["part_name"]
        runtime_state["_current_state"] = "picked"
        runtime_state["_gripper_state"] = "closed"
        runtime_state["_task_ctx"] = {
            "part_name": validated_operator_handoff["part_name"],
            "model_name": validated_operator_handoff["model_name"],
            "origin_resource_location": validated_operator_handoff[
                "origin_resource_location"
            ],
            "origin_pose": deepcopy(
                validated_operator_handoff["world_held_part_pose_at_grasp"]
            ),
            "origin_pose_provenance": deepcopy(
                validated_operator_handoff["origin_pose_provenance"]
            ),
            "resolved_cartesian_positions": {
                "descend": deepcopy(
                    validated_operator_handoff["world_tool0_pose_at_grasp"]
                )
            },
            "held_part_handoff": deepcopy(validated_operator_handoff),
            "operator_confirmed_held_part": True,
            "pick_approach_recording_path": validated_operator_handoff[
                "pick_approach_recording_path"
            ],
            "pick_approach_recording_sha256": validated_operator_handoff[
                "pick_approach_recording_sha256"
            ],
        }
        task_context = dict(runtime_state["_task_ctx"])
        # This is a real custody adoption, not an empty-custody commissioning run.
        independent_place_approach = False
        _commit_runtime_state(agent, runtime_state)
    independent_place_insert = bool(
        manual_function_execution
        and task.name == "place_insert"
        and runtime_state.get("_held_part") in (None, "")
    )
    requested_destination = args.get("destination_location")
    raw_product_geometry = args.get("product_geometry")
    product_target_reference = (
        dict(raw_product_geometry.get("target_reference") or {})
        if isinstance(raw_product_geometry, dict)
        else {}
    )
    assembly_surface_role = (
        product_target_reference.get("surface_role")
        or task_context.get("surface_role")
        or ("assembly_slot" if requested_destination == "assembly_board-v1" else "")
    )
    if (
        independent_place_insert
        and str(getattr(agent, "execution_mode", "") or "") == "physical"
        and requested_destination == "assembly_board-v1"
        and assembly_surface_role == "assembly_slot"
    ):
        return {
            "status": "blocked",
            "content": (
                "Physical place_insert at assembly_board-v1 requires a held part and "
                "the retained pick_grasp/place_approach context. Run those functions "
                "first, then use Supervised Test move_insert until the exact part is "
                "confirmed."
            ),
        }
    if (
        str(getattr(agent, "execution_mode", "") or "") == "physical"
        and _robot_name(agent) == "xarm6"
        and task.name in {"place_approach", "place_insert"}
        and not independent_place_approach
        and not independent_place_insert
        and runtime_state.get("_held_part") not in (None, "")
        and requested_destination == "assembly_board-v1"
        and assembly_surface_role == "assembly_slot"
    ):
        agent.logger.warning(
            "[Robot] %s",
            _PHYSICAL_XARM6_ASSEMBLY_SLOT_INSERT_ERROR,
        )
        return {
            "status": "blocked",
            "content": _PHYSICAL_XARM6_ASSEMBLY_SLOT_INSERT_ERROR,
        }
    if independent_place_approach:
        runtime_state["_task_ctx"] = {}
    elif independent_place_insert:
        try:
            current_z = float(dict(runtime_state.get("_position") or {})["z"])
        except (KeyError, TypeError, ValueError):
            return {
                "status": "blocked",
                "content": (
                    "Independent place_insert requires a finite current position.z "
                    "for the 0.08 m retreat."
                ),
            }
        if not isfinite(current_z):
            return {
                "status": "blocked",
                "content": (
                    "Independent place_insert requires a finite current position.z "
                    "for the 0.08 m retreat."
                ),
            }
        runtime_state["_task_ctx"] = {
            "destination_location": str(args.get("destination_location") or ""),
            "travel_z": current_z + 0.08,
        }

    for guard in task.program.entry_guards:
        # Independent Control-page tests do not claim assembly-sequence state. Normal
        # execution and manual runs with an actual held part retain held_part authority.
        condition = dict(guard.condition or {})
        if (
            manual_function_execution
            and str(condition.get("field") or "").strip() == "resource_state"
            and str(condition.get("operator") or "").strip() == "equals"
            and str(condition.get("value") or "").strip()
            == str(task.program.entry_state or "").strip()
        ):
            continue
        if independent_place_approach and condition.get("field") == "held_part":
            continue
        if independent_place_insert and condition.get("field") in {
            "held_part",
            "task_ctx.destination_location",
        }:
            continue
        if _evaluate_guard(
            guard,
            agent=agent,
            args=args,
            runtime_state=runtime_state,
            step_outputs=step_outputs,
        ):
            continue
        message = _resolve_value(
            guard.message,
            args=args,
            runtime_state=runtime_state,
            step_outputs=step_outputs,
        )
        detail = str(message or f"{task.name} is blocked").strip()
        agent.logger.warning("[Robot] %s", detail)
        return {"status": "blocked", "content": detail}

    if (
        str(getattr(agent, "execution_mode", "") or "") == "physical"
        and task.name == "place_insert"
        and not independent_place_insert
        and assembly_surface_role == "assembly_slot"
    ):
        retained_move_insert_mode = str(
            dict(runtime_state.get("_task_ctx") or {}).get("move_insert_mode")
            or ""
        )
        if retained_move_insert_mode == "force_limited_trial":
            task_context = dict(runtime_state.get("_task_ctx") or {})
            task_context["move_insert_mode"] = "force_limited"
            runtime_state["_task_ctx"] = task_context
            retained_move_insert_mode = "force_limited"
        if retained_move_insert_mode != "force_limited":
            return {
                "status": "blocked",
                "content": (
                    "Physical place_insert at assembly_board-v1 cannot release or "
                    "lift without a retained force_limited move_insert profile from "
                    "place_approach. Run place_approach again, then complete the "
                    "supervised move_insert workflow."
                ),
            }

    physical_overrides: dict[str, dict[str, Any]] = {}
    physical_recording_path: Path | None = None
    execution_mode = str(getattr(agent, "execution_mode", "") or "").strip().lower()
    held_part_handoff: dict[str, Any] = {}
    if task.name == "pick_grasp":
        held_part_handoff, handoff_error = _held_part_handoff_from_context(
            dict(runtime_state.get("_task_ctx") or {}),
            part_name=args.get("part_name"),
        )
        if (
            handoff_error
            and execution_mode == "physical"
            and not manual_function_execution
        ):
            return agent._task_failure(
                handoff_error,
                step="pick_grasp.held_part_handoff_preflight",
                observations={"part_name": args.get("part_name")},
            )
    if (
        execution_mode == "physical"
        and task.name == "place_insert"
        and not independent_place_insert
        and str(args.get("destination_location") or "").strip()
        == "assembly_board-v1"
    ):
        board_lock_error = _physical_place_insert_board_lock_error(
            agent=agent,
            args=args,
            runtime_state=runtime_state,
        )
        if board_lock_error:
            return agent._task_failure(
                board_lock_error,
                step="place_insert.assembly_board_v1_aruco_generation_lock",
                observations={
                    "destination_location": str(args.get("destination_location") or ""),
                    "part_name": str(args.get("part_name") or ""),
                    "move_insert_dispatched": False,
                },
            )
    if execution_mode == "physical" and _cartesian_position_steps(task):
        physical_overrides, physical_recording_path, preflight_error = (
            _load_physical_cartesian_overrides(
                agent=agent,
                task=task,
                args=args,
            )
        )
        if preflight_error:
            return agent._task_failure(
                preflight_error,
                step=f"{task.name}.physical_position_preflight",
                observations={
                    "function_name": task.name,
                    "physical_position_file": (
                        str(physical_recording_path) if physical_recording_path is not None else ""
                    ),
                },
            )
    if (
        execution_mode == "physical"
        and _robot_name(agent) == "ur5e"
        and task.name == "place_approach"
        and not independent_place_approach
        and str(args.get("destination_location") or "").strip()
        == "assembly_board-v1"
        and assembly_surface_role == "assembly_slot"
    ):
        handoff_error = _physical_place_approach_held_part_handoff_error(
            dict(runtime_state.get("_task_ctx") or {}),
            part_name=args.get("part_name"),
        )
        if handoff_error:
            return agent._task_failure(
                handoff_error,
                step="place_approach.held_part_handoff_preflight",
                observations={
                    "destination_location": str(
                        args.get("destination_location") or ""
                    ),
                    "part_name": str(args.get("part_name") or ""),
                },
            )

    failure_part = _resolve_value(
        task.program.failure_part,
        args=args,
        runtime_state=runtime_state,
        step_outputs=step_outputs,
    )
    failure_part_name = str(failure_part or "").strip()
    injected = await agent._maybe_inject_failure(
        function_name=task.name,
        checkpoint="before_execute",
        part_name=failure_part_name,
        call_args=deepcopy(args),
    )
    if injected is not None:
        return injected

    if getattr(agent, "execution_mode", "dry_run") == "dry_run":
        description = _resolve_value(
            task.program.dry_run_description,
            args=args,
            runtime_state=runtime_state,
            step_outputs=step_outputs,
        )
        await agent._simulate_action(
            str(description or f"Executing {task.name}"),
            duration=float(task.program.dry_run_duration or 5.0),
        )

    completed_step_ids: set[str] = set()
    computed_cartesian_positions: dict[str, dict[str, float]] = {}
    computed_cartesian_reference: dict[str, Any] = {}
    computed_cartesian_at = 0.0
    resolved_cartesian_positions: dict[str, dict[str, float]] = {}
    for step in task.program.steps:
        progress_callback = getattr(agent, "_robot_task_progress_callback", None)
        if callable(progress_callback):
            try:
                progress_callback(task.name, step.id)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                agent.logger.warning(
                    "[Robot] %s.%s progress update failed: %s",
                    task.name,
                    step.id,
                    exc,
                )
        if (
            execution_mode == "physical"
            and task.name == "place_insert"
            and not independent_place_insert
            and step.id in {"move_insert", "release_part"}
            and str(args.get("destination_location") or "").strip()
            == "assembly_board-v1"
        ):
            board_lock_error = _physical_place_insert_board_lock_error(
                agent=agent,
                args=args,
                runtime_state=runtime_state,
                require_seated_pose=step.id == "release_part",
            )
            if board_lock_error:
                return agent._task_failure(
                    board_lock_error,
                    step="place_insert.assembly_board_v1_aruco_generation_lock",
                    observations={
                        "destination_location": str(
                            args.get("destination_location") or ""
                        ),
                        "part_name": str(args.get("part_name") or ""),
                        "move_insert_dispatched": (
                            "move_insert" in completed_step_ids
                        ),
                    },
                )
        original_task_context: dict[str, Any] | None = None
        if (
            manual_function_execution
            and task.name == "place_approach"
            and step.id == "compute_place_targets"
            and runtime_state.get("_held_part") not in (None, "")
        ):
            original_task_context = dict(runtime_state.get("_task_ctx") or {})
            runtime_state["_task_ctx"] = {
                **original_task_context,
                "manual_function_execution": True,
            }
        try:
            result = await _execute_task_step(
                agent=agent,
                task=task,
                step=step,
                args=args,
                runtime_state=runtime_state,
                step_outputs=step_outputs,
                physical_overrides=physical_overrides,
            )
        finally:
            if original_task_context is not None:
                runtime_state["_task_ctx"] = original_task_context
        if result.get("skipped"):
            continue
        if not result.get("success"):
            simulation_lift_after_release = (
                str(getattr(agent, "execution_mode", "") or "").strip().lower() == "simulation"
                and task.name == "place_insert"
                and step.id == "lift"
                and "release_part" in completed_step_ids
            )
            simulation_snap_part_to_slot = (
                str(getattr(agent, "execution_mode", "") or "").strip().lower() == "simulation"
                and task.name == "place_insert"
                and step.id == "snap_part_to_slot"
            )
            if (
                step.continue_on_failure and not simulation_snap_part_to_slot
            ) or simulation_lift_after_release:
                raw = dict(result.get("raw") or {})
                agent.logger.warning(
                    "[Robot] %s.%s soft-failed: %s",
                    task.name,
                    step.id,
                    raw.get("message") or "step failed",
                )
                continue
            raw = dict(result.get("raw") or {})
            failure_message = str(raw.get("message") or f"{step.op} failed")
            if task.name == "pick_approach" and step.id == "move_above_part":
                completed_motions = [
                    completed_step.id
                    for completed_step in task.program.steps
                    if completed_step.id in completed_step_ids
                    and completed_step.op in {"move_to_named_pose", "move_cartesian"}
                ]
                completed_detail = ", ".join(completed_motions) or "none"
                failure_message = (
                    f"{failure_message} Completed motion steps: {completed_detail}. "
                    "move_above_part was requested but did not complete; descend was "
                    "not commanded."
                )
            elif task.name == "pick_approach" and step.id == "descend":
                completed_motions = [
                    completed_step.id
                    for completed_step in task.program.steps
                    if completed_step.id in completed_step_ids
                    and completed_step.op in {"move_to_named_pose", "move_cartesian"}
                ]
                completed_detail = ", ".join(completed_motions) or "none"
                failure_message = (
                    f"{failure_message} Completed motion steps: {completed_detail}. "
                    "descend was requested but did not complete."
                )
            elif task.name == "place_approach" and step.id == "move_above_destination":
                completed_motions = [
                    completed_step.id
                    for completed_step in task.program.steps
                    if completed_step.id in completed_step_ids
                    and completed_step.op in {"move_to_named_pose", "move_cartesian"}
                ]
                completed_detail = ", ".join(completed_motions) or "none"
                failure_message = (
                    f"{failure_message} Completed motion steps: {completed_detail}. "
                    "move_above_destination was requested but did not complete; descend "
                    "was not commanded."
                )
            elif task.name == "place_approach" and step.id == "descend":
                completed_motions = [
                    completed_step.id
                    for completed_step in task.program.steps
                    if completed_step.id in completed_step_ids
                    and completed_step.op in {"move_to_named_pose", "move_cartesian"}
                ]
                completed_detail = ", ".join(completed_motions) or "none"
                failure_message = (
                    f"{failure_message} Completed motion steps: {completed_detail}. "
                    "descend was requested but did not complete."
                )
            elif task.name == "place_insert" and step.id == "move_insert":
                failure_message = (
                    f"{failure_message} move_insert did not complete; release_part and "
                    "lift were not commanded. The robot remains positioned with the "
                    "part held and the gripper closed."
                )
            elif task.name == "place_insert" and "release_part" in completed_step_ids:
                failure_message = (
                    f"{failure_message} release_part already completed; the part is no "
                    "longer clamped, and no later place_insert step was commanded."
                )
            if (
                task.name == "pick_approach"
                and step.id in {"detect_parts", "compute_pick_targets", "open_gripper"}
                and "move_to_origin_resource_location" in completed_step_ids
            ):
                failure_message = (
                    f"{failure_message} Completed motion steps: "
                    "move_to_origin_resource_location. move_above_part and descend were "
                    "not commanded."
                )
            if (
                task.name == "place_approach"
                and step.id
                in {"localize_assembly_board_v1", "compute_place_targets"}
                and "move_to_destination_location" in completed_step_ids
            ):
                failure_message = (
                    f"{failure_message} Completed motion steps: "
                    "move_to_destination_location. move_above_destination and descend "
                    "were not commanded."
                )
            failure_observations = _resolve_value(
                step.failure_observations,
                args=args,
                runtime_state=runtime_state,
                step_outputs=step_outputs,
            )
            if not isinstance(failure_observations, dict):
                failure_observations = {}
            if task.name == "place_insert":
                failure_observations["move_insert_dispatched"] = bool(
                    "move_insert" in completed_step_ids
                    or step.id == "move_insert"
                )
            if task.name == "place_insert" and step.id == "move_insert":
                failure_observations["move_insert_result"] = (
                    _sanitized_move_insert_result(raw)
                )
            return agent._task_failure(
                failure_message,
                step=f"{task.name}.{step.id}",
                observations=failure_observations,
            )
        payload = result.get("payload")
        if (
            execution_mode == "physical"
            and task.name == "place_approach"
            and step.id == "compute_place_targets"
            and isinstance(payload, dict)
        ):
            payload, qualification_error = (
                _apply_place_approach_recording_qualification(
                    payload,
                    physical_recording_path,
                )
            )
            if qualification_error:
                return agent._task_failure(
                    qualification_error,
                    step="place_approach.move_insert_qualification",
                    observations={"part_name": str(args.get("part_name") or "")},
                )
        completed_step_ids.add(str(step.id))
        if task.name == "place_insert" and step.id == "release_part":
            # Releasing the part is physically irreversible. Commit custody as soon as
            # the primitive succeeds so a later delay, lift, or bookkeeping failure
            # cannot leave the cached agent claiming that the part is still clamped.
            runtime_state["_held_part"] = None
            runtime_state["_gripper_state"] = "open"
            _commit_runtime_state(agent, runtime_state)
        if step.store_as:
            step_outputs[step.store_as] = deepcopy(payload if payload is not None else {})
            raw_positions, raw_reference = _computed_cartesian_state(
                task=task,
                step_outputs=step_outputs,
            )
            if raw_positions:
                computed_cartesian_positions = raw_positions
                computed_cartesian_reference = raw_reference
                computed_cartesian_at = time.time()
                computed_pose_callback = getattr(
                    agent,
                    "_robot_task_computed_pose_callback",
                    None,
                )
                if callable(computed_pose_callback):
                    try:
                        computed_pose_callback(
                            task.name,
                            deepcopy(computed_cartesian_positions),
                            deepcopy(computed_cartesian_reference),
                            computed_cartesian_at,
                        )
                    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                        agent.logger.warning(
                            "[Robot] %s computed pose update failed: %s",
                            task.name,
                            exc,
                        )
            override_error = _apply_cartesian_overrides_to_targets(
                task=task,
                physical_overrides=physical_overrides,
                step_outputs=step_outputs,
                agent=agent,
            )
            if (
                not override_error
                and execution_mode == "physical"
                and _robot_name(agent) == "ur5e"
                and task.name == "place_approach"
            ):
                place_targets = step_outputs.get("place_targets")
                move_insert_mode = (
                    str(place_targets.get("move_insert_mode") or "")
                    if isinstance(place_targets, dict)
                    else ""
                )
                if move_insert_mode in {"force_limited", "force_limited_trial"}:
                    raw_hard_caps = (
                        raw_product_geometry.get("move_insert_hard_caps")
                        if isinstance(raw_product_geometry, dict)
                        else None
                    )
                    boundary, boundary_error = _validated_move_insert_boundary(
                        start_pose=place_targets.get("target_pose"),
                        targets=place_targets,
                        raw_hard_caps=raw_hard_caps,
                    )
                    if boundary_error:
                        place_targets["move_insert_boundary_ready"] = False
                        place_targets["move_insert_boundary_error"] = boundary_error
                    else:
                        place_targets.update(
                            {
                                "insertion_axis_world": deepcopy(
                                    boundary["insertion_axis_world"]
                                ),
                                "move_insert_timeout_sec": float(
                                    boundary["move_insert_timeout_sec"]
                                ),
                                "move_insert_hard_caps": deepcopy(
                                    boundary["move_insert_hard_caps"]
                                ),
                                "move_insert_hard_caps_sha256": str(
                                    boundary["move_insert_hard_caps_sha256"]
                                ),
                                "move_insert_boundary_metrics": {
                                    field: float(boundary[field])
                                    for field in (
                                        "insertion_depth_m",
                                        "insertion_travel_m",
                                        "lateral_error_m",
                                        "orientation_error_rad",
                                        "learned_start_position_error_m",
                                        "learned_start_orientation_error_rad",
                                    )
                                },
                                "move_insert_boundary_ready": True,
                                "move_insert_boundary_error": "",
                            }
                        )
            if override_error:
                if (
                    task.name == "pick_approach"
                    and "move_to_origin_resource_location" in completed_step_ids
                ):
                    completed_motions = [
                        completed_step.id
                        for completed_step in task.program.steps
                        if completed_step.id in completed_step_ids
                        and completed_step.op in {"move_to_named_pose", "move_cartesian"}
                    ]
                    override_error = (
                        f"{override_error} Completed motion steps: "
                        f"{', '.join(completed_motions)}. move_above_part and descend were "
                        "not commanded."
                    )
                elif (
                    task.name == "place_approach"
                    and "move_to_destination_location" in completed_step_ids
                ):
                    completed_motions = [
                        completed_step.id
                        for completed_step in task.program.steps
                        if completed_step.id in completed_step_ids
                        and completed_step.op in {"move_to_named_pose", "move_cartesian"}
                    ]
                    override_error = (
                        f"{override_error} Completed motion steps: "
                        f"{', '.join(completed_motions)}. move_above_destination and "
                        "descend were not commanded."
                    )
                return agent._task_failure(
                    override_error,
                    step=f"{task.name}.physical_position_preflight",
                    observations={
                        "function_name": task.name,
                        "physical_position_file": (
                            str(physical_recording_path)
                            if physical_recording_path is not None
                            else ""
                        ),
                    },
                )
        if isinstance(payload, dict) and "absolute_position" in payload:
            absolute_position = deepcopy(payload["absolute_position"])
            runtime_state["_position"] = absolute_position
            if step.op in {"move_cartesian", "move_insert"}:
                resolved_cartesian_positions[step.id] = absolute_position
            if step.op == "move_insert":
                task_context = dict(runtime_state.get("_task_ctx") or {})
                retained_positions = dict(
                    task_context.get("resolved_cartesian_positions") or {}
                )
                retained_positions["move_insert"] = deepcopy(absolute_position)
                task_context["resolved_cartesian_positions"] = retained_positions
                task_context["move_insert_result"] = _sanitized_move_insert_result(
                    result.get("raw")
                )
                runtime_state["_task_ctx"] = task_context
                if task.name == "place_insert":
                    runtime_state["_current_state"] = "positioned"
                    runtime_state["_held_part"] = deepcopy(
                        getattr(agent, "_held_part", None)
                    )
                    runtime_state["_gripper_state"] = "closed"
                    _commit_runtime_state(agent, runtime_state)

    injected = await agent._maybe_inject_failure(
        function_name=task.name,
        checkpoint="after_execute_before_commit",
        part_name=failure_part_name,
        call_args=deepcopy(args),
    )
    if injected is not None:
        return injected

    for effect in task.program.effects:
        should_apply = True
        for guard in effect.when:
            if not _evaluate_guard(
                guard,
                agent=agent,
                args=args,
                runtime_state=runtime_state,
                step_outputs=step_outputs,
            ):
                should_apply = False
                break
        if should_apply:
            _apply_effect(effect, args=args, runtime_state=runtime_state, step_outputs=step_outputs)

    if task.name == "pick_grasp" and held_part_handoff:
        task_context = dict(runtime_state.get("_task_ctx") or {})
        task_context["held_part_handoff"] = deepcopy(held_part_handoff)
        runtime_state["_task_ctx"] = task_context

    resolved_cartesian_state_error = _apply_resolved_cartesian_state(
        task=task,
        physical_overrides=physical_overrides,
        computed_positions=computed_cartesian_positions,
        computed_reference=computed_cartesian_reference,
        computed_at=computed_cartesian_at,
        resolved_positions=resolved_cartesian_positions,
        runtime_state=runtime_state,
        step_outputs=step_outputs,
    )
    if resolved_cartesian_state_error:
        return agent._task_failure(
            resolved_cartesian_state_error,
            step=f"{task.name}.resolved_cartesian_state",
            observations={
                "function_name": task.name,
                "part_name": str(args.get("part_name") or ""),
            },
        )
    if independent_place_approach:
        runtime_state["_held_part"] = deepcopy(getattr(agent, "_held_part", None))
        runtime_state["_current_state"] = deepcopy(
            getattr(agent, "_current_state", "")
        )
        runtime_state["_gripper_state"] = deepcopy(
            getattr(agent, "_gripper_state", "")
        )
        runtime_state["_recovery_pose_ref"] = deepcopy(
            getattr(agent, "_recovery_pose_ref", None)
        )
        runtime_state["_task_ctx"] = deepcopy(getattr(agent, "_task_ctx", {}))
    elif independent_place_insert:
        runtime_state["_held_part"] = deepcopy(getattr(agent, "_held_part", None))
        runtime_state["_current_state"] = deepcopy(
            getattr(agent, "_current_state", "")
        )
        runtime_state["_recovery_pose_ref"] = deepcopy(
            getattr(agent, "_recovery_pose_ref", None)
        )
        runtime_state["_task_ctx"] = deepcopy(getattr(agent, "_task_ctx", {}))
    _commit_runtime_state(agent, runtime_state)
    response = _resolve_value(
        task.program.success_response,
        args=args,
        runtime_state=runtime_state,
        step_outputs=step_outputs,
    )
    if not isinstance(response, dict):
        response = {"status": "completed", "content": str(response or "")}
    response.setdefault("status", "completed")
    if operator_held_place_approach:
        retained_task_context = dict(runtime_state.get("_task_ctx") or {})
        move_insert_trial_context_ready = bool(
            retained_task_context.get("move_insert_mode")
            in {"force_limited", "force_limited_trial"}
            and isinstance(retained_task_context.get("move_insert_profile"), dict)
        )
        response.update(
            {
                "content": (
                    "Completed place_approach after adopting operator-confirmed "
                    f"held_part {requested_part_name!r} from the confirmed "
                    "pick_approach.descend handoff; "
                    f"{requested_part_name} remains clamped."
                    + (
                        " The retained context is ready for supervised move_insert "
                        "readiness checks."
                        if move_insert_trial_context_ready
                        else " place_approach did not require move_insert; Supervised "
                        "Test move_insert remains blocked until its protected recipe "
                        "and calibration are ready."
                    )
                ),
                "operator_confirmed_held_part": True,
                "operator_held_part": requested_part_name,
                "held_part_handoff_adopted": True,
                "move_insert_trial_context_ready": move_insert_trial_context_ready,
                "move_insert_authorized": False,
            }
        )
    elif independent_place_approach:
        response["content"] = (
            "Completed independent place_approach with held_part empty; "
            "the RobotAgent pick/place context was preserved."
        )
    elif independent_place_insert:
        response["content"] = (
            "Completed independent place_insert with held_part empty; opened the gripper "
            "and retreated 0.08 m without advancing the RobotAgent pick/place context."
        )
        response.pop("placed_location", None)
    return response


def _place_insert_move_insert_trial_error(
    agent: Any,
    *,
    destination_location: str,
    part_name: str,
    runtime_state: dict[str, Any],
) -> str:
    """Validate the exact positioned state used by a supervised move_insert trial."""
    if str(getattr(agent, "execution_mode", "") or "") != "physical":
        return "Supervised Test move_insert requires physical execution mode."
    if _robot_name(agent) != "ur5e":
        return "Supervised Test move_insert is currently available only for ur5e."
    if destination_location != "assembly_board-v1":
        return "Supervised Test move_insert requires assembly_board-v1."
    if part_name not in _MOVE_INSERT_TRANSLATIONAL_PARTS:
        return (
            "Supervised Test move_insert requires one of the exact supported parts "
            f"{list(_MOVE_INSERT_TRANSLATIONAL_PARTS)}."
        )
    if runtime_state.get("_current_state") != "positioned":
        return "Supervised Test move_insert requires place_approach to finish at positioned."
    if runtime_state.get("_held_part") != part_name:
        return f"Supervised Test move_insert requires ur5e to hold exact part {part_name}."
    if runtime_state.get("_gripper_state") != "closed":
        return "Supervised Test move_insert requires the ur5e gripper to remain closed."
    task_context = dict(runtime_state.get("_task_ctx") or {})
    if task_context.get("destination_location") != destination_location:
        return "Supervised Test move_insert destination does not match place_approach."
    if task_context.get("part_name") != part_name:
        return "Supervised Test move_insert part identity does not match place_approach."
    if task_context.get("move_insert_mode") not in {
        "force_limited",
        "force_limited_trial",
    }:
        return "place_approach did not establish a physical move_insert trial profile."
    if not isinstance(task_context.get("move_insert_profile"), dict):
        return "place_approach did not retain the move_insert profile."
    if not isinstance(task_context.get("held_part_handoff"), dict):
        return "pick_grasp did not retain a complete held-part handoff."
    return _physical_place_insert_board_lock_error(
        agent=agent,
        args={
            "destination_location": destination_location,
            "part_name": part_name,
        },
        runtime_state=runtime_state,
        require_seated_pose=False,
    )


async def execute_place_insert_move_insert_trial(
    agent: Any,
    *,
    destination_location: str,
    part_name: str,
    trial_id: str,
) -> dict[str, Any]:
    """Execute only ``place_insert.move_insert`` and retain the exact part in the gripper."""
    if not isinstance(trial_id, str) or not trial_id or trial_id != trial_id.strip():
        return {
            "status": "blocked",
            "content": "Supervised Test move_insert trial_id is missing or invalid.",
        }

    def no_motion_result(status: str, content: str) -> dict[str, Any]:
        move_insert_result = {
            "success": False,
            "trial_id": trial_id,
            "state_uncertain": False,
            "motion_settled": True,
            "dispatch_attempted": False,
        }
        return {
            "status": status,
            "content": content,
            "trial_id": trial_id,
            "motion_settled": True,
            "dispatch_attempted": False,
            "move_insert_result": move_insert_result,
            "move_insert_result_sha256": _move_insert_result_sha256(
                move_insert_result
            ),
            "trial_ready_for_confirmation": False,
        }

    task = robot_task_registry().get("place_insert")
    if task is None:
        return no_motion_result("failed", "place_insert task is unavailable")
    runtime_state = _build_runtime_state(agent)
    task_context = dict(runtime_state.get("_task_ctx") or {})
    original_mode = str(task_context.get("move_insert_mode") or "")
    error = _place_insert_move_insert_trial_error(
        agent,
        destination_location=destination_location,
        part_name=part_name,
        runtime_state=runtime_state,
    )
    if error:
        return no_motion_result("blocked", error)
    task_context = dict(runtime_state.get("_task_ctx") or {})
    task_context["move_insert_mode"] = "force_limited"
    had_trial_id = "move_insert_trial_id" in task_context
    original_trial_id = task_context.get("move_insert_trial_id")
    task_context["move_insert_trial_id"] = trial_id
    runtime_state["_task_ctx"] = task_context
    move_insert_step = next(
        (step for step in task.program.steps if step.id == "move_insert"),
        None,
    )
    if move_insert_step is None:
        return no_motion_result(
            "failed",
            "place_insert.move_insert is unavailable",
        )
    args = {
        "destination_location": destination_location,
        "part_name": part_name,
    }
    result = await _execute_task_step(
        agent=agent,
        task=task,
        step=move_insert_step,
        args=args,
        runtime_state=runtime_state,
        step_outputs={},
        physical_overrides={},
    )
    raw_result = _sanitized_move_insert_result(result.get("raw"))
    task_context = dict(runtime_state.get("_task_ctx") or {})
    if had_trial_id:
        task_context["move_insert_trial_id"] = original_trial_id
    else:
        task_context.pop("move_insert_trial_id", None)
    runtime_state["_task_ctx"] = task_context
    payload = result.get("payload")
    if isinstance(payload, dict) and "absolute_position" in payload:
        absolute_position, pose_error = _normalized_se3_pose(
            payload.get("absolute_position"),
            label="Supervised Test move_insert final world -> tool0 pose",
        )
        if not pose_error:
            runtime_state["_position"] = deepcopy(absolute_position)
            retained_positions = dict(
                dict(runtime_state.get("_task_ctx") or {}).get(
                    "resolved_cartesian_positions"
                )
                or {}
            )
            retained_positions["move_insert"] = deepcopy(absolute_position)
            task_context = dict(runtime_state.get("_task_ctx") or {})
            task_context["resolved_cartesian_positions"] = retained_positions
            runtime_state["_task_ctx"] = task_context
    settled_success = bool(
        result.get("success")
        and raw_result.get("success") is True
        and raw_result.get("trial_id") == trial_id
        and raw_result.get("state_uncertain") is not True
        and raw_result.get("motion_settled") is True
        and raw_result.get("final_tool0_pose_valid") is True
        and raw_result.get("engagement_detected") is True
        and raw_result.get("seated_detected") is True
    )
    task_context = dict(runtime_state.get("_task_ctx") or {})
    task_context["move_insert_mode"] = original_mode or "force_limited_trial"
    task_context["move_insert_result"] = deepcopy(raw_result)
    task_context["move_insert_trial_result"] = deepcopy(raw_result)
    result_sha256 = _move_insert_result_sha256(raw_result)
    task_context["move_insert_trial_result_sha256"] = result_sha256
    runtime_state["_task_ctx"] = task_context
    runtime_state["_current_state"] = "positioned"
    runtime_state["_held_part"] = part_name
    runtime_state["_gripper_state"] = "closed"
    _commit_runtime_state(agent, runtime_state)
    if not settled_success:
        message = str(
            dict(result.get("raw") or {}).get("message")
            or "move_insert did not establish confirmed engagement and seating"
        )
        return {
            "status": "failed",
            "content": (
                f"{message} {part_name} remains clamped; release_part, lift, and move_home "
                "were not commanded."
            ),
            "move_insert_result": raw_result,
            "move_insert_result_sha256": result_sha256,
            "trial_id": trial_id,
            "trial_ready_for_confirmation": False,
        }
    return {
        "status": "completed",
        "content": (
            "Supervised Test move_insert established engagement and seating. "
            f"{part_name} remains clamped pending Confirm Completion."
        ),
        "move_insert_result": raw_result,
        "move_insert_result_sha256": result_sha256,
        "trial_id": trial_id,
        "trial_ready_for_confirmation": True,
    }


async def complete_place_insert_after_move_insert_trial(
    agent: Any,
    *,
    destination_location: str,
    part_name: str,
    expected_move_insert_result_sha256: str,
) -> dict[str, Any]:
    """Release and lift after a reviewed trial without executing move_insert again."""
    task = robot_task_registry().get("place_insert")
    if task is None:
        return {"status": "failed", "content": "place_insert task is unavailable"}
    runtime_state = _build_runtime_state(agent)
    task_context = dict(runtime_state.get("_task_ctx") or {})
    error = _place_insert_move_insert_trial_error(
        agent,
        destination_location=destination_location,
        part_name=part_name,
        runtime_state=runtime_state,
    )
    if error:
        return {"status": "blocked", "content": error}
    retained_result = _sanitized_move_insert_result(
        task_context.get("move_insert_trial_result")
    )
    retained_sha256 = str(
        task_context.get("move_insert_trial_result_sha256") or ""
    )
    if (
        not expected_move_insert_result_sha256
        or retained_sha256 != expected_move_insert_result_sha256
        or _move_insert_result_sha256(retained_result) != retained_sha256
    ):
        return {
            "status": "blocked",
            "content": "Supervised move_insert trial result identity changed before confirmation.",
        }
    if (
        retained_result.get("success") is not True
        or retained_result.get("state_uncertain") is True
        or retained_result.get("motion_settled") is not True
        or retained_result.get("final_tool0_pose_valid") is not True
        or retained_result.get("engagement_detected") is not True
        or retained_result.get("seated_detected") is not True
    ):
        return {
            "status": "blocked",
            "content": (
                "Confirm Completion requires confirmed engagement, seating, and "
                "stationary settlement evidence."
            ),
        }
    board_error = _physical_place_insert_board_lock_error(
        agent=agent,
        args={
            "destination_location": destination_location,
            "part_name": part_name,
        },
        runtime_state=runtime_state,
        require_seated_pose=True,
    )
    if board_error:
        return {"status": "blocked", "content": board_error}

    args = {
        "destination_location": destination_location,
        "part_name": part_name,
    }
    completed_steps: list[str] = []
    execute_suffix = False
    for step in task.program.steps:
        if step.id == "move_insert":
            execute_suffix = True
            continue
        if not execute_suffix:
            continue
        result = await _execute_task_step(
            agent=agent,
            task=task,
            step=step,
            args=args,
            runtime_state=runtime_state,
            step_outputs={},
            physical_overrides={},
        )
        if result.get("skipped"):
            continue
        if not result.get("success"):
            message = str(
                dict(result.get("raw") or {}).get("message")
                or f"place_insert.{step.id} failed"
            )
            return {
                "status": "failed",
                "content": message,
                "completed_steps": completed_steps,
            }
        completed_steps.append(step.id)
        if step.id == "release_part":
            # Release is physically irreversible. Commit the observed custody state
            # immediately so a later lift failure cannot leave the cached agent
            # claiming that the part is still clamped.
            runtime_state["_held_part"] = None
            runtime_state["_gripper_state"] = "open"
            _commit_runtime_state(agent, runtime_state)
        payload = result.get("payload")
        if isinstance(payload, dict) and "absolute_position" in payload:
            runtime_state["_position"] = deepcopy(payload["absolute_position"])

    for effect in task.program.effects:
        should_apply = all(
            _evaluate_guard(
                guard,
                agent=agent,
                args=args,
                runtime_state=runtime_state,
                step_outputs={},
            )
            for guard in effect.when
        )
        if should_apply:
            _apply_effect(
                effect,
                args=args,
                runtime_state=runtime_state,
                step_outputs={},
            )
    _commit_runtime_state(agent, runtime_state)
    return {
        "status": "completed",
        "content": (
            f"Confirmed and completed place_insert for {part_name} at "
            f"{destination_location}."
        ),
        "placed_location": destination_location,
        "completed_steps": completed_steps,
    }
