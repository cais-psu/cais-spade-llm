"""Execution runtime for registry-backed robot tasks."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

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
    payload.setdefault("origin_pose", {"x": tx, "y": ty, "z": tz})
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

    if step.op == "move_cartesian":
        payload = dict(payload or {})
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


def _safe_recording_name(name: Any) -> str:
    value = str(name or "").strip()
    safe = "".join(
        character if (character.isalnum() or character in "-_") else "_" for character in value
    )
    return safe or "default"


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


def _required_physical_position_steps(task: RobotTaskDefinition) -> tuple[RobotTaskStep, ...]:
    return tuple(step for step in task.program.steps if step.physical_position_required)


def _cartesian_position_steps(task: RobotTaskDefinition) -> tuple[RobotTaskStep, ...]:
    return tuple(step for step in task.program.steps if step.op == "move_cartesian")


def _recording_location(task: RobotTaskDefinition, args: dict[str, Any]) -> tuple[str, str]:
    location_param = str(
        dict(task.program.context_mapping or {}).get("location_param") or ""
    ).strip()
    if not location_param:
        return "", "physical position task has no location_param"
    location = str(args.get(location_param) or "").strip()
    if not location:
        return "", f"{task.name} requires {location_param} for physical position lookup"
    return location, ""


def _recording_part_name(args: dict[str, Any]) -> tuple[str, str]:
    part_name = str(args.get("part_name") or "").strip()
    if not part_name:
        return "", "physical position lookup requires part_name"
    return part_name, ""


def _recorded_joints_error(
    *,
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
        or len(joint_names) != 6
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
    expected_joint_names = _HARDWARE_JOINT_NAMES.get(robot)
    recorded_joint_names = tuple(str(name).strip() for name in joint_names)
    if expected_joint_names is None or set(recorded_joint_names) != set(expected_joint_names):
        return (
            f"physical position joints do not match the exact {robot} six-joint hardware "
            f"set for {task.name}.{step.id}"
        )
    return ""


def _recorded_cartesian_pose(
    *,
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
    if str(pose.get("frame_id") or "").strip() != "world":
        return {}, f"physical position frame must be world for {task.name}.{step.id}"
    expected_child_frame = "link_eef" if robot == "xarm6" else "tool0"
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
    location: str,
    part_name: str,
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
    expected_name = part_name if expected_kind == "detected_part" else location
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
    if name != expected_name:
        return {}, {}, (
            f"relative_reference.name mismatch for {task.name}.{step.id}: "
            f"expected {expected_name!r}, found {name or '<empty>'!r}"
        )
    if not source:
        return {}, {}, f"relative_reference.source is missing for {task.name}.{step.id}"
    is_assembly_board_v1 = (
        task.name == "place_approach" and location == "assembly_board-v1"
    )
    if is_assembly_board_v1 and source != "assembly_board-v1_aruco":
        return {}, {}, (
            "relative_reference.source must be assembly_board-v1_aruco for "
            f"{task.name}.{step.id}"
        )
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
    if is_assembly_board_v1:
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
        reference_pose, reference_pose_error = _normalized_se3_pose(
            raw_reference.get("pose"),
            label=f"relative_reference.pose for {task.name}.{step.id}",
        )
        if reference_pose_error:
            return {}, {}, reference_pose_error
        if any(
            abs(reference_pose[field] - reference_position_m[field]) > 1e-6
            for field in _CARTESIAN_POSITION_FIELDS
        ):
            return {}, {}, (
                "relative_reference.pose translation differs from position_m for "
                f"{task.name}.{step.id}"
            )
        reference.update(
            {
                "camera_role": camera_role,
                "generation": deepcopy(generation),
                "pose": reference_pose,
            }
        )
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
    robot: str,
    task: RobotTaskDefinition,
    step: RobotTaskStep,
    recorded_step: dict[str, Any],
    location: str,
    part_name: str,
) -> tuple[dict[str, Any], str]:
    pose, error = _recorded_cartesian_pose(
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
    is_assembly_board_v1 = (
        task.name == "place_approach" and location == "assembly_board-v1"
    )
    if is_assembly_board_v1 and relative_fields != set(_CARTESIAN_POSITION_FIELDS):
        return {}, (
            "physical place recording must use captured_relative for x, y, and z and "
            f"include full relative_pose for {task.name}.{step.id}"
        )
    if relative_fields and relative_fields != set(_CARTESIAN_POSITION_FIELDS):
        return {}, (
            f"captured_relative must be selected for x, y, and z together for "
            f"{task.name}.{step.id}"
        )
    relative_position_m: dict[str, float] = {}
    relative_pose: dict[str, float] = {}
    relative_reference: dict[str, Any] = {}
    computed_position_m: dict[str, float] = {}
    computed_source = ""
    computed_at: float | None = None
    if relative_fields:
        relative_position_m, relative_reference, relative_error = (
            _recorded_relative_position(
                robot=robot,
                task=task,
                step=step,
                recorded_step=recorded_step,
                location=location,
                part_name=part_name,
            )
        )
        if relative_error:
            return {}, relative_error
        if is_assembly_board_v1:
            relative_pose, relative_pose_error = _recorded_relative_pose(
                task=task,
                step=step,
                recorded_step=recorded_step,
                reference_pose=dict(relative_reference["pose"]),
                captured_pose=pose,
            )
            if relative_pose_error:
                return {}, relative_pose_error
        raw_computed_position = recorded_step.get("computed_position_m")
        if raw_computed_position is not None:
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
    required_steps = _required_physical_position_steps(task)
    cartesian_steps = _cartesian_position_steps(task)
    if not cartesian_steps:
        return {}, None, ""

    robot = _robot_name(agent)
    if robot not in {"ur5e", "xarm6"}:
        return (
            {},
            None,
            f"physical position recording is not supported for {robot or '<unknown>'}",
        )
    location, error = _recording_location(task, args)
    if error:
        return {}, None, error
    part_name, error = _recording_part_name(args)
    if error:
        return {}, None, error

    path = (
        _TAUGHT_FUNCTIONS_ROOT
        / robot
        / task.name
        / (f"{_safe_recording_name(location)}__{_safe_recording_name(part_name)}__hardware.json")
    )
    try:
        with path.open("r", encoding="utf-8") as recording_file:
            payload = json.load(recording_file)
    except FileNotFoundError:
        if required_steps and not allow_missing:
            return {}, path, f"physical position file not found: {path}"
        return {}, path, ""
    except (OSError, json.JSONDecodeError) as exc:
        return {}, path, f"could not load physical position file {path}: {exc}"
    if not isinstance(payload, dict):
        return {}, path, f"physical position file is not a JSON object: {path}"

    if str(payload.get("robot") or "").strip().lower() != robot:
        return {}, path, f"physical position robot mismatch in {path}"
    if str(payload.get("function_name") or "").strip() != task.name:
        return {}, path, f"physical position function mismatch in {path}"
    if str(payload.get("name") or "").strip() != location:
        return {}, path, f"physical position location mismatch in {path}"
    if str(payload.get("part_name") or "").strip() != part_name:
        return {}, path, f"physical position part_name mismatch in {path}"
    if str(payload.get("capture_source") or "").strip().lower() != "hardware":
        return {}, path, f"physical position capture_source must be hardware in {path}"

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return {}, path, f"physical position steps must be a list in {path}"
    recorded_by_id: dict[str, dict[str, Any]] = {}
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            return {}, path, f"physical position step must be an object in {path}"
        step_id = str(raw_step.get("step_name") or "").strip()
        if not step_id:
            return {}, path, f"physical position step_name is empty in {path}"
        if step_id in recorded_by_id:
            return {}, path, f"duplicate physical position step_name {step_id} in {path}"
        recorded_by_id[step_id] = raw_step

    overrides: dict[str, dict[str, Any]] = {}
    required_step_ids = {step.id for step in required_steps}
    for step in cartesian_steps:
        recorded_step = recorded_by_id.get(step.id)
        if recorded_step is None:
            if step.id in required_step_ids and not allow_missing:
                return (
                    {},
                    path,
                    f"physical position step not found: {task.name}.{step.id} in {path}",
                )
            continue
        override, error = _recorded_cartesian_override(
            robot=robot,
            task=task,
            step=step,
            recorded_step=recorded_step,
            location=location,
            part_name=part_name,
        )
        if error:
            return {}, path, error
        overrides[step.id] = override
    return overrides, path, ""


def _apply_resolved_cartesian_state(
    *,
    task: RobotTaskDefinition,
    physical_overrides: dict[str, dict[str, Any]],
    computed_positions: dict[str, dict[str, float]],
    computed_reference: dict[str, Any],
    computed_at: float,
    resolved_positions: dict[str, dict[str, float]],
    runtime_state: dict[str, Any],
    step_outputs: dict[str, Any],
) -> None:
    cartesian_steps = _cartesian_position_steps(task)
    if not cartesian_steps:
        return
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
        localization = step_outputs.get("assembly_board_v1_aruco")
        if isinstance(localization, dict) and localization:
            task_context["assembly_board_v1_aruco"] = deepcopy(localization)
            task_context["assembly_board_v1_aruco_generation"] = deepcopy(
                localization.get("generation")
            )
    runtime_state["_task_ctx"] = task_context


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


def _apply_cartesian_overrides_to_targets(  # noqa: C901, PLR0912 - explicit reference and safety gates.
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
    if task.name == "pick_approach":
        current_reference = dict(targets.get("origin_pose") or {})
        expected_reference_kind = "detected_part"
        expected_reference_name = str(targets.get("part_name") or "").strip()
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
        expected_destination = str(targets.get("destination_location") or "").strip()
        if expected_destination == "assembly_board-v1":
            localization, localization_error = _assembly_board_v1_aruco_payload(
                step_outputs.get("assembly_board_v1_aruco"),
                robot=_robot_name(agent) if agent is not None else "",
                destination_location=expected_destination,
                require_fresh=True,
            )
            if localization_error:
                return localization_error
            current_reference_pose = dict(localization["pose"])
            current_reference = {
                field: current_reference_pose[field]
                for field in _CARTESIAN_POSITION_FIELDS
            }
        else:
            current_reference = dict(targets.get("target_pose") or {})
        expected_reference_kind = "destination_target"
        expected_reference_name = expected_destination
    if not expected_reference_name:
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
        relative_position_m = dict(override.get("relative_position_m") or {})
        relative_pose = dict(override.get("relative_pose") or {})
        relative_reference = dict(override.get("relative_reference") or {})
        if relative_position_m:
            if str(relative_reference.get("kind") or "") != expected_reference_kind:
                return f"{task.name}.{step.id} relative reference kind changed"
            if str(relative_reference.get("name") or "") != expected_reference_name:
                return (
                    f"{task.name}.{step.id} relative reference name changed: expected "
                    f"{expected_reference_name!r}"
                )
            is_assembly_board_v1 = (
                task.name == "place_approach"
                and expected_reference_name == "assembly_board-v1"
            )
            if is_assembly_board_v1:
                if str(relative_reference.get("source") or "") != "assembly_board-v1_aruco":
                    return (
                        f"{task.name}.{step.id} relative reference source changed"
                    )
                if str(relative_reference.get("camera_role") or "").strip().lower() != (
                    _robot_name(agent) if agent is not None else ""
                ):
                    return (
                        f"{task.name}.{step.id} relative reference camera_role changed"
                    )
                if not relative_pose:
                    return f"relative_pose is missing for {task.name}.{step.id}"
                pose = _compose_se3(current_reference_pose, relative_pose)
                override["values"] = {
                    **dict(override.get("values") or {}),
                    **{field: pose[field] for field in ("qx", "qy", "qz", "qw")},
                }
                relative_base = current_reference
            else:
                try:
                    relative_base = (
                        {
                            field: float(pose[field])
                            for field in _CARTESIAN_POSITION_FIELDS
                        }
                        if dict(override.get("computed_position_m") or {})
                        else current_reference
                    )
                    pose.update(
                        {
                            field: relative_base[field] + float(relative_position_m[field])
                            for field in _CARTESIAN_POSITION_FIELDS
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    return f"{task.name}.{step.id} relative XYZ is incomplete"
            override["resolved_reference_position_m"] = deepcopy(current_reference)
            override["resolved_computed_position_m"] = deepcopy(relative_base)
            override["resolved_position_m"] = {
                field: float(pose[field]) for field in _CARTESIAN_POSITION_FIELDS
            }
        if task.name != "place_approach" or expected_reference_name != "assembly_board-v1":
            pose.update(
                {
                    field: float(value)
                    for field, value in dict(override.get("values") or {}).items()
                    if field in (*_CARTESIAN_POSITION_FIELDS, "qx", "qy", "qz", "qw")
                }
            )
        pose_error = _resolved_cartesian_pose_error(
            agent=agent,
            task=task,
            step=step,
            pose=pose,
        )
        if pose_error:
            return pose_error
        targets[pose_key] = pose
    approach_pose = dict(targets.get("approach_pose") or {})
    target_pose = dict(targets.get("target_pose") or {})
    try:
        approach_z = float(approach_pose["z"])
        target_z = float(target_pose["z"])
    except (KeyError, TypeError, ValueError):
        return f"{task.name} taught Cartesian positions are incomplete"
    if not isfinite(approach_z) or not isfinite(target_z):
        return f"{task.name} taught Cartesian Z contains a non-finite value"
    if approach_z <= target_z + 1e-6:
        return (
            f"{task.name} taught move-above Z must remain above the descend Z "
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


async def _execute_task_step(
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
        step.physical_position_required or step.op == "move_cartesian"
    ) and step.id in physical_overrides:
        params.update(dict(physical_overrides[step.id].get("values") or {}))

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


def _physical_place_insert_board_lock_error(
    *,
    agent: Any,
    args: dict[str, Any],
    runtime_state: dict[str, Any],
) -> str:
    task_context = dict(runtime_state.get("_task_ctx") or {})
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
    return ""


async def execute_robot_task(  # noqa: C901, PLR0912, PLR0915
    agent: Any,
    task_name: str,
    manual_function_execution_authority: object | None = None,
    /,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute one exact registry-backed task against the selected robot mode.

    Args:
        agent: RobotAgent-compatible runtime owner.
        task_name: Exact registered robot function name.
        manual_function_execution_authority: Internal identity capability for the
            Control-page manual Function Execution path.
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

    for guard in task.program.entry_guards:
        # Manual Control commissioning bypasses only the assembly sequence token;
        # physical, held-part, gripper, and task-context guards stay authoritative.
        condition = dict(guard.condition or {})
        if (
            manual_function_execution
            and str(condition.get("field") or "").strip() == "resource_state"
            and str(condition.get("operator") or "").strip() == "equals"
            and str(condition.get("value") or "").strip()
            == str(task.program.entry_state or "").strip()
        ):
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

    physical_overrides: dict[str, dict[str, Any]] = {}
    physical_recording_path: Path | None = None
    execution_mode = str(getattr(agent, "execution_mode", "") or "").strip().lower()
    if (
        execution_mode == "physical"
        and task.name == "place_insert"
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
            and step.id == "release_part"
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
                        "destination_location": str(
                            args.get("destination_location") or ""
                        ),
                        "part_name": str(args.get("part_name") or ""),
                    },
                )
        result = await _execute_task_step(
            agent=agent,
            task=task,
            step=step,
            args=args,
            runtime_state=runtime_state,
            step_outputs=step_outputs,
            physical_overrides=physical_overrides,
        )
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
            return agent._task_failure(
                failure_message,
                step=f"{task.name}.{step.id}",
                observations=_resolve_value(
                    step.failure_observations,
                    args=args,
                    runtime_state=runtime_state,
                    step_outputs=step_outputs,
                ),
            )
        payload = result.get("payload")
        completed_step_ids.add(str(step.id))
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
            if step.op == "move_cartesian":
                resolved_cartesian_positions[step.id] = absolute_position

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

    _apply_resolved_cartesian_state(
        task=task,
        physical_overrides=physical_overrides,
        computed_positions=computed_cartesian_positions,
        computed_reference=computed_cartesian_reference,
        computed_at=computed_cartesian_at,
        resolved_positions=resolved_cartesian_positions,
        runtime_state=runtime_state,
        step_outputs=step_outputs,
    )
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
    return response
