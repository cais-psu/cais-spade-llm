"""Pure helpers for `digital_twin` target configuration and status paths."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def digital_twin_target(
    targets: dict[str, dict[str, Any]],
    target: str,
) -> dict[str, Any] | None:
    """Return the configured `digital_twin` target entry."""
    return targets.get(str(target or "").strip().lower())


def digital_twin_slug(cfg: dict[str, Any]) -> str:
    """Return the file-safe slug stored on a `digital_twin` target entry."""
    return str(cfg.get("slug") or "digital_twin").strip()


def digital_twin_status_path(
    targets: dict[str, dict[str, Any]],
    target: str,
    status_root: Path,
) -> Path:
    """Return the aggregate status path for a `digital_twin` target."""
    cfg = digital_twin_target(targets, target) or {}
    return Path(status_root) / f"cais_digital_twin_{digital_twin_slug(cfg)}.json"


def digital_twin_direction_path(
    targets: dict[str, dict[str, Any]],
    target: str,
    status_root: Path,
) -> Path:
    """Return the direction status path for a `digital_twin` target."""
    cfg = digital_twin_target(targets, target) or {}
    return Path(status_root) / f"cais_digital_twin_{digital_twin_slug(cfg)}_direction.json"


def digital_twin_sync_status_path(
    targets: dict[str, dict[str, Any]],
    target: str,
    robot: str,
    status_root: Path,
) -> Path:
    """Return the per-robot sync status path for a `digital_twin` target."""
    cfg = digital_twin_target(targets, target) or {}
    robot_key = str(robot or "").strip().lower()
    if robot_key:
        return Path(status_root) / f"cais_digital_twin_{digital_twin_slug(cfg)}_{robot_key}.json"
    return digital_twin_status_path(targets, target, status_root)


def digital_twin_dual_drag_markers_status_path(
    targets: dict[str, dict[str, Any]],
    target: str,
    status_root: Path,
) -> Path:
    """Return the dual drag markers status path for a `digital_twin` target."""
    cfg = digital_twin_target(targets, target) or {}
    return Path(status_root) / f"cais_digital_twin_{digital_twin_slug(cfg)}_dual_drag_markers.json"


def digital_twin_hardware_processes_for_robot(
    cfg: dict[str, Any],
    robot: str,
) -> dict[str, str]:
    """Return hardware process names for a robot in a `digital_twin` target entry."""
    hardware_processes = cfg.get("hardware_processes") or {}
    if not isinstance(hardware_processes, dict):
        return {}
    robot_key = str(robot or "").strip().lower()
    per_robot = hardware_processes.get(robot_key)
    if isinstance(per_robot, dict):
        return {
            str(key): str(value)
            for key, value in per_robot.items()
            if str(value or "").strip()
        }
    return {
        str(key): str(value)
        for key, value in hardware_processes.items()
        if isinstance(value, str) and str(value or "").strip()
    }


def digital_twin_sync_process_items(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """Return `(robot, process)` sync process entries for a `digital_twin` target."""
    sync_processes = cfg.get("sync_processes") or {}
    if isinstance(sync_processes, dict) and sync_processes:
        return [
            (str(robot).strip().lower(), str(process).strip())
            for robot, process in sync_processes.items()
            if str(robot).strip() and str(process).strip()
        ]
    sync_process = str(cfg.get("sync_process") or "").strip()
    robot = str(cfg.get("robot") or "").strip()
    return [(robot, sync_process)] if sync_process else []


def normalize_digital_twin_sim_mode(mode: object, aliases: dict[str, str]) -> str:
    """Return the configured sim mode value after applying existing aliases."""
    value = str(mode or "").strip()
    return aliases.get(value, value)


def digital_twin_allowed_sim_modes(
    cfg: dict[str, Any],
    sim_modes: tuple[str, ...],
    aliases: dict[str, str],
) -> tuple[str, ...]:
    """Return allowed sim modes for a `digital_twin` target entry."""
    modes = tuple(
        normalize_digital_twin_sim_mode(mode, aliases)
        for mode in (cfg.get("sim_modes") or ())
    )
    canonical_modes = tuple(dict.fromkeys(mode for mode in modes if mode))
    return canonical_modes or sim_modes


def digital_twin_has_multiple_hardware_domains(cfg: dict[str, Any]) -> bool:
    """Return whether a `digital_twin` target uses per-robot hardware domains."""
    return bool(cfg.get("multiple_hardware_domains", False))


def digital_twin_has_per_robot_hardware_processes(cfg: dict[str, Any]) -> bool:
    """Return whether hardware processes are grouped by robot key."""
    hardware_processes = cfg.get("hardware_processes") or {}
    return isinstance(hardware_processes, dict) and any(
        isinstance(value, dict) for value in hardware_processes.values()
    )


def digital_twin_hardware_domain_id(
    cfg: dict[str, Any],
    robot: str,
    domains: dict[str, int],
) -> int:
    """Return the hardware ROS_DOMAIN_ID for a robot in a `digital_twin` target."""
    if digital_twin_has_multiple_hardware_domains(cfg):
        robot_key = str(robot or "").strip().lower()
        return int(domains.get(f"hardware_{robot_key}", domains["hardware"]))
    return int(domains["hardware"])


def digital_twin_process_names(cfg: dict[str, Any]) -> list[str]:
    """Return tracked process names for a `digital_twin` target entry."""
    names: list[str] = []
    for key in ("gazebo_process", "gazebo_moveit_process", "paired_marker_process"):
        value = str(cfg.get(key) or "").strip()
        if value:
            names.append(value)
    for _robot, process in digital_twin_sync_process_items(cfg):
        if process:
            names.append(process)
    hardware_processes = cfg.get("hardware_processes") or {}
    if isinstance(hardware_processes, dict):
        for value in hardware_processes.values():
            if isinstance(value, dict):
                for nested_value in value.values():
                    name = str(nested_value or "").strip()
                    if name and name not in names:
                        names.append(name)
            else:
                name = str(value or "").strip()
                if name and name not in names:
                    names.append(name)
    return names


def robot_function_template_step(
    templates: dict[str, list[dict[str, str]]],
    function_name: str,
    step_name: str,
) -> dict[str, str] | None:
    """Return a taught-function template step by `step_name`."""
    step_key = str(step_name or "").strip()
    for step in templates.get(str(function_name or "").strip(), []):
        if str(step.get("step_name") or "") == step_key:
            return dict(step)
    return None


def robot_function_safe_name(name: object) -> str:
    """Return a filesystem-safe taught-function name segment."""
    value = str(name or "").strip()
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in value)
    return safe or "default"


def robot_function_safe_function_name(function_name: object) -> str:
    """Return a filesystem-safe function name, or empty for an empty input."""
    if not str(function_name or "").strip():
        return ""
    return robot_function_safe_name(function_name)


def robot_function_buffer_key(
    target: str,
    robot: str,
    function_name: str,
    name: str,
) -> str:
    """Return the in-memory taught-function buffer key."""
    return "::".join(
        [
            str(target or "").strip().lower(),
            str(robot or "").strip().lower(),
            str(function_name or "").strip(),
            str(name or "").strip(),
        ]
    )


def robot_function_display_path(path: Path, project_root: Path) -> str:
    """Return a project-relative taught-function path when possible."""
    try:
        return str(path.relative_to(project_root))
    except Exception:
        return str(path)


def robot_function_launch_mode_from_target(_target: str, _cfg: dict[str, Any]) -> str:
    """Return the launch mode used for taught-function storage."""
    return "digital_twin"


def robot_function_storage_source_for_launch(launch_mode: str) -> str:
    """Return taught-function storage source for a launch mode."""
    return "gazebo" if str(launch_mode or "").strip() == "gazebo" else "hardware"


def robot_function_path(
    taught_functions_dir: Path,
    robot: str,
    function_name: str,
    name: str,
    storage_source: str,
) -> Path:
    """Return the taught-function JSON path."""
    robot_key = str(robot or "").strip().lower()
    function_key = robot_function_safe_function_name(function_name)
    safe_name = robot_function_safe_name(name)
    source_key = str(storage_source or "hardware").strip()
    return (
        Path(taught_functions_dir)
        / robot_key
        / function_key
        / f"{safe_name}__{source_key}.json"
    )


def robot_function_step_waypoint(step: dict[str, Any]) -> dict[str, Any] | None:
    """Return the waypoint body stored on a taught-function step."""
    waypoint = dict(step.get("waypoint") or {})
    positions = waypoint.get("joint_positions") or waypoint.get("positions") or []
    if not positions:
        return None
    body: dict[str, Any] = {
        "positions": [float(v) for v in positions],
        "joint_names": list(waypoint.get("joint_names") or []),
        "gripper_joint": waypoint.get("gripper_joint"),
    }
    if waypoint.get("gripper_position") is not None:
        body["gripper"] = float(waypoint.get("gripper_position"))
    elif waypoint.get("gripper") is not None:
        body["gripper"] = float(waypoint.get("gripper"))
    return body


def single_robot_replay_cfg(cfg: dict[str, Any], robot: str) -> dict[str, Any]:
    """Return a single-robot replay cfg derived from a target cfg."""
    robot_key = str(robot or "").strip().lower()
    replay_cfg = dict(cfg)
    replay_cfg["robot"] = robot_key
    replay_cfg["hardware"] = (robot_key,)
    return replay_cfg


def digital_twin_recording_hash(recording: dict[str, Any]) -> str:
    """Return the stable hash used for prepared replay files."""
    body = json.dumps(recording, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def digital_twin_prepared_replay_path(
    cfg: dict[str, Any],
    recording: dict[str, Any],
    replay_target: str,
    source: str,
    status_root: Path,
) -> Path:
    """Return the prepared replay path for a `digital_twin` recording."""
    slug = digital_twin_slug(cfg)
    digest = digital_twin_recording_hash(recording)[:16]
    safe_source = "".join(
        c if (c.isalnum() or c in "-_") else "_"
        for c in str(source or "recording").strip()
    ) or "recording"
    safe_replay = "".join(
        c if (c.isalnum() or c in "-_") else "_"
        for c in str(replay_target or "twin").strip()
    ) or "twin"
    return Path(status_root) / (
        f"cais_digital_twin_{slug}_{safe_source}_{safe_replay}_{digest}_prepared.json"
    )


def digital_twin_is_dual_robots(cfg: dict[str, Any]) -> bool:
    """Return whether a `digital_twin` target entry is `dual robots`."""
    return str(cfg.get("robot") or "").strip().lower() == "dual robots"


def digital_twin_dual_robot_keys(cfg: dict[str, Any]) -> list[str]:
    """Return the configured dual robot keys in supported order."""
    return [
        str(robot).strip().lower()
        for robot in (cfg.get("hardware") or ())
        if str(robot).strip().lower() in {"xarm6", "ur5e"}
    ]


def digital_twin_recovery_metadata(target: str, *, recording_type: str) -> dict[str, Any]:
    """Return recovery metadata for a `digital_twin` recording."""
    return {
        "source": "digital_twin_teach",
        "target": target,
        "recording_type": recording_type,
        "dispatch": "manual",
        "created_at": time.time(),
    }


def waypoints_from_recording(
    cfg: dict[str, Any],
    recording: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return in-memory waypoints from a persisted `digital_twin` recording."""
    waypoints: list[dict[str, Any]] = []
    if digital_twin_is_dual_robots(cfg):
        for waypoint in list(recording.get("waypoints") or []):
            robots = {
                str(robot): dict(body)
                for robot, body in dict(waypoint.get("robots") or {}).items()
                if isinstance(body, dict)
            }
            if robots:
                waypoints.append({"robots": robots, "t": time.time()})
        return waypoints

    joint_names = list(recording.get("joint_names") or [])
    gripper_joint = recording.get("gripper_joint")
    for waypoint in list(recording.get("waypoints") or []):
        body = dict(waypoint or {})
        positions = list(body.get("positions") or [])
        if positions:
            waypoints.append(
                {
                    "positions": positions,
                    "gripper": body.get("gripper"),
                    "joint_names": joint_names,
                    "gripper_joint": gripper_joint,
                    "t": time.time(),
                }
            )
    return waypoints
