"""Execution runtime for registry-backed robot tasks."""

from __future__ import annotations

import json
from copy import deepcopy
from math import isfinite
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
    elif isinstance(primitive_result.get("data"), (dict, list)):
        payload = deepcopy(primitive_result.get("data"))
    elif isinstance(primitive_result.get("observation"), dict):
        payload = deepcopy(dict(primitive_result.get("observation") or {}))

    if step.op == "move_cartesian":
        payload = dict(payload or {})
        payload["absolute_position"] = {
            "x": float(params.get("x", 0.0) or 0.0),
            "y": float(params.get("y", 0.0) or 0.0),
            "z": float(params.get("z", 0.0) or 0.0),
        }
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
    return ""


def _recorded_cartesian_pose(
    *,
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
    joint_error = _recorded_joints_error(task=task, step=step, waypoint=waypoint)
    if joint_error:
        return {}, joint_error
    pose = waypoint.get("pose")
    if not isinstance(pose, dict):
        return {}, f"physical position pose is missing for {task.name}.{step.id}"
    if str(pose.get("frame_id") or "").strip() != "world":
        return {}, f"physical position frame must be world for {task.name}.{step.id}"
    if str(pose.get("child_frame_id") or "").strip() != "tool0":
        return {}, f"physical position child frame must be tool0 for {task.name}.{step.id}"

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


def _load_physical_cartesian_overrides(  # noqa: C901
    *,
    agent: Any,
    task: RobotTaskDefinition,
    args: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], Path | None, str]:
    required_steps = _required_physical_position_steps(task)
    if not required_steps:
        return {}, None, ""

    robot = _robot_name(agent)
    if robot != "ur5e":
        return (
            {},
            None,
            f"physical position recording is currently supported only for ur5e, not {robot or '<unknown>'}",
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
        return {}, path, f"physical position file not found: {path}"
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

    overrides: dict[str, dict[str, float]] = {}
    for step in required_steps:
        recorded_step = recorded_by_id.get(step.id)
        if recorded_step is None:
            return {}, path, f"physical position step not found: {task.name}.{step.id} in {path}"
        pose, error = _recorded_cartesian_pose(
            task=task,
            step=step,
            recorded_step=recorded_step,
        )
        if error:
            return {}, path, error
        overrides[step.id] = pose
    return overrides, path, ""


def _apply_physical_recording_state(
    *,
    task: RobotTaskDefinition,
    physical_overrides: dict[str, dict[str, float]],
    runtime_state: dict[str, Any],
) -> None:
    required_steps = _required_physical_position_steps(task)
    if not required_steps or not physical_overrides:
        return
    above_pose = physical_overrides.get(required_steps[0].id)
    target_pose = physical_overrides.get(required_steps[-1].id)
    if above_pose is not None:
        task_context = dict(runtime_state.get("_task_ctx") or {})
        task_context["travel_z"] = float(above_pose["z"])
        runtime_state["_task_ctx"] = task_context
    if target_pose is not None:
        runtime_state["_position"] = {
            "x": float(target_pose["x"]),
            "y": float(target_pose["y"]),
            "z": float(target_pose["z"]),
        }


async def _execute_task_step(
    *,
    agent: Any,
    task: RobotTaskDefinition,
    step: RobotTaskStep,
    args: dict[str, Any],
    runtime_state: dict[str, Any],
    step_outputs: dict[str, Any],
    physical_overrides: dict[str, dict[str, float]],
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

    params = _resolve_value(
        step.params, args=args, runtime_state=runtime_state, step_outputs=step_outputs
    )
    if not isinstance(params, dict):
        params = {}
    if step.physical_position_required and step.id in physical_overrides:
        params.update(physical_overrides[step.id])

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


async def execute_robot_task(  # noqa: C901
    agent: Any,
    task_name: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute one exact registry-backed task against the selected robot mode.

    Args:
        agent: RobotAgent-compatible runtime owner.
        task_name: Exact registered robot function name.
        **kwargs: Arguments declared by the selected task definition.

    Returns:
        Task completion, block, or failure payload.
    """
    task = robot_task_registry().get(str(task_name or "").strip())
    if task is None:
        return {"status": "failed", "content": f"unknown robot task '{task_name}'"}

    args = deepcopy(dict(kwargs or {}))
    runtime_state = _build_runtime_state(agent)
    step_outputs: dict[str, Any] = {}

    for guard in task.program.entry_guards:
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

    physical_overrides: dict[str, dict[str, float]] = {}
    physical_recording_path: Path | None = None
    execution_mode = str(getattr(agent, "execution_mode", "") or "").strip().lower()
    if execution_mode == "physical" and _required_physical_position_steps(task):
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
    for step in task.program.steps:
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
            return agent._task_failure(
                str(raw.get("message") or f"{step.op} failed"),
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
        if isinstance(payload, dict) and "absolute_position" in payload:
            runtime_state["_position"] = deepcopy(payload["absolute_position"])

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

    _apply_physical_recording_state(
        task=task,
        physical_overrides=physical_overrides,
        runtime_state=runtime_state,
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
