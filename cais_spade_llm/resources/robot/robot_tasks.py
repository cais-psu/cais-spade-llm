"""Unified robot task registry with declarative execution programs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import yaml


def _step_ref(fact_path: str, path: str) -> dict[str, str]:
    return {"$ref": f"event_facts.{fact_path}.{path}"}


def _resource_token(resource_jid: str) -> str:
    token = str(resource_jid or "<RESOURCE_JID>").strip()
    return token or "<RESOURCE_JID>"


def _copy_rows(rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    return deepcopy(list(rows))


def _task_docstring(frontmatter: dict[str, Any], description: str) -> str:
    payload = deepcopy(dict(frontmatter or {}))
    payload["description"] = str(description or "").strip()
    yaml_block = yaml.safe_dump(payload, sort_keys=False).strip()
    return f"---\n{yaml_block}\n---\n{str(description or '').strip()}".strip()

def _arg(name: str) -> dict[str, Any]:
    return {"$arg": str(name)}


def _state(field: str, *path: str) -> dict[str, Any]:
    return {"$state": str(field), "path": list(path)}


def _step_output(name: str, *path: str) -> dict[str, Any]:
    return {"$step": str(name), "path": list(path)}


def _format(template: str, **values: Any) -> dict[str, Any]:
    return {"$format": str(template), "values": dict(values)}


def _first(*values: Any) -> dict[str, Any]:
    return {"$first": list(values)}


def _sub(left: Any, right: Any) -> dict[str, Any]:
    return {"$sub": [left, right]}


@dataclass(frozen=True)
class RobotTaskGuard:
    predicate: str
    args: dict[str, Any] = field(default_factory=dict)
    message: Any = ""
    description: Any = ""


@dataclass(frozen=True)
class RobotTaskStep:
    id: str
    op: str
    executor: str = "primitive"
    params: dict[str, Any] = field(default_factory=dict)
    store_as: str | None = None
    exposed: bool = True
    public_params: dict[str, Any] | None = None
    note: str = ""
    when: tuple[RobotTaskGuard, ...] = ()
    dry_run_output: Any = None
    continue_on_failure: bool = False
    failure_observations: dict[str, Any] = field(default_factory=dict)

    def render_capability_row(self) -> dict[str, Any]:
        params = self.public_params if self.public_params is not None else {
            key: value for key, value in self.params.items() if not str(key).startswith("_")
        }
        row = {
            "primitive": self.op,
            "params": _render_decomposition_value(params),
        }
        if self.note:
            row["note"] = self.note
        return row


@dataclass(frozen=True)
class RobotTaskEffect:
    target: str
    action: str
    value: Any = None
    when: tuple[RobotTaskGuard, ...] = ()
    skip_empty_values: bool = False


@dataclass(frozen=True)
class RobotTaskProgram:
    """Task-level SSOT; bridge/task decompositions are rendered from `steps`."""

    entry_state: str
    success_state: str
    part_in_state: str = ""
    required_context_keys: tuple[str, ...] = ()
    context_mapping: dict[str, Any] = field(default_factory=dict)
    part_transition: dict[str, Any] | None = None
    entry_guards: tuple[RobotTaskGuard, ...] = ()
    steps: tuple[RobotTaskStep, ...] = ()
    effects: tuple[RobotTaskEffect, ...] = ()
    success_response: dict[str, Any] = field(default_factory=dict)
    dry_run_description: Any = ""
    dry_run_duration: float = 5.0
    failure_part: Any = ""
    notes: tuple[str, ...] = ()

    def modeled_transition(self) -> str:
        start = str(self.entry_state or "any").strip() or "any"
        end = str(self.success_state or start).strip() or start
        return f"{start} -> {end}"

    def task_preconditions(self, *, resource_jid: str = "") -> list[str]:
        resolved: list[str] = []
        for guard in self.entry_guards:
            description = _render_decomposition_value(
                guard.description,
                context={"resource_jid": _resource_token(resource_jid)},
            )
            token = str(description or "").strip()
            if token:
                resolved.append(token)
        return resolved

    def render_bridge_steps(self) -> list[dict[str, Any]]:
        return [step.render_capability_row() for step in self.steps if step.exposed]


@dataclass(frozen=True)
class RobotTaskArgument:
    name: str
    type: str
    description: str = ""
    required: bool = False
    default: Any = None

    def param_schema(self) -> dict[str, Any]:
        payload = {"type": str(self.type or "string").strip() or "string"}
        if self.description:
            payload["description"] = str(self.description).strip()
        if self.default is not None:
            payload["default"] = deepcopy(self.default)
        return payload


@dataclass(frozen=True)
class RobotTaskExposure:
    predicates: tuple[RobotTaskGuard, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RobotTaskDefinition:
    name: str
    description: str
    arguments: tuple[RobotTaskArgument, ...]
    program: RobotTaskProgram
    exposure: RobotTaskExposure = field(default_factory=RobotTaskExposure)
    source: str = "robot_tasks.py"
    examples: tuple[str, ...] = ()
    handler: Callable[..., Awaitable[dict[str, Any]]] | None = None

    def argument_properties(self) -> dict[str, dict[str, Any]]:
        return {
            argument.name: argument.param_schema()
            for argument in self.arguments
        }

    def required_argument_names(self) -> list[str]:
        return [argument.name for argument in self.arguments if argument.required]

    def tool_frontmatter(self) -> dict[str, Any]:
        payload = {
            "process": "assembly",
            "resource_type": "robot",
        }
        payload["in_state"] = str(self.program.entry_state or "any").strip() or "any"
        payload["out_state"] = str(self.program.success_state or payload["in_state"]).strip()
        if self.program.part_in_state:
            payload["part_in_state"] = str(self.program.part_in_state)
        if self.program.required_context_keys:
            payload["required_context_keys"] = list(self.program.required_context_keys)
        if self.program.context_mapping:
            payload["context_mapping"] = deepcopy(self.program.context_mapping)
        if self.program.part_transition is not None:
            payload["part_transition"] = deepcopy(self.program.part_transition)
        payload["params"] = self.argument_properties()
        payload["description"] = str(self.description or "").strip()
        return payload

    def rendered_docstring(self) -> str:
        return _task_docstring(self.tool_frontmatter(), self.description)

    def tool_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": str(self.description or "").strip(),
            "parameters": {
                "type": "object",
                "properties": self.argument_properties(),
                "required": self.required_argument_names(),
            },
        }

    def capability_decomposition(self, *, resource_jid: str = "") -> dict[str, Any]:
        # `program.steps` is the authored task SSOT. `bridge_visible_steps` remains
        # a derived rendering for primitive generation and related bridge tooling.
        return {
            "function_name": self.name,
            "source": self.source,
            "modeled_transition": self.program.modeled_transition(),
            "task_preconditions": self.program.task_preconditions(resource_jid=resource_jid),
            "bridge_visible_steps": self.program.render_bridge_steps(),
            "execution_notes": list(self.program.notes),
        }

    def is_enabled(self, capability_context: dict[str, Any]) -> bool:
        return all(
            _evaluate_exposure_guard(predicate, capability_context=capability_context)
            for predicate in self.exposure.predicates
        )


def _is_ref_dict(value: Any, key: str) -> bool:
    return isinstance(value, dict) and key in value and len(value) in {1, 2}


def _path_get(value: Any, path: list[str]) -> Any:
    current = value
    for token in path:
        if isinstance(current, dict):
            current = current.get(token)
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(token)]
            except (TypeError, ValueError, IndexError):
                return None
        else:
            return None
    return current


def _placeholder_for_arg(name: str) -> str:
    token = str(name or "").strip().upper()
    if token == "PART_NAME":
        return "<PART>"
    return f"<{token}>"


def _resolve_value(
    value: Any,
    *,
    args: dict[str, Any],
    runtime_state: dict[str, Any],
    step_outputs: dict[str, Any],
) -> Any:
    if _is_ref_dict(value, "$arg"):
        return deepcopy(args.get(str(value["$arg"])))
    if isinstance(value, dict) and "$state" in value:
        field = str(value.get("$state") or "")
        path = list(value.get("path") or [])
        current = runtime_state.get(field)
        return deepcopy(_path_get(current, path) if path else current)
    if isinstance(value, dict) and "$step" in value:
        step_name = str(value.get("$step") or "")
        path = list(value.get("path") or [])
        current = step_outputs.get(step_name)
        return deepcopy(_path_get(current, path) if path else current)
    if isinstance(value, dict) and "$format" in value:
        resolved_values = {
            key: _resolve_value(item, args=args, runtime_state=runtime_state, step_outputs=step_outputs)
            for key, item in dict(value.get("values") or {}).items()
        }
        return str(value.get("$format") or "").format_map(
            {key: "" if item is None else item for key, item in resolved_values.items()}
        )
    if isinstance(value, dict) and "$first" in value:
        for item in list(value.get("$first") or []):
            resolved = _resolve_value(item, args=args, runtime_state=runtime_state, step_outputs=step_outputs)
            if resolved not in (None, ""):
                return deepcopy(resolved)
        return None
    if isinstance(value, dict) and "$sub" in value:
        left_raw, right_raw = list(value.get("$sub") or [0, 0])[:2]
        left = _resolve_value(left_raw, args=args, runtime_state=runtime_state, step_outputs=step_outputs)
        right = _resolve_value(right_raw, args=args, runtime_state=runtime_state, step_outputs=step_outputs)
        try:
            return float(left or 0.0) - float(right or 0.0)
        except (TypeError, ValueError):
            return 0.0
    if isinstance(value, dict):
        return {
            key: _resolve_value(item, args=args, runtime_state=runtime_state, step_outputs=step_outputs)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_value(item, args=args, runtime_state=runtime_state, step_outputs=step_outputs)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _resolve_value(item, args=args, runtime_state=runtime_state, step_outputs=step_outputs)
            for item in value
        )
    return deepcopy(value)


def _render_decomposition_value(value: Any, *, context: dict[str, Any] | None = None) -> Any:
    context = dict(context or {})
    if _is_ref_dict(value, "$arg"):
        return _placeholder_for_arg(str(value["$arg"]))
    if isinstance(value, dict) and "$step" in value:
        step_name = str(value.get("$step") or "")
        path = ".".join(str(token) for token in list(value.get("path") or []))
        if step_name in {"pick_targets", "place_targets"} and path:
            return _step_ref(f"{step_name}.<PART>", path)
        return {"$ref": f"{step_name}.{path}".rstrip(".")}
    if isinstance(value, dict) and "$format" in value:
        rendered_values = {
            key: _render_decomposition_value(item, context=context)
            for key, item in dict(value.get("values") or {}).items()
        }
        merged = dict(context)
        merged.update(rendered_values)
        return str(value.get("$format") or "").format_map(merged)
    if isinstance(value, dict) and "$first" in value:
        rendered = [
            _render_decomposition_value(item, context=context)
            for item in list(value.get("$first") or [])
        ]
        for item in rendered:
            if item not in (None, ""):
                return item
        return None
    if isinstance(value, dict) and "$sub" in value:
        left, right = list(value.get("$sub") or [0, 0])[:2]
        return {
            "$sub": [
                _render_decomposition_value(left, context=context),
                _render_decomposition_value(right, context=context),
            ]
        }
    if isinstance(value, dict):
        return {key: _render_decomposition_value(item, context=context) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_decomposition_value(item, context=context) for item in value]
    if isinstance(value, tuple):
        return [_render_decomposition_value(item, context=context) for item in value]
    return deepcopy(value)


def _evaluate_named_pose_available(
    pose_name: str,
    *,
    capability_context: dict[str, Any] | None = None,
    agent: Any | None = None,
) -> bool:
    token = str(pose_name or "").strip()
    if not token:
        return False
    named_positions = {}
    if capability_context is not None:
        named_positions = capability_context.get("named_positions") or {}
    elif agent is not None:
        named_positions = getattr(agent, "named_positions", {}) or {}
    if isinstance(named_positions, dict):
        return token in {str(name).strip() for name in named_positions.keys()}
    if isinstance(named_positions, (list, tuple, set)):
        return token in {str(name).strip() for name in named_positions}
    return False


def _evaluate_guard_predicate(
    predicate: str,
    *,
    resolved_args: dict[str, Any],
    agent: Any | None = None,
    runtime_state: dict[str, Any] | None = None,
    capability_context: dict[str, Any] | None = None,
) -> bool:
    if predicate in {"", "always"}:
        return True
    if predicate == "held_part_empty":
        return not bool(dict(runtime_state or {}).get("_held_part"))
    if predicate == "held_part_exists":
        return bool(dict(runtime_state or {}).get("_held_part"))
    if predicate == "task_ctx_key_truthy":
        key = str(dict(resolved_args or {}).get("key") or "").strip()
        return bool(dict(dict(runtime_state or {}).get("_task_ctx") or {}).get(key))
    if predicate == "execution_mode":
        expected = str(dict(resolved_args or {}).get("mode") or "").strip().lower()
        actual = str(getattr(agent, "execution_mode", "") or "").strip().lower() if agent else ""
        return bool(expected) and actual == expected
    if predicate == "named_pose_available":
        pose_name = str(dict(resolved_args or {}).get("pose_name") or "").strip()
        return _evaluate_named_pose_available(
            pose_name,
            capability_context=capability_context,
            agent=agent,
        )
    return True


def _evaluate_guard(
    guard: RobotTaskGuard,
    *,
    agent: Any,
    args: dict[str, Any],
    runtime_state: dict[str, Any],
    step_outputs: dict[str, Any],
) -> bool:
    predicate = str(guard.predicate or "").strip()
    resolved_args = _resolve_value(
        guard.args,
        args=args,
        runtime_state=runtime_state,
        step_outputs=step_outputs,
    )
    return _evaluate_guard_predicate(
        predicate,
        resolved_args=dict(resolved_args or {}),
        agent=agent,
        runtime_state=runtime_state,
    )


def _evaluate_exposure_guard(
    guard: RobotTaskGuard,
    *,
    capability_context: dict[str, Any],
) -> bool:
    return _evaluate_guard_predicate(
        str(guard.predicate or "").strip(),
        resolved_args=dict(guard.args or {}),
        capability_context=capability_context,
    )


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


def _normalize_place_targets(raw: dict[str, Any], *, runtime_state: dict[str, Any]) -> dict[str, Any]:
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
        "_bridge_pose_ref": deepcopy(getattr(agent, "_bridge_pose_ref", None)),
        "_task_ctx": deepcopy(getattr(agent, "_task_ctx", {})),
    }


def _commit_runtime_state(agent: Any, runtime_state: dict[str, Any]) -> None:
    for field in (
        "_held_part",
        "_current_state",
        "_position",
        "_gripper_state",
        "_bridge_pose_ref",
        "_task_ctx",
    ):
        setattr(agent, field, deepcopy(runtime_state.get(field)))


def _primitive_payload_from_result(
    *,
    step: RobotTaskStep,
    params: dict[str, Any],
    primitive_result: dict[str, Any],
    runtime_state: dict[str, Any],
) -> dict[str, Any] | None:
    payload: dict[str, Any] | None = None
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
    elif isinstance(primitive_result.get("data"), dict):
        payload = deepcopy(dict(primitive_result.get("data") or {}))
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


_TAUGHT_FUNCTION_STEP_MAP: dict[str, dict[str, str]] = {
    "pick_approach": {
        "move_above_part": "approach_pose",
        "descend": "pick_pose",
    },
    "pick_grasp": {
        "grasp_part": "grasp_part",
        "lift": "lift_pose",
    },
    "place_approach": {
        "move_above_destination": "approach_pose",
        "descend": "place_pose",
    },
    "place_insert": {
        "release_part": "release_part",
        "lift": "retreat_pose",
    },
    "move_home": {
        "move_home": "home",
    },
}


def _uses_taught_function_replay(agent: Any) -> bool:
    execution_mode = str(getattr(agent, "execution_mode", "") or "").strip().lower()
    return execution_mode not in {"", "dry_run", "simulation"}


def _taught_function_file_name(task_name: str, args: dict[str, Any]) -> str:
    explicit = str(args.get("function_file_name") or "").strip()
    if explicit:
        return explicit
    if task_name in {"pick_approach", "pick_grasp"}:
        return str(args.get("origin_resource_location") or "").strip() or "default"
    if task_name in {"place_approach", "place_insert"}:
        return str(args.get("destination_location") or "").strip() or "default"
    return "default"


async def _execute_task_step(
    *,
    agent: Any,
    task: RobotTaskDefinition,
    step: RobotTaskStep,
    args: dict[str, Any],
    runtime_state: dict[str, Any],
    step_outputs: dict[str, Any],
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

    params = _resolve_value(step.params, args=args, runtime_state=runtime_state, step_outputs=step_outputs)
    if not isinstance(params, dict):
        params = {}

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

    taught_step_name = _TAUGHT_FUNCTION_STEP_MAP.get(task.name, {}).get(step.id)
    if taught_step_name and _uses_taught_function_replay(agent):
        executor = getattr(agent, "_execute_taught_function_step", None)
        if not callable(executor):
            result = {
                "success": False,
                "message": "resource agent missing taught function replay support",
            }
        else:
            result = await executor(
                function_name=task.name,
                taught_function_name=_taught_function_file_name(task.name, args),
                step_name=taught_step_name,
            )
    else:
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
    value = _resolve_value(effect.value, args=args, runtime_state=runtime_state, step_outputs=step_outputs)

    if effect.target == "task_ctx":
        if effect.action == "clear":
            runtime_state["_task_ctx"] = {}
            return
        if effect.action == "merge":
            current = dict(runtime_state.get("_task_ctx") or {})
            incoming = dict(value or {})
            if effect.skip_empty_values:
                incoming = {
                    key: item
                    for key, item in incoming.items()
                    if item not in (None, "")
                }
            current.update(deepcopy(incoming))
            runtime_state["_task_ctx"] = current
            return
        runtime_state["_task_ctx"] = deepcopy(dict(value or {}))
        return

    if effect.action == "clear":
        runtime_state[f"_{effect.target}"] = None if effect.target != "task_ctx" else {}
        return
    runtime_state[f"_{effect.target}"] = deepcopy(value)


async def execute_robot_task(
    agent: Any,
    task_name: str,
    **kwargs: Any,
) -> dict[str, Any]:
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
            if (step.continue_on_failure and not simulation_snap_part_to_slot) or simulation_lift_after_release:
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


def _make_robot_task_handler(function_name: str) -> Callable[..., Awaitable[dict[str, Any]]]:
    async def _handler(self: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._execute_registered_robot_task(function_name, **kwargs)

    _handler.__name__ = function_name
    _handler.__qualname__ = f"RobotAgent.{function_name}"
    return _handler


def _register_robot_task(task: RobotTaskDefinition) -> RobotTaskDefinition:
    handler = _make_robot_task_handler(task.name)
    object.__setattr__(task, "handler", handler)
    handler.__doc__ = task.rendered_docstring()
    handler.__tool_spec__ = task
    return task


_TASK_ORDER = (
    "pick_approach",
    "pick_grasp",
    "place_approach",
    "move_home",
    "place_insert",
)


_ROBOT_TASKS: tuple[RobotTaskDefinition, ...] = (
    _register_robot_task(
        RobotTaskDefinition(
            name="pick_approach",
            description="Approach the part's origin location with empty gripper.",
            arguments=(
                RobotTaskArgument(
                    name="origin_resource_location",
                    type="string",
                    description="Target origin location to approach for picking.",
                    required=True,
                ),
                RobotTaskArgument(
                    name="part_name",
                    type="string",
                    description="Name of the part intended to be picked (for tracking).",
                    required=True,
                ),
                RobotTaskArgument(
                    name="speed",
                    type="number",
                    description="Optional motion speed.",
                ),
                RobotTaskArgument(
                    name="product_geometry",
                    type="object",
                    description="Product geometry payload containing part poses in world frame.",
                ),
                RobotTaskArgument(
                    name="product_jid",
                    type="string",
                    description="JID of the ProductAgent that owns this task.",
                ),
                RobotTaskArgument(name="task_id", type="string"),
            ),
            program=RobotTaskProgram(
                entry_state="idle",
                success_state="at_pick",
                required_context_keys=("origin",),
                context_mapping={
                    "location_param": "origin_resource_location",
                    "location_type": "part_location",
                },
                entry_guards=(
                    RobotTaskGuard(
                        predicate="held_part_empty",
                        message="Cannot move-to-pick while already holding a part.",
                        description="held_part is empty",
                    ),
                ),
                steps=(
                    RobotTaskStep(
                        id="detect_parts",
                        op="detect_parts",
                        executor="primitive",
                        exposed=True,
                        params={"part_name": _arg("part_name")},
                        note="May be skipped only when equivalent observed pose was retrieved.",
                    ),
                    RobotTaskStep(
                        id="compute_pick_targets",
                        op="compute_pick_targets",
                        executor="primitive",
                        exposed=True,
                        store_as="pick_targets",
                        params={
                            "part_name": _arg("part_name"),
                            "product_geometry": _arg("product_geometry"),
                        },
                        dry_run_output={
                            "part_name": _arg("part_name"),
                            "model_name": "",
                            "tx": 0.0,
                            "ty": 0.0,
                            "tz": 0.0,
                            "pick_z": 0.0,
                            "travel_z": 1.2,
                            "part_height": 0.08,
                            "tcp_offset_z": -0.17,
                            "pick_tcp_z": 0.0,
                            "gripper_close_position": None,
                            "start_x": 0.0,
                            "start_y": 0.0,
                            "start_z": 0.0,
                        },
                        failure_observations={"part_name": _arg("part_name")},
                    ),
                    RobotTaskStep(
                        id="open_gripper",
                        op="open_gripper",
                        executor="primitive",
                        exposed=False,
                    ),
                    RobotTaskStep(
                        id="move_above_part",
                        op="move_cartesian",
                        executor="primitive",
                        exposed=True,
                        params={
                            "x": _step_output("pick_targets", "approach_pose", "x"),
                            "y": _step_output("pick_targets", "approach_pose", "y"),
                            "z": _step_output("pick_targets", "approach_pose", "z"),
                            "speed": _arg("speed"),
                        },
                        public_params={
                            "x": _step_ref("pick_targets.<PART>", "approach_pose.x"),
                            "y": _step_ref("pick_targets.<PART>", "approach_pose.y"),
                            "z": _step_ref("pick_targets.<PART>", "approach_pose.z"),
                        },
                        failure_observations={"part_name": _arg("part_name")},
                    ),
                    RobotTaskStep(
                        id="descend",
                        op="move_cartesian",
                        executor="primitive",
                        exposed=True,
                        params={
                            "x": _step_output("pick_targets", "target_pose", "x"),
                            "y": _step_output("pick_targets", "target_pose", "y"),
                            "z": _step_output("pick_targets", "target_pose", "z"),
                        },
                        public_params={
                            "x": _step_ref("pick_targets.<PART>", "target_pose.x"),
                            "y": _step_ref("pick_targets.<PART>", "target_pose.y"),
                            "z": _step_ref("pick_targets.<PART>", "target_pose.z"),
                        },
                        failure_observations={"part_name": _arg("part_name")},
                    ),
                ),
                effects=(
                    RobotTaskEffect(
                        target="task_ctx",
                        action="set",
                        value={
                            "part_name": _step_output("pick_targets", "part_name"),
                            "model_name": _step_output("pick_targets", "model_name"),
                            "tx": _step_output("pick_targets", "tx"),
                            "ty": _step_output("pick_targets", "ty"),
                            "tz": _step_output("pick_targets", "tz"),
                            "pick_z": _step_output("pick_targets", "pick_z"),
                            "travel_z": _step_output("pick_targets", "travel_z"),
                            "part_height": _step_output("pick_targets", "part_height"),
                            "tcp_offset_z": _step_output("pick_targets", "tcp_offset_z"),
                            "pick_tcp_z": _step_output("pick_targets", "pick_tcp_z"),
                            "origin_resource_location": _arg("origin_resource_location"),
                            "origin_pose": _step_output("pick_targets", "origin_pose"),
                            "start_x": _step_output("pick_targets", "start_x"),
                            "start_y": _step_output("pick_targets", "start_y"),
                            "start_z": _step_output("pick_targets", "start_z"),
                        },
                    ),
                    RobotTaskEffect(target="current_state", action="set", value="at_pick"),
                    RobotTaskEffect(
                        target="position",
                        action="set",
                        value={
                            "x": _step_output("pick_targets", "tx"),
                            "y": _step_output("pick_targets", "ty"),
                            "z": _step_output("pick_targets", "pick_z"),
                        },
                    ),
                    RobotTaskEffect(target="bridge_pose_ref", action="set", value=None),
                ),
                success_response={
                    "status": "completed",
                    "content": _format(
                        "Arrived at {origin_resource_location} ready to pick {part_name}.",
                        origin_resource_location=_arg("origin_resource_location"),
                        part_name=_arg("part_name"),
                    ),
                },
                dry_run_description=_format(
                    "Travel empty to pick location {origin_resource_location} for {part_name} (speed={speed})",
                    origin_resource_location=_arg("origin_resource_location"),
                    part_name=_arg("part_name"),
                    speed=_first(_arg("speed"), "default"),
                ),
                dry_run_duration=5.0,
                failure_part=_arg("part_name"),
                notes=(
                    "RobotAgent.pick_approach computes pick geometry, opens the gripper, moves above the part, then descends to the pick pose.",
                    "open_gripper and direct controller pose helpers are hidden from synthesis; use compute_pick_targets plus move_cartesian approach/target poses.",
                ),
            ),
        )
    ),
    _register_robot_task(
        RobotTaskDefinition(
            name="pick_grasp",
            description="Pick a ready part from an origin location.",
            arguments=(
                RobotTaskArgument(
                    name="part_name",
                    type="string",
                    description="Name of the part to pick.",
                    required=True,
                ),
                RobotTaskArgument(
                    name="origin_resource_location",
                    type="string",
                    description="Origin location of the part (printer or fixture).",
                    required=True,
                ),
                RobotTaskArgument(
                    name="gripper",
                    type="string",
                    description="Optional gripper configuration.",
                ),
                RobotTaskArgument(
                    name="product_geometry",
                    type="object",
                    description="Product geometry payload containing part poses in world frame.",
                ),
                RobotTaskArgument(
                    name="product_jid",
                    type="string",
                    description="JID of the ProductAgent that owns this task.",
                ),
                RobotTaskArgument(name="task_id", type="string"),
            ),
            program=RobotTaskProgram(
                entry_state="at_pick",
                success_state="picked",
                part_in_state="ready",
                required_context_keys=("origin",),
                context_mapping={
                    "location_param": "origin_resource_location",
                    "location_type": "current_location",
                },
                part_transition={
                    "completed": {
                        "state": "in_gripper",
                        "location_template": "{resource_jid}_gripper",
                    }
                },
                entry_guards=(
                    RobotTaskGuard(
                        predicate="held_part_empty",
                        message=_format(
                            "Already holding {held_part}; assemble it before picking a new part.",
                            held_part=_state("_held_part"),
                        ),
                        description="resource is already at the pick pose from pick_approach",
                    ),
                    RobotTaskGuard(
                        predicate="always",
                        description="held_part is empty",
                    ),
                    RobotTaskGuard(
                        predicate="always",
                        description="pick target was grounded for the active part",
                    ),
                ),
                steps=(
                    RobotTaskStep(
                        id="grasp_part",
                        op="grasp_part",
                        executor="primitive",
                        exposed=True,
                        params={
                            "model_name": _state("_task_ctx", "model_name"),
                            "part_name": _arg("part_name"),
                            "position": _state("_task_ctx", "gripper_close_position"),
                        },
                        public_params={
                            "model_name": "<MODEL_NAME_FROM_PART_TARGET>",
                            "part_name": _arg("part_name"),
                        },
                        failure_observations={
                            "part_name": _arg("part_name"),
                            "model_name": _state("_task_ctx", "model_name"),
                        },
                    ),
                    RobotTaskStep(
                        id="lift",
                        op="move_relative",
                        executor="primitive",
                        exposed=True,
                        params={
                            "dx": 0.0,
                            "dy": 0.0,
                            "dz": _sub(_state("_task_ctx", "travel_z"), _state("_position", "z")),
                            "speed": 0.45,
                        },
                        public_params={
                            "dx": 0.0,
                            "dy": 0.0,
                            "dz": 0.05,
                            "speed": 0.45,
                        },
                        note="Positive dz lift/retreat after grasp.",
                        failure_observations={"part_name": _arg("part_name")},
                    ),
                ),
                effects=(
                    RobotTaskEffect(target="held_part", action="set", value=_arg("part_name")),
                    RobotTaskEffect(target="current_state", action="set", value="picked"),
                    RobotTaskEffect(target="gripper_state", action="set", value="closed"),
                ),
                success_response={
                    "status": "completed",
                    "content": _format("Picked {part_name}.", part_name=_arg("part_name")),
                    "observations": {
                        "part_name": _arg("part_name"),
                        "origin_pose": {
                            "x": _state("_task_ctx", "tx"),
                            "y": _state("_task_ctx", "ty"),
                            "z": _state("_task_ctx", "tz"),
                        },
                    },
                },
                dry_run_description=_format(
                    "Picking {part_name} from {origin_resource_location} (gripper={gripper})",
                    part_name=_arg("part_name"),
                    origin_resource_location=_arg("origin_resource_location"),
                    gripper=_first(_arg("gripper"), "default"),
                ),
                dry_run_duration=5.0,
                failure_part=_arg("part_name"),
                notes=(
                    "RobotAgent.pick_grasp closes the gripper, attaches the part in simulation, then lifts to travel height.",
                    "close_gripper and attach_part are hidden from synthesis; use grasp_part as the visible composite.",
                ),
            ),
        )
    ),
    _register_robot_task(
        RobotTaskDefinition(
            name="place_approach",
            description="Move the loaded part to its destination location.",
            arguments=(
                RobotTaskArgument(
                    name="destination_location",
                    type="string",
                    description="Destination location to carry the loaded part.",
                    required=True,
                ),
                RobotTaskArgument(
                    name="part_name",
                    type="string",
                    description="Name of the part being moved.",
                    required=True,
                ),
                RobotTaskArgument(
                    name="speed",
                    type="number",
                    description="Optional motion speed while loaded.",
                ),
                RobotTaskArgument(
                    name="product_geometry",
                    type="object",
                    description="Product geometry payload containing target placement poses.",
                ),
                RobotTaskArgument(
                    name="product_jid",
                    type="string",
                    description="JID of the ProductAgent that owns this task.",
                ),
                RobotTaskArgument(name="task_id", type="string"),
            ),
            program=RobotTaskProgram(
                entry_state="picked",
                success_state="positioned",
                part_in_state="in_gripper",
                required_context_keys=("destination",),
                context_mapping={
                    "location_param": "destination_location",
                    "location_type": "reachable_location",
                },
                part_transition={
                    "completed": {
                        "state": "in_transit",
                        "location_template": "{resource_jid}_gripper",
                    }
                },
                entry_guards=(
                    RobotTaskGuard(
                        predicate="held_part_exists",
                        message="Cannot move-loaded without holding a part.",
                        description="held_part exists",
                    ),
                    RobotTaskGuard(
                        predicate="always",
                        description="destination geometry is available from destination_location, product_geometry, or served context",
                    ),
                ),
                steps=(
                    RobotTaskStep(
                        id="compute_place_targets",
                        op="compute_place_targets",
                        executor="primitive",
                        exposed=True,
                        store_as="place_targets",
                        params={
                            "pick_ctx": _state("_task_ctx"),
                            "product_geometry": _arg("product_geometry"),
                            "part_name": _arg("part_name"),
                            "z_adjustment_m": 0.0,
                            "destination_location": _arg("destination_location"),
                        },
                        public_params={
                            "part_name": _arg("part_name"),
                            "destination_location": _arg("destination_location"),
                        },
                        dry_run_output={
                            "slot_x": 0.0,
                            "slot_y": 0.0,
                            "board_top_z": 1.025,
                            "place_z": 1.1,
                            "part_height": 0.08,
                            "destination_location": _arg("destination_location"),
                        },
                        failure_observations={"part_name": _state("_held_part")},
                    ),
                    RobotTaskStep(
                        id="move_above_destination",
                        op="move_cartesian",
                        executor="primitive",
                        exposed=True,
                        params={
                            "x": _step_output("place_targets", "approach_pose", "x"),
                            "y": _step_output("place_targets", "approach_pose", "y"),
                            "z": _step_output("place_targets", "approach_pose", "z"),
                            "speed": _arg("speed"),
                        },
                        public_params={
                            "x": _step_ref("place_targets.<PART>", "approach_pose.x"),
                            "y": _step_ref("place_targets.<PART>", "approach_pose.y"),
                            "z": _step_ref("place_targets.<PART>", "approach_pose.z"),
                        },
                        failure_observations={"part_name": _state("_held_part")},
                    ),
                    RobotTaskStep(
                        id="descend",
                        op="move_cartesian",
                        executor="primitive",
                        exposed=True,
                        params={
                            "x": _step_output("place_targets", "target_pose", "x"),
                            "y": _step_output("place_targets", "target_pose", "y"),
                            "z": _step_output("place_targets", "target_pose", "z"),
                        },
                        public_params={
                            "x": _step_ref("place_targets.<PART>", "target_pose.x"),
                            "y": _step_ref("place_targets.<PART>", "target_pose.y"),
                            "z": _step_ref("place_targets.<PART>", "target_pose.z"),
                        },
                        failure_observations={"part_name": _state("_held_part")},
                    ),
                ),
                effects=(
                    RobotTaskEffect(
                        target="task_ctx",
                        action="merge",
                        value={
                            "slot_x": _step_output("place_targets", "slot_x"),
                            "slot_y": _step_output("place_targets", "slot_y"),
                            "board_top_z": _step_output("place_targets", "board_top_z"),
                            "place_z": _step_output("place_targets", "place_z"),
                            "place_part_origin_z": _step_output("place_targets", "place_part_origin_z"),
                            "part_height": _step_output("place_targets", "part_height"),
                            "destination_location": _arg("destination_location"),
                            "model_name": _step_output("place_targets", "model_name"),
                        },
                        skip_empty_values=True,
                    ),
                    RobotTaskEffect(target="current_state", action="set", value="positioned"),
                    RobotTaskEffect(
                        target="position",
                        action="set",
                        value={
                            "x": _step_output("place_targets", "slot_x"),
                            "y": _step_output("place_targets", "slot_y"),
                            "z": _step_output("place_targets", "place_z"),
                        },
                    ),
                    RobotTaskEffect(target="bridge_pose_ref", action="set", value=None),
                ),
                success_response={
                    "status": "completed",
                    "content": _format(
                        "Reached {destination_location} with {held_part}.",
                        destination_location=_arg("destination_location"),
                        held_part=_state("_held_part"),
                    ),
                },
                dry_run_description=_format(
                    "Move loaded part {held_part} to {destination_location} (speed={speed})",
                    held_part=_state("_held_part"),
                    destination_location=_arg("destination_location"),
                    speed=_first(_arg("speed"), "default"),
                ),
                dry_run_duration=5.0,
                failure_part=_first(_arg("part_name"), _state("_held_part")),
                notes=(
                    "RobotAgent.place_approach computes destination geometry, moves above the destination, then descends to the place pose.",
                    "Direct controller pose helpers are hidden from synthesis; use compute_place_targets plus move_cartesian approach/target poses.",
                ),
            ),
        )
    ),
    _register_robot_task(
        RobotTaskDefinition(
            name="place_insert",
            description="Assemble the currently held part at its final destination.",
            arguments=(
                RobotTaskArgument(
                    name="destination_location",
                    type="string",
                    description="Final assembly location for the part.",
                    required=True,
                ),
                RobotTaskArgument(
                    name="part_name",
                    type="string",
                    description="Name of the part being assembled.",
                    required=True,
                ),
                RobotTaskArgument(
                    name="orientation",
                    type="string",
                    description="Optional placement orientation.",
                ),
                RobotTaskArgument(
                    name="product_geometry",
                    type="object",
                    description="Product geometry payload containing insertion target pose.",
                ),
                RobotTaskArgument(
                    name="product_jid",
                    type="string",
                    description="JID of the ProductAgent that owns this task.",
                ),
                RobotTaskArgument(name="task_id", type="string"),
            ),
            program=RobotTaskProgram(
                entry_state="positioned",
                success_state="placed",
                part_in_state="in_transit",
                required_context_keys=("destination",),
                context_mapping={
                    "location_param": "destination_location",
                    "location_type": "current_location",
                },
                part_transition={
                    "completed": {
                        "state": "assembled",
                        "verify_camera": True,
                        "location_param": "destination_location",
                    }
                },
                entry_guards=(
                    RobotTaskGuard(
                        predicate="held_part_exists",
                        message="No part currently held; run pick_grasp first.",
                        description="held_part exists",
                    ),
                    RobotTaskGuard(
                        predicate="always",
                        description="place_approach already positioned the robot at the target pose",
                    ),
                ),
                steps=(
                    RobotTaskStep(
                        id="release_part",
                        op="release_part",
                        executor="primitive",
                        exposed=True,
                        params={
                            "model_name": _state("_task_ctx", "model_name"),
                            "part_name": _state("_held_part"),
                        },
                        public_params={
                            "model_name": "<MODEL_NAME_FROM_PART_TARGET>",
                            "part_name": _arg("part_name"),
                        },
                        failure_observations={
                            "part_name": _state("_held_part"),
                            "destination_location": _arg("destination_location"),
                        },
                    ),
                    RobotTaskStep(
                        id="snap_part_to_slot",
                        op="snap_part_to_slot",
                        executor="primitive",
                        exposed=False,
                        params={
                            "model_name": _state("_task_ctx", "model_name"),
                            "slot_x": _state("_task_ctx", "slot_x"),
                            "slot_y": _state("_task_ctx", "slot_y"),
                            "part_height": _state("_task_ctx", "part_height"),
                            "board_top_z": _state("_task_ctx", "board_top_z"),
                            "part_origin_z": _state("_task_ctx", "place_part_origin_z"),
                            "destination_location": _arg("destination_location"),
                        },
                        when=(
                            RobotTaskGuard(
                                predicate="execution_mode",
                                args={"mode": "simulation"},
                            ),
                            RobotTaskGuard(
                                predicate="task_ctx_key_truthy",
                                args={"key": "model_name"},
                            ),
                        ),
                        continue_on_failure=True,
                        failure_observations={
                            "part_name": _state("_held_part"),
                            "destination_location": _arg("destination_location"),
                        },
                    ),
                    RobotTaskStep(
                        id="lift",
                        op="move_relative",
                        executor="primitive",
                        exposed=True,
                        params={
                            "dx": 0.0,
                            "dy": 0.0,
                            "dz": _sub(_state("_task_ctx", "travel_z"), _state("_position", "z")),
                            "speed": 0.45,
                        },
                        public_params={
                            "dx": 0.0,
                            "dy": 0.0,
                            "dz": 0.08,
                            "speed": 0.45,
                        },
                        note="Positive dz retreat after release.",
                        failure_observations={
                            "part_name": _state("_held_part"),
                            "destination_location": _arg("destination_location"),
                        },
                    ),
                ),
                effects=(
                    RobotTaskEffect(target="held_part", action="clear"),
                    RobotTaskEffect(target="current_state", action="set", value="placed"),
                    RobotTaskEffect(target="gripper_state", action="set", value="open"),
                    RobotTaskEffect(target="task_ctx", action="clear"),
                ),
                success_response={
                    "status": "completed",
                    "content": _format(
                        "Assembled {placed} at {destination_location}.",
                        placed=_first(_arg("part_name"), _state("_held_part")),
                        destination_location=_arg("destination_location"),
                    ),
                    "placed_location": _arg("destination_location"),
                },
                dry_run_description=_format(
                    "Assembling {held_part} at {destination_location} (orientation={orientation})",
                    held_part=_state("_held_part"),
                    destination_location=_arg("destination_location"),
                    orientation=_first(_arg("orientation"), "default"),
                ),
                dry_run_duration=5.0,
                failure_part=_first(_arg("part_name"), _state("_held_part")),
                notes=(
                    "RobotAgent.place_insert releases the part at the pose established by place_approach, detaches and snaps/settles in simulation, then lifts away.",
                    "open_gripper and detach_part are hidden from synthesis; use release_part as the visible composite.",
                ),
            ),
        )
    ),
    _register_robot_task(
        RobotTaskDefinition(
            name="move_home",
            description="Robot arm move to its home position.",
            arguments=(
                RobotTaskArgument(
                    name="product_jid",
                    type="string",
                    description="JID of the ProductAgent that owns this task.",
                ),
                RobotTaskArgument(name="task_id", type="string"),
            ),
            exposure=RobotTaskExposure(
                predicates=(
                    RobotTaskGuard(
                        predicate="named_pose_available",
                        args={"pose_name": "home"},
                    ),
                ),
            ),
            program=RobotTaskProgram(
                entry_state="any",
                success_state="idle",
                entry_guards=(
                    RobotTaskGuard(
                        predicate="held_part_empty",
                        message="Cannot move home while still holding a part; assemble it first.",
                        description="held_part is empty",
                    ),
                    RobotTaskGuard(
                        predicate="named_pose_available",
                        args={"pose_name": "home"},
                        description="resource exposes a named pose called 'home'",
                    ),
                ),
                steps=(
                    RobotTaskStep(
                        id="move_home",
                        op="move_to_named_pose",
                        executor="primitive",
                        exposed=True,
                        params={
                            "pose_name": "home",
                            "speed": 0.25,
                        },
                        public_params={"pose_name": "home", "speed": 0.25},
                        dry_run_output={
                            "absolute_position": {"x": 0.0, "y": 0.0, "z": 445.0},
                        },
                    ),
                ),
                effects=(
                    RobotTaskEffect(target="current_state", action="set", value="idle"),
                    RobotTaskEffect(
                        target="position",
                        action="set",
                        value={"x": 0.0, "y": 0.0, "z": 445.0},
                    ),
                    RobotTaskEffect(target="bridge_pose_ref", action="set", value="home"),
                    RobotTaskEffect(target="task_ctx", action="clear"),
                ),
                success_response={
                    "status": "completed",
                    "content": "At home position.",
                },
                dry_run_description="Moving arm to home position",
                dry_run_duration=3.0,
                failure_part="",
                notes=(
                    "RobotAgent.move_home maps to the controller's home/named-pose motion when that named pose is available.",
                ),
            ),
        )
    ),
)


def robot_task_registry() -> dict[str, RobotTaskDefinition]:
    return {task.name: task for task in _ROBOT_TASKS}


def robot_task_names() -> tuple[str, ...]:
    registry = robot_task_registry()
    ordered = [name for name in _TASK_ORDER if name in registry]
    ordered.extend(task.name for task in _ROBOT_TASKS if task.name not in ordered)
    return tuple(ordered)


def robot_task_capability_context(
    *,
    static_capabilities: dict[str, Any] | None = None,
    named_positions: dict[str, Any] | None = None,
    controller_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "static_capabilities": deepcopy(static_capabilities or {}),
        "named_positions": deepcopy(named_positions or {}),
        "controller_config": deepcopy(controller_config or {}),
    }


def resolve_robot_task_names(
    *,
    static_capabilities: dict[str, Any] | None = None,
    named_positions: dict[str, Any] | None = None,
    controller_config: dict[str, Any] | None = None,
    requested_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> tuple[str, ...]:
    capability_context = robot_task_capability_context(
        static_capabilities=static_capabilities,
        named_positions=named_positions,
        controller_config=controller_config,
    )
    registry = robot_task_registry()
    enabled = [
        name
        for name in robot_task_names()
        if name in registry and registry[name].is_enabled(capability_context)
    ]
    if requested_names is None:
        return tuple(enabled)
    requested = [str(name or "").strip() for name in requested_names if str(name or "").strip()]
    if not requested:
        return tuple(enabled)
    requested_set = set(requested)
    return tuple(name for name in enabled if name in requested_set)


def robot_task_docstring(function_name: str) -> str:
    task = robot_task_registry().get(str(function_name or "").strip())
    return task.rendered_docstring() if task is not None else ""


def robot_task_capability_decompositions(
    *,
    function_name: str = "",
    resource_jid: str = "",
) -> dict[str, Any]:
    registry = robot_task_registry()
    token = str(function_name or "").strip()
    if token:
        task = registry.get(token)
        return task.capability_decomposition(resource_jid=resource_jid) if task is not None else {}
    return {
        name: task.capability_decomposition(resource_jid=resource_jid)
        for name, task in registry.items()
    }


__all__ = [
    "RobotTaskArgument",
    "RobotTaskDefinition",
    "RobotTaskEffect",
    "RobotTaskExposure",
    "RobotTaskGuard",
    "RobotTaskProgram",
    "RobotTaskStep",
    "execute_robot_task",
    "resolve_robot_task_names",
    "robot_task_capability_context",
    "robot_task_capability_decompositions",
    "robot_task_docstring",
    "robot_task_names",
    "robot_task_registry",
]
