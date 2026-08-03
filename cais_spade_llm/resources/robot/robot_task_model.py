"""Shared declarative models and value helpers for robot tasks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

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
    condition: dict[str, Any] = field(default_factory=dict)


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
    physical_position_required: bool = False

    def render_capability_row(self) -> dict[str, Any]:
        params = (
            self.public_params
            if self.public_params is not None
            else {key: value for key, value in self.params.items() if not str(key).startswith("_")}
        )
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
    """Task-level SSOT; recovery/task decompositions are rendered from `steps`."""

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

    def render_recovery_steps(self) -> list[dict[str, Any]]:
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
        return {argument.name: argument.param_schema() for argument in self.arguments}

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
        # `program.steps` is the authored task SSOT. `recovery_visible_steps` remains
        # a derived rendering for primitive generation and related recovery tooling.
        return {
            "function_name": self.name,
            "source": self.source,
            "modeled_transition": self.program.modeled_transition(),
            "task_preconditions": self.program.task_preconditions(resource_jid=resource_jid),
            "recovery_visible_steps": self.program.render_recovery_steps(),
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
            key: _resolve_value(
                item, args=args, runtime_state=runtime_state, step_outputs=step_outputs
            )
            for key, item in dict(value.get("values") or {}).items()
        }
        return str(value.get("$format") or "").format_map(
            {key: "" if item is None else item for key, item in resolved_values.items()}
        )
    if isinstance(value, dict) and "$first" in value:
        for item in list(value.get("$first") or []):
            resolved = _resolve_value(
                item, args=args, runtime_state=runtime_state, step_outputs=step_outputs
            )
            if resolved not in (None, ""):
                return deepcopy(resolved)
        return None
    if isinstance(value, dict) and "$sub" in value:
        left_raw, right_raw = list(value.get("$sub") or [0, 0])[:2]
        left = _resolve_value(
            left_raw, args=args, runtime_state=runtime_state, step_outputs=step_outputs
        )
        right = _resolve_value(
            right_raw, args=args, runtime_state=runtime_state, step_outputs=step_outputs
        )
        try:
            return float(left or 0.0) - float(right or 0.0)
        except (TypeError, ValueError):
            return 0.0
    if isinstance(value, dict):
        return {
            key: _resolve_value(
                item, args=args, runtime_state=runtime_state, step_outputs=step_outputs
            )
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
        return {
            key: _render_decomposition_value(item, context=context) for key, item in value.items()
        }
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
        return token in {str(name).strip() for name in named_positions}
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
    if guard.condition:
        field_name = str(guard.condition.get("field") or "").strip()
        operator = str(guard.condition.get("operator") or "equals").strip()
        if field_name.startswith("task_ctx."):
            context_key = field_name.split(".", 1)[1]
            actual = dict(runtime_state.get("_task_ctx") or {}).get(context_key)
        else:
            runtime_field = {
                "held_part": "_held_part",
                "resource_state": "_current_state",
                "gripper_state": "_gripper_state",
                "resource_location": "_recovery_pose_ref",
            }.get(field_name, f"_{field_name}")
            actual = runtime_state.get(runtime_field)
        expected = _resolve_value(
            guard.condition.get("value"),
            args=args,
            runtime_state=runtime_state,
            step_outputs=step_outputs,
        )
        if operator == "exists":
            return actual not in (None, "")
        if operator == "not_equals":
            return actual != expected
        return actual == expected

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
