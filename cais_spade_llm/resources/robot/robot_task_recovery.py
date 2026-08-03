"""Recovery DES compiler derived from robot task definitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .robot_task_model import RobotTaskDefinition, RobotTaskGuard
from .robot_task_registry import robot_task_names, robot_task_registry


def _recovery_condition_descriptor(
    guard: RobotTaskGuard,
) -> tuple[str, dict[str, Any]] | None:
    condition = dict(guard.condition or {})
    field_name = str(condition.get("field") or "").strip()
    if not field_name:
        return None
    operator = str(condition.get("operator") or "equals").strip()
    if operator == "exists":
        return field_name, {"exists": True}
    value = condition.get("value")
    if isinstance(value, dict) and str(value.get("$arg") or "").strip():
        parameter_name = str(value.get("$arg") or "").strip()
        key = "not_equals_from_param" if operator == "not_equals" else "equals_from_param"
        return field_name, {key: parameter_name}
    key = "not_equals" if operator == "not_equals" else "equals"
    return field_name, {key: deepcopy(value)}


def _recovery_update_descriptor(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        parameter_name = str(value.get("$arg") or "").strip()
        if parameter_name:
            return {"set_from_param": parameter_name}
        if any(str(key).startswith("$") for key in value):
            return None
    return {"set": deepcopy(value)}


def _robot_task_context_fields(tasks: list[RobotTaskDefinition]) -> set[str]:
    fields: set[str] = set()
    for task in tasks:
        for guard in task.program.entry_guards:
            field_name = str(dict(guard.condition or {}).get("field") or "").strip()
            if field_name.startswith("task_ctx."):
                fields.add(field_name)
        for effect in task.program.effects:
            if effect.target != "task_ctx" or not isinstance(effect.value, dict):
                continue
            for key, value in effect.value.items():
                if _recovery_update_descriptor(value) is not None:
                    fields.add(f"task_ctx.{key}")
    return fields


def robot_recovery_des_descriptor(  # noqa: C901, PLR0912
    *,
    resource_jid: str,
    snapshot: dict[str, Any],
    task_names: tuple[str, ...] | list[str] | None = None,
    reachable_locations: list[Any] | tuple[Any, ...] | None = None,
    named_poses: list[Any] | tuple[Any, ...] | None = None,
    marked_state_conditions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile the robot task registry into the private recovery DES view."""
    registry = robot_task_registry()
    selected_names = tuple(task_names or robot_task_names())
    selected_tasks = [registry[name] for name in selected_names if name in registry]
    if not selected_tasks:
        return {}

    current_state = snapshot.get("current_state")
    current_location = snapshot.get("current_location")
    held_part = snapshot.get("held_part")
    resource_states: list[Any] = [current_state]
    part_states: list[Any] = [None]
    part_locations: list[Any] = [None, resource_jid]
    part_locations.extend(reachable_locations or [])
    locations: list[Any] = [current_location]
    locations.extend(reachable_locations or [])
    locations.extend(named_poses or [])
    events: list[dict[str, Any]] = []
    task_context_fields = _robot_task_context_fields(selected_tasks)

    for task in selected_tasks:
        program = task.program
        if program.entry_state != "any":
            resource_states.append(program.entry_state)
        resource_states.append(program.success_state)
        guards: dict[str, dict[str, Any]] = {}
        if program.part_in_state:
            part_states.append(program.part_in_state)
        for guard in program.entry_guards:
            compiled_guard = _recovery_condition_descriptor(guard)
            if compiled_guard is not None:
                field_name, condition = compiled_guard
                guards[field_name] = condition

        updates: dict[str, dict[str, Any]] = {"resource_state": {"set": program.success_state}}
        for effect in program.effects:
            if effect.target == "current_state" and effect.action == "set":
                compiled_update = _recovery_update_descriptor(effect.value)
                if compiled_update is not None:
                    updates["resource_state"] = compiled_update
            elif effect.target == "held_part":
                if effect.action == "clear":
                    updates["held_part"] = {"set": None}
                else:
                    compiled_update = _recovery_update_descriptor(effect.value)
                    if compiled_update is not None:
                        updates["held_part"] = compiled_update
            elif (
                effect.target == "recovery_pose_ref"
                and effect.action == "set"
                and effect.value is not None
            ):
                compiled_update = _recovery_update_descriptor(effect.value)
                if compiled_update is not None:
                    updates["resource_location"] = compiled_update
            elif effect.target == "task_ctx":
                if effect.action == "clear":
                    for field_name in task_context_fields:
                        updates[field_name] = {"set": None}
                elif isinstance(effect.value, dict):
                    for key, value in effect.value.items():
                        compiled_update = _recovery_update_descriptor(value)
                        if compiled_update is not None:
                            updates[f"task_ctx.{key}"] = compiled_update

        context_mapping = dict(program.context_mapping or {})
        context_location_param = str(context_mapping.get("location_param") or "").strip()
        if context_location_param:
            updates["resource_location"] = {"set_from_param": context_location_param}

        completed_part_transition = dict(dict(program.part_transition or {}).get("completed") or {})
        if completed_part_transition:
            part_state = completed_part_transition.get("state")
            if part_state not in (None, ""):
                updates["part_state"] = {"set": deepcopy(part_state)}
                part_states.append(deepcopy(part_state))
            location_template = str(
                completed_part_transition.get("location_template") or ""
            ).strip()
            part_location_param = str(completed_part_transition.get("location_param") or "").strip()
            if location_template:
                resolved_location = location_template.replace("{resource_jid}", resource_jid)
                updates["part_location"] = {"set": resolved_location}
                part_locations.append(resolved_location)
            elif part_location_param:
                updates["part_location"] = {"set_from_param": part_location_param}

        events.append(
            {
                "event_name": task.name,
                "controllable": True,
                "observable": True,
                "guards": guards,
                "updates": updates,
                "parameter_bindings": (
                    {
                        context_location_param: {
                            "location_type": str(context_mapping.get("location_type") or "").strip()
                        }
                    }
                    if context_location_param
                    else {}
                ),
                "requires_part_binding": any(
                    argument.name == "part_name" for argument in task.arguments
                ),
                "recovery_visible_steps": program.render_recovery_steps(),
                "source": task.source,
            }
        )

    def _domain(values: list[Any]) -> list[Any]:
        result: list[Any] = []
        for value in values:
            if value not in result:
                result.append(deepcopy(value))
        return result

    state_variables: dict[str, dict[str, Any]] = {
        "resource_state": {
            "scope": "resource",
            "domain": _domain(resource_states),
        },
        "resource_location": {
            "scope": "resource",
            "domain": _domain(locations),
        },
        "held_part": {
            "scope": "resource",
            "domain": _domain([held_part, None]),
        },
        "part_state": {"scope": "part", "domain": _domain(part_states)},
        "part_location": {
            "scope": "part",
            "domain": _domain(part_locations),
        },
    }
    for field_name in sorted(task_context_fields):
        state_variables[field_name] = {
            "scope": "resource",
            "domain": _domain([None, *locations]),
            "private": True,
        }

    current_valuation = {
        "resource_state": current_state,
        "resource_location": current_location,
        "held_part": held_part,
    }
    current_valuation.update({field_name: None for field_name in task_context_fields})

    return {
        "state_variables": state_variables,
        "current_valuation": current_valuation,
        "events": events,
        "marked_state_conditions": deepcopy(marked_state_conditions or []),
    }


__all__ = ["robot_recovery_des_descriptor"]
