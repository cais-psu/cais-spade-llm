"""Derive a resource-owned transition from declared primitive contracts."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from copy import deepcopy
from itertools import islice, product
from math import isfinite
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)

_MISSING = object()
_CONDITIONS = {"equals", "not_equals", "in", "not_in", "exists", "equals_from_param", "not_equals_from_param"}
_EFFECTS = {"set", "set_unknown", "set_from_param", "set_from_param_any_of", "pose_absolute_from_params", "pose_relative_from_params"}
_RESOURCE_NAMES = {"resource_state": ("current_state",), "resource_location": ("current_location",)}
_PART_NAMES = {
    "part_state": ("current_state", "state"),
    "part_location": ("current_location", "location"),
    "part_holder_resource_jid": ("current_holder_resource_jid", "holder_resource_jid"),
}


def _equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _get(mapping: dict, field: str) -> Any:
    if not isinstance(mapping, dict):
        return _MISSING
    if field in mapping:
        return mapping[field]
    value = mapping
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _write(mapping: dict, field: str, value: Any = _MISSING) -> None:
    if field in mapping:
        if value is _MISSING:
            mapping.pop(field)
        else:
            mapping[field] = deepcopy(value)
    if "." not in field:
        if value is not _MISSING:
            mapping[field] = deepcopy(value)
        return
    keys = field.split(".")
    target = mapping
    for key in keys[:-1]:
        if not isinstance(target.get(key), dict):
            if value is _MISSING:
                return
            target[key] = {}
        target = target[key]
    if value is _MISSING:
        target.pop(keys[-1], None)
    else:
        target[keys[-1]] = deepcopy(value)


def _field(field: str, declarations: dict) -> tuple[str, str]:
    if field in declarations:
        return declarations[field].get("scope", "resource"), field
    for name, aliases in _RESOURCE_NAMES.items():
        if field in aliases and name in declarations:
            return "resource", name
    return "resource", field


def _read_initial(field: str, scope: str, resource: dict, part: dict) -> Any:
    source = part if scope == "part" else resource
    names = (field, *(_PART_NAMES if scope == "part" else _RESOURCE_NAMES).get(field, ()))
    for name in names:
        value = _get(source, name)
        if value is not _MISSING:
            return deepcopy(value)
    if scope == "resource":
        facets = source.get("resource_facets")
        for row in [source.get("resource_core", {}), *(facets.values() if isinstance(facets, dict) else [])]:
            for name in names:
                value = _get(row, name)
                if value is not _MISSING:
                    return deepcopy(value)
    return _MISSING


def _conditions(conditions: dict, state: dict, params: dict, declarations: dict) -> tuple[bool, str | None]:
    """Keep unavailable facts distinct from explicit null and failed predicates."""
    for field, condition in conditions.items():
        if not isinstance(condition, dict) or set(condition) - _CONDITIONS:
            return False, f"Unsupported primitive condition for {field!r}"
        scope, key = _field(field, declarations)
        value = state[scope].get(key, _MISSING)
        if value is _MISSING:
            return False, f"Missing primitive condition fact {field!r}"
        for operator, expected in condition.items():
            if operator.endswith("_from_param"):
                if not isinstance(expected, str) or expected not in params:
                    return False, f"Missing primitive condition parameter {expected!r}"
                expected = params[expected]
            if operator in {"in", "not_in"} and not isinstance(expected, list):
                return False, f"Malformed primitive condition {operator!r} for {field!r}"
            if operator == "exists" and type(expected) is not bool:
                return False, f"Malformed primitive exists condition for {field!r}"
            if operator in {"equals", "equals_from_param"}:
                passed = _equal(value, expected)
            elif operator in {"not_equals", "not_equals_from_param"}:
                passed = not _equal(value, expected)
            elif operator in {"in", "not_in"}:
                passed = any(_equal(value, item) for item in expected)
                if operator == "not_in":
                    passed = not passed
            else:
                passed = (value is not None) == expected
            if not passed:
                return False, None
    return True, None


def _apply(effects: dict, state: dict, params: dict, declarations: dict) -> tuple[dict | None, str | None]:  # noqa: C901, PLR0912, PLR0915 - explicit fail-closed effect operators
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_primitives import (
        apply_effects_to_snapshot,
    )

    result = deepcopy(state)
    for field, effect in effects.items():
        if not isinstance(effect, dict) or len(effect) != 1 or set(effect) - _EFFECTS:
            return None, f"Unsupported primitive effect for {field!r}"
        scope, key = _field(field, declarations)
        operator, binding = next(iter(effect.items()))
        if operator == "set_unknown":
            if binding is not True:
                return None, f"Malformed set_unknown effect for {field!r}"
            result[scope].pop(key, None)
            if key not in result["unknown"][scope]:
                result["unknown"][scope].append(key)
            continue
        if operator == "set_from_param" and (not isinstance(binding, str) or binding not in params):
            return None, f"Missing primitive effect parameter {binding!r}"
        if operator == "set_from_param_any_of":
            if not isinstance(binding, list) or not all(isinstance(name, str) for name in binding):
                return None, f"Malformed parameter alternatives for {field!r}"
            if not any(name in params and params[name] not in (None, "") for name in binding):
                return None, f"No recorded primitive effect parameter for {field!r}"
        if operator.startswith("pose_"):
            if not isinstance(binding, list) or len(binding) != 3 or any(
                not isinstance(name, str) or name not in params or type(params[name]) not in (int, float)
                or not isfinite(params[name]) for name in binding
            ):
                return None, f"Missing pose effect parameters for {field!r}"
            if operator == "pose_relative_from_params":
                pose = result[scope].get(key)
                if not isinstance(pose, dict) or any(type(pose.get(axis)) not in (int, float) or not isfinite(pose[axis]) for axis in ("x", "y", "z")):
                    return None, f"Missing starting pose for relative effect {field!r}"
        # Reuse the primitive engine only after checking its permissive defaults.
        source = {"resource_type": "resource"}
        if key in result[scope]:
            source["value"] = deepcopy(result[scope][key])
        projected = apply_effects_to_snapshot({"params": params}, {"effects": {"value": effect}}, source)
        value = projected.get("resource_core", {}).get("value", _MISSING)
        if value is _MISSING:
            return None, f"Primitive effect did not establish {field!r}"
        declaration = declarations.get(key, {})
        domain = declaration.get("domain")
        grounded_held_part = (
            key == "held_part" and value == params.get("part_name") and bool(value)
            and (effect.get("set_from_param") == "part_name"
                 or "part_name" in effect.get("set_from_param_any_of", []))
        )
        if isinstance(domain, list) and not grounded_held_part and not any(_equal(value, item) for item in domain):
            return None, f"Primitive effect {field!r} is outside its declared state domain"
        result[scope][key] = deepcopy(value)
        if key in result["unknown"][scope]:
            result["unknown"][scope].remove(key)
    return result, None


def _evidence_rows(entry: dict, key: str, params: dict) -> tuple[list[tuple], str | None]:
    contract = entry.get("required_evidence") or {}
    if not isinstance(contract, dict) or set(contract) - {"requires", "establishes"}:
        return [], "Unsupported primitive evidence contract"
    rows = contract.get(key, [])
    if not isinstance(rows, list):
        return [], "Malformed primitive evidence requirements"
    result = []
    for row in rows:
        if isinstance(row, str):
            result.append((row, None))
        elif isinstance(row, dict) and set(row) <= {"fact", "part_param"} and isinstance(row.get("fact"), str):
            name = row.get("part_param")
            if name is not None and (not isinstance(name, str) or name not in params or not isinstance(params[name], str)):
                return [], f"Missing evidence binding {name!r}"
            result.append((row["fact"], params.get(name)))
        else:
            return [], "Conditional or unsupported primitive evidence needs an owner evaluator"
    return result, None


def _parameter_rows(entry: dict, bindings: dict, choices: dict, limit: int) -> tuple[list[dict], bool, str | None, int]:
    properties, required = entry.get("params", {}), entry.get("required_params", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return [], False, "Primitive parameter schema is missing or malformed", 0
    schema = {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        return [], False, f"Malformed primitive parameter schema: {exc.message}", 0
    omitted = object()
    rows, complete = {}, True
    for name, definition in properties.items():
        if not isinstance(definition, dict):
            return [], False, f"Primitive parameter {name!r} requires an explicit schema object", 0
        values = []
        if name in bindings:
            values = [deepcopy(bindings[name])]
        else:
            if name not in required:
                values.append(omitted)
            if "const" in definition:
                values.append(deepcopy(definition["const"]))
            elif "enum" in definition:
                values.extend(deepcopy(definition["enum"]))
            else:
                values.extend(deepcopy(choices.get(name, [])))
                if "default" in definition:
                    values.append(deepcopy(definition["default"]))
                complete = False
        if not values:
            return [], False, f"No recorded or finite declared value for parameter {name!r}", 0
        rows[name] = values
    if any(name not in properties for name in required):
        return [], False, "Required parameter has no declared schema", 0
    combinations = product(*(rows[name] for name in rows))
    validator = Draft202012Validator(schema)
    assignments = []
    trials = 0
    for combination in islice(combinations, limit):
        trials += 1
        params = {name: value for name, value in zip(rows, combination, strict=True) if value is not omitted}
        if not any(validator.iter_errors(params)):
            assignments.append(params)
    if next(combinations, None) is not None:
        complete = False
    return assignments, complete, None, trials


def _runtime_views(state: dict, resource: dict, part: dict) -> tuple[dict, dict]:
    result, part_result = deepcopy(resource), deepcopy(part)
    for scope, target, aliases in (("resource", result, _RESOURCE_NAMES), ("part", part_result, _PART_NAMES)):
        facets = target.get("resource_facets")
        mirrors = [row for row in [target.get("resource_core", {}), *(facets.values() if isinstance(facets, dict) else [])]
                   if isinstance(row, dict)]
        for field in state["unknown"][scope]:
            for name in (field, *aliases.get(field, ())):
                for row in [target, *mirrors]:
                    _write(row, name)
        for field, value in state[scope].items():
            for name in (field, *aliases.get(field, ())):
                _write(target, name, value)
                for row in mirrors:
                    if _get(row, name) is not _MISSING:
                        _write(row, name, value)
    return result, part_result


def _contract_guards(guards: Any) -> tuple[dict, str | None]:
    from cais_spade_llm.resources.robot.robot_task_model import RobotTaskGuard
    from cais_spade_llm.resources.robot.robot_task_recovery import _recovery_condition_descriptor

    if not isinstance(guards, list):
        return {}, "Complete task entry and step guards are required"
    conditions = {}
    for row in guards:
        if not isinstance(row, dict):
            return {}, "Malformed task guard"
        condition = row.get("condition") or {}
        if not condition:
            if row.get("predicate", "always") not in ("", "always"):
                return {}, "Task predicate requires an owning resource evaluator"
            continue
        if (not isinstance(condition, dict) or not isinstance(condition.get("field"), str)
                or condition.get("operator", "equals") not in {"equals", "not_equals", "exists"}):
            return {}, "Unsupported task guard condition"
        value = condition.get("value")
        if isinstance(value, dict) and any(str(key).startswith("$") for key in value) and not (
            set(value) == {"$arg"} and isinstance(value["$arg"], str)
        ):
            return {}, "Task guard requires unavailable value bindings"
        compiled = _recovery_condition_descriptor(RobotTaskGuard(predicate="always", condition=condition))
        if compiled is None:
            return {}, "Task guard is unavailable"
        field, predicate = compiled
        current = conditions.setdefault(field, {})
        if set(current) & set(predicate):
            return {}, "Repeated task guard operators require separate evaluation"
        current.update(predicate)
        if condition.get("operator") == "exists":
            current["not_equals"] = ""
    return conditions, None


def _contract_value(value: Any, bindings: dict, snapshot: dict) -> tuple[Any, str | None]:  # noqa: C901, PLR0912 - validate every task-reference form before the shared resolver
    """Use the task resolver only after checking that every reference exists."""
    from cais_spade_llm.resources.robot.robot_task_model import _resolve_value

    runtime = dict(snapshot)
    for field, alias in {"held_part": "_held_part", "current_state": "_current_state",
                         "current_pose": "_position", "task_ctx": "_task_ctx"}.items():
        if field in snapshot:
            runtime[alias] = snapshot[field]
    if isinstance(value, dict) and any(str(key).startswith("$") for key in value):
        if set(value) == {"$arg"} and isinstance(value["$arg"], str):
            if value["$arg"] not in bindings:
                return None, f"Missing task parameter {value['$arg']!r}"
        elif set(value) <= {"$state", "path"} and isinstance(value.get("$state"), str):
            item = runtime.get(value["$state"], _MISSING)
            path = value.get("path", [])
            if not isinstance(path, list):
                return None, "Malformed task state reference"
            for key in path:
                if not isinstance(key, str) or not isinstance(item, dict) or key not in item:
                    return None, "Missing task state reference"
                item = item[key]
            if item is _MISSING:
                return None, "Missing task state reference"
        else:
            return None, "Task expression requires unavailable output or arithmetic evidence"
        return _resolve_value(value, args=bindings, runtime_state=runtime, step_outputs={}), None
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            resolved, error = _contract_value(item, bindings, snapshot)
            if error:
                return None, error
            result[key] = resolved
        return result, None
    if isinstance(value, list):
        result = []
        for item in value:
            resolved, error = _contract_value(item, bindings, snapshot)
            if error:
                return None, error
            result.append(resolved)
        return result, None
    return deepcopy(value), None


def validate_recovery_transition(  # noqa: C901, PLR0912, PLR0915 - one bounded proof search
    *, task: dict[str, Any], recovery_des_model: dict[str, Any], recovery_snapshot: dict[str, Any],
    part_context: dict[str, Any] | None = None,
    validate_primitive: Callable[..., dict[str, Any]] | None = None,
    parameter_evidence: dict[str, Any] | None = None, max_expansions: int = 4096,
    required_task_contract: str | None = None,
) -> dict[str, Any]:
    """Prove one proposed transition using ordered, declared primitive effects.

    Args:
        task: One LLM event with exact public start and end conditions.
        recovery_des_model: Owning resource state declarations and primitive catalog.
        recovery_snapshot: Recorded or previously derived resource facts.
        part_context: Corresponding recorded or previously derived part facts.
        validate_primitive: Optional nonexecuting owner evaluator. Receives primitive,
            params, recovery_snapshot and part_context at that intermediate state.
        parameter_evidence: Recorded parameter bindings, never guessed coordinates.
        max_expansions: Maximum parameter trials and expanded search states, up to 4096.
        required_task_contract: Internal nominal check requiring exactly this complete
            registered task from the starting state, without prerequisite recovery.

    Returns:
        Existing feasibility status with ordered witnesses and derived state.
        FEASIBLE covers declared abstract effects only; it does not authorize execution.
    """
    part_context = part_context or {}
    declarations = recovery_des_model.get("state_variables")
    catalog = recovery_des_model.get("primitive_catalog")
    start, goal = task.get("expected_start_state"), task.get("expected_end_state")
    support = {
        "semantics": "primitive_transition",
        "descriptor_fingerprint": recovery_des_model.get("descriptor_fingerprint", ""),
        "model_fingerprint": recovery_validation_fingerprint(recovery_des_model),
        "task_fingerprint": recovery_validation_fingerprint(task),
        "snapshot_fingerprint": recovery_validation_fingerprint(recovery_snapshot),
        "part_context_fingerprint": recovery_validation_fingerprint(part_context),
        "part_name": task.get("part_name"),
        "required_task_contract": required_task_contract,
        "expected_start_state": deepcopy(start), "expected_end_state": deepcopy(goal),
        "derived_end_state": {}, "primitives": [], "trace": [], "unresolved": [],
        "search": {"limit": max_expansions, "expanded_states": 0, "parameter_trials": 0, "limit_reached": False},
    }

    def finish(status: str, reason: str, *, state: dict | None = None, trace: list | None = None) -> dict:
        if state is not None:
            support["derived_end_state"] = {field: deepcopy(state[_field(field, declarations)[0]][_field(field, declarations)[1]]) for field in goal}
            support["terminal_state"] = deepcopy(state)
            resource, part = _runtime_views(state, recovery_snapshot, part_context)
            support.update(projected_snapshot=resource, projected_part_context=part)
        support["trace"] = deepcopy(trace or [])
        support["primitives"] = [{"primitive": row["primitive"], "params": deepcopy(row["params"])} for row in trace or [] if row.get("primitive")]
        return {"allowed": status == "FEASIBLE", "feasibility_status": status,
                "constraint_code": "" if status == "FEASIBLE" else "resource_validation_unavailable" if status == "NEEDS_CONTEXT" else "primitive_support_infeasible",
                "reason": reason, "primitive_support": deepcopy(support), "evidence": deepcopy(support),
                "derived_end_state": deepcopy(support["derived_end_state"]), "execution_authorized": False}

    def unresolved(reason: str) -> None:
        if reason not in support["unresolved"]:
            support["unresolved"].append(reason)

    if type(max_expansions) is not int or not 1 <= max_expansions <= 4096:
        return finish("NEEDS_CONTEXT", "Search bound must be an integer between 1 and 4096")
    if required_task_contract is not None and (not isinstance(required_task_contract, str) or not required_task_contract):
        return finish("NEEDS_CONTEXT", "Required task contract must be an exact nonempty identifier")
    if not isinstance(declarations, dict) or not declarations or not isinstance(catalog, list) or not catalog:
        return finish("NEEDS_CONTEXT", "Resource state declarations and primitive catalog are required")
    if not isinstance(start, dict) or not isinstance(goal, dict) or not goal:
        return finish("INFEASIBLE", "Candidate start and nonempty end conditions must be objects")
    if any(not isinstance(name, str) or not isinstance(row, dict) or row.get("scope", "resource") not in {"resource", "part"}
           or not isinstance(row.get("domain"), list) for name, row in declarations.items()):
        return finish("NEEDS_CONTEXT", "Resource state declarations are malformed")
    for field in set(start) | set(goal):
        declaration = declarations.get(field) if isinstance(field, str) else None
        if (not isinstance(declaration, dict) or declaration.get("private") is True
                or field.startswith("_") or field.startswith("task_ctx.")
                or declaration.get("scope", "resource") not in {"resource", "part"}):
            return finish("INFEASIBLE", f"Candidate field {field!r} is not publicly declared by this resource")
        if declaration.get("scope") == "part" and not task.get("part_name"):
            return finish("INFEASIBLE", f"Part field {field!r} requires part_name")

    bindings = deepcopy(parameter_evidence or {})
    if not isinstance(bindings, dict):
        return finish("NEEDS_CONTEXT", "Recorded parameter evidence must be an object")
    authored = task.get("params") or {}
    if not isinstance(authored, dict):
        return finish("INFEASIBLE", "Candidate params must be an object")
    authored = {**authored, **({"part_name": task["part_name"]} if task.get("part_name") else {})}
    for name, value in authored.items():
        if name in bindings and not _equal(bindings[name], value):
            return finish("INFEASIBLE", f"Candidate parameter {name!r} conflicts with recorded evidence")
        bindings[name] = deepcopy(value)
    support["parameter_evidence"] = deepcopy(parameter_evidence or {})

    fields = set(declarations)
    valid_catalog = []
    for entry in catalog:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            unresolved("Malformed primitive catalog entry")
            continue
        if any(not isinstance(entry.get(key, {}), dict) for key in ("preconditions", "capability_constraints", "effects", "required_evidence")) or "effects" not in entry:
            unresolved(f"Primitive {entry['name']!r} has missing or malformed effects/conditions")
            continue
        if any(not isinstance(field, str) for key in ("preconditions", "capability_constraints", "effects")
               for field in entry.get(key, {})):
            unresolved(f"Primitive {entry['name']!r} has a malformed field reference")
            continue
        evidence = entry.get("required_evidence", {})
        if any(not isinstance(evidence.get(key, []), list) for key in ("requires", "establishes")):
            unresolved(f"Primitive {entry['name']!r} has malformed evidence requirements")
            continue
        valid_catalog.append(entry)
        for key in ("preconditions", "capability_constraints", "effects"):
            fields.update(entry.get(key, {}))
    primitive_catalog = {entry["name"]: entry for entry in valid_catalog}
    for event in recovery_des_model.get("events", []):
        if not isinstance(event, dict) or not isinstance(event.get("composition_contract"), dict):
            continue
        contract = event["composition_contract"]
        if (not isinstance(event.get("updates"), dict) or not isinstance(event.get("event_name"), str)
                or any(not isinstance(field, str) for field in event["updates"])
                or not isinstance(event.get("guards", {}), dict)
                or any(not isinstance(field, str) for field in event.get("guards", {}))):
            unresolved("Task completion contract has malformed declared updates")
            continue
        if not isinstance(contract.get("effects", []), list):
            unresolved(f"{event['event_name']}: task completion effects are malformed")
            continue
        guards, error = _contract_guards(contract.get("entry_guards"))
        if error:
            unresolved(f"{event['event_name']}: {error}")
        entry = {"name": event["event_name"], "params": contract.get("params", {}),
                 "required_params": contract.get("required_params", []),
                 "effects": event["updates"], "preconditions": guards,
                 "_task_contract": deepcopy(event)}
        valid_catalog.append(entry)
        fields.update(event["updates"])
        fields.update(guards)
        if isinstance(event.get("guards", {}), dict):
            fields.update(event.get("guards", {}))
        for step in contract.get("steps", []) if isinstance(contract.get("steps", []), list) else []:
            if isinstance(step, dict):
                step_guards, _ = _contract_guards(step.get("when", []))
                fields.update(step_guards)
    initial = {"resource": {}, "part": {}, "unknown": {"resource": [], "part": []}}
    for field in fields:
        scope, key = _field(field, declarations)
        if scope not in initial:
            return finish("NEEDS_CONTEXT", f"Unsupported state scope for {field!r}")
        value = _read_initial(key, scope, recovery_snapshot, part_context)
        if value is not _MISSING:
            initial[scope][key] = value
    for field, expected in start.items():
        scope, key = _field(field, declarations)
        if key not in initial[scope]:
            return finish("NEEDS_CONTEXT", f"Candidate starting fact {field!r} is unavailable")
        if not _equal(initial[scope][key], expected):
            return finish("INFEASIBLE", f"Candidate starting fact {field!r} differs from recorded/projected state")

    needed = {_field(field, declarations) for field in goal}
    needed_evidence = set()
    selected = []
    while True:
        changed = False
        for entry in valid_catalog:
            if entry in selected:
                continue
            establishes = (entry.get("required_evidence") or {}).get("establishes", [])
            fact_names = {row if isinstance(row, str) else row["fact"] for row in establishes
                          if isinstance(row, str) or isinstance(row, dict) and isinstance(row.get("fact"), str)}
            if not (needed & {_field(field, declarations) for field in entry["effects"]} or needed_evidence & fact_names):
                continue
            selected.append(entry)
            changed = True
            for key in ("preconditions", "capability_constraints"):
                needed.update(_field(field, declarations) for field in entry.get(key, {}))
            if "_task_contract" in entry:
                event = entry["_task_contract"]
                if isinstance(event.get("guards", {}), dict):
                    needed.update(_field(field, declarations) for field in event.get("guards", {}))
                contract_steps = event["composition_contract"].get("steps", [])
                for step in contract_steps if isinstance(contract_steps, list) else []:
                    if isinstance(step, dict) and step.get("op") in primitive_catalog:
                        primitive = primitive_catalog[step["op"]]
                        for key in ("preconditions", "capability_constraints"):
                            needed.update(_field(field, declarations) for field in primitive.get(key, {}))
            for row in (entry.get("required_evidence") or {}).get("requires", []):
                if isinstance(row, str):
                    needed_evidence.add(row)
                elif isinstance(row, dict) and isinstance(row.get("fact"), str):
                    needed_evidence.add(row["fact"])
        if not changed:
            break
    if required_task_contract is not None:
        selected = [entry for entry in valid_catalog if "_task_contract" in entry and entry["name"] == required_task_contract]
        if not selected:
            return finish("NEEDS_CONTEXT", "The exact required task has no complete declared composition contract")
    support["backward_primitives"] = [entry["name"] for entry in selected]
    choices: dict[str, list] = {}
    for entry in selected:
        for field, effect in entry["effects"].items():
            if not isinstance(effect, dict):
                continue
            names = ([effect["set_from_param"]] if "set_from_param" in effect else effect.get("set_from_param_any_of", []))
            scope, key = _field(field, declarations)
            values = list(declarations.get(key, {}).get("domain", []))
            if key in initial[scope]:
                values.append(initial[scope][key])
            for goal_field, value in goal.items():
                if _field(goal_field, declarations) == (scope, key):
                    values.append(value)
            if isinstance(names, list):
                for name in names:
                    if isinstance(name, str):
                        choices.setdefault(name, []).extend(value for value in values if not any(_equal(value, old) for old in choices.get(name, [])))

    def complete(state: dict) -> bool:
        return all(key in state[scope] and _equal(state[scope][key], value)
                   for field, value in goal.items() for scope, key in [_field(field, declarations)])

    def advance(entry: dict, params: dict, state: dict, facts: frozenset, trace: list) -> tuple | None:
        if "part_name" in params and task.get("part_name") and params["part_name"] != task["part_name"]:
            return None
        for group in ("preconditions", "capability_constraints"):
            passed, error = _conditions(entry.get(group, {}), state, params, declarations)
            if error:
                unresolved(f"{entry['name']}: {error}")
            if not passed:
                return None
        requires, error = _evidence_rows(entry, "requires", params)
        if error:
            unresolved(f"{entry['name']}: {error}")
            return None
        if not set(requires) <= facts:
            unresolved(f"{entry['name']}: required evidence facts are unavailable")
            return None
        establishes, error = _evidence_rows(entry, "establishes", params)
        if error:
            unresolved(f"{entry['name']}: {error}")
            return None
        next_state, error = _apply(entry["effects"], state, params, declarations)
        if error:
            unresolved(f"{entry['name']}: {error}")
            return None
        capability = None
        if validate_primitive is not None:
            resource, part = _runtime_views(state, recovery_snapshot, part_context)
            capability = validate_primitive(primitive=entry["name"], params=deepcopy(params),
                                            recovery_snapshot=resource, part_context=part)
            if not isinstance(capability, dict) or capability.get("allowed") is not True or capability.get("feasibility_status") != "FEASIBLE":
                if not isinstance(capability, dict) or capability.get("feasibility_status") != "INFEASIBLE":
                    unresolved(f"{entry['name']}: owner capability evidence is unavailable")
                return None
        next_facts = facts | frozenset(establishes)
        row = {"step_index": len(trace), "primitive": entry["name"], "params": deepcopy(params),
               "declaration_fingerprint": recovery_validation_fingerprint(entry),
               "declared_effects": deepcopy(entry["effects"]),
               "start_state": deepcopy(state), "end_state": deepcopy(next_state),
               "required_evidence": deepcopy(requires), "established_evidence": deepcopy(establishes),
               "capability_evidence": deepcopy(capability)}
        next_trace = [*trace, row]
        return next_state, next_facts, next_trace

    def advance_contract(entry: dict, params: dict, state: dict, facts: frozenset, trace: list) -> tuple | None:  # noqa: C901, PLR0912, PLR0915 - ordered complete-contract proof stages
        event = entry["_task_contract"]
        contract = event["composition_contract"]

        def missing(reason: str) -> None:
            unresolved(f"{entry['name']}: {reason}")

        guards, error = _contract_guards(contract.get("entry_guards"))
        steps, support_steps = contract.get("steps"), event.get("primitive_support")
        if error or not isinstance(steps, list) or not steps or not isinstance(support_steps, list):
            missing(error or "Complete task program and primitive support are required")
            return None
        if any(not isinstance(step, dict) for step in steps + support_steps):
            missing("Malformed task program step")
            return None
        if [step.get("primitive") for step in support_steps] != [step.get("op") for step in steps if step.get("executor", "primitive") == "primitive"]:
            missing("Complete task steps do not match the declared primitive support")
            return None
        for conditions in (guards, event.get("guards", {})):
            if not isinstance(conditions, dict):
                missing("Malformed task entry conditions")
                return None
            passed, error = _conditions(conditions, state, params, declarations)
            if error:
                missing(error)
            if not passed:
                return None
        for field, value in (("resource_state", contract.get("entry_state")), ("part_state", contract.get("part_in_state"))):
            if value in (None, "", "any"):
                continue
            passed, error = _conditions({field: {"equals": value}}, state, params, declarations)
            if error:
                missing(error)
            if not passed:
                return None
        snapshot, _ = _runtime_views(state, recovery_snapshot, part_context)
        required_context = contract.get("required_context_keys", [])
        if not isinstance(required_context, list) or any(not isinstance(key, str) for key in required_context):
            missing("Malformed required task context")
            return None
        for key in required_context:
            value = _get(snapshot, f"task_ctx.{key}")
            if value is _MISSING or value is None:
                missing(f"Required task context {key!r} is unavailable")
                return None
        current_state, current_facts, current_trace = state, facts, trace
        for step in steps:
            conditions, error = _contract_guards(step.get("when", []))
            if error:
                missing(error)
                return None
            passed, error = _conditions(conditions, current_state, params, declarations)
            if error:
                missing(error)
                return None
            if not passed:
                continue
            if step.get("executor", "primitive") != "primitive" or step.get("op") not in primitive_catalog:
                missing("Task step has no declared nonexecuting primitive contract")
                return None
            if support["search"]["parameter_trials"] >= max_expansions:
                support["search"]["limit_reached"] = True
                missing("Bounded primitive search reached its limit inside task contract")
                return None
            snapshot, _ = _runtime_views(current_state, recovery_snapshot, part_context)
            resolved, error = _contract_value(step.get("params", {}), params, snapshot)
            if error or not isinstance(resolved, dict):
                missing(error or "Malformed task primitive parameters")
                return None
            primitive = primitive_catalog[step["op"]]
            schema = {"type": "object", "properties": primitive.get("params", {}),
                      "required": primitive.get("required_params", []), "additionalProperties": False}
            support["search"]["parameter_trials"] += 1
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                missing(f"Task primitive parameter schema is malformed: {exc.message}")
                return None
            if any(Draft202012Validator(schema).iter_errors(resolved)):
                missing("Task primitive parameter binding is incomplete or invalid")
                return None
            advanced = advance(primitive, resolved, current_state, current_facts, current_trace)
            if advanced is None:
                return None
            current_state, current_facts, current_trace = advanced
        transition = contract.get("part_transition") or {}
        completed = transition.get("completed", {}) if isinstance(transition, dict) else None
        if not isinstance(completed, dict):
            missing("Malformed task completion requirements")
            return None
        if any(str(key).startswith("verify_") and value is not False for key, value in completed.items()):
            missing("Task completion requires verification evidence after its primitive sequence")
            return None
        part_effects = {}
        if "state" in completed:
            part_effects["part_state"] = {"set": completed["state"]}
        if completed.get("location_template"):
            resource_jid = recovery_des_model.get("resource_jid") or task.get("resource_jid")
            template = completed["location_template"]
            if not isinstance(template, str) or not isinstance(resource_jid, str):
                missing("Task part location template has no resource binding")
                return None
            part_effects["part_location"] = {"set": template.replace("{resource_jid}", resource_jid)}
        elif completed.get("location_param"):
            part_effects["part_location"] = {"set_from_param": completed["location_param"]}
        for effect in contract.get("effects", []):
            if not isinstance(effect, dict) or not isinstance(effect.get("target"), str):
                missing("Malformed task completion effect")
                return None
            target = effect["target"]
            if declarations.get(target, {}).get("scope") != "part":
                continue
            if effect.get("action") == "clear":
                part_effects[target] = {"set": None}
            elif effect.get("action") == "set":
                value, error = _contract_value(effect.get("value"), params, snapshot)
                if error:
                    missing(error)
                    return None
                part_effects[target] = {"set": value}
            else:
                missing("Unsupported task part completion effect")
                return None
        for field, effect in event["updates"].items():
            if declarations.get(field, {}).get("scope") != "part":
                continue
            if field not in part_effects:
                missing(f"Part field {field!r} has no declared task completion effect")
                return None
            declared_state, error = _apply({field: part_effects[field]}, current_state, params, declarations)
            summary_state, summary_error = _apply({field: effect}, current_state, params, declarations)
            if error or summary_error or not _equal(declared_state["part"], summary_state["part"]):
                missing(error or summary_error or f"Part field {field!r} differs from its declared task completion effect")
                return None
        if len(current_trace) == len(trace):
            missing("Task contract did not establish a primitive witness")
            return None
        next_state, error = _apply(event["updates"], current_state, params, declarations)
        if error:
            missing(error)
            return None
        row = {"step_index": len(current_trace), "primitive": None, "task_contract": event["event_name"],
               "primitive_start_index": len(trace), "primitive_end_index": len(current_trace),
               "params": deepcopy(params), "declaration_fingerprint": recovery_validation_fingerprint(event),
               "declared_effects": deepcopy(event["updates"]), "start_state": deepcopy(current_state),
               "end_state": deepcopy(next_state), "required_evidence": [], "established_evidence": []}
        return next_state, current_facts, [*current_trace, row]

    queue = deque([(initial, frozenset(), [])])
    seen = set()
    while queue:
        if support["search"]["expanded_states"] >= max_expansions or support["search"]["parameter_trials"] >= max_expansions:
            support["search"]["limit_reached"] = True
            unresolved("Bounded primitive search reached its limit before establishing the transition")
            break
        state, facts, trace = queue.popleft()
        marker = recovery_validation_fingerprint({"state": state, "facts": sorted(facts, key=repr)})
        if marker in seen:
            continue
        seen.add(marker)
        support["search"]["expanded_states"] += 1
        if trace and complete(state):
            return finish("FEASIBLE", "Declared primitive effects establish every proposed end condition", state=state, trace=trace)
        for entry in selected:
            remaining = max_expansions - support["search"]["parameter_trials"]
            if remaining <= 0:
                break
            rows, exhaustive, error, trials = _parameter_rows(entry, bindings, choices, remaining)
            support["search"]["parameter_trials"] += trials
            if error:
                unresolved(f"{entry['name']}: {error}")
            if not exhaustive:
                unresolved(f"{entry['name']}: parameter evidence does not exhaust its declared domain")
            for params in rows:
                advanced = (advance_contract if "_task_contract" in entry else advance)(entry, params, state, facts, trace)
                if advanced is None:
                    continue
                next_state, next_facts, next_trace = advanced
                if complete(next_state):
                    return finish("FEASIBLE", "Declared primitive effects establish every proposed end condition", state=next_state, trace=next_trace)
                if required_task_contract is None:
                    queue.append((next_state, next_facts, next_trace))
    if not selected or any(not any(_field(field, declarations) in {_field(item, declarations) for item in entry["effects"]} for entry in selected)
                           for field in goal if not _equal(initial[_field(field, declarations)[0]].get(_field(field, declarations)[1], _MISSING), goal[field])):
        unresolved("No declared primitive effect establishes every required changed end field")
    if support["unresolved"]:
        return finish("NEEDS_CONTEXT", "; ".join(support["unresolved"]))
    return finish("INFEASIBLE", "No ordered primitive witness satisfies the proposed transition in the finite declared state space")


__all__ = ["validate_recovery_transition"]
