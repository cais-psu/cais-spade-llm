from __future__ import annotations

"""RA-owned backward primitive derivation from declared DES successors."""

from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


def _result(status: str, reason: str, **evidence: Any) -> dict[str, Any]:
    return {
        "allowed": status == "FEASIBLE",
        "feasibility_status": status,
        "constraint_code": (
            "resource_validation_unavailable"
            if status == "NEEDS_CONTEXT"
            else "primitive_support_infeasible"
            if status == "INFEASIBLE"
            else ""
        ),
        "reason": reason,
        "evidence": evidence,
    }


def _equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _bind_parameter(bindings: dict[str, Any], name: str, value: Any) -> bool:
    if name in bindings and not _equal(bindings[name], value):
        return False
    bindings[name] = deepcopy(value)
    return True


def _supported_guard(guard: Any) -> bool:
    return isinstance(guard, dict) and not set(guard) - {
        "equals", "not_equals", "exists", "equals_from_param", "not_equals_from_param", "in", "not_in",
    } and all(
        key not in guard or isinstance(guard[key], list) for key in ("in", "not_in")
    ) and ("exists" not in guard or type(guard["exists"]) is bool) and all(
        key not in guard or isinstance(guard[key], str) and bool(guard[key])
        for key in ("equals_from_param", "not_equals_from_param")
    )


def _successor_bindings(  # noqa: C901, PLR0912 - exact DES guard/update cases
    task: dict[str, Any], event: dict[str, Any], model: dict[str, Any],
    *, missing: list[str] | None = None,
) -> dict[str, Any] | None:
    """Match a predefined successor, without requiring its predecessor now."""
    end = task["expected_end_state"]
    declarations = model["state_variables"]
    missing = missing if missing is not None else []
    if not isinstance(event, dict) or not isinstance(event.get("event_name"), str):
        missing.append("Malformed resource transition")
        return None
    updates, guards = event.get("updates"), event.get("guards", {})
    declared_bindings = event.get("parameter_bindings", {})
    if not all(isinstance(row, dict) for row in (updates, guards, declared_bindings)):
        missing.append(f"{event['event_name']}: transition declarations are unavailable")
        return None
    bindings = deepcopy(task.get("params") or {})
    if not isinstance(bindings, dict):
        missing.append("Candidate params must be an object")
        return None
    if task.get("part_name") and not _bind_parameter(bindings, "part_name", task["part_name"]):
        return None
    for name, condition in declared_bindings.items():
        if (isinstance(condition, dict) and set(condition) == {"location_type"}
                and isinstance(condition["location_type"], str)
                and condition["location_type"] in {"part_location", "current_location", "reachable_location"}):
            # These declarations constrain nominal context. Successor updates
            # below bind their values; they do not require that predecessor now.
            continue
        if (isinstance(condition, dict) and condition
                and not set(condition) - {"required", "type", "reference"}
                and ("required" not in condition or type(condition["required"]) is bool)
                and condition.get("reference") in (None, "part_name")):
            schema = {key: deepcopy(value) for key, value in condition.items() if key == "type"}
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError:
                missing.append(f"{event['event_name']}: unsupported parameter binding {name!r}")
                return None
            if name in bindings and list(Draft202012Validator(schema).iter_errors(bindings[name])):
                return None
            # Required source values are resolved by owner parameter preparation.
            # A declaration alone supplies no observation or completion evidence.
            continue
        if not isinstance(condition, dict) or set(condition) != {"equals"}:
            missing.append(f"{event['event_name']}: unsupported parameter binding {name!r}")
            return None
        if not _bind_parameter(bindings, name, condition["equals"]):
            return None
    fields = {field: value for field, value in end.items() if field in declarations
              and declarations[field].get("private") is not True
              and declarations[field].get("scope", "resource") == "resource"}
    if not fields:
        return None
    # Bind all updates first so preserved-field matching is independent of JSON order.
    for field, value in fields.items():
        if field in updates:
            update = updates[field]
            if not isinstance(update, dict) or len(update) != 1:
                missing.append(f"{event['event_name']}: unsupported successor update for {field!r}")
                return None
            if "set" in update:
                if not _equal(update["set"], value):
                    return None
            elif "set_from_param" in update:
                name = update["set_from_param"]
                if not isinstance(name, str) or not name:
                    missing.append(f"{event['event_name']}: malformed parameter update for {field!r}")
                    return None
                if not _bind_parameter(bindings, name, value):
                    return None
            else:
                missing.append(f"{event['event_name']}: unsupported successor update for {field!r}")
                return None
            if declarations[field].get("domain_complete") is True and not any(
                _equal(value, item) for item in declarations[field]["domain"]
            ):
                return None
    for field, guard in guards.items():
        declaration = declarations.get(field, {})
        if declaration.get("private") is True or declaration.get("scope", "resource") != "resource":
            continue
        if not declaration or not _supported_guard(guard):
            missing.append(f"{event['event_name']}: unsupported public resource guard for {field!r}")
            return None
    for field, update in updates.items():
        declaration = declarations.get(field, {})
        if declaration.get("private") is True or declaration.get("scope", "resource") != "resource":
            continue
        if (not declaration or not isinstance(update, dict)
                or set(update) not in ({"set"}, {"set_from_param"})
                or "set_from_param" in update and (
                    not isinstance(update["set_from_param"], str) or not update["set_from_param"]
                )):
            missing.append(f"{event['event_name']}: unsupported public resource update for {field!r}")
            return None
    for name, condition in declared_bindings.items():
        if "type" in condition and name in bindings:
            if list(Draft202012Validator({"type": condition["type"]}).iter_errors(bindings[name])):
                return None
        if condition.get("reference") == "part_name" and name in bindings:
            domains = [row["domain"] for row in declarations.values()
                       if row.get("reference") == "part_name" and isinstance(row.get("domain"), list)]
            if not domains:
                missing.append(f"{event['event_name']}: recorded part identities are unavailable")
                return None
            if not any(_equal(bindings[name], value) for domain in domains for value in domain):
                return None
        location_type = condition.get("location_type")
        if location_type is None or name not in bindings:
            continue
        field = "part_location" if location_type == "part_location" else "resource_location"
        declaration = declarations.get(field)
        if not isinstance(declaration, dict):
            missing.append(f"{event['event_name']}: missing {field!r} declaration for parameter {name!r}")
            return None
        if not any(_equal(bindings[name], value) for value in declaration["domain"]):
            return None
    for field, value in fields.items():
        if field not in updates:
            guard = guards.get(field, {})
            if not _supported_guard(guard):
                missing.append(f"{event['event_name']}: unsupported preserved-field guard for {field!r}")
                return None
            if "equals" in guard and not _equal(guard["equals"], value):
                return None
            if "not_equals" in guard and _equal(guard["not_equals"], value):
                return None
            if "exists" in guard and type(guard["exists"]) is not bool:
                missing.append(f"{event['event_name']}: malformed exists guard for {field!r}")
                return None
            if "exists" in guard and (value is not None) != guard["exists"]:
                return None
            for operator in ("in", "not_in"):
                if operator in guard:
                    if not isinstance(guard[operator], list):
                        missing.append(f"{event['event_name']}: malformed {operator} guard for {field!r}")
                        return None
                    member = any(_equal(value, item) for item in guard[operator])
                    if member != (operator == "in"):
                        return None
            grounded_binding = False
            if "equals_from_param" in guard:
                name = guard["equals_from_param"]
                if not isinstance(name, str) or not name:
                    missing.append(f"{event['event_name']}: malformed parameter guard for {field!r}")
                    return None
                grounded_binding = name in bindings and _equal(bindings[name], value)
                if not _bind_parameter(bindings, name, value):
                    return None
            if "not_equals_from_param" in guard:
                name = guard["not_equals_from_param"]
                if not isinstance(name, str) or name not in bindings:
                    missing.append(f"{event['event_name']}: missing preserved-field parameter for {field!r}")
                    return None
                if _equal(bindings[name], value):
                    return None
            # A captured domain may omit a future grounded parameter value. Only
            # an explicit positive binding permits it; complete domains stay closed.
            if not any(_equal(value, item) for item in declarations[field]["domain"]) and not (
                grounded_binding and declarations[field].get("domain_complete") is not True
            ):
                return None
    return bindings


def derive_primitive_union(  # noqa: C901 - preserve exact matching and missing-model evidence
    *, task: dict[str, Any], recovery_des_model: dict[str, Any], model_complete: bool | None = None,
    recovery_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect primitives from successors that support a requested change.

    Args:
        task: Candidate carrying the proposed resource and product valuations.
        recovery_des_model: Private RA transition model and primitive support.
        model_complete: Explicit completeness of the captured transition declarations.
        recovery_snapshot: Recorded or accepted projected starting facts supplied by the owner.

    Returns:
        Unordered support with all contributing bindings and valuation coverage.
        Product effects and execution prerequisites remain composition obligations.
    """
    missing_context: list[str] = []
    declarations = recovery_des_model.get("state_variables")
    events = recovery_des_model.get("events")
    end = task.get("expected_end_state")
    if not isinstance(end, dict) or not end:
        missing_context.append("Nonempty expected_end_state is required")
        end = {}
    if not isinstance(declarations, dict) or not declarations:
        missing_context.append("Resource state declarations are unavailable")
        declarations = {}
    else:
        invalid = [name for name, row in declarations.items() if not isinstance(name, str)
                   or not isinstance(row, dict) or row.get("scope", "resource") not in {"resource", "part"}
                   or not isinstance(row.get("domain"), list)
                   or "domain_complete" in row and type(row["domain_complete"]) is not bool]
        if invalid:
            missing_context.append(f"Malformed state declarations: {invalid!r}")
            declarations = {name: row for name, row in declarations.items() if name not in invalid}
    if not isinstance(events, list):
        missing_context.append("Resource transition declarations are unavailable")
        events = []
    complete = model_complete if model_complete is not None else recovery_des_model.get("model_complete", True)
    if complete is not True:
        missing_context.append("Resource transition declarations are incomplete")
    for field in end:
        if field not in declarations:
            missing_context.append(f"Requested field {field!r} has no state declaration")
        elif declarations[field].get("private") is True:
            missing_context.append(f"Requested field {field!r} is not public")
    covered = [
        field for field in declarations
        if field in end
        and declarations[field].get("private") is not True
        and declarations[field].get("scope", "resource") == "resource"
    ]
    if not covered:
        missing_context.append("No declared public resource end condition is available")
    start = deepcopy(task.get("expected_start_state", {}))
    if not isinstance(start, dict):
        missing_context.append("Recorded or projected expected_start_state is unavailable")
        start = {}
    for field in covered:
        if field not in start and recovery_snapshot is not None and field in recovery_snapshot:
            start[field] = deepcopy(recovery_snapshot[field])
    changes = [field for field in covered if field in start and not _equal(start[field], end[field])]
    preserved = [field for field in covered if field in start and _equal(start[field], end[field])]
    unknown = [field for field in covered if field not in start]
    transitions: list[dict[str, Any]] = []
    primitives: dict[str, dict[str, Any]] = {}
    missing_support: list[str] = []
    matching_model = {**recovery_des_model, "state_variables": declarations}
    matching_task = {**task, "expected_end_state": end}
    for event in events:
        # The nominal-continuation checker revalidates a recorded task, which
        # may still require execution even when its public end values hold.
        nominal = bool(task.get("required_task_contract"))
        if nominal and isinstance(event, dict) and event.get("event_name") != task["required_task_contract"]:
            continue
        if not nominal and isinstance(event, dict) and isinstance(event.get("updates"), dict) and not (
            set(changes + unknown).intersection(event["updates"])
        ):
            continue
        bindings = _successor_bindings(matching_task, event, matching_model, missing=missing_context)
        if bindings is None:
            continue
        updates = event.get("updates", {})
        supported_changes = [field for field in changes if field in updates]
        if not nominal and not supported_changes:
            missing_context.extend(f"Starting value for {field!r} is unavailable" for field in unknown if field in updates)
            continue
        transition = {
            "event_name": event["event_name"],
            "parameter_bindings": deepcopy(bindings),
            "valuation_coverage": {
                field: {
                    "value": deepcopy(end[field]),
                    "basis": "update" if field in event.get("updates", {}) else "preserved",
                }
                for field in covered
            },
            "composition_prerequisites": deepcopy(event.get("guards") or {}),
            "composition_contract": deepcopy(event.get("composition_contract") or {}),
            "supported_changes": supported_changes,
            "preserved_conditions": {field: deepcopy(end[field]) for field in covered if field not in updates},
            "additional_effects": {field: deepcopy(value) for field, value in updates.items() if field not in end},
            "primitives": [],
        }
        transitions.append(transition)
        steps = event.get("primitive_support")
        if not isinstance(steps, list):
            missing_support.append(event["event_name"])
            continue
        for index, step in enumerate(steps):
            if not isinstance(step, dict) or not isinstance(step.get("primitive"), str):
                missing_support.append(event["event_name"])
                continue
            name = step["primitive"]
            transition["primitives"].append(name)
            row = primitives.setdefault(name, {"primitive": name, "contributors": []})
            row["contributors"].append({
                "event_name": event["event_name"],
                "step_index": index,
                "parameter_bindings": deepcopy(bindings),
                "step": deepcopy(step),
                "covered_valuation_fields": list(covered),
            })
    support = {
        "semantics": "successor_primitive_union",
        "descriptor_fingerprint": recovery_des_model.get("descriptor_fingerprint", ""),
        "expected_start_state": deepcopy(start),
        "expected_end_state": deepcopy(end),
        "requested_changes": changes,
        "preserved_valuation_fields": preserved,
        "unknown_start_fields": unknown,
        "covered_valuation_fields": covered,
        "deferred_valuation_fields": [field for field in end if field not in covered],
        "matching_transitions": transitions,
        "missing_support": list(dict.fromkeys(missing_support)),
        "model_complete": not missing_context,
        "missing_context": list(dict.fromkeys(missing_context)),
        "primitives": list(primitives.values()),
    }
    if len(transitions) == 1:
        support["event_name"] = transitions[0]["event_name"]
        support["parameter_bindings"] = deepcopy(transitions[0]["parameter_bindings"])
    return support


__all__ = ["derive_primitive_union"]
