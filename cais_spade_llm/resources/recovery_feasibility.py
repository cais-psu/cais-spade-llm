"""RA-owned backward primitive derivation and forward capability validation."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from itertools import product
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


def _successor_bindings(  # noqa: C901, PLR0912 - exact DES guard/update cases
    task: dict[str, Any], event: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any] | None:
    """Match a predefined successor, without requiring its predecessor now."""
    end = task["expected_end_state"]
    declarations = model["state_variables"]
    if "resource_state" not in end or "resource_state" not in event.get("updates", {}):
        return None
    bindings = deepcopy(task.get("params") or {})
    if task.get("part_name"):
        bindings["part_name"] = task["part_name"]
    for name, condition in event.get("parameter_bindings", {}).items():
        if "equals" in condition and not _bind_parameter(bindings, name, condition["equals"]):
            return None
    for field, value in end.items():
        if (
            field not in declarations
            or declarations[field].get("private") is True
            or declarations[field].get("scope", "resource") != "resource"
        ):
            continue
        update = event.get("updates", {}).get(field)
        if update is not None:
            if "set" in update:
                if not _equal(update["set"], value):
                    return None
            elif "set_from_param" in update:
                name = update["set_from_param"]
                if not _bind_parameter(bindings, name, value):
                    return None
            else:
                return None
        else:
            # An unmodified field must be possible in the nominal predecessor.
            # It is not sufficient that it appears in the generated end state.
            guard = event.get("guards", {}).get(field, {})
            if "equals" in guard and not _equal(guard["equals"], value):
                return None
            if "not_equals" in guard and _equal(guard["not_equals"], value):
                return None
            if "exists" in guard and (value is not None) != guard["exists"]:
                return None
            if "equals_from_param" in guard:
                name = guard["equals_from_param"]
                if not _bind_parameter(bindings, name, value):
                    return None
            if "not_equals_from_param" in guard:
                name = guard["not_equals_from_param"]
                if name in bindings and _equal(bindings[name], value):
                    return None
            if not any(_equal(value, item) for item in declarations[field].get("domain", [])):
                return None
    return bindings


def _bound_value(value: Any, bindings: dict[str, Any]) -> tuple[bool, Any]:
    if isinstance(value, dict):
        if set(value) == {"$arg"}:
            name = value["$arg"]
            return name in bindings, deepcopy(bindings.get(name))
        if any(str(key).startswith("$") for key in value):
            return False, None
        result = {}
        for key, item in value.items():
            known, resolved = _bound_value(item, bindings)
            if not known:
                return False, None
            result[key] = resolved
        return True, result
    if isinstance(value, list):
        items = [_bound_value(item, bindings) for item in value]
        return all(known for known, _ in items), [resolved for _, resolved in items]
    return True, deepcopy(value)


def _parameter_assignments(
    step: dict[str, Any], entry: dict[str, Any], bindings: dict[str, Any]
) -> tuple[list[dict[str, Any]], bool]:
    """Enumerate declared finite assignments; never invent continuous values."""
    properties = entry.get("params") or {}
    required = set(entry.get("required_params") or [])
    authored = step.get("params") or {}
    omitted = object()
    values: dict[str, list[Any]] = {}
    complete = True
    for name in dict.fromkeys([*authored, *properties, *sorted(required)]):
        schema = properties.get(name, {})
        if name in authored:
            known, value = _bound_value(authored[name], bindings)
        else:
            known, value = name in bindings, bindings.get(name)
        if known:
            values[name] = [value]
            continue
        choices = [omitted] if name not in required and name not in authored else []
        if "const" in schema:
            choices.append(deepcopy(schema["const"]))
        elif "enum" in schema:
            choices.extend(deepcopy(schema["enum"]))
        else:
            if "default" in schema:
                choices.append(deepcopy(schema["default"]))
            complete = False
        if not choices:
            return [], False
        values[name] = choices
    assignments = [
        {name: value for name, value in zip(values, row, strict=True) if value is not omitted}
        for row in product(*(values[name] for name in values))
    ]
    return assignments, complete


def _precondition_result(entry: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any] | None:
    for field, condition in (entry.get("capability_constraints") or {}).items():
        value: Any = snapshot
        for key in field.split("."):
            if not isinstance(value, dict) or key not in value:
                return _result("NEEDS_CONTEXT", f"Primitive condition {field!r} is unavailable")
            value = value[key]
        if not isinstance(condition, dict):
            return _result("NEEDS_CONTEXT", f"Primitive condition {field!r} is malformed")
        if set(condition) - {"equals", "not_equals", "in", "not_in", "exists"}:
            return _result("NEEDS_CONTEXT", f"Primitive condition {field!r} needs an evaluator")
        passed = (
            ("equals" not in condition or _equal(value, condition["equals"]))
            and ("not_equals" not in condition or not _equal(value, condition["not_equals"]))
            and ("in" not in condition or any(_equal(value, item) for item in condition["in"]))
            and (
                "not_in" not in condition
                or not any(_equal(value, item) for item in condition["not_in"])
            )
            and ("exists" not in condition or (value is not None) == condition["exists"])
        )
        if not passed:
            return _result(
                "INFEASIBLE",
                f"Primitive condition {field!r} is not satisfied",
                field=field,
                actual=deepcopy(value),
                condition=deepcopy(condition),
            )
    return None


def validate_primitive_support(
    *,
    task: dict[str, Any],
    recovery_des_model: dict[str, Any],
    recovery_snapshot: dict[str, Any],
    validate_primitive: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Derive alternatives and require a capability witness for each primitive.

    Args:
        task: Grounded candidate with exact expected start and end conditions.
        recovery_des_model: Fresh RA-owned events and primitive contracts.
        recovery_snapshot: Fresh snapshot with accepted prefix effects overlaid.
        validate_primitive: Resource-owned, non-executing physical evaluator.

    Returns:
        Feasibility, selected primitive support, and evidence for every attempted
        alternative. Primitive Composition must still establish ordering,
        compatible parameter bindings, and the complete intended condition.
    """
    catalog = {row["name"]: row for row in recovery_des_model.get("primitive_catalog", [])}
    alternatives: list[dict[str, Any]] = []
    missing_context = False
    for event in recovery_des_model.get("events", []):
        bindings = _successor_bindings(task, event, recovery_des_model)
        if bindings is None:
            continue
        steps = event.get("primitive_support") or []
        alternative: dict[str, Any] = {
            "event_name": event["event_name"],
            "parameter_bindings": deepcopy(bindings),
            "primitives": [],
        }
        alternatives.append(alternative)
        if not steps:
            alternative.update(_result("NEEDS_CONTEXT", "Matching event has no primitive support"))
            missing_context = True
            continue
        statuses: list[str] = []
        for step in steps:
            name = step["primitive"]
            entry = catalog.get(name)
            primitive_result: dict[str, Any]
            if entry is None:
                primitive_result = _result(
                    "NEEDS_CONTEXT", f"Primitive {name!r} is absent from the RA catalog"
                )
            else:
                primitive_result = _check_primitive(
                    step=step,
                    entry=entry,
                    bindings=bindings,
                    snapshot=recovery_snapshot,
                    validate_primitive=validate_primitive,
                )
            alternative["primitives"].append({"primitive": name, **primitive_result})
            statuses.append(primitive_result["feasibility_status"])
        status = (
            "INFEASIBLE"
            if "INFEASIBLE" in statuses
            else "NEEDS_CONTEXT"
            if "NEEDS_CONTEXT" in statuses
            else "FEASIBLE"
        )
        alternative["feasibility_status"] = status
        if status == "FEASIBLE":
            support = {
                "event_name": event["event_name"],
                "parameter_bindings": deepcopy(bindings),
                "primitives": deepcopy(alternative["primitives"]),
                "descriptor_fingerprint": recovery_des_model.get("descriptor_fingerprint", ""),
                "expected_start_state": deepcopy(task.get("expected_start_state") or {}),
                "expected_end_state": deepcopy(task["expected_end_state"]),
            }
            return {
                **_result(
                    "FEASIBLE",
                    "One matching transition has feasible primitive support",
                    alternatives=alternatives,
                ),
                "primitive_support": support,
            }
        missing_context |= status == "NEEDS_CONTEXT"
    if not alternatives:
        result = _result("INFEASIBLE", "No predefined successor supports expected_end_state")
        result["constraint_code"] = "unsupported_successor_condition"
        return result
    return _result(
        "NEEDS_CONTEXT" if missing_context else "INFEASIBLE",
        "No matching transition has demonstrated feasible primitive support",
        alternatives=alternatives,
    )


def _check_primitive(
    *,
    step: dict[str, Any],
    entry: dict[str, Any],
    bindings: dict[str, Any],
    snapshot: dict[str, Any],
    validate_primitive: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    condition_failure = _precondition_result(entry, snapshot)
    if condition_failure is not None:
        return condition_failure
    schema = {
        "type": "object",
        "properties": entry.get("params") or {},
        "required": entry.get("required_params") or [],
        "additionalProperties": False,
    }
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        return _result(
            "NEEDS_CONTEXT", "RA primitive parameter domain is malformed", error=exc.message
        )
    validator = Draft202012Validator(schema)
    assignments, complete = _parameter_assignments(step, entry, bindings)
    unknown = not complete
    findings = []
    for params in assignments:
        errors = [error.message for error in validator.iter_errors(params)]
        if errors:
            findings.append({"params": params, "errors": errors})
            continue
        result = validate_primitive(primitive=entry["name"], params=deepcopy(params))
        if not isinstance(result, dict):
            result = _result("NEEDS_CONTEXT", "Primitive evaluator returned no evidence")
        status = result.get("feasibility_status")
        if result.get("allowed") is True and status == "FEASIBLE":
            return {**result, "params": deepcopy(params)}
        unknown |= status not in {"FEASIBLE", "INFEASIBLE"}
        findings.append({"params": params, "result": deepcopy(result)})
    return _result(
        "NEEDS_CONTEXT" if unknown else "INFEASIBLE",
        "No satisfying primitive parameter assignment was established",
        parameter_checks=findings,
    )


__all__ = ["validate_primitive_support"]
