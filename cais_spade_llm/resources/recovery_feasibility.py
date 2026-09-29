"""RA-owned backward primitive derivation and forward capability validation."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from itertools import islice, product
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
    rows = product(*(values[name] for name in values))
    assignments = [
        {name: value for name, value in zip(values, row, strict=True) if value is not omitted}
        for row in islice(rows, 4096)
    ]
    if next(rows, None) is not None:
        complete = False
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


def derive_primitive_union(
    *, task: dict[str, Any], recovery_des_model: dict[str, Any]
) -> dict[str, Any]:
    """Collect primitive identities from every matching successor transition.

    Args:
        task: Candidate carrying the proposed resource and product valuations.
        recovery_des_model: Private RA transition model and primitive support.

    Returns:
        Unordered support with all contributing bindings and valuation coverage.
        Product effects and execution prerequisites remain composition obligations.
    """
    declarations = recovery_des_model.get("state_variables") or {}
    end = task["expected_end_state"]
    covered = [
        field for field in end
        if field in declarations
        and declarations[field].get("private") is not True
        and declarations[field].get("scope", "resource") == "resource"
    ]
    transitions: list[dict[str, Any]] = []
    primitives: dict[str, dict[str, Any]] = {}
    missing_support: list[str] = []
    for event in recovery_des_model.get("events", []):
        bindings = _successor_bindings(task, event, recovery_des_model)
        if bindings is None:
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
        "expected_start_state": deepcopy(task.get("expected_start_state") or {}),
        "expected_end_state": deepcopy(end),
        "covered_valuation_fields": covered,
        "deferred_valuation_fields": [field for field in end if field not in covered],
        "matching_transitions": transitions,
        "missing_support": list(dict.fromkeys(missing_support)),
        "primitives": list(primitives.values()),
    }
    if len(transitions) == 1:
        support["event_name"] = transitions[0]["event_name"]
        support["parameter_bindings"] = deepcopy(transitions[0]["parameter_bindings"])
    return support


def validate_primitive_support(
    *,
    task: dict[str, Any],
    recovery_des_model: dict[str, Any],
    recovery_snapshot: dict[str, Any],
    validate_primitive: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Require a capability witness for every primitive in the successor union.

    Args:
        task: Grounded candidate with exact expected start and end conditions.
        recovery_des_model: Fresh RA-owned events and primitive contracts.
        recovery_snapshot: Fresh snapshot with accepted prefix effects overlaid.
        validate_primitive: Resource-owned, non-executing physical evaluator.

    Returns:
        Feasibility, the complete primitive union, and attempted witnesses.
        Composition must still establish ordering, compatible bindings, and
        the complete intended successor condition.
    """
    support = derive_primitive_union(task=task, recovery_des_model=recovery_des_model)
    if not support["matching_transitions"]:
        result = _result("INFEASIBLE", "No predefined successor supports expected_end_state")
        result["constraint_code"] = "unsupported_successor_condition"
        return {**result, "primitive_support": support}
    catalog = {row["name"]: row for row in recovery_des_model.get("primitive_catalog", [])}
    statuses = ["NEEDS_CONTEXT"] if support["missing_support"] else []
    for row in support["primitives"]:
        name = row["primitive"]
        entry = catalog.get(name)
        attempts: list[dict[str, Any]] = []
        witness = None
        if entry is not None:
            for contributor in row["contributors"]:
                check = _check_primitive(
                    step=contributor["step"], entry=entry,
                    bindings=contributor["parameter_bindings"],
                    snapshot=recovery_snapshot, validate_primitive=validate_primitive,
                )
                attempts.append({"event_name": contributor["event_name"], **check})
                if check["feasibility_status"] == "FEASIBLE":
                    witness = check
                    break
        if witness is None:
            unresolved = entry is None or any(
                check["feasibility_status"] == "NEEDS_CONTEXT" for check in attempts
            )
            witness = _result(
                "NEEDS_CONTEXT" if unresolved else "INFEASIBLE",
                f"No demonstrated capability witness for primitive {name!r}",
            )
        row.update(witness)
        row["parameter_attempts"] = attempts
        row["composition_contract"] = {
            key: deepcopy((entry or {}).get(key))
            for key in ("preconditions", "effects", "capability_constraints", "required_evidence")
            if key in (entry or {})
        }
        statuses.append(row["feasibility_status"])
    status = (
        "INFEASIBLE" if "INFEASIBLE" in statuses
        else "NEEDS_CONTEXT" if "NEEDS_CONTEXT" in statuses
        else "FEASIBLE" if support["primitives"]
        else "INFEASIBLE"
    )
    return {
        **_result(
            status,
            "Every primitive in the successor union has a capability witness"
            if status == "FEASIBLE"
            else f"{status}: the successor union is empty or lacks required capability evidence",
            matching_transitions=deepcopy(support["matching_transitions"]),
            primitive_union=[row["primitive"] for row in support["primitives"]],
            missing_support=deepcopy(support["missing_support"]),
        ),
        "primitive_support": support,
    }


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
            return {
                **result, "params": deepcopy(params),
                "parameter_checks": [
                    *findings, {"params": deepcopy(params), "result": deepcopy(result)}
                ],
            }
        unknown |= status not in {"FEASIBLE", "INFEASIBLE"}
        findings.append({"params": params, "result": deepcopy(result)})
    return _result(
        "NEEDS_CONTEXT" if unknown else "INFEASIBLE",
        "No satisfying primitive parameter assignment was established",
        parameter_checks=findings,
    )


__all__ = ["derive_primitive_union", "validate_primitive_support"]
