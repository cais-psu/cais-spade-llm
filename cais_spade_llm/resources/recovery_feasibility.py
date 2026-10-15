from __future__ import annotations

"""RA-owned backward primitive derivation and forward capability validation."""

from collections.abc import Callable
from copy import deepcopy
from itertools import combinations, islice, product
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.resources.recovery_transition_derivation import validate_recovery_transition


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


def _bound_value(value: Any, bindings: dict[str, Any]) -> tuple[bool, Any]:
    if isinstance(value, dict):
        if set(value) == {"$arg"}:
            name = value["$arg"]
            return (name in bindings, deepcopy(bindings.get(name))) if isinstance(name, str) else (False, None)
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
    step: dict[str, Any], entry: dict[str, Any], bindings: dict[str, Any], *, limit: int = 4096,
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
            if not known and name in bindings:
                known, value = True, deepcopy(bindings[name])
        else:
            known, value = name in bindings, bindings.get(name)
        if known:
            values[name] = [value]
            continue
        choices = [omitted] if name not in required else []
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
        for row in islice(rows, limit)
    ]
    if next(rows, None) is not None:
        complete = False
    return assignments, complete


def _precondition_result(entry: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any] | None:
    constraints = entry.get("capability_constraints", {})
    if not isinstance(constraints, dict):
        return _result("NEEDS_CONTEXT", "Primitive capability constraints are malformed")
    for field, condition in constraints.items():
        if not isinstance(field, str):
            return _result("NEEDS_CONTEXT", "Primitive capability field is malformed")
        value: Any = snapshot
        for key in field.split("."):
            if not isinstance(value, dict) or key not in value:
                return _result("NEEDS_CONTEXT", f"Primitive condition {field!r} is unavailable")
            value = value[key]
        if not isinstance(condition, dict):
            return _result("NEEDS_CONTEXT", f"Primitive condition {field!r} is malformed")
        if set(condition) - {"equals", "not_equals", "in", "not_in", "exists"}:
            return _result("NEEDS_CONTEXT", f"Primitive condition {field!r} needs an evaluator")
        if (any(key in condition and not isinstance(condition[key], list) for key in ("in", "not_in"))
                or "exists" in condition and type(condition["exists"]) is not bool):
            return _result("NEEDS_CONTEXT", f"Primitive condition {field!r} is malformed")
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


def _public_condition_support(task: dict, model: dict) -> dict:
    """Separate unsupported conditions and report proven incompatible pairs.

    At most 64 pairs are checked for concise diagnostics. An omitted pair is
    not evidence of support; incomplete pair checks never establish rejection.
    """
    conditions = []
    for field, value in sorted(task['expected_end_state'].items()):
        declaration = model.get('state_variables', {}).get(field, {})
        if not declaration or declaration.get('private') or declaration.get('scope', 'resource') != 'resource':
            continue
        partial = {**task, 'expected_end_state': {field: deepcopy(value)}}
        unknown, supported = not model.get('model_complete', True), False
        start = task.get('expected_start_state') or {}
        changed = field in start and not _equal(start[field], value)
        unknown |= field not in start
        for event in model.get('events', []):
            if changed and field not in event.get('updates', {}):
                continue
            missing = []
            bindings = _successor_bindings(partial, event, model, missing=missing)
            unknown |= bool(missing)
            supported |= bindings is not None
        conditions.append({'field': field, 'actual': deepcopy(value),
                           'supported': True if supported else None if unknown else False})
    unsupported_combinations = []
    start = task.get('expected_start_state') or {}
    for pair in islice(combinations(conditions, 2), 64):
        if not all(row['supported'] is True for row in pair):
            continue
        end = {row['field']: deepcopy(row['actual']) for row in pair}
        changed = {field for field, value in end.items() if field in start and not _equal(start[field], value)}
        if not changed or any(field not in start for field in end):
            continue
        partial = {**task, 'expected_end_state': end}
        unknown, supported = not model.get('model_complete', True), False
        for event in model.get('events', []):
            if not changed.intersection(event.get('updates', {})):
                continue
            missing = []
            bindings = _successor_bindings(partial, event, model, missing=missing)
            unknown |= bool(missing)
            supported |= bindings is not None
        if not unknown and not supported:
            unsupported_combinations.append([{'field': row['field'], 'actual': deepcopy(row['actual'])} for row in pair])
    return {'conditions': conditions,
            'combination_supported': False,
            'individual_conditions_supported': bool(conditions) and all(row['supported'] is True for row in conditions),
            **({'unsupported_combinations': unsupported_combinations} if unsupported_combinations else {}),
            'source': {'descriptor_fingerprint': model.get('descriptor_fingerprint')}}


def _successor_condition_conflicts(task: dict, model: dict) -> list[dict]:
    """Explain public contradictions with the other proposed conditions fixed.

    A diagnostic never selects a transition. All declarations compatible with
    the remaining proposed fields must establish the reported condition.
    Private contributor identities remain in the validation record only.
    """
    end = task["expected_end_state"]
    declarations = model["state_variables"]
    start = task.get("expected_start_state") or {}
    changes = {field for field, declaration in declarations.items()
               if field in end and field in start and declaration.get("private") is not True
               and declaration.get("scope", "resource") == "resource" and not _equal(start[field], end[field])}
    result = []
    for field, declaration in declarations.items():
        if field not in end:
            continue
        proposed = end[field]
        if declaration.get("private") is True or declaration.get("scope", "resource") != "resource":
            continue
        partial = {**task, "expected_end_state": {key: value for key, value in end.items() if key != field}}
        values, sources, unresolved = [], [], False
        for index, event in enumerate(model["events"]):
            # A declaration preserving the intended change cannot explain a
            # conflict. Do not turn its unrelated effect into revision advice.
            if not (changes - {field}).intersection(event.get("updates", {})):
                continue
            missing = []
            bindings = _successor_bindings(partial, event, model, missing=missing)
            if missing:
                unresolved = True
            if bindings is None:
                continue
            update = event.get("updates", {}).get(field, {})
            guard = event.get("guards", {}).get(field, {})
            if "set" in update:
                value = update["set"]
            elif "set_from_param" in update and update["set_from_param"] in bindings:
                value = bindings[update["set_from_param"]]
            elif not update and "equals" in guard:
                value = guard["equals"]
            elif not update and "equals_from_param" in guard and guard["equals_from_param"] in bindings:
                value = bindings[guard["equals_from_param"]]
            else:
                unresolved = True
                continue
            if value is not None and type(value) not in (str, int, float, bool):
                unresolved = True
                continue
            if not any(_equal(value, item) for item in values):
                values.append(deepcopy(value))
            sources.append({"event_index": index, "field": field,
                            "section": "updates" if update else "guards"})
        if unresolved or not values or any(_equal(proposed, item) for item in values):
            continue
        result.append({"field": field, "actual": deepcopy(proposed),
            "condition_operator": "equals" if len(values) == 1 else "in",
            "expected": values[0] if len(values) == 1 else values,
            "evidence_source": "resource declarations supporting requested changes",
            "source": {"descriptor_fingerprint": model.get("descriptor_fingerprint"),
                       "declarations": sources, "fixed_conditions": deepcopy(partial["expected_end_state"])}})
    return result


def validate_primitive_support(  # noqa: C901, PLR0912 - retain contributor-scoped witness audit
    *,
    task: dict[str, Any],
    recovery_des_model: dict[str, Any],
    recovery_snapshot: dict[str, Any],
    validate_primitive: Callable[..., dict[str, Any]],
    parameter_evidence: dict[str, Any] | None = None,
    parameter_witnesses: list[dict[str, Any]] | None = None,
    evaluate_condition: Callable[..., dict[str, Any]] | None = None,
    model_complete: bool | None = None,
    max_parameter_trials: int = 4096,
) -> dict[str, Any]:
    """Require a capability witness for every primitive in the successor union.

    Args:
        task: Grounded candidate with exact expected start and end conditions.
        recovery_des_model: Fresh RA-owned events and primitive contracts.
        recovery_snapshot: Fresh snapshot with accepted prefix effects overlaid.
        validate_primitive: Resource-owned, non-executing physical evaluator.
        parameter_evidence: Resource-observed existential capability witnesses.
            Retained for callers using the historical flat binding format.
        parameter_witnesses: Recorded params scoped by primitive, event_name and
            step_index, retaining source, captured_at and revision metadata.
        evaluate_condition: Owner evaluation of declared optional contributor conditions
            against the same starting facts. No prior primitive effects are used.
        model_complete: Whether the saved resource transition model is complete.
        max_parameter_trials: Shared assignment limit across the entire union.

    Returns:
        Feasibility, the complete primitive union, and attempted witnesses.
        Composition must still establish ordering, compatible bindings, and
        the complete intended successor condition.
    """
    support = derive_primitive_union(task=task, recovery_des_model=recovery_des_model,
                                    model_complete=model_complete, recovery_snapshot=recovery_snapshot)
    snapshot = deepcopy(recovery_snapshot)
    support["snapshot_fingerprint"] = recovery_validation_fingerprint(snapshot)
    support["search"] = {"limit": max_parameter_trials, "parameter_trials": 0, "limit_reached": False}
    if parameter_evidence:
        support["parameter_evidence"] = deepcopy(parameter_evidence)
    if parameter_witnesses is not None:
        support["parameter_witnesses"] = deepcopy(parameter_witnesses)
    if type(max_parameter_trials) is not int or max_parameter_trials < 1:
        return {**_result("NEEDS_CONTEXT", "Parameter search bound must be a positive integer"),
                "primitive_support": support}
    if parameter_evidence is not None and not isinstance(parameter_evidence, dict):
        return {**_result("NEEDS_CONTEXT", "Recorded parameter evidence must be an object"), "primitive_support": support}
    if parameter_witnesses is not None and (not isinstance(parameter_witnesses, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("primitive"), str)
        or not isinstance(row.get("event_name"), str) or type(row.get("step_index")) is not int
        or row["step_index"] < 0 or not isinstance(row.get("params"), dict)
        or not isinstance(row.get("source"), str) or not row["source"].strip()
        for row in parameter_witnesses
    )):
        return {**_result("NEEDS_CONTEXT", "Parameter witnesses require exact primitive/event_name/step_index scope, params and source"),
                "primitive_support": support}
    for witness in parameter_witnesses or []:
        if (witness.get("snapshot_fingerprint", support["snapshot_fingerprint"]) != support["snapshot_fingerprint"]
                or witness.get("model_fingerprint", recovery_validation_fingerprint(recovery_des_model))
                != recovery_validation_fingerprint(recovery_des_model)):
            return {**_result("NEEDS_CONTEXT", "Parameter witness belongs to different observations or declarations"),
                    "primitive_support": support}
    if not support["matching_transitions"]:
        if support["missing_context"] == ["No declared public resource end condition is available"]:
            # Complete declarations establish this as a proposal error. Asking
            # for more observations cannot supply the omitted resource condition.
            return {**_result("INFEASIBLE", "expected_end_state requires a declared public resource condition; "
                              "part conditions alone cannot establish resource primitive support"),
                    "constraint_code": "unsupported_successor_condition", "primitive_support": support}
        result = _result("INFEASIBLE" if support["model_complete"] else "NEEDS_CONTEXT",
                         "No declared resource successor supports a requested change in expected_end_state" if support["model_complete"]
                         else "; ".join(support["missing_context"]))
        if support["model_complete"]:
            result["constraint_code"] = "unsupported_successor_condition"
            conflicts = _successor_condition_conflicts({**task, "expected_start_state": support["expected_start_state"]}, recovery_des_model)
            result["evidence"]["mismatches"] = conflicts
            result["evidence"]["condition_support"] = _public_condition_support(
                {**task, "expected_start_state": support["expected_start_state"]}, recovery_des_model)
            if conflicts:
                result["reason"] += ": " + "; ".join(
                    f"{row['field']} proposed {row['actual']!r}; supported {row['condition_operator']} {row['expected']!r} "
                    "in declarations supporting the requested changes" for row in conflicts)
        elif task.get("resource_jid") or recovery_des_model.get("resource_jid"):
            jid = task.get("resource_jid") or recovery_des_model["resource_jid"]
            result["context_requests"] = []
            if any(reason.startswith("Starting value for ") for reason in support["missing_context"]):
                result["context_requests"].append(f"/resources/{jid}/recovery_snapshot")
            if any(not reason.startswith("Starting value for ") for reason in support["missing_context"]):
                result["context_requests"].append(f"/resources/{jid}/recovery_des_model")
        return {**result, "primitive_support": support}
    catalog = {}
    raw_catalog = recovery_des_model.get("primitive_catalog")
    if not isinstance(raw_catalog, list):
        support["missing_context"].append("Primitive catalog is unavailable")
        raw_catalog = []
    for entry in raw_catalog:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or entry["name"] in catalog:
            support["missing_context"].append("Primitive catalog identities are malformed or duplicated")
            continue
        catalog[entry["name"]] = deepcopy(entry)
    statuses = ["NEEDS_CONTEXT"] if support["missing_support"] or support["missing_context"] else []
    missing_parameters = []
    for row in support["primitives"]:
        name = row["primitive"]
        entry = catalog.get(name)
        attempts: list[dict[str, Any]] = []
        witness = None
        condition_checks = []
        contributors = []
        for contributor in row["contributors"]:
            when = contributor["step"].get("when", [])
            condition = ({"status": "applicable", "reason": "No optional condition is declared"}
                         if when == [] else evaluate_condition(
                             step=deepcopy(contributor["step"]),
                             bindings=deepcopy(contributor["parameter_bindings"]), snapshot=deepcopy(snapshot))
                         if evaluate_condition is not None else
                         {"status": "NEEDS_CONTEXT", "reason": "Declared optional condition requires an owner evaluator"})
            if condition.get("status") not in {"applicable", "not_applicable", "NEEDS_CONTEXT"}:
                condition = {"status": "NEEDS_CONTEXT", "reason": "Owner condition result is unsupported"}
            condition_checks.append({"event_name": contributor["event_name"],
                                     "step_index": contributor["step_index"], **deepcopy(condition)})
            if condition["status"] == "applicable":
                contributors.append(contributor)
            else:
                attempts.append({"event_name": contributor["event_name"], "step_index": contributor["step_index"],
                    **_result("NOT_APPLICABLE" if condition["status"] == "not_applicable" else "NEEDS_CONTEXT",
                              condition["reason"]), "condition": deepcopy(condition)})
        if entry is not None:
            for contributor in contributors:
                recorded = [item for item in parameter_witnesses or [] if item["primitive"] == name
                            and item["event_name"] == contributor["event_name"]
                            and item["step_index"] == contributor["step_index"]
                            and ("parameter_bindings" not in item
                                 or item["parameter_bindings"] == contributor["parameter_bindings"])]
                for evidence in recorded or [None]:
                    bindings = deepcopy(contributor["parameter_bindings"])
                    conflicts = []
                    step = deepcopy(contributor["step"])
                    omitted = evidence.get("omitted_params", []) if evidence is not None else []
                    invalid_omission = (not isinstance(omitted, list)
                        or any(not isinstance(key, str) or key not in (step.get("params") or {})
                               or key in entry.get("required_params", []) or key in evidence["params"] for key in omitted)
                        or bool(omitted) and (evidence.get("source_kind") != "resource_owned_calculation"
                                             or not evidence.get("omission_reason")))
                    for source in (parameter_evidence or {}, evidence["params"] if evidence is not None else {}):
                        for key, value in source.items():
                            if not _bind_parameter(bindings, key, value):
                                conflicts.append(key)
                    if invalid_omission:
                        check = _result("NEEDS_CONTEXT", "Optional parameter omission lacks a resource-owned applicability decision")
                    elif conflicts:
                        check = _result("NEEDS_CONTEXT", "Recorded parameter evidence conflicts with declared bindings",
                                        conflicting_parameters=list(dict.fromkeys(conflicts)))
                    else:
                        for key in omitted:
                            step["params"].pop(key)
                        check = _check_primitive(
                            step=step, entry=entry, bindings=bindings,
                            snapshot=deepcopy(snapshot), validate_primitive=validate_primitive,
                            budget=support["search"],
                        )
                    if evidence is not None:
                        check["parameter_witness"] = deepcopy(evidence)
                    attempts.append({"event_name": contributor["event_name"], "step_index": contributor["step_index"], **check})
                    if check["feasibility_status"] == "FEASIBLE":
                        witness = check
                        break
                if witness is not None:
                    break
        unknown_condition = any(check["status"] == "NEEDS_CONTEXT" for check in condition_checks)
        if unknown_condition:
            if witness is not None:
                row["capability_witness"] = deepcopy(witness)
            witness = _result("NEEDS_CONTEXT", "A declared contributor condition remains unresolved")
        elif not contributors:
            witness = _result("NOT_APPLICABLE", "Every declared contributor condition is proven false")
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
        row["condition_checks"] = condition_checks
        row["composition_contract"] = {
            key: deepcopy((entry or {}).get(key))
            for key in ("preconditions", "effects", "capability_constraints", "required_evidence")
            if key in (entry or {})
        }
        statuses.append(row["feasibility_status"])
        if row["feasibility_status"] == "NEEDS_CONTEXT":
            for attempt in attempts:
                if attempt["feasibility_status"] == "NEEDS_CONTEXT":
                    missing_parameters.append({"primitive": name, "event_name": attempt["event_name"],
                        "step_index": attempt["step_index"], "reason": attempt["reason"],
                        "unresolved_parameters": deepcopy(attempt.get("evidence", {}).get("unresolved_parameters", {}))})
    status = (
        "INFEASIBLE" if "INFEASIBLE" in statuses
        else "NEEDS_CONTEXT" if "NEEDS_CONTEXT" in statuses
        else "FEASIBLE" if support["primitives"]
        else "INFEASIBLE"
    )
    result = {
        **_result(
            status,
            "Every applicable primitive in the successor union has a capability witness"
            if status == "FEASIBLE"
            else f"{status}: the successor union is empty or lacks required capability evidence",
            matching_transitions=deepcopy(support["matching_transitions"]),
            primitive_union=[row["primitive"] for row in support["primitives"]],
            missing_support=deepcopy(support["missing_support"]),
        ),
        "primitive_support": support,
    }
    if status == "INFEASIBLE":
        checks = [deepcopy(checked["result"])
            for row in support["primitives"] if row["feasibility_status"] == "INFEASIBLE"
            for attempt in row["parameter_attempts"]
            for checked in attempt.get("evidence", {}).get("parameter_checks", [])
            if checked.get("result", {}).get("feasibility_status") == "INFEASIBLE"]
        result["evidence"]["capability_result"] = {"feasibility_status": "INFEASIBLE", "checks": [
            {"feasibility_status": row["feasibility_status"], "reason": row.get("reason"),
             **{key: deepcopy(row.get("evidence", {})[key]) for key in
                ("checked_pose", "gripper_reach", "pose_frame") if key in row.get("evidence", {})}}
            for row in checks]}
        reasons = list(dict.fromkeys(
            row["reason"] for row in checks if isinstance(row.get("reason"), str) and row["reason"]
        ))
        if reasons:
            result["reason"] = "INFEASIBLE: " + "; ".join(reasons)
    if status == "NEEDS_CONTEXT":
        result["missing_parameters"] = missing_parameters
        jid = task.get("resource_jid") or recovery_des_model.get("resource_jid")
        if jid:
            refs = []
            if support["missing_context"] or support["missing_support"] or any(name not in catalog for name in (row["primitive"] for row in support["primitives"])):
                refs.append(f"/resources/{jid}/recovery_des_model")
            if missing_parameters:
                refs.append(f"/resources/{jid}/parameter_witnesses")
            result["context_requests"] = refs
    return result


def _check_primitive(
    *,
    step: dict[str, Any],
    entry: dict[str, Any],
    bindings: dict[str, Any],
    snapshot: dict[str, Any],
    validate_primitive: Callable[..., dict[str, Any]],
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    condition_failure = _precondition_result(entry, snapshot)
    if condition_failure is not None:
        return condition_failure
    if (not isinstance(step.get("params", {}), dict) or not isinstance(entry.get("params", {}), dict)
            or any(not isinstance(row, dict) for row in entry.get("params", {}).values())):
        return _result("NEEDS_CONTEXT", "Primitive parameter declarations must be objects")
    for name, value in step.get("params", {}).items():
        known, resolved = _bound_value(value, bindings)
        if known and name in bindings and not _equal(resolved, bindings[name]):
            return _result("NEEDS_CONTEXT", "Recorded parameter evidence conflicts with authored primitive binding",
                           conflicting_parameters=[name])
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
    budget = budget if budget is not None else {"limit": 4096, "parameter_trials": 0, "limit_reached": False}
    remaining = budget["limit"] - budget["parameter_trials"]
    if remaining <= 0:
        budget["limit_reached"] = True
        return _result("NEEDS_CONTEXT", "Shared primitive parameter search reached its limit")
    assignments, complete = _parameter_assignments(step, entry, bindings, limit=remaining)
    unknown = not complete
    findings = []
    for params in assignments:
        budget["parameter_trials"] += 1
        errors = [error.message for error in validator.iter_errors(params)]
        if errors:
            findings.append({"params": params, "errors": errors})
            continue
        result = validate_primitive(primitive=entry["name"], params=deepcopy(params)) if callable(validate_primitive) else None
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
    if not complete and budget["parameter_trials"] >= budget["limit"]:
        budget["limit_reached"] = True
    unresolved = {name: deepcopy(value) for name, value in step.get("params", {}).items()
                  if not _bound_value(value, bindings)[0] and name not in bindings}
    for name in entry.get("required_params", []):
        if name not in bindings and name not in step.get("params", {}) and not any(
            key in entry.get("params", {}).get(name, {}) for key in ("const", "enum", "default")
        ):
            unresolved[name] = {"required": True}
    return _result(
        "NEEDS_CONTEXT" if unknown else "INFEASIBLE",
        "No satisfying primitive parameter assignment was established",
        parameter_checks=findings,
        unresolved_parameters=unresolved,
    )


__all__ = ["derive_primitive_union", "validate_primitive_support", "validate_recovery_transition"]
