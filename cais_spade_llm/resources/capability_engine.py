"""Private configuration-driven capability evaluation for Resource Agents."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_GUARD_OPERATORS = {
    "equals",
    "equals_from_param",
    "exists",
    "in",
    "not_equals",
    "not_equals_from_param",
}
_UPDATE_OPERATORS = {
    "set",
    "set_from_param",
    "set_from_param_any_of",
}
_SINGLE_RUNTIME_FACT_OPERATORS = {
    "equals_from_param",
    "not_equals_from_param",
    "set_from_param",
}

_COMPOUND_STATE_FIELDS = {
    "resource_state": ("resource_state", "resource_location"),
    "part_state": ("part_state", "part_location"),
}


def _exact_value_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def compound_recovery_state(state: dict[str, Any] | None) -> dict[str, Any]:
    """Pack configured state/location fields into recovery compound states."""
    flat_state = deepcopy(dict(state or {}))
    compound: dict[str, Any] = {
        str(field_name): deepcopy(value)
        for field_name, value in flat_state.items()
        if str(field_name)
        not in {
            "resource_state",
            "resource_location",
            "part_state",
            "part_location",
        }
    }
    for compound_field, (condition_field, location_field) in (
        _COMPOUND_STATE_FIELDS.items()
    ):
        condition_present = condition_field in flat_state
        location_present = location_field in flat_state
        if not condition_present and not location_present:
            continue
        compound_value: dict[str, Any] = {}
        if condition_present:
            compound_value["condition"] = deepcopy(
                flat_state.get(condition_field)
            )
        if location_present:
            compound_value["location"] = deepcopy(
                flat_state.get(location_field)
            )
        compound[compound_field] = compound_value
    return compound


def configured_recovery_state(state: dict[str, Any] | None) -> dict[str, Any]:
    """Unpack recovery compound states into configured atomic fields."""
    compound_state = deepcopy(dict(state or {}))
    flat: dict[str, Any] = {}
    for raw_field_name, value in compound_state.items():
        field_name = str(raw_field_name)
        component_fields = _COMPOUND_STATE_FIELDS.get(field_name)
        if component_fields is None or not isinstance(value, dict):
            flat[field_name] = deepcopy(value)
            continue
        condition_field, location_field = component_fields
        components = dict(value)
        if "condition" in components:
            flat[condition_field] = deepcopy(components.get("condition"))
        if "location" in components:
            flat[location_field] = deepcopy(components.get("location"))
    return flat


def compound_state_values(
    state_values: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pack known condition and location values for prompt publication."""
    flat_values = deepcopy(dict(state_values or {}))
    compound: dict[str, Any] = {
        str(field_name): deepcopy(value)
        for field_name, value in flat_values.items()
        if str(field_name)
        not in {
            "resource_state",
            "resource_location",
            "part_state",
            "part_location",
        }
    }
    for compound_field, (condition_field, location_field) in (
        _COMPOUND_STATE_FIELDS.items()
    ):
        if condition_field not in flat_values and location_field not in flat_values:
            continue
        compound_value: dict[str, Any] = {}
        if condition_field in flat_values:
            compound_value["condition"] = deepcopy(
                flat_values.get(condition_field) or []
            )
        if location_field in flat_values:
            compound_value["location"] = deepcopy(
                flat_values.get(location_field) or []
            )
        compound[compound_field] = compound_value
    return compound


def configured_capability_errors(
    capabilities: dict[str, Any] | None,
    *,
    executable_names: set[str] | None = None,
    runtime_fact_names: set[str] | None = None,
    executable_required_facts: dict[str, set[str]] | None = None,
) -> list[str]:
    """Return errors in one resource's configured transition capabilities."""
    raw_capabilities = dict(capabilities or {})
    raw_state_variables = raw_capabilities.get("state_variables")
    raw_events = raw_capabilities.get("events")
    errors: list[str] = []
    if not isinstance(raw_state_variables, dict) or not raw_state_variables:
        return ["static_capabilities.state_variables must be a nonempty object"]
    if not isinstance(raw_events, list) or not raw_events:
        return ["static_capabilities.events must be a nonempty array"]
    supported_runtime_facts = set(runtime_fact_names or set())
    state_fields = {
        str(field_name)
        for field_name in raw_state_variables
        if str(field_name)
    }
    for raw_field_name, raw_declaration in raw_state_variables.items():
        field_name = str(raw_field_name or "")
        if not field_name or not isinstance(raw_declaration, dict):
            errors.append(
                "static_capabilities state variable declarations must be objects"
            )
            continue
        scope = str(raw_declaration.get("scope") or "resource")
        if scope not in {"resource", "part"}:
            errors.append(
                f"static_capabilities.state_variables.{field_name}.scope must be "
                "'resource' or 'part'"
            )
        domain = raw_declaration.get("domain")
        if not isinstance(domain, list) or not domain:
            errors.append(
                f"static_capabilities.state_variables.{field_name}.domain must be "
                "a nonempty array"
            )
        elif any(
            any(
                _exact_value_equal(value, previous)
                for previous in domain[:value_index]
            )
            for value_index, value in enumerate(domain)
        ):
            errors.append(
                f"static_capabilities.state_variables.{field_name}.domain must "
                "not contain duplicate values"
            )
        parameter_values = raw_declaration.get("parameter_values")
        if parameter_values is not None and (
            not isinstance(parameter_values, list)
            or not all(
                isinstance(parameter_name, str) and parameter_name
                for parameter_name in parameter_values
            )
        ):
            errors.append(
                f"static_capabilities.state_variables.{field_name}."
                "parameter_values must be a string array"
            )

    seen_event_names: set[str] = set()
    for event_index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            errors.append(
                f"static_capabilities.events[{event_index}] must be an object"
            )
            continue
        event_name = str(raw_event.get("event_name") or "")
        if not event_name:
            errors.append(
                f"static_capabilities.events[{event_index}].event_name must be nonempty"
            )
            continue
        if event_name in seen_event_names:
            errors.append(
                f"static_capabilities event '{event_name}' is duplicated"
            )
        seen_event_names.add(event_name)
        if executable_names is not None and event_name not in executable_names:
            errors.append(
                f"static_capabilities event '{event_name}' has no executable function"
            )
        if "parameter_bindings" in raw_event:
            errors.append(
                f"static_capabilities event '{event_name}' must not declare "
                "parameter_bindings"
            )
        requires_part_binding = raw_event.get("requires_part_binding")
        if requires_part_binding is not None and not isinstance(
            requires_part_binding,
            bool,
        ):
            errors.append(
                f"static_capabilities event '{event_name}' "
                "requires_part_binding must be boolean"
            )
        elif (
            requires_part_binding is True
            and runtime_fact_names is not None
            and "part_name" not in supported_runtime_facts
        ):
            errors.append(
                f"static_capabilities event '{event_name}' requires unsupported "
                "runtime fact 'part_name'"
            )
        for required_fact in sorted(
            dict(executable_required_facts or {}).get(event_name, set())
        ):
            if (
                runtime_fact_names is not None
                and required_fact not in supported_runtime_facts
            ):
                errors.append(
                    f"static_capabilities event '{event_name}' executable requires "
                    f"unsupported runtime fact '{required_fact}'"
                )

        for boolean_field in ("controllable", "observable"):
            value = raw_event.get(boolean_field)
            if value is not None and not isinstance(value, bool):
                errors.append(
                    f"static_capabilities event '{event_name}' "
                    f"{boolean_field} must be boolean"
                )
        for section_name, supported_operators in (
            ("guards", _GUARD_OPERATORS),
            ("updates", _UPDATE_OPERATORS),
        ):
            section = raw_event.get(section_name) or {}
            if not isinstance(section, dict):
                errors.append(
                    f"static_capabilities event '{event_name}' "
                    f"{section_name} must be an object"
                )
                continue
            for raw_field_name, raw_expression in section.items():
                field_name = str(raw_field_name or "")
                if field_name not in state_fields:
                    errors.append(
                        f"static_capabilities event '{event_name}' {section_name} "
                        f"references undeclared field '{field_name}'"
                    )
                if not isinstance(raw_expression, dict) or len(raw_expression) != 1:
                    errors.append(
                        f"static_capabilities event '{event_name}' "
                        f"{section_name}.{field_name} must contain one operator"
                    )
                    continue
                operator = str(next(iter(raw_expression)))
                if operator not in supported_operators:
                    errors.append(
                        f"static_capabilities event '{event_name}' "
                        f"{section_name}.{field_name} uses unsupported operator "
                        f"'{operator}'"
                    )
                    continue
                operand = raw_expression.get(operator)
                if operator == "exists" and operand is not True:
                    errors.append(
                        f"static_capabilities event '{event_name}' "
                        f"{section_name}.{field_name}.exists must be true"
                    )
                elif operator == "in" and (
                    not isinstance(operand, list) or not operand
                ):
                    errors.append(
                        f"static_capabilities event '{event_name}' "
                        f"{section_name}.{field_name}.in must be a nonempty array"
                    )
                elif operator in _SINGLE_RUNTIME_FACT_OPERATORS and not (
                    isinstance(operand, str) and operand
                ):
                    errors.append(
                        f"static_capabilities event '{event_name}' "
                        f"{section_name}.{field_name}.{operator} must be a string"
                    )
                elif operator == "set_from_param_any_of" and (
                    not isinstance(operand, list)
                    or not operand
                    or not all(
                        isinstance(parameter_name, str) and parameter_name
                        for parameter_name in operand
                    )
                ):
                    errors.append(
                        f"static_capabilities event '{event_name}' "
                        f"{section_name}.{field_name}.{operator} must be a "
                        "nonempty string array"
                    )

                referenced_facts = (
                    [operand]
                    if operator in _SINGLE_RUNTIME_FACT_OPERATORS
                    and isinstance(operand, str)
                    else list(operand)
                    if operator == "set_from_param_any_of"
                    and isinstance(operand, list)
                    else []
                )
                for runtime_fact_name in referenced_facts:
                    if (
                        runtime_fact_names is not None
                        and runtime_fact_name not in supported_runtime_facts
                    ):
                        errors.append(
                            f"static_capabilities event '{event_name}' "
                            f"{section_name}.{field_name}.{operator} references "
                            f"unsupported runtime fact '{runtime_fact_name}'"
                        )

                declaration = dict(
                    raw_state_variables.get(field_name) or {}
                )
                domain = declaration.get("domain") or []
                if operator in {"equals", "not_equals", "set"} and not any(
                    _exact_value_equal(operand, domain_value)
                    for domain_value in domain
                ):
                    errors.append(
                        f"static_capabilities event '{event_name}' "
                        f"{section_name}.{field_name}.{operator} uses a value "
                        "outside the declared domain"
                    )
                elif operator == "in" and any(
                    not any(
                        _exact_value_equal(candidate, domain_value)
                        for domain_value in domain
                    )
                    for candidate in (operand or [])
                ):
                    errors.append(
                        f"static_capabilities event '{event_name}' "
                        f"{section_name}.{field_name}.in uses a value outside "
                        "the declared domain"
                    )

    return errors


def bind_configured_capabilities(
    resource_agent: Any,
    *,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one Resource Agent's startup-loaded capability declarations."""
    if resource_agent is None:
        return {}
    static_capabilities = deepcopy(
        getattr(resource_agent, "static_capabilities", {}) or {}
    )
    raw_capabilities = deepcopy(
        capabilities
        if isinstance(capabilities, dict)
        else {
            "state_variables": static_capabilities.get("state_variables"),
            "events": static_capabilities.get("events"),
        }
    )
    if not isinstance(raw_capabilities.get("state_variables"), dict):
        return {}
    if not isinstance(raw_capabilities.get("events"), list):
        return {}
    return {
        "resource_jid": str(getattr(resource_agent, "jid", "") or "").strip(),
        "state_variables": deepcopy(raw_capabilities["state_variables"]),
        "events": deepcopy(raw_capabilities["events"]),
    }


def capability_value_in_domain(
    value: Any,
    declaration: dict[str, Any],
    *,
    runtime_facts: dict[str, Any] | None = None,
    runtime_fact_tables: list[dict[str, Any]] | None = None,
) -> bool:
    """Return whether an exact value belongs to a configured dynamic domain."""
    if any(
        _exact_value_equal(value, domain_value)
        for domain_value in (declaration.get("domain") or [])
    ):
        return True
    fact_tables = [
        dict(row)
        for row in (
            runtime_fact_tables
            if runtime_fact_tables is not None
            else [runtime_facts or {}]
        )
        if isinstance(row, dict)
    ]
    return any(
        runtime_fact_name in fact_table
        and _exact_value_equal(value, fact_table.get(runtime_fact_name))
        for runtime_fact_name in (declaration.get("parameter_values") or [])
        for fact_table in fact_tables
    )


def capability_state_mismatches(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return exact field mismatches between two capability valuations."""
    return [
        {
            "field": str(field_name),
            "expected": deepcopy(expected_value),
            "actual": deepcopy(actual.get(str(field_name))),
        }
        for field_name, expected_value in expected.items()
        if not _exact_value_equal(
            expected_value,
            actual.get(str(field_name)),
        )
    ]


def capability_state_error(
    capabilities: dict[str, Any],
    *,
    state_name: str,
    state: dict[str, Any],
    runtime_facts: dict[str, Any] | None = None,
    runtime_fact_tables: list[dict[str, Any]] | None = None,
    current_valuation: dict[str, Any] | None = None,
    allow_generated_values: bool = False,
) -> dict[str, Any] | None:
    """Return the first undeclared field or out-of-domain state error."""
    state_variables = dict(capabilities.get("state_variables") or {})
    current_state = dict(current_valuation or {})
    undeclared_fields = sorted(
        str(field_name)
        for field_name in state
        if str(field_name) not in state_variables
    )
    if undeclared_fields:
        return {
            "allowed": False,
            "constraint_code": "disallowed_outline_state_field",
            "reason": "candidate uses fields not declared by the Resource Agent",
            "evidence": {"state_fields": undeclared_fields},
        }
    if allow_generated_values:
        return None
    for field_name, value in state.items():
        declaration = dict(state_variables.get(str(field_name)) or {})
        if capability_value_in_domain(
            value,
            declaration,
            runtime_facts=runtime_facts,
            runtime_fact_tables=runtime_fact_tables,
        ):
            continue
        if (
            declaration.get("parameter_values")
            and str(field_name) in current_state
            and _exact_value_equal(
                value,
                current_state.get(str(field_name)),
            )
        ):
            continue
        return {
            "allowed": False,
            "constraint_code": "state_value_outside_ra_domain",
            "reason": (
                f"{state_name}.{field_name} is outside the configured "
                "Resource Agent domain"
            ),
            "evidence": {
                "field": f"{state_name}.{field_name}",
                "value": deepcopy(value),
            },
        }
    return None


def configured_event_runtime_fact_names(
    event: dict[str, Any],
) -> set[str]:
    """Return exact fact names that one event requires simultaneously."""
    required: set[str] = set()
    for section_name in ("guards", "updates"):
        for raw_expression in dict(
            event.get(section_name) or {}
        ).values():
            expression = dict(raw_expression or {})
            for operator in _SINGLE_RUNTIME_FACT_OPERATORS:
                runtime_fact_name = expression.get(operator)
                if isinstance(runtime_fact_name, str) and runtime_fact_name:
                    required.add(runtime_fact_name)
    return required


def configured_event_successor(
    event: dict[str, Any],
    valuation: dict[str, Any],
    *,
    state_variables: dict[str, Any],
    runtime_facts: dict[str, Any],
    required_runtime_facts: set[str] | None = None,
    evaluate_guards: bool = True,
) -> dict[str, Any]:
    """Calculate one configured event successor using exact-name substitution."""
    missing_runtime_facts = sorted(
        runtime_fact_name
        for runtime_fact_name in (required_runtime_facts or set())
        if runtime_facts.get(runtime_fact_name) in (None, "")
    )
    if missing_runtime_facts:
        return {
            "enabled": False,
            "constraint_code": "runtime_fact_unavailable",
            "reason": "configured capability runtime facts are unavailable",
            "missing_runtime_facts": missing_runtime_facts,
        }

    guard_items = (
        dict(event.get("guards") or {}).items()
        if evaluate_guards
        else ()
    )
    for raw_field_name, raw_expression in guard_items:
        field_name = str(raw_field_name)
        expression = dict(raw_expression or {})
        actual = valuation.get(field_name)
        if "equals" in expression:
            satisfied = _exact_value_equal(actual, expression.get("equals"))
        elif "not_equals" in expression:
            satisfied = not _exact_value_equal(
                actual,
                expression.get("not_equals"),
            )
        elif "in" in expression:
            satisfied = any(
                _exact_value_equal(actual, expected)
                for expected in (expression.get("in") or [])
            )
        elif expression.get("exists") is True:
            satisfied = actual not in (None, "")
        elif "equals_from_param" in expression:
            runtime_fact_name = str(
                expression.get("equals_from_param") or ""
            )
            satisfied = (
                runtime_facts.get(runtime_fact_name) not in (None, "")
                and _exact_value_equal(
                    actual,
                    runtime_facts.get(runtime_fact_name),
                )
            )
        elif "not_equals_from_param" in expression:
            runtime_fact_name = str(
                expression.get("not_equals_from_param") or ""
            )
            satisfied = (
                runtime_facts.get(runtime_fact_name) not in (None, "")
                and not _exact_value_equal(
                    actual,
                    runtime_facts.get(runtime_fact_name),
                )
            )
        else:
            satisfied = False
        if not satisfied:
            return {
                "enabled": False,
                "constraint_code": "unsatisfied_guard_predicate",
                "reason": "configured capability guards are not satisfied",
                "guard_field": field_name,
            }

    successor = deepcopy(valuation)
    for raw_field_name, raw_update in dict(event.get("updates") or {}).items():
        field_name = str(raw_field_name)
        update = dict(raw_update or {})
        if "set" in update:
            successor[field_name] = deepcopy(update.get("set"))
            continue
        if "set_from_param" in update:
            runtime_fact_name = str(update.get("set_from_param") or "")
            if runtime_facts.get(runtime_fact_name) in (None, ""):
                return {
                    "enabled": False,
                    "constraint_code": "runtime_fact_unavailable",
                    "reason": "configured capability runtime facts are unavailable",
                    "missing_runtime_facts": [runtime_fact_name],
                }
            successor[field_name] = deepcopy(
                runtime_facts.get(runtime_fact_name)
            )
            continue
        runtime_fact_names = [
            str(runtime_fact_name)
            for runtime_fact_name in (
                update.get("set_from_param_any_of") or []
            )
            if str(runtime_fact_name)
        ]
        selected_runtime_fact = next(
            (
                deepcopy(runtime_facts.get(runtime_fact_name))
                for runtime_fact_name in runtime_fact_names
                if runtime_facts.get(runtime_fact_name) not in (None, "")
            ),
            None,
        )
        if selected_runtime_fact is None:
            return {
                "enabled": False,
                "constraint_code": "runtime_fact_unavailable",
                "reason": "configured capability runtime facts are unavailable",
                "missing_runtime_facts": runtime_fact_names,
            }
        successor[field_name] = selected_runtime_fact

    for raw_field_name in dict(event.get("updates") or {}):
        field_name = str(raw_field_name)
        declaration = dict(state_variables.get(field_name) or {})
        value = successor.get(field_name)
        if declaration and capability_value_in_domain(
            value,
            declaration,
            runtime_facts=runtime_facts,
        ):
            continue
        return {
            "enabled": False,
            "constraint_code": "state_value_outside_ra_domain",
            "reason": (
                f"configured capability successor field '{field_name}' "
                "is outside the configured Resource Agent domain"
            ),
            "evidence": {
                "field": field_name,
                "value": deepcopy(value),
            },
        }
    return {
        "enabled": True,
        "constraint_code": "",
        "reason": "configured capability guards are satisfied",
        "successor": successor,
    }


__all__ = [
    "bind_configured_capabilities",
    "capability_state_error",
    "capability_state_mismatches",
    "capability_value_in_domain",
    "compound_recovery_state",
    "compound_state_values",
    "configured_recovery_state",
    "configured_capability_errors",
    "configured_event_runtime_fact_names",
    "configured_event_successor",
]
