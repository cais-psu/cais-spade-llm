"""Derive a resource-owned transition from declared primitive contracts."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

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


