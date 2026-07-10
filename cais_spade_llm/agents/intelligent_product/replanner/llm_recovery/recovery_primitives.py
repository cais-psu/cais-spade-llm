"""Recovery primitive-program validation and projection helpers.

Resource and robot primitive catalogs are owned by ``cais_spade_llm.resources``.
This module only handles recovery-authored primitive programs: context reference
resolution, semantic projection, event facts, output extraction, and validation.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from cais_spade_llm.resources.resource_profile import (
    get_resource_profile,
    resource_event_fact_key,
    resource_snapshot_field_value,
    resource_snapshot_set_field,
)

_MISSING = object()
_KNOWN_CONTEXT_ROOTS = {"resource", "resources", "parts", "recovery_resources", "event_facts"}


def _normalized_symbol(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    token = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", token)
    token = token.replace("-", "_").replace(" ", "_")
    return token.strip("_").lower()


def _mapping_lookup(mapping: dict[str, Any], token: str) -> tuple[Any, bool]:
    if token in mapping:
        return mapping[token], True
    normalized_token = _normalized_symbol(token)
    for key, value in mapping.items():
        if _normalized_symbol(key) == normalized_token:
            return value, True
    return None, False


def _list_index(sequence: list[Any], token: str) -> tuple[Any, bool]:
    try:
        index = int(token)
    except (TypeError, ValueError):
        return None, False
    if index < 0 or index >= len(sequence):
        return None, False
    return sequence[index], True


def _iter_ref_tokens(ref: str) -> list[str]:
    text = str(ref or "").strip()
    if not text:
        return []
    if text.startswith("/"):
        return [token for token in text.split("/") if token]
    return [token for token in text.split(".") if token]


def _is_event_fact_ref(ref: str, grounding_context: dict[str, Any] | None = None) -> bool:
    text = str(ref or "").strip()
    if not text:
        return False
    if text.startswith("/event_facts/"):
        return True
    tokens = _iter_ref_tokens(text)
    if not tokens:
        return False
    first = _normalized_symbol(tokens[0])
    if first == "event_facts":
        return True
    roots = {
        _normalized_symbol(key) for key in dict(grounding_context or {}).keys() if str(key).strip()
    }
    return first not in roots and first not in _KNOWN_CONTEXT_ROOTS and len(tokens) > 1


def resolve_context_ref(
    ref: str,
    grounding_context: dict[str, Any] | None = None,
    *,
    event_facts: dict[str, Any] | None = None,
    step_outputs: dict[str, Any] | None = None,
) -> Any:
    """Resolve a dotted or JSON-pointer-like path against grounding context."""
    tokens = _iter_ref_tokens(ref)
    if not tokens:
        raise KeyError("empty context_ref")

    combined = deepcopy(dict(grounding_context or {}))
    existing_event_facts = dict(combined.get("event_facts") or {})
    if event_facts:
        existing_event_facts.update(deepcopy(event_facts))
    if step_outputs:
        existing_event_facts.update(deepcopy(step_outputs))
    combined["event_facts"] = existing_event_facts

    first = tokens[0]
    current: Any = _MISSING
    if _normalized_symbol(first) == "event_facts":
        current = combined["event_facts"]
        tokens = tokens[1:]
    else:
        current, found = _mapping_lookup(combined, first)
        if not found:
            current, found = _mapping_lookup(combined["event_facts"], first)
            if not found:
                raise KeyError(f"context_ref root '{first}' was not found")
        tokens = tokens[1:]

    for token in tokens:
        if isinstance(current, dict):
            current, found = _mapping_lookup(current, token)
            if not found:
                raise KeyError(f"context_ref token '{token}' was not found in object")
            continue
        if isinstance(current, list):
            current, found = _list_index(current, token)
            if not found:
                raise KeyError(f"context_ref token '{token}' was not a valid list index")
            continue
        raise KeyError(f"context_ref token '{token}' could not be resolved from scalar")
    return deepcopy(current)


def resolve_param_refs(
    value: Any,
    grounding_context: dict[str, Any] | None = None,
    *,
    event_facts: dict[str, Any] | None = None,
    preserve_event_fact_refs: bool = False,
    step_outputs: dict[str, Any] | None = None,
    preserve_step_output_refs: bool = False,
) -> Any:
    """Resolve ``context_ref`` dictionaries and reference-like strings."""
    preserve_event_fact_refs = preserve_event_fact_refs or preserve_step_output_refs
    if isinstance(value, dict):
        if set(value.keys()) == {"context_ref"}:
            ref = str(value.get("context_ref") or "").strip()
            if not ref:
                return None
            if preserve_event_fact_refs and _is_event_fact_ref(ref, grounding_context):
                return deepcopy(value)
            return resolve_context_ref(
                ref,
                grounding_context,
                event_facts=event_facts,
                step_outputs=step_outputs,
            )
        return {
            str(key): resolve_param_refs(
                item,
                grounding_context,
                event_facts=event_facts,
                step_outputs=step_outputs,
                preserve_event_fact_refs=preserve_event_fact_refs,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            resolve_param_refs(
                item,
                grounding_context,
                event_facts=event_facts,
                step_outputs=step_outputs,
                preserve_event_fact_refs=preserve_event_fact_refs,
            )
            for item in value
        ]
    if isinstance(value, str):
        ref = value.strip()
        if not ref:
            return value
        if preserve_event_fact_refs and _is_event_fact_ref(ref, grounding_context):
            return value
        try:
            return resolve_context_ref(
                ref,
                grounding_context,
                event_facts=event_facts,
                step_outputs=step_outputs,
            )
        except Exception:
            return value
    return deepcopy(value)


def resolve_step_param_refs(
    steps: list[dict[str, Any]],
    grounding_context: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    """Resolve planner-known context refs while preserving step-output refs."""
    resolved_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        try:
            normalized = dict(step)
            normalized["params"] = resolve_param_refs(
                step.get("params") or {},
                grounding_context,
                preserve_step_output_refs=True,
            )
            resolved_steps.append(normalized)
        except Exception as exc:
            primitive = str(step.get("primitive", "")).strip()
            return [], f"step {index} {primitive}: {exc}"
    return resolved_steps, None


def _collect_context_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        if set(value.keys()) == {"context_ref"}:
            ref = str(value.get("context_ref") or "").strip()
            return [ref] if ref else []
        for item in value.values():
            refs.extend(_collect_context_refs(item))
        return refs
    if isinstance(value, list):
        for item in value:
            refs.extend(_collect_context_refs(item))
        return refs
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("/") or text.startswith("event_facts."):
            return [text]
    return refs


def _assign_nested_mapping(target: dict[str, Any], path: str, value: Any) -> None:
    tokens = [token for token in str(path or "").split(".") if token]
    if not tokens:
        return
    current = target
    for token in tokens[:-1]:
        child = current.get(token)
        if not isinstance(child, dict):
            child = {}
            current[token] = child
        current = child
    current[tokens[-1]] = deepcopy(value)


def event_fact_key_for_primitive(
    *,
    primitive: str,
    params: dict[str, Any],
    resource_type: str = "",
) -> tuple[str | None, str | None]:
    profile = get_resource_profile(resource_type or _infer_resource_type_for_primitive(primitive))
    return resource_event_fact_key(profile, primitive, params)


def expand_composite_steps(
    steps: list[dict[str, Any]] | None,
    primitive_catalog: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Validate the incoming step shape while keeping v4 expansion as pass-through."""
    names = {
        str(entry.get("name") or "").strip()
        for entry in (primitive_catalog or [])
        if isinstance(entry, dict) and str(entry.get("name") or "").strip()
    }
    expanded: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps or []):
        if not isinstance(raw_step, dict):
            raise TypeError(f"primitive step {index} must be an object")
        if raw_step.get("steps") or raw_step.get("primitive_steps"):
            raise ValueError("composite step expansion is not supported in v4 yet")
        primitive = str(raw_step.get("primitive") or "").strip()
        if not primitive:
            raise ValueError(f"primitive step {index} is missing 'primitive'")
        if names and primitive not in names:
            raise ValueError(f"unknown primitive '{primitive}'")
        expanded.append(
            {
                "primitive": primitive,
                "params": deepcopy(raw_step.get("params") or {}),
            }
        )
    return expanded


def _effect_value_from_params(params: dict[str, Any], param_name: str) -> Any:
    value = params.get(str(param_name))
    return deepcopy(value)


def _apply_effect_spec(
    field: str,
    effect_spec: dict[str, Any],
    params: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    profile: Any,
) -> dict[str, Any]:
    updated = deepcopy(snapshot)
    if not isinstance(effect_spec, dict):
        return updated

    if "set" in effect_spec:
        return resource_snapshot_set_field(
            updated,
            field,
            deepcopy(effect_spec.get("set")),
            profile=profile,
        )
    if effect_spec.get("set_unknown"):
        return resource_snapshot_set_field(updated, field, None, profile=profile)
    if "set_from_param" in effect_spec:
        param_name = str(effect_spec.get("set_from_param") or "").strip()
        return resource_snapshot_set_field(
            updated,
            field,
            _effect_value_from_params(params, param_name),
            profile=profile,
        )
    if "set_from_param_any_of" in effect_spec:
        for candidate in effect_spec.get("set_from_param_any_of") or []:
            value = _effect_value_from_params(params, str(candidate))
            if value not in (None, ""):
                return resource_snapshot_set_field(updated, field, value, profile=profile)
        return updated
    if "pose_absolute_from_params" in effect_spec:
        keys = list(effect_spec.get("pose_absolute_from_params") or [])
        pose = {}
        for axis in keys[:3]:
            axis_name = str(axis or "").strip()
            if axis_name:
                pose[axis_name] = deepcopy(params.get(axis_name))
        return resource_snapshot_set_field(updated, field, pose, profile=profile)
    if "pose_relative_from_params" in effect_spec:
        base_pose = dict(resource_snapshot_field_value(updated, field, profile=profile) or {})
        keys = list(effect_spec.get("pose_relative_from_params") or [])
        x = float(base_pose.get("x", 0.0) or 0.0) + float(params.get(str(keys[0]), 0.0) or 0.0)
        y = float(base_pose.get("y", 0.0) or 0.0) + float(params.get(str(keys[1]), 0.0) or 0.0)
        z = float(base_pose.get("z", 0.0) or 0.0) + float(params.get(str(keys[2]), 0.0) or 0.0)
        return resource_snapshot_set_field(
            updated, field, {"x": x, "y": y, "z": z}, profile=profile
        )
    return updated


def _refresh_canonical_mirrors(snapshot: dict[str, Any]) -> dict[str, Any]:
    refreshed = deepcopy(snapshot or {})
    resource_core = dict(refreshed.get("resource_core") or {})
    for field in (
        "resource_jid",
        "resource_type",
        "current_state",
        "current_location",
        "availability",
        "active_work",
        "occupancy",
    ):
        if field in resource_core:
            refreshed[field] = deepcopy(resource_core.get(field))
    for facet_values in (refreshed.get("resource_facets") or {}).values():
        if not isinstance(facet_values, dict):
            continue
        for field, value in facet_values.items():
            refreshed[field] = deepcopy(value)
    return refreshed


def apply_effects_to_snapshot(
    step: dict[str, Any],
    primitive_meta: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Apply semantic effect frontmatter onto a recovery snapshot."""
    updated = deepcopy(snapshot or {})
    params = dict(step.get("params") or {})
    profile = get_resource_profile(
        str(
            dict(updated.get("resource_core") or {}).get("resource_type")
            or updated.get("resource_type")
            or "resource"
        )
    )
    effects = dict(primitive_meta.get("effects") or {})
    for field, effect_spec in effects.items():
        updated = _apply_effect_spec(
            str(field), dict(effect_spec or {}), params, updated, profile=profile
        )
    return _refresh_canonical_mirrors(updated)


def preview_step_output(
    *,
    primitive: str,
    params: dict[str, Any],
    snapshot: dict[str, Any] | None = None,
    grounding_context: dict[str, Any] | None = None,
    resource_type: str = "",
) -> tuple[dict[str, Any] | None, str | None]:
    """Preview an observation/generation output from params and current context."""
    profile = get_resource_profile(resource_type or _infer_resource_type_for_primitive(primitive))
    resolver = dict(profile.preview_output_map or {}).get(str(primitive or "").strip())
    if resolver is None:
        return None, f"primitive '{primitive}' does not define preview output"
    return resolver(
        deepcopy(params or {}),
        deepcopy(snapshot or {}),
        deepcopy(grounding_context or {}),
    )


def _infer_resource_type_for_primitive(primitive: str) -> str:
    target = str(primitive or "").strip()
    if not target:
        return "resource"
    for resource_type in ("robot", "printer", "resource"):
        profile = get_resource_profile(resource_type)
        if target in dict(profile.extract_output_map or {}) or target in dict(
            profile.preview_output_map or {}
        ):
            return resource_type
    return "resource"


def extract_step_output(
    *,
    primitive: str,
    params: dict[str, Any],
    step_result: dict[str, Any],
    resource_type: str = "",
) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize stored step outputs into deterministic recovery context payloads."""
    profile = get_resource_profile(resource_type or _infer_resource_type_for_primitive(primitive))
    resolver = dict(profile.extract_output_map or {}).get(str(primitive or "").strip())
    if resolver is not None:
        return resolver(deepcopy(params or {}), deepcopy(step_result or {}))

    normalized_result = dict(step_result or {})
    if isinstance(normalized_result.get("observation"), dict):
        return deepcopy(normalized_result.get("observation")), None
    if isinstance(normalized_result.get("data"), dict):
        return deepcopy(normalized_result.get("data")), None
    if normalized_result.get("success"):
        return deepcopy(normalized_result), None
    return None, f"primitive '{primitive}' does not define extractable output"


def _precondition_failed_message(field: str, condition: dict[str, Any], actual: Any) -> str | None:
    if not isinstance(condition, dict):
        return None
    if condition.get("exists") is True and actual in (None, ""):
        return f"precondition failed: '{field}' must exist"
    if "equals" in condition and actual != condition.get("equals"):
        return f"precondition failed: '{field}' expected {condition.get('equals')!r}, actual={actual!r}"
    if "not_equals" in condition and actual == condition.get("not_equals"):
        return f"precondition failed: '{field}' must not equal {condition.get('not_equals')!r}"
    return None


def validate_and_project_steps_with_trace(
    steps: list[dict[str, Any]] | None,
    primitive_catalog: list[dict[str, Any]] | None,
    snapshot: dict[str, Any],
    *,
    grounding_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a primitive sequence and return projection trace metadata."""
    try:
        normalized_steps = expand_composite_steps(steps or [], primitive_catalog or [])
    except Exception as exc:
        return {
            "valid": False,
            "projected_snapshot": deepcopy(snapshot or {}),
            "validation_error": str(exc),
            "normalized_steps": [],
            "step_results": [],
            "event_facts": {},
        }

    catalog_by_name = {
        str(entry.get("name") or "").strip(): entry
        for entry in (primitive_catalog or [])
        if isinstance(entry, dict) and str(entry.get("name") or "").strip()
    }
    projected = deepcopy(snapshot or {})
    runtime_resource_type = (
        str(
            dict(projected.get("resource_core") or {}).get("resource_type")
            or projected.get("resource_type")
            or "resource"
        ).strip()
        or "resource"
    )
    event_facts: dict[str, Any] = {}
    step_results: list[dict[str, Any]] = []

    for step_index, step in enumerate(normalized_steps):
        primitive = str(step.get("primitive") or "").strip()
        primitive_meta = dict(catalog_by_name.get(primitive) or {})
        step_result: dict[str, Any] = {
            "step_index": step_index,
            "primitive": primitive,
            "params": deepcopy(step.get("params") or {}),
            "context_refs": _collect_context_refs(step.get("params") or {}),
            "start_snapshot": deepcopy(projected),
        }
        if not primitive_meta:
            validation_error = f"unknown primitive '{primitive}' at step {step_index}"
            step_result["validation_error"] = validation_error
            step_results.append(step_result)
            return {
                "valid": False,
                "projected_snapshot": projected,
                "validation_error": validation_error,
                "normalized_steps": deepcopy(normalized_steps),
                "step_results": step_results,
                "event_facts": deepcopy(event_facts),
            }

        try:
            resolved_params = resolve_param_refs(
                dict(step.get("params") or {}),
                grounding_context or {},
                event_facts=event_facts,
            )
        except Exception as exc:
            validation_error = f"param resolution failed at step {step_index} ({primitive}): {exc}"
            step_result["validation_error"] = validation_error
            step_results.append(step_result)
            return {
                "valid": False,
                "projected_snapshot": projected,
                "validation_error": validation_error,
                "normalized_steps": deepcopy(normalized_steps),
                "step_results": step_results,
                "event_facts": deepcopy(event_facts),
            }
        step_result["resolved_params"] = deepcopy(resolved_params)

        allowed_params = {
            str(param_name).strip()
            for param_name in dict(primitive_meta.get("params") or {})
            if str(param_name).strip()
        }
        unexpected_params = sorted(
            str(param_name).strip()
            for param_name in resolved_params.keys()
            if str(param_name).strip() and str(param_name).strip() not in allowed_params
        )
        if unexpected_params:
            allowed_description = (
                f"allowed params={sorted(allowed_params)}"
                if allowed_params
                else "primitive accepts no params"
            )
            validation_error = (
                f"unexpected params {unexpected_params} at step {step_index} "
                f"({primitive}); {allowed_description}"
            )
            step_result["validation_error"] = validation_error
            step_results.append(step_result)
            return {
                "valid": False,
                "projected_snapshot": projected,
                "validation_error": validation_error,
                "normalized_steps": deepcopy(normalized_steps),
                "step_results": step_results,
                "event_facts": deepcopy(event_facts),
            }

        for required_param in primitive_meta.get("required_params") or []:
            param_name = str(required_param or "").strip()
            if not param_name:
                continue
            if param_name not in resolved_params or resolved_params.get(param_name) is None:
                validation_error = (
                    f"missing required param '{param_name}' at step {step_index} ({primitive})"
                )
                step_result["validation_error"] = validation_error
                step_results.append(step_result)
                return {
                    "valid": False,
                    "projected_snapshot": projected,
                    "validation_error": validation_error,
                    "normalized_steps": deepcopy(normalized_steps),
                    "step_results": step_results,
                    "event_facts": deepcopy(event_facts),
                }

        preconditions = dict(primitive_meta.get("preconditions") or {})
        for field, condition in preconditions.items():
            actual = resource_snapshot_field_value(projected, str(field))
            failed = _precondition_failed_message(str(field), dict(condition or {}), actual)
            if failed:
                validation_error = f"{failed} at step {step_index} ({primitive})"
                step_result["validation_error"] = validation_error
                step_results.append(step_result)
                return {
                    "valid": False,
                    "projected_snapshot": projected,
                    "validation_error": validation_error,
                    "normalized_steps": deepcopy(normalized_steps),
                    "step_results": step_results,
                    "event_facts": deepcopy(event_facts),
                }

        preview_input = {**step, "params": deepcopy(resolved_params)}
        projected = apply_effects_to_snapshot(preview_input, primitive_meta, projected)
        step_result["projected_snapshot"] = deepcopy(projected)

        event_fact_key, event_fact_error = event_fact_key_for_primitive(
            primitive=primitive,
            params=resolved_params,
            resource_type=runtime_resource_type,
        )
        if event_fact_error is not None:
            validation_error = f"{event_fact_error} at step {step_index} ({primitive})"
            step_result["validation_error"] = validation_error
            step_results.append(step_result)
            return {
                "valid": False,
                "projected_snapshot": projected,
                "validation_error": validation_error,
                "normalized_steps": deepcopy(normalized_steps),
                "step_results": step_results,
                "event_facts": deepcopy(event_facts),
            }
        if event_fact_key:
            step_result["event_fact_path"] = f"event_facts.{event_fact_key}"
            preview, preview_error = preview_step_output(
                primitive=primitive,
                params=resolved_params,
                snapshot=projected,
                grounding_context={
                    **deepcopy(grounding_context or {}),
                    "event_facts": deepcopy(event_facts),
                },
                resource_type=runtime_resource_type,
            )
            step_result["preview_error"] = preview_error
            if preview_error is not None:
                validation_error = f"{preview_error} at step {step_index} ({primitive})"
                step_result["validation_error"] = validation_error
                step_results.append(step_result)
                return {
                    "valid": False,
                    "projected_snapshot": projected,
                    "validation_error": validation_error,
                    "normalized_steps": deepcopy(normalized_steps),
                    "step_results": step_results,
                    "event_facts": deepcopy(event_facts),
                }
            _assign_nested_mapping(event_facts, event_fact_key, preview)
            step_result["preview_output"] = deepcopy(preview)
        step_results.append(step_result)

    return {
        "valid": True,
        "projected_snapshot": projected,
        "validation_error": None,
        "normalized_steps": deepcopy(normalized_steps),
        "step_results": step_results,
        "event_facts": deepcopy(event_facts),
    }


def validate_and_project_steps(
    steps: list[dict[str, Any]] | None,
    primitive_catalog: list[dict[str, Any]] | None,
    snapshot: dict[str, Any],
    *,
    grounding_context: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any], str | None]:
    """Validate a primitive sequence and project its semantic effects forward."""
    result = validate_and_project_steps_with_trace(
        steps,
        primitive_catalog,
        snapshot,
        grounding_context=grounding_context,
    )
    return (
        bool(result.get("valid")),
        deepcopy(dict(result.get("projected_snapshot") or {})),
        result.get("validation_error"),
    )


def expected_snapshot_from_recovery_snapshot(
    snapshot: dict[str, Any],
    *,
    resource_type: str = "resource",
) -> dict[str, Any]:
    """Keep stable resource-profile fields for start-state validation."""
    profile = get_resource_profile(resource_type)
    return {
        field: resource_snapshot_field_value(snapshot, field, profile=profile)
        for field in profile.snapshot_fields
    }


def _compare_subset(actual: Any, expected: Any, path: str = "") -> tuple[bool, str | None]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"{path or 'value'} expected object, actual={type(actual).__name__}"
        for key, expected_value in expected.items():
            next_path = f"{path}.{key}" if path else str(key)
            if key not in actual:
                return False, f"{next_path} missing from actual snapshot"
            ok, message = _compare_subset(actual.get(key), expected_value, next_path)
            if not ok:
                return False, message
        return True, None
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False, f"{path or 'value'} expected list, actual={type(actual).__name__}"
        if len(actual) < len(expected):
            return (
                False,
                f"{path or 'value'} expected list length>={len(expected)}, actual={len(actual)}",
            )
        for index, expected_item in enumerate(expected):
            ok, message = _compare_subset(actual[index], expected_item, f"{path}[{index}]")
            if not ok:
                return False, message
        return True, None
    if actual != expected:
        return False, f"{path or 'value'} expected={expected!r} actual={actual!r}"
    return True, None


def snapshot_matches_expected(
    actual: dict[str, Any], expected: dict[str, Any]
) -> tuple[bool, str | None]:
    """Compare an expected snapshot subset against the runtime snapshot."""
    resource_type = (
        str(
            expected.get("resource_type")
            or dict(expected.get("resource_core") or {}).get("resource_type")
            or actual.get("resource_type")
            or dict(actual.get("resource_core") or {}).get("resource_type")
            or "resource"
        )
        .strip()
        .lower()
        or "resource"
    )
    profile = get_resource_profile(resource_type)
    comparable_fields = {
        "current_state",
        "current_location",
        "active_work",
        "current_pose_ref",
        *profile.snapshot_fields,
    }
    compared_fields: set[str] = set()
    for field in comparable_fields:
        if field not in expected:
            continue
        actual_value = resource_snapshot_field_value(actual, field, profile=profile)
        expected_value = resource_snapshot_field_value(expected, field, profile=profile)
        if actual_value == expected_value:
            compared_fields.add(field)
            continue
        equivalence_resolver = getattr(profile, "snapshot_equivalence_resolver", None)
        if callable(equivalence_resolver) and equivalence_resolver(
            field=field,
            actual_snapshot=actual,
            projected_snapshot=expected,
            actual_value=actual_value,
            projected_value=expected_value,
            profile=profile,
        ):
            compared_fields.add(field)
            continue
        return False, f"{field} expected={expected_value!r} actual={actual_value!r}"
    if compared_fields:
        remaining_expected = {
            key: value for key, value in expected.items() if key not in compared_fields
        }
        if not remaining_expected:
            return True, None
        return _compare_subset(actual, remaining_expected)
    return _compare_subset(actual, expected)


__all__ = [
    "apply_effects_to_snapshot",
    "expand_composite_steps",
    "expected_snapshot_from_recovery_snapshot",
    "extract_step_output",
    "resolve_param_refs",
    "resolve_step_param_refs",
    "snapshot_matches_expected",
    "validate_and_project_steps",
    "validate_and_project_steps_with_trace",
]
