"""Active v4 bridge primitive semantics.

This module is the active bridge/runtime semantic layer. It builds primitive
catalogs directly from live resource methods, projects bridge snapshots using
resource profiles, and validates primitive step sequences for the current v4
bridge path.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
import inspect
import re

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_resource_normalization import (
    bridge_resource_capabilities,
    normalize_bridge_resource,
    resolve_bridge_resource_type,
)
from cais_spade_llm.function_analyzer import FunctionAnalyzer
from cais_spade_llm.resources.resource_profile import (
    get_resource_profile,
    get_resource_profile_for_agent,
    resource_store_as_contract,
    resource_snapshot_availability,
    resource_snapshot_field_value,
    resource_snapshot_set_field,
)


_MISSING = object()
_KNOWN_CONTEXT_ROOTS = {"resource", "resources", "parts", "bridge_resources", "step_outputs"}


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


def _is_step_output_ref(ref: str, grounding_context: dict[str, Any] | None = None) -> bool:
    text = str(ref or "").strip()
    if not text:
        return False
    if text.startswith("/step_outputs/"):
        return True
    tokens = _iter_ref_tokens(text)
    if not tokens:
        return False
    first = _normalized_symbol(tokens[0])
    if first == "step_outputs":
        return True
    roots = {
        _normalized_symbol(key)
        for key in dict(grounding_context or {}).keys()
        if str(key).strip()
    }
    return first not in roots and first not in _KNOWN_CONTEXT_ROOTS and len(tokens) > 1


def _build_catalog_owner(resource_agent: Any) -> tuple[Any, Any]:
    profile = get_resource_profile_for_agent(resource_agent)
    owner = resource_agent
    if profile.primitive_owner_resolver is not None:
        try:
            owner = profile.primitive_owner_resolver(resource_agent) or resource_agent
        except Exception:
            owner = resource_agent
    return owner, profile


def _raw_bridge_primitive_names(resource_agent: Any) -> list[str]:
    raw = getattr(resource_agent, "_BRIDGE_PRIMITIVES", ()) or ()
    if isinstance(raw, (list, tuple)):
        names = [str(name or "").strip() for name in raw if str(name or "").strip()]
    else:
        names = sorted({str(name or "").strip() for name in raw if str(name or "").strip()})
    return [name for name in names if name]


def _callable_for_primitive(resource_agent: Any, owner: Any, primitive_name: str) -> Any:
    fn = getattr(resource_agent, primitive_name, None)
    if callable(fn):
        return fn
    fn = getattr(owner, primitive_name, None)
    if callable(fn):
        return fn
    return None


def _schema_properties_for_function(fn: Any) -> tuple[dict[str, Any], list[str], str]:
    analyzer = FunctionAnalyzer()
    analyzed = analyzer.analyze_function(fn)
    parameters = dict(analyzed.get("parameters") or {})
    properties = deepcopy(parameters.get("properties") or {})
    signature = inspect.signature(fn)
    required: list[str] = []
    for param_name, parameter in signature.parameters.items():
        if str(param_name) == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.default is inspect._empty:
            required.append(str(param_name))
    description = str(analyzed.get("description") or "").strip()
    return properties, required, description


def _primitive_summary(
    *,
    description: str,
    preconditions: dict[str, Any],
    effects: dict[str, Any],
) -> str:
    base = description or "Bridge primitive"
    pre_keys = ", ".join(sorted(str(key) for key in preconditions.keys())) if preconditions else ""
    effect_keys = ", ".join(sorted(str(key) for key in effects.keys())) if effects else ""
    detail_parts: list[str] = []
    if pre_keys:
        detail_parts.append(f"pre: {pre_keys}")
    if effect_keys:
        detail_parts.append(f"effects: {effect_keys}")
    if not detail_parts:
        return base
    return f"{base} ({'; '.join(detail_parts)})"


def _effective_catalog_required_params(
    required: list[str],
    *,
    profile: Any,
    primitive_name: str,
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    contract = resource_store_as_contract(profile, primitive_name)
    for raw_name in list(required or []) + list(contract.get("required_params") or []):
        token = str(raw_name or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        merged.append(token)
    return merged


def _resource_type_for_agent(resource_agent: Any, *, snapshot: dict[str, Any] | None = None) -> str:
    static_capabilities = deepcopy(getattr(resource_agent, "static_capabilities", {}) or {})
    return resolve_bridge_resource_type(
        resource=resource_agent,
        snapshot=snapshot or {},
        modeled_state={},
        static_capabilities=static_capabilities,
    )


def resolve_context_ref(
    ref: str,
    grounding_context: dict[str, Any] | None = None,
    *,
    step_outputs: dict[str, Any] | None = None,
) -> Any:
    """Resolve a dotted or JSON-pointer-like path against grounding context."""
    tokens = _iter_ref_tokens(ref)
    if not tokens:
        raise KeyError("empty context_ref")

    combined = deepcopy(dict(grounding_context or {}))
    existing_step_outputs = dict(combined.get("step_outputs") or {})
    if step_outputs:
        existing_step_outputs.update(deepcopy(step_outputs))
    combined["step_outputs"] = existing_step_outputs

    first = tokens[0]
    current: Any = _MISSING
    if _normalized_symbol(first) == "step_outputs":
        current = combined["step_outputs"]
        tokens = tokens[1:]
    else:
        current, found = _mapping_lookup(combined, first)
        if not found:
            current, found = _mapping_lookup(combined["step_outputs"], first)
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
    step_outputs: dict[str, Any] | None = None,
    preserve_step_output_refs: bool = False,
) -> Any:
    """Resolve ``context_ref`` dictionaries and reference-like strings."""
    if isinstance(value, dict):
        if set(value.keys()) == {"context_ref"}:
            ref = str(value.get("context_ref") or "").strip()
            if not ref:
                return None
            if preserve_step_output_refs and _is_step_output_ref(ref, grounding_context):
                return deepcopy(value)
            return resolve_context_ref(ref, grounding_context, step_outputs=step_outputs)
        return {
            str(key): resolve_param_refs(
                item,
                grounding_context,
                step_outputs=step_outputs,
                preserve_step_output_refs=preserve_step_output_refs,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            resolve_param_refs(
                item,
                grounding_context,
                step_outputs=step_outputs,
                preserve_step_output_refs=preserve_step_output_refs,
            )
            for item in value
        ]
    if isinstance(value, str):
        ref = value.strip()
        if not ref:
            return value
        if preserve_step_output_refs and _is_step_output_ref(ref, grounding_context):
            return value
        try:
            return resolve_context_ref(ref, grounding_context, step_outputs=step_outputs)
        except Exception:
            return value
    return deepcopy(value)


def build_execution_primitive_catalog(resource_agent: Any) -> list[dict[str, Any]]:
    """Build the execution primitive catalog from the live bridge surface."""
    if resource_agent is None:
        return []
    owner, profile = _build_catalog_owner(resource_agent)
    resource_type = _resource_type_for_agent(resource_agent)
    entries: list[dict[str, Any]] = []
    for primitive_name in _raw_bridge_primitive_names(resource_agent):
        fn = _callable_for_primitive(resource_agent, owner, primitive_name)
        if not callable(fn):
            continue
        properties, required, description = _schema_properties_for_function(fn)
        required = _effective_catalog_required_params(
            required,
            profile=profile,
            primitive_name=primitive_name,
        )
        frontmatter = FunctionAnalyzer._extract_yaml_frontmatter(fn) or {}
        preconditions = deepcopy(frontmatter.get("preconditions") or {})
        effects = deepcopy(frontmatter.get("effects") or {})
        observation_schema = deepcopy(
            dict(profile.observation_output_schema_map or {}).get(primitive_name) or {}
        )
        entry = {
            "name": primitive_name,
            "resource_type": resource_type,
            "description": str(frontmatter.get("description") or description or "").strip(),
            "params": properties,
            "required_params": required,
            "preconditions": preconditions,
            "effects": effects,
            "primitive_kind": str(
                dict(profile.primitive_kind_map or {}).get(primitive_name) or ""
            ).strip(),
            "output_schema": observation_schema,
            "synthesis_hidden": bool(frontmatter.get("synthesis_hidden", False)),
        }
        entry["semantic_summary"] = _primitive_summary(
            description=str(entry.get("description") or ""),
            preconditions=preconditions,
            effects=effects,
        )
        entries.append(entry)
    return entries


def _composite_parameter_schema(
    *,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    return deepcopy(properties or {}), list(required or [])


def _robot_prompt_composites(
    primitive_catalog: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    primitive_names = {
        str(entry.get("name") or "").strip()
        for entry in (primitive_catalog or [])
        if isinstance(entry, dict) and str(entry.get("name") or "").strip()
    }
    required_robot_primitives = {
        "open_gripper",
        "close_gripper",
        "attach_part",
        "detach_part",
    }
    if not required_robot_primitives <= primitive_names:
        return []

    composites: list[dict[str, Any]] = []

    grasp_params, grasp_required = _composite_parameter_schema(
        properties={
            "model_name": {
                "type": "string",
                "description": "Controller model name to attach after grasp.",
            },
            "part_name": {
                "type": "string",
                "description": "Canonical part name for held-part tracking.",
            },
            "position": {
                "type": "number",
                "description": "Optional gripper closing position.",
            },
        },
        required=["model_name"],
    )
    grasp_entry = {
        "name": "grasp_part",
        "resource_type": "robot",
        "description": (
            "Bridge-only composite primitive that closes the gripper and attaches "
            "the targeted part to the robot."
        ),
        "params": grasp_params,
        "required_params": grasp_required,
        "preconditions": {"held_part": {"equals": None}},
        "effects": {
            "gripper_state": {"set": "closed"},
            "held_part": {"set_from_param_any_of": ["part_name", "model_name"]},
        },
        "primitive_kind": "pick",
        "output_schema": {},
        "composite_expansion": [
            {"primitive": "close_gripper", "params_from_parent": ["position"]},
            {"primitive": "attach_part", "params_from_parent": ["model_name", "part_name"]},
        ],
    }
    grasp_entry["semantic_summary"] = _primitive_summary(
        description=str(grasp_entry.get("description") or ""),
        preconditions=dict(grasp_entry.get("preconditions") or {}),
        effects=dict(grasp_entry.get("effects") or {}),
    )
    composites.append(grasp_entry)

    release_params, release_required = _composite_parameter_schema(
        properties={
            "model_name": {
                "type": "string",
                "description": "Optional controller model name to detach.",
            },
            "assume_released_if_open": {
                "type": "boolean",
                "description": (
                    "Treat an already-open gripper as an idempotent release when true."
                ),
            },
        },
        required=[],
    )
    release_entry = {
        "name": "release_part",
        "resource_type": "robot",
        "description": (
            "Bridge-only composite primitive that opens the gripper and detaches "
            "the currently held part."
        ),
        "params": release_params,
        "required_params": release_required,
        "preconditions": {"held_part": {"exists": True}},
        "effects": {
            "gripper_state": {"set": "open"},
            "held_part": {"set": None},
        },
        "primitive_kind": "release",
        "output_schema": {},
        "composite_expansion": [
            {"primitive": "open_gripper", "params_from_parent": []},
            {
                "primitive": "detach_part",
                "params_from_parent": ["model_name", "assume_released_if_open"],
            },
        ],
    }
    release_entry["semantic_summary"] = _primitive_summary(
        description=str(release_entry.get("description") or ""),
        preconditions=dict(release_entry.get("preconditions") or {}),
        effects=dict(release_entry.get("effects") or {}),
    )
    composites.append(release_entry)

    return composites


def _prompt_hidden_primitive_names(resource_type: str) -> set[str]:
    if resource_type == "robot":
        return {"open_gripper", "close_gripper", "attach_part", "detach_part"}
    return set()


def filter_synthesis_primitive_catalog(
    primitive_catalog: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Filter the LLM-facing primitive surface using explicit visibility only."""
    resource_type = ""
    for raw_entry in primitive_catalog or []:
        if not isinstance(raw_entry, dict):
            continue
        resource_type = str(raw_entry.get("resource_type") or "").strip()
        if resource_type:
            break
    hidden_names = _prompt_hidden_primitive_names(resource_type)
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_entry in primitive_catalog or []:
        if not isinstance(raw_entry, dict):
            continue
        name = str(raw_entry.get("name") or "").strip()
        if not name or name in seen:
            continue
        if name in hidden_names:
            continue
        if bool(raw_entry.get("synthesis_hidden")):
            continue
        entry = deepcopy(raw_entry)
        entry.pop("synthesis_hidden", None)
        filtered.append(entry)
        seen.add(name)

    for composite in _robot_prompt_composites(primitive_catalog or []):
        name = str(composite.get("name") or "").strip()
        if not name or name in seen:
            continue
        filtered.append(deepcopy(composite))
        seen.add(name)
    return filtered


def build_synthesis_primitive_catalog(
    resource_agent: Any | None = None,
    *,
    primitive_catalog: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the LLM-facing synthesis catalog directly from execution primitives."""
    source_catalog = primitive_catalog
    if source_catalog is None and isinstance(resource_agent, list):
        source_catalog = resource_agent
        resource_agent = None
    if source_catalog is None:
        source_catalog = build_execution_primitive_catalog(resource_agent)
    return filter_synthesis_primitive_catalog(source_catalog or [])


def build_primitive_catalog(resource_agent: Any) -> list[dict[str, Any]]:
    """Compatibility alias for callers expecting the older name."""
    return build_execution_primitive_catalog(resource_agent)


def build_primitive_reference_card(primitive_catalog: list[dict[str, Any]] | None) -> str:
    """Render a compact human-readable reference card for prompt builders."""
    lines: list[str] = []
    for entry in filter_synthesis_primitive_catalog(primitive_catalog or []):
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        description = str(entry.get("description") or entry.get("semantic_summary") or "").strip()
        params = dict(entry.get("params") or {})
        required = {
            str(item).strip() for item in (entry.get("required_params") or []) if str(item).strip()
        }
        param_tokens: list[str] = []
        for param_name, schema in params.items():
            if not isinstance(schema, dict):
                continue
            type_name = str(schema.get("type") or "string").strip()
            suffix = " required" if param_name in required else ""
            param_tokens.append(f"{param_name}:{type_name}{suffix}")
        param_text = ", ".join(param_tokens) if param_tokens else "(no params)"
        lines.append(f"- {name}: {description}")
        lines.append(f"  params: {param_text}")
    return "\n".join(lines).strip()


def _snapshot_builder_payload(resource_agent: Any, profile: Any) -> dict[str, Any]:
    if profile.snapshot_builder is not None:
        built = profile.snapshot_builder(resource_agent)
        if isinstance(built, dict):
            return deepcopy(built)
    if hasattr(resource_agent, "_snapshot_state"):
        try:
            built = resource_agent._snapshot_state()
        except Exception:
            built = None
        if isinstance(built, dict):
            return deepcopy(built)
    return {}


def get_resource_bridge_snapshot(resource_agent: Any) -> dict[str, Any]:
    """Build the canonical bridge snapshot for a resource without recursion."""
    if resource_agent is None:
        return {}
    profile = get_resource_profile_for_agent(resource_agent)
    raw_snapshot = _snapshot_builder_payload(resource_agent, profile)
    resource_jid = str(
        getattr(resource_agent, "jid", "") or getattr(resource_agent, "agent_name", "") or ""
    ).strip()
    resource_type = _resource_type_for_agent(resource_agent, snapshot=raw_snapshot)
    normalized_resource = normalize_bridge_resource(
        resource_jid=resource_jid,
        resource_type=resource_type,
        snapshot=raw_snapshot,
        modeled_state={},
    )
    execution_catalog = build_execution_primitive_catalog(resource_agent)
    normalized_resource["bridge_adapter"] = bridge_resource_capabilities(
        resource_type,
        primitive_catalog=execution_catalog,
    )
    return normalized_resource


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
                **(
                    {"store_as": str(raw_step.get("store_as") or "").strip()}
                    if str(raw_step.get("store_as") or "").strip()
                    else {}
                ),
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
        return resource_snapshot_set_field(updated, field, {"x": x, "y": y, "z": z}, profile=profile)
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
    """Apply semantic effect frontmatter onto a bridge snapshot."""
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
        updated = _apply_effect_spec(str(field), dict(effect_spec or {}), params, updated, profile=profile)
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
        if target in dict(profile.extract_output_map or {}) or target in dict(profile.preview_output_map or {}):
            return resource_type
    return "resource"


def extract_step_output(
    *,
    primitive: str,
    params: dict[str, Any],
    step_result: dict[str, Any],
    resource_type: str = "",
) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize stored step outputs into deterministic bridge context payloads."""
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


def validate_and_project_steps(
    steps: list[dict[str, Any]] | None,
    primitive_catalog: list[dict[str, Any]] | None,
    snapshot: dict[str, Any],
    *,
    grounding_context: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any], str | None]:
    """Validate a primitive sequence and project its semantic effects forward."""
    try:
        normalized_steps = expand_composite_steps(steps or [], primitive_catalog or [])
    except Exception as exc:
        return False, deepcopy(snapshot or {}), str(exc)

    catalog_by_name = {
        str(entry.get("name") or "").strip(): entry
        for entry in (primitive_catalog or [])
        if isinstance(entry, dict) and str(entry.get("name") or "").strip()
    }
    projected = deepcopy(snapshot or {})
    runtime_resource_type = str(
        dict(projected.get("resource_core") or {}).get("resource_type")
        or projected.get("resource_type")
        or "resource"
    ).strip() or "resource"
    step_outputs: dict[str, Any] = {}

    for step_index, step in enumerate(normalized_steps):
        primitive = str(step.get("primitive") or "").strip()
        primitive_meta = dict(catalog_by_name.get(primitive) or {})
        if not primitive_meta:
            return False, projected, f"unknown primitive '{primitive}' at step {step_index}"

        try:
            resolved_params = resolve_param_refs(
                dict(step.get("params") or {}),
                grounding_context or {},
                step_outputs=step_outputs,
            )
        except Exception as exc:
            return False, projected, f"param resolution failed at step {step_index} ({primitive}): {exc}"

        for required_param in primitive_meta.get("required_params") or []:
            param_name = str(required_param or "").strip()
            if not param_name:
                continue
            if param_name not in resolved_params or resolved_params.get(param_name) is None:
                return (
                    False,
                    projected,
                    f"missing required param '{param_name}' at step {step_index} ({primitive})",
                )

        preconditions = dict(primitive_meta.get("preconditions") or {})
        for field, condition in preconditions.items():
            actual = resource_snapshot_field_value(projected, str(field))
            failed = _precondition_failed_message(str(field), dict(condition or {}), actual)
            if failed:
                return False, projected, f"{failed} at step {step_index} ({primitive})"

        preview_input = {**step, "params": deepcopy(resolved_params)}
        projected = apply_effects_to_snapshot(preview_input, primitive_meta, projected)

        store_as = str(step.get("store_as") or "").strip()
        if store_as:
            preview, preview_error = preview_step_output(
                primitive=primitive,
                params=resolved_params,
                snapshot=projected,
                grounding_context={
                    **deepcopy(grounding_context or {}),
                    "step_outputs": deepcopy(step_outputs),
                },
                resource_type=runtime_resource_type,
            )
            if preview_error is None:
                step_outputs[store_as] = preview

    return True, projected, None


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
            return False, f"{path or 'value'} expected list length>={len(expected)}, actual={len(actual)}"
        for index, expected_item in enumerate(expected):
            ok, message = _compare_subset(actual[index], expected_item, f"{path}[{index}]")
            if not ok:
                return False, message
        return True, None
    if actual != expected:
        return False, f"{path or 'value'} expected={expected!r} actual={actual!r}"
    return True, None


def snapshot_matches_expected(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str | None]:
    """Compare an expected snapshot subset against the runtime snapshot."""
    return _compare_subset(actual, expected)


def sync_agent_from_bridge_snapshot(resource_agent: Any, snapshot: dict[str, Any]) -> None:
    """Apply canonical snapshot fields back onto the live resource agent."""
    if resource_agent is None:
        return
    profile = get_resource_profile_for_agent(resource_agent)

    current_state = resource_snapshot_field_value(snapshot, "current_state", profile=profile)
    setattr(resource_agent, "_current_state", current_state)

    current_location = resource_snapshot_field_value(snapshot, "current_location", profile=profile)
    if hasattr(resource_agent, "_current_location") or current_location is not None:
        setattr(resource_agent, "_current_location", current_location)

    availability = resource_snapshot_availability(snapshot, profile=profile)
    if hasattr(resource_agent, "_availability") or availability:
        setattr(resource_agent, "_availability", availability)

    for field, target in dict(profile.sync_map or {}).items():
        value = resource_snapshot_field_value(snapshot, str(field), profile=profile)
        if callable(target):
            target(resource_agent, deepcopy(value))
            continue
        attr_name = str(target or "").strip()
        if attr_name:
            setattr(resource_agent, attr_name, deepcopy(value))


__all__ = [
    "apply_effects_to_snapshot",
    "build_execution_primitive_catalog",
    "build_primitive_reference_card",
    "build_synthesis_primitive_catalog",
    "expand_composite_steps",
    "extract_step_output",
    "filter_synthesis_primitive_catalog",
    "get_resource_bridge_snapshot",
    "resolve_param_refs",
    "snapshot_matches_expected",
    "sync_agent_from_bridge_snapshot",
    "validate_and_project_steps",
]
