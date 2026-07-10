"""Generic resource primitive catalog and recovery snapshot helpers."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_resource_adapter import (
    adapt_recovery_resource_snapshot,
    recovery_resource_capabilities,
    resolve_recovery_resource_type,
)
from cais_spade_llm.function_analyzer import FunctionAnalyzer
from cais_spade_llm.resources.resource_profile import (
    get_resource_profile_for_agent,
    resource_event_fact_contract,
    resource_snapshot_availability,
    resource_snapshot_field_value,
)


def primitive_summary(
    *,
    description: str,
    preconditions: dict[str, Any],
    effects: dict[str, Any],
) -> str:
    base = description or "Recovery primitive"
    pre_keys = ", ".join(sorted(str(key) for key in preconditions)) if preconditions else ""
    effect_keys = ", ".join(sorted(str(key) for key in effects)) if effects else ""
    detail_parts: list[str] = []
    if pre_keys:
        detail_parts.append(f"pre: {pre_keys}")
    if effect_keys:
        detail_parts.append(f"effects: {effect_keys}")
    if not detail_parts:
        return base
    return f"{base} ({'; '.join(detail_parts)})"


def _build_catalog_owner(resource_agent: Any) -> tuple[Any, Any]:
    profile = get_resource_profile_for_agent(resource_agent)
    owner = resource_agent
    if profile.primitive_owner_resolver is not None:
        try:
            owner = profile.primitive_owner_resolver(resource_agent) or resource_agent
        except Exception:
            owner = resource_agent
    return owner, profile


def _raw_recovery_primitive_names(resource_agent: Any) -> list[str]:
    raw = getattr(resource_agent, "_RECOVERY_PRIMITIVES", ()) or ()
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


def _effective_catalog_required_params(
    required: list[str],
    *,
    profile: Any,
    primitive_name: str,
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    contract = resource_event_fact_contract(profile, primitive_name)
    for raw_name in list(required or []) + list(contract.get("required_params") or []):
        token = str(raw_name or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        merged.append(token)
    return merged


def _resource_type_for_agent(
    resource_agent: Any,
    *,
    snapshot: dict[str, Any] | None = None,
) -> str:
    static_capabilities = deepcopy(getattr(resource_agent, "static_capabilities", {}) or {})
    return resolve_recovery_resource_type(
        resource=resource_agent,
        snapshot=snapshot or {},
        modeled_state={},
        static_capabilities=static_capabilities,
    )


def build_execution_primitive_catalog(resource_agent: Any) -> list[dict[str, Any]]:
    """Build the execution primitive catalog from a resource recovery surface."""
    if resource_agent is None:
        return []
    owner, profile = _build_catalog_owner(resource_agent)
    resource_type = _resource_type_for_agent(resource_agent)
    entries: list[dict[str, Any]] = []
    for primitive_name in _raw_recovery_primitive_names(resource_agent):
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
        entry["semantic_summary"] = primitive_summary(
            description=str(entry.get("description") or ""),
            preconditions=preconditions,
            effects=effects,
        )
        entries.append(entry)
    return entries


def filter_synthesis_primitive_catalog(
    primitive_catalog: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Filter a generic LLM-facing primitive surface."""
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_entry in primitive_catalog or []:
        if not isinstance(raw_entry, dict):
            continue
        name = str(raw_entry.get("name") or "").strip()
        if not name or name in seen:
            continue
        if bool(raw_entry.get("synthesis_hidden")):
            continue
        entry = deepcopy(raw_entry)
        entry.pop("synthesis_hidden", None)
        filtered.append(entry)
        seen.add(name)
    return filtered


def build_synthesis_primitive_catalog(
    resource_agent: Any | None = None,
    *,
    primitive_catalog: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a generic LLM-facing synthesis catalog from execution primitives."""
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


def _execution_catalog_for_snapshot(resource_agent: Any) -> list[dict[str, Any]]:
    method = getattr(resource_agent, "recovery_execution_primitive_catalog", None)
    if callable(method):
        try:
            catalog = method()
            if isinstance(catalog, list):
                return deepcopy(catalog)
        except Exception:
            pass
    return build_execution_primitive_catalog(resource_agent)


def get_resource_recovery_snapshot(resource_agent: Any) -> dict[str, Any]:
    """Build the canonical recovery snapshot for a resource without recursion."""
    if resource_agent is None:
        return {}
    profile = get_resource_profile_for_agent(resource_agent)
    raw_snapshot = _snapshot_builder_payload(resource_agent, profile)
    resource_jid = str(
        getattr(resource_agent, "jid", "") or getattr(resource_agent, "agent_name", "") or ""
    ).strip()
    resource_type = _resource_type_for_agent(resource_agent, snapshot=raw_snapshot)
    adapted_resource = adapt_recovery_resource_snapshot(
        resource_jid=resource_jid,
        resource_type=resource_type,
        snapshot=raw_snapshot,
        modeled_state={},
    )
    execution_catalog = _execution_catalog_for_snapshot(resource_agent)
    adapted_resource["recovery_adapter"] = recovery_resource_capabilities(
        resource_type,
        primitive_catalog=execution_catalog,
    )
    return adapted_resource


def sync_agent_from_recovery_snapshot(resource_agent: Any, snapshot: dict[str, Any]) -> None:
    """Apply canonical snapshot fields back onto the live resource agent."""
    if resource_agent is None:
        return
    profile = get_resource_profile_for_agent(resource_agent)

    current_state = resource_snapshot_field_value(snapshot, "current_state", profile=profile)
    resource_agent._current_state = current_state

    current_location = resource_snapshot_field_value(snapshot, "current_location", profile=profile)
    if hasattr(resource_agent, "_current_location") or current_location is not None:
        resource_agent._current_location = current_location

    availability = resource_snapshot_availability(snapshot, profile=profile)
    if hasattr(resource_agent, "_availability") or availability:
        resource_agent._availability = availability

    for field, target in dict(profile.sync_map or {}).items():
        value = resource_snapshot_field_value(snapshot, str(field), profile=profile)
        if callable(target):
            target(resource_agent, deepcopy(value))
            continue
        attr_name = str(target or "").strip()
        if attr_name:
            setattr(resource_agent, attr_name, deepcopy(value))


__all__ = [
    "build_execution_primitive_catalog",
    "build_primitive_catalog",
    "build_primitive_reference_card",
    "build_synthesis_primitive_catalog",
    "filter_synthesis_primitive_catalog",
    "get_resource_recovery_snapshot",
    "primitive_summary",
    "sync_agent_from_recovery_snapshot",
]
