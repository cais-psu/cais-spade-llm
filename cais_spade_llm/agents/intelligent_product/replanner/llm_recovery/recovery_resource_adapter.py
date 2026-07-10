"""Recovery runtime snapshot adapter helpers.

The recovery consumes heterogeneous runtime snapshots from robots, printers, and
other resources. This module adapts those snapshots into one stable recovery
shape. It does not rewrite or reinterpret LLM-authored recovery proposals.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cais_spade_llm.resources.resource_profile import (
    all_registered_capability_flags,
    get_resource_profile,
    resource_snapshot_availability,
    resource_type_from_value,
)


def resolve_recovery_resource_type(
    *,
    resource: Any | None = None,
    snapshot: dict[str, Any] | None = None,
    modeled_state: dict[str, Any] | None = None,
    static_capabilities: dict[str, Any] | None = None,
) -> str:
    """Resolve a recovery resource type using the current precedence order."""
    recovery_type_hook = getattr(resource, "recovery_resource_type", None)
    recovery_type_value = ""
    if callable(recovery_type_hook):
        try:
            recovery_type_value = recovery_type_hook()
        except Exception:
            recovery_type_value = ""
    for candidate in (
        (snapshot or {}).get("resource_type"),
        (snapshot or {}).get("resource_core", {}).get("resource_type"),
        modeled_state.get("resource_type") if isinstance(modeled_state, dict) else None,
        static_capabilities.get("resource_type") if isinstance(static_capabilities, dict) else None,
        recovery_type_value,
    ):
        token = resource_type_from_value(candidate)
        if token != "resource" or str(candidate or "").strip():
            return token

    adapted_snapshot: dict[str, Any] = {}
    fallback_snapshot: dict[str, Any] = {}
    if resource is not None:
        get_recovery_snapshot = getattr(resource, "get_recovery_snapshot", None)
        if callable(get_recovery_snapshot):
            try:
                maybe_snapshot = get_recovery_snapshot()
            except Exception:
                maybe_snapshot = {}
            if isinstance(maybe_snapshot, dict):
                adapted_snapshot = dict(maybe_snapshot)
        if not adapted_snapshot:
            snapshot_fn = getattr(resource, "_snapshot_state", None)
            if callable(snapshot_fn):
                try:
                    maybe_snapshot = snapshot_fn()
                except Exception:
                    maybe_snapshot = {}
                if isinstance(maybe_snapshot, dict):
                    fallback_snapshot = dict(maybe_snapshot)

    for candidate in (
        adapted_snapshot.get("resource_type"),
        adapted_snapshot.get("resource_core", {}).get("resource_type"),
        fallback_snapshot.get("resource_type"),
    ):
        token = resource_type_from_value(candidate)
        if token != "resource" or str(candidate or "").strip():
            return token
    return "resource"


def _adapt_occupancy(
    *,
    resource_type: str,
    current_location: Any,
    snapshot_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = get_resource_profile(resource_type)
    if profile.occupancy_builder:
        return profile.occupancy_builder(
            current_location,
            dict(snapshot_fields or {}),
        )
    occupancy: dict[str, Any] = {}
    if current_location not in (None, ""):
        occupancy["location"] = deepcopy(current_location)
    return occupancy


def adapt_recovery_resource_snapshot(
    *,
    resource_jid: str,
    resource_type: str,
    snapshot: dict[str, Any] | None,
    modeled_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapt a raw resource snapshot into the stable recovery runtime shape."""
    raw_snapshot = dict(snapshot or {})
    if isinstance(raw_snapshot.get("resource_core"), dict):
        current_state = (
            str(
                raw_snapshot.get("current_state")
                if raw_snapshot.get("current_state") is not None
                else raw_snapshot.get("resource_core", {}).get("current_state")
                or (modeled_state or {}).get("resource_state")
                or ""
            ).strip()
            or "unknown"
        )
        current_location = (
            raw_snapshot.get("current_location")
            if raw_snapshot.get("current_location") is not None
            else raw_snapshot.get("resource_core", {}).get("current_location")
            if raw_snapshot.get("resource_core", {}).get("current_location") is not None
            else (modeled_state or {}).get("current_location")
        )
    else:
        current_state = (
            str(
                raw_snapshot.get("current_state")
                if raw_snapshot.get("current_state") is not None
                else (modeled_state or {}).get("resource_state") or ""
            ).strip()
            or "unknown"
        )
        current_location = (
            raw_snapshot.get("current_location")
            if raw_snapshot.get("current_location") is not None
            else (modeled_state or {}).get("current_location")
        )

    adapted_type = resource_type_from_value(resource_type)
    profile = get_resource_profile(adapted_type)
    active_work = raw_snapshot.get("active_work")
    if active_work in (None, ""):
        active_work = raw_snapshot.get("active_job")
    resource_core = {
        "resource_jid": str(resource_jid or "").strip(),
        "resource_type": adapted_type,
        "current_state": current_state,
        "current_location": deepcopy(current_location),
        "availability": resource_snapshot_availability(
            raw_snapshot,
            profile=profile,
        ),
        "active_work": deepcopy(active_work),
        "occupancy": _adapt_occupancy(
            resource_type=adapted_type,
            current_location=current_location,
            snapshot_fields=raw_snapshot,
        ),
    }

    resource_facets: dict[str, Any] = {}
    if profile.facet_builder and str(profile.facet_key or "").strip():
        resource_facets[str(profile.facet_key)] = profile.facet_builder(raw_snapshot)

    adapted = {
        "resource_type": resource_core["resource_type"],
        "current_state": resource_core["current_state"],
        "current_location": resource_core["current_location"],
        "active_work": resource_core["active_work"],
        "availability": resource_core["availability"],
        "occupancy": deepcopy(resource_core["occupancy"]),
        "resource_core": resource_core,
        "resource_facets": resource_facets,
    }
    for facet_values in resource_facets.values():
        if not isinstance(facet_values, dict):
            continue
        for field, value in facet_values.items():
            adapted[field] = deepcopy(value)
    return adapted


def recovery_resource_capabilities(
    resource_type: str,
    *,
    primitive_catalog: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive recovery-visible capability flags from the resource profile."""
    adapted = resource_type_from_value(resource_type)
    profile = get_resource_profile(adapted)
    has_primitives = bool(primitive_catalog)
    capabilities = {
        "resource_type": adapted,
        "supports_executable_recovery": has_primitives,
        "observation_families": list(profile.observation_families),
        "example_families": list(profile.example_families),
    }
    for flag_name in all_registered_capability_flags():
        capabilities[flag_name] = False
    for flag_name, flag_value in dict(profile.capability_flags or {}).items():
        capabilities[str(flag_name)] = bool(flag_value and has_primitives)
    return capabilities


__all__ = [
    "adapt_recovery_resource_snapshot",
    "recovery_resource_capabilities",
    "resolve_recovery_resource_type",
]
