"""Resource-type bridge adapters and canonical bridge schemas."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cais_spade_llm.resources.resource_profile import (
    all_registered_capability_flags,
    get_resource_profile,
    resource_snapshot_availability,
    resource_type_from_value,
)


def bridge_resource_type(
    *,
    resource: Any | None = None,
    snapshot: dict[str, Any] | None = None,
    modeled_state: dict[str, Any] | None = None,
    static_capabilities: dict[str, Any] | None = None,
) -> str:
    for candidate in (
        (snapshot or {}).get("resource_type"),
        (snapshot or {}).get("resource_core", {}).get("resource_type"),
        modeled_state.get("resource_type") if isinstance(modeled_state, dict) else None,
        static_capabilities.get("resource_type") if isinstance(static_capabilities, dict) else None,
        getattr(resource, "bridge_resource_type", lambda: "")() if resource is not None else "",
    ):
        token = resource_type_from_value(candidate)
        if token != "resource" or str(candidate or "").strip():
            return token

    bridge_snapshot: dict[str, Any] = {}
    raw_snapshot: dict[str, Any] = {}
    if resource is not None:
        get_bridge_snapshot = getattr(resource, "get_bridge_snapshot", None)
        if callable(get_bridge_snapshot):
            try:
                maybe_snapshot = get_bridge_snapshot()
            except Exception:
                maybe_snapshot = {}
            if isinstance(maybe_snapshot, dict):
                bridge_snapshot = dict(maybe_snapshot)
        if not bridge_snapshot:
            snapshot_fn = getattr(resource, "_snapshot_state", None)
            if callable(snapshot_fn):
                try:
                    maybe_snapshot = snapshot_fn()
                except Exception:
                    maybe_snapshot = {}
                if isinstance(maybe_snapshot, dict):
                    raw_snapshot = dict(maybe_snapshot)

    for candidate in (
        bridge_snapshot.get("resource_type"),
        bridge_snapshot.get("resource_core", {}).get("resource_type"),
        raw_snapshot.get("resource_type"),
    ):
        token = resource_type_from_value(candidate)
        if token != "resource" or str(candidate or "").strip():
            return token
    return "resource"

def _normalize_occupancy(
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


def canonical_bridge_resource(
    *,
    resource_jid: str,
    resource_type: str,
    snapshot: dict[str, Any] | None,
    modeled_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_snapshot = dict(snapshot or {})
    if isinstance(raw_snapshot.get("resource_core"), dict):
        current_state = str(
            raw_snapshot.get("current_state")
            if raw_snapshot.get("current_state") is not None
            else raw_snapshot.get("resource_core", {}).get("current_state")
            or (modeled_state or {}).get("resource_state")
            or ""
        ).strip() or "unknown"
        current_location = (
            raw_snapshot.get("current_location")
            if raw_snapshot.get("current_location") is not None
            else raw_snapshot.get("resource_core", {}).get("current_location")
            if raw_snapshot.get("resource_core", {}).get("current_location") is not None
            else (modeled_state or {}).get("current_location")
        )
    else:
        current_state = str(
            raw_snapshot.get("current_state")
            if raw_snapshot.get("current_state") is not None
            else (modeled_state or {}).get("resource_state")
            or ""
        ).strip() or "unknown"
        current_location = (
            raw_snapshot.get("current_location")
            if raw_snapshot.get("current_location") is not None
            else (modeled_state or {}).get("current_location")
        )

    normalized_type = resource_type_from_value(resource_type)
    profile = get_resource_profile(normalized_type)
    active_work = raw_snapshot.get("active_work")
    if active_work in (None, ""):
        active_work = raw_snapshot.get("active_job")
    resource_core = {
        "resource_jid": str(resource_jid or "").strip(),
        "resource_type": normalized_type,
        "current_state": current_state,
        "current_location": deepcopy(current_location),
        "availability": resource_snapshot_availability(
            raw_snapshot,
            profile=profile,
        ),
        "active_work": deepcopy(active_work),
        "occupancy": _normalize_occupancy(
            resource_type=normalized_type,
            current_location=current_location,
            snapshot_fields=raw_snapshot,
        ),
    }

    resource_facets: dict[str, Any] = {}
    if profile.facet_builder and str(profile.facet_key or "").strip():
        resource_facets[str(profile.facet_key)] = profile.facet_builder(raw_snapshot)

    canonical = {
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
            canonical[field] = deepcopy(value)
    return canonical


def bridge_adapter_capabilities(
    resource_type: str,
    *,
    primitive_catalog: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = resource_type_from_value(resource_type)
    profile = get_resource_profile(normalized)
    has_primitives = bool(primitive_catalog)
    capabilities = {
        "resource_type": normalized,
        "supports_executable_bridge": has_primitives,
        "observation_families": list(profile.observation_families),
        "example_families": list(profile.example_families),
    }
    for flag_name in all_registered_capability_flags():
        capabilities[flag_name] = False
    for flag_name, flag_value in dict(profile.capability_flags or {}).items():
        capabilities[str(flag_name)] = bool(flag_value and has_primitives)
    return capabilities


def canonical_operation_family_from_legacy(
    event: dict[str, Any],
    *,
    resource_type: str = "",
    bridge_resources: dict[str, Any] | None = None,
) -> str:
    operation_family = str(event.get("operation_family", "") or "").strip().lower()
    if operation_family:
        return operation_family

    resolved_resource_type = resource_type_from_value(
        resource_type
        or event.get("resource_type")
        or (
            (dict((bridge_resources or {}).get(str(event.get("resource_jid", "") or "").strip()) or {}))
            .get("resource_type")
        )
        or dict(
            dict((bridge_resources or {}).get(str(event.get("resource_jid", "") or "").strip()) or {}).get(
                "bridge_snapshot"
            )
            or {}
        ).get("resource_type")
    )
    profile = get_resource_profile(resolved_resource_type)
    if profile.event_family_resolver:
        resolved = str(profile.event_family_resolver(dict(event or {})) or "").strip().lower()
        if resolved:
            return resolved
    return "bridge"


def canonical_bridge_event(
    event: dict[str, Any],
    *,
    resource_type: str = "",
    bridge_resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = deepcopy(event or {})
    operation_family = canonical_operation_family_from_legacy(
        normalized,
        resource_type=resource_type,
        bridge_resources=bridge_resources,
    )
    targets = dict(normalized.get("targets") or {})
    part_name = str(normalized.get("part_name", "") or "").strip()
    if part_name and not targets.get("part_names"):
        targets["part_names"] = [part_name]
    expected_part_delta = dict(normalized.get("expected_part_delta") or {})
    location_to = (
        targets.get("location")
        or expected_part_delta.get("location_to")
    )
    if location_to not in (None, ""):
        targets["location"] = location_to

    projected_effects = deepcopy(normalized.get("projected_effects") or {})
    resource_effects = dict(projected_effects.get("resource") or {})
    if not resource_effects:
        resource_delta = dict(normalized.get("expected_resource_delta") or {})
        if resource_delta:
            resource_effects["current_state"] = {
                "from": resource_delta.get("from"),
                "to": resource_delta.get("to"),
            }
    occupancy_effects = dict(projected_effects.get("occupancy") or {})
    if not occupancy_effects:
        if location_to not in (None, ""):
            occupancy_effects["location"] = deepcopy(location_to)
        elif operation_family in {"clear", "home"}:
            occupancy_effects["location"] = None
    part_effects = dict(projected_effects.get("parts") or {})
    if not part_effects and part_name and expected_part_delta:
        part_effects[part_name] = {
            "state": expected_part_delta.get("to"),
            "location": expected_part_delta.get("location_to"),
        }
    work_effects = dict(projected_effects.get("work") or {})
    if not work_effects:
        projected_effects["work"] = {}

    projected_effects["resource"] = resource_effects
    projected_effects["occupancy"] = occupancy_effects
    projected_effects["parts"] = part_effects
    projected_effects["work"] = work_effects

    normalized["operation_family"] = operation_family
    normalized["targets"] = targets
    normalized["projected_effects"] = projected_effects
    return normalized


def _bridge_constraint_location(rule: dict[str, Any]) -> str:
    context = dict(rule.get("context") or {})
    for key in ("destination", "location", "zone", "area"):
        token = str(context.get(key, "") or "").strip()
        if token:
            return token
    return ""


def canonical_bridge_constraint_from_rule(
    rule: dict[str, Any],
    *,
    resource_jids: list[str],
) -> dict[str, Any] | None:
    location = _bridge_constraint_location(rule)
    if not location or len(resource_jids) < 2:
        return None
    return {
        "rule_id": str(rule.get("id", "") or "").strip(),
        "constraint_type": str(rule.get("constraint_type", "") or "").strip(),
        "location": location,
        "resource_jids": list(resource_jids),
        "generated_interpretation": str(
            rule.get("generated_interpretation", "") or rule.get("raw_text", "") or ""
        ).strip(),
        "raw_text": str(rule.get("raw_text", "") or "").strip(),
        "source": "safety_rule",
    }
