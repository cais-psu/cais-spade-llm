"""Generic resource profile registry and snapshot helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


FacetBuilder = Callable[[dict[str, Any]], dict[str, Any]]
OccupancyBuilder = Callable[[Any, dict[str, Any]], dict[str, Any]]
SnapshotBuilder = Callable[[Any], dict[str, Any]]
AvailabilityResolver = Callable[[dict[str, Any], str], str]
PrimitiveOwnerResolver = Callable[[Any], Any | None]
SyncTarget = str | Callable[[Any, Any], None]
EventFamilyResolver = Callable[[dict[str, Any]], str]
EventContractValidator = Callable[..., str | None]
CompilerHook = Callable[..., dict[str, Any]]
StateProjector = Callable[..., None]
EventStateValidator = Callable[..., str | None]
PreviewOutputResolver = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]],
    tuple[dict[str, Any] | None, str | None],
]
ExtractOutputResolver = Callable[
    [dict[str, Any], dict[str, Any]],
    tuple[dict[str, Any] | None, str | None],
]
CarriedEntityLocationBuilder = Callable[[str, dict[str, Any] | None], str]


def resource_type_from_value(value: Any) -> str:
    token = str(value or "").strip().lower()
    return token or "resource"


@dataclass(frozen=True)
class ResourceProfile:
    resource_type: str
    snapshot_fields: tuple[str, ...] = ("current_state",)
    facet_key: str = ""
    facet_builder: FacetBuilder | None = None
    occupancy_builder: OccupancyBuilder | None = None
    snapshot_builder: SnapshotBuilder | None = None
    availability_resolver: AvailabilityResolver | None = None
    primitive_owner_resolver: PrimitiveOwnerResolver | None = None
    sync_map: Mapping[str, SyncTarget] = field(default_factory=dict)
    primitive_kind_map: Mapping[str, str] = field(default_factory=dict)
    event_family_resolver: EventFamilyResolver | None = None
    event_contract_validator: EventContractValidator | None = None
    family_to_primitive: Mapping[str, str] = field(default_factory=dict)
    compiler_map: Mapping[str, str | CompilerHook] = field(default_factory=dict)
    state_projector: StateProjector | None = None
    event_state_validator: EventStateValidator | None = None
    capability_flags: Mapping[str, bool] = field(default_factory=dict)
    observation_families: tuple[str, ...] = ()
    example_families: tuple[str, ...] = ("generic_bridge",)
    observation_output_schema_map: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    preview_output_map: Mapping[str, PreviewOutputResolver] = field(default_factory=dict)
    extract_output_map: Mapping[str, ExtractOutputResolver] = field(default_factory=dict)
    carried_entity_field: str = ""
    carried_entity_location_builder: CarriedEntityLocationBuilder | None = None
    prompt_addendum: str = ""
    repair_example: str = ""


def resource_snapshot_field_value(
    snapshot: dict[str, Any] | None,
    field: str,
    *,
    profile: ResourceProfile | None = None,
) -> Any:
    raw_snapshot = dict(snapshot or {})
    active_profile = profile or get_resource_profile(
        resource_type_from_value(
            raw_snapshot.get("resource_type")
            or dict(raw_snapshot.get("resource_core") or {}).get("resource_type")
        )
    )

    resource_core = dict(raw_snapshot.get("resource_core") or {})
    if field in resource_core and resource_core.get(field) is not None:
        return deepcopy(resource_core.get(field))

    facet_key = str(active_profile.facet_key or "").strip()
    if facet_key:
        facet = dict((raw_snapshot.get("resource_facets") or {}).get(facet_key) or {})
        if field in facet and facet.get(field) is not None:
            return deepcopy(facet.get(field))

    if field in raw_snapshot and raw_snapshot.get(field) is not None:
        return deepcopy(raw_snapshot.get(field))

    if field in resource_core:
        return deepcopy(resource_core.get(field))
    if facet_key:
        facet = dict((raw_snapshot.get("resource_facets") or {}).get(facet_key) or {})
        if field in facet:
            return deepcopy(facet.get(field))
    if field in raw_snapshot:
        return deepcopy(raw_snapshot.get(field))
    return None


def resource_snapshot_set_field(
    snapshot: dict[str, Any] | None,
    field: str,
    value: Any,
    *,
    profile: ResourceProfile | None = None,
) -> dict[str, Any]:
    raw_snapshot = deepcopy(dict(snapshot or {}))
    active_profile = profile or get_resource_profile(
        resource_type_from_value(
            raw_snapshot.get("resource_type")
            or dict(raw_snapshot.get("resource_core") or {}).get("resource_type")
        )
    )
    field_name = str(field or "").strip()
    if not field_name:
        return raw_snapshot

    core_fields = {
        "resource_jid",
        "resource_type",
        "current_state",
        "current_location",
        "availability",
        "active_work",
        "occupancy",
    }
    facet_key = str(active_profile.facet_key or "").strip()
    resource_core = dict(raw_snapshot.get("resource_core") or {})

    if field_name in core_fields or field_name in resource_core or not facet_key:
        resource_core[field_name] = deepcopy(value)
        raw_snapshot["resource_core"] = resource_core
        if field_name in core_fields:
            raw_snapshot[field_name] = deepcopy(value)
        else:
            raw_snapshot.pop(field_name, None)
        return raw_snapshot

    resource_facets = dict(raw_snapshot.get("resource_facets") or {})
    facet = dict(resource_facets.get(facet_key) or {})
    facet[field_name] = deepcopy(value)
    resource_facets[facet_key] = facet
    raw_snapshot["resource_facets"] = resource_facets
    raw_snapshot.pop(field_name, None)
    return raw_snapshot


def resource_snapshot_has_field(
    snapshot: dict[str, Any] | None,
    field: str,
    *,
    profile: ResourceProfile | None = None,
) -> bool:
    raw_snapshot = dict(snapshot or {})
    active_profile = profile or get_resource_profile(
        resource_type_from_value(
            raw_snapshot.get("resource_type")
            or dict(raw_snapshot.get("resource_core") or {}).get("resource_type")
        )
    )
    if field in raw_snapshot:
        return True
    if field in dict(raw_snapshot.get("resource_core") or {}):
        return True
    facet_key = str(active_profile.facet_key or "").strip()
    if facet_key and field in dict((raw_snapshot.get("resource_facets") or {}).get(facet_key) or {}):
        return True
    return False


def resource_snapshot_fields_map(
    snapshot: dict[str, Any] | None,
    fields: tuple[str, ...] | list[str],
    *,
    profile: ResourceProfile | None = None,
) -> dict[str, Any]:
    return {
        str(field): resource_snapshot_field_value(snapshot, str(field), profile=profile)
        for field in fields
    }


def resource_snapshot_availability(
    snapshot: dict[str, Any] | None,
    *,
    profile: ResourceProfile | None = None,
) -> str:
    raw_snapshot = dict(snapshot or {})
    raw_value = resource_snapshot_field_value(raw_snapshot, "availability", profile=profile)
    availability = str(raw_value or "").strip().lower()
    if availability:
        return availability
    current_state = str(
        resource_snapshot_field_value(raw_snapshot, "current_state", profile=profile) or ""
    ).strip()
    active_profile = profile or get_resource_profile(
        resource_type_from_value(
            raw_snapshot.get("resource_type")
            or dict(raw_snapshot.get("resource_core") or {}).get("resource_type")
        )
    )
    if active_profile.availability_resolver is not None:
        resolved = str(active_profile.availability_resolver(raw_snapshot, current_state) or "").strip().lower()
        if resolved:
            return resolved
    return "available"


def resource_snapshot_carried_entity(
    snapshot: dict[str, Any] | None,
    *,
    profile: ResourceProfile | None = None,
) -> Any:
    raw_snapshot = dict(snapshot or {})
    active_profile = profile or get_resource_profile(
        resource_type_from_value(
            raw_snapshot.get("resource_type")
            or dict(raw_snapshot.get("resource_core") or {}).get("resource_type")
        )
    )
    field = str(active_profile.carried_entity_field or "").strip()
    if not field:
        return None
    return resource_snapshot_field_value(raw_snapshot, field, profile=active_profile)


def resource_snapshot_carried_entity_location(
    *,
    resource_jid: str,
    snapshot: dict[str, Any] | None,
    profile: ResourceProfile | None = None,
) -> str:
    raw_snapshot = dict(snapshot or {})
    active_profile = profile or get_resource_profile(
        resource_type_from_value(
            raw_snapshot.get("resource_type")
            or dict(raw_snapshot.get("resource_core") or {}).get("resource_type")
        )
    )
    if active_profile.carried_entity_location_builder is None:
        return ""
    return str(
        active_profile.carried_entity_location_builder(
            str(resource_jid or "").strip(),
            raw_snapshot,
        )
        or ""
    ).strip()


_PROFILE_REGISTRY: dict[str, ResourceProfile] = {}
_BUILTINS_REGISTERED = False


_DEFAULT_PROFILE = ResourceProfile(
    resource_type="resource",
    snapshot_fields=("current_state",),
    example_families=("generic_bridge",),
)


def _ensure_builtin_resource_profiles_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True

    # Import for side effects: each module registers its own built-in profile.
    from cais_spade_llm.resources.machine import printer_profile  # noqa: F401
    from cais_spade_llm.resources.robot import robot_profile  # noqa: F401


def register_resource_profile(profile: ResourceProfile) -> None:
    _PROFILE_REGISTRY[resource_type_from_value(profile.resource_type)] = profile


def get_resource_profile(resource_type: str) -> ResourceProfile:
    _ensure_builtin_resource_profiles_registered()
    return _PROFILE_REGISTRY.get(
        resource_type_from_value(resource_type),
        _DEFAULT_PROFILE,
    )


def get_resource_profile_for_agent(agent: Any) -> ResourceProfile:
    profile = getattr(agent, "_RESOURCE_PROFILE", None)
    if isinstance(profile, ResourceProfile):
        return profile

    resource_type = ""
    bridge_resource_type = getattr(agent, "bridge_resource_type", None)
    if callable(bridge_resource_type):
        try:
            resource_type = str(bridge_resource_type() or "").strip().lower()
        except Exception:
            resource_type = ""
    if not resource_type:
        static_capabilities = getattr(agent, "static_capabilities", {}) or {}
        resource_type = str(static_capabilities.get("resource_type", "") or "").strip().lower()
    return get_resource_profile(resource_type or "resource")


def all_registered_operation_kinds() -> set[str]:
    _ensure_builtin_resource_profiles_registered()
    kinds: set[str] = set()
    for profile in _PROFILE_REGISTRY.values():
        kinds.update(
            str(kind).strip()
            for kind in profile.primitive_kind_map.values()
            if str(kind).strip()
        )
        kinds.update(
            str(kind).strip()
            for kind in profile.family_to_primitive.keys()
            if str(kind).strip()
        )
        kinds.update(
            str(kind).strip()
            for kind in profile.compiler_map.keys()
            if str(kind).strip()
        )
    kinds.update({"bridge", "clear", "home"})
    return {kind for kind in kinds if kind}


def all_registered_capability_flags() -> set[str]:
    _ensure_builtin_resource_profiles_registered()
    flags: set[str] = set()
    for profile in _PROFILE_REGISTRY.values():
        flags.update(
            str(flag).strip()
            for flag in (profile.capability_flags or {}).keys()
            if str(flag).strip()
        )
    return flags
