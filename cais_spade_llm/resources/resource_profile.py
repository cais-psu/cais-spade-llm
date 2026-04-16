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
EventTargetResolver = Callable[[dict[str, Any]], str]
EventContractValidator = Callable[..., str | None]
CompilerHook = Callable[..., dict[str, Any]]
StateProjector = Callable[..., None]
EventStateValidator = Callable[..., str | None]
PrimitiveSequenceValidator = Callable[..., list[dict[str, Any]]]
CapabilityDecompositionProvider = Callable[..., Any]
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
    primitive_trace_fact_map: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    event_family_resolver: EventFamilyResolver | None = None
    event_target_resolver: EventTargetResolver | None = None
    event_contract_validator: EventContractValidator | None = None
    family_to_primitive: Mapping[str, str] = field(default_factory=dict)
    compiler_map: Mapping[str, str | CompilerHook] = field(default_factory=dict)
    state_projector: StateProjector | None = None
    event_state_validator: EventStateValidator | None = None
    primitive_sequence_validator: PrimitiveSequenceValidator | None = None
    capability_decomposition_provider: CapabilityDecompositionProvider | None = None
    capability_flags: Mapping[str, bool] = field(default_factory=dict)
    observation_families: tuple[str, ...] = ()
    grounding_observation_primitives: tuple[str, ...] = ()
    grounding_observation_fact_map: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    example_families: tuple[str, ...] = ("generic_bridge",)
    observation_output_schema_map: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    preview_output_map: Mapping[str, PreviewOutputResolver] = field(default_factory=dict)
    extract_output_map: Mapping[str, ExtractOutputResolver] = field(default_factory=dict)
    store_as_contract_map: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    primitive_event_target_contract_map: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    expected_end_state_projection_map: Mapping[str, dict[str, Any] | str] = field(
        default_factory=dict
    )
    carried_entity_field: str = ""
    carried_entity_location_builder: CarriedEntityLocationBuilder | None = None
    prompt_addendum: str = ""
    repair_example: str = ""


def resource_store_as_contract(
    profile: ResourceProfile | None,
    primitive_name: str,
) -> dict[str, Any]:
    primitive_token = str(primitive_name or "").strip()
    if profile is None or not primitive_token:
        return {}
    contract = dict((profile.store_as_contract_map or {}).get(primitive_token) or {})
    required_params = [
        str(param).strip()
        for param in (contract.get("required_params") or [])
        if str(param).strip()
    ]
    any_of_param_sets = [
        [
            str(param).strip()
            for param in (param_set or [])
            if str(param).strip()
        ]
        for param_set in (contract.get("any_of_param_sets") or [])
        if isinstance(param_set, (list, tuple))
    ]
    normalized: dict[str, Any] = {}
    if required_params:
        normalized["required_params"] = required_params
    if any_of_param_sets:
        normalized["any_of_param_sets"] = any_of_param_sets
    return normalized


def resource_capability_decompositions(
    profile: ResourceProfile | None,
    *,
    function_name: str = "",
    primitive_catalog: list[dict[str, Any]] | None = None,
    resource_jid: str = "",
) -> Any:
    """Return code-provided decomposition examples for modeled capabilities."""
    if profile is None or profile.capability_decomposition_provider is None:
        return {}
    payload = profile.capability_decomposition_provider(
        function_name=str(function_name or "").strip(),
        primitive_catalog=deepcopy(primitive_catalog or []),
        resource_jid=str(resource_jid or "").strip(),
    )
    return deepcopy(payload)


def resource_event_target(
    profile: ResourceProfile | None,
    outline_event: dict[str, Any] | None,
) -> str:
    if profile is None or profile.event_target_resolver is None:
        return ""
    return str(profile.event_target_resolver(dict(outline_event or {})) or "").strip()


def resource_primitive_event_target_contract(
    profile: ResourceProfile | None,
    primitive_name: str,
) -> dict[str, Any]:
    primitive_token = str(primitive_name or "").strip()
    if profile is None or not primitive_token:
        return {}
    raw_contract = dict(
        (profile.primitive_event_target_contract_map or {}).get(primitive_token) or {}
    )
    raw_param_fields = raw_contract.get("param_fields")
    if isinstance(raw_param_fields, str):
        raw_param_fields = [raw_param_fields]
    param_fields = [
        str(field).strip()
        for field in (raw_param_fields or [])
        if str(field).strip()
    ]
    raw_event_families = raw_contract.get("event_families")
    if isinstance(raw_event_families, str):
        raw_event_families = [raw_event_families]
    event_families = [
        str(family).strip().lower()
        for family in (raw_event_families or [])
        if str(family).strip()
    ]
    raw_conditions = raw_contract.get("conditions") or raw_contract.get("when")
    if isinstance(raw_conditions, str):
        raw_conditions = [raw_conditions]
    conditions = [
        str(token).strip().lower()
        for token in (raw_conditions or [])
        if str(token).strip()
    ]
    normalized: dict[str, Any] = {}
    if param_fields:
        normalized["param_fields"] = param_fields
    if event_families:
        normalized["event_families"] = event_families
    if conditions:
        normalized["conditions"] = conditions
    constraint_code = str(raw_contract.get("constraint_code") or "").strip()
    if constraint_code:
        normalized["constraint_code"] = constraint_code
    return normalized


def resource_expected_end_state_projection_map(
    profile: ResourceProfile | None,
) -> dict[str, dict[str, Any]]:
    def _normalize(
        mapping: Mapping[str, dict[str, Any] | str] | None,
    ) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for expected_field, raw_spec in dict(mapping or {}).items():
            expected_token = str(expected_field or "").strip()
            if not expected_token:
                continue
            if isinstance(raw_spec, str):
                snapshot_field = str(raw_spec).strip()
                compare_when_present = False
            elif isinstance(raw_spec, dict):
                snapshot_field = str(
                    raw_spec.get("snapshot_field")
                    or raw_spec.get("field")
                    or ""
                ).strip()
                compare_when_present = bool(raw_spec.get("compare_when_present"))
            else:
                continue
            if not snapshot_field:
                continue
            normalized[expected_token] = {
                "snapshot_field": snapshot_field,
                "compare_when_present": compare_when_present,
            }
        return normalized

    merged = _normalize(_DEFAULT_PROFILE.expected_end_state_projection_map)
    if profile is not None and profile is not _DEFAULT_PROFILE:
        merged.update(_normalize(profile.expected_end_state_projection_map))
    return merged


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
    field_name = str(field or "").strip()
    if not field_name:
        return None

    def _nested_get(mapping: dict[str, Any], dotted_field: str) -> Any:
        current: Any = mapping
        for token in dotted_field.split("."):
            if not isinstance(current, dict) or token not in current:
                return None
            current = current.get(token)
        return deepcopy(current)

    resource_core = dict(raw_snapshot.get("resource_core") or {})
    if "." in field_name:
        value = _nested_get(resource_core, field_name)
        if value is not None:
            return value

        facet_key = str(active_profile.facet_key or "").strip()
        if facet_key:
            facet = dict((raw_snapshot.get("resource_facets") or {}).get(facet_key) or {})
            value = _nested_get(facet, field_name)
            if value is not None:
                return value

        value = _nested_get(raw_snapshot, field_name)
        if value is not None:
            return value

    if field_name in resource_core and resource_core.get(field_name) is not None:
        return deepcopy(resource_core.get(field_name))

    facet_key = str(active_profile.facet_key or "").strip()
    if facet_key:
        facet = dict((raw_snapshot.get("resource_facets") or {}).get(facet_key) or {})
        if field_name in facet and facet.get(field_name) is not None:
            return deepcopy(facet.get(field_name))

    if field_name in raw_snapshot and raw_snapshot.get(field_name) is not None:
        return deepcopy(raw_snapshot.get(field_name))

    if field_name in resource_core:
        return deepcopy(resource_core.get(field_name))
    if facet_key:
        facet = dict((raw_snapshot.get("resource_facets") or {}).get(facet_key) or {})
        if field_name in facet:
            return deepcopy(facet.get(field_name))
    if field_name in raw_snapshot:
        return deepcopy(raw_snapshot.get(field_name))
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

    def _nested_set(
        mapping: dict[str, Any],
        dotted_field: str,
        next_value: Any,
    ) -> dict[str, Any]:
        target = mapping
        tokens = [token for token in dotted_field.split(".") if token]
        if not tokens:
            return target
        for token in tokens[:-1]:
            child = target.get(token)
            if not isinstance(child, dict):
                child = {}
            target[token] = child
            target = child
        target[tokens[-1]] = deepcopy(next_value)
        return mapping

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
    root_field = field_name.split(".", 1)[0]

    if "." in field_name:
        if root_field in core_fields or root_field in resource_core or not facet_key:
            resource_core = _nested_set(resource_core, field_name, value)
            raw_snapshot["resource_core"] = resource_core
            if root_field in core_fields:
                top_level_value = dict(raw_snapshot.get(root_field) or {})
                if not isinstance(top_level_value, dict):
                    top_level_value = {}
                raw_snapshot[root_field] = _nested_set(top_level_value, field_name.split(".", 1)[1], value)
            return raw_snapshot

        resource_facets = dict(raw_snapshot.get("resource_facets") or {})
        facet = dict(resource_facets.get(facet_key) or {})
        facet = _nested_set(facet, field_name, value)
        resource_facets[facet_key] = facet
        raw_snapshot["resource_facets"] = resource_facets
        return raw_snapshot

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
    expected_end_state_projection_map={
        "resource_state": "current_state",
        "resource_location": {
            "snapshot_field": "current_location",
            "compare_when_present": True,
        },
    },
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
