"""Build an obligation-centric RecoveryContext from system state.

This module bridges existing v1 bridge infrastructure (PreparedBridgeRequest,
bridge_session helpers) into the v2 universal repair language.  The key
difference from the existing ``grounding_context`` is that this context is
**obligation-centric** — it presents all resources and obligations equally
rather than centering on one stuck ``ra_jid``.
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.mutation_types import (
    RecoveryContext,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.part_state_semantics import (
    part_state_is_carried,
    part_state_is_stably_grounded,
    part_state_requires_external_localization,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.bridge_adapters import (
    canonical_bridge_resource,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.primitive_semantics import (
    build_primitive_catalog,
    build_synthesis_primitive_catalog,
    preview_step_output,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.observation_policy import (
    primitive_supports_part_grounding,
    semantic_observation_candidates_from_grounding_candidates,
)
from cais_spade_llm.prompts import (
    _bridge_generalize_location_summary,
)
from cais_spade_llm.resources.robot.place_geometry_resolution import (
    has_place_geometry_fields,
    resolve_place_geometry,
)
from cais_spade_llm.resources.resource_profile import (
    get_resource_profile,
    resource_snapshot_fields_map,
)

logger = logging.getLogger(__name__)

_TRUSTED_POSE_SOURCES = frozenset({"live_observation", "controller_runtime"})
_DEGRADED_RESOURCE_STATES = frozenset({"error", "fault", "offline"})
_DEGRADED_AVAILABILITY_STATES = frozenset({"unavailable", "offline", "degraded"})


def _prompt_location_summary(value: Any) -> Any:
    """Generalize symbolic location tokens before they are shown to the LLM."""
    if isinstance(value, str):
        return _bridge_generalize_location_summary(value)
    return deepcopy(value)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_recovery_context(
    prepared_bridge_request: dict[str, Any],
    *,
    planner: Any | None = None,
    resource_agents: dict[str, Any] | None = None,
    observation_store: dict[str, Any] | None = None,
    discovered_constraints: list[dict[str, Any]] | None = None,
) -> RecoveryContext:
    """Build a :class:`RecoveryContext` from v1 bridge request + planner state.

    Parameters
    ----------
    prepared_bridge_request:
        The standard v1 prepared bridge request dict (contains
        ``bridge_resources``, ``part_tracker``, ``obligation_targets``,
        ``tools_catalog``, ``goal_state``, ``bridge_safety_context``, etc.).
    planner:
        Optional ``ProcessPlanner`` instance — used to extract pending tasks.
    resource_agents:
        Optional mapping of resource JID → agent instance — used to build
        primitive catalogs if not already in the bridge request.
    observation_store:
        Outputs from prior top-level ``observe`` turns.
    discovered_constraints:
        Constraints extracted from prior validator rejections (for
        fresh-prompt constraint accumulation).
    """
    bridge_resources = deepcopy(prepared_bridge_request.get("bridge_resources") or {})

    # ----- resource snapshots (symmetric across all resources) -----
    resource_snapshots = _build_resource_snapshots(bridge_resources)

    # ----- workspace bounds (from resource agents' static capabilities) -----
    if resource_agents:
        for jid, snap in resource_snapshots.items():
            agent = resource_agents.get(jid)
            if agent is None:
                continue
            caps = getattr(agent, "static_capabilities", None)
            if isinstance(caps, dict) and caps.get("workspace_bounds"):
                snap["workspace_bounds"] = deepcopy(caps["workspace_bounds"])

    # ----- part states -----
    part_states = _build_part_states(
        deepcopy(prepared_bridge_request.get("part_tracker") or {}),
    )

    # ----- pending tasks -----
    pending_tasks = _build_pending_tasks(bridge_resources, planner)

    # ----- active obligations (from safety context + obligation targets) -----
    active_obligations = _build_active_obligations(prepared_bridge_request)

    # ----- goal state -----
    goal_state = str(prepared_bridge_request.get("goal_state", "") or "").strip()

    # ----- priority context -----
    priority_context = dict(prepared_bridge_request.get("priority_context") or {})

    # ----- available task actions (from tools catalog) -----
    available_task_actions = list(prepared_bridge_request.get("tools_catalog") or [])

    # ----- available primitives (per resource) -----
    available_primitives = _build_available_primitives(
        bridge_resources, resource_agents,
    )

    # ----- capability degradations -----
    capability_degradations = _build_capability_degradations(
        bridge_resources, resource_snapshots,
    )

    product_geometry = dict(getattr(planner, "product_geometry", None) or {})
    if not product_geometry:
        product_geometry = dict(
            getattr(getattr(planner, "product_agent", None), "product_geometry", None) or {}
        )

    grounded_environment_facts = _build_grounded_environment_facts(
        part_states=part_states,
        active_obligations=active_obligations,
        resource_snapshots=resource_snapshots,
        capability_degradations=capability_degradations,
        product_geometry=product_geometry,
        resource_agents=resource_agents,
    )

    return RecoveryContext(
        resource_snapshots=resource_snapshots,
        part_states=part_states,
        pending_tasks=pending_tasks,
        active_obligations=active_obligations,
        goal_state=goal_state,
        priority_context=priority_context,
        available_task_actions=available_task_actions,
        available_primitives=available_primitives,
        capability_degradations=capability_degradations,
        observation_store=dict(observation_store or {}),
        discovered_constraints=list(discovered_constraints or []),
        grounded_environment_facts=grounded_environment_facts,
    )


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _build_resource_snapshots(
    bridge_resources: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Extract canonical snapshot per resource — symmetric, no focused JID."""
    snapshots: dict[str, dict[str, Any]] = {}
    for resource_jid, raw_entry in bridge_resources.items():
        jid = str(resource_jid or "").strip()
        if not jid:
            continue
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        snapshot = dict(entry.get("bridge_snapshot") or {})
        modeled_state = dict(entry.get("modeled_state") or {})
        resource_type = str(
            snapshot.get("resource_type")
            or dict(snapshot.get("resource_core") or {}).get("resource_type")
            or entry.get("resource_type")
            or "resource"
        ).strip().lower() or "resource"

        canonical = canonical_bridge_resource(
            resource_jid=jid,
            resource_type=resource_type,
            snapshot=snapshot,
            modeled_state=modeled_state,
        )
        resource_core = dict(canonical.get("resource_core") or {})
        resource_facets = dict(canonical.get("resource_facets") or {})

        profile = get_resource_profile(resource_type)
        # Collect all field names from resource_core + resource_facets.
        all_field_names: list[str] = list(resource_core.keys())
        for facet in resource_facets.values():
            if isinstance(facet, dict):
                all_field_names.extend(facet.keys())
        flat_fields = resource_snapshot_fields_map(
            canonical, tuple(all_field_names), profile=profile,
        )

        snapshots[jid] = {
            "resource_type": resource_type,
            "resource_jid": jid,
            "current_state": (
                resource_core.get("current_state")
                or canonical.get("current_state")
                or snapshot.get("current_state")
                or modeled_state.get("resource_state")
            ),
            "occupancy": deepcopy(
                resource_core.get("occupancy")
                or canonical.get("occupancy")
                or {}
            ),
            "resource_core": deepcopy(resource_core),
            "resource_facets": deepcopy(resource_facets),
            **flat_fields,
        }
    return snapshots


def _build_part_states(
    part_tracker: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Extract part states from the part tracker."""
    parts: dict[str, dict[str, Any]] = {}
    for part_name, raw_info in part_tracker.items():
        name = str(part_name or "").strip()
        if not name:
            continue
        info = raw_info if isinstance(raw_info, dict) else {}
        parts[name] = {
            "state": info.get("state"),
            "location": info.get("location"),
            **{k: v for k, v in info.items() if k not in ("state", "location")},
        }
    return parts


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _geometry_for_part(
    product_geometry: dict[str, Any] | None,
    *,
    part_name: str,
    expected_location: str,
) -> dict[str, Any] | None:
    token = str(expected_location or "").strip().lower()
    if not token or "assembly_board" not in token:
        return None
    if not isinstance(product_geometry, dict):
        return None

    board = dict(product_geometry.get("assembly_board") or {})
    parts = dict(product_geometry.get("parts") or {})
    slot_xy = dict(board.get("slots") or {}).get(part_name)
    if not isinstance(slot_xy, (list, tuple)) or len(slot_xy) < 2:
        return None

    return {
        "slot_xy": list(slot_xy[:2]),
        "part_height_m": dict(parts.get("heights_m") or {}).get(part_name),
        "model_name": dict(parts.get("model_map") or {}).get(part_name),
        "slot_floor_z_m": board.get("slot_floor_z_m"),
        "board_center": board.get("center") or {},
    }


def _default_preview_snapshot(resource_snapshots: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for snap in resource_snapshots.values():
        row = dict(snap or {})
        if str(row.get("resource_type") or "").strip().lower() == "robot":
            return row
    return {
        "resource_type": "robot",
        "resource_core": {"resource_type": "robot"},
    }


def _preview_place_targets_for_observed_part(
    *,
    part_name: str,
    expected_location: str,
    observed_pose: dict[str, Any] | None,
    product_geometry: dict[str, Any] | None,
    resource_snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    geometry = _geometry_for_part(
        product_geometry,
        part_name=part_name,
        expected_location=expected_location,
    )
    if geometry is None or not isinstance(observed_pose, dict):
        return None

    preview_snapshot = _default_preview_snapshot(resource_snapshots)
    pick_ctx, pick_err = preview_step_output(
        primitive="compute_pick_targets",
        params={"part_name": part_name, "product_geometry": geometry},
        snapshot=preview_snapshot,
        grounding_context={"parts": {part_name: {"observed_pose": observed_pose}}},
    )
    if pick_err or not isinstance(pick_ctx, dict):
        return None

    place_preview, place_err = preview_step_output(
        primitive="compute_place_targets",
        params={
            "part_name": part_name,
            "product_geometry": geometry,
            "pick_ctx": pick_ctx,
        },
        snapshot=preview_snapshot,
        grounding_context={},
    )
    if place_err or not isinstance(place_preview, dict):
        return None
    return place_preview


def _is_explicit_non_assembly_location_token(value: Any) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    lower = token.lower()
    if lower in {"unknown", "none", "null"}:
        return False
    if "assembly_board" in lower:
        return False
    if "_gripper" in lower or lower.endswith("gripper"):
        return False
    return True


def _explicit_staging_destination_for_part(info: dict[str, Any]) -> str:
    for key in (
        "staging_destination",
        "origin_resource_location",
        "last_known_location",
        "location",
    ):
        value = info.get(key)
        if _is_explicit_non_assembly_location_token(value):
            return str(value).strip()
    return ""


def _staging_destination_detail_for_part(
    *,
    part_name: str,
    destination: str,
    resource_snapshots: dict[str, dict[str, Any]],
    resource_agents: dict[str, Any] | None,
    product_geometry: dict[str, Any] | None,
) -> dict[str, Any]:
    token = str(destination or "").strip()
    if not part_name or not token:
        return {}

    geometry = resolve_place_geometry(
        part_name=part_name,
        destination_location=token,
        execution_mode="simulation",
    )
    geometry_backed = has_place_geometry_fields(geometry)
    detail = {
        "destination": token,
        "placement_support": (
            "geometry_backed"
            if geometry_backed
            else "anchor_only"
        ),
    }
    if geometry_backed:
        detail["preferred_release_pattern"] = (
            "compute_place_targets_then_approach_target_release"
        )
    else:
        detail["preferred_release_pattern"] = (
            "move_to_destination_descend_release_retreat"
        )
        explicit_geometry = _staging_anchor_geometry_for_part(
            part_name=part_name,
            destination=token,
            resource_snapshots=resource_snapshots,
            resource_agents=resource_agents,
            product_geometry=product_geometry,
        )
        if explicit_geometry:
            detail["explicit_product_geometry"] = explicit_geometry
            detail["preferred_release_pattern"] = (
                "compute_place_targets_with_explicit_product_geometry"
            )
    return detail


def _carrier_resource_for_part(
    *,
    part_name: str,
    resource_snapshots: dict[str, dict[str, Any]],
) -> str:
    for resource_jid, snapshot in dict(resource_snapshots or {}).items():
        if str(dict(snapshot or {}).get("held_part") or "").strip() == str(part_name or "").strip():
            return str(resource_jid or "").strip()
    return ""


def _part_model_defaults(
    product_geometry: dict[str, Any] | None,
    *,
    part_name: str,
) -> dict[str, Any]:
    parts = dict((product_geometry or {}).get("parts") or {})
    return {
        "part_height_m": dict(parts.get("heights_m") or {}).get(part_name),
        "model_name": dict(parts.get("model_map") or {}).get(part_name),
    }


def _normalize_staging_area_geometry(
    area: dict[str, Any],
    *,
    destination: str,
    part_name: str,
    product_geometry: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(area, dict):
        return {}

    geometry = dict(area or {})
    anchor_pose = dict(geometry.get("anchor_pose") or {})
    board_center = dict(geometry.get("board_center") or {})
    if not board_center and anchor_pose:
        board_center = {
            axis: anchor_pose.get(axis)
            for axis in ("x", "y", "z")
            if anchor_pose.get(axis) is not None
        }
    slot_xy = geometry.get("slot_xy")
    if slot_xy is None:
        slot_xy = [0.0, 0.0]

    normalized = {
        "anchor_location": str(destination or "").strip(),
        "slot_xy": list(slot_xy[:2]) if isinstance(slot_xy, (list, tuple)) and len(slot_xy) >= 2 else [0.0, 0.0],
        "board_center": board_center,
        "slot_floor_z_m": geometry.get("slot_floor_z_m", board_center.get("z")),
    }
    defaults = _part_model_defaults(product_geometry, part_name=part_name)
    part_height = geometry.get("part_height_m", defaults.get("part_height_m"))
    model_name = geometry.get("model_name", defaults.get("model_name"))
    if part_height is not None:
        normalized["part_height_m"] = part_height
    if model_name:
        normalized["model_name"] = model_name
    if not has_place_geometry_fields(normalized):
        return {}
    return normalized


def _staging_anchor_geometry_for_part(
    *,
    part_name: str,
    destination: str,
    resource_snapshots: dict[str, dict[str, Any]],
    resource_agents: dict[str, Any] | None,
    product_geometry: dict[str, Any] | None,
) -> dict[str, Any]:
    token = str(destination or "").strip()
    if not token:
        return {}

    preferred_resource = _carrier_resource_for_part(
        part_name=part_name,
        resource_snapshots=resource_snapshots,
    )
    candidate_jids: list[str] = []
    if preferred_resource:
        candidate_jids.append(preferred_resource)
    for resource_jid in dict(resource_agents or {}).keys():
        jid = str(resource_jid or "").strip()
        if jid and jid not in candidate_jids:
            candidate_jids.append(jid)

    for resource_jid in candidate_jids:
        agent = dict(resource_agents or {}).get(resource_jid)
        caps = getattr(agent, "static_capabilities", None) if agent is not None else None
        staging_areas = dict(caps.get("staging_areas") or {}) if isinstance(caps, dict) else {}
        area = dict(staging_areas.get(token) or {})
        normalized = _normalize_staging_area_geometry(
            area,
            destination=token,
            part_name=part_name,
            product_geometry=product_geometry,
        )
        if normalized:
            return normalized
    return {}


def _build_grounded_environment_facts(
    *,
    part_states: dict[str, dict[str, Any]],
    active_obligations: list[dict[str, Any]],
    resource_snapshots: dict[str, dict[str, Any]],
    capability_degradations: list[dict[str, Any]],
    product_geometry: dict[str, Any] | None,
    resource_agents: dict[str, Any] | None,
) -> dict[str, Any]:
    observed_part_sources: dict[str, dict[str, Any]] = {}
    required_destinations: dict[str, str] = {}
    placement_previews: dict[str, dict[str, Any]] = {}
    staging_destinations: dict[str, str] = {}
    staging_destination_details: dict[str, dict[str, Any]] = {}
    resource_capability_facts: dict[str, dict[str, Any]] = {}
    degradation_facts: list[dict[str, Any]] = []
    resume_reentry_facts: list[dict[str, Any]] = []

    for part_name, info in part_states.items():
        row = dict(info or {})
        state = str(row.get("state") or "").strip().lower()
        if (
            part_state_requires_external_localization(state)
            and part_observation_status(row) == "observed"
        ):
            observed_pose = dict(row.get("observed_pose") or {})
            source_row: dict[str, Any] = {
                "pose_source": str(row.get("pose_source") or "").strip(),
            }
            if observed_pose:
                source_row["observed_pose"] = {
                    axis: _float_or_none(observed_pose.get(axis))
                    for axis in ("x", "y", "z")
                    if _float_or_none(observed_pose.get(axis)) is not None
                }
            observed_part_sources[str(part_name)] = source_row
        if part_state_is_carried(state):
            staging_destination = _explicit_staging_destination_for_part(row)
            if staging_destination:
                staging_destinations[str(part_name)] = staging_destination
                detail = _staging_destination_detail_for_part(
                    part_name=str(part_name),
                    destination=staging_destination,
                    resource_snapshots=resource_snapshots,
                    resource_agents=resource_agents,
                    product_geometry=product_geometry,
                )
                if detail:
                    staging_destination_details[str(part_name)] = detail

    for obligation in active_obligations or []:
        ob_type = str(obligation.get("type") or "").strip().lower()
        ob_class = str(obligation.get("obligation_class") or "").strip().lower()
        entity_kind = str(obligation.get("entity_kind") or "").strip()
        entity = str(obligation.get("entity") or "").strip()
        field = str(obligation.get("field") or "").strip()
        expected = str(obligation.get("expected") or "").strip()
        if ob_class == "resume_entry":
            if entity and field:
                resume_reentry_facts.append(
                    {
                        "entity": entity,
                        "field": field,
                        "expected": expected,
                    }
                )
            continue
        if ob_type == "safety":
            continue
        if entity_kind == "part" and field == "location" and expected:
            required_destinations[entity] = expected

    for part_name, expected_location in required_destinations.items():
        info = dict(part_states.get(part_name) or {})
        observed_pose = dict(info.get("observed_pose") or {})
        preview = _preview_place_targets_for_observed_part(
            part_name=part_name,
            expected_location=expected_location,
            observed_pose=observed_pose,
            product_geometry=product_geometry,
            resource_snapshots=resource_snapshots,
        )
        if preview:
            placement_previews[part_name] = {
                "destination": expected_location,
                "approach_pose": dict(preview.get("approach_pose") or {}),
                "target_pose": dict(preview.get("target_pose") or {}),
            }

    for jid, snap in resource_snapshots.items():
        row: dict[str, Any] = {
            "current_state": snap.get("current_state"),
        }
        if snap.get("held_part") is not None:
            row["held_part"] = snap.get("held_part")
        if snap.get("gripper_state") is not None:
            row["gripper_state"] = snap.get("gripper_state")
        occupancy = dict(snap.get("occupancy") or {})
        if occupancy.get("location") is not None:
            row["location"] = occupancy.get("location")
        if row:
            resource_capability_facts[str(jid)] = row

    for item in capability_degradations or []:
        if not isinstance(item, dict):
            continue
        row = {
            "resource_jid": str(item.get("resource_jid") or "").strip(),
            "resource_state": str(item.get("resource_state") or "").strip(),
            "availability": str(item.get("availability") or "").strip(),
            "reason": str(item.get("reason") or "").strip(),
        }
        row = {k: v for k, v in row.items() if v}
        if row:
            degradation_facts.append(row)

    facts: dict[str, Any] = {}
    if observed_part_sources:
        facts["observed_part_sources"] = observed_part_sources
    if required_destinations:
        facts["required_destinations"] = required_destinations
    if placement_previews:
        facts["placement_previews"] = placement_previews
    if staging_destinations:
        facts["staging_destinations"] = staging_destinations
    if staging_destination_details:
        facts["staging_destination_details"] = staging_destination_details
    if resource_capability_facts:
        facts["resource_capability_facts"] = resource_capability_facts
    if degradation_facts:
        facts["degradation_facts"] = degradation_facts
    if resume_reentry_facts:
        facts["resume_reentry_facts"] = resume_reentry_facts
    return facts


def _build_pending_tasks(
    bridge_resources: dict[str, dict[str, Any]],
    planner: Any | None,
) -> list[dict[str, Any]]:
    """Collect pending tasks from all resources (not just the focused one)."""
    pending: list[dict[str, Any]] = []

    # Prefer planner nodes if available.
    if planner is not None:
        nodes = getattr(planner, "nodes", None) or []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if node.get("type") != "task":
                continue
            status = str(node.get("status", "")).strip()
            if status in ("pending", "running"):
                pending.append(deepcopy(node))
        if pending:
            return pending

    # Fallback: extract from bridge_resources pending_tasks.
    for resource_jid, raw_entry in bridge_resources.items():
        jid = str(resource_jid or "").strip()
        if not jid:
            continue
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        for task in (entry.get("pending_tasks") or []):
            if isinstance(task, dict):
                task_copy = deepcopy(task)
                task_copy.setdefault("resource_jid", jid)
                pending.append(task_copy)
    return pending


def _build_active_obligations(
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge obligation targets + safety rules into a flat list."""
    obligations: list[dict[str, Any]] = []

    # Obligation targets (goal-level).
    for target in (prepared_bridge_request.get("obligation_targets") or []):
        if isinstance(target, dict):
            obligations.append(deepcopy(target))

    # Safety rules from bridge_safety_context.
    safety_ctx = prepared_bridge_request.get("bridge_safety_context") or {}
    for rule in (safety_ctx.get("safety_rules") or []):
        if isinstance(rule, dict):
            obligations.append({
                "type": "safety",
                "rule_id": rule.get("id", ""),
                "text": rule.get("text") or rule.get("raw_text") or "",
                "ltlf": rule.get("ltlf", ""),
                "aps": rule.get("aps", []),
            })

    # Marked reentry conditions.
    marked_ctx = prepared_bridge_request.get("marked_reentry_context") or {}
    for cond in (marked_ctx.get("marked_reentry_conditions") or []):
        if isinstance(cond, dict):
            obligations.append({
                "type": "reentry_condition",
                **deepcopy(cond),
            })

    return [_annotate_obligation(obligation) for obligation in obligations]


def _annotate_obligation(
    obligation: dict[str, Any],
) -> dict[str, Any]:
    """Attach generic policy metadata used by prompts and validation."""
    annotated = deepcopy(obligation)
    obligation_type = str(annotated.get("type") or "").strip().lower()
    obligation_kind = str(annotated.get("kind") or "").strip().lower()
    obligation_role = str(annotated.get("role") or "").strip().lower()

    obligation_class = "bridge_goal"
    must_satisfy_before_resume = True
    label = "bridge-goal"

    if obligation_type == "safety":
        obligation_class = "safety"
        label = "safety"
    elif (
        obligation_kind.startswith("resume_entry_")
        or obligation_role == "resume_suffix"
    ):
        obligation_class = "resume_entry"
        label = "resume-precondition"
    elif obligation_kind in (
        "focused_resource_terminal_state",
        "bridge_part_goal",
        "bridge_part_goal_location",
    ) or obligation_role == "bridge_replaced":
        obligation_class = "bridge_goal"
        label = "bridge-goal"

    annotated["obligation_class"] = obligation_class
    annotated["must_satisfy_before_resume"] = must_satisfy_before_resume
    annotated["prompt_label"] = label
    return annotated


def _build_available_primitives(
    bridge_resources: dict[str, dict[str, Any]],
    resource_agents: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Collect primitive catalogs per resource."""
    primitives: dict[str, list[dict[str, Any]]] = {}
    for resource_jid, raw_entry in bridge_resources.items():
        jid = str(resource_jid or "").strip()
        if not jid:
            continue
        entry = raw_entry if isinstance(raw_entry, dict) else {}

        # Use pre-built catalog from bridge request if available.
        catalog = entry.get("primitive_catalog")
        if isinstance(catalog, list) and catalog:
            primitives[jid] = deepcopy(catalog)
            continue

        # Build from agent if available.
        if resource_agents and jid in resource_agents:
            agent = resource_agents[jid]
            try:
                catalog = build_synthesis_primitive_catalog(agent)
                if isinstance(catalog, list):
                    primitives[jid] = catalog
                    continue
            except Exception:
                logger.debug(
                    "Failed to build primitive catalog for %s", jid, exc_info=True,
                )

        primitives[jid] = []
    return primitives


def _build_capability_degradations(
    bridge_resources: dict[str, dict[str, Any]],
    resource_snapshots: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Identify resources with degraded or unavailable capabilities."""
    degradations: list[dict[str, Any]] = []
    for resource_jid, snapshot in resource_snapshots.items():
        current_state = str(snapshot.get("current_state") or "").strip().lower()
        availability = str(
            dict(snapshot.get("resource_core") or {}).get("availability", "")
        ).strip().lower()
        if current_state in ("error", "fault", "offline") or availability in (
            "unavailable", "offline", "degraded",
        ):
            degradations.append({
                "resource_jid": resource_jid,
                "resource_state": current_state,
                "availability": availability,
                "reason": f"state={current_state}, availability={availability}",
            })
    return degradations


def _part_pose_source(info: dict[str, Any]) -> str:
    return str((info or {}).get("pose_source") or "").strip().lower()


def part_pose_is_trusted(info: dict[str, Any]) -> bool:
    observed_pose = (info or {}).get("observed_pose")
    return isinstance(observed_pose, dict) and _part_pose_source(info) in _TRUSTED_POSE_SOURCES


def part_observation_status(info: dict[str, Any]) -> str:
    observed_pose = (info or {}).get("observed_pose")
    if not isinstance(observed_pose, dict):
        return "UNOBSERVED"
    if part_pose_is_trusted(info):
        return "observed"
    return "untrusted_pose"


def _reachability_pose(info: dict[str, Any]) -> dict[str, Any] | None:
    if part_pose_is_trusted(info):
        pose = (info or {}).get("observed_pose")
        if isinstance(pose, dict):
            return deepcopy(pose)
    pose = (info or {}).get("last_known_pose")
    if isinstance(pose, dict):
        return deepcopy(pose)
    return None


def _prompt_obligation_is_currently_satisfied(
    ctx: RecoveryContext,
    obligation: dict[str, Any],
) -> bool:
    entity = str(obligation.get("entity") or "").strip()
    field = str(obligation.get("field") or "").strip()
    expected = obligation.get("expected")
    if not entity or not field:
        return False

    if entity in (ctx.part_states or {}):
        part_entry = dict((ctx.part_states or {}).get(entity) or {})
        return part_entry.get(field) == expected

    resource_entry = dict((ctx.resource_snapshots or {}).get(entity) or {})
    if resource_entry:
        actual = resource_entry.get(field)
        if actual is None and isinstance(resource_entry.get("resource_core"), dict):
            actual = dict(resource_entry.get("resource_core") or {}).get(field)
        return actual == expected
    return False


def _collect_relevant_entities(ctx: RecoveryContext) -> tuple[set[str], set[str]]:
    relevant_parts: set[str] = set()
    relevant_resources: set[str] = set()

    for ob in ctx.active_obligations:
        entity_kind = str(ob.get("entity_kind", "")).strip().lower()
        entity = str(ob.get("entity", "")).strip()
        resource_jid = str(ob.get("resource_jid", "")).strip()

        if entity_kind == "part" and entity:
            relevant_parts.add(entity)
        if entity_kind == "resource" and entity:
            relevant_resources.add(entity)
        if resource_jid:
            relevant_resources.add(resource_jid)

    for part_name, info in ctx.part_states.items():
        state = str((info or {}).get("state", "")).strip().lower()
        if part_state_requires_external_localization(state) or part_state_is_carried(state):
            relevant_parts.add(part_name)

    for resource_jid, snap in ctx.resource_snapshots.items():
        current_state = str((snap or {}).get("current_state", "")).strip().lower()
        held_part = str((snap or {}).get("held_part", "")).strip()
        if current_state not in ("", "idle") or held_part:
            relevant_resources.add(resource_jid)
        if held_part and held_part in relevant_parts:
            relevant_resources.add(resource_jid)

    if not relevant_parts:
        relevant_parts.update(ctx.part_states.keys())
    if not relevant_resources:
        relevant_resources.update(ctx.resource_snapshots.keys())

    return relevant_resources, relevant_parts


def _extract_part_refs_from_value(
    value: Any,
    *,
    known_parts: set[str],
) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            refs.update(
                _extract_part_refs_from_value(item, known_parts=known_parts)
            )
        return refs
    if isinstance(value, (list, tuple, set)):
        for item in value:
            refs.update(
                _extract_part_refs_from_value(item, known_parts=known_parts)
            )
        return refs
    if not isinstance(value, str):
        return refs

    text = value.strip()
    if not text:
        return refs
    if text in known_parts:
        refs.add(text)
    for token in re.split(r"[^A-Za-z0-9_@-]+", text):
        if token in known_parts:
            refs.add(token)
    return refs


def _obligation_part_refs(ctx: RecoveryContext) -> set[str]:
    known_parts = {
        str(name).strip()
        for name in (ctx.part_states or {})
        if str(name).strip()
    }
    refs: set[str] = set()
    for obligation in ctx.active_obligations:
        if str(obligation.get("type", "")).strip().lower() == "safety":
            continue
        refs.update(
            _extract_part_refs_from_value(obligation, known_parts=known_parts)
        )
    return refs


def _continuation_part_refs(ctx: RecoveryContext) -> set[str]:
    known_parts = {
        str(name).strip()
        for name in (ctx.part_states or {})
        if str(name).strip()
    }
    refs: set[str] = set()
    for task in ctx.pending_tasks:
        refs.update(
            _extract_part_refs_from_value(
                dict(task or {}).get("params") or task,
                known_parts=known_parts,
            )
        )
    return refs


def _observation_resource_options(
    ctx: RecoveryContext,
    *,
    relevant_resources: set[str],
    part_name: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    options: dict[str, list[dict[str, Any]]] = {}
    for resource_jid, catalog in (ctx.available_primitives or {}).items():
        if relevant_resources and resource_jid not in relevant_resources:
            continue
        primitives: list[str] = []
        for entry in catalog or []:
            if not isinstance(entry, dict):
                continue
            semantics = entry.get("bridge_semantics") or {}
            if not bool(semantics.get("top_level_observation_admissible")):
                continue
            if part_name and not primitive_supports_part_grounding(entry):
                continue
            primitive_name = str(entry.get("name") or "").strip()
            if primitive_name:
                primitives.append(primitive_name)
        if primitives:
            options[resource_jid] = [{
                "resource_jid": resource_jid,
                "primitives": sorted(set(primitives)),
            }]
    return options


def _rank_observers_for_part(
    part_name: str,
    *,
    ctx: RecoveryContext,
    observation_options: dict[str, list[dict[str, Any]]],
    relevant_resources: set[str],
    trusted_reachability: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Rank observers by executability value instead of generic visibility."""
    ranked: list[dict[str, Any]] = []
    reachability_by_resource = {
        str(item.get("resource_jid") or "").strip(): dict(item)
        for item in (trusted_reachability.get(part_name) or [])
        if str(item.get("resource_jid") or "").strip()
    }

    for resource_jid, rows in observation_options.items():
        if relevant_resources and resource_jid not in relevant_resources:
            continue
        snapshot = dict(ctx.resource_snapshots.get(resource_jid) or {})
        current_state = str(snapshot.get("current_state") or "").strip().lower()
        held_part = str(snapshot.get("held_part") or "").strip()
        score = 0
        reasons: list[str] = []

        if current_state in ("error", "fault", "offline", "recovery_required", "failed"):
            score -= 90
            reasons.append(f"resource state '{current_state}' is degraded")
        elif current_state in ("idle", ""):
            score += 25
            reasons.append("resource is immediately available for sensing")
        else:
            score += 10
            reasons.append(f"resource state '{current_state}' remains usable")

        if held_part and held_part != part_name:
            score -= 10
            reasons.append(f"already carrying {held_part}")

        reachability = reachability_by_resource.get(resource_jid) or {}
        if "reachable" in reachability:
            if bool(reachability.get("reachable")):
                score += 35
                reasons.append(
                    "trusted reachability suggests this resource can manipulate the grounded entity next"
                )
            else:
                score -= 20
                reasons.append(str(reachability.get("reason") or "outside trusted workspace bounds"))

        for row in rows:
            candidate = deepcopy(row)
            primitives = [
                str(name).strip()
                for name in (candidate.get("primitives") or [])
                if str(name).strip()
            ]
            candidate["primitives"] = primitives
            if primitives:
                candidate["recommended_primitive"] = primitives[0]
            candidate["score"] = score
            candidate["rank_reasons"] = reasons
            ranked.append(candidate)

    ranked.sort(
        key=lambda row: (
            -int(row.get("score") or 0),
            str(row.get("resource_jid") or ""),
        )
    )
    return ranked


def _grounding_gap_blocked_transition(
    part_name: str,
    *,
    needs_continuation_grounding: bool,
    needs_obligation_grounding: bool,
    needs_pose_grounding: bool,
) -> str:
    if needs_continuation_grounding or needs_obligation_grounding:
        return (
            f"choose the next executable manipulation for {part_name} "
            "without depending on an untrusted pose"
        )
    if needs_pose_grounding:
        return f"ground the pose-dependent recovery branch for {part_name}"
    return f"resolve the next uncertain transition involving {part_name}"


def _trusted_reachability_hints(
    ctx: RecoveryContext,
    *,
    relevant_resources: set[str],
    require_trusted_pose: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    hints: dict[str, list[dict[str, Any]]] = {}
    for part_name, info in ctx.part_states.items():
        if require_trusted_pose and not part_pose_is_trusted(info):
            continue
        pose = _reachability_pose(info)
        if not isinstance(pose, dict):
            continue
        pose_label = "trusted pose" if part_pose_is_trusted(info) else "last-known pose"
        part_hints: list[dict[str, Any]] = []
        for resource_jid, snap in ctx.resource_snapshots.items():
            if relevant_resources and resource_jid not in relevant_resources:
                continue
            bounds = snap.get("workspace_bounds")
            if not isinstance(bounds, dict):
                continue
            violations: list[str] = []
            for axis in ("x", "y", "z"):
                val = pose.get(axis)
                if val is None:
                    continue
                lo = bounds.get(f"{axis}_min_m")
                hi = bounds.get(f"{axis}_max_m")
                if lo is not None and float(val) < float(lo):
                    violations.append(f"{axis}={val} < {axis}_min_m={lo}")
                if hi is not None and float(val) > float(hi):
                    violations.append(f"{axis}={val} > {axis}_max_m={hi}")
            part_hints.append({
                "resource_jid": resource_jid,
                "reachable": not violations,
                "reason": (
                    f"within workspace_bounds ({pose_label})"
                    if not violations
                    else "; ".join(violations) + f" ({pose_label})"
                ),
            })
        if part_hints:
            hints[str(part_name)] = part_hints
    return hints


def _resource_is_degraded(
    ctx: RecoveryContext,
    resource_jid: str,
) -> bool:
    snapshot = dict(ctx.resource_snapshots.get(resource_jid) or {})
    current_state = str(snapshot.get("current_state") or "").strip().lower()
    availability = str(
        dict(snapshot.get("resource_core") or {}).get("availability") or ""
    ).strip().lower()
    if current_state in _DEGRADED_RESOURCE_STATES:
        return True
    if availability in _DEGRADED_AVAILABILITY_STATES:
        return True
    for row in (ctx.capability_degradations or []):
        if str(row.get("resource_jid") or "").strip() == resource_jid:
            return True
    return False


def _executor_feasibility_for_part(
    part_name: str,
    *,
    ctx: RecoveryContext,
    trusted_reachability: dict[str, list[dict[str, Any]]],
    best_effort_reachability: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    trusted_rows = list(trusted_reachability.get(part_name) or [])
    fallback_rows = list(best_effort_reachability.get(part_name) or [])
    info = dict(ctx.part_states.get(part_name) or {})
    if trusted_rows:
        reachability_basis = "trusted_pose"
        rows = trusted_rows
    elif fallback_rows:
        reachability_basis = (
            "best_effort_pose"
            if part_observation_status(info) == "UNOBSERVED"
            else "last_known_pose"
        )
        rows = fallback_rows
    else:
        reachability_basis = "unknown"
        rows = []

    healthy_reachable: list[str] = []
    degraded_reachable: list[str] = []
    for row in rows:
        resource_jid = str(row.get("resource_jid") or "").strip()
        if not resource_jid or not bool(row.get("reachable")):
            continue
        if _resource_is_degraded(ctx, resource_jid):
            degraded_reachable.append(resource_jid)
        else:
            healthy_reachable.append(resource_jid)

    unique_reachable = healthy_reachable + [
        jid for jid in degraded_reachable
        if jid not in healthy_reachable
    ]
    sole_reachable_executor = unique_reachable[0] if len(unique_reachable) == 1 else None

    executor_blocker_class = ""
    executor_blocker_reason = ""
    if reachability_basis == "trusted_pose":
        if not unique_reachable:
            executor_blocker_class = "no_reachable_executor"
            executor_blocker_reason = (
                "trusted reachability finds no resource that can execute the required manipulation"
            )
        elif not healthy_reachable and sole_reachable_executor:
            executor_blocker_class = "sole_executor_degraded"
            executor_blocker_reason = (
                f"trusted reachability leaves only degraded executor {sole_reachable_executor}"
            )
        elif not healthy_reachable and degraded_reachable:
            executor_blocker_class = "all_reachable_executors_degraded"
            executor_blocker_reason = (
                "trusted reachability leaves only degraded reachable executors"
            )
    elif not healthy_reachable and sole_reachable_executor:
        executor_blocker_class = "sole_executor_degraded"
        executor_blocker_reason = (
            f"{reachability_basis.replace('_', '-')} reachability leaves only degraded "
            f"executor {sole_reachable_executor}"
        )
    elif not healthy_reachable and degraded_reachable:
        executor_blocker_class = "all_reachable_executors_degraded"
        executor_blocker_reason = (
            f"{reachability_basis.replace('_', '-')} reachability leaves only degraded reachable executors"
        )

    return {
        "healthy_reachable_executors": sorted(set(healthy_reachable)),
        "degraded_reachable_executors": sorted(set(degraded_reachable)),
        "sole_reachable_executor": sole_reachable_executor,
        "executor_blocker_class": executor_blocker_class,
        "executor_blocker_reason": executor_blocker_reason,
        "reachability_basis": reachability_basis,
    }


def build_grounding_assessment(ctx: RecoveryContext) -> dict[str, Any]:
    """Compute generic grounding gaps and ranked observation candidates."""
    relevant_resources, relevant_parts = _collect_relevant_entities(ctx)
    obligation_parts = _obligation_part_refs(ctx)
    continuation_parts = _continuation_part_refs(ctx)
    trusted_reachability = _trusted_reachability_hints(
        ctx,
        relevant_resources=relevant_resources,
        require_trusted_pose=True,
    )
    best_effort_reachability = _trusted_reachability_hints(
        ctx,
        relevant_resources=relevant_resources,
        require_trusted_pose=False,
    )

    grounding_gaps: list[dict[str, Any]] = []
    candidate_observations: list[dict[str, Any]] = []
    required_parts: list[str] = []

    for part_name in sorted(relevant_parts):
        info = dict(ctx.part_states.get(part_name) or {})
        state = str(info.get("state", "")).strip().lower()
        observation_status = part_observation_status(info)
        if observation_status == "observed":
            continue

        needs_pose_grounding = part_state_requires_external_localization(state)
        needs_continuation_grounding = (
            part_name in continuation_parts
            and not part_state_is_stably_grounded(state)
        )
        needs_obligation_grounding = (
            part_name in obligation_parts
            and not part_state_is_stably_grounded(state)
        )
        needs_grounding = (
            needs_pose_grounding
            or needs_continuation_grounding
            or needs_obligation_grounding
        )
        if not needs_grounding:
            continue

        blocker_reasons: list[str] = []
        blocker_score = 0
        if needs_continuation_grounding:
            blocker_reasons.append("needed by continuation or pending work")
            blocker_score += 100
        if needs_obligation_grounding:
            blocker_reasons.append("referenced by an active non-safety obligation")
            blocker_score += 80
        if needs_pose_grounding:
            blocker_reasons.append(f"state '{state}' is execution-uncertain")
            blocker_score += 60
        if observation_status == "UNOBSERVED":
            blocker_reasons.append("no trusted pose is available")
            blocker_score += 30
        elif observation_status == "untrusted_pose":
            blocker_reasons.append("pose is present but not trusted")
            blocker_score += 20

        if blocker_score <= 0:
            continue

        observation_options = _observation_resource_options(
            ctx,
            relevant_resources=relevant_resources,
            part_name=part_name,
        )
        observers = _rank_observers_for_part(
            part_name,
            ctx=ctx,
            observation_options=observation_options,
            relevant_resources=relevant_resources,
            trusted_reachability=trusted_reachability,
        )
        executor_feasibility = _executor_feasibility_for_part(
            part_name,
            ctx=ctx,
            trusted_reachability=trusted_reachability,
            best_effort_reachability=best_effort_reachability,
        )
        observation_admissible = True
        observation_policy_reason = ""
        blocker_class = str(executor_feasibility.get("executor_blocker_class") or "").strip()
        if blocker_class == "sole_executor_degraded":
            observation_admissible = False
            observation_policy_reason = (
                "the sole trusted reachable executor is degraded, so capability restoration "
                "or suffix replacement must be planned before new manipulation grounding"
            )
        elif blocker_class in {"all_reachable_executors_degraded", "no_reachable_executor"}:
            observation_admissible = False
            observation_policy_reason = str(
                executor_feasibility.get("executor_blocker_reason") or ""
            ).strip() or "no healthy executable branch is currently available"
        elif not observers:
            observation_admissible = False
            observation_policy_reason = (
                "no relevant resource currently advertises a top-level sensing primitive "
                "for this grounding gap"
            )

        if observers and observation_admissible:
            blocker_score += min(10, len(observers) * 3)
        elif not observers:
            blocker_reasons.append(
                "no relevant resource currently advertises top-level sensing primitives"
            )
        elif observation_policy_reason:
            blocker_reasons.append(observation_policy_reason)

        blocked_transition = _grounding_gap_blocked_transition(
            part_name,
            needs_continuation_grounding=needs_continuation_grounding,
            needs_obligation_grounding=needs_obligation_grounding,
            needs_pose_grounding=needs_pose_grounding,
        )
        gap = {
            "part_name": part_name,
            "state": info.get("state"),
            "location_summary": _prompt_location_summary(info.get("location")),
            "observation_status": observation_status,
            "pose_source": info.get("pose_source"),
            "blocked_transition": blocked_transition,
            "affected_entities": [part_name],
            "why_state_is_insufficient": "; ".join(blocker_reasons),
            "smallest_admissible_observation_batch": 1,
            "reasons": blocker_reasons,
            "priority_score": blocker_score,
            **executor_feasibility,
            "observation_admissible": observation_admissible,
        }
        if observation_policy_reason:
            gap["observation_policy_reason"] = observation_policy_reason
        if info.get("last_known_location"):
            gap["last_known_location_summary"] = _prompt_location_summary(
                info.get("last_known_location")
            )
        if trusted_reachability.get(part_name):
            gap["trusted_reachability"] = deepcopy(
                trusted_reachability[part_name]
            )
        elif best_effort_reachability.get(part_name):
            gap["best_effort_reachability"] = deepcopy(
                best_effort_reachability[part_name]
            )
        grounding_gaps.append(gap)

        if observation_admissible and observers:
            required_parts.append(part_name)
            candidate = deepcopy(gap)
            candidate["recommended_observer"] = deepcopy(observers[0])
            candidate["observer_options"] = deepcopy(observers)
            candidate_observations.append(candidate)

    candidate_observations.sort(
        key=lambda row: (-int(row.get("priority_score") or 0), str(row.get("part_name") or "")),
    )
    grounding_gaps.sort(
        key=lambda row: (-int(row.get("priority_score") or 0), str(row.get("part_name") or "")),
    )

    return {
        "required_observation_parts": [
            str(part_name).strip()
            for part_name in required_parts
            if str(part_name).strip()
        ],
        "executor_first_parts": [
            str(item.get("part_name") or "").strip()
            for item in grounding_gaps
            if str(item.get("part_name") or "").strip()
            and str(item.get("executor_blocker_class") or "").strip()
            in {
                "sole_executor_degraded",
                "all_reachable_executors_degraded",
                "no_reachable_executor",
            }
        ],
        "grounding_gaps": grounding_gaps,
        "candidate_observations": candidate_observations,
        "semantic_observation_candidates": (
            semantic_observation_candidates_from_grounding_candidates(
                candidate_observations,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Serialization for LLM prompt
# ---------------------------------------------------------------------------

def recovery_context_to_prompt_dict(ctx: RecoveryContext) -> dict[str, Any]:
    """Serialize a :class:`RecoveryContext` to a compact dict for LLM prompt.

    Strips internal fields that the LLM does not need to see.
    """
    relevant_resources, relevant_parts = _collect_relevant_entities(ctx)

    resources: dict[str, Any] = {}
    for jid, snap in ctx.resource_snapshots.items():
        if jid not in relevant_resources:
            continue
        resources[jid] = {
            "resource_type": snap.get("resource_type"),
            "current_state": snap.get("current_state"),
            "held_part": snap.get("held_part"),
            "gripper_state": snap.get("gripper_state"),
            "occupancy": snap.get("occupancy"),
            "workspace_bounds": snap.get("workspace_bounds"),
        }
        # Only include non-None fields.
        resources[jid] = {k: v for k, v in resources[jid].items() if v is not None}

    parts: dict[str, Any] = {}
    for name, info in ctx.part_states.items():
        if name not in relevant_parts:
            continue
        part_entry: dict[str, Any] = {
            "state": info.get("state"),
            "location_summary": _prompt_location_summary(info.get("location")),
        }
        if info.get("last_known_location"):
            part_entry["last_known_location_summary"] = _prompt_location_summary(
                info.get("last_known_location")
            )
        # Flag observation status for displaced/misplaced parts.
        state = str(info.get("state", "")).strip().lower()
        if part_state_requires_external_localization(state):
            observation_status = part_observation_status(info)
            if observation_status == "UNOBSERVED":
                part_entry["observation_status"] = "UNOBSERVED"
            else:
                part_entry["observation_status"] = observation_status
            pose_source = str(info.get("pose_source", "")).strip()
            if pose_source:
                part_entry["pose_source"] = pose_source
        # Include origin info so the LLM knows where to return/stow a held part.
        origin_loc = str(info.get("origin_resource_location") or "").strip()
        if origin_loc:
            part_entry["origin_resource_location"] = origin_loc
        origin_pose = info.get("origin_pose")
        if isinstance(origin_pose, dict) and {"x", "y", "z"} <= set(origin_pose):
            part_entry["origin_pose"] = origin_pose
        parts[name] = part_entry

    # Reachability analysis: for each critical part, check which resources
    # can reach it based on workspace_bounds vs observed/last-known pose.
    reachability: dict[str, dict[str, Any]] = {}
    for name, info in ctx.part_states.items():
        if name not in relevant_parts:
            continue
        state = str(info.get("state", "")).strip().lower()
        if not part_state_requires_external_localization(state):
            continue
        pose = _reachability_pose(info)
        if not isinstance(pose, dict):
            # No pose available — can't compute reachability.
            reachability[name] = {
                jid: {"reachable": "unknown", "reason": "no pose data available"}
                for jid in relevant_resources
            }
            continue
        pose_source = "trusted observed_pose" if part_pose_is_trusted(info) else "last_known_pose"
        part_reach: dict[str, Any] = {}
        for jid, snap in ctx.resource_snapshots.items():
            if jid not in relevant_resources:
                continue
            bounds = snap.get("workspace_bounds")
            if not isinstance(bounds, dict):
                part_reach[jid] = {
                    "reachable": True,
                    "reason": f"no workspace_bounds configured ({pose_source})",
                }
                continue
            violations: list[str] = []
            for axis in ("x", "y", "z"):
                val = pose.get(axis)
                if val is None:
                    continue
                lo = bounds.get(f"{axis}_min_m")
                hi = bounds.get(f"{axis}_max_m")
                if lo is not None and float(val) < float(lo):
                    violations.append(f"{axis}={val} < {axis}_min_m={lo}")
                if hi is not None and float(val) > float(hi):
                    violations.append(f"{axis}={val} > {axis}_max_m={hi}")
            if violations:
                part_reach[jid] = {
                    "reachable": False,
                    "reason": f"{'; '.join(violations)} ({pose_source})",
                }
            else:
                part_reach[jid] = {
                    "reachable": True,
                    "reason": f"within workspace_bounds ({pose_source})",
                }
        reachability[name] = part_reach

    obligations: list[str] = []
    for ob in ctx.active_obligations:
        ob_type = str(ob.get("type", "")).strip()
        obligation_class = str(ob.get("obligation_class") or "").strip().lower()
        prompt_label = str(ob.get("prompt_label") or "").strip() or "goal"
        if ob_type != "safety" and _prompt_obligation_is_currently_satisfied(ctx, ob):
            continue

        if ob_type == "safety":
            # Prefer human-readable text; fall back to AP descriptions.
            text = str(ob.get("text") or "").strip()
            if not text or text.startswith("("):
                # Build readable description from atomic propositions.
                aps = ob.get("aps") or []
                ap_descs = [
                    str(ap.get("full") or ap.get("label") or "")
                    for ap in aps if isinstance(ap, dict)
                ]
                if ap_descs:
                    text = "constraint over: " + ", ".join(ap_descs)
                elif not text:
                    text = str(ob.get("ltlf") or "").strip()
            if text:
                obligations.append(f"[safety] {text}")

        elif obligation_class == "resume_entry":
            entity = ob.get("entity", "")
            field_name = ob.get("field", "")
            expected = ob.get("expected", "")
            reason = ob.get("reason") or ob.get("source_function_name", "")
            if entity and field_name:
                line = f"[resume-precondition] {entity}.{field_name} must be {expected}"
                if reason:
                    line += f" (required by: {reason})"
                obligations.append(line)

        else:
            entity = ob.get("entity", "")
            field_name = ob.get("field", "")
            expected = ob.get("expected", "")
            if entity and field_name:
                obligations.append(
                    f"[{prompt_label}] {entity}.{field_name} must reach {expected}"
                )

    pending: list[str] = []
    for task in ctx.pending_tasks:
        task_resource = str(task.get("resource_jid", "")).strip()
        if task_resource and task_resource not in relevant_resources:
            continue
        fn = task.get("function_name", "?")
        rjid = task.get("resource_jid", "?")
        pending.append(f"{fn} on {rjid}")
        if len(pending) >= 8:
            break

    result: dict[str, Any] = {
        "resources": resources,
        "parts": parts,
        "obligations": obligations,
        "goal": ctx.goal_state,
    }
    if reachability:
        result["reachability_analysis"] = reachability
    if pending:
        result["pending_tasks"] = pending
    if ctx.capability_degradations:
        result["degraded_resources"] = ctx.capability_degradations
    if ctx.discovered_constraints:
        result["discovered_constraints"] = [
            c.get("constraint", "") for c in ctx.discovered_constraints
            if c.get("constraint")
        ]

    grounding = build_grounding_assessment(ctx)
    if grounding.get("required_observation_parts"):
        result["required_observation_parts"] = deepcopy(
            grounding["required_observation_parts"]
        )
    if grounding.get("executor_first_parts"):
        result["executor_first_parts"] = deepcopy(
            grounding["executor_first_parts"]
        )
    if grounding.get("grounding_gaps"):
        result["grounding_gaps"] = deepcopy(grounding["grounding_gaps"])
    if grounding.get("candidate_observations"):
        result["candidate_observations"] = deepcopy(
            grounding["candidate_observations"]
        )
    if grounding.get("semantic_observation_candidates"):
        result["semantic_observation_candidates"] = deepcopy(
            grounding["semantic_observation_candidates"]
        )

    grounded_facts = dict(ctx.grounded_environment_facts or {})
    prompt_grounded_facts: dict[str, Any] = {}
    if grounded_facts.get("observed_part_sources"):
        prompt_grounded_facts["observed_part_sources"] = {
            str(name): deepcopy(row)
            for name, row in dict(grounded_facts.get("observed_part_sources") or {}).items()
            if str(name) in relevant_parts
        }
    if grounded_facts.get("required_destinations"):
        prompt_grounded_facts["required_destinations"] = {
            str(name): str(value)
            for name, value in dict(grounded_facts.get("required_destinations") or {}).items()
            if str(name) in relevant_parts
        }
    if grounded_facts.get("staging_destinations"):
        prompt_grounded_facts["staging_destinations"] = {
            str(name): str(value)
            for name, value in dict(grounded_facts.get("staging_destinations") or {}).items()
            if str(name) in relevant_parts
        }
    if grounded_facts.get("staging_destination_details"):
        prompt_grounded_facts["staging_destination_details"] = {
            str(name): deepcopy(row)
            for name, row in dict(grounded_facts.get("staging_destination_details") or {}).items()
            if str(name) in relevant_parts
        }
    if grounded_facts.get("resource_capability_facts"):
        prompt_grounded_facts["resource_capability_facts"] = {
            str(jid): deepcopy(row)
            for jid, row in dict(grounded_facts.get("resource_capability_facts") or {}).items()
            if str(jid) in relevant_resources
        }
    if grounded_facts.get("degradation_facts"):
        degrade_rows = []
        for row in list(grounded_facts.get("degradation_facts") or []):
            if not isinstance(row, dict):
                continue
            resource_jid = str(row.get("resource_jid") or "").strip()
            if resource_jid and resource_jid not in relevant_resources:
                continue
            degrade_rows.append(deepcopy(row))
        if degrade_rows:
            prompt_grounded_facts["degradation_facts"] = degrade_rows
    if grounded_facts.get("resume_reentry_facts"):
        reentry_rows = []
        for row in list(grounded_facts.get("resume_reentry_facts") or []):
            if not isinstance(row, dict):
                continue
            entity = str(row.get("entity") or "").strip()
            if entity and entity not in relevant_resources and entity not in relevant_parts:
                continue
            reentry_rows.append(deepcopy(row))
        if reentry_rows:
            prompt_grounded_facts["resume_reentry_facts"] = reentry_rows
    if prompt_grounded_facts:
        # Keep geometry-derived placement previews internal to runtime validation;
        # prompts should only see lower-bias grounded facts such as observations
        # and required destinations.
        result["grounded_environment_facts"] = prompt_grounded_facts

    return result
