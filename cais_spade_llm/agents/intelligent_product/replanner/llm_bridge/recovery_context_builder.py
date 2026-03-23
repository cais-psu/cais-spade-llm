"""Build an obligation-centric RecoveryContext from system state.

This module bridges existing v1 bridge infrastructure (PreparedBridgeRequest,
bridge_session helpers) into the v2 universal repair language.  The key
difference from the existing ``grounding_context`` is that this context is
**obligation-centric** — it presents all resources and obligations equally
rather than centering on one stuck ``ra_jid``.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.mutation_types import (
    RecoveryContext,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_adapters import (
    canonical_bridge_resource,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
    build_primitive_catalog,
    build_synthesis_primitive_catalog,
)
from cais_spade_llm.resources.resource_profile import (
    get_resource_profile,
    resource_snapshot_fields_map,
)

logger = logging.getLogger(__name__)


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

    return obligations


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
                "reason": f"state={current_state}, availability={availability}",
            })
    return degradations


# ---------------------------------------------------------------------------
# Serialization for LLM prompt
# ---------------------------------------------------------------------------

def recovery_context_to_prompt_dict(ctx: RecoveryContext) -> dict[str, Any]:
    """Serialize a :class:`RecoveryContext` to a compact dict for LLM prompt.

    Strips internal fields that the LLM does not need to see.
    """
    resources: dict[str, Any] = {}
    for jid, snap in ctx.resource_snapshots.items():
        resources[jid] = {
            "resource_type": snap.get("resource_type"),
            "current_state": snap.get("current_state"),
            "held_part": snap.get("held_part"),
            "gripper_state": snap.get("gripper_state"),
            "occupancy": snap.get("occupancy"),
        }
        # Only include non-None fields.
        resources[jid] = {k: v for k, v in resources[jid].items() if v is not None}

    parts: dict[str, Any] = {}
    for name, info in ctx.part_states.items():
        parts[name] = {
            "state": info.get("state"),
            "location": info.get("location"),
        }

    obligations: list[str] = []
    for ob in ctx.active_obligations:
        ob_type = str(ob.get("type", "")).strip()
        ob_kind = str(ob.get("kind", "")).strip()

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

        elif ob_type == "reentry_condition" or ob_kind.startswith("resume_entry_"):
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
            # Goal obligations.
            entity = ob.get("entity", "")
            field_name = ob.get("field", "")
            expected = ob.get("expected", "")
            if entity and field_name:
                obligations.append(
                    f"[goal] {entity}.{field_name} must reach {expected}"
                )

    pending: list[str] = []
    for task in ctx.pending_tasks[:10]:
        fn = task.get("function_name", "?")
        rjid = task.get("resource_jid", "?")
        pending.append(f"{fn} on {rjid}")

    result: dict[str, Any] = {
        "resources": resources,
        "parts": parts,
        "obligations": obligations,
        "goal": ctx.goal_state,
    }
    if pending:
        result["pending_tasks"] = pending
    if ctx.capability_degradations:
        result["degraded_resources"] = ctx.capability_degradations
    if ctx.discovered_constraints:
        result["discovered_constraints"] = [
            c.get("constraint", "") for c in ctx.discovered_constraints
            if c.get("constraint")
        ]

    return result
