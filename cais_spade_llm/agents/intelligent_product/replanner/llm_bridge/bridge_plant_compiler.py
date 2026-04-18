"""Compile and validate an LLM-generated recovery plant automaton.

The plant compiler sits between the LLM domain-generation output and the
DES solver.  It ensures every state, event, and transition in the plant
uses vocabulary that is grounded in the current world state (known
resources, parts, locations, workspace bounds).

Validation is split into two stages:
  Level 1 — vocabulary & structure (before solving)
  Level 2 — physical feasibility (after solution trace is found, delegated
             to the existing CCA / robot-agent validators)
"""

from __future__ import annotations

import json
import logging
from collections import deque
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_contract_semantics import (
    apply_event_contract_effects,
    event_location_ref,
    infer_event_semantics,
    normalize_event_contract,
    normalize_state_metadata_entry,
    snapshot_signature,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation finding helper
# ---------------------------------------------------------------------------


def _plant_finding(
    *,
    constraint_code: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
    event_name: str | None = None,
    severity: str = "error",
) -> dict[str, Any]:
    """Build a structured validation finding for plant compilation."""
    finding: dict[str, Any] = {
        "constraint_owner": "plant_compiler",
        "constraint_family": "plant_validation",
        "constraint_code": constraint_code,
        "reason": reason,
        "severity": str(severity or "error"),
    }
    if evidence:
        finding["evidence"] = deepcopy(evidence)
    if event_name:
        finding["event_name"] = event_name
    return finding


# ---------------------------------------------------------------------------
# Known vocabulary extraction
# ---------------------------------------------------------------------------


def _collect_known_locations(
    bridge_resources: dict[str, Any],
    symbolic_parts: dict[str, Any],
) -> set[str]:
    """Collect all location tokens that are considered grounded."""
    locations: set[str] = set()
    for jid, res in (bridge_resources or {}).items():
        res = dict(res or {})
        if jid:
            locations.add(jid)
            locations.add(f"{jid}_gripper")
        current_location = str(res.get("current_location") or "").strip()
        if current_location:
            locations.add(current_location)
        named_poses = res.get("named_poses")
        if isinstance(named_poses, dict):
            locations.update(
                str(k).strip() for k in named_poses if str(k).strip()
            )
        elif isinstance(named_poses, list):
            locations.update(
                str(p).strip() for p in named_poses if str(p).strip()
            )
    for pname, prow in (symbolic_parts or {}).items():
        prow = dict(prow or {})
        for field in (
            "current_location",
            "origin_location",
            "goal_location",
        ):
            token = str(prow.get(field) or "").strip()
            if token:
                locations.add(token)
        if prow.get("observed_pose") and isinstance(prow["observed_pose"], dict):
            locations.add("observed_pose")
            locations.add(f"{pname}_observed_pose")
    return locations


def _collect_known_named_poses(
    bridge_resources: dict[str, Any],
) -> dict[str, set[str]]:
    """Return {resource_jid: set_of_named_pose_tokens}."""
    result: dict[str, set[str]] = {}
    for jid, res in (bridge_resources or {}).items():
        res = dict(res or {})
        poses: set[str] = set()
        named_poses = res.get("named_poses")
        if isinstance(named_poses, dict):
            poses.update(str(k).strip() for k in named_poses if str(k).strip())
        elif isinstance(named_poses, list):
            poses.update(str(p).strip() for p in named_poses if str(p).strip())
        result[jid] = poses
    return result


def _workspace_bounds_for_resource(
    resource_jid: str,
    bridge_resources: dict[str, Any],
) -> dict[str, float] | None:
    """Return workspace bounds dict for a resource, or None."""
    res = dict((bridge_resources or {}).get(resource_jid) or {})
    caps = dict(res.get("static_capabilities") or res.get("capabilities") or {})
    bounds = caps.get("workspace_bounds")
    if not bounds:
        bounds = res.get("workspace_bounds")
    if isinstance(bounds, dict) and bounds:
        return dict(bounds)
    return None


def _is_pose_in_workspace(
    pose: dict[str, Any],
    bounds: dict[str, float],
) -> tuple[bool, str]:
    """Check if a pose is within workspace bounds."""
    violations: list[str] = []
    for axis in ("x", "y", "z"):
        val = pose.get(axis)
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        lo = bounds.get(f"{axis}_min_m")
        hi = bounds.get(f"{axis}_max_m")
        if lo is not None and val < float(lo):
            violations.append(f"{axis}={val:.4f} < {axis}_min_m={float(lo):.4f}")
        if hi is not None and val > float(hi):
            violations.append(f"{axis}={val:.4f} > {axis}_max_m={float(hi):.4f}")
    if violations:
        return False, "; ".join(violations)
    return True, ""


# ---------------------------------------------------------------------------
# Plant parsing
# ---------------------------------------------------------------------------


def _parse_plant_events(
    raw_events: list[dict[str, Any]] | dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Normalize events from LLM response into {event_name: event_dict}."""
    events: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    if isinstance(raw_events, dict):
        for name, edict in raw_events.items():
            normalized, legacy_fields = normalize_event_contract(
                {**dict(edict or {}), "name": str(name).strip()},
            )
            event_name = str(normalized.get("name") or "").strip()
            if not event_name:
                continue
            events[event_name] = normalized
            if legacy_fields:
                warnings.append(_plant_finding(
                    constraint_code="deprecated_event_fields",
                    reason=(
                        f"Event '{event_name}' used deprecated field(s): "
                        f"{', '.join(legacy_fields)}. They were accepted for compatibility "
                        "but ignored semantically."
                    ),
                    event_name=event_name,
                    evidence={"deprecated_fields": legacy_fields},
                    severity="warning",
                ))
    elif isinstance(raw_events, list):
        for edict in raw_events:
            normalized, legacy_fields = normalize_event_contract(dict(edict or {}))
            event_name = str(normalized.get("name") or "").strip()
            if not event_name:
                continue
            events[event_name] = normalized
            if legacy_fields:
                warnings.append(_plant_finding(
                    constraint_code="deprecated_event_fields",
                    reason=(
                        f"Event '{event_name}' used deprecated field(s): "
                        f"{', '.join(legacy_fields)}. They were accepted for compatibility "
                        "but ignored semantically."
                    ),
                    event_name=event_name,
                    evidence={"deprecated_fields": legacy_fields},
                    severity="warning",
                ))
    return events, warnings


def _normalize_state_metadata(
    raw_metadata: Any,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Normalize audit-only metadata maps keyed by plant state name."""
    metadata: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    if not isinstance(raw_metadata, dict):
        return metadata, warnings
    for raw_state_name, raw_entry in raw_metadata.items():
        state_name = str(raw_state_name or "").strip()
        if not state_name:
            continue
        entry, legacy_fields = normalize_state_metadata_entry(raw_entry)
        metadata[state_name] = entry
        if legacy_fields:
            warnings.append(_plant_finding(
                constraint_code="deprecated_state_metadata_fields",
                reason=(
                    f"State metadata for '{state_name}' used deprecated field(s): "
                    f"{', '.join(legacy_fields)}. They were accepted for compatibility "
                    "but ignored semantically."
                ),
                evidence={
                    "state_name": state_name,
                    "deprecated_fields": legacy_fields,
                },
                severity="warning",
            ))
    return metadata, warnings


def _build_transition_table(
    events: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[str, str]]]:
    """Build {from_state: [(event_name, to_state), ...]} from event defs."""
    table: dict[str, list[tuple[str, str]]] = {}
    for ename, edict in events.items():
        src = str(edict.get("from") or edict.get("source") or "").strip()
        dst = str(edict.get("to") or edict.get("target") or edict.get("destination") or "").strip()
        if not src or not dst:
            continue
        table.setdefault(src, []).append((ename, dst))
    return table


# ---------------------------------------------------------------------------
# Structure validation
# ---------------------------------------------------------------------------


def _validate_plant_structure(
    plant: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate structural integrity of the plant automaton."""
    findings: list[dict[str, Any]] = []
    states = set(plant.get("states") or set())
    events = dict(plant.get("events") or {})
    initial = str(plant.get("initial") or "").strip()
    marked = set(plant.get("marked") or set())
    transitions = dict(plant.get("transitions") or {})
    state_metadata = dict(plant.get("state_metadata") or {})
    marked_state_metadata = dict(plant.get("marked_state_metadata") or {})

    if not initial:
        findings.append(_plant_finding(
            constraint_code="missing_initial_state",
            reason="Plant has no initial state.",
        ))
    elif initial not in states:
        findings.append(_plant_finding(
            constraint_code="initial_state_not_in_states",
            reason=f"Initial state '{initial}' is not in the state set.",
        ))

    if not marked:
        findings.append(_plant_finding(
            constraint_code="missing_marked_states",
            reason="Plant has no marked (goal) states.",
        ))
    for m in marked:
        if m not in states:
            findings.append(_plant_finding(
                constraint_code="marked_state_not_in_states",
                reason=f"Marked state '{m}' is not in the state set.",
            ))

    for state_name in state_metadata:
        if state_name not in states:
            findings.append(_plant_finding(
                constraint_code="state_metadata_unbound",
                reason=f"State metadata references unknown state '{state_name}'.",
            ))

    for state_name in marked_state_metadata:
        if state_name not in states:
            findings.append(_plant_finding(
                constraint_code="marked_state_metadata_unbound",
                reason=f"Marked-state metadata references unknown state '{state_name}'.",
            ))

    for ename, edict in events.items():
        src = str(edict.get("from") or "").strip()
        dst = str(edict.get("to") or "").strip()
        if src and src not in states:
            findings.append(_plant_finding(
                constraint_code="event_source_not_in_states",
                reason=f"Event '{ename}' source '{src}' not in state set.",
                event_name=ename,
            ))
        if dst and dst not in states:
            findings.append(_plant_finding(
                constraint_code="event_target_not_in_states",
                reason=f"Event '{ename}' target '{dst}' not in state set.",
                event_name=ename,
            ))
        if not src or not dst:
            findings.append(_plant_finding(
                constraint_code="event_missing_endpoints",
                reason=f"Event '{ename}' missing source or target state.",
                event_name=ename,
            ))

    # Check determinism — no two events from the same state with the same name
    for src, edges in transitions.items():
        seen_events: set[str] = set()
        for ename, _ in edges:
            if ename in seen_events:
                findings.append(_plant_finding(
                    constraint_code="nondeterministic_transition",
                    reason=f"State '{src}' has duplicate event '{ename}'.",
                    event_name=ename,
                ))
            seen_events.add(ename)

    # Check reachability from initial to at least one marked state
    if initial and initial in states and marked:
        reachable = _reachable_states(initial, transitions)
        if not (reachable & marked):
            findings.append(_plant_finding(
                constraint_code="marked_unreachable",
                reason="No marked state is reachable from the initial state in the plant.",
                evidence={
                    "reachable": sorted(reachable),
                    "marked": sorted(marked),
                },
            ))

    return findings


def _reachable_states(
    initial: str,
    transitions: dict[str, list[tuple[str, str]]],
) -> set[str]:
    """BFS reachability from initial state."""
    visited: set[str] = set()
    queue: deque[str] = deque([initial])
    while queue:
        s = queue.popleft()
        if s in visited:
            continue
        visited.add(s)
        for _, dst in transitions.get(s, []):
            if dst not in visited:
                queue.append(dst)
    return visited


# ---------------------------------------------------------------------------
# Vocabulary validation
# ---------------------------------------------------------------------------


def _validate_plant_vocabulary(
    plant: dict[str, Any],
    *,
    known_resources: set[str],
    known_parts: set[str],
    known_locations: set[str],
    workspace_bounds_by_resource: dict[str, dict[str, float] | None],
) -> list[dict[str, Any]]:
    """Validate that all event tokens reference grounded vocabulary."""
    findings: list[dict[str, Any]] = []
    events = dict(plant.get("events") or {})

    for ename, edict in events.items():
        resource_jid = str(edict.get("resource_jid") or "").strip()
        part_name = str(edict.get("part_name") or "").strip()
        location_ref = event_location_ref(edict)
        pose = edict.get("pose")

        if resource_jid and resource_jid not in known_resources:
            findings.append(_plant_finding(
                constraint_code="resource_unbound",
                reason=f"Event '{ename}' references unknown resource '{resource_jid}'.",
                event_name=ename,
                evidence={"resource_jid": resource_jid, "known": sorted(known_resources)},
            ))

        if part_name and part_name not in known_parts:
            findings.append(_plant_finding(
                constraint_code="part_unbound",
                reason=f"Event '{ename}' references unknown part '{part_name}'.",
                event_name=ename,
                evidence={"part_name": part_name, "known": sorted(known_parts)},
            ))

        if location_ref and location_ref not in known_locations:
            if not (location_ref.endswith("_observed_pose") or location_ref == "observed_pose"):
                findings.append(_plant_finding(
                    constraint_code="unknown_location_token",
                    reason=f"Event '{ename}' location_ref '{location_ref}' is not a known location.",
                    event_name=ename,
                    evidence={"location_ref": location_ref},
                ))

        # Workspace check if pose provided
        if isinstance(pose, dict) and resource_jid:
            bounds = workspace_bounds_by_resource.get(resource_jid)
            if bounds:
                in_ws, ws_reason = _is_pose_in_workspace(pose, bounds)
                if not in_ws:
                    findings.append(_plant_finding(
                        constraint_code="workspace_unreachable",
                        reason=f"Event '{ename}' pose is outside {resource_jid} workspace: {ws_reason}",
                        event_name=ename,
                        evidence={"pose": pose, "bounds": bounds, "resource_jid": resource_jid},
                    ))

    return findings


def _annotate_events_with_semantics(
    plant: dict[str, Any],
    bridge_resources: dict[str, Any],
    symbolic_parts: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve event grounding and infer rolling symbolic semantics."""
    events = dict(plant.get("events") or {})
    transitions = dict(plant.get("transitions") or {})
    initial = str(plant.get("initial") or "").strip()
    findings: list[dict[str, Any]] = []
    if not initial:
        plant["events"] = events
        return findings

    state_snapshots: dict[str, dict[str, Any]] = {
        initial: snapshot_signature(
            resources=deepcopy(bridge_resources or {}),
            parts=deepcopy(symbolic_parts or {}),
        )
    }
    queue: deque[str] = deque([initial])
    processed_edges: set[tuple[str, str]] = set()

    while queue:
        state_name = queue.popleft()
        snapshot = dict(state_snapshots.get(state_name) or {})
        if not snapshot:
            continue
        pre_resources = deepcopy(dict(snapshot.get("resources") or {}))
        pre_parts = deepcopy(dict(snapshot.get("parts") or {}))
        for ename, dst in transitions.get(state_name, []):
            edge_key = (state_name, ename)
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)
            edict = dict(events.get(ename) or {})
            if not edict:
                continue

            part_name = str(edict.get("part_name") or "").strip()
            location_ref = event_location_ref(edict)
            if (
                part_name
                and location_ref in ("observed_pose", f"{part_name}_observed_pose")
                and not edict.get("pose")
            ):
                observed = dict((symbolic_parts or {}).get(part_name) or {}).get("observed_pose")
                if isinstance(observed, dict) and observed:
                    edict["pose"] = deepcopy(observed)

            next_resources = deepcopy(pre_resources)
            next_parts = deepcopy(pre_parts)
            apply_event_contract_effects(
                edict,
                resources=next_resources,
                parts=next_parts,
            )
            semantics = infer_event_semantics(
                edict,
                pre_resources=pre_resources,
                pre_parts=pre_parts,
                post_resources=next_resources,
                post_parts=next_parts,
            )
            resource_jid = str(edict.get("resource_jid") or "").strip()
            edict["semantic_witness"] = semantics
            edict["pre_state"] = {
                "resource": deepcopy(dict(pre_resources.get(resource_jid) or {})),
                "part": deepcopy(dict(pre_parts.get(part_name) or {})) if part_name else {},
            }
            edict["post_state"] = {
                "resource": deepcopy(dict(next_resources.get(resource_jid) or {})),
                "part": deepcopy(dict(next_parts.get(part_name) or {})) if part_name else {},
            }
            events[ename] = edict

            candidate_snapshot = snapshot_signature(
                resources=next_resources,
                parts=next_parts,
            )
            if dst not in state_snapshots:
                state_snapshots[dst] = candidate_snapshot
                queue.append(dst)
                continue

            current_snapshot = state_snapshots[dst]
            if json.dumps(current_snapshot, sort_keys=True, default=str) != json.dumps(
                candidate_snapshot,
                sort_keys=True,
                default=str,
            ):
                findings.append(_plant_finding(
                    constraint_code="inconsistent_state_merge",
                    reason=(
                        f"Plant state '{dst}' is reached with inconsistent symbolic snapshots. "
                        "All incoming paths to a merged state must agree."
                    ),
                    event_name=ename,
                    evidence={
                        "state_name": dst,
                        "incoming_event": ename,
                        "existing_snapshot": deepcopy(current_snapshot),
                        "candidate_snapshot": deepcopy(candidate_snapshot),
                    },
                ))

    plant["events"] = events
    plant["state_snapshots"] = state_snapshots
    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_plant_from_llm_response(
    raw_response: dict[str, Any] | str,
    bridge_resources: dict[str, Any],
    symbolic_parts: dict[str, Any],
) -> dict[str, Any]:
    """Parse, validate, and annotate an LLM-generated recovery plant.

    Parameters
    ----------
    raw_response
        The JSON-decoded LLM response containing the plant automaton.
    bridge_resources
        ``{resource_jid: resource_dict}`` from the prepared bridge request.
    symbolic_parts
        ``{part_name: part_dict}`` from session state.

    Returns
    -------
    dict with:
        ``status``: ``"valid"`` | ``"invalid"``
        ``plant``: the compiled plant dict (if valid)
        ``findings``: list of validation findings (if any)
    """
    if isinstance(raw_response, str):
        try:
            raw_response = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            return {
                "status": "invalid",
                "plant": None,
                "findings": [_plant_finding(
                    constraint_code="parse_error",
                    reason="LLM response is not valid JSON.",
                )],
            }

    raw_response = dict(raw_response or {})

    # Extract plant from response — may be nested under "plant" key
    plant_raw = raw_response.get("plant") or raw_response

    # Parse states
    raw_states = plant_raw.get("states") or []
    if isinstance(raw_states, list):
        states = {str(s).strip() for s in raw_states if str(s).strip()}
    elif isinstance(raw_states, set):
        states = {str(s).strip() for s in raw_states if str(s).strip()}
    else:
        states = set()

    # Parse events
    raw_events = plant_raw.get("events") or []
    events, event_warnings = _parse_plant_events(raw_events)

    # Infer states from events if states list is empty
    if not states and events:
        for edict in events.values():
            src = str(edict.get("from") or "").strip()
            dst = str(edict.get("to") or "").strip()
            if src:
                states.add(src)
            if dst:
                states.add(dst)

    initial = str(plant_raw.get("initial") or "").strip()
    raw_marked = plant_raw.get("marked") or plant_raw.get("goal_states") or []
    if isinstance(raw_marked, str):
        marked = {raw_marked.strip()}
    elif isinstance(raw_marked, (list, set)):
        marked = {str(m).strip() for m in raw_marked if str(m).strip()}
    else:
        marked = set()

    transitions = _build_transition_table(events)
    state_metadata, state_metadata_warnings = _normalize_state_metadata(plant_raw.get("state_metadata"))
    marked_state_metadata, marked_state_metadata_warnings = _normalize_state_metadata(
        plant_raw.get("marked_state_metadata"),
    )

    plant: dict[str, Any] = {
        "states": states,
        "events": events,
        "initial": initial,
        "marked": marked,
        "transitions": transitions,
        "state_metadata": state_metadata,
        "marked_state_metadata": marked_state_metadata,
    }

    # Validate structure
    all_findings: list[dict[str, Any]] = []
    all_findings.extend(event_warnings)
    all_findings.extend(state_metadata_warnings)
    all_findings.extend(marked_state_metadata_warnings)
    all_findings.extend(_validate_plant_structure(plant))

    # Validate vocabulary
    known_resources = set((bridge_resources or {}).keys())
    known_parts = set((symbolic_parts or {}).keys())
    known_locations = _collect_known_locations(bridge_resources, symbolic_parts)
    workspace_bounds: dict[str, dict[str, float] | None] = {}
    for jid in known_resources:
        workspace_bounds[jid] = _workspace_bounds_for_resource(jid, bridge_resources)

    all_findings.extend(_validate_plant_vocabulary(
        plant,
        known_resources=known_resources,
        known_parts=known_parts,
        known_locations=known_locations,
        workspace_bounds_by_resource=workspace_bounds,
    ))
    all_findings.extend(
        _annotate_events_with_semantics(plant, bridge_resources, symbolic_parts)
    )

    has_errors = any(str(f.get("severity") or "error").strip().lower() != "warning" for f in all_findings)
    status = "invalid" if has_errors else "valid"
    return {
        "status": status,
        "plant": plant,
        "findings": all_findings,
    }


def plant_findings_summary(findings: list[dict[str, Any]]) -> str:
    """Render validation findings as a human-readable summary for LLM feedback."""
    if not findings:
        return "(none)"
    lines: list[str] = []
    for f in findings:
        code = str(f.get("constraint_code") or "").strip()
        reason = str(f.get("reason") or "").strip()
        ename = str(f.get("event_name") or "").strip()
        severity = str(f.get("severity") or "error").strip().lower()
        prefix = f"[{severity}:{code}]" if code else f"[{severity}]"
        event_tag = f" (event: {ename})" if ename else ""
        lines.append(f"- {prefix} {reason}{event_tag}")
    return "\n".join(lines)


__all__ = [
    "compile_plant_from_llm_response",
    "plant_findings_summary",
]
