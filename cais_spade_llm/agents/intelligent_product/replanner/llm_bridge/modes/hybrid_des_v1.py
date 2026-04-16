"""Hybrid DES bridge execution engine: LLM domain author + deterministic DES solver.

The LLM generates a recovery plant automaton (state-transition model for
actions that don't exist in any pre-modeled domain).  The DES solver
composes that plant with safety DFA specifications and finds the optimal
recovery trace via BFS on the product automaton.  When primitive catalogs
are available, a cursor-based primitive grounding phase turns each
validated DES event into executable controller primitive steps.

Phases
------
evaluate_grounding → domain_generation (LLM) → compose_and_solve (DFA)
        → validate_plan → primitive_generation (LLM, optional) → finalize
        ↑                                       |
        └── feedback if plant/solver/validation/primitive grounding fails

Self-contained: no imports from multi_turn or multi_turn_v2.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_des_solver import (
    solver_diagnostic_summary,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_plant_compiler import (
    compile_plant_from_llm_response,
    plant_findings_summary,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
    filter_synthesis_primitive_catalog,
    validate_and_project_steps,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_common import (
    append_revision_entry,
    apply_des_action_effects as _shared_apply_des_action_effects,
    build_des_session_seed,
    build_recovery_gap_state as _shared_build_recovery_gap_state,
    collect_recovery_blockers as _shared_collect_recovery_blockers,
    extract_ap_descriptors as _shared_extract_ap_descriptors,
    extract_safety_dfas as _shared_extract_safety_dfas,
    feasibility_findings_summary as _shared_feasibility_findings_summary,
    handle_compose_and_solve as _shared_handle_compose_and_solve,
    handle_evaluate_grounding as _shared_handle_evaluate_grounding,
    handle_finalize as _shared_handle_finalize,
    handle_validate_plan as _shared_handle_validate_plan,
    revision_history_summary_text,
    validate_action_feasibility as _shared_validate_action_feasibility,
    write_des_per_turn_artifact,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.prompts.hybrid_des_v1 import (
    build_hybrid_des_prompt_input,
    build_hybrid_domain_generation_prompt,
    build_hybrid_feedback_prompt,
    hybrid_des_phase_response_schema,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_TURNS = 6
_DEFAULT_MAX_DOMAIN_REVISIONS = 3

_TRANSITIONS: dict[str, dict[str, str]] = {
    "evaluate_grounding": {
        "grounding_required": "evaluate_grounding",
        "grounding_satisfied": "domain_generation",
    },
    "domain_generation": {
        "plant_valid": "compose_and_solve",
        "plant_invalid": "domain_generation",
    },
    "compose_and_solve": {
        "solved": "validate_plan",
        "unsolvable": "domain_generation",
        "safety_blocked": "domain_generation",
    },
    "validate_plan": {
        "all_feasible": "primitive_generation",
        "all_feasible_no_primitives": "finalize",
        "infeasible": "domain_generation",
    },
    "primitive_generation": {
        "primitive_event_ready": "primitive_generation",
        "need_primitive_revision": "primitive_generation",
        "need_domain_revision": "domain_generation",
        "draft_ready": "finalize",
    },
    "finalize": {
        "accepted": "finalize",
    },
}


def _transition_phase(current_phase: str, decision: str) -> str:
    phase_map = _TRANSITIONS.get(current_phase, {})
    next_phase = phase_map.get(decision)
    if next_phase is None:
        _logger.warning(
            "[HybridDES] No transition for phase=%s decision=%s; staying",
            current_phase, decision,
        )
        return current_phase
    return next_phase


# ---------------------------------------------------------------------------
# Session seed
# ---------------------------------------------------------------------------


def build_hybrid_session_seed(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    """Build the initial session state for a hybrid DES bridge run."""
    seed = build_des_session_seed(
        prepared_bridge_request,
        engine_name="hybrid_des_v1",
    )
    seed["hybrid_engine"] = "des_v1"
    seed["primitive_generation_cursor"] = 0
    seed["accepted_primitive_program"] = []
    seed["primitive_rejection_feedback"] = []
    return seed


# ---------------------------------------------------------------------------
# Observation blockers  (Component 1)
# ---------------------------------------------------------------------------


def _compute_observation_blockers(
    symbolic_parts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Identify parts that need observation before they can be manipulated.

    Detection is purely structural — it does not rely on any hardcoded
    vocabulary of failure-state names. A part is observation-blocked iff
    the system needs to act on it (its current location diverges from its
    goal, or its current location is unknown) AND no usable observed pose
    is available; or the part's ``location_basis`` explicitly requires
    sensor confirmation that has not yet been recorded.
    """
    blockers: list[dict[str, Any]] = []
    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict):
            continue
        observed_pose = row.get("observed_pose")
        has_pose = isinstance(observed_pose, dict) and bool(observed_pose)
        current_location = str(row.get("current_location") or "").strip()
        goal_location = str(row.get("goal_location") or "").strip()
        location_basis = str(row.get("location_basis") or "").strip().lower()

        needs_observation = False
        reason_parts: list[str] = []

        location_unknown = current_location == ""
        diverges_from_goal = (
            current_location != "" and goal_location != ""
            and current_location != goal_location
        )
        needs_action = location_unknown or diverges_from_goal

        if needs_action and not has_pose:
            needs_observation = True
            if location_unknown:
                reason_parts.append(f"part '{part_name}' has no current_location")
            else:
                reason_parts.append(
                    f"part '{part_name}' current_location='{current_location}' "
                    f"differs from goal_location='{goal_location}' but has no observed_pose"
                )

        if location_basis == "sensor_observation" and not has_pose:
            needs_observation = True
            reason_parts.append(
                f"part '{part_name}' location_basis is 'sensor_observation' "
                "but no sensor data available"
            )

        if needs_observation:
            blockers.append({
                "kind": "observation_required",
                "part_name": part_name,
                "description": (
                    f"OBSERVATION REQUIRED: {'; '.join(reason_parts)}. "
                    f"You MUST include a 'detect_parts' or 'observe' event "
                    f"for '{part_name}' BEFORE any pick/place action involving "
                    f"this part. Without observation, the target pose is unknown "
                    f"and the action will fail physical reachability validation."
                ),
                "reason": "; ".join(reason_parts),
            })
    return blockers


# ---------------------------------------------------------------------------
# Additional runtime blockers (terminal / order / workspace / reachability)
# ---------------------------------------------------------------------------


def _compute_terminal_state_blockers(
    symbolic_resources: dict[str, dict[str, Any]],
    *,
    extra_terminal_state_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Resources blocked from executing recovery actions.

    Detection is structural — no hardcoded failure-state vocabulary. A
    resource is terminal-blocked if any of these structural fields say so:
    - ``available`` is explicitly False
    - ``is_blocked`` / ``is_faulted`` / ``is_error`` / ``is_terminal`` is truthy
    - ``fault`` / ``error`` / ``error_code`` is a non-empty value
    - ``current_state`` matches a name supplied via ``extra_terminal_state_names``
      (deployment-level configuration; default empty so no vocabulary is baked in).
    """
    extras = {str(s).strip().lower() for s in (extra_terminal_state_names or set())}
    blockers: list[dict[str, Any]] = []
    for jid, row in (symbolic_resources or {}).items():
        if not isinstance(row, dict):
            continue
        state = str(row.get("current_state") or "").strip().lower()
        reasons: list[str] = []
        if row.get("available") is False:
            reasons.append("available=False")
        for flag in ("is_blocked", "is_faulted", "is_error", "is_terminal"):
            if bool(row.get(flag)):
                reasons.append(f"{flag}=True")
        for fault_field in ("fault", "error", "error_code"):
            value = row.get(fault_field)
            if value not in (None, "", 0, False, [], {}):
                reasons.append(f"{fault_field}={value!r}")
        if state and state in extras:
            reasons.append(f"current_state='{state}' matches configured terminal-state vocabulary")

        if reasons:
            blockers.append({
                "kind": "resource_terminal_state",
                "resource_jid": jid,
                "current_state": state,
                "description": (
                    f"RESOURCE BLOCKED: '{jid}' shows terminal-state evidence "
                    f"({'; '.join(reasons)}). Recovery plant must include a "
                    f"reset/recover transition for this resource before assigning "
                    f"further actions to it, or route the work to a different resource."
                ),
                "reason": "; ".join(reasons),
            })
    return blockers


def _compute_assembly_order_blockers(
    llm_input: dict[str, Any],
    symbolic_parts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parts whose goal references a predecessor that is not yet at its goal.

    Detection is purely structural. A part is "at its goal" iff its
    ``current_location`` equals its ``goal_location`` (both non-empty).
    A part with a ``goal_location`` string that references another part's
    name as a substring (e.g., 'on top of part_X', 'into part_X.cavity')
    declares an order dependency on that predecessor; the blocker fires
    when the predecessor is not yet at its goal. No state-name vocabulary
    is consulted.
    """
    def _at_goal(row: dict[str, Any]) -> bool:
        cur = str(row.get("current_location") or "").strip()
        goal = str(row.get("goal_location") or "").strip()
        return bool(cur) and bool(goal) and cur == goal

    blockers: list[dict[str, Any]] = []
    part_names = [str(p).strip() for p in (symbolic_parts or {}).keys() if p]
    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict):
            continue
        if _at_goal(row):
            continue
        goal_location = str(row.get("goal_location") or "").strip()
        if not goal_location:
            continue
        for predecessor in part_names:
            if predecessor == part_name:
                continue
            if predecessor and predecessor in goal_location:
                pred_row = dict(symbolic_parts.get(predecessor) or {})
                if not _at_goal(pred_row):
                    pred_cur = str(pred_row.get("current_location") or "").strip() or "unknown"
                    pred_goal = str(pred_row.get("goal_location") or "").strip() or "unknown"
                    blockers.append({
                        "kind": "assembly_order",
                        "part_name": part_name,
                        "predecessor": predecessor,
                        "predecessor_current_location": pred_cur,
                        "predecessor_goal_location": pred_goal,
                        "description": (
                            f"ORDER BLOCKED: '{part_name}' goal_location references "
                            f"predecessor '{predecessor}' which is not yet at its goal "
                            f"(current='{pred_cur}', goal='{pred_goal}'). Recovery "
                            f"plant must place '{predecessor}' at its goal before "
                            f"placing '{part_name}', or sequence them in a single "
                            f"accepting trace."
                        ),
                        "reason": (
                            f"part '{part_name}' depends on '{predecessor}' "
                            f"(current='{pred_cur}', goal='{pred_goal}')"
                        ),
                    })
                break
    return blockers


def _compute_shared_workspace_blockers(
    bridge_resources: dict[str, dict[str, Any]],
    symbolic_parts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cross-region deadlock: a part is in resource B's workspace but assigned to A.

    A part with an ``observed_pose`` (or ``current_location``) inside
    resource B's workspace bounds, when its ``assigned_resource_jid`` is
    resource A, indicates a handoff is required and the two-robot
    workspaces are sharing state in a way that can deadlock.
    """
    blockers: list[dict[str, Any]] = []

    def _bounds_of(jid: str) -> dict[str, Any] | None:
        res = dict((bridge_resources or {}).get(jid) or {})
        caps = dict(res.get("static_capabilities") or res.get("capabilities") or {})
        bounds = caps.get("workspace_bounds") or res.get("workspace_bounds")
        return dict(bounds) if isinstance(bounds, dict) else None

    def _pose_in_bounds(pose: dict[str, Any], bounds: dict[str, Any]) -> bool:
        try:
            x, y, z = float(pose["x"]), float(pose["y"]), float(pose["z"])
        except (KeyError, TypeError, ValueError):
            return False
        for axis, val in (("x", x), ("y", y), ("z", z)):
            lo = bounds.get(f"{axis}_min_m")
            hi = bounds.get(f"{axis}_max_m")
            if lo is not None and val < float(lo):
                return False
            if hi is not None and val > float(hi):
                return False
        return True

    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict):
            continue
        assigned = str(row.get("assigned_resource_jid") or "").strip()
        if not assigned:
            continue
        pose = row.get("observed_pose")
        if not isinstance(pose, dict):
            continue
        for jid in bridge_resources or {}:
            if jid == assigned:
                continue
            other_bounds = _bounds_of(jid)
            if not other_bounds:
                continue
            if _pose_in_bounds(pose, other_bounds):
                blockers.append({
                    "kind": "shared_workspace",
                    "part_name": part_name,
                    "assigned_resource_jid": assigned,
                    "host_resource_jid": jid,
                    "description": (
                        f"DEADLOCK CANDIDATE: '{part_name}' is assigned to "
                        f"'{assigned}' but its observed pose lies inside the "
                        f"workspace of '{jid}'. Recovery plant must include "
                        f"a handoff (pick by '{jid}', transfer, place by "
                        f"'{assigned}') or reassignment to clear the deadlock."
                    ),
                    "reason": (
                        f"part '{part_name}' assigned='{assigned}' but "
                        f"in '{jid}' workspace"
                    ),
                })
                break
    return blockers


def _compute_reachability_blockers(
    planner: Any,
    symbolic_parts: dict[str, dict[str, Any]],
    bridge_resources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parts whose current pose lies outside every known resource workspace.

    Without a candidate resource able to reach the pose, no plant can
    succeed without prior relocation/observation.
    """
    blockers: list[dict[str, Any]] = []

    def _bounds(jid: str) -> dict[str, Any] | None:
        res = dict((bridge_resources or {}).get(jid) or {})
        caps = dict(res.get("static_capabilities") or res.get("capabilities") or {})
        bounds = caps.get("workspace_bounds") or res.get("workspace_bounds")
        return dict(bounds) if isinstance(bounds, dict) else None

    def _in(pose: dict[str, Any], bounds: dict[str, Any]) -> bool:
        try:
            x, y, z = float(pose["x"]), float(pose["y"]), float(pose["z"])
        except (KeyError, TypeError, ValueError):
            return False
        for axis, val in (("x", x), ("y", y), ("z", z)):
            lo = bounds.get(f"{axis}_min_m")
            hi = bounds.get(f"{axis}_max_m")
            if lo is not None and val < float(lo):
                return False
            if hi is not None and val > float(hi):
                return False
        return True

    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict):
            continue
        pose = row.get("observed_pose")
        if not isinstance(pose, dict):
            continue
        reachable_jids: list[str] = []
        for jid in bridge_resources or {}:
            b = _bounds(jid)
            if b and _in(pose, b):
                reachable_jids.append(jid)
        if not reachable_jids:
            blockers.append({
                "kind": "reachability",
                "part_name": part_name,
                "observed_pose": deepcopy(pose),
                "description": (
                    f"UNREACHABLE: '{part_name}' observed pose is outside every "
                    f"known resource workspace. Recovery plant cannot reach this "
                    f"pose directly; an external relocation or pose correction "
                    f"event must precede any pick involving '{part_name}'."
                ),
                "reason": f"part '{part_name}' pose outside all workspaces",
            })
    return blockers


def _collect_recovery_blockers(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> list[dict[str, Any]]:
    del planner
    return _shared_collect_recovery_blockers(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )


# ---------------------------------------------------------------------------
# Recovery gap state (resource + part state for prompt)
# ---------------------------------------------------------------------------


def _build_recovery_gap_state(
    session_state: dict[str, Any],
) -> dict[str, Any]:
    """Build the resource/part state for the prompt from session symbolic state."""
    return _shared_build_recovery_gap_state(session_state)


def _plant_summary(plant: dict[str, Any] | None) -> dict[str, Any]:
    plant = dict(plant or {})
    events = plant.get("events") or []
    return {
        "state_count": len(plant.get("states") or []),
        "event_count": len(events) if isinstance(events, list) else len(dict(events)),
        "initial": plant.get("initial"),
        "marked": list(plant.get("marked") or []),
    }


def _has_primitive_catalogs(prepared_bridge_request: dict[str, Any]) -> bool:
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    if bridge_session.get("enable_primitive_generation") is False:
        return False
    for entry in dict(prepared_bridge_request.get("bridge_resources") or {}).values():
        if isinstance(entry, dict) and entry.get("primitive_catalog"):
            return True
    return bool(bridge_session.get("enable_primitive_generation"))


def _report_json(value: Any) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=True)


def _render_compose_report(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
    decision: str,
    next_phase: str,
) -> tuple[str, dict[str, Any]]:
    solver_result = dict(session_state.get("solver_result") or {})
    action_sequence = list(session_state.get("action_sequence") or [])
    safety_dfas = _extract_safety_dfas(planner, prepared_bridge_request)
    result = {
        "phase": "compose_and_solve",
        "llm_call": False,
        "decision": decision,
        "next_phase": next_phase,
        "input_plant_summary": _plant_summary(session_state.get("current_plant")),
        "safety_dfa_count": len(safety_dfas),
        "solver_status": solver_result.get("status"),
        "product_states_explored": solver_result.get("product_states_explored", 0),
        "trace": deepcopy(solver_result.get("trace") or []),
        "action_sequence": deepcopy(action_sequence),
    }
    report = "\n".join([
        "Hybrid DES Compose And Solve Report",
        "No LLM call was made in this phase.",
        f"Decision: {decision}",
        f"Next phase: {next_phase}",
        "",
        "Input Plant Summary",
        _report_json(result["input_plant_summary"]),
        "",
        f"Safety DFA Count: {result['safety_dfa_count']}",
        f"Solver Status: {result['solver_status']}",
        f"Product States Explored: {result['product_states_explored']}",
        "",
        "Selected Trace",
        _report_json(result["trace"]),
        "",
        "Selected Action Sequence",
        _report_json(result["action_sequence"]),
    ])
    return report + "\n", result


def _findings_for_task(
    findings: list[dict[str, Any]],
    task_id: str,
) -> list[dict[str, Any]]:
    return [
        deepcopy(row)
        for row in findings
        if isinstance(row, dict) and str(row.get("task_id") or "").strip() == task_id
    ]


def _validate_plan_steps_report(
    *,
    action_sequence: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    initial_resources: dict[str, dict[str, Any]],
    initial_parts: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    projected_resources = deepcopy(initial_resources)
    projected_parts = deepcopy(initial_parts)
    step_rows: list[dict[str, Any]] = []
    for index, action in enumerate(action_sequence):
        task_id = f"RECOVERY_SEQ{index + 1}"
        resource_jid = str(action.get("resource_jid") or "").strip()
        part_name = str(action.get("part_name") or "").strip()
        resource_before = deepcopy(dict(projected_resources.get(resource_jid) or {}))
        part_before = deepcopy(dict(projected_parts.get(part_name) or {})) if part_name else {}
        step_findings = _findings_for_task(findings, task_id)
        row = {
            "task_id": task_id,
            "action": deepcopy(action),
            "pre_state": {
                "resource": {
                    "resource_jid": resource_jid,
                    "current_state": resource_before.get("current_state"),
                    "current_location": resource_before.get("current_location"),
                    "held_part": resource_before.get("held_part"),
                    "gripper_state": resource_before.get("gripper_state"),
                },
                "part": {
                    "part_name": part_name or None,
                    "current_location": part_before.get("current_location"),
                    "current_holder_resource_jid": part_before.get("current_holder_resource_jid"),
                    "observed_pose": part_before.get("observed_pose"),
                } if part_name else {},
            },
            "findings": step_findings,
        }
        if not step_findings:
            _apply_hybrid_action_effects(
                action,
                resources=projected_resources,
                parts=projected_parts,
            )
            resource_after = deepcopy(dict(projected_resources.get(resource_jid) or {}))
            part_after = deepcopy(dict(projected_parts.get(part_name) or {})) if part_name else {}
            row["projected_effect"] = {
                "resource": {
                    "current_state": resource_after.get("current_state"),
                    "current_location": resource_after.get("current_location"),
                    "held_part": resource_after.get("held_part"),
                    "gripper_state": resource_after.get("gripper_state"),
                },
                "part": {
                    "current_location": part_after.get("current_location"),
                    "current_holder_resource_jid": part_after.get("current_holder_resource_jid"),
                    "part_state": part_after.get("part_state"),
                } if part_name else {},
            }
        step_rows.append(row)
    return step_rows, projected_resources, projected_parts


def _render_validate_report(
    *,
    session_state: dict[str, Any],
    decision: str,
    next_phase: str,
    initial_resources: dict[str, dict[str, Any]],
    initial_parts: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    action_sequence = list(session_state.get("action_sequence") or [])
    findings = [
        deepcopy(row)
        for row in (session_state.get("feasibility_findings") or [])
        if isinstance(row, dict)
    ]
    step_rows, projected_resources, projected_parts = _validate_plan_steps_report(
        action_sequence=action_sequence,
        findings=findings,
        initial_resources=initial_resources,
        initial_parts=initial_parts,
    )
    unassigned_findings = [
        deepcopy(row)
        for row in findings
        if isinstance(row, dict) and not str(row.get("task_id") or "").strip()
    ]
    result = {
        "phase": "validate_plan",
        "llm_call": False,
        "decision": decision,
        "next_phase": next_phase,
        "feasibility_finding_count": len(findings),
        "steps": step_rows,
        "unassigned_findings": unassigned_findings,
        "final_projected_resources": deepcopy(projected_resources),
        "final_projected_parts": deepcopy(projected_parts),
    }
    report = "\n".join([
        "Hybrid DES Validate Plan Report",
        "No LLM call was made in this phase.",
        f"Decision: {decision}",
        f"Next phase: {next_phase}",
        f"Feasibility Findings: {len(findings)}",
        "",
        "Step Checks",
        _report_json(step_rows),
        "",
        "Unassigned Findings",
        _report_json(unassigned_findings),
        "",
        "Final Projected Resource State",
        _report_json(projected_resources),
        "",
        "Final Projected Part State",
        _report_json(projected_parts),
    ])
    return report + "\n", result


def _primitive_catalog_for_resource(
    prepared_bridge_request: dict[str, Any],
    resource_jid: str,
) -> list[dict[str, Any]]:
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    entry = dict(bridge_resources.get(resource_jid) or {})
    return filter_synthesis_primitive_catalog(
        [
            dict(row)
            for row in (entry.get("primitive_catalog") or [])
            if isinstance(row, dict)
        ]
    )


def _primitive_projected_context(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    cursor: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    resources = deepcopy(
        dict(
            session_state.get("primitive_generation_initial_resources")
            or session_state.get("symbolic_resources")
            or {}
        )
    )
    parts = deepcopy(
        dict(
            session_state.get("primitive_generation_initial_parts")
            or session_state.get("symbolic_parts")
            or {}
        )
    )
    for action in list(session_state.get("action_sequence") or [])[:max(cursor, 0)]:
        _apply_hybrid_action_effects(action, resources=resources, parts=parts)
    for resource_jid, entry in dict(prepared_bridge_request.get("bridge_resources") or {}).items():
        if not isinstance(entry, dict):
            continue
        row = resources.setdefault(str(resource_jid), {"resource_jid": str(resource_jid)})
        for field in (
            "resource_type",
            "current_pose",
            "current_pose_ref",
            "workspace_bounds",
            "named_poses",
            "available_named_poses",
            "reachability",
            "staging_areas",
        ):
            if row.get(field) in (None, "", [], {}) and entry.get(field) not in (None, "", [], {}):
                row[field] = deepcopy(entry.get(field))
    return resources, parts


def _active_primitive_event(
    session_state: dict[str, Any],
) -> tuple[int, dict[str, Any] | None, list[dict[str, Any]]]:
    action_sequence = [
        dict(row)
        for row in (session_state.get("action_sequence") or [])
        if isinstance(row, dict)
    ]
    cursor = int(session_state.get("primitive_generation_cursor") or 0)
    if cursor < 0:
        cursor = 0
        session_state["primitive_generation_cursor"] = 0
    if cursor >= len(action_sequence):
        return cursor, None, action_sequence
    active_event = deepcopy(action_sequence[cursor])
    active_event.setdefault("event_index", cursor)
    active_event.setdefault("outline_id", f"RECOVERY_SEQ{cursor + 1}")
    return cursor, active_event, action_sequence


def _render_primitive_generation_prompt(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> str:
    cursor, active_event, action_sequence = _active_primitive_event(session_state)
    resource_jid = str(dict(active_event or {}).get("resource_jid") or "").strip()
    resources, parts = _primitive_projected_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        cursor=cursor,
    )
    primitive_catalog = _primitive_catalog_for_resource(prepared_bridge_request, resource_jid)
    sections = [
        "Task and Role",
        (
            "You are grounding one validated DES recovery event into executable "
            "controller primitive steps.\n"
            "Current phase: Event-to-Primitive Grounding.\n"
            "Implement only the active recovery event; do not re-plan the event sequence."
        ),
        "",
        "Accepted Recovery Event Sequence",
        _report_json(action_sequence),
        "",
        "Active Recovery Event",
        _report_json(active_event or {}),
        "",
        "Event Grounding Cursor",
        _report_json({
            "active_index": cursor,
            "accepted_event_count": len(action_sequence),
            "remaining_events_after_this": max(len(action_sequence) - cursor - 1, 0),
        }),
    ]
    feedback = [
        deepcopy(row)
        for row in (session_state.get("primitive_rejection_feedback") or [])
        if isinstance(row, dict)
    ]
    if feedback:
        sections.extend(["", "Primitive Rejection Feedback", _report_json(feedback)])
    part_name = str(dict(active_event or {}).get("part_name") or "").strip()
    sections.extend([
        "",
        "Current Resource State",
        _report_json(dict(resources.get(resource_jid) or {})),
        "",
        "Current Part State",
        _report_json(dict(parts.get(part_name) or {}) if part_name else {}),
        "",
        "Session Observation Store",
        _report_json(dict(session_state.get("observation_store") or {})),
        "",
        "Active Resource Primitive Catalog",
        _report_json(primitive_catalog),
        "",
        "Output Constraints",
        "- Use only primitives listed in Active Resource Primitive Catalog.",
        "- primitive_steps must be ordered controller primitive calls.",
        "- Each primitive step must include primitive and params.",
        "- Do not invent observations, resources, parts, or grounded locations.",
        "- Use primitive_event_ready when primitive_steps are ready for validation.",
        "- Use need_primitive_revision when prior primitive feedback needs another attempt.",
        "- Use need_domain_revision only when the active recovery event is not implementable with the listed primitives.",
    ])
    return "\n".join(sections).strip() + "\n"


def _primitive_expected_held_part(action: dict[str, Any]) -> Any:
    expected_effect = action.get("expected_effect")
    if isinstance(expected_effect, dict):
        resource_effect = expected_effect.get("resource")
        if isinstance(resource_effect, dict) and "held_part" in resource_effect:
            return resource_effect.get("held_part")
    return "__skip__"


def _record_primitive_domain_revision(
    session_state: dict[str, Any],
    feedback: list[dict[str, Any]],
) -> None:
    """Carry primitive-level impossibility back into domain-generation feedback."""
    session_state["primitive_rejection_feedback"] = deepcopy(feedback)
    session_state["feasibility_findings"] = deepcopy(feedback)
    session_state["domain_revision_count"] = int(
        session_state.get("domain_revision_count") or 0
    ) + 1
    append_revision_entry(
        session_state,
        plant=session_state.get("current_plant"),
        solver_result=session_state.get("solver_result"),
        feasibility_findings=feedback,
    )


async def _handle_primitive_generation(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    del planner
    cursor, active_event, action_sequence = _active_primitive_event(session_state)
    turn_entry: dict[str, Any] = {
        "primitive_generation_cursor": cursor,
        "active_recovery_event": deepcopy(active_event),
        "accepted_action_sequence": deepcopy(action_sequence),
    }
    if active_event is None:
        session_state["status"] = "running"
        return "draft_ready", turn_entry

    resource_jid = str(active_event.get("resource_jid") or "").strip()
    primitive_catalog = _primitive_catalog_for_resource(prepared_bridge_request, resource_jid)
    if not primitive_catalog:
        feedback = [{
            "event_index": cursor,
            "event_name": active_event.get("name"),
            "resource_jid": resource_jid,
            "constraint_code": "primitive_catalog_missing",
            "reason": f"no primitive catalog is available for {resource_jid}",
        }]
        _record_primitive_domain_revision(session_state, feedback)
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        return "need_domain_revision", turn_entry

    response_decision = str(parsed_response.get("decision") or "").strip()
    primitive_steps = [
        dict(row)
        for row in (parsed_response.get("primitive_steps") or [])
        if isinstance(row, dict)
    ]
    turn_entry["primitive_steps"] = deepcopy(primitive_steps)
    if response_decision == "need_domain_revision":
        feedback = [{
            "event_index": cursor,
            "event_name": active_event.get("name"),
            "resource_jid": resource_jid,
            "constraint_code": "domain_revision_requested",
            "reason": "LLM reported the active recovery event is not implementable",
        }]
        _record_primitive_domain_revision(session_state, feedback)
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        return "need_domain_revision", turn_entry

    schema_errors: list[str] = []
    if response_decision not in {"primitive_event_ready", "need_primitive_revision"}:
        schema_errors.append("decision must be primitive_event_ready, need_primitive_revision, or need_domain_revision")
    raw_event_index = parsed_response.get("event_index")
    try:
        event_index = int(raw_event_index) if raw_event_index is not None else -1
    except (TypeError, ValueError):
        event_index = -1
    if event_index != cursor:
        schema_errors.append(f"event_index must match active event index {cursor}")
    if str(parsed_response.get("resource_jid") or "").strip() != resource_jid:
        schema_errors.append(f"resource_jid must match active resource {resource_jid!r}")
    if not primitive_steps:
        schema_errors.append("primitive_steps must contain at least one primitive step")
    for index, step in enumerate(primitive_steps):
        if not str(step.get("primitive") or "").strip():
            schema_errors.append(f"primitive_steps[{index}] must include primitive")
        if not isinstance(step.get("params"), dict):
            schema_errors.append(f"primitive_steps[{index}] must include params object")
    if schema_errors:
        feedback = [{
            "event_index": cursor,
            "event_name": active_event.get("name"),
            "resource_jid": resource_jid,
            "constraint_code": "primitive_schema_violation",
            "reason": "; ".join(schema_errors),
        }]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        return "need_primitive_revision", turn_entry

    resources, parts = _primitive_projected_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        cursor=cursor,
    )
    start_snapshot = deepcopy(dict(resources.get(resource_jid) or {}))
    part_name = str(active_event.get("part_name") or "").strip()
    grounding_context = {
        "active_recovery_event": deepcopy(active_event),
        "resource": deepcopy(start_snapshot),
        "part": deepcopy(dict(parts.get(part_name) or {})) if part_name else {},
        "resources_by_jid": deepcopy(resources),
        "parts_by_name": deepcopy(parts),
        "observation_store": deepcopy(session_state.get("observation_store") or {}),
    }
    valid, projected_snapshot, validation_error = validate_and_project_steps(
        primitive_steps,
        primitive_catalog,
        start_snapshot,
        grounding_context=grounding_context,
    )
    turn_entry["start_snapshot"] = deepcopy(start_snapshot)
    turn_entry["projected_snapshot"] = deepcopy(projected_snapshot)
    if not valid:
        feedback = [{
            "event_index": cursor,
            "event_name": active_event.get("name"),
            "resource_jid": resource_jid,
            "constraint_code": "primitive_validation_failed",
            "reason": validation_error or "primitive validation failed",
        }]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        return "need_primitive_revision", turn_entry

    expected_held = _primitive_expected_held_part(active_event)
    if expected_held != "__skip__" and projected_snapshot.get("held_part") != expected_held:
        feedback = [{
            "event_index": cursor,
            "event_name": active_event.get("name"),
            "resource_jid": resource_jid,
            "constraint_code": "primitive_projection_mismatch",
            "reason": (
                f"projected held_part expected={expected_held!r} "
                f"actual={projected_snapshot.get('held_part')!r}"
            ),
        }]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        return "need_primitive_revision", turn_entry

    accepted_program = list(session_state.get("accepted_primitive_program") or [])
    accepted_row = {
        "event_index": cursor,
        "event_name": active_event.get("name"),
        "resource_jid": resource_jid,
        "part_name": part_name or None,
        "description": str(active_event.get("description") or "").strip(),
        "primitive_steps": deepcopy(primitive_steps),
        "projected_snapshot": deepcopy(projected_snapshot),
    }
    accepted_program.append(accepted_row)
    session_state["accepted_primitive_program"] = deepcopy(accepted_program)
    session_state["primitive_rejection_feedback"] = []
    session_state["primitive_generation_cursor"] = cursor + 1
    turn_entry["accepted_primitive_macro"] = deepcopy(accepted_row)
    if cursor + 1 >= len(action_sequence):
        return "draft_ready", turn_entry
    return "primitive_event_ready", turn_entry


# ---------------------------------------------------------------------------
# Phase handlers
# ---------------------------------------------------------------------------


async def _handle_evaluate_grounding(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle evaluate_grounding phase — check if missing poses require camera execution."""
    session_state["grounding_checkpoint"] = {
        "symbolic_parts": deepcopy(session_state.get("symbolic_parts") or {}),
        "symbolic_resources": deepcopy(session_state.get("symbolic_resources") or {}),
        "current_phase": str(session_state.get("current_phase") or ""),
        "turn_index": int(session_state.get("turn_index") or 0),
    }
    return await _shared_handle_evaluate_grounding(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        engine_name="hybrid",
        session_state_key="hybrid_session_state",
    )


async def _handle_domain_generation(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle domain_generation phase — validate LLM-produced plant."""
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})

    # Extract plant from LLM response
    plant_response = parsed_response.get("plant") or parsed_response
    compile_result = compile_plant_from_llm_response(
        plant_response,
        bridge_resources,
        symbolic_parts,
    )

    turn_entry: dict[str, Any] = {
        "compile_result_status": compile_result["status"],
        "plant_findings": deepcopy(compile_result.get("findings") or []),
    }

    if compile_result["status"] == "valid":
        session_state["current_plant"] = deepcopy(compile_result["plant"])
        session_state["plant_findings"] = []
        session_state["last_rejected_plant"] = None
        _logger.info("[HybridDES] Plant compiled successfully.")
        return "plant_valid", turn_entry

    session_state["last_rejected_plant"] = deepcopy(compile_result.get("plant") or plant_response)
    session_state["plant_findings"] = deepcopy(compile_result.get("findings") or [])
    session_state["domain_revision_count"] = int(
        session_state.get("domain_revision_count") or 0
    ) + 1
    append_revision_entry(
        session_state,
        plant=compile_result.get("plant") or plant_response,
        plant_findings=compile_result.get("findings") or [],
    )
    _logger.warning(
        "[HybridDES] Plant invalid (%d findings). Revision %d.",
        len(compile_result.get("findings") or []),
        int(session_state.get("domain_revision_count") or 0),
    )
    return "plant_invalid", turn_entry


async def _handle_compose_and_solve(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle compose_and_solve phase — run DES solver (no LLM call)."""
    decision, turn_entry = await _shared_handle_compose_and_solve(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        planner=planner,
    )
    next_phase = _transition_phase("compose_and_solve", decision)
    report_text, phase_result = _render_compose_report(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        planner=planner,
        decision=decision,
        next_phase=next_phase,
    )
    turn_entry["report_text"] = report_text
    turn_entry["phase_result"] = deepcopy(phase_result)
    if decision in {"unsolvable", "safety_blocked"}:
        append_revision_entry(
            session_state,
            plant=session_state.get("current_plant"),
            solver_result=session_state.get("solver_result"),
        )
    return decision, turn_entry


async def _handle_validate_plan(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle validate_plan phase — check physical feasibility of each step.

    Uses rolling symbolic state projection so that each step is validated
    against the projected state *after* all preceding steps have been applied.
    """
    initial_resources = deepcopy(dict(session_state.get("symbolic_resources") or {}))
    initial_parts = deepcopy(dict(session_state.get("symbolic_parts") or {}))
    decision, turn_entry = await _shared_handle_validate_plan(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        planner=planner,
    )
    if decision == "all_feasible":
        session_state["primitive_generation_initial_resources"] = deepcopy(initial_resources)
        session_state["primitive_generation_initial_parts"] = deepcopy(initial_parts)
        if not _has_primitive_catalogs(prepared_bridge_request):
            decision = "all_feasible_no_primitives"
    next_phase = _transition_phase("validate_plan", decision)
    report_text, phase_result = _render_validate_report(
        session_state=session_state,
        decision=decision,
        next_phase=next_phase,
        initial_resources=initial_resources,
        initial_parts=initial_parts,
    )
    turn_entry["report_text"] = report_text
    turn_entry["phase_result"] = deepcopy(phase_result)
    if decision == "infeasible":
        append_revision_entry(
            session_state,
            plant=session_state.get("current_plant"),
            solver_result=session_state.get("solver_result"),
            feasibility_findings=session_state.get("feasibility_findings"),
        )
    return decision, turn_entry


async def _handle_finalize(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle finalize phase — build the bridge proposal from validated trace."""
    return await _shared_handle_finalize(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        engine_name="hybrid_des_v1",
    )


# ---------------------------------------------------------------------------
# Phase handler dispatch
# ---------------------------------------------------------------------------

_PHASE_HANDLERS: dict[str, Any] = {
    "evaluate_grounding": _handle_evaluate_grounding,
    "domain_generation": _handle_domain_generation,
    "compose_and_solve": _handle_compose_and_solve,
    "validate_plan": _handle_validate_plan,
    "primitive_generation": _handle_primitive_generation,
    "finalize": _handle_finalize,
}


# ---------------------------------------------------------------------------
# Helper: extract safety DFAs and AP descriptors from planner context
# ---------------------------------------------------------------------------


def _extract_safety_dfas(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return _shared_extract_safety_dfas(planner, prepared_bridge_request)


def _extract_ap_descriptors(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    return _shared_extract_ap_descriptors(planner, prepared_bridge_request)


# ---------------------------------------------------------------------------
# Helper: validate a single action's physical feasibility
# ---------------------------------------------------------------------------


async def _validate_action_feasibility(
    *,
    action: dict[str, Any],
    step_index: int,
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    pre_resources: dict[str, dict[str, Any]] | None = None,
    pre_parts: dict[str, dict[str, Any]] | None = None,
    action_sequence: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return await _shared_validate_action_feasibility(
        action=action,
        step_index=step_index,
        planner=planner,
        prepared_bridge_request=prepared_bridge_request,
        session_state=session_state,
        pre_resources=pre_resources,
        pre_parts=pre_parts,
        action_sequence=action_sequence,
    )


# ---------------------------------------------------------------------------
# Component 4: Symbolic state projection for action effects
# ---------------------------------------------------------------------------


def _apply_hybrid_action_effects(
    action: dict[str, Any],
    *,
    resources: dict[str, dict[str, Any]],
    parts: dict[str, dict[str, Any]],
) -> None:
    _shared_apply_des_action_effects(
        action,
        resources=resources,
        parts=parts,
    )


def _feasibility_findings_summary(findings: list[dict[str, Any]]) -> str:
    """Render feasibility findings as text for LLM feedback."""
    return _shared_feasibility_findings_summary(findings)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_phase_prompt(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    *,
    planner: Any = None,
) -> tuple[dict[str, Any], str]:
    """Build the prompt for the current phase."""
    current_phase = str(session_state.get("current_phase") or "").strip()
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    recovery_gap_state = _build_recovery_gap_state(session_state)
    current_recovery_blockers = _collect_recovery_blockers(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        planner=planner,
    )
    ap_descriptors = _extract_ap_descriptors(planner, prepared_bridge_request)
    prompt_input = build_hybrid_des_prompt_input(
        phase=current_phase,
        llm_input=llm_input,
        session_state=session_state,
        bridge_resources=bridge_resources,
        recovery_gap_state=recovery_gap_state,
        current_recovery_blockers=current_recovery_blockers,
        ap_descriptors=ap_descriptors,
        persistent_constraint_summary=list(
            session_state.get("persistent_constraint_summary") or []
        ),
        revision_history_summary_text=revision_history_summary_text(session_state),
    )

    # Check if this is a revision (feedback prompt) or initial generation
    domain_revision_count = int(session_state.get("domain_revision_count") or 0)

    if current_phase == "domain_generation" and domain_revision_count > 0:
        # Build feedback prompt with diagnostics
        previous_plant = (
            session_state.get("last_rejected_plant")
            or session_state.get("current_plant")
        )
        previous_plant_json = ""
        if previous_plant:
            # Serialize plant for display, converting sets to lists
            serializable_plant = deepcopy(previous_plant)
            if isinstance(serializable_plant.get("states"), set):
                serializable_plant["states"] = sorted(serializable_plant["states"])
            if isinstance(serializable_plant.get("marked"), set):
                serializable_plant["marked"] = sorted(serializable_plant["marked"])
            previous_plant_json = json.dumps(
                serializable_plant, indent=2, default=str, ensure_ascii=False,
            )

        solver_result = session_state.get("solver_result") or {}
        feasibility_findings = session_state.get("feasibility_findings") or []
        solver_diag = ""
        if solver_result and not (
            solver_result.get("status") == "solved" and feasibility_findings
        ):
            solver_diag = solver_diagnostic_summary(solver_result)

        prompt_text = build_hybrid_feedback_prompt(
            prompt_input,
            plant_findings_text=plant_findings_summary(
                session_state.get("plant_findings") or [],
            ),
            solver_diagnostic_text=solver_diag,
            feasibility_findings_text=_feasibility_findings_summary(
                feasibility_findings,
            ),
            previous_plant_json=previous_plant_json,
            persistent_constraint_summary=list(
                session_state.get("persistent_constraint_summary") or []
            ),
            revision_history_summary_text=revision_history_summary_text(session_state),
        )
    else:
        prompt_text = build_hybrid_domain_generation_prompt(prompt_input)
    if current_phase == "primitive_generation":
        prompt_text = _render_primitive_generation_prompt(
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )

    return prompt_input, prompt_text


# ---------------------------------------------------------------------------
# Per-turn artifact writing
# ---------------------------------------------------------------------------


def _write_per_turn_artifact(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    turn_entry: dict[str, Any],
    *,
    write_session_transcript: bool = False,
) -> None:
    """Write debug artifacts for the current turn."""
    del turn_entry
    write_des_per_turn_artifact(
        prepared_bridge_request,
        session_state,
        phase_label=str(session_state.get("current_phase") or "hybrid"),
        debug_session_key="hybrid_session",
        write_session_transcript=write_session_transcript,
    )


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------


async def execute_hybrid_des_bridge(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run the hybrid DES bridge loop.

    LLM generates recovery plant → DES solver finds safe trace → validators
    check feasibility → finalize proposal.

    Parameters
    ----------
    planner
        The process planner (must have ``product_agent.ask_llm_structured``).
    prepared_bridge_request
        The prepared bridge request dict.
    session_state
        Optional existing session state for resumption.

    Returns
    -------
    Finalized proposal dict, or None if turn budget exhausted.
    """
    product_agent = getattr(planner, "product_agent", None)
    ask_llm_structured = getattr(product_agent, "ask_llm_structured", None)
    if not callable(ask_llm_structured):
        raise RuntimeError(
            "product_agent.ask_llm_structured is required for hybrid DES bridge execution"
        )

    if session_state is not None:
        session_state = deepcopy(session_state)
    else:
        session_state = deepcopy(
            prepared_bridge_request.get("hybrid_session_seed")
            or build_hybrid_session_seed(prepared_bridge_request)
        )
    session_state["status"] = "running"

    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["hybrid_session"] = deepcopy(session_state)
    prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
    if hasattr(planner, "_set_last_bridge_debug"):
        planner._set_last_bridge_debug(bridge_debug)

    max_turns = int(session_state.get("max_turns") or _DEFAULT_MAX_TURNS)

    while int(session_state.get("turn_index") or 0) < max_turns:
        session_state["turn_index"] = int(session_state.get("turn_index") or 0) + 1
        current_phase = str(
            session_state.get("current_phase") or "domain_generation"
        ).strip().lower()
        turn_idx = int(session_state.get("turn_index") or 0)

        _logger.info(
            "[HybridDES] Turn %d/%d | phase=%s | revisions=%d",
            turn_idx, max_turns, current_phase,
            int(session_state.get("domain_revision_count") or 0),
        )

        # Phases that require an LLM call
        if current_phase in {"domain_generation", "primitive_generation"}:
            prompt_input, prompt_text = _build_phase_prompt(
                prepared_bridge_request, session_state, planner=planner,
            )
            response_schema = hybrid_des_phase_response_schema(
                current_phase,
                vocab=(
                    dict(prompt_input.get("domain_vocabulary") or {})
                    if current_phase == "domain_generation"
                    else None
                ),
            )
            raw_response = await ask_llm_structured(
                prompt=prompt_text,
                response_format=response_schema,
            )
            parsed_response = deepcopy(
                raw_response if isinstance(raw_response, dict) else {}
            )
        else:
            # compose_and_solve, validate_plan, finalize, evaluate_grounding — no LLM call
            prompt_input = {}
            prompt_text = ""
            parsed_response = {}

        # Dispatch to handler
        handler = _PHASE_HANDLERS.get(current_phase)
        if handler is None:
            _logger.error("[HybridDES] No handler for phase=%s", current_phase)
            session_state["status"] = "error"
            break

        decision, turn_entry = await handler(
            session_state=session_state,
            parsed_response=parsed_response,
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
        )

        # Record turn
        turn_entry["turn_index"] = turn_idx
        turn_entry["phase"] = current_phase
        turn_entry["decision"] = decision
        if prompt_text:
            turn_entry["prompt_text"] = prompt_text
            turn_entry["raw_response"] = deepcopy(parsed_response)
        session_state.setdefault("turns", []).append(deepcopy(turn_entry))

        # Transition
        next_phase = _transition_phase(current_phase, decision)
        session_state["current_phase"] = next_phase

        # Terminal check
        if current_phase == "finalize" and decision == "accepted":
            session_state["status"] = "completed"
            
        if current_phase == "evaluate_grounding" and decision == "grounding_required":
            # State is already set to 'paused_after_grounding' inside the handler
            pass

        # Update debug
        bridge_debug["hybrid_session"] = deepcopy(session_state)
        bridge_debug["status"] = str(session_state.get("status") or "running")
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        if hasattr(planner, "_set_last_bridge_debug"):
            planner._set_last_bridge_debug(bridge_debug)
        is_final_turn = session_state.get("status") in (
            "completed", "paused_after_grounding",
        )
        _write_per_turn_artifact(
            prepared_bridge_request, session_state, turn_entry,
            write_session_transcript=is_final_turn,
        )

        if is_final_turn:
            break

    # Stash session state
    prepared_bridge_request["hybrid_session_state"] = deepcopy(session_state)

    if session_state.get("status") not in ("completed", "paused_after_grounding"):
        session_state["status"] = "turn_budget_exhausted"
        prepared_bridge_request["hybrid_session_state"] = deepcopy(session_state)
        bridge_debug["hybrid_session"] = deepcopy(session_state)
        bridge_debug["status"] = "turn_budget_exhausted"
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        if hasattr(planner, "_set_last_bridge_debug"):
            planner._set_last_bridge_debug(bridge_debug)
        _logger.warning(
            "[HybridDES] Turn budget exhausted (%d turns) in phase=%s",
            max_turns,
            str(session_state.get("current_phase") or ""),
        )
        _write_per_turn_artifact(
            prepared_bridge_request, session_state, {},
            write_session_transcript=True,
        )
        return None

    if session_state.get("status") == "paused_after_grounding":
        return None
        
    return deepcopy(session_state.get("proposal"))


__all__ = [
    "build_hybrid_session_seed",
    "execute_hybrid_des_bridge",
]
