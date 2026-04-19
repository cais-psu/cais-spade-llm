"""Outline-phase helpers for the multi-turn v2 bridge."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    multi_turn_v2 as _shared,
)

_logger = logging.getLogger(__name__)


def _clear_primitive_escalation_state(session_state: dict[str, Any]) -> None:
    session_state["primitive_escalation_diagnostics"] = []
    session_state["primitive_rejection_feedback"] = []
    session_state["primitive_served_context"] = {}
    session_state["primitive_context_errors"] = []
    session_state["primitive_input_diagnostics"] = []
    session_state["primitive_event_guard"] = {}


def _projected_outline_validation_context(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})

    resources_by_jid: dict[str, dict[str, Any]] = {}
    for row in (observed_runtime_state.get("resources") or []):
        if not isinstance(row, dict):
            continue
        resource_jid = str(row.get("resource_jid") or "").strip()
        if resource_jid:
            resources_by_jid[resource_jid] = deepcopy(row)
    for resource_jid, row in dict(session_state.get("symbolic_resources") or {}).items():
        token = str(resource_jid or "").strip()
        if token and isinstance(row, dict):
            resources_by_jid[token] = deepcopy(row)
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    for resource_jid, raw_entry in bridge_resources.items():
        token = str(resource_jid or "").strip()
        if not token or not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        bridge_snapshot = dict(entry.get("bridge_snapshot") or {})
        static_capabilities = dict(entry.get("static_capabilities") or {})
        resource_row = resources_by_jid.setdefault(token, {"resource_jid": token})
        for key in (
            "named_poses",
            "available_named_poses",
            "supported_recovery_states",
            "available_recovery_states",
            "reachability",
            "reachable_locations",
            "known_locations",
            "staging_areas",
            "workspace_bounds",
        ):
            if resource_row.get(key) not in (None, "", [], {}):
                continue
            if static_capabilities.get(key) not in (None, "", [], {}):
                resource_row[key] = deepcopy(static_capabilities.get(key))
            elif bridge_snapshot.get(key) not in (None, "", [], {}):
                resource_row[key] = deepcopy(bridge_snapshot.get(key))

    parts_by_name: dict[str, dict[str, Any]] = {}
    for row in (llm_input.get("part_facts") or []):
        if not isinstance(row, dict):
            continue
        part_name = str(row.get("part_name") or "").strip()
        if part_name:
            parts_by_name[part_name] = deepcopy(row)
    for part_name, row in dict(session_state.get("symbolic_parts") or {}).items():
        token = str(part_name or "").strip()
        if token and isinstance(row, dict):
            parts_by_name[token] = deepcopy(row)

    for entry in dict(session_state.get("observation_store") or {}).values():
        if not isinstance(entry, dict):
            continue
        part_name = str(entry.get("part_name") or "").strip()
        if not part_name:
            continue
        is_new_part = part_name not in parts_by_name
        part_row = parts_by_name.setdefault(part_name, {"part_name": part_name})
        pose = dict(entry.get("pose") or {})
        if not pose and entry.get("x") is not None:
            pose = {"x": entry.get("x"), "y": entry.get("y"), "z": entry.get("z")}
        if pose:
            part_row["observed_pose"] = deepcopy(pose)
        if (
            is_new_part
            and part_row.get("current_location") in (None, "")
            and entry.get("current_location") not in (None, "")
        ):
            part_row["current_location"] = deepcopy(entry.get("current_location"))
        holder = str(entry.get("current_holder_resource_jid") or "").strip()
        if (
            is_new_part
            and not str(part_row.get("current_holder_resource_jid") or "").strip()
            and holder
        ):
            part_row["current_holder_resource_jid"] = holder
    return resources_by_jid, parts_by_name


async def _handle_outline_single_pass(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Single-pass: LLM proposes all tasks at once, accept without validation."""
    turn_entry: dict[str, Any] = {}

    outline_tasks = _shared._parsed_response_rows(
        parsed_response,
        primary_key="transition_trace",
        legacy_key="outline_tasks",
    )
    turn_entry["transition_trace"] = deepcopy(outline_tasks)
    turn_entry["outline_tasks"] = deepcopy(outline_tasks)

    if not outline_tasks:
        turn_entry["error"] = (
            "outline response missing transition_trace (legacy outline_tasks)"
        )
        _logger.warning("[MultiTurnV2] outline single_pass: no transition_trace")
        return "need_revision", turn_entry

    session_state["accepted_outline_prefix"] = deepcopy(outline_tasks)
    _shared._sync_des_recovery_aliases(session_state, turn_entry=turn_entry)

    _logger.info(
        "[MultiTurnV2] outline single_pass: accepted %d recovery events",
        len(outline_tasks),
    )

    session_state["status"] = "paused_after_outline_turn"
    return "outline_ready", turn_entry


async def _handle_outline_incremental(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Incremental: one task at a time, no validation."""
    turn_entry: dict[str, Any] = {}

    next_task = _shared._parsed_response_object(
        parsed_response,
        primary_key="next_transition",
        legacy_key="next_task",
    )
    lookahead_tasks = _shared._parsed_response_rows(
        parsed_response,
        primary_key="transition_suffix",
        legacy_key="lookahead_tasks",
    )

    turn_entry["next_transition"] = deepcopy(next_task)
    turn_entry["next_task"] = deepcopy(next_task)
    turn_entry["transition_suffix"] = deepcopy(lookahead_tasks)
    if lookahead_tasks:
        turn_entry["lookahead_tasks"] = deepcopy(lookahead_tasks)

    if not next_task or not str(next_task.get("outline_id") or "").strip():
        turn_entry["error"] = (
            "outline response missing next_transition with outline_id "
            "(legacy next_task)"
        )
        _logger.warning("[MultiTurnV2] outline incremental: no next_transition")
        return "need_revision", turn_entry

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(next_task))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = deepcopy(lookahead_tasks)
    _shared._sync_des_recovery_aliases(session_state, turn_entry=turn_entry)

    _shared._apply_task_effects_to_symbolic_state(next_task, session_state)

    outline_complete = not lookahead_tasks
    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurnV2] outline incremental: accepted event %s (prefix now %d events, complete=%s)",
        str(next_task.get("outline_id") or "").strip(),
        len(accepted_prefix),
        outline_complete,
    )

    session_state["status"] = "paused_after_outline_turn"
    return decision, turn_entry


async def _handle_outline_incremental_validated(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Incremental with validation: one task at a time, validate before accepting."""
    turn_entry: dict[str, Any] = {}

    next_task = _shared._parsed_response_object(
        parsed_response,
        primary_key="next_transition",
        legacy_key="next_task",
    )
    lookahead_tasks = _shared._parsed_response_rows(
        parsed_response,
        primary_key="transition_suffix",
        legacy_key="lookahead_tasks",
    )

    turn_entry["next_transition"] = deepcopy(next_task)
    turn_entry["next_task"] = deepcopy(next_task)
    turn_entry["transition_suffix"] = deepcopy(lookahead_tasks)
    if lookahead_tasks:
        turn_entry["lookahead_tasks"] = deepcopy(lookahead_tasks)

    if not next_task or not str(next_task.get("outline_id") or "").strip():
        turn_entry["error"] = (
            "outline response missing next_transition with outline_id "
            "(legacy next_task)"
        )
        _logger.warning("[MultiTurnV2] outline incremental_validated: no next_transition")
        return "need_revision", turn_entry

    findings, grounded_action = _shared._validate_single_outline_task(
        planner=planner,
        task=next_task,
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    if grounded_action:
        turn_entry["grounded_action"] = deepcopy(grounded_action)

    if findings:
        turn_entry["validation_findings"] = deepcopy(findings)
        session_state["outline_validation_findings"] = _shared._merge_outline_validation_findings(
            list(session_state.get("outline_validation_findings") or []),
            findings,
        )
        turn_entry["transition_validation"] = {
            "status": "rejected",
            "findings": deepcopy(findings),
        }
        session_state["transition_validation"] = deepcopy(
            turn_entry["transition_validation"]
        )
        _logger.info(
            "[MultiTurnV2] outline incremental_validated: rejected event %s (%d findings)",
            str(next_task.get("outline_id") or "").strip(),
            len(findings),
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(next_task))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = deepcopy(lookahead_tasks)

    _shared._apply_task_effects_to_symbolic_state(next_task, session_state)
    session_state["outline_validation_findings"] = _shared._prune_resolved_outline_validation_findings(
        list(session_state.get("outline_validation_findings") or []),
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    _shared._sync_des_recovery_aliases(
        session_state,
        turn_entry=turn_entry,
        transition_validation={"status": "passed", "findings": []},
    )

    outline_complete = not lookahead_tasks
    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurnV2] outline incremental_validated: accepted event %s "
        "(prefix now %d events, complete=%s)",
        str(next_task.get("outline_id") or "").strip(),
        len(accepted_prefix),
        outline_complete,
    )

    session_state["status"] = "paused_after_outline_turn"
    return decision, turn_entry


async def _handle_outline_incremental_candidates_validated(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Incremental with multiple candidate next tasks and deterministic selection."""
    turn_entry: dict[str, Any] = {}
    sequence_index = _shared._next_recovery_sequence_index(session_state)
    session_state["outline_validation_findings"] = []
    session_state["pruned_actions"] = _shared._active_v2_pruned_actions(
        session_state,
        prepared_bridge_request,
    )

    candidate_response_rows = _shared._parsed_response_rows_any(
        parsed_response,
        keys=("candidate_events", "candidate_transitions", "candidate_tasks"),
    )
    candidate_tasks = [
        _shared._normalize_candidate_task(
            task=dict(row),
            sequence_index=sequence_index,
            candidate_index=candidate_index,
        )
        for candidate_index, row in enumerate(candidate_response_rows)
    ]
    turn_entry["candidate_events"] = deepcopy(candidate_tasks)
    turn_entry["candidate_transitions"] = deepcopy(candidate_tasks)
    turn_entry["candidate_tasks"] = deepcopy(candidate_tasks)

    candidate_bound = int(
        session_state.get("des_candidate_bound")
        or session_state.get("candidate_bound")
        or _shared._DEFAULT_CANDIDATE_BOUND
    )
    if not (1 <= len(candidate_tasks) <= candidate_bound):
        turn_entry["error"] = (
            "outline response must include 1 to "
            f"{candidate_bound} candidate_events "
            "(compatibility candidate_transitions/candidate_tasks)"
        )
        _logger.warning(
            "[MultiTurnV2] outline incremental_candidates_validated: expected 1-%d candidate_events, got %d",
            candidate_bound,
            len(candidate_tasks),
        )
        return "need_revision", turn_entry

    candidate_evaluations: list[dict[str, Any]] = []
    valid_candidates: list[dict[str, Any]] = []
    for candidate_index, task in enumerate(candidate_tasks):
        surface_task = deepcopy(task)
        working_task = deepcopy(task)
        evaluation: dict[str, Any] = {
            "candidate_index": candidate_index,
            "surface_task": deepcopy(surface_task),
            "task": deepcopy(working_task),
        }

        pruned_row = _shared._matching_active_v2_pruned_action(
            task=working_task,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if pruned_row is not None:
            evaluation["valid"] = False
            evaluation["validation_findings"] = [
                _shared._retarget_candidate_finding_to_task(
                    dict(pruned_row.get("guard") or {}),
                    working_task,
                )
            ]
            evaluation["pruned_match"] = True
            candidate_evaluations.append(evaluation)
            continue

        normalized_task, schema_findings = _shared._derive_candidate_outline_task(
            candidate_task=working_task,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if schema_findings:
            evaluation["valid"] = False
            evaluation["validation_findings"] = deepcopy(schema_findings)
            candidate_evaluations.append(evaluation)
            continue
        evaluation["normalized_task"] = deepcopy(normalized_task)

        findings, grounded_action = _shared._validate_single_outline_task(
            planner=planner,
            task=dict(normalized_task or {}),
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        evaluation["valid"] = not findings
        evaluation["validation_findings"] = deepcopy(findings)
        if grounded_action:
            evaluation["grounded_action"] = deepcopy(grounded_action)
        if findings:
            candidate_evaluations.append(evaluation)
            continue

        progress_score, progress_detail = _shared._candidate_progress_score(
            task=dict(normalized_task or {}),
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        evaluation["progress_score"] = progress_score
        evaluation["progress_detail"] = deepcopy(progress_detail)
        valid_candidates.append(evaluation)
        candidate_evaluations.append(evaluation)

    progress_candidates = [
        row for row in valid_candidates
        if int(row.get("progress_score") or 0) > 0
    ]

    if not progress_candidates:
        for row in candidate_evaluations:
            if not bool(row.get("valid")):
                continue
            row["valid"] = False
            row["validation_findings"] = [
                _shared._no_blocker_reduction_finding(task=dict(row.get("task") or {}))
            ]
        _shared._promote_durable_candidate_rejections(
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
            candidate_evaluations=candidate_evaluations,
        )
        turn_entry["candidate_evaluations"] = deepcopy(candidate_evaluations)
        feedback_rows = _shared._candidate_feedback_rows(candidate_evaluations)
        session_state["candidate_rejection_feedback"] = deepcopy(feedback_rows)
        turn_entry["candidate_rejection_feedback"] = deepcopy(feedback_rows)
        turn_entry["transition_validation"] = {
            "status": "rejected",
            "findings": deepcopy(feedback_rows),
        }
        session_state["transition_validation"] = deepcopy(
            turn_entry["transition_validation"]
        )
        session_state["rejected_turn_thought"] = str(
            parsed_response.get("thought") or ""
        ).strip()
        _logger.info(
            "[MultiTurnV2] outline incremental_candidates_validated: rejected all %d candidates",
            len(candidate_tasks),
        )
        stagnation = int(session_state.get("outline_stagnation_count") or 0) + 1
        session_state["outline_stagnation_count"] = stagnation
        status_counts: dict[str, int] = {}
        for row in candidate_evaluations:
            if not isinstance(row, dict):
                continue
            findings = [
                dict(f)
                for f in (row.get("validation_findings") or [])
                if isinstance(f, dict)
            ]
            if not findings:
                continue
            status = _shared._finding_event_status_for_logging(findings[0])
            status_counts[status] = int(status_counts.get(status) or 0) + 1
        status_summary = ", ".join(
            f"{status}={count}"
            for status, count in sorted(status_counts.items())
        ) or "none"
        rejection_codes = [
            str(f.get("constraint_code") or "unknown")
            for row in candidate_evaluations
            if isinstance(row, dict)
            for f in (row.get("validation_findings") or [])
            if isinstance(f, dict)
        ]
        _logger.info(
            "[DES] Stagnation %d — status_counts: %s",
            stagnation, status_summary,
        )
        _logger.debug(
            "[DES] Stagnation %d — rejection codes: %s",
            stagnation, rejection_codes,
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry
    turn_entry["candidate_evaluations"] = deepcopy(candidate_evaluations)

    session_state["outline_stagnation_count"] = 0

    selected = max(
        progress_candidates,
        key=lambda row: (
            int(dict(row.get("progress_detail") or {}).get("resolved_direct_blockers") or 0),
            int(dict(row.get("progress_detail") or {}).get("blocker_part_acquired") or 0)
            + int(dict(row.get("progress_detail") or {}).get("freed_resource_for_blocker") or 0),
            int(dict(row.get("progress_detail") or {}).get("preparatory_transit") or 0),
            -int(dict(row.get("progress_detail") or {}).get("remaining_blocked_issues") or 0),
            -int(row.get("candidate_index") or 0),
        ),
    )
    selected_candidate_index = int(selected.get("candidate_index") or 0)
    selected_candidate_task = deepcopy(dict(selected.get("task") or {}))
    selected_normalized_task = deepcopy(dict(selected.get("normalized_task") or {}))
    selected_next_task = _shared._commit_selected_candidate_task(
        task=selected_normalized_task,
        sequence_index=sequence_index,
    )
    selected_grounded_action = deepcopy(dict(selected.get("grounded_action") or {}))

    turn_entry["selected_candidate_index"] = selected_candidate_index
    turn_entry["selected_transition"] = deepcopy(selected_next_task)
    turn_entry["selected_candidate_task"] = deepcopy(selected_candidate_task)
    turn_entry["selected_next_task"] = deepcopy(selected_next_task)
    turn_entry["next_transition"] = deepcopy(selected_next_task)
    turn_entry["next_task"] = deepcopy(selected_next_task)
    if selected_grounded_action:
        turn_entry["grounded_action"] = deepcopy(selected_grounded_action)

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(selected_next_task))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = []
    session_state["candidate_rejection_feedback"] = []
    session_state["rejected_turn_thought"] = ""
    _shared._sync_des_recovery_aliases(
        session_state,
        turn_entry=turn_entry,
        transition_validation={
            "status": "passed",
            "selected_candidate_index": selected_candidate_index,
        },
    )

    pre_state = _shared._symbolic_state_fingerprint(
        session_state.get("symbolic_resources") or {},
        session_state.get("symbolic_parts") or {},
    )
    _shared._apply_task_effects_to_symbolic_state(selected_next_task, session_state)
    session_state["pruned_actions"] = _shared._active_v2_pruned_actions(
        session_state,
        prepared_bridge_request,
    )
    post_state = _shared._symbolic_state_fingerprint(
        session_state.get("symbolic_resources") or {},
        session_state.get("symbolic_parts") or {},
    )

    event_name = str(
        selected_next_task.get("outline_id") or f"e_{len(session_state.get('des_trace') or [])}"
    ).strip()
    _shared._extend_plant_with_event(
        session_state.get("des_plant") or {},
        event_name=event_name,
        event_dict={
            "name": str(selected_next_task.get("event_name") or "").strip(),
            "resource_jid": str(selected_next_task.get("resource_jid") or "").strip(),
            "part_name": str(selected_next_task.get("part_name") or "").strip() or None,
            "target_ref": str(selected_next_task.get("target_ref") or "").strip() or None,
            "description": str(selected_next_task.get("description") or "").strip(),
        },
        from_state=pre_state,
        to_state=post_state,
    )
    session_state["des_current_state"] = post_state
    session_state.setdefault("des_trace", []).append(event_name)

    safety_dfas = session_state.get("des_safety_dfas") or {}
    if safety_dfas:
        _violates, new_q, _violated_ids = _shared._advance_des_safety_state(
            candidate_event={
                "name": str(selected_next_task.get("event_name") or "").strip(),
                "resource_jid": str(selected_next_task.get("resource_jid") or "").strip(),
                "part_name": str(selected_next_task.get("part_name") or "").strip() or None,
                "target_ref": str(selected_next_task.get("target_ref") or "").strip() or None,
            },
            current_safety_q=tuple(session_state.get("des_safety_dfa_vector") or ()),
            safety_dfas=safety_dfas,
            ap_descriptors=session_state.get("des_ap_descriptors") or [],
        )
        session_state["des_safety_dfa_vector"] = new_q

    visited = session_state.setdefault("des_visited_states", [])
    if post_state in visited:
        _logger.warning("[DES] Cycle detected: state %s already visited", post_state)
        session_state["status"] = "des_cycle_detected"
    else:
        visited.append(post_state)

    _logger.info(
        "[DES] Plant extended: %s -[%s]-> %s (trace len=%d)",
        pre_state, event_name, post_state, len(session_state.get("des_trace") or []),
    )

    remaining_findings, remaining_conditions = _shared._remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    outline_complete = remaining_findings == 0 and remaining_conditions == 0

    if outline_complete and (session_state.get("des_safety_dfas") or {}):
        plant = session_state.get("des_plant") or {}
        final_state = session_state.get("des_current_state") or ""
        if final_state and final_state not in (plant.get("marked") or []):
            plant.setdefault("marked", []).append(final_state)
        solver_result = _shared.compose_and_solve(
            plant=plant,
            safety_dfas=session_state.get("des_safety_dfas") or {},
            ap_descriptors=session_state.get("des_ap_descriptors") or [],
        )
        solver_status = str(solver_result.get("status") or "").strip()
        if solver_status != "solved":
            diagnostic = _shared.solver_diagnostic_summary(solver_result)
            _logger.warning(
                "[DES] Composition gate FAILED (%s): %s", solver_status, diagnostic,
            )
            session_state.setdefault("phase_feedback", []).append({
                "phase": "outline",
                "issue": "des_safety_composition_failed",
                "diagnostic": diagnostic,
            })
            outline_complete = False
        else:
            _logger.info("[DES] Composition gate PASSED — outline is safety-verified.")

    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurnV2] outline incremental_candidates_validated: selected candidate %d (%s) "
        "(progress=%d, prefix now %d events, complete=%s)",
        selected_candidate_index + 1,
        str(selected_next_task.get("outline_id") or "").strip(),
        int(selected.get("progress_score") or 0),
        len(accepted_prefix),
        outline_complete,
    )

    session_state["status"] = "paused_after_outline_turn"
    return decision, turn_entry


async def _handle_outline_phase(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Dispatch outline handling according to configured outline mode."""
    outline_mode = str(
        session_state.get("outline_mode") or "incremental"
    ).strip().lower()
    decision: str
    turn_entry: dict[str, Any]
    if outline_mode == "single_pass":
        decision, turn_entry = await _handle_outline_single_pass(
            session_state=session_state,
            parsed_response=parsed_response,
        )
    elif outline_mode == "incremental_validated":
        decision, turn_entry = await _handle_outline_incremental_validated(
            session_state=session_state,
            parsed_response=parsed_response,
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
        )
    elif outline_mode == "incremental_candidates_validated":
        decision, turn_entry = await _handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response=parsed_response,
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
        )
    else:
        decision, turn_entry = await _handle_outline_incremental(
            session_state=session_state,
            parsed_response=parsed_response,
        )

    if decision in {"outline_ready", "need_next_task"}:
        _clear_primitive_escalation_state(session_state)
    return decision, turn_entry


__all__ = [
    "_projected_outline_validation_context",
    "_handle_outline_single_pass",
    "_handle_outline_incremental",
    "_handle_outline_incremental_validated",
    "_handle_outline_incremental_candidates_validated",
    "_handle_outline_phase",
]
