"""Outline runtime handlers for the DES recovery bridge."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_automaton import (
    compose_and_solve,
    solver_diagnostic_summary,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_outline_helpers import (
    _DEFAULT_CANDIDATE_BOUND,
    active_des_recovery_pruned_actions,
    advance_des_safety_state,
    apply_task_effects_to_symbolic_state,
    candidate_feedback_rows,
    candidate_progress_score,
    commit_selected_candidate_task,
    derive_candidate_outline_task,
    extend_plant_with_event,
    finding_event_status_for_logging,
    matching_active_des_recovery_pruned_action,
    merge_outline_validation_findings,
    next_recovery_sequence_index,
    no_blocker_reduction_finding,
    normalize_candidate_task,
    parsed_response_object,
    parsed_response_rows,
    parsed_response_rows_any,
    promote_durable_candidate_rejections,
    prune_resolved_outline_validation_findings,
    remaining_blocked_issue_counts,
    retarget_candidate_finding_to_task,
    symbolic_state_fingerprint,
    sync_des_recovery_aliases,
    validate_single_outline_task,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.des_recovery_validation_context import (
    projected_outline_validation_context,
)

_logger = logging.getLogger(__name__)


def _clear_primitive_escalation_state(session_state: dict[str, Any]) -> None:
    session_state["primitive_escalation_diagnostics"] = []
    session_state["primitive_rejection_feedback"] = []
    session_state["primitive_served_context"] = {}
    session_state["primitive_context_errors"] = []
    session_state["primitive_input_diagnostics"] = []
    session_state["primitive_event_guard"] = {}


async def _handle_outline_single_pass(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Single-pass: LLM proposes all tasks at once, accept without validation."""
    turn_entry: dict[str, Any] = {}

    outline_tasks = parsed_response_rows(
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
        _logger.warning("[DesRecovery] outline single_pass: no transition_trace")
        return "need_revision", turn_entry

    session_state["accepted_outline_prefix"] = deepcopy(outline_tasks)
    sync_des_recovery_aliases(session_state, turn_entry=turn_entry)

    _logger.info(
        "[DesRecovery] outline single_pass: accepted %d recovery events",
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

    next_task = parsed_response_object(
        parsed_response,
        primary_key="next_transition",
        legacy_key="next_task",
    )
    lookahead_tasks = parsed_response_rows(
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
        _logger.warning("[DesRecovery] outline incremental: no next_transition")
        return "need_revision", turn_entry

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(next_task))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = deepcopy(lookahead_tasks)
    sync_des_recovery_aliases(session_state, turn_entry=turn_entry)

    apply_task_effects_to_symbolic_state(next_task, session_state)

    outline_complete = not lookahead_tasks
    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[DesRecovery] outline incremental: accepted event %s (prefix now %d events, complete=%s)",
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

    next_task = parsed_response_object(
        parsed_response,
        primary_key="next_transition",
        legacy_key="next_task",
    )
    lookahead_tasks = parsed_response_rows(
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
        _logger.warning("[DesRecovery] outline incremental_validated: no next_transition")
        return "need_revision", turn_entry

    findings, grounded_action = validate_single_outline_task(
        planner=planner,
        task=next_task,
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    if grounded_action:
        turn_entry["grounded_action"] = deepcopy(grounded_action)

    if findings:
        turn_entry["validation_findings"] = deepcopy(findings)
        session_state["outline_validation_findings"] = merge_outline_validation_findings(
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
            "[DesRecovery] outline incremental_validated: rejected event %s (%d findings)",
            str(next_task.get("outline_id") or "").strip(),
            len(findings),
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(next_task))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = deepcopy(lookahead_tasks)

    apply_task_effects_to_symbolic_state(next_task, session_state)
    session_state["outline_validation_findings"] = prune_resolved_outline_validation_findings(
        list(session_state.get("outline_validation_findings") or []),
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    sync_des_recovery_aliases(
        session_state,
        turn_entry=turn_entry,
        transition_validation={"status": "passed", "findings": []},
    )

    outline_complete = not lookahead_tasks
    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[DesRecovery] outline incremental_validated: accepted event %s "
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
    sequence_index = next_recovery_sequence_index(session_state)
    session_state["outline_validation_findings"] = []
    session_state["pruned_actions"] = active_des_recovery_pruned_actions(
        session_state,
        prepared_bridge_request,
    )

    candidate_response_rows = parsed_response_rows_any(
        parsed_response,
        keys=("candidate_events", "candidate_transitions", "candidate_tasks"),
    )
    candidate_tasks = [
        normalize_candidate_task(
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
        or _DEFAULT_CANDIDATE_BOUND
    )
    if not (1 <= len(candidate_tasks) <= candidate_bound):
        turn_entry["error"] = (
            "outline response must include 1 to "
            f"{candidate_bound} candidate_events "
            "(compatibility candidate_transitions/candidate_tasks)"
        )
        _logger.warning(
            "[DesRecovery] outline incremental_candidates_validated: expected 1-%d candidate_events, got %d",
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

        pruned_row = matching_active_des_recovery_pruned_action(
            task=working_task,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if pruned_row is not None:
            evaluation["valid"] = False
            evaluation["validation_findings"] = [
                retarget_candidate_finding_to_task(
                    dict(pruned_row.get("guard") or {}),
                    working_task,
                )
            ]
            evaluation["pruned_match"] = True
            candidate_evaluations.append(evaluation)
            continue

        normalized_task, schema_findings = derive_candidate_outline_task(
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

        findings, grounded_action = validate_single_outline_task(
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

        progress_score, progress_detail = candidate_progress_score(
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
                no_blocker_reduction_finding(task=dict(row.get("task") or {}))
            ]
        promote_durable_candidate_rejections(
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
            candidate_evaluations=candidate_evaluations,
        )
        turn_entry["candidate_evaluations"] = deepcopy(candidate_evaluations)
        feedback_rows = candidate_feedback_rows(candidate_evaluations)
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
            "[DesRecovery] outline incremental_candidates_validated: rejected all %d candidates",
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
            status = finding_event_status_for_logging(findings[0])
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
    selected_next_task = commit_selected_candidate_task(
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
    sync_des_recovery_aliases(
        session_state,
        turn_entry=turn_entry,
        transition_validation={
            "status": "passed",
            "selected_candidate_index": selected_candidate_index,
        },
    )

    pre_state = symbolic_state_fingerprint(
        session_state.get("symbolic_resources") or {},
        session_state.get("symbolic_parts") or {},
    )
    apply_task_effects_to_symbolic_state(selected_next_task, session_state)
    session_state["pruned_actions"] = active_des_recovery_pruned_actions(
        session_state,
        prepared_bridge_request,
    )
    post_state = symbolic_state_fingerprint(
        session_state.get("symbolic_resources") or {},
        session_state.get("symbolic_parts") or {},
    )

    event_name = str(
        selected_next_task.get("outline_id") or f"e_{len(session_state.get('des_trace') or [])}"
    ).strip()
    extend_plant_with_event(
        session_state.get("des_plant") or {},
        event_name=event_name,
        event_dict={
            "name": str(
                selected_next_task.get("event_name")
                or selected_next_task.get("action_name")
                or ""
            ).strip(),
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
        _violates, new_q, _violated_ids = advance_des_safety_state(
            candidate_event={
                "name": str(
                    selected_next_task.get("event_name")
                    or selected_next_task.get("action_name")
                    or ""
                ).strip(),
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

    remaining_findings, remaining_conditions = remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    outline_complete = remaining_findings == 0 and remaining_conditions == 0

    if outline_complete and (session_state.get("des_safety_dfas") or {}):
        plant = session_state.get("des_plant") or {}
        final_state = session_state.get("des_current_state") or ""
        if final_state and final_state not in (plant.get("marked") or []):
            plant.setdefault("marked", []).append(final_state)
        solver_result = compose_and_solve(
            plant=plant,
            safety_dfas=session_state.get("des_safety_dfas") or {},
            ap_descriptors=session_state.get("des_ap_descriptors") or [],
        )
        solver_status = str(solver_result.get("status") or "").strip()
        if solver_status != "solved":
            diagnostic = solver_diagnostic_summary(solver_result)
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
        "[DesRecovery] outline incremental_candidates_validated: selected candidate %d (%s) "
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
    "projected_outline_validation_context",
    "_handle_outline_single_pass",
    "_handle_outline_incremental",
    "_handle_outline_incremental_validated",
    "_handle_outline_incremental_candidates_validated",
    "_handle_outline_phase",
]
