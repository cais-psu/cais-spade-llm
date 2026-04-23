"""Outline-phase helpers for the multi-turn bridge."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_validation_service import (
    projected_outline_validation_context as _service_projected_outline_validation_context,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    multi_turn as _shared,
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
    return _service_projected_outline_validation_context(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )


async def _handle_outline_single_pass(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
    parsed_response: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Single-pass: LLM proposes all tasks at once, Product derives the trace."""
    turn_entry: dict[str, Any] = {}

    surface_trace = _shared._parsed_response_rows(
        parsed_response,
        primary_key="transition_trace",
    )
    turn_entry["transition_trace_surface"] = deepcopy(surface_trace)

    if not surface_trace:
        turn_entry["error"] = "outline response missing transition_trace"
        _logger.warning("[MultiTurn] outline single_pass: no transition_trace")
        return "need_revision", turn_entry

    derived_trace: list[dict[str, Any]] = []
    working_session_state = deepcopy(session_state)
    base_sequence_index = _shared._next_recovery_sequence_index(session_state)
    for index, surface_event in enumerate(surface_trace, start=base_sequence_index):
        working_surface = deepcopy(dict(surface_event or {}))
        validated_task, schema_findings = _shared._derive_candidate_outline_task(
            candidate_task=working_surface,
            session_state=working_session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if schema_findings or not validated_task:
            turn_entry["validation_findings"] = deepcopy(schema_findings)
            session_state["outline_validation_findings"] = deepcopy(schema_findings)
            session_state["status"] = "paused_after_outline_turn"
            return "need_revision", turn_entry
        findings, grounded_action = _shared._validate_single_outline_task(
            planner=planner,
            task=dict(validated_task or {}),
            session_state=working_session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if findings or not validated_task:
            turn_entry["validation_findings"] = deepcopy(findings)
            session_state["outline_validation_findings"] = deepcopy(findings)
            session_state["status"] = "paused_after_outline_turn"
            return "need_revision", turn_entry
        if grounded_action:
            turn_entry.setdefault("grounded_trace", []).append(deepcopy(grounded_action))
        committed_task = _shared._commit_selected_candidate_task(
            task=working_surface,
            sequence_index=index,
        )
        derived_trace.append(deepcopy(committed_task))
        _shared._apply_task_effects_to_symbolic_state(
            committed_task,
            working_session_state,
        )

    session_state["accepted_outline_prefix"] = deepcopy(derived_trace)
    turn_entry["transition_trace"] = deepcopy(derived_trace)
    _shared._sync_des_recovery_aliases(session_state, turn_entry=turn_entry)

    _logger.info(
        "[MultiTurn] outline single_pass: accepted %d recovery events",
        len(derived_trace),
    )

    session_state["status"] = "ready_for_primitive_generation"
    return "outline_ready", turn_entry


async def _handle_outline_incremental(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
    parsed_response: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Incremental: one task at a time, Product derives and validates the event."""
    turn_entry: dict[str, Any] = {}

    next_surface_transition = _shared._parsed_response_object(
        parsed_response,
        primary_key="next_transition",
    )
    transition_suffix_surface = _shared._parsed_response_rows(
        parsed_response,
        primary_key="transition_suffix",
    )

    turn_entry["next_transition_surface"] = deepcopy(next_surface_transition)
    turn_entry["transition_suffix_surface"] = deepcopy(transition_suffix_surface)

    if not next_surface_transition:
        turn_entry["error"] = "outline response missing next_transition"
        _logger.warning("[MultiTurn] outline incremental: no next_transition")
        return "need_revision", turn_entry

    sequence_index = _shared._next_recovery_sequence_index(session_state)
    surface_transition = deepcopy(dict(next_surface_transition or {}))
    validated_task, schema_findings = _shared._derive_candidate_outline_task(
        candidate_task=surface_transition,
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    if schema_findings or not validated_task:
        turn_entry["validation_findings"] = deepcopy(schema_findings)
        session_state["outline_validation_findings"] = deepcopy(schema_findings)
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry
    findings, grounded_action = _shared._validate_single_outline_task(
        planner=planner,
        task=dict(validated_task or {}),
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    next_transition = deepcopy(surface_transition)
    if findings or not next_transition:
        turn_entry["validation_findings"] = deepcopy(findings)
        session_state["outline_validation_findings"] = deepcopy(findings)
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry
    if grounded_action:
        turn_entry["grounded_action"] = deepcopy(grounded_action)
    next_transition = _shared._commit_selected_candidate_task(
        task=surface_transition,
        sequence_index=sequence_index,
    )

    transition_suffix: list[dict[str, Any]] = []
    working_session_state = deepcopy(session_state)
    _shared._apply_task_effects_to_symbolic_state(next_transition, working_session_state)
    for index, raw_suffix in enumerate(transition_suffix_surface, start=1):
        surface_suffix = deepcopy(dict(raw_suffix or {}))
        validated_suffix, suffix_schema_findings = _shared._derive_candidate_outline_task(
            candidate_task=surface_suffix,
            session_state=working_session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if suffix_schema_findings or not validated_suffix:
            break
        suffix_findings, _suffix_grounded_action = _shared._validate_single_outline_task(
            planner=planner,
            task=dict(validated_suffix or {}),
            session_state=working_session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if suffix_findings or not validated_suffix:
            break
        committed_suffix = _shared._commit_selected_candidate_task(
            task=surface_suffix,
            sequence_index=sequence_index + index,
        )
        transition_suffix.append(deepcopy(committed_suffix))
        _shared._apply_task_effects_to_symbolic_state(
            committed_suffix,
            working_session_state,
        )

    turn_entry["next_transition"] = deepcopy(next_transition)
    turn_entry["transition_suffix"] = deepcopy(transition_suffix)

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(next_transition))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = deepcopy(transition_suffix)
    _shared._sync_des_recovery_aliases(session_state, turn_entry=turn_entry)

    _shared._apply_task_effects_to_symbolic_state(next_transition, session_state)

    outline_complete = not transition_suffix
    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurn] outline incremental: accepted event %s (prefix now %d events, complete=%s)",
        str(next_transition.get("outline_id") or "").strip(),
        len(accepted_prefix),
        outline_complete,
    )

    session_state["status"] = (
        "ready_for_primitive_generation"
        if outline_complete
        else "paused_after_outline_turn"
    )
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

    next_transition = _shared._parsed_response_object(
        parsed_response,
        primary_key="next_transition",
    )
    transition_suffix = _shared._parsed_response_rows(
        parsed_response,
        primary_key="transition_suffix",
    )

    turn_entry["next_transition"] = deepcopy(next_transition)
    turn_entry["transition_suffix"] = deepcopy(transition_suffix)

    if not next_transition:
        turn_entry["error"] = "outline response missing next_transition"
        _logger.warning("[MultiTurn] outline incremental_validated: no next_transition")
        return "need_revision", turn_entry

    sequence_index = _shared._next_recovery_sequence_index(session_state)
    surface_transition = deepcopy(dict(next_transition or {}))
    validated_task, schema_findings = _shared._derive_candidate_outline_task(
        candidate_task=surface_transition,
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    if schema_findings or not validated_task:
        turn_entry["next_transition"] = deepcopy(surface_transition)
        turn_entry["validation_findings"] = deepcopy(schema_findings)
        session_state["outline_validation_findings"] = _shared._merge_outline_validation_findings(
            list(session_state.get("outline_validation_findings") or []),
            schema_findings,
        )
        turn_entry["transition_validation"] = {
            "status": "rejected",
            "findings": deepcopy(schema_findings),
        }
        session_state["transition_validation"] = deepcopy(
            turn_entry["transition_validation"]
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry
    findings, grounded_action = _shared._validate_single_outline_task(
        planner=planner,
        task=dict(validated_task or {}),
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    next_transition = _shared._commit_selected_candidate_task(
        task=surface_transition,
        sequence_index=sequence_index,
    )
    turn_entry["next_transition"] = deepcopy(next_transition)
    if grounded_action:
        turn_entry["grounded_action"] = deepcopy(grounded_action)

    if findings or not next_transition:
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
            "[MultiTurn] outline incremental_validated: rejected event %s (%d findings)",
            str(next_transition.get("outline_id") or "").strip(),
            len(findings),
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(next_transition))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = deepcopy(transition_suffix)

    _shared._apply_task_effects_to_symbolic_state(next_transition, session_state)
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

    outline_complete = not transition_suffix
    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurn] outline incremental_validated: accepted event %s "
        "(prefix now %d events, complete=%s)",
        str(next_transition.get("outline_id") or "").strip(),
        len(accepted_prefix),
        outline_complete,
    )

    session_state["status"] = (
        "ready_for_primitive_generation"
        if outline_complete
        else "paused_after_outline_turn"
    )
    return decision, turn_entry


async def _handle_outline_incremental_candidates_validated(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Incremental with multiple candidate next tasks and LLM-owned selection."""
    turn_entry: dict[str, Any] = {}
    sequence_index = _shared._next_recovery_sequence_index(session_state)
    session_state["outline_validation_findings"] = []
    session_state["pruned_actions"] = _shared._active_pruned_actions(
        session_state,
        prepared_bridge_request,
    )

    candidate_events = [
        dict(row)
        for row in _shared._parsed_response_rows(
            parsed_response,
            primary_key="candidate_events",
        )
    ]
    turn_entry["candidate_events"] = deepcopy(candidate_events)

    candidate_bound = int(
        session_state.get("candidate_bound")
        or _shared._DEFAULT_CANDIDATE_BOUND
    )
    if not (1 <= len(candidate_events) <= candidate_bound):
        turn_entry["error"] = (
            "outline response must include 1 to "
            f"{candidate_bound} candidate_events"
        )
        _logger.warning(
            "[MultiTurn] outline incremental_candidates_validated: expected 1-%d candidate_events, got %d",
            candidate_bound,
            len(candidate_events),
        )
        return "need_revision", turn_entry

    selected_candidate_index_raw = parsed_response.get("selected_candidate_index")
    if (
        not isinstance(selected_candidate_index_raw, int)
        or isinstance(selected_candidate_index_raw, bool)
    ):
        finding = _shared._candidate_schema_finding(
            task={},
            reason=(
                "outline response must include integer selected_candidate_index "
                "pointing at candidate_events"
            ),
            evidence={"field": "selected_candidate_index"},
        )
        turn_entry["validation_findings"] = [deepcopy(finding)]
        turn_entry["transition_validation"] = {
            "status": "rejected",
            "findings": [deepcopy(finding)],
        }
        session_state["outline_validation_findings"] = [deepcopy(finding)]
        session_state["transition_validation"] = deepcopy(
            turn_entry["transition_validation"]
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry
    selected_candidate_index = int(selected_candidate_index_raw)
    turn_entry["selected_candidate_index"] = selected_candidate_index
    if not (0 <= selected_candidate_index < len(candidate_events)):
        finding = _shared._candidate_schema_finding(
            task={},
            reason=(
                "selected_candidate_index must reference an item in candidate_events"
            ),
            evidence={
                "field": "selected_candidate_index",
                "selected_candidate_index": selected_candidate_index,
                "candidate_count": len(candidate_events),
            },
        )
        turn_entry["validation_findings"] = [deepcopy(finding)]
        turn_entry["transition_validation"] = {
            "status": "rejected",
            "findings": [deepcopy(finding)],
        }
        session_state["outline_validation_findings"] = [deepcopy(finding)]
        session_state["transition_validation"] = deepcopy(
            turn_entry["transition_validation"]
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry

    candidate_evaluations: list[dict[str, Any]] = []
    for candidate_index, task in enumerate(candidate_events):
        surface_task = deepcopy(task)
        working_task = deepcopy(task)
        evaluation: dict[str, Any] = {
            "candidate_index": candidate_index,
            "surface_task": deepcopy(surface_task),
            "task": deepcopy(working_task),
        }

        validated_task, schema_findings = _shared._derive_candidate_outline_task(
            candidate_task=working_task,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if schema_findings or not validated_task:
            evaluation["valid"] = False
            evaluation["validation_findings"] = deepcopy(schema_findings)
            candidate_evaluations.append(evaluation)
            continue
        evaluation["validated_task"] = deepcopy(validated_task)

        pruned_row = _shared._matching_active_pruned_action(
            task=validated_task,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if pruned_row is not None:
            evaluation["valid"] = False
            evaluation["validation_findings"] = [
                _shared._retarget_candidate_finding_to_task(
                    dict(pruned_row.get("guard") or {}),
                    dict(validated_task or working_task),
                )
            ]
            evaluation["pruned_match"] = True
            candidate_evaluations.append(evaluation)
            continue

        findings, grounded_action = _shared._validate_single_outline_task(
            planner=planner,
            task=dict(validated_task or {}),
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        evaluation["valid"] = bool(validated_task) and not findings
        evaluation["validation_findings"] = deepcopy(findings)
        if grounded_action:
            evaluation["grounded_action"] = deepcopy(grounded_action)
        candidate_evaluations.append(evaluation)

    turn_entry["candidate_evaluations"] = deepcopy(candidate_evaluations)
    _shared._promote_durable_candidate_rejections(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        candidate_evaluations=candidate_evaluations,
    )

    selected = next(
        (
            row
            for row in candidate_evaluations
            if isinstance(row, dict)
            and int(row.get("candidate_index", -1)) == selected_candidate_index
        ),
        None,
    )
    feedback_rows = _shared._candidate_feedback_rows(candidate_evaluations)
    selected_valid = bool(selected and selected.get("valid"))
    if not selected_valid:
        accumulated_feedback = _shared._merge_candidate_rejection_feedback(
            list(session_state.get("candidate_rejection_feedback") or []),
            feedback_rows,
        )
        session_state["candidate_rejection_feedback"] = deepcopy(accumulated_feedback)
        turn_entry["candidate_rejection_feedback"] = deepcopy(accumulated_feedback)
        turn_entry["transition_validation"] = {
            "status": "rejected",
            "selected_candidate_index": selected_candidate_index,
            "findings": deepcopy(feedback_rows),
        }
        session_state["transition_validation"] = deepcopy(
            turn_entry["transition_validation"]
        )
        session_state["rejected_turn_thought"] = str(
            parsed_response.get("thought") or ""
        ).strip()
        _logger.info(
            "[MultiTurn] outline incremental_candidates_validated: rejected selected candidate %d",
            selected_candidate_index + 1,
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
            "[MultiTurn] Stagnation %d — status_counts: %s",
            stagnation, status_summary,
        )
        _logger.debug(
            "[MultiTurn] Stagnation %d — rejection codes: %s",
            stagnation, rejection_codes,
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry

    session_state["outline_stagnation_count"] = 0

    selected_candidate_task = deepcopy(dict(selected.get("task") or {}))
    selected_validated_task = deepcopy(dict(selected.get("validated_task") or {}))
    selected_transition = _shared._commit_selected_candidate_task(
        task=selected_validated_task,
        sequence_index=sequence_index,
    )
    selected_grounded_action = deepcopy(dict(selected.get("grounded_action") or {}))

    turn_entry["selected_candidate_index"] = selected_candidate_index
    turn_entry["selected_transition"] = deepcopy(selected_transition)
    turn_entry["selected_candidate_task"] = deepcopy(selected_candidate_task)
    turn_entry["next_transition"] = deepcopy(selected_transition)
    turn_entry["transition_validation"] = {
        "status": "passed",
        "selected_candidate_index": selected_candidate_index,
    }
    if selected_grounded_action:
        turn_entry["grounded_action"] = deepcopy(selected_grounded_action)

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.append(deepcopy(selected_transition))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = []
    session_state["candidate_rejection_feedback"] = []
    session_state["rejected_turn_thought"] = ""
    _shared._sync_des_recovery_aliases(
        session_state,
        turn_entry=turn_entry,
        transition_validation=turn_entry["transition_validation"],
    )

    _shared._apply_task_effects_to_symbolic_state(selected_transition, session_state)
    session_state["pruned_actions"] = _shared._active_pruned_actions(
        session_state,
        prepared_bridge_request,
    )

    remaining_findings, remaining_conditions = _shared._remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    outline_complete = remaining_findings == 0 and remaining_conditions == 0

    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurn] outline incremental_candidates_validated: accepted LLM-selected candidate %d (%s) "
        "(prefix now %d events, complete=%s)",
        selected_candidate_index + 1,
        str(selected_transition.get("outline_id") or "").strip(),
        len(accepted_prefix),
        outline_complete,
    )

    session_state["status"] = (
        "ready_for_primitive_generation"
        if outline_complete
        else "paused_after_outline_turn"
    )
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
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
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
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
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
