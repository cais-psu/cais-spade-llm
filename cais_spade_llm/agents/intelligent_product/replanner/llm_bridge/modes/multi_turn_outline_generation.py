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
        "ready_for_primitive_generation" if outline_complete else "paused_after_outline_turn"
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
        session_state["transition_validation"] = deepcopy(turn_entry["transition_validation"])
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
        session_state["transition_validation"] = deepcopy(turn_entry["transition_validation"])
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
    session_state["outline_validation_findings"] = (
        _shared._prune_resolved_outline_validation_findings(
            list(session_state.get("outline_validation_findings") or []),
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
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
        "ready_for_primitive_generation" if outline_complete else "paused_after_outline_turn"
    )
    return decision, turn_entry


def _recovery_selection_mode(session_state: dict[str, Any]) -> str:
    mode = str(session_state.get("recovery_selection_mode") or "pure_llm").strip().lower()
    return mode if mode in {"pure_llm", "neurosymbolic"} else "pure_llm"


def _action_horizon(session_state: dict[str, Any]) -> str:
    horizon = str(session_state.get("action_horizon") or "1").strip().lower()
    return horizon if horizon in {"1", "k", "full"} else "1"


def _action_horizon_k(session_state: dict[str, Any]) -> int:
    try:
        return max(1, int(session_state.get("action_horizon_k") or 3))
    except (TypeError, ValueError):
        return 3


def _action_horizon_steps(session_state: dict[str, Any], *, action_horizon: str) -> int | str:
    raw_steps = session_state.get("action_horizon_steps")
    if raw_steps not in (None, ""):
        return raw_steps
    if action_horizon == "full":
        return "full"
    if action_horizon == "k":
        return _action_horizon_k(session_state)
    return 1


def _candidate_count(session_state: dict[str, Any]) -> int | str:
    raw_candidate_count = session_state.get("candidate_count", "auto")
    if isinstance(raw_candidate_count, str):
        candidate_count_token = raw_candidate_count.strip().lower()
        if candidate_count_token in {"auto", "n"}:
            return "auto"
        try:
            return max(1, int(candidate_count_token))
        except ValueError:
            return "auto"
    try:
        return max(1, int(raw_candidate_count))
    except (TypeError, ValueError):
        return "auto"


def _candidate_sequences_from_response(
    parsed_response: dict[str, Any],
    *,
    action_horizon: str,
) -> list[dict[str, Any]]:
    if action_horizon == "1":
        return [
            {
                "candidate_index": candidate_index,
                "surface_events": [dict(row)],
            }
            for candidate_index, row in enumerate(
                _shared._parsed_response_rows(
                    parsed_response,
                    primary_key="candidate_events",
                )
            )
        ]

    candidate_traces: list[dict[str, Any]] = []
    for candidate_index, row in enumerate(
        _shared._parsed_response_rows(
            parsed_response,
            primary_key="candidate_traces",
        )
    ):
        trace = dict(row or {})
        candidate_traces.append(
            {
                "candidate_index": candidate_index,
                "surface_events": [
                    dict(event) for event in (trace.get("events") or []) if isinstance(event, dict)
                ],
                "rationale": str(trace.get("rationale") or "").strip(),
            }
        )
    if candidate_traces:
        return candidate_traces
    return [
        {
            "candidate_index": candidate_index,
            "surface_events": [dict(row)],
        }
        for candidate_index, row in enumerate(
            _shared._parsed_response_rows(
                parsed_response,
                primary_key="candidate_events",
            )
        )
    ]


def _llm_selected_candidate_index(parsed_response: dict[str, Any]) -> int | None:
    raw_index = parsed_response.get("selected_candidate_index")
    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
        return None
    return int(raw_index)


def _resource_switch_count(events: list[dict[str, Any]]) -> int:
    resource_jids = [
        str(event.get("resource_jid") or "").strip()
        for event in events
        if str(event.get("resource_jid") or "").strip()
    ]
    if not resource_jids:
        return 0
    return sum(
        1
        for previous, current in zip(resource_jids, resource_jids[1:], strict=False)
        if previous != current
    )


def _selection_score(
    *,
    progress_score: int,
    remaining_blocked_issues: int,
    event_count: int,
    resource_switch_count: int,
) -> int:
    return (
        int(progress_score) * 100
        - int(remaining_blocked_issues) * 50
        - int(event_count) * 10
        - int(resource_switch_count) * 2
    )


def _validate_candidate_sequence(
    *,
    candidate: dict[str, Any],
    sequence_index: int,
    action_horizon: str,
    action_horizon_k: int,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> dict[str, Any]:
    candidate_index = int(candidate.get("candidate_index") or 0)
    surface_events = [
        deepcopy(row) for row in (candidate.get("surface_events") or []) if isinstance(row, dict)
    ]
    evaluation: dict[str, Any] = {
        "candidate_index": candidate_index,
        "surface_events": deepcopy(surface_events),
        "event_count": len(surface_events),
        "action_horizon": action_horizon,
    }
    if surface_events:
        evaluation["surface_task"] = deepcopy(surface_events[0])
        evaluation["task"] = deepcopy(surface_events[0])
    if str(candidate.get("rationale") or "").strip():
        evaluation["rationale"] = str(candidate.get("rationale") or "").strip()

    if not surface_events:
        evaluation["valid"] = False
        evaluation["validation_findings"] = [
            _shared._candidate_schema_finding(
                task={},
                reason="candidate trace must include at least one event",
                evidence={"candidate_index": candidate_index},
            )
        ]
        return evaluation
    if action_horizon == "1" and len(surface_events) != 1:
        evaluation["valid"] = False
        evaluation["validation_findings"] = [
            _shared._candidate_schema_finding(
                task=surface_events[0],
                reason="action_horizon=1 candidate must include exactly one event",
                evidence={"candidate_index": candidate_index, "event_count": len(surface_events)},
            )
        ]
        return evaluation
    if action_horizon == "k" and len(surface_events) > action_horizon_k:
        evaluation["valid"] = False
        evaluation["validation_findings"] = [
            _shared._candidate_schema_finding(
                task=surface_events[0],
                reason="action_horizon=k candidate exceeds action_horizon_k",
                evidence={
                    "candidate_index": candidate_index,
                    "event_count": len(surface_events),
                    "action_horizon_k": action_horizon_k,
                },
            )
        ]
        return evaluation

    working_session_state = deepcopy(session_state)
    validated_events: list[dict[str, Any]] = []
    committed_events: list[dict[str, Any]] = []
    grounded_actions: list[dict[str, Any]] = []
    progress_score = 0
    progress_details: list[dict[str, Any]] = []

    for event_index, surface_event in enumerate(surface_events):
        working_task = deepcopy(surface_event)
        evaluation["task"] = deepcopy(working_task)
        validated_task, schema_findings = _shared._derive_candidate_outline_task(
            candidate_task=working_task,
            session_state=working_session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if schema_findings or not validated_task:
            evaluation["valid"] = False
            evaluation["validation_findings"] = deepcopy(schema_findings)
            evaluation["failed_event_index"] = event_index
            return evaluation
        evaluation["validated_task"] = deepcopy(validated_task)

        pruned_row = _shared._matching_active_pruned_action(
            task=validated_task,
            session_state=working_session_state,
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
            evaluation["failed_event_index"] = event_index
            return evaluation

        findings, grounded_action = _shared._validate_single_outline_task(
            planner=planner,
            task=dict(validated_task or {}),
            session_state=working_session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        if findings:
            evaluation["valid"] = False
            evaluation["validation_findings"] = deepcopy(findings)
            evaluation["failed_event_index"] = event_index
            return evaluation
        if grounded_action:
            grounded_actions.append(deepcopy(grounded_action))

        event_progress_score, event_progress_detail = _shared._candidate_progress_score(
            task=dict(validated_task or {}),
            session_state=working_session_state,
            prepared_bridge_request=prepared_bridge_request,
        )
        progress_score += int(event_progress_score or 0)
        progress_details.append(deepcopy(event_progress_detail or {}))

        committed_event = _shared._commit_selected_candidate_task(
            task=dict(validated_task or {}),
            sequence_index=sequence_index + event_index,
        )
        accepted_prefix = list(working_session_state.get("accepted_outline_prefix") or [])
        accepted_prefix.append(deepcopy(committed_event))
        working_session_state["accepted_outline_prefix"] = accepted_prefix
        _shared._sync_des_recovery_aliases(working_session_state)
        _shared._apply_task_effects_to_symbolic_state(
            committed_event,
            working_session_state,
        )
        validated_events.append(deepcopy(validated_task))
        committed_events.append(deepcopy(committed_event))

    remaining_findings, remaining_conditions = _shared._remaining_blocked_issue_counts(
        session_state=working_session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    remaining_blocked_issues = int(remaining_findings or 0) + int(remaining_conditions or 0)
    if action_horizon == "full" and remaining_blocked_issues > 0:
        evaluation["valid"] = False
        evaluation["validation_findings"] = [
            _shared._candidate_schema_finding(
                task=committed_events[-1] if committed_events else {},
                reason="action_horizon=full candidate did not reconnect to the nominal plant",
                evidence={
                    "candidate_index": candidate_index,
                    "remaining_blocked_issues": remaining_blocked_issues,
                },
            )
        ]
        return evaluation

    resource_switches = _resource_switch_count(committed_events)
    evaluation["valid"] = True
    evaluation["validation_findings"] = []
    evaluation["validated_events"] = deepcopy(validated_events)
    evaluation["committed_events"] = deepcopy(committed_events)
    evaluation["grounded_actions"] = deepcopy(grounded_actions)
    evaluation["progress_score"] = progress_score
    evaluation["progress_details"] = deepcopy(progress_details)
    evaluation["remaining_blocked_issues"] = remaining_blocked_issues
    evaluation["resource_switch_count"] = resource_switches
    evaluation["selection_score"] = _selection_score(
        progress_score=progress_score,
        remaining_blocked_issues=remaining_blocked_issues,
        event_count=len(committed_events),
        resource_switch_count=resource_switches,
    )
    if validated_events:
        evaluation["task"] = deepcopy(validated_events[0])
        evaluation["validated_task"] = deepcopy(validated_events[0])
    if grounded_actions:
        evaluation["grounded_action"] = deepcopy(grounded_actions[0])
    return evaluation


def _select_neurosymbolic_candidate(
    candidate_evaluations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    valid_candidates = [
        row for row in candidate_evaluations if isinstance(row, dict) and bool(row.get("valid"))
    ]
    if not valid_candidates:
        return None
    return max(
        valid_candidates,
        key=lambda row: (
            int(row.get("selection_score") or 0),
            -int(row.get("remaining_blocked_issues") or 0),
            -int(row.get("event_count") or 0),
            -int(row.get("candidate_index") or 0),
        ),
    )


async def _handle_outline_incremental_candidates_validated(  # noqa: PLR0915
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Incremental with multiple candidate rows or traces and selectable ownership."""
    turn_entry: dict[str, Any] = {}
    sequence_index = _shared._next_recovery_sequence_index(session_state)
    recovery_selection_mode = _recovery_selection_mode(session_state)
    action_horizon = _action_horizon(session_state)
    action_horizon_k = _action_horizon_k(session_state)
    action_horizon_steps = _action_horizon_steps(
        session_state,
        action_horizon=action_horizon,
    )
    candidate_count = _candidate_count(session_state)
    turn_entry["recovery_selection_mode"] = recovery_selection_mode
    turn_entry["action_horizon"] = action_horizon_steps
    turn_entry["candidate_count"] = candidate_count
    session_state["outline_validation_findings"] = []
    session_state["pruned_actions"] = _shared._active_pruned_actions(
        session_state,
        prepared_bridge_request,
    )

    candidate_sequences = _candidate_sequences_from_response(
        parsed_response,
        action_horizon=action_horizon,
    )
    if action_horizon == "1":
        turn_entry["candidate_events"] = [
            deepcopy(row["surface_events"][0])
            for row in candidate_sequences
            if row.get("surface_events")
        ]
    else:
        turn_entry["candidate_traces"] = [
            {
                "events": deepcopy(row.get("surface_events") or []),
                **(
                    {"rationale": str(row.get("rationale") or "").strip()}
                    if str(row.get("rationale") or "").strip()
                    else {}
                ),
            }
            for row in candidate_sequences
        ]
    candidate_bound = int(session_state.get("candidate_bound") or _shared._DEFAULT_CANDIDATE_BOUND)
    if candidate_count == "auto":
        count_is_valid = 1 <= len(candidate_sequences) <= candidate_bound
        count_error = (
            "outline response must include at least 1 candidate "
            "and stay within the runtime schema limit"
        )
        expected_count_log = "1-%d"
        expected_count_args = (candidate_bound,)
    else:
        count_is_valid = len(candidate_sequences) == int(candidate_count)
        count_error = (
            f"outline response must include exactly {candidate_count} candidate"
            f"{'' if int(candidate_count) == 1 else 's'}"
        )
        expected_count_log = "%d"
        expected_count_args = (int(candidate_count),)
    if not count_is_valid:
        turn_entry["error"] = count_error
        _logger.warning(
            "[MultiTurn] outline incremental_candidates_validated: expected "
            + expected_count_log
            + " candidates, got %d",
            *expected_count_args,
            len(candidate_sequences),
        )
        return "need_revision", turn_entry

    llm_selected_candidate_index = _llm_selected_candidate_index(parsed_response)
    if llm_selected_candidate_index is not None:
        turn_entry["llm_selected_candidate_index"] = llm_selected_candidate_index
    if recovery_selection_mode == "pure_llm" and llm_selected_candidate_index is None:
        finding = _shared._candidate_schema_finding(
            task={},
            reason=(
                "outline response must include integer selected_candidate_index "
                "pointing at the proposed candidates"
            ),
            evidence={"field": "selected_candidate_index"},
        )
        turn_entry["validation_findings"] = [deepcopy(finding)]
        turn_entry["transition_validation"] = {
            "status": "rejected",
            "findings": [deepcopy(finding)],
        }
        session_state["outline_validation_findings"] = [deepcopy(finding)]
        session_state["transition_validation"] = deepcopy(turn_entry["transition_validation"])
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry
    if (
        recovery_selection_mode == "pure_llm"
        and llm_selected_candidate_index is not None
        and not (0 <= llm_selected_candidate_index < len(candidate_sequences))
    ):
        finding = _shared._candidate_schema_finding(
            task={},
            reason="selected_candidate_index must reference a proposed candidate",
            evidence={
                "field": "selected_candidate_index",
                "selected_candidate_index": llm_selected_candidate_index,
                "candidate_count": len(candidate_sequences),
            },
        )
        turn_entry["validation_findings"] = [deepcopy(finding)]
        turn_entry["transition_validation"] = {
            "status": "rejected",
            "findings": [deepcopy(finding)],
        }
        session_state["outline_validation_findings"] = [deepcopy(finding)]
        session_state["transition_validation"] = deepcopy(turn_entry["transition_validation"])
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry

    candidate_evaluations = [
        _validate_candidate_sequence(
            candidate=dict(candidate),
            sequence_index=sequence_index,
            action_horizon=action_horizon,
            action_horizon_k=action_horizon_k,
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
        )
        for candidate in candidate_sequences
    ]

    turn_entry["candidate_evaluations"] = deepcopy(candidate_evaluations)
    _shared._promote_durable_candidate_rejections(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        candidate_evaluations=candidate_evaluations,
    )

    if recovery_selection_mode == "pure_llm":
        selected_candidate_index = int(llm_selected_candidate_index or 0)
        selected_by = "pure_llm"
        selected = next(
            (
                row
                for row in candidate_evaluations
                if isinstance(row, dict)
                and int(row.get("candidate_index", -1)) == selected_candidate_index
            ),
            None,
        )
    else:
        selected = _select_neurosymbolic_candidate(candidate_evaluations)
        selected_candidate_index = (
            int(selected.get("candidate_index") or 0) if isinstance(selected, dict) else -1
        )
        selected_by = "neurosymbolic"

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
            "selected_by": selected_by,
            "findings": deepcopy(feedback_rows),
        }
        session_state["transition_validation"] = deepcopy(turn_entry["transition_validation"])
        session_state["rejected_turn_thought"] = str(parsed_response.get("thought") or "").strip()
        _logger.info(
            "[MultiTurn] outline incremental_candidates_validated: rejected selected candidate %d (%s)",
            selected_candidate_index + 1 if selected_candidate_index >= 0 else 0,
            selected_by,
        )
        stagnation = int(session_state.get("outline_stagnation_count") or 0) + 1
        session_state["outline_stagnation_count"] = stagnation
        status_counts: dict[str, int] = {}
        for row in candidate_evaluations:
            if not isinstance(row, dict):
                continue
            findings = [
                dict(f) for f in (row.get("validation_findings") or []) if isinstance(f, dict)
            ]
            if not findings:
                continue
            status = _shared._finding_event_status_for_logging(findings[0])
            status_counts[status] = int(status_counts.get(status) or 0) + 1
        status_summary = (
            ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
            or "none"
        )
        rejection_codes = [
            str(f.get("constraint_code") or "unknown")
            for row in candidate_evaluations
            if isinstance(row, dict)
            for f in (row.get("validation_findings") or [])
            if isinstance(f, dict)
        ]
        _logger.info(
            "[MultiTurn] Stagnation %d — status_counts: %s",
            stagnation,
            status_summary,
        )
        _logger.debug(
            "[MultiTurn] Stagnation %d — rejection codes: %s",
            stagnation,
            rejection_codes,
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry

    session_state["outline_stagnation_count"] = 0

    selected_candidate_task = deepcopy(dict(selected.get("task") or {}))
    selected_committed_events = [
        deepcopy(row) for row in (selected.get("committed_events") or []) if isinstance(row, dict)
    ]
    selected_transition = (
        deepcopy(selected_committed_events[0]) if selected_committed_events else {}
    )
    selected_grounded_actions = [
        deepcopy(row) for row in (selected.get("grounded_actions") or []) if isinstance(row, dict)
    ]

    turn_entry["selected_candidate_index"] = selected_candidate_index
    turn_entry["selected_by"] = selected_by
    turn_entry["selection_score"] = int(selected.get("selection_score") or 0)
    turn_entry["selected_transition"] = deepcopy(selected_transition)
    turn_entry["selected_transition_sequence"] = deepcopy(selected_committed_events)
    turn_entry["selected_candidate_task"] = deepcopy(selected_candidate_task)
    turn_entry["selected_candidate_trace"] = deepcopy(selected.get("surface_events") or [])
    turn_entry["next_transition"] = deepcopy(selected_transition)
    turn_entry["transition_validation"] = {
        "status": "passed",
        "selected_candidate_index": selected_candidate_index,
        "selected_by": selected_by,
    }
    if llm_selected_candidate_index is not None:
        turn_entry["llm_selected_candidate_index"] = llm_selected_candidate_index
    if selected_grounded_actions:
        turn_entry["grounded_action"] = deepcopy(selected_grounded_actions[0])
        turn_entry["grounded_actions"] = deepcopy(selected_grounded_actions)

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.extend(deepcopy(selected_committed_events))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = []
    session_state["candidate_rejection_feedback"] = []
    session_state["rejected_turn_thought"] = ""
    _shared._sync_des_recovery_aliases(
        session_state,
        turn_entry=turn_entry,
        transition_validation=turn_entry["transition_validation"],
    )

    for selected_event in selected_committed_events:
        _shared._apply_task_effects_to_symbolic_state(selected_event, session_state)
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
        "[MultiTurn] outline incremental_candidates_validated: accepted %s-selected candidate %d (%s) "
        "(events=%d, score=%d, prefix now %d events, complete=%s)",
        selected_by,
        selected_candidate_index + 1,
        str(selected_transition.get("outline_id") or "").strip(),
        len(selected_committed_events),
        int(selected.get("selection_score") or 0),
        len(accepted_prefix),
        outline_complete,
    )

    session_state["status"] = (
        "ready_for_primitive_generation" if outline_complete else "paused_after_outline_turn"
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
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()
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
