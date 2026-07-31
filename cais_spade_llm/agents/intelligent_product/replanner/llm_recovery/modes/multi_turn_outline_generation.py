"""Outline-phase helpers for the multi-turn recovery."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
    multi_turn as _shared,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_validation_service import (
    PHYSICAL_FEASIBILITY,
    SAFETY,
    SYNTAX_AND_GROUNDING_VALIDATION,
    TRANSITION_FEASIBILITY,
    build_recovery_physical_validation_input,
    build_recovery_safety_validation_input,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_validation_service import (
    projected_outline_validation_context as _service_projected_outline_validation_context,
)
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
    recovery_validation_stage,
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
    prepared_recovery_request: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    return _service_projected_outline_validation_context(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )


def _pa_state_fingerprint(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
) -> str:
    """Fingerprint the ProductAgent's exact projected recovery state and blockers."""
    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    return recovery_validation_fingerprint(
        {
            "resources": {
                resource_jid: deepcopy(resources_by_jid[resource_jid])
                for resource_jid in sorted(resources_by_jid)
            },
            "parts": {
                part_name: deepcopy(parts_by_name[part_name])
                for part_name in sorted(parts_by_name)
            },
            "recovery_blockers": sorted(
                [
                    list(_shared._candidate_recovery_blocker_key(row))
                    for row in _shared._active_candidate_recovery_blockers(
                        session_state=session_state,
                        prepared_recovery_request=prepared_recovery_request,
                    )
                    if isinstance(row, dict)
                ],
                key=lambda row: json.dumps(row, sort_keys=True, default=str),
            ),
            "unresolved_condition_ids": sorted(
                _unresolved_condition_ids(
                    session_state=session_state,
                    prepared_recovery_request=prepared_recovery_request,
                )
            ),
            "safety_dfa_states": {
                str(rule_id): str(state)
                for rule_id, state in sorted(
                    dict(
                        session_state.get("projected_safety_dfa_states") or {}
                    ).items()
                )
            },
        }
    )


async def _handle_outline_single_pass(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
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
            prepared_recovery_request=prepared_recovery_request,
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
            prepared_recovery_request=prepared_recovery_request,
        )
        if findings or not validated_task:
            turn_entry["validation_findings"] = deepcopy(findings)
            session_state["outline_validation_findings"] = deepcopy(findings)
            session_state["status"] = "paused_after_outline_turn"
            return "need_revision", turn_entry
        if grounded_action:
            turn_entry.setdefault("grounded_trace", []).append(deepcopy(grounded_action))
        committed_task = _shared._commit_selected_candidate_task(
            task=validated_task,
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
    prepared_recovery_request: dict[str, Any],
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
        prepared_recovery_request=prepared_recovery_request,
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
        prepared_recovery_request=prepared_recovery_request,
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
        task=validated_task,
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
            prepared_recovery_request=prepared_recovery_request,
        )
        if suffix_schema_findings or not validated_suffix:
            break
        suffix_findings, _suffix_grounded_action = _shared._validate_single_outline_task(
            planner=planner,
            task=dict(validated_suffix or {}),
            session_state=working_session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        if suffix_findings or not validated_suffix:
            break
        committed_suffix = _shared._commit_selected_candidate_task(
            task=validated_suffix,
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
    prepared_recovery_request: dict[str, Any],
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
        prepared_recovery_request=prepared_recovery_request,
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
        prepared_recovery_request=prepared_recovery_request,
    )
    next_transition = _shared._commit_selected_candidate_task(
        task=validated_task,
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
            prepared_recovery_request=prepared_recovery_request,
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
    if _recovery_selection_mode(session_state) == "neurosymbolic":
        return "auto"
    if _action_horizon(session_state) == "1":
        return _shared._ONE_STEP_CANDIDATE_COUNT
    raw_candidate_count = session_state.get("candidate_count", "auto")
    if isinstance(raw_candidate_count, str):
        candidate_count_token = raw_candidate_count.strip().lower()
        if candidate_count_token in {"adaptive", "auto", "n"}:
            return "auto"
        try:
            return max(1, int(candidate_count_token))
        except ValueError:
            return "auto"
    try:
        return max(1, int(raw_candidate_count))
    except (TypeError, ValueError):
        return "auto"


def _exact_condition_identifier(condition: dict[str, Any]) -> str:
    identifier = str(condition.get("condition_id") or condition.get("id") or "").strip()
    if identifier:
        return identifier
    return json.dumps(condition, sort_keys=True, separators=(",", ":"), default=str)


def _unresolved_condition_ids(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
) -> set[str]:
    return {
        _exact_condition_identifier(condition)
        for condition in _shared._active_continuation_conditions(
            prepared_recovery_request
        )
        if not _shared._continuation_condition_satisfied(
            condition,
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
    }


def _unresolved_conditions(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return exact nominal conditions not satisfied by projected state."""
    return [
        deepcopy(condition)
        for condition in _shared._active_continuation_conditions(
            prepared_recovery_request
        )
        if not _shared._continuation_condition_satisfied(
            condition,
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
    ]


def _latest_enabled_capability_results(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
) -> list[dict[str, Any]]:
    del prepared_recovery_request
    return [
        deepcopy(row)
        for row in (session_state.get("latest_enabled_capability_results") or [])
        if isinstance(row, dict)
    ]


def _recovery_relevant_resource_ids(
    *,
    unresolved_condition_ids: set[str],
    prepared_recovery_request: dict[str, Any],
    resource_ids: set[str],
) -> set[str]:
    conditions = [
        condition
        for condition in _shared._active_continuation_conditions(
            prepared_recovery_request
        )
        if _exact_condition_identifier(condition) in unresolved_condition_ids
    ]
    exact_resources = {
        str(condition.get("entity") or "").strip()
        for condition in conditions
        if str(condition.get("entity_kind") or "").strip().lower() == "resource"
        and str(condition.get("entity") or "").strip() in resource_ids
    }
    has_part_or_supervisor_obligation = any(
        str(condition.get("entity_kind") or "").strip().lower() == "part"
        for condition in conditions
    ) or len(conditions) < len(unresolved_condition_ids)
    if has_part_or_supervisor_obligation:
        exact_resources.update(resource_ids)
    return exact_resources or set(resource_ids)


def _nominal_reentry_event_id(task: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task_id": str(task.get("task_id") or "").strip(),
            "resource_jid": str(task.get("resource_jid") or "").strip(),
            "function_name": str(task.get("function_name") or "").strip(),
            "part_name": str(task.get("part_name") or "").strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _nominal_task_tool_row(
    *,
    task: dict[str, Any],
    tools_catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    function_name = str(task.get("function_name") or "").strip()
    resource_name = str(task.get("resource_jid") or "").split("@", 1)[0].strip()
    fallback: dict[str, Any] = {}
    for raw_row in tools_catalog:
        if not isinstance(raw_row, dict):
            continue
        row = dict(raw_row)
        if str(row.get("function") or "").strip() != function_name:
            continue
        owner = str(row.get("function_owner_agent") or "").split("@", 1)[0].strip()
        if owner == resource_name:
            return row
        if not fallback:
            fallback = row
    return fallback


def _task_reaches_pending_task(
    *,
    task_id: str,
    pending_task_ids: set[str],
    successors_by_task_id: dict[str, set[str]],
) -> bool:
    if task_id in pending_task_ids:
        return True
    frontier = list(successors_by_task_id.get(task_id) or set())
    visited: set[str] = set()
    while frontier:
        successor_id = frontier.pop()
        if successor_id in pending_task_ids:
            return True
        if successor_id in visited:
            continue
        visited.add(successor_id)
        frontier.extend(successors_by_task_id.get(successor_id) or set())
    return False


def _nominal_reentry_event_rows(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build private exact nominal events on affected continuation paths."""
    recovery_resources = dict(prepared_recovery_request.get("recovery_resources") or {})
    pending_task_ids = {
        str(task.get("id") or "").strip()
        for raw_resource in recovery_resources.values()
        if isinstance(raw_resource, dict)
        for task in (raw_resource.get("pending_tasks") or [])
        if isinstance(task, dict) and str(task.get("id") or "").strip()
    }
    if not pending_task_ids:
        return []

    task_rows = [
        deepcopy(task)
        for raw_tasks in dict(
            prepared_recovery_request.get("requirement_task_index") or {}
        ).values()
        if isinstance(raw_tasks, list)
        for task in raw_tasks
        if isinstance(task, dict) and str(task.get("task_id") or "").strip()
    ]
    if not task_rows:
        return []
    task_by_id = {
        str(task.get("task_id") or "").strip(): task for task in task_rows
    }
    successors_by_task_id = {
        task_id: {
            str(successor_id).strip()
            for successor_id in (task.get("successors") or [])
            if str(successor_id).strip()
        }
        for task_id, task in task_by_id.items()
    }
    tasks_by_requirement: dict[str, list[dict[str, Any]]] = {}
    task_requirement_map = dict(
        prepared_recovery_request.get("task_requirement_map") or {}
    )
    for task_id, task in task_by_id.items():
        requirement_id = str(task_requirement_map.get(task_id) or "").strip()
        if requirement_id:
            tasks_by_requirement.setdefault(requirement_id, []).append(task)

    pending_requirement_ids = {
        str(task_requirement_map.get(task_id) or "").strip()
        for task_id in pending_task_ids
        if str(task_requirement_map.get(task_id) or "").strip()
    }
    tools_catalog = [
        deepcopy(row)
        for row in (prepared_recovery_request.get("tools_catalog") or [])
        if isinstance(row, dict)
    ]
    del session_state
    rows: list[dict[str, Any]] = []
    for requirement_id in sorted(pending_requirement_ids):
        requirement_tasks = tasks_by_requirement.get(requirement_id) or []
        requirement_pending_ids = {
            task_id
            for task_id in pending_task_ids
            if str(task_requirement_map.get(task_id) or "").strip() == requirement_id
        }
        affected_parts = {
            str(task.get("part_name") or "").strip()
            for task in requirement_tasks
            if str(task.get("task_id") or "").strip() in requirement_pending_ids
            and str(task.get("part_name") or "").strip()
        }
        affected_resources = {
            str(task.get("resource_jid") or "").strip()
            for task in requirement_tasks
            if str(task.get("task_id") or "").strip() in requirement_pending_ids
            and str(task.get("resource_jid") or "").strip()
        }
        for task in requirement_tasks:
            task_id = str(task.get("task_id") or "").strip()
            resource_jid = str(task.get("resource_jid") or "").strip()
            part_name = str(task.get("part_name") or "").strip()
            if resource_jid not in affected_resources:
                continue
            if affected_parts and part_name and part_name not in affected_parts:
                continue
            if str(task.get("status") or "").strip() not in {"completed", "pending"}:
                continue
            if not _task_reaches_pending_task(
                task_id=task_id,
                pending_task_ids=requirement_pending_ids,
                successors_by_task_id=successors_by_task_id,
            ):
                continue
            tool_row = _nominal_task_tool_row(
                task=task,
                tools_catalog=tools_catalog,
            )
            if not tool_row:
                continue
            params = dict(task.get("params") or {})
            context_mapping = dict(tool_row.get("context_mapping") or {})
            location_param = str(context_mapping.get("location_param") or "").strip()
            location_value = params.get(location_param) if location_param else None
            safety_task = {
                "outline_id": task_id,
                "event_name": str(task.get("function_name") or "").strip(),
                "resource_jid": resource_jid,
                "part_name": part_name or None,
                "action_target": {
                    "target_location": deepcopy(location_value),
                },
            }
            rows.append(
                {
                    "event_id": _nominal_reentry_event_id(task),
                    "task": deepcopy(task),
                    "tool": deepcopy(tool_row),
                    "safety_task": safety_task,
                    "safety_signature": {
                        "inferable_primary_part": part_name,
                        "task_kind": "part_handling" if part_name else "resource_action",
                        "changes_part_world": bool(
                            part_name
                            and (
                                tool_row.get("part_transition")
                                or location_value not in (None, "")
                            )
                        ),
                    },
                }
            )
    return rows


def _admissible_recovery_enabled_event_ids(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    unresolved_condition_ids: set[str],
) -> set[str]:
    if not unresolved_condition_ids:
        return set()
    return {
        str(row.get("event_id") or "").strip()
        for row in _latest_enabled_capability_results(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        if row.get("allowed") is True
        and str(row.get("event_id") or "").strip()
    }


def _recovery_relevant_part_names(
    *,
    unresolved_condition_ids: set[str],
    prepared_recovery_request: dict[str, Any],
    available_part_names: set[str],
) -> set[str]:
    relevant = {
        str(condition.get("entity") or "").strip()
        for condition in _shared._active_continuation_conditions(
            prepared_recovery_request
        )
        if _exact_condition_identifier(condition) in unresolved_condition_ids
        and str(condition.get("entity_kind") or "").strip().lower() == "part"
        and str(condition.get("entity") or "").strip() in available_part_names
    }
    return relevant or set(available_part_names)


def _symbolically_enabled_recovery_event_instances(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    unresolved_condition_ids: set[str],
) -> list[dict[str, Any]]:
    if not unresolved_condition_ids:
        return []
    return [
        {
            "event_id": str(row.get("event_id") or ""),
            "resource_jid": str(dict(row.get("task") or {}).get("resource_jid") or ""),
            "part_name": dict(row.get("task") or {}).get("part_name"),
            "task": deepcopy(dict(row.get("task") or {})),
            "transition_feasibility": deepcopy(
                row.get("transition_feasibility") or {}
            ),
            "physical_feasibility": deepcopy(row.get("physical_feasibility") or {}),
        }
        for row in _latest_enabled_capability_results(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        if row.get("allowed") is True
    ]


def _admissible_nominal_reentry_event_ids(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    cca_admissible_event_ids: set[str],
) -> set[str]:
    nominal_ids = {
        str(row.get("event_id") or "").strip()
        for row in _nominal_reentry_event_rows(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        if str(row.get("event_id") or "").strip()
    }
    return nominal_ids & set(cca_admissible_event_ids)


async def _agent_filtered_recovery_enabledness(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    unresolved_condition_ids: set[str],
    planner: Any,
) -> dict[str, Any]:
    """Return capability instances admitted by Resource Agents and CCA."""
    result: dict[str, Any] = {
        "symbolically_enabled_event_ids": [],
        "ra_admissible_event_ids": [],
        "cca_admissible_event_ids": [],
        "admissible_event_ids": [],
        "admissible_nominal_reentry_event_ids": [],
        "resource_enabled_goal_recovery_event_ids": [],
        "cca_admissible_goal_recovery_event_ids": [],
        "admissible_goal_recovery_event_ids": [],
        "enabled_capability_results": [],
        "event_evaluations": [],
        "safety_rule_fingerprint": "",
        "live_safety_dfa_state_fingerprint": "",
        "future_goal_query_complete": True,
    }
    if not unresolved_condition_ids:
        return result

    product_agent = getattr(planner, "product_agent", None)
    request_physical = getattr(
        product_agent, "request_recovery_outline_physical_validation", None
    )
    request_safety = getattr(
        product_agent, "request_recovery_outline_safety_validation", None
    )
    if not callable(request_physical) or not callable(request_safety):
        result["unavailable_reason"] = "recovery validation transport is unavailable"
        return result

    recovery_session_id = str(
        session_state.get("recovery_session_id")
        or session_state.get("session_id")
        or dict(prepared_recovery_request.get("recovery_session") or {}).get(
            "session_id"
        )
        or prepared_recovery_request.get("recovery_session_id")
        or ""
    ).strip()
    turn_index = int(
        session_state.get("turn_index")
        or len(list(session_state.get("turns") or [])) + 1
    )
    state_fingerprint = _pa_state_fingerprint(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    configured_resource_ids = {
        str(resource_jid)
        for resource_jid in dict(
            prepared_recovery_request.get("recovery_resources") or {}
        )
        if str(resource_jid)
    }
    relevant_resources = _recovery_relevant_resource_ids(
        unresolved_condition_ids=unresolved_condition_ids,
        prepared_recovery_request=prepared_recovery_request,
        resource_ids=configured_resource_ids,
    )
    relevant_parts = _recovery_relevant_part_names(
        unresolved_condition_ids=unresolved_condition_ids,
        prepared_recovery_request=prepared_recovery_request,
        available_part_names=set(parts_by_name),
    )
    goal_conditions = [
        {
            **deepcopy(condition),
            "condition_id": _exact_condition_identifier(condition),
        }
        for condition in _unresolved_conditions(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
    ]

    async def validate_resource_group(
        resource_jid: str,
    ) -> tuple[str, dict[str, Any] | Exception]:
        part_contexts = [
            {
                "part_name": part_name,
                **deepcopy(dict(parts_by_name.get(part_name) or {})),
            }
            for part_name in sorted(relevant_parts)
        ]
        payload = {
            "recovery_session_id": recovery_session_id,
            "turn_index": turn_index,
            "state_fingerprint": state_fingerprint,
            "candidates": [],
            "enabledness_query": {
                "projected_resource_snapshot": deepcopy(
                    resources_by_jid.get(resource_jid) or {}
                ),
                "part_contexts": deepcopy(part_contexts),
            },
            "future_goal_query": {
                "projected_resource_snapshot": deepcopy(
                    resources_by_jid.get(resource_jid) or {}
                ),
                "part_contexts": deepcopy(part_contexts),
                "goal_conditions": deepcopy(goal_conditions),
            },
        }
        try:
            reply = await request_physical(
                resource_jid=resource_jid,
                payload=payload,
                timeout_s=10.0,
            )
            return resource_jid, reply
        except Exception as exc:  # noqa: BLE001 - enabledness must fail closed
            return resource_jid, exc

    ra_replies = await asyncio.gather(
        *[
            validate_resource_group(resource_jid)
            for resource_jid in sorted(relevant_resources)
        ]
    )
    evaluations_by_event_id: dict[str, dict[str, Any]] = {}
    ra_admissible_instances: list[dict[str, Any]] = []
    future_goal_instances: list[dict[str, Any]] = []
    seen_future_goal_event_ids: set[str] = set()
    instance_by_index: dict[int, dict[str, Any]] = {}
    next_enabledness_index = 0
    for resource_jid, reply_or_error in ra_replies:
        if isinstance(reply_or_error, Exception):
            result["future_goal_query_complete"] = False
            result["event_evaluations"].append(
                {
                    "resource_jid": resource_jid,
                    "ra_status": "unavailable",
                    "cca_status": "skipped",
                    "constraint_codes": ["resource_validation_unavailable"],
                    "reason": str(reply_or_error),
                }
            )
            continue
        reply = dict(reply_or_error or {})
        if not isinstance(
            reply.get("future_goal_capability_results"),
            list,
        ):
            result["future_goal_query_complete"] = False
        for raw_row in reply.get("enabled_capability_results") or []:
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            task = deepcopy(dict(row.get("task") or {}))
            if str(task.get("resource_jid") or "") != resource_jid:
                continue
            event_id = str(row.get("event_id") or "")
            if not event_id:
                continue
            transition_feasibility = deepcopy(
                dict(row.get("transition_feasibility") or {})
            )
            atomic_transitions = [
                deepcopy(transition_row)
                for transition_row in (
                    row.get("_cca_atomic_transitions") or []
                )
                if isinstance(transition_row, dict)
            ]
            public_row = {
                key: deepcopy(value)
                for key, value in row.items()
                if not str(key).startswith("_")
            }
            instance = {
                **public_row,
                "enabledness_index": next_enabledness_index,
                "resource_jid": resource_jid,
                "part_name": task.get("part_name"),
                "task": task,
                "_cca_atomic_transitions": atomic_transitions,
                "safety_input": build_recovery_safety_validation_input(
                    task=task,
                    session_state=session_state,
                    prepared_recovery_request=prepared_recovery_request,
                    calculated_successor=deepcopy(
                        transition_feasibility.get(
                            "calculated_successor"
                        )
                        or {}
                    ),
                ),
            }
            instance_by_index[next_enabledness_index] = instance
            next_enabledness_index += 1
            result["enabled_capability_results"].append(
                {
                    key: deepcopy(value)
                    for key, value in instance.items()
                    if not str(key).startswith("_")
                    and key != "safety_input"
                }
            )
            transition_allowed = bool(
                dict(row.get("transition_feasibility") or {}).get("allowed")
            )
            if transition_allowed:
                result["symbolically_enabled_event_ids"].append(event_id)
            evaluation = {
                "event_id": event_id,
                "resource_jid": resource_jid,
                "part_name": task.get("part_name"),
                "ra_status": "passed" if bool(row.get("allowed")) else "rejected",
                "cca_status": "skipped",
                "constraint_codes": sorted(
                    {
                        str(
                            dict(row.get(result_name) or {}).get(
                                "constraint_code"
                            )
                            or ""
                        ).strip()
                        for result_name in (
                            "transition_feasibility",
                            "physical_feasibility",
                        )
                        if str(
                            dict(row.get(result_name) or {}).get(
                                "constraint_code"
                            )
                            or ""
                        ).strip()
                    }
                ),
            }
            evaluations_by_event_id[event_id] = evaluation
            evaluation["ra_status"] = "passed" if bool(row.get("allowed")) else "rejected"
            evaluation["ra_mocked"] = bool(reply.get("mocked"))
            if bool(row.get("allowed")):
                ra_admissible_instances.append(instance)
        for raw_row in reply.get("future_goal_capability_results") or []:
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            task = deepcopy(dict(row.get("task") or {}))
            if str(task.get("resource_jid") or "") != resource_jid:
                continue
            event_id = str(row.get("event_id") or "").strip()
            if not event_id or event_id in seen_future_goal_event_ids:
                continue
            seen_future_goal_event_ids.add(event_id)
            future_instance = {
                "enabledness_index": next_enabledness_index,
                "query_kind": "future_goal",
                "event_id": event_id,
                "resource_jid": resource_jid,
                "part_name": task.get("part_name"),
                "task": task,
                "resource_enabled": bool(
                    row.get("resource_enabled") is True
                ),
                "signature": deepcopy(row.get("signature") or {}),
                "_cca_atomic_transitions": [
                    deepcopy(transition_row)
                    for transition_row in (
                        row.get("_cca_atomic_transitions") or []
                    )
                    if isinstance(transition_row, dict)
                ],
            }
            instance_by_index[next_enabledness_index] = future_instance
            next_enabledness_index += 1
            future_goal_instances.append(future_instance)

    result["symbolically_enabled_event_ids"] = sorted(
        set(result["symbolically_enabled_event_ids"])
    )
    result["ra_admissible_event_ids"] = sorted(
        {str(row.get("event_id") or "") for row in ra_admissible_instances}
    )
    result["resource_enabled_goal_recovery_event_ids"] = sorted(
        {
            str(row.get("event_id") or "")
            for row in future_goal_instances
            if row.get("resource_enabled") is True
            and str(row.get("event_id") or "")
        }
    )
    if not ra_admissible_instances and not future_goal_instances:
        result["event_evaluations"] = [
            evaluations_by_event_id[event_id]
            for event_id in sorted(evaluations_by_event_id)
        ]
        return result

    nominal_reentry_events = _nominal_reentry_event_rows(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    cca_candidates: list[dict[str, Any]] = []
    for instance in ra_admissible_instances:
        safety_input = deepcopy(instance.get("safety_input") or {})
        safety_input.setdefault("llm_input", {})["nominal_reentry_events"] = [
            {
                "event_id": str(row.get("event_id") or ""),
                "task": deepcopy(row.get("safety_task") or {}),
                "signature": deepcopy(row.get("safety_signature") or {}),
            }
            for row in nominal_reentry_events
            if str(row.get("event_id") or "")
        ]
        cca_candidates.append(
            {
                "candidate_index": int(instance.get("enabledness_index") or 0),
                "event_id": str(instance.get("event_id") or ""),
                "task": deepcopy(instance.get("task") or {}),
                "safety_input": safety_input,
                "_cca_atomic_transitions": deepcopy(
                    instance.get("_cca_atomic_transitions") or []
                ),
                **(
                    {
                        "safety_dfa_states_before": deepcopy(
                            session_state.get("projected_safety_dfa_states") or {}
                        )
                    }
                    if session_state.get("projected_safety_dfa_states")
                    else {}
                ),
                **(
                    {
                        "safety_rule_fingerprint": str(
                            session_state.get("projected_safety_rule_fingerprint")
                            or ""
                        ).strip()
                    }
                    if str(
                        session_state.get("projected_safety_rule_fingerprint") or ""
                    ).strip()
                    else {}
                ),
                **(
                    {
                        "live_safety_dfa_state_fingerprint": str(
                            session_state.get("live_safety_dfa_state_fingerprint")
                            or ""
                        ).strip()
                    }
                    if str(
                        session_state.get("live_safety_dfa_state_fingerprint") or ""
                    ).strip()
                    else {}
                ),
            }
        )
    for instance in future_goal_instances:
        task = deepcopy(instance.get("task") or {})
        safety_input = build_recovery_safety_validation_input(
            task=task,
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
            calculated_successor=deepcopy(
                dict(task.get("expected_end_state") or {})
            ),
        )
        cca_candidates.append(
            {
                "candidate_index": int(
                    instance.get("enabledness_index") or 0
                ),
                "event_id": str(instance.get("event_id") or ""),
                "task": task,
                "safety_input": safety_input,
                "_cca_atomic_transitions": deepcopy(
                    instance.get("_cca_atomic_transitions") or []
                ),
                **(
                    {
                        "safety_dfa_states_before": deepcopy(
                            session_state.get(
                                "projected_safety_dfa_states"
                            )
                            or {}
                        )
                    }
                    if session_state.get("projected_safety_dfa_states")
                    else {}
                ),
            }
        )
    try:
        cca_reply = await request_safety(
            payload={
                "recovery_session_id": recovery_session_id,
                "turn_index": turn_index,
                "state_fingerprint": state_fingerprint,
                "candidates": cca_candidates,
            },
            timeout_s=10.0,
        )
    except Exception as exc:  # noqa: BLE001 - enabledness must fail closed
        result["future_goal_query_complete"] = False
        for instance in ra_admissible_instances:
            evaluation = evaluations_by_event_id[str(instance.get("event_id") or "")]
            evaluation["cca_status"] = "unavailable"
            evaluation["constraint_codes"] = sorted(
                set(evaluation.get("constraint_codes") or [])
                | {"safety_validation_unavailable"}
            )
            evaluation["reason"] = str(exc)
        result["event_evaluations"] = [
            evaluations_by_event_id[event_id]
            for event_id in sorted(evaluations_by_event_id)
        ]
        return result

    result["safety_rule_fingerprint"] = str(
        cca_reply.get("safety_rule_fingerprint") or ""
    ).strip()
    result["live_safety_dfa_state_fingerprint"] = str(
        cca_reply.get("live_safety_dfa_state_fingerprint") or ""
    ).strip()
    cca_admissible: set[str] = set()
    nominal_reentry_admissible: set[str] = set()
    goal_recovery_admissible: set[str] = set()
    for raw_row in cca_reply.get("results") or []:
        if not isinstance(raw_row, dict):
            continue
        row = dict(raw_row)
        instance = instance_by_index.get(int(row.get("candidate_index") or 0))
        if not instance:
            continue
        event_id = str(instance.get("event_id") or "")
        if instance.get("query_kind") == "future_goal":
            if bool(row.get("is_safe")):
                goal_recovery_admissible.add(event_id)
            continue
        evaluation = evaluations_by_event_id[event_id]
        findings = [
            dict(item)
            for item in (row.get("findings") or [])
            if isinstance(item, dict)
        ]
        evaluation["cca_status"] = "passed" if bool(row.get("is_safe")) else "rejected"
        evaluation["cca_mocked"] = bool(cca_reply.get("mocked"))
        evaluation["safety_dfa_states_before"] = deepcopy(
            row.get("safety_dfa_states_before") or {}
        )
        evaluation["safety_dfa_states_after"] = deepcopy(
            row.get("safety_dfa_states_after") or {}
        )
        nominal_reentry_admissible.update(
            str(event_id)
            for event_id in (
                row.get("admissible_nominal_reentry_event_ids_after")
                or row.get("admissible_nominal_reentry_event_ids")
                or []
            )
            if str(event_id)
        )
        if findings:
            evaluation["findings"] = deepcopy(
                list(evaluation.get("findings") or [])
                + [
                    {
                        key: deepcopy(finding.get(key))
                        for key in (
                            "validation_category",
                            "constraint_code",
                            "constraint_owner",
                            "resource_jid",
                            "part_name",
                            "reason",
                        )
                        if finding.get(key) not in (None, "", [], {})
                    }
                    for finding in findings
                ]
            )
            evaluation["constraint_codes"] = sorted(
                set(evaluation.get("constraint_codes") or [])
                | {
                    str(item.get("constraint_code") or "").strip()
                    for item in findings
                    if str(item.get("constraint_code") or "").strip()
                }
            )
        if bool(row.get("is_safe")):
            cca_admissible.add(event_id)

    result["cca_admissible_event_ids"] = sorted(cca_admissible)
    result["admissible_nominal_reentry_event_ids"] = sorted(
        nominal_reentry_admissible
    )
    result["cca_admissible_goal_recovery_event_ids"] = sorted(
        goal_recovery_admissible
    )
    result["admissible_goal_recovery_event_ids"] = sorted(
        set(result["resource_enabled_goal_recovery_event_ids"])
        & goal_recovery_admissible
    )
    result["admissible_event_ids"] = sorted(
        set(result["symbolically_enabled_event_ids"])
        & set(result["ra_admissible_event_ids"])
        & cca_admissible
    )
    result["event_evaluations"] = [
        evaluations_by_event_id[event_id]
        for event_id in sorted(evaluations_by_event_id)
    ]
    return result


async def _populate_agent_filtered_enabledness(
    *,
    candidate_evaluations: list[dict[str, Any]],
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    planner: Any,
) -> None:
    current_unresolved = _unresolved_condition_ids(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    valid_rows = [
        row
        for row in candidate_evaluations
        if isinstance(row, dict) and bool(row.get("valid"))
    ]
    current_task = asyncio.create_task(
        _agent_filtered_recovery_enabledness(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
            unresolved_condition_ids=current_unresolved,
            planner=planner,
        )
    )
    after_tasks: list[tuple[dict[str, Any], asyncio.Task[dict[str, Any]]]] = []
    for evaluation in valid_rows:
        projected_session = _candidate_projected_session(
            session_state=session_state,
            evaluation=evaluation,
        )
        remaining = _unresolved_condition_ids(
            session_state=projected_session,
            prepared_recovery_request=prepared_recovery_request,
        )
        after_tasks.append(
            (
                evaluation,
                asyncio.create_task(
                    _agent_filtered_recovery_enabledness(
                        session_state=projected_session,
                        prepared_recovery_request=prepared_recovery_request,
                        unresolved_condition_ids=remaining,
                        planner=planner,
                    )
                ),
            )
        )
    current = await current_task
    session_state["latest_enabled_capability_results"] = deepcopy(
        current.get("enabled_capability_results") or []
    )
    for evaluation, task in after_tasks:
        after = await task
        evaluation["agent_filtered_enabledness"] = True
        evaluation["admissible_recovery_enabled_event_ids_before"] = deepcopy(
            current.get("admissible_event_ids") or []
        )
        evaluation["admissible_recovery_enabled_event_ids_after"] = deepcopy(
            after.get("admissible_event_ids") or []
        )
        evaluation["recovery_enabledness_validation_before"] = deepcopy(current)
        evaluation["recovery_enabledness_validation_after"] = deepcopy(after)
        evaluation["admissible_nominal_reentry_event_ids_before"] = deepcopy(
            current.get("admissible_nominal_reentry_event_ids") or []
        )
        evaluation["admissible_nominal_reentry_event_ids_after"] = deepcopy(
            after.get("admissible_nominal_reentry_event_ids") or []
        )
        evaluation["cca_admissible_goal_recovery_event_ids_before"] = deepcopy(
            current.get("cca_admissible_goal_recovery_event_ids") or []
        )
        evaluation["cca_admissible_goal_recovery_event_ids_after"] = deepcopy(
            after.get("cca_admissible_goal_recovery_event_ids") or []
        )
        evaluation["admissible_goal_recovery_event_ids_before"] = deepcopy(
            current.get("admissible_goal_recovery_event_ids") or []
        )
        evaluation["admissible_goal_recovery_event_ids_after"] = deepcopy(
            after.get("admissible_goal_recovery_event_ids") or []
        )
        evaluation["future_goal_evaluation_complete"] = bool(
            current.get("future_goal_query_complete") is True
            and after.get("future_goal_query_complete") is True
        )


def _candidate_realizer_event_ids(
    *,
    candidate: dict[str, Any],
    evaluation: dict[str, Any],
) -> set[str]:
    del evaluation
    surface_events = [
        dict(row) for row in (candidate.get("surface_events") or []) if isinstance(row, dict)
    ]
    if len(surface_events) != 1:
        return set()
    task = surface_events[0]
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    if not resource_jid:
        return set()
    event_names = {str(task.get("event_name") or "").strip()}
    return {
        json.dumps(
            {
                "resource_jid": resource_jid,
                "event_name": event_name,
                **({"part_name": part_name} if part_name else {}),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for event_name in event_names
    }


def _candidate_effect_identifier(candidate: dict[str, Any]) -> str:
    surface_events = [
        row
        for row in (candidate.get("surface_events") or [])
        if isinstance(row, dict)
    ]
    effects = [
        {
            "resource_jid": str(row.get("resource_jid") or "").strip(),
            "part_name": str(row.get("part_name") or "").strip(),
            "expected_end_state": deepcopy(row.get("expected_end_state") or {}),
        }
        for row in surface_events
    ]
    return f"candidate_{recovery_validation_fingerprint(effects)[:16]}"


def _candidate_authored_delta_fields(
    candidate: dict[str, Any],
    evaluation: dict[str, Any],
) -> list[str]:
    """Return exact authored fields changed by one neurosymbolic candidate."""
    surface_events = [
        row
        for row in (
            evaluation.get("validated_events")
            or candidate.get("surface_events")
            or []
        )
        if isinstance(row, dict)
    ]
    if len(surface_events) != 1:
        return []
    task = surface_events[0]
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})
    delta_fields: list[str] = []
    for field_name, value in end_state.items():
        if field_name in {"resource_state", "part_state"} and isinstance(
            value,
            dict,
        ):
            start_components = dict(start_state.get(field_name) or {})
            for component_name, component_value in value.items():
                if (
                    component_name not in start_components
                    or start_components.get(component_name)
                    != component_value
                ):
                    delta_fields.append(f"{field_name}.{component_name}")
            continue
        if field_name not in start_state or start_state.get(field_name) != value:
            delta_fields.append(field_name)
    return sorted(delta_fields)


def _classify_candidate_selection_progress(
    *,
    candidate: dict[str, Any],
    evaluation: dict[str, Any],
    progressing: bool,
) -> tuple[bool, list[str]]:
    """Classify no-op and label-only effects after model-based comparison."""
    delta_fields = _candidate_authored_delta_fields(candidate, evaluation)
    if not delta_fields:
        return False, ["no_state_change"]
    if progressing:
        return True, []
    if all(
        field_name
        in {
            "resource_state",
            "resource_state.condition",
            "current_state",
            "part_state",
            "part_state.condition",
        }
        for field_name in delta_fields
    ):
        return False, ["label_only_state_change"]
    return False, []


def _candidate_projected_session(
    *,
    session_state: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    projected = deepcopy(session_state)
    projected["symbolic_resources"] = deepcopy(
        evaluation.get("projected_symbolic_resources") or {}
    )
    projected["symbolic_parts"] = deepcopy(
        evaluation.get("projected_symbolic_parts") or {}
    )
    projected["projected_safety_dfa_states"] = deepcopy(
        evaluation.get("safety_dfa_states_after") or {}
    )
    return projected


def _recovery_event_binding(event_id: str) -> dict[str, str]:
    """Return the exact resource/part binding encoded in one private event id."""
    try:
        payload = json.loads(str(event_id or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    resource_jid = str(payload.get("resource_jid") or "").strip()
    part_name = str(payload.get("part_name") or "").strip()
    if not resource_jid or not part_name:
        return {}
    return {
        "resource_jid": resource_jid,
        "part_name": part_name,
    }


def _next_modeled_continuation_binding(
    selected: dict[str, Any],
) -> dict[str, str]:
    """Keep a modeled chain only for the selected task's exact resource and part."""
    task = dict(selected.get("task") or {})
    resource_jid = _shared._task_resource_jid(task)
    part_name = _shared._task_part_name(task)
    if not resource_jid or not part_name:
        return {}
    evidence = dict(selected.get("selection_evidence") or {})
    event_ids = evidence.get("admissible_recovery_enabled_event_ids_after") or []
    for event_id in sorted(str(item) for item in event_ids if str(item).strip()):
        binding = _recovery_event_binding(event_id)
        if binding == {"resource_jid": resource_jid, "part_name": part_name}:
            return binding
    return {}


def _apply_neurosymbolic_comparison(  # noqa: C901
    *,
    candidate_sequences: list[dict[str, Any]],
    candidate_evaluations: list[dict[str, Any]],
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
) -> list[str]:
    current_unresolved = _unresolved_condition_ids(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    filtered_current_rows = [
        row
        for row in candidate_evaluations
        if isinstance(row, dict)
        and bool(row.get("agent_filtered_enabledness"))
        and "admissible_recovery_enabled_event_ids_before" in row
    ]
    current_enabled_events = (
        {
            str(item).strip()
            for item in (
                filtered_current_rows[0].get(
                    "admissible_recovery_enabled_event_ids_before"
                )
                or []
            )
            if str(item).strip()
        }
        if filtered_current_rows
        else _admissible_recovery_enabled_event_ids(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
            unresolved_condition_ids=current_unresolved,
        )
    )
    current_cca_nominal_reentry_events = {
        str(item).strip()
        for row in candidate_evaluations
        if isinstance(row, dict) and bool(row.get("valid"))
        for item in (row.get("admissible_nominal_reentry_event_ids_before") or [])
        if str(item).strip()
    }
    current_nominal_reentry_events = _admissible_nominal_reentry_event_ids(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        cca_admissible_event_ids=current_cca_nominal_reentry_events,
    )
    current_cca_goal_recovery_events = {
        str(item).strip()
        for row in candidate_evaluations
        if isinstance(row, dict) and bool(row.get("valid"))
        for item in (
            row.get("cca_admissible_goal_recovery_event_ids_before") or []
        )
        if str(item).strip()
    }
    current_admissible_goal_recovery_events = {
        str(item).strip()
        for row in candidate_evaluations
        if isinstance(row, dict) and bool(row.get("valid"))
        for item in (
            row.get("admissible_goal_recovery_event_ids_before") or []
        )
        if str(item).strip()
    }
    candidate_by_index = {
        int(row.get("candidate_index") or 0): row for row in candidate_sequences
    }
    candidates_by_successor_class: dict[str, list[dict[str, Any]]] = {}
    progressing: list[dict[str, Any]] = []
    for evaluation in candidate_evaluations:
        candidate_index = int(evaluation.get("candidate_index") or 0)
        candidate = candidate_by_index.get(candidate_index, {})
        candidate_id = _candidate_effect_identifier(candidate)
        evaluation["candidate_id"] = candidate_id
        evaluation["selection_constraint_codes"] = []
        candidates_by_successor_class.setdefault(candidate_id, []).append(evaluation)
        if not bool(evaluation.get("valid")):
            evaluation["selection_status"] = "excluded_invalid"
            continue

        projected_session = _candidate_projected_session(
            session_state=session_state,
            evaluation=evaluation,
        )
        remaining = _unresolved_condition_ids(
            session_state=projected_session,
            prepared_recovery_request=prepared_recovery_request,
        )
        cleared = current_unresolved - remaining
        introduced = remaining - current_unresolved
        enabled_events_after = (
            {
                str(item).strip()
                for item in (
                    evaluation.get("admissible_recovery_enabled_event_ids_after")
                    or []
                )
                if str(item).strip()
            }
            if bool(evaluation.get("agent_filtered_enabledness"))
            else _admissible_recovery_enabled_event_ids(
                session_state=projected_session,
                prepared_recovery_request=prepared_recovery_request,
                unresolved_condition_ids=remaining,
            )
        )
        nominal_reentry_events_after = _admissible_nominal_reentry_event_ids(
            session_state=projected_session,
            prepared_recovery_request=prepared_recovery_request,
            cca_admissible_event_ids={
                str(item).strip()
                for item in (
                    evaluation.get("admissible_nominal_reentry_event_ids") or []
                )
                if str(item).strip()
            },
        )
        cca_goal_recovery_events_after = {
            str(item).strip()
            for item in (
                evaluation.get("cca_admissible_goal_recovery_event_ids_after")
                or []
            )
            if str(item).strip()
        }
        admissible_goal_recovery_events_after = {
            str(item).strip()
            for item in (
                evaluation.get("admissible_goal_recovery_event_ids_after")
                or []
            )
            if str(item).strip()
        }
        newly_enabled = enabled_events_after - current_enabled_events
        disabled_events = current_enabled_events - enabled_events_after
        newly_enabled_nominal_reentry_events = (
            nominal_reentry_events_after - current_nominal_reentry_events
        )
        newly_cca_admissible_goal_recovery_events = (
            cca_goal_recovery_events_after - current_cca_goal_recovery_events
        )
        newly_admissible_goal_recovery_events = (
            admissible_goal_recovery_events_after
            - current_admissible_goal_recovery_events
        )
        future_goal_evaluation_complete = bool(
            evaluation.get("future_goal_evaluation_complete") is True
        )
        realizer_events = _candidate_realizer_event_ids(
            candidate=candidate,
            evaluation=evaluation,
        )
        authored_delta_fields = set(
            _candidate_authored_delta_fields(candidate, evaluation)
        )
        understood_effect_fields = authored_delta_fields & {
            "held_part",
            "part_holder_resource_jid",
            "resource_state.location",
            "part_state.location",
        }
        continuation_enabled_events_before = current_enabled_events - realizer_events
        progressing_candidate = (
            remaining < current_unresolved and not introduced
        ) or (
            remaining == current_unresolved
            and bool(newly_enabled)
        ) or (
            remaining == current_unresolved
            and future_goal_evaluation_complete
            and bool(newly_admissible_goal_recovery_events)
        ) or (
            remaining == current_unresolved
            and future_goal_evaluation_complete
            and bool(newly_cca_admissible_goal_recovery_events)
        ) or (
            remaining == current_unresolved
            and enabled_events_after == current_enabled_events
            and cca_goal_recovery_events_after
            == current_cca_goal_recovery_events
            and bool(newly_enabled_nominal_reentry_events)
        )
        (
            progressing_candidate,
            evaluation["selection_constraint_codes"],
        ) = _classify_candidate_selection_progress(
            candidate=candidate,
            evaluation=evaluation,
            progressing=progressing_candidate,
        )
        evidence = {
            "open_recovery_obligation_ids_before": sorted(current_unresolved),
            "cleared_recovery_obligation_ids": sorted(cleared),
            "open_recovery_obligation_ids_after": sorted(remaining),
            "introduced_recovery_obligation_ids": sorted(introduced),
            "admissible_recovery_enabled_event_ids_before": sorted(
                current_enabled_events
            ),
            "consumed_recovery_event_ids": sorted(realizer_events),
            "continuation_enabled_event_ids_before": sorted(
                continuation_enabled_events_before
            ),
            "admissible_recovery_enabled_event_ids_after": sorted(
                enabled_events_after
            ),
            "newly_enabled_recovery_event_ids": sorted(newly_enabled),
            "disabled_recovery_event_ids": sorted(disabled_events),
            "understood_effect_fields": sorted(
                understood_effect_fields
            ),
            "cca_admissible_goal_recovery_event_ids_before": sorted(
                current_cca_goal_recovery_events
            ),
            "cca_admissible_goal_recovery_event_ids_after": sorted(
                cca_goal_recovery_events_after
            ),
            "newly_cca_admissible_goal_recovery_event_ids": sorted(
                newly_cca_admissible_goal_recovery_events
            ),
            "admissible_goal_recovery_event_ids_before": sorted(
                current_admissible_goal_recovery_events
            ),
            "admissible_goal_recovery_event_ids_after": sorted(
                admissible_goal_recovery_events_after
            ),
            "newly_admissible_goal_recovery_event_ids": sorted(
                newly_admissible_goal_recovery_events
            ),
            "future_goal_evaluation_complete": (
                future_goal_evaluation_complete
            ),
            "admissible_nominal_reentry_event_ids_after": sorted(
                nominal_reentry_events_after
            ),
            "admissible_nominal_reentry_event_ids_before": sorted(
                current_nominal_reentry_events
            ),
            "newly_enabled_nominal_reentry_event_ids": sorted(
                newly_enabled_nominal_reentry_events
            ),
            "safety_dfa_states_before": deepcopy(
                evaluation.get("safety_dfa_states_before") or {}
            ),
            "safety_dfa_states_after": deepcopy(
                evaluation.get("safety_dfa_states_after") or {}
            ),
        }
        evaluation["selection_evidence"] = evidence
        evaluation["admissible_recovery_enabled_event_ids_after"] = sorted(
            enabled_events_after
        )
        evaluation["cca_admissible_goal_recovery_event_ids_after"] = sorted(
            cca_goal_recovery_events_after
        )
        evaluation["admissible_goal_recovery_event_ids_after"] = sorted(
            admissible_goal_recovery_events_after
        )
        evaluation["admissible_nominal_reentry_event_ids_after"] = sorted(
            nominal_reentry_events_after
        )
        evaluation["progressing"] = progressing_candidate
        evaluation["selection_status"] = (
            "eligible" if progressing_candidate else "excluded_no_progress"
        )
        if progressing_candidate:
            progressing.append(evaluation)

    if not progressing:
        return []

    minimum_obligation_count = min(
        len(
            dict(candidate.get("selection_evidence") or {}).get(
                "open_recovery_obligation_ids_after"
            )
            or []
        )
        for candidate in progressing
    )
    obligation_preferred = [
        candidate
        for candidate in progressing
        if len(
            dict(candidate.get("selection_evidence") or {}).get(
                "open_recovery_obligation_ids_after"
            )
            or []
        )
        == minimum_obligation_count
    ]
    for candidate in progressing:
        if candidate not in obligation_preferred:
            candidate["selection_status"] = "dominated_open_obligations"

    goal_enabled_preferred: list[dict[str, Any]] = []
    for candidate in obligation_preferred:
        candidate_events = set(
            candidate.get("admissible_goal_recovery_event_ids_after") or []
        )
        dominators = [
            other
            for other in obligation_preferred
            if other is not candidate
            and set(
                other.get("admissible_goal_recovery_event_ids_after") or []
            )
            > candidate_events
        ]
        candidate["dominated_by_candidate_ids"] = sorted(
            {
                str(other.get("candidate_id") or "")
                for other in dominators
                if str(other.get("candidate_id") or "")
            }
        )
        if dominators:
            candidate["selection_status"] = (
                "dominated_goal_recovery_enabledness"
            )
        else:
            goal_enabled_preferred.append(candidate)

    recovery_preferred: list[dict[str, Any]] = []
    for candidate in goal_enabled_preferred:
        candidate_events = set(
            candidate.get("admissible_recovery_enabled_event_ids_after") or []
        )
        dominators = [
            other
            for other in goal_enabled_preferred
            if other is not candidate
            and set(other.get("admissible_recovery_enabled_event_ids_after") or [])
            > candidate_events
        ]
        candidate["dominated_by_candidate_ids"] = sorted(
            {
                str(other.get("candidate_id") or "")
                for other in dominators
                if str(other.get("candidate_id") or "")
            }
        )
        if dominators:
            candidate["selection_status"] = "dominated_recovery_enabledness"
        else:
            recovery_preferred.append(candidate)

    cca_goal_preferred: list[dict[str, Any]] = []
    for candidate in recovery_preferred:
        candidate_events = set(
            candidate.get("cca_admissible_goal_recovery_event_ids_after") or []
        )
        dominators = [
            other
            for other in recovery_preferred
            if other is not candidate
            and set(
                other.get("cca_admissible_goal_recovery_event_ids_after") or []
            )
            > candidate_events
        ]
        if dominators:
            candidate["selection_status"] = "dominated_cca_goal_admissibility"
            candidate["dominated_by_candidate_ids"] = sorted(
                {
                    str(other.get("candidate_id") or "")
                    for other in dominators
                    if str(other.get("candidate_id") or "")
                }
            )
        else:
            cca_goal_preferred.append(candidate)

    nominal_preferred: list[dict[str, Any]] = []
    for candidate in cca_goal_preferred:
        candidate_events = set(
            candidate.get("admissible_nominal_reentry_event_ids_after") or []
        )
        dominators = [
            other
            for other in cca_goal_preferred
            if other is not candidate
            and set(other.get("admissible_nominal_reentry_event_ids_after") or [])
            > candidate_events
        ]
        if dominators:
            candidate["selection_status"] = "dominated_nominal_reentry_enabledness"
            candidate["dominated_by_candidate_ids"] = sorted(
                {
                    str(other.get("candidate_id") or "")
                    for other in dominators
                    if str(other.get("candidate_id") or "")
                }
            )
        else:
            nominal_preferred.append(candidate)

    representative_candidate_id = min(
        str(candidate.get("candidate_id") or "")
        for candidate in nominal_preferred
        if str(candidate.get("candidate_id") or "")
    )
    representative_rows = [
        candidate
        for candidate in nominal_preferred
        if str(candidate.get("candidate_id") or "") == representative_candidate_id
    ]
    representative = min(
        representative_rows,
        key=lambda candidate: str(
            dict(candidate.get("task") or {}).get("outline_id") or ""
        ),
    )
    symbolically_tied_candidate_ids = sorted(
        {
            str(candidate.get("candidate_id") or "")
            for candidate in nominal_preferred
            if str(candidate.get("candidate_id") or "")
        }
    )
    tie_representative_used = len(nominal_preferred) > 1
    for candidate in nominal_preferred:
        if candidate is representative:
            candidate["selection_status"] = "nondominated"
            candidate["tie_representative_evidence"] = {
                "used": tie_representative_used,
                "rule": "smallest_exact_effect_candidate_id",
                "representative_candidate_id": representative_candidate_id,
                "symbolically_tied_candidate_ids": symbolically_tied_candidate_ids,
                "operational_superiority_claimed": False,
            }
        else:
            candidate["selection_status"] = "stable_representative_not_selected"
            candidate["dominated_by_candidate_ids"] = [representative_candidate_id]
    nondominated = [representative]
    for successor_class_id, rows in candidates_by_successor_class.items():
        if len(rows) <= 1:
            continue
        for row in rows:
            row["equivalent_successor_class_id"] = successor_class_id
    return sorted(str(row.get("candidate_id") or "") for row in nondominated)


def _selection_revision_cca_fingerprints(
    *,
    session_state: dict[str, Any],
    candidate_evaluations: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    safety_rule_fingerprints = sorted(
        {
            str(row.get("safety_rule_fingerprint") or "")
            for row in candidate_evaluations
            if str(row.get("safety_rule_fingerprint") or "")
        }
    )
    if not safety_rule_fingerprints:
        safety_rule_fingerprints = sorted(
            {
                str(value)
                for value in (
                    session_state.get("selection_revision_safety_rule_fingerprints")
                    or [session_state.get("projected_safety_rule_fingerprint")]
                )
                if str(value or "")
            }
        )

    live_dfa_fingerprints = sorted(
        {
            str(row.get("live_safety_dfa_state_fingerprint") or "")
            for row in candidate_evaluations
            if str(row.get("live_safety_dfa_state_fingerprint") or "")
        }
    )
    if not live_dfa_fingerprints:
        live_dfa_fingerprints = sorted(
            {
                str(value)
                for value in (
                    session_state.get("selection_revision_live_dfa_fingerprints")
                    or [session_state.get("live_safety_dfa_state_fingerprint")]
                )
                if str(value or "")
            }
        )
    return safety_rule_fingerprints, live_dfa_fingerprints


def _selection_revision_fingerprint(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    candidate_evaluations: list[dict[str, Any]],
) -> str:
    safety_rule_fingerprints, live_dfa_fingerprints = (
        _selection_revision_cca_fingerprints(
            session_state=session_state,
            candidate_evaluations=candidate_evaluations,
        )
    )
    return recovery_validation_fingerprint(
        {
            "pa_state_fingerprint": _pa_state_fingerprint(
                session_state=session_state,
                prepared_recovery_request=prepared_recovery_request,
            ),
            "safety_rule_fingerprints": safety_rule_fingerprints,
            "live_safety_dfa_state_fingerprints": live_dfa_fingerprints,
            "projected_safety_dfa_states": deepcopy(
                session_state.get("projected_safety_dfa_states") or {}
            ),
        }
    )


def _selection_failure_fingerprint(
    candidate_evaluations: list[dict[str, Any]],
) -> str:
    rows: list[dict[str, Any]] = []
    for evaluation in candidate_evaluations:
        if not isinstance(evaluation, dict):
            continue
        constraint_codes = {
            str(finding.get("constraint_code") or "").strip()
            for finding in (evaluation.get("validation_findings") or [])
            if isinstance(finding, dict)
            and str(finding.get("constraint_code") or "").strip()
        }
        constraint_codes.update(
            str(code).strip()
            for code in (evaluation.get("selection_constraint_codes") or [])
            if str(code).strip()
        )
        rows.append(
            {
                "candidate_id": str(evaluation.get("candidate_id") or "").strip(),
                "valid": bool(evaluation.get("valid")),
                "selection_status": str(
                    evaluation.get("selection_status") or ""
                ).strip(),
                "constraint_codes": sorted(constraint_codes),
            }
        )
    return recovery_validation_fingerprint(rows)


def _register_selection_revision(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    candidate_evaluations: list[dict[str, Any]],
) -> tuple[str, int, int, int, int]:
    revision_fingerprint = _selection_revision_fingerprint(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        candidate_evaluations=candidate_evaluations,
    )
    same_state = (
        str(session_state.get("selection_revision_fingerprint") or "")
        == revision_fingerprint
    )
    if same_state:
        revision_count = int(session_state.get("selection_revision_count") or 0) + 1
    else:
        revision_count = 1
        if (
            str(session_state.get("active_selection_ambiguity_fingerprint") or "")
            != revision_fingerprint
        ):
            session_state["active_selection_ambiguity_feedback"] = {}
            session_state["active_selection_ambiguity_fingerprint"] = ""

    failure_fingerprint = _selection_failure_fingerprint(candidate_evaluations)
    if (
        same_state
        and str(
            session_state.get("selection_repeated_failure_fingerprint") or ""
        )
        == failure_fingerprint
    ):
        repeated_failure_count = int(
            session_state.get("selection_repeated_failure_count") or 0
        ) + 1
    else:
        repeated_failure_count = 1

    safety_rule_fingerprints, live_dfa_fingerprints = (
        _selection_revision_cca_fingerprints(
            session_state=session_state,
            candidate_evaluations=candidate_evaluations,
        )
    )
    session_state["selection_revision_fingerprint"] = revision_fingerprint
    session_state["selection_revision_count"] = revision_count
    session_state["selection_repeated_failure_fingerprint"] = failure_fingerprint
    session_state["selection_repeated_failure_count"] = repeated_failure_count
    session_state["selection_revision_safety_rule_fingerprints"] = deepcopy(
        safety_rule_fingerprints
    )
    session_state["selection_revision_live_dfa_fingerprints"] = deepcopy(
        live_dfa_fingerprints
    )
    revision_limit = max(
        1,
        int(session_state.get("selection_revision_limit") or 6),
    )
    repeated_failure_limit = max(
        1,
        int(session_state.get("selection_repeated_failure_limit") or 3),
    )
    return (
        revision_fingerprint,
        revision_count,
        revision_limit,
        repeated_failure_count,
        repeated_failure_limit,
    )


def _pa_candidate_revision_targets(
    *,
    candidate_evaluations: list[dict[str, Any]],
    candidate_bound: int,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for evaluation in sorted(
        (
            row
            for row in candidate_evaluations
            if isinstance(row, dict) and not bool(row.get("valid"))
        ),
        key=lambda row: int(row.get("candidate_index") or 0),
    ):
        rejected_roles = {
            str(stage.get("validator_role") or "").strip()
            for stage in (evaluation.get("validation_stages") or [])
            if isinstance(stage, dict)
            and str(stage.get("status") or "").strip() == "rejected"
        }
        if rejected_roles != {"PA"}:
            continue
        task = dict(evaluation.get("task") or {})
        expected_end_state = dict(task.get("expected_end_state") or {})
        resource_jid = str(task.get("resource_jid") or "").strip()
        if not resource_jid or not expected_end_state:
            continue
        part_name = str(task.get("part_name") or "").strip()
        constraint_codes = sorted(
            {
                str(finding.get("constraint_code") or "").strip()
                for finding in (evaluation.get("validation_findings") or [])
                if isinstance(finding, dict)
                and str(finding.get("constraint_code") or "").strip()
            }
        )
        if any(
            code in {"expected_start_state_mismatch", "validation_state_stale"}
            for code in constraint_codes
        ):
            continue
        targets.append(
            {
                "candidate_id": str(evaluation.get("candidate_id") or "").strip(),
                "candidate_index": int(evaluation.get("candidate_index") or 0),
                "resource_jid": resource_jid,
                **({"part_name": part_name} if part_name else {}),
                "expected_end_state": deepcopy(expected_end_state),
                "constraint_codes": constraint_codes,
            }
        )
        if len(targets) >= max(1, int(candidate_bound)):
            break
    return targets


def _active_candidate_revision_targets(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
) -> list[dict[str, Any]]:
    targets = [
        deepcopy(row)
        for row in (session_state.get("candidate_revision_targets") or [])
        if isinstance(row, dict)
    ]
    if not targets:
        return []
    current_fingerprint = _pa_state_fingerprint(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    if (
        str(session_state.get("candidate_revision_state_fingerprint") or "")
        != current_fingerprint
    ):
        session_state["candidate_revision_targets"] = []
        session_state["candidate_revision_state_fingerprint"] = ""
        session_state["selection_revision_count"] = 0
        session_state["selection_revision_fingerprint"] = ""
        session_state["selection_repeated_failure_count"] = 0
        session_state["selection_repeated_failure_fingerprint"] = ""
        return []
    return targets


def _candidate_revision_requirement_findings(
    *,
    candidate_sequences: list[dict[str, Any]],
    revision_targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    available_candidate_indexes = set(range(len(candidate_sequences)))
    findings: list[dict[str, Any]] = []
    for target in revision_targets:
        target_resource_jid = str(target.get("resource_jid") or "").strip()
        target_part_name = str(target.get("part_name") or "").strip()
        rejected_end_state = dict(target.get("expected_end_state") or {})
        matched_candidate_index: int | None = None
        for candidate_index in sorted(available_candidate_indexes):
            surface_events = list(
                candidate_sequences[candidate_index].get("surface_events") or []
            )
            if len(surface_events) != 1 or not isinstance(surface_events[0], dict):
                continue
            candidate_task = dict(surface_events[0])
            if (
                str(candidate_task.get("resource_jid") or "").strip()
                != target_resource_jid
                or str(candidate_task.get("part_name") or "").strip()
                != target_part_name
            ):
                continue
            if dict(candidate_task.get("expected_end_state") or {}) == rejected_end_state:
                continue
            matched_candidate_index = candidate_index
            break
        if matched_candidate_index is not None:
            available_candidate_indexes.remove(matched_candidate_index)
            continue
        findings.append(
            _shared._candidate_schema_finding(
                task={
                    "outline_id": str(target.get("candidate_id") or ""),
                    "resource_jid": target_resource_jid,
                    **({"part_name": target_part_name} if target_part_name else {}),
                },
                reason=(
                    "The response omitted a materially revised candidate for a "
                    "prior PA-rejected resource/part identity."
                ),
                evidence={
                    "candidate_id": str(target.get("candidate_id") or ""),
                    "resource_jid": target_resource_jid,
                    **({"part_name": target_part_name} if target_part_name else {}),
                    "rejected_expected_end_state": deepcopy(rejected_end_state),
                    "constraint_codes": deepcopy(
                        target.get("constraint_codes") or []
                    ),
                },
            )
        )
    return findings


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


def _validator_jids(planner: Any, resource_jid: str = "") -> tuple[str, str, str]:
    product_agent = getattr(planner, "product_agent", None)
    product_jid = str(getattr(product_agent, "jid", "") or "").strip()
    cca_jid = str(getattr(product_agent, "cca_jid", "") or "").strip()
    return product_jid, str(resource_jid or "").strip(), cca_jid


def _unavailable_validation_finding(
    *,
    task: dict[str, Any],
    validation_category: str,
    validator_role: str,
    constraint_code: str,
    reason: str,
) -> dict[str, Any]:
    owner = "resource" if validator_role == "RA" else "cca"
    return _shared.annotate_validation_finding(
        {
            "task_id": str(task.get("outline_id") or "").strip(),
            "resource_jid": _shared._task_resource_jid(task) or None,
            "part_name": _shared._task_part_name(task) or None,
            "validation_category": validation_category,
            "constraint_owner": owner,
            "constraint_family": (
                "resource_feasibility" if validator_role == "RA" else "safety"
            ),
            "constraint_code": constraint_code,
            "reason": reason,
            "retriable": True,
        }
    )


def _skipped_validation_stage(
    *,
    category: str,
    role: str,
    jid: str,
    mocked: bool = False,
) -> dict[str, Any]:
    return recovery_validation_stage(
        validation_category=category,
        validator_role=role,
        validator_jid=jid,
        status="skipped",
        findings=[],
        mocked=mocked,
    )


def _append_pa_validation_stages(
    *,
    stages: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    product_jid: str,
    state_fingerprint: str,
) -> None:
    syntax_findings = [
        deepcopy(row)
        for row in findings
        if str(row.get("validation_category") or "").strip()
        == SYNTAX_AND_GROUNDING_VALIDATION
    ]
    stages.append(
        recovery_validation_stage(
            validation_category=SYNTAX_AND_GROUNDING_VALIDATION,
            validator_role="PA",
            validator_jid=product_jid,
            status="rejected" if syntax_findings else "passed",
            findings=syntax_findings,
            state_fingerprint=state_fingerprint,
        )
    )


async def _validate_candidate_sequence(  # noqa: C901, PLR0912, PLR0915
    *,
    candidate: dict[str, Any],
    sequence_index: int,
    action_horizon: str,
    action_horizon_k: int,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
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

    product_agent = getattr(planner, "product_agent", None)
    product_jid, _, cca_jid = _validator_jids(planner)
    state_fingerprint = _pa_state_fingerprint(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    validation_stages: list[dict[str, Any]] = []
    evaluation["pa_state_fingerprint"] = state_fingerprint
    recovery_session_id = str(
        session_state.get("recovery_session_id")
        or session_state.get("session_id")
        or dict(prepared_recovery_request.get("recovery_session") or {}).get(
            "session_id"
        )
        or prepared_recovery_request.get("recovery_session_id")
        or ""
    ).strip()
    turn_index = int(
        session_state.get("turn_index")
        or len(list(session_state.get("turns") or [])) + 1
    )

    def reject_before_agent_validation(
        *,
        findings: list[dict[str, Any]],
        task: dict[str, Any],
        failed_event_index: int = 0,
    ) -> dict[str, Any]:
        annotated = [_shared.annotate_validation_finding(row) for row in findings]
        _append_pa_validation_stages(
            stages=validation_stages,
            findings=annotated,
            product_jid=product_jid,
            state_fingerprint=state_fingerprint,
        )
        resource_jid = _shared._task_resource_jid(task)
        validation_stages.append(
            _skipped_validation_stage(
                category=TRANSITION_FEASIBILITY,
                role="RA",
                jid=resource_jid,
            )
        )
        validation_stages.append(
            _skipped_validation_stage(
                category=PHYSICAL_FEASIBILITY,
                role="RA",
                jid=resource_jid,
            )
        )
        validation_stages.append(
            _skipped_validation_stage(category=SAFETY, role="CCA", jid=cca_jid)
        )
        evaluation["valid"] = False
        evaluation["validation_findings"] = deepcopy(annotated)
        evaluation["validation_stages"] = deepcopy(validation_stages)
        evaluation["failed_event_index"] = failed_event_index
        return evaluation

    if not surface_events:
        return reject_before_agent_validation(
            findings=[
                _shared._candidate_schema_finding(
                task={},
                reason="candidate trace must include at least one event",
                evidence={"candidate_index": candidate_index},
                )
            ],
            task={},
        )
    if action_horizon == "1" and len(surface_events) != 1:
        return reject_before_agent_validation(
            findings=[
                _shared._candidate_schema_finding(
                task=surface_events[0],
                reason="action_horizon=1 candidate must include exactly one event",
                evidence={"candidate_index": candidate_index, "event_count": len(surface_events)},
                )
            ],
            task=surface_events[0],
        )
    if action_horizon == "k" and len(surface_events) > action_horizon_k:
        return reject_before_agent_validation(
            findings=[
                _shared._candidate_schema_finding(
                task=surface_events[0],
                reason="action_horizon=k candidate exceeds action_horizon_k",
                evidence={
                    "candidate_index": candidate_index,
                    "event_count": len(surface_events),
                    "action_horizon_k": action_horizon_k,
                },
                )
            ],
            task=surface_events[0],
        )

    working_session_state = deepcopy(session_state)
    validated_events: list[dict[str, Any]] = []
    committed_events: list[dict[str, Any]] = []
    grounded_actions: list[dict[str, Any]] = []
    safety_dfa_states_before: dict[str, str] = {}
    safety_dfa_states_after: dict[str, str] = deepcopy(
        working_session_state.get("projected_safety_dfa_states") or {}
    )
    admissible_nominal_reentry_event_ids_before: list[str] = []
    admissible_nominal_reentry_event_ids_after: list[str] = []
    cca_admissible_goal_recovery_event_ids_before: list[str] = []
    cca_admissible_goal_recovery_event_ids_after: list[str] = []

    for event_index, surface_event in enumerate(surface_events):
        working_task = deepcopy(surface_event)
        evaluation["task"] = deepcopy(working_task)
        validated_task, schema_findings = _shared._derive_candidate_outline_task(
            candidate_task=working_task,
            session_state=working_session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        if schema_findings or not validated_task:
            return reject_before_agent_validation(
                findings=list(schema_findings or []),
                task=working_task,
                failed_event_index=event_index,
            )
        evaluation["validated_task"] = deepcopy(validated_task)

        findings, grounded_action = _shared._validate_single_outline_task(
            planner=planner,
            task=dict(validated_task or {}),
            session_state=working_session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        pa_findings = [
            _shared.annotate_validation_finding(row)
            for row in findings
            if isinstance(row, dict)
        ]
        _append_pa_validation_stages(
            stages=validation_stages,
            findings=pa_findings,
            product_jid=product_jid,
            state_fingerprint=state_fingerprint,
        )
        if findings:
            resource_jid = _shared._task_resource_jid(validated_task)
            validation_stages.append(
                _skipped_validation_stage(
                    category=TRANSITION_FEASIBILITY,
                    role="RA",
                    jid=resource_jid,
                )
            )
            validation_stages.append(
                _skipped_validation_stage(
                    category=PHYSICAL_FEASIBILITY,
                    role="RA",
                    jid=resource_jid,
                )
            )
            validation_stages.append(
                _skipped_validation_stage(category=SAFETY, role="CCA", jid=cca_jid)
            )
            evaluation["valid"] = False
            evaluation["validation_findings"] = deepcopy(pa_findings)
            evaluation["validation_stages"] = deepcopy(validation_stages)
            evaluation["failed_event_index"] = event_index
            return evaluation
        if grounded_action:
            grounded_actions.append(deepcopy(grounded_action))

        resource_jid = _shared._task_resource_jid(validated_task)
        physical_input = build_recovery_physical_validation_input(
            task=validated_task,
            grounded_action=dict(grounded_action or {}),
            session_state=working_session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        physical_request = {
            "recovery_session_id": recovery_session_id,
            "turn_index": turn_index,
            "candidate_index": candidate_index,
            "state_fingerprint": state_fingerprint,
            "candidate_task": deepcopy(validated_task),
            "grounded_action": deepcopy(grounded_action or {}),
            "candidates": [
                {
                    "candidate_index": candidate_index,
                    "task": deepcopy(validated_task),
                    "physical_input": deepcopy(physical_input),
                }
            ],
        }
        ra_started_at = time.perf_counter()
        try:
            if not callable(
                getattr(product_agent, "request_recovery_outline_physical_validation", None)
            ):
                raise RuntimeError(
                    "ProductAgent physical-validation message transport is unavailable"
                )
            ra_reply = await product_agent.request_recovery_outline_physical_validation(
                resource_jid=resource_jid,
                payload=physical_request,
                timeout_s=10.0,
            )
            ra_result = next(
                (
                    dict(row)
                    for row in (ra_reply.get("results") or [])
                    if isinstance(row, dict)
                    and int(row.get("candidate_index") or 0) == candidate_index
                ),
                None,
            )
            if ra_result is None:
                raise RuntimeError("ResourceAgent reply omitted the candidate result")
        except Exception as exc:  # noqa: BLE001 - validator transport must fail closed
            ra_finding = _unavailable_validation_finding(
                task=validated_task,
                validation_category=TRANSITION_FEASIBILITY,
                validator_role="RA",
                constraint_code="resource_validation_unavailable",
                reason=str(exc),
            )
            validation_stages.append(
                recovery_validation_stage(
                    validation_category=TRANSITION_FEASIBILITY,
                    validator_role="RA",
                    validator_jid=resource_jid,
                    status="unavailable",
                    findings=[ra_finding],
                    latency_ms=(time.perf_counter() - ra_started_at) * 1000.0,
                    state_fingerprint=state_fingerprint,
                )
            )
            validation_stages.append(
                _skipped_validation_stage(
                    category=PHYSICAL_FEASIBILITY,
                    role="RA",
                    jid=resource_jid,
                )
            )
            validation_stages.append(
                _skipped_validation_stage(category=SAFETY, role="CCA", jid=cca_jid)
            )
            evaluation["valid"] = False
            evaluation["validation_findings"] = [ra_finding]
            evaluation["validation_stages"] = deepcopy(validation_stages)
            evaluation["failed_event_index"] = event_index
            return evaluation

        transition_findings = [
            _shared.annotate_validation_finding(row)
            for row in (ra_result.get("transition_findings") or [])
            if isinstance(row, dict)
        ]
        validation_stages.append(
            recovery_validation_stage(
                validation_category=TRANSITION_FEASIBILITY,
                validator_role="RA",
                validator_jid=str(ra_reply.get("validator_jid") or resource_jid),
                status=(
                    "passed"
                    if bool(
                        dict(
                            ra_result.get("transition_feasibility") or {}
                        ).get("allowed")
                    )
                    else "rejected"
                ),
                findings=transition_findings,
                request_id=str(ra_reply.get("request_id") or ""),
                latency_ms=ra_reply.get("latency_ms"),
                state_fingerprint=state_fingerprint,
                mocked=bool(ra_reply.get("mocked")),
            )
        )
        if transition_findings or not bool(
            dict(ra_result.get("transition_feasibility") or {}).get(
                "allowed"
            )
        ):
            validation_stages.append(
                _skipped_validation_stage(
                    category=PHYSICAL_FEASIBILITY,
                    role="RA",
                    jid=resource_jid,
                    mocked=bool(ra_reply.get("mocked")),
                )
            )
            validation_stages.append(
                _skipped_validation_stage(
                    category=SAFETY,
                    role="CCA",
                    jid=cca_jid,
                    mocked=bool(ra_reply.get("mocked")),
                )
            )
            evaluation["valid"] = False
            evaluation["validation_findings"] = deepcopy(transition_findings)
            evaluation["validation_stages"] = deepcopy(validation_stages)
            evaluation["failed_event_index"] = event_index
            return evaluation

        physical_findings = [
            _shared.annotate_validation_finding(row)
            for row in (ra_result.get("physical_findings") or [])
            if isinstance(row, dict)
        ]
        validation_stages.append(
            recovery_validation_stage(
                validation_category=PHYSICAL_FEASIBILITY,
                validator_role="RA",
                validator_jid=str(ra_reply.get("validator_jid") or resource_jid),
                status=(
                    "passed"
                    if bool(
                        dict(
                            ra_result.get("physical_feasibility") or {}
                        ).get("allowed")
                    )
                    else "rejected"
                ),
                findings=physical_findings,
                request_id=str(ra_reply.get("request_id") or ""),
                latency_ms=ra_reply.get("latency_ms"),
                state_fingerprint=state_fingerprint,
                mocked=bool(ra_reply.get("mocked")),
            )
        )
        if physical_findings or not bool(
            dict(ra_result.get("physical_feasibility") or {}).get("allowed")
        ):
            validation_stages.append(
                _skipped_validation_stage(
                    category=SAFETY,
                    role="CCA",
                    jid=cca_jid,
                    mocked=bool(ra_reply.get("mocked")),
                )
            )
            evaluation["valid"] = False
            evaluation["validation_findings"] = deepcopy(physical_findings)
            evaluation["validation_stages"] = deepcopy(validation_stages)
            evaluation["failed_event_index"] = event_index
            return evaluation

        transition_feasibility = dict(
            ra_result.get("transition_feasibility") or {}
        )
        calculated_successor = deepcopy(
            dict(
                transition_feasibility.get("calculated_successor")
                or {}
            )
        )
        resource_validated_task = deepcopy(validated_task)
        resource_validated_task["expected_end_state"] = deepcopy(
            calculated_successor
        )
        atomic_transitions = [
            deepcopy(transition_row)
            for transition_row in (
                ra_result.get("_cca_atomic_transitions") or []
            )
            if isinstance(transition_row, dict)
        ]
        safety_input = build_recovery_safety_validation_input(
            task=resource_validated_task,
            session_state=working_session_state,
            prepared_recovery_request=prepared_recovery_request,
            calculated_successor=calculated_successor,
        )
        nominal_reentry_events = _nominal_reentry_event_rows(
            session_state=working_session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        safety_input.setdefault("llm_input", {})["nominal_reentry_events"] = [
            {
                "event_id": str(row.get("event_id") or ""),
                "task": deepcopy(row.get("safety_task") or {}),
                "signature": deepcopy(row.get("safety_signature") or {}),
            }
            for row in nominal_reentry_events
            if str(row.get("event_id") or "")
        ]
        safety_request = {
            "recovery_session_id": recovery_session_id,
            "turn_index": turn_index,
            "candidate_index": candidate_index,
            "state_fingerprint": state_fingerprint,
            "candidate_task": deepcopy(resource_validated_task),
            "grounded_action": deepcopy(grounded_action or {}),
            "candidates": [
                {
                    "candidate_index": candidate_index,
                    "task": deepcopy(resource_validated_task),
                    "safety_input": safety_input,
                    "_cca_atomic_transitions": atomic_transitions,
                    **(
                        {
                            "safety_dfa_states_before": deepcopy(
                                working_session_state.get(
                                    "projected_safety_dfa_states"
                                )
                                or {}
                            )
                        }
                        if working_session_state.get("projected_safety_dfa_states")
                        else {}
                    ),
                    **(
                        {
                            "safety_rule_fingerprint": str(
                                working_session_state.get(
                                    "projected_safety_rule_fingerprint"
                                )
                                or ""
                            ).strip()
                        }
                        if str(
                            working_session_state.get(
                                "projected_safety_rule_fingerprint"
                            )
                            or ""
                        ).strip()
                        else {}
                    ),
                    **(
                        {
                            "live_safety_dfa_state_fingerprint": str(
                                working_session_state.get(
                                    "live_safety_dfa_state_fingerprint"
                                )
                                or ""
                            ).strip()
                        }
                        if str(
                            working_session_state.get(
                                "live_safety_dfa_state_fingerprint"
                            )
                            or ""
                        ).strip()
                        else {}
                    ),
                }
            ],
        }
        cca_started_at = time.perf_counter()
        try:
            if not callable(
                getattr(product_agent, "request_recovery_outline_safety_validation", None)
            ):
                raise RuntimeError(
                    "ProductAgent safety-validation message transport is unavailable"
                )
            cca_reply = await product_agent.request_recovery_outline_safety_validation(
                payload=safety_request,
                timeout_s=10.0,
            )
            cca_result = next(
                (
                    dict(row)
                    for row in (cca_reply.get("results") or [])
                    if isinstance(row, dict)
                    and int(row.get("candidate_index") or 0) == candidate_index
                ),
                None,
            )
            if cca_result is None:
                raise RuntimeError("CCA reply omitted the candidate result")
            reply_rule_fingerprint = str(
                cca_reply.get("safety_rule_fingerprint") or ""
            ).strip()
            expected_rule_fingerprint = str(
                working_session_state.get("projected_safety_rule_fingerprint") or ""
            ).strip()
            if (
                expected_rule_fingerprint
                and reply_rule_fingerprint != expected_rule_fingerprint
            ):
                raise RuntimeError("CCA safety-rule fingerprint changed")
            reply_live_state_fingerprint = str(
                cca_reply.get("live_safety_dfa_state_fingerprint") or ""
            ).strip()
            expected_live_state_fingerprint = str(
                working_session_state.get("live_safety_dfa_state_fingerprint") or ""
            ).strip()
            if (
                expected_live_state_fingerprint
                and reply_live_state_fingerprint != expected_live_state_fingerprint
            ):
                raise RuntimeError("CCA live safety DFA state changed")
        except Exception as exc:  # noqa: BLE001 - validator transport must fail closed
            cca_finding = _unavailable_validation_finding(
                task=validated_task,
                validation_category=SAFETY,
                validator_role="CCA",
                constraint_code="safety_validation_unavailable",
                reason=str(exc),
            )
            validation_stages.append(
                recovery_validation_stage(
                    validation_category=SAFETY,
                    validator_role="CCA",
                    validator_jid=cca_jid,
                    status="unavailable",
                    findings=[cca_finding],
                    latency_ms=(time.perf_counter() - cca_started_at) * 1000.0,
                    state_fingerprint=state_fingerprint,
                )
            )
            evaluation["valid"] = False
            evaluation["validation_findings"] = [cca_finding]
            evaluation["validation_stages"] = deepcopy(validation_stages)
            evaluation["failed_event_index"] = event_index
            return evaluation

        cca_findings = [
            _shared.annotate_validation_finding(row)
            for row in (cca_result.get("findings") or [])
            if isinstance(row, dict)
        ]
        validation_stages.append(
            recovery_validation_stage(
                validation_category=SAFETY,
                validator_role="CCA",
                validator_jid=str(cca_reply.get("validator_jid") or cca_jid),
                status="passed" if bool(cca_result.get("is_safe")) else "rejected",
                findings=cca_findings,
                request_id=str(cca_reply.get("request_id") or ""),
                latency_ms=cca_reply.get("latency_ms"),
                state_fingerprint=state_fingerprint,
                mocked=bool(cca_reply.get("mocked")),
            )
        )
        if cca_findings or not bool(cca_result.get("is_safe")):
            evaluation["valid"] = False
            evaluation["validation_findings"] = deepcopy(cca_findings)
            evaluation["validation_stages"] = deepcopy(validation_stages)
            evaluation["failed_event_index"] = event_index
            return evaluation
        event_safety_dfa_states_before = {
            str(rule_id): str(state)
            for rule_id, state in dict(
                cca_result.get("safety_dfa_states_before") or {}
            ).items()
        }
        event_safety_dfa_states_after = {
            str(rule_id): str(state)
            for rule_id, state in dict(
                cca_result.get("safety_dfa_states_after") or {}
            ).items()
        }
        if event_index == 0:
            safety_dfa_states_before = deepcopy(event_safety_dfa_states_before)
        safety_dfa_states_after = deepcopy(event_safety_dfa_states_after)
        working_session_state["projected_safety_dfa_states"] = deepcopy(
            event_safety_dfa_states_after
        )
        working_session_state["projected_safety_rule_fingerprint"] = str(
            cca_reply.get("safety_rule_fingerprint") or ""
        ).strip()
        working_session_state["live_safety_dfa_state_fingerprint"] = str(
            cca_reply.get("live_safety_dfa_state_fingerprint") or ""
        ).strip()
        event_cca_admissible_goal_recovery_event_ids_before = [
            str(item).strip()
            for item in (
                cca_result.get("cca_admissible_goal_recovery_event_ids_before")
                or []
            )
            if str(item).strip()
        ]
        event_cca_admissible_goal_recovery_event_ids_after = [
            str(item).strip()
            for item in (
                cca_result.get("cca_admissible_goal_recovery_event_ids_after")
                or cca_result.get("cca_admissible_goal_recovery_event_ids")
                or []
            )
            if str(item).strip()
        ]
        if event_index == 0:
            cca_admissible_goal_recovery_event_ids_before = deepcopy(
                event_cca_admissible_goal_recovery_event_ids_before
            )
        cca_admissible_goal_recovery_event_ids_after = deepcopy(
            event_cca_admissible_goal_recovery_event_ids_after
        )
        event_admissible_nominal_reentry_event_ids_before = [
            str(item).strip()
            for item in (
                cca_result.get("admissible_nominal_reentry_event_ids_before") or []
            )
            if str(item).strip()
        ]
        event_admissible_nominal_reentry_event_ids_after = [
            str(item).strip()
            for item in (
                cca_result.get("admissible_nominal_reentry_event_ids_after")
                or cca_result.get("admissible_nominal_reentry_event_ids")
                or []
            )
            if str(item).strip()
        ]
        if event_index == 0:
            admissible_nominal_reentry_event_ids_before = deepcopy(
                event_admissible_nominal_reentry_event_ids_before
            )
        admissible_nominal_reentry_event_ids_after = deepcopy(
            event_admissible_nominal_reentry_event_ids_after
        )
        evaluation["admissible_nominal_reentry_event_ids_before"] = deepcopy(
            admissible_nominal_reentry_event_ids_before
        )
        evaluation["admissible_nominal_reentry_event_ids_after"] = deepcopy(
            admissible_nominal_reentry_event_ids_after
        )
        evaluation["admissible_nominal_reentry_event_ids"] = deepcopy(
            admissible_nominal_reentry_event_ids_after
        )

        committed_event = _shared._commit_selected_candidate_task(
            task=dict(resource_validated_task or {}),
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
        validated_events.append(deepcopy(resource_validated_task))
        committed_events.append(deepcopy(committed_event))

    remaining_findings, remaining_conditions = _shared._remaining_blocked_issue_counts(
        session_state=working_session_state,
        prepared_recovery_request=prepared_recovery_request,
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
    evaluation["validation_stages"] = deepcopy(validation_stages)
    evaluation["safety_dfa_states_before"] = deepcopy(safety_dfa_states_before)
    evaluation["safety_dfa_states_after"] = deepcopy(safety_dfa_states_after)
    evaluation["safety_rule_fingerprint"] = str(
        working_session_state.get("projected_safety_rule_fingerprint") or ""
    ).strip()
    evaluation["live_safety_dfa_state_fingerprint"] = str(
        working_session_state.get("live_safety_dfa_state_fingerprint") or ""
    ).strip()
    evaluation["admissible_nominal_reentry_event_ids_before"] = deepcopy(
        admissible_nominal_reentry_event_ids_before
    )
    evaluation["admissible_nominal_reentry_event_ids_after"] = deepcopy(
        admissible_nominal_reentry_event_ids_after
    )
    evaluation["cca_admissible_goal_recovery_event_ids_before"] = deepcopy(
        cca_admissible_goal_recovery_event_ids_before
    )
    evaluation["cca_admissible_goal_recovery_event_ids_after"] = deepcopy(
        cca_admissible_goal_recovery_event_ids_after
    )
    projected_resources, projected_parts = _projected_outline_validation_context(
        session_state=working_session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    evaluation["projected_symbolic_resources"] = {
        resource_jid: deepcopy(projected_resources[resource_jid])
        for resource_jid in sorted(projected_resources)
    }
    evaluation["projected_symbolic_parts"] = {
        part_name: deepcopy(projected_parts[part_name])
        for part_name in sorted(projected_parts)
    }
    evaluation["remaining_blocked_issues"] = remaining_blocked_issues
    evaluation["resource_switch_count"] = resource_switches
    if validated_events:
        evaluation["task"] = deepcopy(validated_events[0])
        evaluation["validated_task"] = deepcopy(validated_events[0])
    if grounded_actions:
        evaluation["grounded_action"] = deepcopy(grounded_actions[0])
    return evaluation


async def _handle_outline_incremental_candidates_validated(  # noqa: C901, PLR0912, PLR0915
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
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
        prepared_recovery_request,
    )

    candidate_sequences = _candidate_sequences_from_response(
        parsed_response,
        action_horizon=action_horizon,
    )
    if recovery_selection_mode == "neurosymbolic" and action_horizon != "1":
        turn_entry["error"] = "neurosymbolic recovery selection requires action_horizon=1"
        return "need_revision", turn_entry
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

    revision_targets = (
        _active_candidate_revision_targets(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        if recovery_selection_mode == "neurosymbolic"
        else []
    )
    revision_requirement_findings = _candidate_revision_requirement_findings(
        candidate_sequences=candidate_sequences,
        revision_targets=revision_targets,
    )
    if revision_targets:
        turn_entry["candidate_revision_targets"] = deepcopy(revision_targets)

    llm_selected_candidate_index = _llm_selected_candidate_index(parsed_response)
    if (
        recovery_selection_mode == "neurosymbolic"
        and "selected_candidate_index" in parsed_response
    ):
        finding = _shared._candidate_schema_finding(
            task={},
            reason=(
                "neurosymbolic outline responses must not include "
                "selected_candidate_index"
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
    if recovery_selection_mode == "pure_llm" and llm_selected_candidate_index is not None and not (
        0 <= llm_selected_candidate_index < len(candidate_sequences)
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

    if revision_requirement_findings:
        product_jid, _, cca_jid = _validator_jids(planner)
        state_fingerprint = _pa_state_fingerprint(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        candidate_evaluations = []
        for candidate in candidate_sequences:
            candidate_index = int(candidate.get("candidate_index") or 0)
            surface_events = [
                deepcopy(row)
                for row in (candidate.get("surface_events") or [])
                if isinstance(row, dict)
            ]
            task = deepcopy(surface_events[0]) if surface_events else {}
            validation_stages: list[dict[str, Any]] = []
            _append_pa_validation_stages(
                stages=validation_stages,
                findings=revision_requirement_findings,
                product_jid=product_jid,
                state_fingerprint=state_fingerprint,
            )
            validation_stages.append(
                _skipped_validation_stage(
                    category=PHYSICAL_FEASIBILITY,
                    role="RA",
                    jid=_shared._task_resource_jid(task),
                )
            )
            validation_stages.append(
                _skipped_validation_stage(
                    category=SAFETY,
                    role="CCA",
                    jid=cca_jid,
                )
            )
            candidate_evaluations.append(
                {
                    "candidate_index": candidate_index,
                    "valid": False,
                    "task": task,
                    "surface_events": surface_events,
                    "validation_findings": deepcopy(
                        revision_requirement_findings
                    ),
                    "validation_stages": validation_stages,
                    "pa_state_fingerprint": state_fingerprint,
                }
            )
    else:
        candidate_evaluations = list(
            await asyncio.gather(
                *[
                    _validate_candidate_sequence(
                        candidate=dict(candidate),
                        sequence_index=sequence_index,
                        action_horizon=action_horizon,
                        action_horizon_k=action_horizon_k,
                        session_state=session_state,
                        prepared_recovery_request=prepared_recovery_request,
                        planner=planner,
                    )
                    for candidate in candidate_sequences
                ]
            )
        )

    if recovery_selection_mode == "neurosymbolic":
        await _populate_agent_filtered_enabledness(
            candidate_evaluations=candidate_evaluations,
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )

    turn_entry["candidate_evaluations"] = deepcopy(candidate_evaluations)
    _shared._promote_durable_candidate_rejections(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        candidate_evaluations=candidate_evaluations,
    )

    selected_candidate_index: int | None
    nondominated_candidate_ids: list[str] = []
    if recovery_selection_mode == "neurosymbolic":
        nondominated_candidate_ids = _apply_neurosymbolic_comparison(
            candidate_sequences=candidate_sequences,
            candidate_evaluations=candidate_evaluations,
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        turn_entry["candidate_evaluations"] = deepcopy(candidate_evaluations)
        turn_entry["nondominated_candidate_ids"] = deepcopy(
            nondominated_candidate_ids
        )
        if len(nondominated_candidate_ids) != 1:
            valid_evaluations = [
                row
                for row in candidate_evaluations
                if isinstance(row, dict) and bool(row.get("valid"))
            ]
            selection_status = (
                "selection_ambiguous"
                if len(nondominated_candidate_ids) > 1
                else "need_revision"
            )
            (
                revision_fingerprint,
                revision_count,
                revision_limit,
                repeated_failure_count,
                repeated_failure_limit,
            ) = (
                _register_selection_revision(
                    session_state=session_state,
                    prepared_recovery_request=prepared_recovery_request,
                    candidate_evaluations=candidate_evaluations,
                )
            )
            active_ambiguity_finding = dict(
                session_state.get("active_selection_ambiguity_feedback") or {}
            )
            if (
                str(session_state.get("active_selection_ambiguity_fingerprint") or "")
                != revision_fingerprint
            ):
                active_ambiguity_finding = {}
                session_state["active_selection_ambiguity_feedback"] = {}
                session_state["active_selection_ambiguity_fingerprint"] = ""
            if (
                revision_count >= revision_limit
                or repeated_failure_count >= repeated_failure_limit
            ):
                selection_status = "selection_unresolved"
            comparison_evidence = [
                {
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "selection_status": str(row.get("selection_status") or ""),
                    "selection_evidence": deepcopy(
                        row.get("selection_evidence") or {}
                    ),
                    "dominated_by_candidate_ids": deepcopy(
                        row.get("dominated_by_candidate_ids") or []
                    ),
                }
                for row in candidate_evaluations
                if isinstance(row, dict)
            ]
            model_finding: dict[str, Any] | None = None
            if selection_status == "selection_unresolved":
                model_finding = _shared.annotate_validation_finding(
                    {
                        "validation_category": "model_based_selection",
                        "constraint_owner": "product",
                        "constraint_family": "model_based_selection",
                        "constraint_code": "selection_unresolved",
                        "reason": (
                            "The unchanged plant and supervisor state reached an "
                            "unresolved revision safeguard."
                        ),
                        "evidence": {
                            "selection_revision_count": revision_count,
                            "selection_revision_limit": revision_limit,
                            "selection_repeated_failure_count": (
                                repeated_failure_count
                            ),
                            "selection_repeated_failure_limit": (
                                repeated_failure_limit
                            ),
                            "plant_supervisor_fingerprint": revision_fingerprint,
                        },
                    }
                )
            elif len(nondominated_candidate_ids) > 1:
                model_finding = _shared.annotate_validation_finding(
                    {
                        "validation_category": "model_based_selection",
                        "constraint_owner": "product",
                        "constraint_family": "model_based_selection",
                        "constraint_code": "selection_ambiguous",
                        "reason": (
                            "Multiple equivalent or incomparable nondominated "
                            "candidates remain."
                        ),
                        "evidence": {
                            "nondominated_candidate_ids": deepcopy(
                                nondominated_candidate_ids
                            ),
                            "candidate_comparison": comparison_evidence,
                        },
                    }
                )
                session_state["active_selection_ambiguity_feedback"] = deepcopy(
                    model_finding
                )
                session_state["active_selection_ambiguity_fingerprint"] = (
                    revision_fingerprint
                )
                active_ambiguity_finding = deepcopy(model_finding)
            elif valid_evaluations:
                no_progress_comparison_evidence = []
                for row in candidate_evaluations:
                    if not isinstance(row, dict):
                        continue
                    task = dict(row.get("task") or {})
                    part_name = str(task.get("part_name") or "").strip()
                    no_progress_comparison_evidence.append(
                        {
                            "candidate_id": str(row.get("candidate_id") or ""),
                            "selection_status": str(
                                row.get("selection_status") or ""
                            ),
                            "selection_constraint_codes": deepcopy(
                                row.get("selection_constraint_codes") or []
                            ),
                            "resource_jid": str(
                                task.get("resource_jid") or ""
                            ).strip(),
                            **({"part_name": part_name} if part_name else {}),
                            "expected_end_state": deepcopy(
                                task.get("expected_end_state") or {}
                            ),
                            "selection_evidence": deepcopy(
                                row.get("selection_evidence") or {}
                            ),
                            "dominated_by_candidate_ids": deepcopy(
                                row.get("dominated_by_candidate_ids") or []
                            ),
                        }
                    )
                model_finding = _shared.annotate_validation_finding(
                    {
                        "validation_category": "model_based_selection",
                        "constraint_owner": "product",
                        "constraint_family": "model_based_selection",
                        "constraint_code": "no_progressing_candidate",
                        "reason": (
                            "Validated candidates neither reduce open recovery "
                            "obligations, enable a previously inadmissible "
                            "recovery-relevant event, make a new goal-relevant "
                            "future recovery event CCA-admissible, nor enable a "
                            "previously inadmissible exact nominal-reentry event."
                        ),
                        "evidence": {
                            "candidate_comparison": no_progress_comparison_evidence
                        },
                    }
                )

            if revision_requirement_findings:
                next_revision_targets = deepcopy(revision_targets)
            else:
                next_revision_targets = _pa_candidate_revision_targets(
                    candidate_evaluations=candidate_evaluations,
                    candidate_bound=candidate_bound,
                )
            session_state["candidate_revision_targets"] = deepcopy(
                next_revision_targets
            )
            session_state["candidate_revision_state_fingerprint"] = (
                _pa_state_fingerprint(
                    session_state=session_state,
                    prepared_recovery_request=prepared_recovery_request,
                )
                if next_revision_targets
                else ""
            )
            turn_entry["candidate_revision_targets"] = deepcopy(
                next_revision_targets
            )

            feedback_rows = _shared._merge_applicable_candidate_feedback(
                session_state=session_state,
                prepared_recovery_request=prepared_recovery_request,
                current_feedback_rows=_shared._candidate_feedback_rows(
                    candidate_evaluations
                ),
                prune_current=False,
            )
            transition_findings = [
                deepcopy(finding)
                for evaluation in candidate_evaluations
                if isinstance(evaluation, dict) and not evaluation.get("valid")
                for finding in (evaluation.get("validation_findings") or [])
                if isinstance(finding, dict)
            ]
            if model_finding is not None:
                feedback_rows.append(deepcopy(model_finding))
                transition_findings.append(deepcopy(model_finding))
            if (
                selection_status != "selection_unresolved"
                and active_ambiguity_finding
                and str(active_ambiguity_finding.get("constraint_code") or "")
                == "selection_ambiguous"
                and not any(
                    str(row.get("constraint_code") or "")
                    == "selection_ambiguous"
                    for row in feedback_rows
                    if isinstance(row, dict)
                )
            ):
                feedback_rows.append(deepcopy(active_ambiguity_finding))
                transition_findings.append(deepcopy(active_ambiguity_finding))
            session_state["candidate_rejection_feedback"] = deepcopy(
                feedback_rows
            )
            turn_entry["candidate_rejection_feedback"] = deepcopy(feedback_rows)
            turn_entry["selected_by"] = "neurosymbolic"
            turn_entry["selection_status"] = selection_status
            turn_entry["transition_validation"] = {
                "status": selection_status,
                "selected_by": "neurosymbolic",
                "findings": transition_findings,
            }
            session_state["transition_validation"] = deepcopy(
                turn_entry["transition_validation"]
            )
            session_state["status"] = (
                "selection_unresolved"
                if selection_status == "selection_unresolved"
                else "paused_after_outline_turn"
            )
            return selection_status, turn_entry
        selected = next(
            (
                row
                for row in candidate_evaluations
                if str(row.get("candidate_id") or "")
                == nondominated_candidate_ids[0]
                and str(row.get("selection_status") or "") == "nondominated"
            ),
            None,
        )
        selected_candidate_index = (
            int(selected.get("candidate_index") or 0) if selected else None
        )
        selected_by = "neurosymbolic"
    else:
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

    feedback_rows = _shared._merge_applicable_candidate_feedback(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        current_feedback_rows=_shared._candidate_feedback_rows(
            [selected] if selected else []
        ),
        prune_current=False,
    )
    selected_valid = bool(selected and selected.get("valid"))
    if not selected_valid or selected_candidate_index is None:
        session_state["candidate_rejection_feedback"] = deepcopy(feedback_rows)
        turn_entry["candidate_rejection_feedback"] = deepcopy(feedback_rows)
        turn_entry["transition_validation"] = {
            "status": "rejected",
            "selected_by": selected_by,
            "findings": deepcopy(feedback_rows),
        }
        if recovery_selection_mode == "pure_llm":
            turn_entry["transition_validation"]["selected_candidate_index"] = (
                selected_candidate_index
            )
        session_state["transition_validation"] = deepcopy(turn_entry["transition_validation"])
        _logger.info(
            "[MultiTurn] outline incremental_candidates_validated: rejected selected candidate %d (%s)",
            selected_candidate_index + 1 if selected_candidate_index is not None else 0,
            selected_by,
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry

    current_pa_fingerprint = _pa_state_fingerprint(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    selected_pa_fingerprint = str(selected.get("pa_state_fingerprint") or "")
    if current_pa_fingerprint != selected_pa_fingerprint:
        finding = _shared.annotate_validation_finding(
            {
                "task_id": str(dict(selected.get("task") or {}).get("outline_id") or ""),
                "resource_jid": _shared._task_resource_jid(
                    dict(selected.get("task") or {})
                )
                or None,
                "part_name": _shared._task_part_name(dict(selected.get("task") or {}))
                or None,
                "validation_category": TRANSITION_FEASIBILITY,
                "constraint_owner": "product",
                "constraint_family": "transition_staleness",
                "constraint_code": "validation_state_stale",
                "reason": "PA projected state changed before candidate commit.",
            }
        )
        feedback_rows = _shared._merge_applicable_candidate_feedback(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
            current_feedback_rows=_shared._candidate_feedback_rows(
                [{**dict(selected), "validation_findings": [finding]}]
            ),
            prune_current=False,
        )
        session_state["candidate_rejection_feedback"] = deepcopy(feedback_rows)
        turn_entry["candidate_rejection_feedback"] = deepcopy(feedback_rows)
        turn_entry["transition_validation"] = {
            "status": "rejected",
            "selected_by": selected_by,
            "findings": [deepcopy(finding)],
        }
        if recovery_selection_mode == "pure_llm":
            turn_entry["transition_validation"]["selected_candidate_index"] = (
                selected_candidate_index
            )
        session_state["transition_validation"] = deepcopy(
            turn_entry["transition_validation"]
        )
        session_state["status"] = "paused_after_outline_turn"
        return "need_revision", turn_entry

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

    if recovery_selection_mode == "pure_llm":
        turn_entry["selected_candidate_index"] = selected_candidate_index
    turn_entry["selected_by"] = selected_by
    turn_entry["selection_status"] = "selected"
    if recovery_selection_mode == "neurosymbolic":
        turn_entry["selection_evidence"] = deepcopy(
            selected.get("selection_evidence") or {}
        )
        turn_entry["tie_representative_evidence"] = deepcopy(
            selected.get("tie_representative_evidence") or {}
        )
        turn_entry["nondominated_candidate_ids"] = deepcopy(
            nondominated_candidate_ids
        )
    turn_entry["selected_transition"] = deepcopy(selected_transition)
    turn_entry["selected_transition_sequence"] = deepcopy(selected_committed_events)
    turn_entry["selected_candidate_task"] = deepcopy(selected_candidate_task)
    turn_entry["selected_candidate_trace"] = deepcopy(selected.get("surface_events") or [])
    turn_entry["next_transition"] = deepcopy(selected_transition)
    turn_entry["transition_validation"] = {
        "status": "passed",
        "selected_by": selected_by,
    }
    if recovery_selection_mode == "pure_llm":
        turn_entry["transition_validation"]["selected_candidate_index"] = (
            selected_candidate_index
        )
    if llm_selected_candidate_index is not None:
        turn_entry["llm_selected_candidate_index"] = llm_selected_candidate_index
    if selected_grounded_actions:
        turn_entry["grounded_action"] = deepcopy(selected_grounded_actions[0])
        turn_entry["grounded_actions"] = deepcopy(selected_grounded_actions)

    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    accepted_prefix.extend(deepcopy(selected_committed_events))
    session_state["accepted_outline_prefix"] = accepted_prefix
    session_state["outline_lookahead"] = []
    session_state["selection_revision_count"] = 0
    session_state["selection_revision_fingerprint"] = ""
    session_state["selection_repeated_failure_count"] = 0
    session_state["selection_repeated_failure_fingerprint"] = ""
    session_state["selection_revision_safety_rule_fingerprints"] = []
    session_state["selection_revision_live_dfa_fingerprints"] = []
    session_state["candidate_revision_targets"] = []
    session_state["candidate_revision_state_fingerprint"] = ""
    session_state["active_selection_ambiguity_feedback"] = {}
    session_state["active_selection_ambiguity_fingerprint"] = ""
    session_state["modeled_continuation_binding"] = (
        _next_modeled_continuation_binding(selected)
        if recovery_selection_mode == "neurosymbolic"
        else {}
    )
    session_state["projected_safety_dfa_states"] = deepcopy(
        selected.get("safety_dfa_states_after") or {}
    )
    session_state["projected_safety_rule_fingerprint"] = str(
        selected.get("safety_rule_fingerprint") or ""
    ).strip()
    session_state["live_safety_dfa_state_fingerprint"] = str(
        selected.get("live_safety_dfa_state_fingerprint") or ""
    ).strip()
    _shared._sync_des_recovery_aliases(
        session_state,
        turn_entry=turn_entry,
        transition_validation=turn_entry["transition_validation"],
    )

    for selected_event in selected_committed_events:
        _shared._apply_task_effects_to_symbolic_state(selected_event, session_state)
    applicable_feedback_rows: list[dict[str, Any]] = []
    for evaluation in candidate_evaluations:
        if not isinstance(evaluation, dict) or bool(evaluation.get("valid")):
            continue
        findings = _shared._prune_resolved_outline_validation_findings(
            [
                deepcopy(item)
                for item in (evaluation.get("validation_findings") or [])
                if isinstance(item, dict)
            ],
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        if not findings:
            continue
        applicable_feedback_rows.append(
            {
                "candidate_index": int(evaluation.get("candidate_index") or 0),
                "task": deepcopy(evaluation.get("task") or {}),
                "validation_findings": findings,
            }
        )
    enabledness_after = dict(
        selected.get("recovery_enabledness_validation_after") or {}
    )
    for event_evaluation in enabledness_after.get("event_evaluations") or []:
        if not isinstance(event_evaluation, dict):
            continue
        findings = _shared._prune_resolved_outline_validation_findings(
            [
                deepcopy(item)
                for item in (event_evaluation.get("findings") or [])
                if isinstance(item, dict)
            ],
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        if not findings:
            continue
        applicable_feedback_rows.append(
            {
                "candidate_index": -1,
                "task": {
                    "resource_jid": str(
                        event_evaluation.get("resource_jid") or ""
                    ).strip(),
                    "part_name": event_evaluation.get("part_name"),
                },
                "validation_findings": findings,
            }
        )
    applicable_feedback_rows = _shared._merge_applicable_candidate_feedback(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        current_feedback_rows=applicable_feedback_rows,
        prune_current=True,
    )
    session_state["candidate_rejection_feedback"] = deepcopy(
        applicable_feedback_rows
    )
    turn_entry["candidate_rejection_feedback"] = deepcopy(
        applicable_feedback_rows
    )
    session_state["pruned_actions"] = _shared._active_pruned_actions(
        session_state,
        prepared_recovery_request,
    )

    remaining_findings, remaining_conditions = _shared._remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    nominal_reentry_required = bool(
        _nominal_reentry_event_rows(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
    )
    nominal_reentry_ready = bool(
        selected.get("admissible_nominal_reentry_event_ids_after") or []
    )
    outline_complete = (
        remaining_findings == 0
        and remaining_conditions == 0
        and (not nominal_reentry_required or nominal_reentry_ready)
    )

    decision = "outline_ready" if outline_complete else "need_next_task"

    _logger.info(
        "[MultiTurn] outline incremental_candidates_validated: accepted %s-selected candidate %d (%s) "
        "(events=%d, prefix now %d events, complete=%s)",
        selected_by,
        selected_candidate_index + 1,
        str(selected_transition.get("outline_id") or "").strip(),
        len(selected_committed_events),
        len(accepted_prefix),
        outline_complete,
    )

    if outline_complete:
        session_state["status"] = "ready_for_primitive_generation"
    elif session_state.get("modeled_continuation_binding"):
        session_state["status"] = "running"
    else:
        session_state["status"] = "paused_after_outline_turn"
    return decision, turn_entry


async def _handle_outline_phase(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Dispatch outline handling according to configured outline mode."""
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()
    decision: str
    turn_entry: dict[str, Any]
    if outline_mode == "single_pass":
        decision, turn_entry = await _handle_outline_single_pass(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
            parsed_response=parsed_response,
        )
    elif outline_mode == "incremental_validated":
        decision, turn_entry = await _handle_outline_incremental_validated(
            session_state=session_state,
            parsed_response=parsed_response,
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    elif outline_mode == "incremental_candidates_validated":
        decision, turn_entry = await _handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response=parsed_response,
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    else:
        decision, turn_entry = await _handle_outline_incremental(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
            parsed_response=parsed_response,
        )

    remaining_findings, remaining_conditions = _shared._remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    turn_entry["remaining_blocked_issue_count"] = int(
        remaining_findings or 0
    ) + int(remaining_conditions or 0)

    if decision in {"outline_ready", "need_next_task"}:
        _clear_primitive_escalation_state(session_state)
    return decision, turn_entry


def _modeled_candidate_row(instance: dict[str, Any]) -> dict[str, Any]:
    """Keep the same candidate surface used for LLM-authored bridge actions."""
    task = dict(instance.get("task") or {})
    event_id = str(instance.get("event_id") or "").strip()
    row = {
        "outline_id": str(task.get("outline_id") or event_id).strip(),
        "event_name": deepcopy(task.get("event_name")),
        "resource_jid": deepcopy(task.get("resource_jid")),
        "expected_end_state": deepcopy(task.get("expected_end_state") or {}),
        "rationale": str(
            task.get("rationale") or "Configured enabled continuation."
        ),
    }
    part_name = str(task.get("part_name") or "").strip()
    if part_name:
        row["part_name"] = part_name
    return row


def _attach_modeled_task_steps(
    *,
    session_state: dict[str, Any],
    turn_entry: dict[str, Any],
    instance_by_outline_id: dict[str, dict[str, Any]],
) -> str:
    selected_candidate = dict(turn_entry.get("selected_candidate_task") or {})
    source_outline_id = str(selected_candidate.get("outline_id") or "").strip()
    instance = dict(instance_by_outline_id.get(source_outline_id) or {})
    modeled_task_steps = [
        deepcopy(row)
        for row in (instance.get("recovery_visible_steps") or [])
        if isinstance(row, dict)
    ]
    candidate_source = (
        "robot_task_program"
        if modeled_task_steps
        else "configured_capability"
    )

    selected_outline_ids = {
        str(row.get("outline_id") or "").strip()
        for row in (turn_entry.get("selected_transition_sequence") or [])
        if isinstance(row, dict) and str(row.get("outline_id") or "").strip()
    }
    for row in session_state.get("accepted_outline_prefix") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("outline_id") or "").strip() not in selected_outline_ids:
            continue
        row["candidate_source"] = candidate_source
        row["llm_called"] = False
        if modeled_task_steps:
            row["modeled_task_steps"] = deepcopy(modeled_task_steps)
    for key in (
        "selected_candidate_task",
        "selected_transition",
        "next_transition",
    ):
        row = turn_entry.get(key)
        if isinstance(row, dict):
            row["candidate_source"] = candidate_source
            row["llm_called"] = False
            if modeled_task_steps:
                row["modeled_task_steps"] = deepcopy(modeled_task_steps)
    for row in turn_entry.get("selected_transition_sequence") or []:
        if isinstance(row, dict):
            row["candidate_source"] = candidate_source
            row["llm_called"] = False
            if modeled_task_steps:
                row["modeled_task_steps"] = deepcopy(modeled_task_steps)
    return candidate_source


async def _try_handle_modeled_continuation(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """Validate and select the next enabled RobotTaskProgram transition."""
    if (
        _recovery_selection_mode(session_state) != "neurosymbolic"
        or str(session_state.get("outline_mode") or "").strip().lower()
        != "incremental_candidates_validated"
        or _action_horizon(session_state) != "1"
    ):
        return None
    binding = {
        key: str(value or "").strip()
        for key, value in dict(
            session_state.get("modeled_continuation_binding") or {}
        ).items()
        if key in {"resource_jid", "part_name"}
    }
    if not binding.get("resource_jid") or not binding.get("part_name"):
        return None

    unresolved_condition_ids = _unresolved_condition_ids(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    enabledness = await _agent_filtered_recovery_enabledness(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        unresolved_condition_ids=unresolved_condition_ids,
        planner=planner,
    )
    admissible_ids = set(enabledness.get("admissible_event_ids") or [])
    instances = [
        row
        for row in (enabledness.get("enabled_capability_results") or [])
        if isinstance(row, dict)
        and str(row.get("event_id") or "") in admissible_ids
        if str(row.get("resource_jid") or "").strip()
        == binding["resource_jid"]
        and str(row.get("part_name") or "").strip() == binding["part_name"]
    ]
    if not instances:
        session_state["modeled_continuation_binding"] = {}
        return None

    candidate_bound = max(
        1,
        int(session_state.get("candidate_bound") or _shared._DEFAULT_CANDIDATE_BOUND),
    )
    instances = instances[:candidate_bound]
    modeled_candidates = [
        _modeled_candidate_row(dict(instance)) for instance in instances
    ]
    parsed_response = {"candidate_events": modeled_candidates}
    instance_by_outline_id = {
        str(candidate.get("outline_id") or "").strip(): instance
        for candidate, instance in zip(
            modeled_candidates,
            instances,
            strict=True,
        )
    }
    working_state = deepcopy(session_state)
    configured_candidate_count = working_state.get("candidate_count", "auto")
    working_state["candidate_count"] = "auto"
    decision, turn_entry = await _handle_outline_phase(
        session_state=working_state,
        parsed_response=parsed_response,
        prepared_recovery_request=prepared_recovery_request,
        planner=planner,
    )
    if decision not in {"need_next_task", "outline_ready"}:
        session_state["modeled_continuation_binding"] = {}
        return None
    selection_evidence = dict(turn_entry.get("selection_evidence") or {})
    unique_nondominated = (
        len(turn_entry.get("nondominated_candidate_ids") or []) == 1
        and dict(turn_entry.get("tie_representative_evidence") or {}).get(
            "used"
        )
        is not True
    )
    goal_advancing = bool(
        selection_evidence.get("cleared_recovery_obligation_ids")
        or selection_evidence.get(
            "newly_admissible_goal_recovery_event_ids"
        )
        or selection_evidence.get(
            "newly_cca_admissible_goal_recovery_event_ids"
        )
        or selection_evidence.get(
            "newly_enabled_nominal_reentry_event_ids"
        )
    )
    if not unique_nondominated or not goal_advancing:
        session_state["modeled_continuation_binding"] = {}
        return None

    candidate_source = _attach_modeled_task_steps(
        session_state=working_state,
        turn_entry=turn_entry,
        instance_by_outline_id=instance_by_outline_id,
    )
    working_state["candidate_count"] = configured_candidate_count
    turn_entry["candidate_source"] = candidate_source
    turn_entry["llm_called"] = False
    session_state.clear()
    session_state.update(working_state)
    return decision, turn_entry, parsed_response


__all__ = [
    "_projected_outline_validation_context",
    "_handle_outline_single_pass",
    "_handle_outline_incremental",
    "_handle_outline_incremental_validated",
    "_handle_outline_incremental_candidates_validated",
    "_handle_outline_phase",
    "_try_handle_modeled_continuation",
]
