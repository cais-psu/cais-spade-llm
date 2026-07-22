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
    compile_grounded_recovery_outline_task,
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


def _recovery_des_models(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    models = {
        str(resource_jid): deepcopy(dict(descriptor or {}))
        for resource_jid, descriptor in dict(
            session_state.get("recovery_des_models") or {}
        ).items()
        if str(resource_jid) and isinstance(descriptor, dict)
    }
    for resource_jid, raw_entry in dict(
        prepared_recovery_request.get("recovery_resources") or {}
    ).items():
        descriptor = dict(dict(raw_entry or {}).get("recovery_des_model") or {})
        if descriptor:
            models.setdefault(str(resource_jid), deepcopy(descriptor))
    return models


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


def _recovery_relevant_event_ids(
    *,
    models: dict[str, dict[str, Any]],
    relevant_resource_ids: set[str],
    prepared_recovery_request: dict[str, Any],
    unresolved_condition_ids: set[str],
) -> set[str]:
    relevant_fields: set[str] = set()
    for condition in _shared._active_continuation_conditions(
        prepared_recovery_request
    ):
        if _exact_condition_identifier(condition) not in unresolved_condition_ids:
            continue
        field_name = str(condition.get("field") or "").strip()
        if field_name:
            relevant_fields.add(field_name)
        if str(condition.get("entity_kind") or "").strip().lower() == "part":
            relevant_fields.update({"held_part", "part_state", "part_location"})

    event_rows: list[tuple[str, dict[str, Any]]] = []
    for resource_jid in sorted(relevant_resource_ids):
        for event in dict(models.get(resource_jid) or {}).get("events") or []:
            if not isinstance(event, dict) or event.get("controllable") is not True:
                continue
            event_id = json.dumps(
                {
                    "resource_jid": resource_jid,
                    "event_name": str(event.get("event_name") or ""),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            event_rows.append((event_id, event))

    selected: set[str] = set()
    for pass_index in range(2):
        changed = True
        while changed:
            changed = False
            for event_id, event in event_rows:
                update_fields = {
                    str(field) for field in dict(event.get("updates") or {})
                }
                if event_id in selected or not (update_fields & relevant_fields):
                    continue
                selected.add(event_id)
                relevant_fields.update(
                    str(field) for field in dict(event.get("guards") or {})
                )
                changed = True
        if selected or pass_index > 0:
            break
        # If the product obligation does not name an RA-local variable, start
        # backward relevance at the RA's own guard variables. This remains
        # descriptor-driven and does not inspect event names or resource types.
        for _event_id, event in event_rows:
            relevant_fields.update(
                str(field) for field in dict(event.get("guards") or {})
            )
    return selected


def _efa_guard_is_satisfied(condition: Any, actual: Any) -> bool:
    if not isinstance(condition, dict):
        return True
    if condition.get("exists") is True and actual in (None, ""):
        return False
    if "equals" in condition and actual != condition.get("equals"):
        return False
    return not (
        "not_equals" in condition and actual == condition.get("not_equals")
    )


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
    models = _recovery_des_models(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
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
            resource_model = dict(models.get(resource_jid) or {})
            exact_ra_event = next(
                (
                    deepcopy(event)
                    for event in (resource_model.get("events") or [])
                    if isinstance(event, dict)
                    and str(event.get("event_name") or "").strip()
                    == str(task.get("function_name") or "").strip()
                ),
                {},
            )
            if not tool_row or not exact_ra_event:
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
                    "ra_event": exact_ra_event,
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
    return sorted(rows, key=lambda row: str(row.get("event_id") or ""))


def _admissible_recovery_enabled_event_ids(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    unresolved_condition_ids: set[str],
) -> set[str]:
    if not unresolved_condition_ids:
        return set()
    models = _recovery_des_models(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    if not models:
        return set()
    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    relevant_resources = _recovery_relevant_resource_ids(
        unresolved_condition_ids=unresolved_condition_ids,
        prepared_recovery_request=prepared_recovery_request,
        resource_ids=set(models),
    )
    relevant_event_ids = _recovery_relevant_event_ids(
        models=models,
        relevant_resource_ids=relevant_resources,
        prepared_recovery_request=prepared_recovery_request,
        unresolved_condition_ids=unresolved_condition_ids,
    )
    enabled: set[str] = set()
    for resource_jid in sorted(relevant_resources):
        resource_row = dict(resources_by_jid.get(resource_jid) or {})
        resource_model = dict(models.get(resource_jid) or {})
        state_variables = dict(resource_model.get("state_variables") or {})
        for event in resource_model.get("events") or []:
            if not isinstance(event, dict) or event.get("controllable") is not True:
                continue
            event_id = json.dumps(
                {
                    "resource_jid": resource_jid,
                    "event_name": str(event.get("event_name") or ""),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if event_id not in relevant_event_ids:
                continue
            guards = dict(event.get("guards") or {})
            resource_guards = {
                str(field_name): condition
                for field_name, condition in guards.items()
                if str(
                    dict(state_variables.get(str(field_name)) or {}).get("scope")
                    or "resource"
                )
                == "resource"
            }
            part_guards = {
                str(field_name): condition
                for field_name, condition in guards.items()
                if str(
                    dict(state_variables.get(str(field_name)) or {}).get("scope")
                    or "resource"
                )
                == "part"
            }
            resource_enabled = all(
                _efa_guard_is_satisfied(condition, resource_row.get(str(field_name)))
                for field_name, condition in resource_guards.items()
            )
            if not resource_enabled:
                continue
            if not part_guards:
                enabled.add(event_id)
                continue
            for part_name, part_row in sorted(parts_by_name.items()):
                if all(
                    _efa_guard_is_satisfied(
                        condition,
                        dict(part_row or {}).get(str(field_name)),
                    )
                    for field_name, condition in part_guards.items()
                ):
                    enabled.add(
                        json.dumps(
                            {
                                "resource_jid": resource_jid,
                                "event_name": str(event.get("event_name") or ""),
                                "part_name": str(part_name),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
    return enabled


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


def _recovery_event_requires_part_binding(
    *,
    event: dict[str, Any],
    state_variables: dict[str, Any],
) -> bool:
    for field_name in set(dict(event.get("guards") or {})) | set(
        dict(event.get("updates") or {})
    ):
        declaration = dict(state_variables.get(str(field_name)) or {})
        if str(declaration.get("scope") or "resource").strip() == "part":
            return True
    held_part_update = dict(dict(event.get("updates") or {}).get("held_part") or {})
    return bool(
        str(held_part_update.get("set_from_param") or "").strip() == "part_name"
        or "part_name" in (held_part_update.get("set_from_param_any_of") or [])
    )


def _recovery_event_update_value(
    *,
    field_name: str,
    update: Any,
    resource_jid: str,
    part_name: str,
    part_row: dict[str, Any],
) -> tuple[bool, Any]:
    if not isinstance(update, dict):
        return False, None
    if "set" in update:
        return True, deepcopy(update.get("set"))
    parameter_names: list[str] = []
    if str(update.get("set_from_param") or "").strip():
        parameter_names.append(str(update.get("set_from_param") or "").strip())
    parameter_names.extend(
        str(item).strip()
        for item in (update.get("set_from_param_any_of") or [])
        if str(item).strip()
    )
    for parameter_name in parameter_names:
        if parameter_name == "part_name" and part_name:
            return True, part_name
        if field_name == "part_location":
            for source_name in ("goal_location", "current_location", "origin_location"):
                value = part_row.get(source_name)
                if value not in (None, ""):
                    return True, deepcopy(value)
        if field_name == "resource_location":
            for source_name in ("current_location", "origin_location", "goal_location"):
                value = part_row.get(source_name)
                if value not in (None, ""):
                    return True, deepcopy(value)
        if parameter_name in {"resource_jid", "holder_resource_jid"}:
            return True, resource_jid
    return False, None


def _recovery_event_instance_task(
    *,
    event_id: str,
    resource_jid: str,
    event: dict[str, Any],
    resource_row: dict[str, Any],
    part_name: str,
    part_row: dict[str, Any],
    state_variables: dict[str, Any],
) -> dict[str, Any] | None:
    start_state: dict[str, Any] = {}
    end_state: dict[str, Any] = {}
    for field_name, raw_declaration in state_variables.items():
        declaration = dict(raw_declaration or {})
        scope = str(declaration.get("scope") or "resource").strip()
        if scope == "part" and not part_name:
            continue
        source_row = part_row if scope == "part" else resource_row
        value = _exact_state_field_value(source_row, str(field_name))
        start_state[str(field_name)] = deepcopy(value)
        end_state[str(field_name)] = deepcopy(value)

    if "resource_state" not in start_state:
        start_state["resource_state"] = _exact_state_field_value(
            resource_row, "resource_state"
        )
        end_state["resource_state"] = deepcopy(start_state["resource_state"])
    if "held_part" not in start_state:
        start_state["held_part"] = deepcopy(resource_row.get("held_part"))
        end_state["held_part"] = deepcopy(resource_row.get("held_part"))
    if part_name:
        if "part_state" not in start_state:
            start_state["part_state"] = _exact_state_field_value(
                part_row, "part_state"
            )
            end_state["part_state"] = deepcopy(start_state["part_state"])
        if "part_location" not in start_state:
            start_state["part_location"] = _exact_state_field_value(
                part_row, "part_location"
            )
            end_state["part_location"] = deepcopy(start_state["part_location"])

    applied_update = False
    for field_name, update in dict(event.get("updates") or {}).items():
        resolved, value = _recovery_event_update_value(
            field_name=str(field_name),
            update=update,
            resource_jid=resource_jid,
            part_name=part_name,
            part_row=part_row,
        )
        if not resolved:
            continue
        end_state[str(field_name)] = deepcopy(value)
        applied_update = True
    if not applied_update:
        return None

    start_held_part = str(start_state.get("held_part") or "").strip()
    end_held_part = str(end_state.get("held_part") or "").strip()
    action_target: dict[str, Any] = {}
    if part_name and start_held_part != part_name and end_held_part == part_name:
        if dict(part_row.get("observed_pose") or {}):
            action_target["source_location"] = "observed_pose"
        elif part_row.get("current_location") not in (None, ""):
            action_target["source_location"] = deepcopy(
                part_row.get("current_location")
            )
    if part_name and start_held_part == part_name and end_held_part != part_name:
        if end_state.get("part_location") not in (None, ""):
            action_target["target_location"] = deepcopy(
                end_state.get("part_location")
            )
    if (
        part_name
        and "target_location" not in action_target
        and end_state.get("part_location") not in (None, "")
        and end_state.get("part_location") != start_state.get("part_location")
        and not str(end_state.get("part_location") or "").endswith("_gripper")
    ):
        action_target["target_location"] = deepcopy(end_state.get("part_location"))

    task: dict[str, Any] = {
        "outline_id": f"enabledness_{recovery_validation_fingerprint(event_id)[:16]}",
        "event_name": str(event.get("event_name") or "").strip(),
        "resource_jid": resource_jid,
        "expected_start_state": start_state,
        "expected_end_state": end_state,
        "rationale": "",
    }
    if part_name:
        task["part_name"] = part_name
    if action_target:
        task["action_target"] = action_target
    return task


def _goal_relevant_recovery_event_rows(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    unresolved_condition_ids: set[str],
) -> list[dict[str, Any]]:
    if not unresolved_condition_ids:
        return []
    models = _recovery_des_models(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    relevant_resources = _recovery_relevant_resource_ids(
        unresolved_condition_ids=unresolved_condition_ids,
        prepared_recovery_request=prepared_recovery_request,
        resource_ids=set(models),
    )
    relevant_event_ids = _recovery_relevant_event_ids(
        models=models,
        relevant_resource_ids=relevant_resources,
        prepared_recovery_request=prepared_recovery_request,
        unresolved_condition_ids=unresolved_condition_ids,
    )
    relevant_parts = _recovery_relevant_part_names(
        unresolved_condition_ids=unresolved_condition_ids,
        prepared_recovery_request=prepared_recovery_request,
        available_part_names=set(parts_by_name),
    )
    rows: list[dict[str, Any]] = []
    for resource_jid in sorted(relevant_resources):
        resource_row = dict(resources_by_jid.get(resource_jid) or {})
        resource_model = dict(models.get(resource_jid) or {})
        state_variables = dict(resource_model.get("state_variables") or {})
        for raw_event in resource_model.get("events") or []:
            if not isinstance(raw_event, dict) or raw_event.get("controllable") is not True:
                continue
            event = dict(raw_event)
            base_event_id = json.dumps(
                {
                    "resource_jid": resource_jid,
                    "event_name": str(event.get("event_name") or ""),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if base_event_id not in relevant_event_ids:
                continue
            part_bindings = (
                sorted(relevant_parts)
                if _recovery_event_requires_part_binding(
                    event=event,
                    state_variables=state_variables,
                )
                else [""]
            )
            for part_name in part_bindings:
                part_row = dict(parts_by_name.get(part_name) or {}) if part_name else {}
                event_id = json.dumps(
                    {
                        "resource_jid": resource_jid,
                        "event_name": str(event.get("event_name") or ""),
                        **({"part_name": part_name} if part_name else {}),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                task = _recovery_event_instance_task(
                    event_id=event_id,
                    resource_jid=resource_jid,
                    event=event,
                    resource_row=resource_row,
                    part_name=part_name,
                    part_row=part_row,
                    state_variables=state_variables,
                )
                if task is None:
                    continue
                update_fields = set(dict(event.get("updates") or {}))
                rows.append(
                    {
                        "event_id": event_id,
                        "task": task,
                        "signature": {
                            "inferable_primary_part": part_name,
                            "task_kind": (
                                "part_handling" if part_name else "resource_action"
                            ),
                            "changes_part_world": bool(
                                part_name
                                and update_fields
                                & {"held_part", "part_state", "part_location"}
                            ),
                        },
                    }
                )
    return sorted(rows, key=lambda row: str(row.get("event_id") or ""))


def _symbolically_enabled_recovery_event_instances(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    unresolved_condition_ids: set[str],
) -> list[dict[str, Any]]:
    if not unresolved_condition_ids:
        return []
    models = _recovery_des_models(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    relevant_resources = _recovery_relevant_resource_ids(
        unresolved_condition_ids=unresolved_condition_ids,
        prepared_recovery_request=prepared_recovery_request,
        resource_ids=set(models),
    )
    relevant_event_ids = _recovery_relevant_event_ids(
        models=models,
        relevant_resource_ids=relevant_resources,
        prepared_recovery_request=prepared_recovery_request,
        unresolved_condition_ids=unresolved_condition_ids,
    )
    relevant_parts = _recovery_relevant_part_names(
        unresolved_condition_ids=unresolved_condition_ids,
        prepared_recovery_request=prepared_recovery_request,
        available_part_names=set(parts_by_name),
    )
    instances: list[dict[str, Any]] = []
    for resource_jid in sorted(relevant_resources):
        resource_row = dict(resources_by_jid.get(resource_jid) or {})
        resource_model = dict(models.get(resource_jid) or {})
        state_variables = dict(resource_model.get("state_variables") or {})
        for raw_event in resource_model.get("events") or []:
            if not isinstance(raw_event, dict) or raw_event.get("controllable") is not True:
                continue
            event = dict(raw_event)
            base_event_id = json.dumps(
                {
                    "resource_jid": resource_jid,
                    "event_name": str(event.get("event_name") or ""),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if base_event_id not in relevant_event_ids:
                continue
            part_bindings = (
                sorted(relevant_parts)
                if _recovery_event_requires_part_binding(
                    event=event,
                    state_variables=state_variables,
                )
                else [""]
            )
            for part_name in part_bindings:
                part_row = dict(parts_by_name.get(part_name) or {}) if part_name else {}
                guards = dict(event.get("guards") or {})
                if not all(
                    _efa_guard_is_satisfied(
                        condition,
                        _exact_state_field_value(
                            part_row
                            if str(
                                dict(state_variables.get(str(field_name)) or {}).get(
                                    "scope"
                                )
                                or "resource"
                            ).strip()
                            == "part"
                            else resource_row,
                            str(field_name),
                        ),
                    )
                    for field_name, condition in guards.items()
                ):
                    continue
                event_id = json.dumps(
                    {
                        "resource_jid": resource_jid,
                        "event_name": str(event.get("event_name") or ""),
                        **({"part_name": part_name} if part_name else {}),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                task = _recovery_event_instance_task(
                    event_id=event_id,
                    resource_jid=resource_jid,
                    event=event,
                    resource_row=resource_row,
                    part_name=part_name,
                    part_row=part_row,
                    state_variables=state_variables,
                )
                if task is None:
                    continue
                grounding = compile_grounded_recovery_outline_task(
                    task,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                    location_validation_mode="relaxed",
                )
                grounded_action = dict(grounding.get("grounded_action") or {})
                if not grounded_action:
                    continue
                physical_input = build_recovery_physical_validation_input(
                    task=task,
                    grounded_action=grounded_action,
                    session_state=session_state,
                    prepared_recovery_request=prepared_recovery_request,
                )
                physical_input["use_projected_recovery_snapshot"] = True
                instances.append(
                    {
                        "event_id": event_id,
                        "resource_jid": resource_jid,
                        "part_name": part_name or None,
                        "task": task,
                        "grounded_action": grounded_action,
                        "physical_input": physical_input,
                        "safety_input": build_recovery_safety_validation_input(
                            task=task,
                            session_state=session_state,
                            prepared_recovery_request=prepared_recovery_request,
                        ),
                    }
                )
    return sorted(instances, key=lambda row: str(row.get("event_id") or ""))


def _exact_state_field_value(row: dict[str, Any], field_name: str) -> Any:
    aliases = {
        "resource_state": "current_state",
        "resource_location": "current_location",
        "part_state": "current_state",
        "part_location": "current_location",
    }
    if field_name in row:
        return row.get(field_name)
    return row.get(aliases.get(field_name, field_name))


def _resource_reachable_location_tokens(resource_row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for field_name in ("reachability", "reachable_locations", "known_locations"):
        raw_value = resource_row.get(field_name)
        if isinstance(raw_value, (dict, list, tuple, set)):
            tokens.update(
                str(token).strip() for token in raw_value if str(token).strip()
            )
    return tokens


def _nominal_reentry_guard_is_enabled(
    *,
    row: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
) -> bool:
    task = dict(row.get("task") or {})
    tool = dict(row.get("tool") or {})
    ra_event = dict(row.get("ra_event") or {})
    resource_jid = str(task.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    part_row = dict(parts_by_name.get(part_name) or {}) if part_name else {}
    if not resource_jid or not resource_row:
        return False

    required_resource_state = str(tool.get("in_state") or "").strip()
    actual_resource_state = str(
        _exact_state_field_value(resource_row, "resource_state") or ""
    ).strip()
    if (
        required_resource_state
        and required_resource_state.lower() != "any"
        and actual_resource_state != required_resource_state
    ):
        return False
    required_part_state = str(tool.get("part_in_state") or "").strip()
    if required_part_state and (
        not part_name
        or str(
            _exact_state_field_value(part_row, "part_state") or ""
        ).strip()
        != required_part_state
    ):
        return False

    params = dict(task.get("params") or {})
    context_mapping = dict(tool.get("context_mapping") or {})
    location_param = str(context_mapping.get("location_param") or "").strip()
    location_type = str(context_mapping.get("location_type") or "").strip()
    location_value = params.get(location_param) if location_param else None
    if location_param and location_value not in (None, ""):
        if location_type == "part_location":
            if not part_name or _exact_state_field_value(
                part_row, "part_location"
            ) != location_value:
                return False
        elif location_type == "current_location":
            if _exact_state_field_value(
                resource_row, "resource_location"
            ) != location_value:
                return False
        elif (
            location_type == "reachable_location"
            and str(location_value)
            not in _resource_reachable_location_tokens(resource_row)
        ):
            return False

    state_variables = dict(row.get("state_variables") or {})
    for field_name, condition in dict(ra_event.get("guards") or {}).items():
        scope = str(
            dict(state_variables.get(str(field_name)) or {}).get("scope")
            or ("part" if str(field_name).startswith("part_") else "resource")
        ).strip()
        source_row = part_row if scope == "part" else resource_row
        if not _efa_guard_is_satisfied(
            condition,
            _exact_state_field_value(source_row, str(field_name)),
        ):
            return False
    return True


def _admissible_nominal_reentry_event_ids(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    cca_admissible_event_ids: set[str],
) -> set[str]:
    resources_by_jid, parts_by_name = _projected_outline_validation_context(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    models = _recovery_des_models(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    enabled: set[str] = set()
    for row in _nominal_reentry_event_rows(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    ):
        event_id = str(row.get("event_id") or "").strip()
        resource_jid = str(dict(row.get("task") or {}).get("resource_jid") or "").strip()
        enriched_row = deepcopy(row)
        enriched_row["state_variables"] = deepcopy(
            dict(dict(models.get(resource_jid) or {}).get("state_variables") or {})
        )
        if (
            event_id
            and event_id in cca_admissible_event_ids
            and _nominal_reentry_guard_is_enabled(
                row=enriched_row,
                resources_by_jid=resources_by_jid,
                parts_by_name=parts_by_name,
            )
        ):
            enabled.add(event_id)
    return enabled


async def _agent_filtered_recovery_enabledness(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    unresolved_condition_ids: set[str],
    planner: Any,
) -> dict[str, Any]:
    """Return recovery events admitted by symbolic, RA, and CCA validation."""
    instances = _symbolically_enabled_recovery_event_instances(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        unresolved_condition_ids=unresolved_condition_ids,
    )
    result: dict[str, Any] = {
        "symbolically_enabled_event_ids": [
            str(row.get("event_id") or "") for row in instances
        ],
        "ra_admissible_event_ids": [],
        "cca_admissible_event_ids": [],
        "admissible_event_ids": [],
        "event_evaluations": [],
        "ra_snapshot_fingerprints": {},
        "ra_descriptor_fingerprints": {},
        "safety_rule_fingerprint": "",
        "live_safety_dfa_state_fingerprint": "",
    }
    if not instances:
        return result

    product_agent = getattr(planner, "product_agent", None)
    request_physical = getattr(
        product_agent, "request_recovery_outline_physical_validation", None
    )
    request_safety = getattr(
        product_agent, "request_recovery_outline_safety_validation", None
    )
    if not callable(request_physical) or not callable(request_safety):
        result["event_evaluations"] = [
            {
                "event_id": str(row.get("event_id") or ""),
                "resource_jid": str(row.get("resource_jid") or ""),
                "part_name": row.get("part_name"),
                "ra_status": "unavailable",
                "cca_status": "skipped",
                "constraint_codes": ["resource_validation_unavailable"],
            }
            for row in instances
        ]
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
    instances_by_resource: dict[str, list[dict[str, Any]]] = {}
    for index, instance in enumerate(instances):
        instance["enabledness_index"] = index
        instances_by_resource.setdefault(
            str(instance.get("resource_jid") or ""), []
        ).append(instance)

    async def validate_resource_group(
        resource_jid: str,
        rows: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any] | Exception]:
        payload = {
            "recovery_session_id": recovery_session_id,
            "turn_index": turn_index,
            "state_fingerprint": state_fingerprint,
            "candidates": [
                {
                    "candidate_index": int(row.get("enabledness_index") or 0),
                    "event_id": str(row.get("event_id") or ""),
                    "task": deepcopy(row.get("task") or {}),
                    "physical_input": deepcopy(row.get("physical_input") or {}),
                }
                for row in rows
            ],
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
            validate_resource_group(resource_jid, rows)
            for resource_jid, rows in sorted(instances_by_resource.items())
        ]
    )
    evaluations_by_event_id = {
        str(row.get("event_id") or ""): {
            "event_id": str(row.get("event_id") or ""),
            "resource_jid": str(row.get("resource_jid") or ""),
            "part_name": row.get("part_name"),
            "ra_status": "unavailable",
            "cca_status": "skipped",
            "constraint_codes": [],
        }
        for row in instances
    }
    ra_admissible_instances: list[dict[str, Any]] = []
    instance_by_index = {
        int(row.get("enabledness_index") or 0): row for row in instances
    }
    expected_models = _recovery_des_models(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    for resource_jid, reply_or_error in ra_replies:
        resource_instances = instances_by_resource.get(resource_jid) or []
        if isinstance(reply_or_error, Exception):
            for instance in resource_instances:
                evaluation = evaluations_by_event_id[str(instance.get("event_id") or "")]
                evaluation["constraint_codes"] = ["resource_validation_unavailable"]
                evaluation["reason"] = str(reply_or_error)
            continue
        reply = dict(reply_or_error or {})
        snapshot = dict(reply.get("snapshot") or {})
        snapshot_fingerprint = str(reply.get("snapshot_fingerprint") or "").strip()
        if (
            not snapshot_fingerprint
            or snapshot_fingerprint != recovery_validation_fingerprint(snapshot)
        ):
            for instance in resource_instances:
                evaluation = evaluations_by_event_id[str(instance.get("event_id") or "")]
                evaluation["constraint_codes"] = ["resource_validation_unavailable"]
                evaluation["reason"] = "ResourceAgent enabledness snapshot fingerprint is invalid"
            continue
        expected_descriptor_fingerprint = str(
            dict(expected_models.get(resource_jid) or {}).get("descriptor_fingerprint")
            or ""
        ).strip()
        try:
            _, descriptor_fingerprint = _verified_recovery_des_model(
                ra_reply=reply,
                expected_fingerprint=expected_descriptor_fingerprint,
            )
        except Exception as exc:  # noqa: BLE001 - descriptor mismatch fails closed
            for instance in resource_instances:
                evaluation = evaluations_by_event_id[str(instance.get("event_id") or "")]
                evaluation["constraint_codes"] = ["resource_validation_unavailable"]
                evaluation["reason"] = str(exc)
            continue
        result["ra_snapshot_fingerprints"][resource_jid] = snapshot_fingerprint
        result["ra_descriptor_fingerprints"][resource_jid] = descriptor_fingerprint
        for raw_row in reply.get("results") or []:
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            instance = instance_by_index.get(int(row.get("candidate_index") or 0))
            if not instance or str(instance.get("resource_jid") or "") != resource_jid:
                continue
            event_id = str(instance.get("event_id") or "")
            evaluation = evaluations_by_event_id[event_id]
            findings = [
                dict(item)
                for item in (row.get("findings") or [])
                if isinstance(item, dict)
            ]
            evaluation["ra_status"] = "passed" if bool(row.get("allowed")) else "rejected"
            evaluation["ra_mocked"] = bool(reply.get("mocked"))
            evaluation["constraint_codes"] = sorted(
                {
                    str(item.get("constraint_code") or "").strip()
                    for item in findings
                    if str(item.get("constraint_code") or "").strip()
                }
            )
            if findings:
                evaluation["findings"] = [
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
                    | (
                        {
                            "evidence": {
                                evidence_key: deepcopy(
                                    dict(finding.get("evidence") or {}).get(
                                        evidence_key
                                    )
                                )
                                for evidence_key in (
                                    "checked_pose",
                                    "workspace_bounds",
                                )
                                if dict(finding.get("evidence") or {}).get(
                                    evidence_key
                                )
                                not in (None, "", [], {})
                            }
                        }
                        if any(
                            dict(finding.get("evidence") or {}).get(evidence_key)
                            not in (None, "", [], {})
                            for evidence_key in ("checked_pose", "workspace_bounds")
                        )
                        else {}
                    )
                    for finding in findings
                ]
            if bool(row.get("allowed")):
                ra_admissible_instances.append(instance)

    result["ra_admissible_event_ids"] = sorted(
        {str(row.get("event_id") or "") for row in ra_admissible_instances}
    )
    if not ra_admissible_instances:
        result["event_evaluations"] = [
            evaluations_by_event_id[event_id]
            for event_id in sorted(evaluations_by_event_id)
        ]
        return result

    nominal_reentry_events = _nominal_reentry_event_rows(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    goal_recovery_events = _goal_relevant_recovery_event_rows(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        unresolved_condition_ids=unresolved_condition_ids,
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
        safety_input.setdefault("llm_input", {})["goal_recovery_events"] = [
            {
                "event_id": str(row.get("event_id") or ""),
                "task": deepcopy(row.get("task") or {}),
                "signature": deepcopy(row.get("signature") or {}),
            }
            for row in goal_recovery_events
            if str(row.get("event_id") or "")
        ]
        cca_candidates.append(
            {
                "candidate_index": int(instance.get("enabledness_index") or 0),
                "event_id": str(instance.get("event_id") or ""),
                "task": deepcopy(instance.get("task") or {}),
                "safety_input": safety_input,
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
    for raw_row in cca_reply.get("results") or []:
        if not isinstance(raw_row, dict):
            continue
        row = dict(raw_row)
        instance = instance_by_index.get(int(row.get("candidate_index") or 0))
        if not instance:
            continue
        event_id = str(instance.get("event_id") or "")
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


def _candidate_realizer_event_ids(
    *,
    candidate: dict[str, Any],
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
) -> set[str]:
    surface_events = [
        dict(row) for row in (candidate.get("surface_events") or []) if isinstance(row, dict)
    ]
    if len(surface_events) != 1:
        return set()
    task = surface_events[0]
    resource_jid = str(task.get("resource_jid") or "").strip()
    end_state = dict(task.get("expected_end_state") or {})
    if not resource_jid or not end_state:
        return set()
    models = _recovery_des_models(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    realized: set[str] = set()
    for event in dict(models.get(resource_jid) or {}).get("events") or []:
        if not isinstance(event, dict) or event.get("controllable") is not True:
            continue
        matched_update = False
        incompatible_update = False
        for field_name, update in dict(event.get("updates") or {}).items():
            if str(field_name) == "current_state":
                continue
            candidate_field = {
                "current_location": "resource_location",
            }.get(str(field_name), str(field_name))
            if candidate_field not in end_state or not isinstance(update, dict):
                continue
            expected_value: Any = object()
            if "set" in update:
                expected_value = update.get("set")
            elif "set_from_param" in update:
                expected_value = task.get(str(update.get("set_from_param") or ""))
            elif "set_from_param_any_of" in update:
                expected_value = next(
                    (
                        task.get(str(param_name))
                        for param_name in (update.get("set_from_param_any_of") or [])
                        if task.get(str(param_name)) not in (None, "")
                    ),
                    object(),
                )
            if type(expected_value) is object:
                continue
            if end_state.get(candidate_field) != expected_value:
                incompatible_update = True
                break
            matched_update = True
        if matched_update and not incompatible_update:
            realized.add(
                json.dumps(
                    {
                        "resource_jid": resource_jid,
                        "event_name": str(event.get("event_name") or ""),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    return realized


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
    return sorted(
        field_name
        for field_name, value in end_state.items()
        if field_name not in start_state or start_state.get(field_name) != value
    )


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
        field_name in {"resource_state", "current_state", "part_state"}
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
        newly_enabled = enabled_events_after - current_enabled_events
        disabled_events = current_enabled_events - enabled_events_after
        newly_enabled_nominal_reentry_events = (
            nominal_reentry_events_after - current_nominal_reentry_events
        )
        newly_cca_admissible_goal_recovery_events = (
            cca_goal_recovery_events_after - current_cca_goal_recovery_events
        )
        realizer_events = _candidate_realizer_event_ids(
            candidate=candidate,
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        continuation_enabled_events_before = current_enabled_events - realizer_events
        progressing_candidate = (
            remaining < current_unresolved and not introduced
        ) or (
            remaining == current_unresolved
            and bool(newly_enabled)
        ) or (
            remaining == current_unresolved
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
            "cca_admissible_goal_recovery_event_ids_before": sorted(
                current_cca_goal_recovery_events
            ),
            "cca_admissible_goal_recovery_event_ids_after": sorted(
                cca_goal_recovery_events_after
            ),
            "newly_cca_admissible_goal_recovery_event_ids": sorted(
                newly_cca_admissible_goal_recovery_events
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

    recovery_preferred: list[dict[str, Any]] = []
    for candidate in obligation_preferred:
        candidate_events = set(
            candidate.get("admissible_recovery_enabled_event_ids_after") or []
        )
        dominators = [
            other
            for other in obligation_preferred
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
    descriptor_fingerprints = {
        str(resource_jid): str(
            dict(descriptor or {}).get("descriptor_fingerprint") or ""
        )
        for resource_jid, descriptor in _recovery_des_models(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        ).items()
    }
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
            "recovery_des_model_fingerprints": descriptor_fingerprints,
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
    transition_findings = [
        deepcopy(row)
        for row in findings
        if str(row.get("validation_category") or "").strip()
        != SYNTAX_AND_GROUNDING_VALIDATION
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
    stages.append(
        recovery_validation_stage(
            validation_category=TRANSITION_FEASIBILITY,
            validator_role="PA",
            validator_jid=product_jid,
            status=(
                "skipped"
                if syntax_findings
                else "rejected" if transition_findings else "passed"
            ),
            findings=transition_findings,
            state_fingerprint=state_fingerprint,
        )
    )


def _ra_snapshot_staleness_findings(
    *,
    task: dict[str, Any],
    snapshot: dict[str, Any],
    session_state: dict[str, Any],
) -> list[dict[str, Any]]:
    resource_jid = _shared._task_resource_jid(task)
    if any(
        isinstance(row, dict)
        and _shared._task_resource_jid(row) == resource_jid
        for row in (session_state.get("accepted_outline_prefix") or [])
    ):
        return []
    start_state = dict(task.get("expected_start_state") or {})
    occupancy = dict(snapshot.get("occupancy") or {})
    actual_by_field = {
        "resource_state": (
            snapshot.get("resource_state")
            if "resource_state" in snapshot
            else snapshot.get("current_state")
        ),
        "resource_location": (
            snapshot.get("resource_location")
            if "resource_location" in snapshot
            else snapshot.get("current_location")
            if "current_location" in snapshot
            else occupancy.get("location")
        ),
        "held_part": snapshot.get("held_part"),
    }
    mismatches = [
        {
            "field": field_name,
            "expected": deepcopy(start_state.get(field_name)),
            "actual": deepcopy(actual_by_field.get(field_name)),
        }
        for field_name in ("resource_state", "resource_location", "held_part")
        if field_name in start_state
        and actual_by_field.get(field_name) != start_state.get(field_name)
    ]
    if not mismatches:
        return []
    return [
        _shared.annotate_validation_finding(
            {
                "task_id": str(task.get("outline_id") or "").strip(),
                "resource_jid": _shared._task_resource_jid(task) or None,
                "part_name": _shared._task_part_name(task) or None,
                "validation_category": TRANSITION_FEASIBILITY,
                "constraint_owner": "product",
                "constraint_family": "transition_staleness",
                "constraint_code": "validation_state_stale",
                "reason": "ResourceAgent live state changed after the PA projection.",
                "evidence": {"mismatches": mismatches},
                "retriable": True,
            }
        )
    ]


def _verified_recovery_des_model(
    *,
    ra_reply: dict[str, Any],
    expected_fingerprint: str = "",
) -> tuple[dict[str, Any], str]:
    """Verify one live RA descriptor without changing any formal token."""
    recovery_des_model = dict(ra_reply.get("recovery_des_model") or {})
    reply_fingerprint = str(
        ra_reply.get("recovery_des_model_fingerprint") or ""
    ).strip()
    descriptor_payload = deepcopy(recovery_des_model)
    embedded_fingerprint = str(
        descriptor_payload.pop("descriptor_fingerprint", "") or ""
    ).strip()
    if (
        not recovery_des_model
        or not reply_fingerprint
        or reply_fingerprint != embedded_fingerprint
        or reply_fingerprint != recovery_validation_fingerprint(descriptor_payload)
    ):
        raise RuntimeError("ResourceAgent recovery DES descriptor is unavailable or invalid")
    if expected_fingerprint and reply_fingerprint != str(expected_fingerprint).strip():
        raise RuntimeError("ResourceAgent recovery DES descriptor fingerprint changed")
    return recovery_des_model, reply_fingerprint


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
        "recovery_des_models": {},
        "recovery_des_model_fingerprints": {},
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
            ra_snapshot = dict(ra_reply.get("snapshot") or {})
            snapshot_fingerprint = str(
                ra_reply.get("snapshot_fingerprint") or ""
            ).strip()
            if not snapshot_fingerprint or snapshot_fingerprint != (
                recovery_validation_fingerprint(ra_snapshot)
            ):
                raise RuntimeError("ResourceAgent reply snapshot fingerprint is invalid")
            expected_recovery_des_model_fingerprint = str(
                dict(
                    _recovery_des_models(
                        session_state=session_state,
                        prepared_recovery_request=prepared_recovery_request,
                    ).get(resource_jid)
                    or {}
                ).get("descriptor_fingerprint")
                or ""
            ).strip()
            recovery_des_model, recovery_des_model_fingerprint = (
                _verified_recovery_des_model(
                    ra_reply=ra_reply,
                    expected_fingerprint=expected_recovery_des_model_fingerprint,
                )
            )
            evaluation["recovery_des_models"][resource_jid] = deepcopy(
                recovery_des_model
            )
            evaluation["recovery_des_model_fingerprints"][resource_jid] = (
                recovery_des_model_fingerprint
            )
        except Exception as exc:  # noqa: BLE001 - validator transport must fail closed
            ra_finding = _unavailable_validation_finding(
                task=validated_task,
                validation_category=PHYSICAL_FEASIBILITY,
                validator_role="RA",
                constraint_code="resource_validation_unavailable",
                reason=str(exc),
            )
            validation_stages.append(
                recovery_validation_stage(
                    validation_category=PHYSICAL_FEASIBILITY,
                    validator_role="RA",
                    validator_jid=resource_jid,
                    status="unavailable",
                    findings=[ra_finding],
                    latency_ms=(time.perf_counter() - ra_started_at) * 1000.0,
                    state_fingerprint=state_fingerprint,
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

        ra_findings = [
            _shared.annotate_validation_finding(row)
            for row in (ra_result.get("findings") or [])
            if isinstance(row, dict)
        ]
        validation_stages.append(
            recovery_validation_stage(
                validation_category=PHYSICAL_FEASIBILITY,
                validator_role="RA",
                validator_jid=str(ra_reply.get("validator_jid") or resource_jid),
                status="passed" if bool(ra_result.get("allowed")) else "rejected",
                findings=ra_findings,
                request_id=str(ra_reply.get("request_id") or ""),
                latency_ms=ra_reply.get("latency_ms"),
                state_fingerprint=state_fingerprint,
                snapshot_fingerprint=str(ra_reply.get("snapshot_fingerprint") or ""),
                mocked=bool(ra_reply.get("mocked")),
            )
        )
        if ra_findings or not bool(ra_result.get("allowed")):
            validation_stages.append(
                _skipped_validation_stage(
                    category=SAFETY,
                    role="CCA",
                    jid=cca_jid,
                    mocked=bool(ra_reply.get("mocked")),
                )
            )
            evaluation["valid"] = False
            evaluation["validation_findings"] = deepcopy(ra_findings)
            evaluation["validation_stages"] = deepcopy(validation_stages)
            evaluation["failed_event_index"] = event_index
            return evaluation

        stale_findings = _ra_snapshot_staleness_findings(
            task=validated_task,
            snapshot=ra_snapshot,
            session_state=working_session_state,
        )
        if stale_findings:
            validation_stages.append(
                recovery_validation_stage(
                    validation_category=TRANSITION_FEASIBILITY,
                    validator_role="PA",
                    validator_jid=product_jid,
                    status="rejected",
                    findings=stale_findings,
                    state_fingerprint=state_fingerprint,
                    snapshot_fingerprint=str(
                        ra_reply.get("snapshot_fingerprint") or ""
                    ),
                    mocked=bool(ra_reply.get("mocked")),
                )
            )
            validation_stages.append(
                _skipped_validation_stage(category=SAFETY, role="CCA", jid=cca_jid)
            )
            evaluation["valid"] = False
            evaluation["validation_findings"] = deepcopy(stale_findings)
            evaluation["validation_stages"] = deepcopy(validation_stages)
            evaluation["failed_event_index"] = event_index
            return evaluation

        safety_input = build_recovery_safety_validation_input(
            task=validated_task,
            session_state=working_session_state,
            prepared_recovery_request=prepared_recovery_request,
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
        goal_recovery_events = _goal_relevant_recovery_event_rows(
            session_state=working_session_state,
            prepared_recovery_request=prepared_recovery_request,
            unresolved_condition_ids=_unresolved_condition_ids(
                session_state=working_session_state,
                prepared_recovery_request=prepared_recovery_request,
            ),
        )
        safety_input.setdefault("llm_input", {})["goal_recovery_events"] = [
            {
                "event_id": str(row.get("event_id") or ""),
                "task": deepcopy(row.get("task") or {}),
                "signature": deepcopy(row.get("signature") or {}),
            }
            for row in goal_recovery_events
            if str(row.get("event_id") or "")
        ]
        safety_request = {
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
                    "safety_input": safety_input,
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
                snapshot_fingerprint=str(
                    cca_reply.get("safety_rule_fingerprint") or ""
                ),
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

    session_state["status"] = (
        "ready_for_primitive_generation" if outline_complete else "paused_after_outline_turn"
    )
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


__all__ = [
    "_projected_outline_validation_context",
    "_handle_outline_single_pass",
    "_handle_outline_incremental",
    "_handle_outline_incremental_validated",
    "_handle_outline_incremental_candidates_validated",
    "_handle_outline_phase",
]
