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
    safety_condition_ids: list[str] | None = None,
) -> set[str]:
    unresolved = {
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
    unresolved.update(
        str(item).strip()
        for item in (
            safety_condition_ids
            if safety_condition_ids is not None
            else session_state.get("projected_safety_condition_identifiers") or []
        )
        if str(item).strip()
    )
    return unresolved


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
            "expected_start_state": deepcopy(row.get("expected_start_state") or {}),
            "expected_end_state": deepcopy(row.get("expected_end_state") or {}),
        }
        for row in surface_events
    ]
    return f"candidate_{recovery_validation_fingerprint(effects)[:16]}"


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
    projected["projected_safety_condition_identifiers"] = deepcopy(
        evaluation.get("remaining_safety_condition_identifiers") or []
    )
    return projected


def _apply_neurosymbolic_comparison(
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
    current_enabled_events = _admissible_recovery_enabled_event_ids(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        unresolved_condition_ids=current_unresolved,
    )
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
            safety_condition_ids=list(
                evaluation.get("remaining_safety_condition_identifiers") or []
            ),
        )
        cleared = current_unresolved - remaining
        introduced = remaining - current_unresolved
        enabled_events_after = _admissible_recovery_enabled_event_ids(
            session_state=projected_session,
            prepared_recovery_request=prepared_recovery_request,
            unresolved_condition_ids=remaining,
        )
        newly_enabled = enabled_events_after - current_enabled_events
        disabled_events = current_enabled_events - enabled_events_after
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
        evaluation["progressing"] = progressing_candidate
        evaluation["selection_status"] = (
            "eligible" if progressing_candidate else "excluded_no_progress"
        )
        if progressing_candidate:
            progressing.append(evaluation)

    nondominated: list[dict[str, Any]] = []
    for candidate in progressing:
        candidate_remaining = set(
            dict(candidate.get("selection_evidence") or {}).get(
                "open_recovery_obligation_ids_after"
            )
            or []
        )
        candidate_enabled_events = set(
            candidate.get("admissible_recovery_enabled_event_ids_after") or []
        )
        dominated_by: list[str] = []
        for other in progressing:
            if other is candidate:
                continue
            other_remaining = set(
                dict(other.get("selection_evidence") or {}).get(
                    "open_recovery_obligation_ids_after"
                )
                or []
            )
            other_enabled_events = set(
                other.get("admissible_recovery_enabled_event_ids_after") or []
            )
            if (
                other_remaining <= candidate_remaining
                and other_enabled_events >= candidate_enabled_events
                and (
                    other_remaining < candidate_remaining
                    or other_enabled_events > candidate_enabled_events
                )
            ):
                dominated_by.append(str(other.get("candidate_id") or ""))
        candidate["dominated_by_candidate_ids"] = sorted(
            token for token in dominated_by if token
        )
        if dominated_by:
            candidate["selection_status"] = "dominated"
        else:
            candidate["selection_status"] = "nondominated"
            nondominated.append(candidate)
    for successor_class_id, rows in candidates_by_successor_class.items():
        if len(rows) <= 1:
            continue
        for row in rows:
            row["equivalent_successor_class_id"] = successor_class_id
            if row in nondominated:
                row["selection_status"] = "equivalent_nondominated"
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


def _register_selection_revision(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    candidate_evaluations: list[dict[str, Any]],
) -> tuple[str, int, int]:
    revision_fingerprint = _selection_revision_fingerprint(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        candidate_evaluations=candidate_evaluations,
    )
    if (
        str(session_state.get("selection_revision_fingerprint") or "")
        == revision_fingerprint
    ):
        revision_count = int(session_state.get("selection_revision_count") or 0) + 1
    else:
        revision_count = 1
        if (
            str(session_state.get("active_selection_ambiguity_fingerprint") or "")
            != revision_fingerprint
        ):
            session_state["active_selection_ambiguity_feedback"] = {}
            session_state["active_selection_ambiguity_fingerprint"] = ""

    safety_rule_fingerprints, live_dfa_fingerprints = (
        _selection_revision_cca_fingerprints(
            session_state=session_state,
            candidate_evaluations=candidate_evaluations,
        )
    )
    session_state["selection_revision_fingerprint"] = revision_fingerprint
    session_state["selection_revision_count"] = revision_count
    session_state["selection_revision_safety_rule_fingerprints"] = deepcopy(
        safety_rule_fingerprints
    )
    session_state["selection_revision_live_dfa_fingerprints"] = deepcopy(
        live_dfa_fingerprints
    )
    revision_limit = max(
        1,
        int(session_state.get("selection_revision_limit") or 3),
    )
    return revision_fingerprint, revision_count, revision_limit


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
    cleared_safety_condition_identifiers: list[str] = []
    remaining_safety_condition_identifiers: list[str] = []
    safety_dfa_states_before: dict[str, str] = {}
    safety_dfa_states_after: dict[str, str] = deepcopy(
        working_session_state.get("projected_safety_dfa_states") or {}
    )

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

        pruned_row = _shared._matching_active_pruned_action(
            task=validated_task,
            session_state=working_session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        if pruned_row is not None:
            return reject_before_agent_validation(
                findings=[
                    _shared._retarget_candidate_finding_to_task(
                    dict(pruned_row.get("guard") or {}),
                    dict(validated_task or working_task),
                    )
                ],
                task=validated_task,
                failed_event_index=event_index,
            )

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
        active_safety_condition_ids = [
            str(row.get("condition_id") or row.get("id") or "").strip()
            for row in _shared._active_candidate_recovery_blockers(
                session_state=working_session_state,
                prepared_recovery_request=prepared_recovery_request,
            )
            if isinstance(row, dict)
            and str(row.get("condition_id") or row.get("id") or "").strip()
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
                    "active_safety_condition_ids": active_safety_condition_ids,
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
        cleared_safety_condition_identifiers.extend(
            str(item).strip()
            for item in (
                cca_result.get("cleared_safety_condition_identifiers") or []
            )
            if str(item).strip()
        )
        remaining_safety_condition_identifiers = [
            str(item).strip()
            for item in (
                cca_result.get("remaining_safety_condition_identifiers") or []
            )
            if str(item).strip()
        ]
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
        working_session_state["projected_safety_condition_identifiers"] = deepcopy(
            remaining_safety_condition_identifiers
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
    evaluation["cleared_safety_condition_identifiers"] = list(
        dict.fromkeys(cleared_safety_condition_identifiers)
    )
    evaluation["remaining_safety_condition_identifiers"] = deepcopy(
        remaining_safety_condition_identifiers
    )
    evaluation["safety_dfa_states_before"] = deepcopy(safety_dfa_states_before)
    evaluation["safety_dfa_states_after"] = deepcopy(safety_dfa_states_after)
    evaluation["safety_rule_fingerprint"] = str(
        working_session_state.get("projected_safety_rule_fingerprint") or ""
    ).strip()
    evaluation["live_safety_dfa_state_fingerprint"] = str(
        working_session_state.get("live_safety_dfa_state_fingerprint") or ""
    ).strip()
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
            revision_fingerprint, revision_count, revision_limit = (
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
            if revision_count >= revision_limit:
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
                            "The same plant and supervisor fingerprints produced "
                            "three unresolved revisions."
                        ),
                        "evidence": {
                            "selection_revision_count": revision_count,
                            "selection_revision_limit": revision_limit,
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
                model_finding = _shared.annotate_validation_finding(
                    {
                        "validation_category": "model_based_selection",
                        "constraint_owner": "product",
                        "constraint_family": "model_based_selection",
                        "constraint_code": "no_progressing_candidate",
                        "reason": (
                            "Validated candidates neither reduce open recovery "
                            "obligations nor enable a previously disabled "
                            "recovery-relevant RA event."
                        ),
                        "evidence": {"candidate_comparison": comparison_evidence},
                    }
                )

            feedback_rows = _shared._candidate_feedback_rows(
                candidate_evaluations
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

    feedback_rows = _shared._candidate_feedback_rows([selected] if selected else [])
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
        feedback_rows = _shared._candidate_feedback_rows(
            [{**dict(selected), "validation_findings": [finding]}]
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
    session_state["candidate_rejection_feedback"] = []
    session_state["selection_revision_count"] = 0
    session_state["selection_revision_fingerprint"] = ""
    session_state["selection_revision_safety_rule_fingerprints"] = []
    session_state["selection_revision_live_dfa_fingerprints"] = []
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
    session_state["projected_safety_condition_identifiers"] = deepcopy(
        selected.get("remaining_safety_condition_identifiers") or []
    )
    _shared._sync_des_recovery_aliases(
        session_state,
        turn_entry=turn_entry,
        transition_validation=turn_entry["transition_validation"],
    )

    for selected_event in selected_committed_events:
        _shared._apply_task_effects_to_symbolic_state(selected_event, session_state)
    session_state["pruned_actions"] = _shared._active_pruned_actions(
        session_state,
        prepared_recovery_request,
    )

    remaining_findings, remaining_conditions = _shared._remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    outline_complete = (
        remaining_findings == 0
        and remaining_conditions == 0
        and not list(selected.get("remaining_safety_condition_identifiers") or [])
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
