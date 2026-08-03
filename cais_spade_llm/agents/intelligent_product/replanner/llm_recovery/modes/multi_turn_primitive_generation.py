"""Primitive-generation helpers for the multi-turn recovery.

LLM-authored plan flow:

  The LLM authors ``primitive_steps`` directly in the event-local response format. There is
  no mechanical composer, no deterministic retrieval ranker, and no recipe
  constants. Context is pulled agentically: when the LLM sets
  ``decision = "need_context"`` with ``context_requests``, the recovery returns
  just those fields on the next turn. When ``decision = "primitive_steps_ready"``
  the authored ``primitive_steps`` are shape-validated and then run through the
  existing trace validator.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
    compact_multi_turn_runtime_session,
    write_recovery_artifacts,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_primitives import (
    validate_and_project_steps_with_trace,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_validation_service import (
    annotate_validation_finding,
)
from cais_spade_llm.resources.resource_primitives import (
    filter_synthesis_primitive_catalog,
)
from cais_spade_llm.resources.resource_profile import (
    get_resource_profile,
    resource_capability_decompositions,
)

from .multi_turn_outline_generation import _projected_outline_validation_context

_logger = logging.getLogger(__name__)

_MEMO_MAX_ENTRIES = 10
_LOG_FIELD_TRUNCATE = 200
_STUCK_GUARD_SAME_SIGNATURE_LIMIT = 5
_STUCK_GUARD_NO_PROGRESS_LIMIT = 6
_RESOURCE_OUTLINE_STATE_FIELDS = ("resource_state", "held_part", "resource_location")
_PART_OUTLINE_STATE_FIELDS = ("part_state", "part_location", "part_holder_resource_jid")


def _outline_resource_jid(outline_event: dict[str, Any] | None) -> str:
    if not isinstance(outline_event, dict):
        return ""
    return str(outline_event.get("resource_jid") or "").strip()


def _outline_part_name(outline_event: dict[str, Any] | None) -> str:
    if not isinstance(outline_event, dict):
        return ""
    return str(outline_event.get("part_name") or "").strip()


def _outline_target_ref(outline_event: dict[str, Any] | None) -> str:
    if not isinstance(outline_event, dict):
        return ""
    action_target = dict(outline_event.get("action_target") or {})
    end_state = dict(outline_event.get("expected_end_state") or {})
    return str(
        outline_event.get("target_ref")
        or action_target.get("target_ref")
        or action_target.get("target_location")
        or end_state.get("part_location")
        or ""
    ).strip()


def _outline_source_ref(outline_event: dict[str, Any] | None) -> str:
    if not isinstance(outline_event, dict):
        return ""
    action_target = dict(outline_event.get("action_target") or {})
    start_state = dict(outline_event.get("expected_start_state") or {})
    return str(
        outline_event.get("source_ref")
        or action_target.get("source_ref")
        or action_target.get("source_location")
        or start_state.get("part_location")
        or ""
    ).strip()


def _outline_description(outline_event: dict[str, Any] | None) -> str:
    if not isinstance(outline_event, dict):
        return ""
    return str(outline_event.get("description") or outline_event.get("rationale") or "").strip()


# ---------------------------------------------------------------------------
# Cursor / active-event helpers
# ---------------------------------------------------------------------------


def _active_primitive_outline_event(
    session_state: dict[str, Any],
) -> tuple[int, dict[str, Any] | None, list[dict[str, Any]]]:
    accepted_prefix = [
        dict(row)
        for row in (session_state.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    accepted_outline_ids = {
        str(row.get("outline_id") or "").strip()
        for row in (session_state.get("accepted_primitive_program") or [])
        if isinstance(row, dict) and str(row.get("outline_id") or "").strip()
    }
    cursor = max(0, int(session_state.get("primitive_generation_cursor") or 0))
    for index, row in enumerate(accepted_prefix):
        outline_id = str(row.get("outline_id") or "").strip()
        if outline_id and outline_id in accepted_outline_ids:
            continue
        cursor = index
        session_state["primitive_generation_cursor"] = cursor
        return cursor, deepcopy(row), accepted_prefix
    session_state["primitive_generation_cursor"] = len(accepted_prefix)
    return len(accepted_prefix), None, accepted_prefix


def _missing_primitive_outline_events(
    session_state: dict[str, Any],
) -> list[dict[str, Any]]:
    accepted_prefix = [
        dict(row)
        for row in (session_state.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    accepted_outline_ids = {
        str(row.get("outline_id") or "").strip()
        for row in (session_state.get("accepted_primitive_program") or [])
        if isinstance(row, dict) and str(row.get("outline_id") or "").strip()
    }
    missing: list[dict[str, Any]] = []
    for row in accepted_prefix:
        outline_id = str(row.get("outline_id") or "").strip()
        if outline_id and outline_id in accepted_outline_ids:
            continue
        missing.append(deepcopy(row))
    return missing


def _primitive_feedback_row(
    *,
    outline_event: dict[str, Any] | None,
    constraint_code: str,
    reason: str,
    finding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = dict(outline_event or {})
    row: dict[str, Any] = {
        "outline_id": str(event.get("outline_id") or "").strip(),
        "resource_jid": _outline_resource_jid(event),
        "part_name": _outline_part_name(event) or None,
        "constraint_code": str(constraint_code or "").strip() or "primitive_validation_failed",
        "reason": str(reason or "").strip() or "primitive validation failed",
    }
    if isinstance(finding, dict):
        if str(finding.get("constraint_code") or "").strip():
            row["constraint_code"] = str(finding.get("constraint_code") or "").strip()
        if str(finding.get("reason") or "").strip():
            row["reason"] = str(finding.get("reason") or "").strip()
        for key in (
            "validation_category",
            "constraint_owner",
            "constraint_family",
            "evidence",
            "guard",
            "failed_axes",
            "step_index",
            "primitive",
            "task_id",
        ):
            if key in finding and finding.get(key) not in (None, "", [], {}):
                row[key] = deepcopy(finding.get(key))
    return row


def _primitive_event_guard_key(
    *,
    cursor: int,
    outline_event: dict[str, Any] | None,
) -> dict[str, Any]:
    event = dict(outline_event or {})
    return {
        "cursor": int(cursor),
        "outline_id": str(event.get("outline_id") or "").strip(),
        "resource_jid": _outline_resource_jid(event),
    }


def _clear_primitive_event_guard(session_state: dict[str, Any]) -> None:
    session_state["primitive_event_guard"] = {}


def _reset_primitive_event_guard(
    session_state: dict[str, Any],
    *,
    cursor: int,
    outline_event: dict[str, Any] | None,
) -> None:
    session_state["primitive_event_guard"] = {
        "event_key": _primitive_event_guard_key(
            cursor=cursor,
            outline_event=outline_event,
        ),
        "no_progress_turns": 0,
        "last_signature": "",
        "same_signature_streak": 0,
        "feedback_history": [],
        "context_error_history": [],
        "input_diagnostic_history": [],
    }


def _primitive_no_progress_signature(
    *,
    cursor: int,
    outline_event: dict[str, Any] | None,
    response_decision: str,
    feedback_rows: list[dict[str, Any]] | None = None,
    context_errors: list[dict[str, Any]] | None = None,
) -> str:
    event_key = _primitive_event_guard_key(
        cursor=cursor,
        outline_event=outline_event,
    )
    feedback_signature = tuple(
        (
            str(row.get("constraint_code") or "").strip(),
            str(row.get("reason") or "").strip(),
        )
        for row in (feedback_rows or [])
        if isinstance(row, dict)
    )
    context_error_signature = tuple(
        (
            str(row.get("context_ref") or "").strip(),
            str(row.get("reason") or "").strip(),
        )
        for row in (context_errors or [])
        if isinstance(row, dict)
    )
    return repr(
        (
            event_key["cursor"],
            event_key["outline_id"],
            event_key["resource_jid"],
            str(response_decision or "").strip(),
            feedback_signature,
            context_error_signature,
        )
    )


def _record_primitive_no_progress(
    session_state: dict[str, Any],
    *,
    cursor: int,
    outline_event: dict[str, Any] | None,
    response_decision: str,
    feedback_rows: list[dict[str, Any]] | None = None,
    context_errors: list[dict[str, Any]] | None = None,
    input_diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    event_key = _primitive_event_guard_key(
        cursor=cursor,
        outline_event=outline_event,
    )
    prior_guard = dict(session_state.get("primitive_event_guard") or {})
    prior_event_key = dict(prior_guard.get("event_key") or {})
    same_event = prior_event_key == event_key
    signature = _primitive_no_progress_signature(
        cursor=cursor,
        outline_event=outline_event,
        response_decision=response_decision,
        feedback_rows=feedback_rows,
        context_errors=context_errors,
    )
    same_signature = same_event and str(prior_guard.get("last_signature") or "") == signature
    updated_guard = {
        "event_key": event_key,
        "no_progress_turns": (
            int(prior_guard.get("no_progress_turns") or 0) + 1 if same_event else 1
        ),
        "last_signature": signature,
        "same_signature_streak": (
            int(prior_guard.get("same_signature_streak") or 0) + 1 if same_signature else 1
        ),
        "feedback_history": (list(prior_guard.get("feedback_history") or []) if same_event else [])
        + [deepcopy(row) for row in (feedback_rows or []) if isinstance(row, dict)],
        "context_error_history": (
            list(prior_guard.get("context_error_history") or []) if same_event else []
        )
        + [deepcopy(row) for row in (context_errors or []) if isinstance(row, dict)],
        "input_diagnostic_history": (
            list(prior_guard.get("input_diagnostic_history") or []) if same_event else []
        )
        + [deepcopy(row) for row in (input_diagnostics or []) if isinstance(row, dict)],
    }
    session_state["primitive_event_guard"] = deepcopy(updated_guard)
    return updated_guard


def _primitive_event_should_escalate(guard_state: dict[str, Any]) -> bool:
    return (
        int(guard_state.get("same_signature_streak") or 0) >= _STUCK_GUARD_SAME_SIGNATURE_LIMIT
        or int(guard_state.get("no_progress_turns") or 0) >= _STUCK_GUARD_NO_PROGRESS_LIMIT
    )


def _primitive_rows_summary(rows: list[dict[str, Any]] | None) -> str:
    items: list[str] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        label = str(
            row.get("constraint_code") or row.get("context_ref") or row.get("reason") or ""
        ).strip()
        reason = str(row.get("reason") or "").strip()
        if label and reason and label != reason:
            items.append(f"{label}: {reason}")
        elif reason:
            items.append(reason)
        elif label:
            items.append(label)
    return "; ".join(items) if items else ""


def _count_named_rows(
    rows: list[dict[str, Any]] | None,
    *,
    key: str,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        token = str(row.get(key) or "").strip()
        if token:
            counts[token] += 1
    return counts


def _counter_summary(counter: Counter[str], *, limit: int = 3) -> str:
    parts = [f"{token} x{count}" for token, count in counter.most_common(limit) if token]
    return ", ".join(parts)


def _primitive_event_stuck_reason(
    *,
    guard_state: dict[str, Any],
    feedback_rows: list[dict[str, Any]] | None = None,
    context_errors: list[dict[str, Any]] | None = None,
    input_diagnostics: list[dict[str, Any]] | None = None,
) -> str:
    reason_parts = [
        (
            "primitive authoring stalled on the active event after "
            f"{int(guard_state.get('no_progress_turns') or 0)} no-progress turns "
            f"(same-signature streak={int(guard_state.get('same_signature_streak') or 0)})"
        )
    ]
    feedback_history = list(guard_state.get("feedback_history") or [])
    if not feedback_history:
        feedback_history = [dict(row) for row in (feedback_rows or []) if isinstance(row, dict)]
    feedback_counts = _count_named_rows(feedback_history, key="constraint_code")
    if feedback_counts:
        reason_parts.append(f"repeated validator blockers: {_counter_summary(feedback_counts)}")
    elif feedback_rows:
        feedback_summary = _primitive_rows_summary(feedback_rows)
        if feedback_summary:
            reason_parts.append(f"last validator feedback: {feedback_summary}")
    context_error_history = list(guard_state.get("context_error_history") or [])
    if not context_error_history:
        context_error_history = [
            dict(row) for row in (context_errors or []) if isinstance(row, dict)
        ]
    context_error_counts = _count_named_rows(context_error_history, key="context_ref")
    if context_error_counts:
        reason_parts.append(
            f"repeated context request errors: {_counter_summary(context_error_counts)}"
        )
    input_diagnostic_history = list(guard_state.get("input_diagnostic_history") or [])
    if not input_diagnostic_history:
        input_diagnostic_history = [
            dict(row) for row in (input_diagnostics or []) if isinstance(row, dict)
        ]
    if input_diagnostic_history:
        input_summary = _primitive_rows_summary(input_diagnostic_history[:2])
        if input_summary:
            reason_parts.append(f"input diagnostics: {input_summary}")
    return ". ".join(reason_parts)


def _record_primitive_escalation_diagnostic(
    session_state: dict[str, Any],
    *,
    cursor: int,
    outline_event: dict[str, Any] | None,
    guard_state: dict[str, Any],
    feedback_rows: list[dict[str, Any]] | None = None,
    context_errors: list[dict[str, Any]] | None = None,
    input_diagnostics: list[dict[str, Any]] | None = None,
    reason: str,
) -> list[dict[str, Any]]:
    event = dict(outline_event or {})
    trigger = (
        "same_signature_streak"
        if int(guard_state.get("same_signature_streak") or 0) >= _STUCK_GUARD_SAME_SIGNATURE_LIMIT
        else "no_progress_turns"
    )
    diagnostics = [
        {
            "outline_id": str(event.get("outline_id") or "").strip(),
            "resource_jid": _outline_resource_jid(event),
            "part_name": _outline_part_name(event) or None,
            "cursor": int(cursor),
            "trigger": trigger,
            "no_progress_turns": int(guard_state.get("no_progress_turns") or 0),
            "same_signature_streak": int(guard_state.get("same_signature_streak") or 0),
            "reason": str(reason or "").strip(),
            "repeated_validator_blockers": [
                {"constraint_code": code, "count": count}
                for code, count in _count_named_rows(
                    list(guard_state.get("feedback_history") or []),
                    key="constraint_code",
                ).most_common()
            ],
            "repeated_context_errors": [
                {"context_ref": ref, "count": count}
                for ref, count in _count_named_rows(
                    list(guard_state.get("context_error_history") or []),
                    key="context_ref",
                ).most_common()
            ],
            "input_diagnostics": deepcopy(
                [
                    dict(row)
                    for row in (
                        list(guard_state.get("input_diagnostic_history") or [])
                        or list(input_diagnostics or [])
                    )
                    if isinstance(row, dict)
                ]
            ),
            "last_validator_feedback": deepcopy(
                [dict(row) for row in (feedback_rows or []) if isinstance(row, dict)]
            ),
            "last_context_errors": deepcopy(
                [dict(row) for row in (context_errors or []) if isinstance(row, dict)]
            ),
        }
    ]
    session_state["primitive_escalation_diagnostics"] = deepcopy(diagnostics)
    return diagnostics


def _capability_decomposition_names_for_resource(
    *,
    prepared_recovery_request: dict[str, Any],
    resource_jid: str,
) -> list[str]:
    payload = _capability_decomposition_for_resource(
        prepared_recovery_request=prepared_recovery_request,
        resource_jid=resource_jid,
        function_name="",
    )
    if not isinstance(payload, dict):
        return []
    return sorted(
        name for name, row in payload.items() if str(name).strip() and isinstance(row, dict)
    )


# ---------------------------------------------------------------------------
# Catalog / snapshot / grounding helpers
# ---------------------------------------------------------------------------


def _primitive_catalog_for_resource(
    prepared_recovery_request: dict[str, Any],
    resource_jid: str,
) -> list[dict[str, Any]]:
    recovery_resources = dict(prepared_recovery_request.get("recovery_resources") or {})
    recovery_entry = dict(recovery_resources.get(resource_jid) or {})
    return filter_synthesis_primitive_catalog(
        [
            dict(row)
            for row in (recovery_entry.get("primitive_catalog") or [])
            if isinstance(row, dict)
        ]
    )


def _resource_type_for_resource(
    prepared_recovery_request: dict[str, Any],
    resource_jid: str,
) -> str:
    recovery_entry = _resource_recovery_entry(prepared_recovery_request, resource_jid)
    recovery_snapshot = dict(recovery_entry.get("recovery_snapshot") or {})
    resource_core = dict(recovery_snapshot.get("resource_core") or {})
    return (
        str(
            recovery_entry.get("resource_type")
            or recovery_snapshot.get("resource_type")
            or resource_core.get("resource_type")
            or ""
        )
        .strip()
        .lower()
    )


def _capability_decomposition_for_resource(
    *,
    prepared_recovery_request: dict[str, Any],
    resource_jid: str,
    function_name: str = "",
) -> Any:
    resource_type = _resource_type_for_resource(prepared_recovery_request, resource_jid)
    profile = get_resource_profile(resource_type or "resource")
    return resource_capability_decompositions(
        profile,
        function_name=function_name,
        primitive_catalog=_primitive_catalog_for_resource(
            prepared_recovery_request,
            resource_jid,
        ),
        resource_jid=resource_jid,
    )


def _resource_recovery_entry(
    prepared_recovery_request: dict[str, Any],
    resource_jid: str,
) -> dict[str, Any]:
    return dict(dict(prepared_recovery_request.get("recovery_resources") or {}).get(resource_jid) or {})


def _resource_snapshot_with_caps(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    resource_jid: str,
    projected_resources_by_jid: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if projected_resources_by_jid is None:
        resources_by_jid, _ = _projected_outline_validation_context(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
    else:
        resources_by_jid = deepcopy(projected_resources_by_jid)
    recovery_entry = _resource_recovery_entry(prepared_recovery_request, resource_jid)
    recovery_snapshot = dict(recovery_entry.get("recovery_snapshot") or {})
    static_capabilities = dict(recovery_entry.get("static_capabilities") or {})
    snapshot = deepcopy(dict(resources_by_jid.get(resource_jid) or {}))
    snapshot.setdefault("resource_jid", resource_jid)
    for key in (
        "resource_type",
        "current_pose",
        "current_pose_ref",
        "workspace_bounds",
        "named_poses",
        "available_named_poses",
        "reachability",
        "staging_areas",
        "static_capabilities",
    ):
        if snapshot.get(key) not in (None, "", [], {}):
            continue
        value = recovery_entry.get(key)
        if value in (None, "", [], {}):
            value = recovery_snapshot.get(key)
        if value in (None, "", [], {}):
            value = static_capabilities.get(key)
        if value not in (None, "", [], {}):
            snapshot[key] = deepcopy(value)
    if static_capabilities:
        snapshot.setdefault("static_capabilities", deepcopy(static_capabilities))
    return snapshot


def _apply_entity_scoped_outline_state(
    *,
    resource_jid: str,
    part_name: str,
    outline_state: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    outline_id: str = "",
    fact_producers: dict[tuple[str, str, str], str | None] | None = None,
) -> None:
    if resource_jid:
        resource_row = dict(resources_by_jid.get(resource_jid) or {"resource_jid": resource_jid})
        if "resource_state" in outline_state:
            value = deepcopy(outline_state.get("resource_state"))
            resource_row["resource_state"] = value
            resource_row["current_state"] = value
            if fact_producers is not None:
                fact_producers[("resource", resource_jid, "resource_state")] = outline_id or None
        if "held_part" in outline_state:
            value = deepcopy(outline_state.get("held_part"))
            resource_row["held_part"] = value
            if fact_producers is not None:
                fact_producers[("resource", resource_jid, "held_part")] = outline_id or None
        if "resource_location" in outline_state:
            value = deepcopy(outline_state.get("resource_location"))
            resource_row["resource_location"] = value
            resource_row["current_location"] = value
            if fact_producers is not None:
                fact_producers[("resource", resource_jid, "resource_location")] = outline_id or None
        resources_by_jid[resource_jid] = deepcopy(resource_row)

    if part_name:
        part_row = dict(parts_by_name.get(part_name) or {"part_name": part_name})
        if "part_state" in outline_state:
            value = deepcopy(outline_state.get("part_state"))
            part_row["part_state"] = value
            part_row["current_state"] = value
            if fact_producers is not None:
                fact_producers[("part", part_name, "part_state")] = outline_id or None
        if "part_location" in outline_state:
            value = deepcopy(outline_state.get("part_location"))
            part_row["part_location"] = value
            part_row["current_location"] = value
            if fact_producers is not None:
                fact_producers[("part", part_name, "part_location")] = outline_id or None
        if "part_holder_resource_jid" in outline_state:
            value = deepcopy(outline_state.get("part_holder_resource_jid"))
            part_row["part_holder_resource_jid"] = value
            part_row["current_holder_resource_jid"] = value
            if fact_producers is not None:
                fact_producers[("part", part_name, "part_holder_resource_jid")] = outline_id or None
        parts_by_name[part_name] = deepcopy(part_row)


def _primitive_propagated_outline_entities(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[tuple[str, str, str], str | None],
]:
    llm_input = dict(prepared_recovery_request.get("llm_input") or {})
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})

    resources_by_jid: dict[str, dict[str, Any]] = {}
    for row in observed_runtime_state.get("resources") or []:
        if not isinstance(row, dict):
            continue
        resource_jid = str(row.get("resource_jid") or "").strip()
        if resource_jid:
            resources_by_jid[resource_jid] = deepcopy(row)
    for resource_jid, row in dict(session_state.get("base_symbolic_resources") or {}).items():
        token = str(resource_jid or "").strip()
        if token and isinstance(row, dict):
            resources_by_jid[token] = deepcopy(row)

    parts_by_name: dict[str, dict[str, Any]] = {}
    for row in llm_input.get("part_facts") or []:
        if not isinstance(row, dict):
            continue
        part_name = str(row.get("part_name") or "").strip()
        if part_name:
            parts_by_name[part_name] = deepcopy(row)
    for part_name, row in dict(session_state.get("base_symbolic_parts") or {}).items():
        token = str(part_name or "").strip()
        if token and isinstance(row, dict):
            parts_by_name[token] = deepcopy(row)

    for entry in dict(session_state.get("observation_store") or {}).values():
        if not isinstance(entry, dict):
            continue
        part_name = str(entry.get("part_name") or "").strip()
        if not part_name:
            continue
        part_row = parts_by_name.setdefault(part_name, {"part_name": part_name})
        pose = dict(entry.get("pose") or {})
        if not pose and entry.get("x") is not None:
            pose = {"x": entry.get("x"), "y": entry.get("y"), "z": entry.get("z")}
        if pose:
            part_row["observed_pose"] = deepcopy(pose)
    fact_producers: dict[tuple[str, str, str], str | None] = {}
    for row in session_state.get("accepted_primitive_program") or []:
        if not isinstance(row, dict):
            continue
        _apply_entity_scoped_outline_state(
            resource_jid=str(row.get("resource_jid") or "").strip(),
            part_name=str(row.get("part_name") or "").strip(),
            outline_state=dict(row.get("projected_outline_state") or {}),
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
            outline_id=str(row.get("outline_id") or "").strip(),
            fact_producers=fact_producers,
        )
    return resources_by_jid, parts_by_name, fact_producers


def _primitive_current_outline_state(
    *,
    outline_event: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def _field_value(row: dict[str, Any], field_name: str) -> tuple[bool, Any]:
        alias_map = {
            "resource_state": ("resource_state", "current_state", "state"),
            "resource_location": ("resource_location", "current_location", "location"),
            "held_part": ("held_part",),
            "part_state": ("part_state", "current_state", "state"),
            "part_location": ("part_location", "current_location", "location", "current_pose_ref"),
            "part_holder_resource_jid": (
                "part_holder_resource_jid",
                "current_holder_resource_jid",
            ),
        }
        for key in alias_map.get(field_name, (field_name,)):
            if key in row:
                return True, deepcopy(row.get(key))
        return False, None

    expected_start = dict(outline_event.get("expected_start_state") or {})
    if not expected_start:
        return {}
    resource_jid = _outline_resource_jid(outline_event)
    part_name = _outline_part_name(outline_event)
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    part_row = dict(parts_by_name.get(part_name) or {})

    current_outline_state: dict[str, Any] = {}
    for field_name in _RESOURCE_OUTLINE_STATE_FIELDS:
        if field_name in expected_start:
            has_value, value = _field_value(resource_row, field_name)
            if has_value:
                current_outline_state[field_name] = value
    for field_name in _PART_OUTLINE_STATE_FIELDS:
        if field_name in expected_start:
            has_value, value = _field_value(part_row, field_name)
            if has_value:
                current_outline_state[field_name] = value
    return current_outline_state


def _previous_projected_snapshot_for_resource(
    session_state: dict[str, Any],
    resource_jid: str,
) -> dict[str, Any] | None:
    for row in reversed(list(session_state.get("accepted_primitive_program") or [])):
        if not isinstance(row, dict):
            continue
        if str(row.get("resource_jid") or "").strip() != resource_jid:
            continue
        projected_snapshot = dict(row.get("projected_snapshot") or {})
        if projected_snapshot:
            return deepcopy(projected_snapshot)
    return None


def _primitive_start_state_conflict_diagnostic(
    *,
    outline_event: dict[str, Any],
    current_outline_state: dict[str, Any],
    fact_producers: dict[tuple[str, str, str], str | None],
) -> dict[str, Any] | None:
    expected_start = dict(outline_event.get("expected_start_state") or {})
    if not expected_start:
        return None

    mismatches: list[dict[str, Any]] = []
    resource_jid = _outline_resource_jid(outline_event)
    part_name = _outline_part_name(outline_event)
    compared_fields: list[str] = []
    for field_name, expected_value in expected_start.items():
        fact_key: tuple[str, str, str] | None = None
        if field_name in _RESOURCE_OUTLINE_STATE_FIELDS and resource_jid:
            fact_key = ("resource", resource_jid, field_name)
        elif field_name in _PART_OUTLINE_STATE_FIELDS and part_name:
            fact_key = ("part", part_name, field_name)
        if fact_key is None or fact_key not in fact_producers:
            continue
        compared_fields.append(field_name)

        if field_name not in current_outline_state:
            mismatches.append(
                {
                    "field": field_name,
                    "entity_scope": fact_key[0] if fact_key else None,
                    "entity_id": fact_key[1] if fact_key else None,
                    "expected": deepcopy(expected_value),
                    "actual": None,
                    "actual_unavailable": True,
                    "previous_outline_id": fact_producers.get(fact_key) if fact_key else None,
                }
            )
            continue
        actual_value = current_outline_state.get(field_name)
        if actual_value != expected_value:
            mismatches.append(
                {
                    "field": field_name,
                    "entity_scope": fact_key[0] if fact_key else None,
                    "entity_id": fact_key[1] if fact_key else None,
                    "expected": deepcopy(expected_value),
                    "actual": deepcopy(actual_value),
                    "actual_unavailable": False,
                    "previous_outline_id": fact_producers.get(fact_key) if fact_key else None,
                }
            )
    if not mismatches:
        return None

    mismatch_text = ", ".join(
        (
            f"{row['field']} expected={row['expected']!r} actual=unavailable"
            if row.get("actual_unavailable")
            else f"{row['field']} expected={row['expected']!r} actual={row['actual']!r}"
        )
        for row in mismatches
    )
    prior_outline_ids = sorted(
        {
            str(row.get("previous_outline_id") or "").strip()
            for row in mismatches
            if str(row.get("previous_outline_id") or "").strip()
        }
    )
    return {
        "outline_id": str(outline_event.get("outline_id") or "").strip(),
        "resource_jid": resource_jid,
        "part_name": part_name or None,
        "constraint_code": "primitive_start_state_contradiction",
        "reason": (
            "accepted primitive prefix contradicts the active event's exact "
            f"expected_start_state: {mismatch_text}"
        ),
        "evidence": {
            "expected_start_state": {
                field_name: deepcopy(expected_start.get(field_name))
                for field_name in compared_fields
            },
            "current_propagated_outline_state": {
                field_name: deepcopy(current_outline_state.get(field_name))
                for field_name in compared_fields
                if field_name in current_outline_state
            },
            "mismatches": deepcopy(mismatches),
            "previous_outline_id": prior_outline_ids[0] if len(prior_outline_ids) == 1 else None,
            "previous_outline_ids": prior_outline_ids,
        },
    }


def _apply_exact_start_to_executor_resource_snapshot(
    *,
    snapshot: dict[str, Any],
    expected_start: dict[str, Any],
) -> dict[str, Any]:
    projected = deepcopy(dict(snapshot or {}))
    field_map = {
        "resource_state": "current_state",
        "resource_location": "current_location",
        "held_part": "held_part",
    }
    for expected_key, snapshot_key in field_map.items():
        if expected_key in expected_start:
            projected[snapshot_key] = deepcopy(expected_start.get(expected_key))
    if "held_part" in expected_start:
        projected["gripper_state"] = (
            "closed" if expected_start.get("held_part") not in (None, "") else "open"
        )
    return projected


def _build_literal_resource_start_snapshot(
    *,
    base_snapshot: dict[str, Any],
    exact_state: dict[str, Any],
) -> dict[str, Any]:
    literal = deepcopy(dict(base_snapshot or {}))
    for field_name in ("resource_state", "resource_location", "held_part"):
        if field_name in exact_state:
            literal[field_name] = deepcopy(exact_state.get(field_name))
    if "resource_state" in exact_state:
        literal.pop("current_state", None)
    if "resource_location" in exact_state:
        literal.pop("current_location", None)
    if "held_part" in literal:
        literal.pop("gripper_state", None)
    return literal


def _build_literal_part_start_snapshot(
    *,
    base_snapshot: dict[str, Any],
    outline_event: dict[str, Any],
    part_name: str,
) -> dict[str, Any]:
    literal = deepcopy(dict(base_snapshot or {}))
    if part_name != _outline_part_name(outline_event):
        return literal
    expected_start = dict(outline_event.get("expected_start_state") or {})
    if "part_state" in expected_start:
        literal["part_state"] = deepcopy(expected_start.get("part_state"))
        literal.pop("current_state", None)
    if "part_location" in expected_start:
        literal["part_location"] = deepcopy(expected_start.get("part_location"))
        literal.pop("current_location", None)
        literal.pop("current_holder_resource_jid", None)
    return literal


def _apply_exact_start_to_executor_part_snapshot(
    *,
    snapshot: dict[str, Any],
    outline_event: dict[str, Any],
) -> dict[str, Any]:
    projected = deepcopy(dict(snapshot or {}))
    expected_start = dict(outline_event.get("expected_start_state") or {})
    if "part_state" in expected_start:
        projected["part_state"] = deepcopy(expected_start.get("part_state"))
        projected["current_state"] = deepcopy(expected_start.get("part_state"))
    if "part_location" in expected_start:
        projected["part_location"] = deepcopy(expected_start.get("part_location"))
        projected["current_location"] = deepcopy(expected_start.get("part_location"))
    part_name = _outline_part_name(outline_event)
    if part_name and "held_part" in expected_start:
        holder_jid = (
            _outline_resource_jid(outline_event)
            if expected_start.get("held_part") == part_name
            else None
        )
        projected["current_holder_resource_jid"] = holder_jid
        projected["part_holder_resource_jid"] = holder_jid
    return projected


def _primitive_resource_start_views(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    outline_event: dict[str, Any],
) -> dict[str, Any]:
    resource_jid = _outline_resource_jid(outline_event)
    propagated_resources, propagated_parts, fact_producers = _primitive_propagated_outline_entities(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    previous_projected_snapshot = dict(
        _previous_projected_snapshot_for_resource(session_state, resource_jid) or {}
    )
    if previous_projected_snapshot:
        base_snapshot = deepcopy(previous_projected_snapshot)
        base_snapshot.setdefault("resource_jid", resource_jid)
    else:
        base_snapshot = _resource_snapshot_with_caps(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
            resource_jid=resource_jid,
            projected_resources_by_jid=propagated_resources,
        )
    current_outline_state = _primitive_current_outline_state(
        outline_event=outline_event,
        resources_by_jid=propagated_resources,
        parts_by_name=propagated_parts,
    )
    contradiction = _primitive_start_state_conflict_diagnostic(
        outline_event=outline_event,
        current_outline_state=current_outline_state,
        fact_producers=fact_producers,
    )
    expected_start = dict(outline_event.get("expected_start_state") or {})
    prompt_exact_state = (
        deepcopy(current_outline_state) if contradiction else deepcopy(expected_start)
    )
    prompt_snapshot = _build_literal_resource_start_snapshot(
        base_snapshot=base_snapshot,
        exact_state=prompt_exact_state,
    )
    executor_snapshot = deepcopy(base_snapshot)
    if contradiction is None:
        executor_snapshot = _apply_exact_start_to_executor_resource_snapshot(
            snapshot=executor_snapshot,
            expected_start=expected_start,
        )
    return {
        "executor_snapshot": executor_snapshot,
        "prompt_snapshot": prompt_snapshot,
        "diagnostics": [deepcopy(contradiction)] if contradiction else [],
        "propagated_resources_by_jid": deepcopy(propagated_resources),
        "propagated_parts_by_name": deepcopy(propagated_parts),
        "current_outline_state": deepcopy(current_outline_state),
    }


def _primitive_start_snapshot(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    outline_event: dict[str, Any],
) -> dict[str, Any]:
    return deepcopy(
        dict(
            _primitive_resource_start_views(
                session_state=session_state,
                prepared_recovery_request=prepared_recovery_request,
                outline_event=outline_event,
            ).get("executor_snapshot")
            or {}
        )
    )


def _merge_grounded_part_context(
    *,
    prepared_recovery_request: dict[str, Any],
    parts_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    parts_root = deepcopy(
        dict(dict(prepared_recovery_request.get("grounding_context") or {}).get("parts") or {})
    )
    for part_name, row in parts_by_name.items():
        token = str(part_name or "").strip()
        if not token:
            continue
        entry = parts_root.setdefault(token, {})
        for key, value in dict(row or {}).items():
            if key not in entry or entry.get(key) in (None, "", [], {}):
                entry[key] = deepcopy(value)
    return parts_root


def _merge_grounded_resource_context(
    *,
    prepared_recovery_request: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    resources_root: dict[str, Any] = {}
    recovery_resources = dict(prepared_recovery_request.get("recovery_resources") or {})
    for resource_jid, row in resources_by_jid.items():
        token = str(resource_jid or "").strip()
        if not token:
            continue
        entry = deepcopy(dict(row or {}))
        recovery_entry = dict(recovery_resources.get(token) or {})
        recovery_snapshot = dict(recovery_entry.get("recovery_snapshot") or {})
        static_capabilities = dict(recovery_entry.get("static_capabilities") or {})
        for source in (recovery_snapshot, static_capabilities):
            for key, value in source.items():
                if key not in entry or entry.get(key) in (None, "", [], {}):
                    entry[key] = deepcopy(value)
        resources_root[token] = entry
    return resources_root


def _primitive_grounding_context(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    outline_event: dict[str, Any],
) -> dict[str, Any]:
    start_state_views = _primitive_resource_start_views(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        outline_event=outline_event,
    )
    resources_by_jid = deepcopy(start_state_views.get("propagated_resources_by_jid") or {})
    parts_by_name = deepcopy(start_state_views.get("propagated_parts_by_name") or {})
    resource_jid = _outline_resource_jid(outline_event)
    part_name = _outline_part_name(outline_event)
    if resource_jid:
        resources_by_jid[resource_jid] = deepcopy(
            dict(start_state_views.get("executor_snapshot") or {})
        )
    if part_name:
        part_snapshot = dict(parts_by_name.get(part_name) or {"part_name": part_name})
        if start_state_views.get("diagnostics"):
            parts_by_name[part_name] = deepcopy(part_snapshot)
        else:
            parts_by_name[part_name] = _apply_exact_start_to_executor_part_snapshot(
                snapshot=part_snapshot,
                outline_event=outline_event,
            )
    parts_root = _merge_grounded_part_context(
        prepared_recovery_request=prepared_recovery_request,
        parts_by_name=parts_by_name,
    )
    resources_root = _merge_grounded_resource_context(
        prepared_recovery_request=prepared_recovery_request,
        resources_by_jid=resources_by_jid,
    )
    active_part_root = dict(parts_root.get(part_name) or {})
    active_part_row = deepcopy(dict(parts_by_name.get(part_name) or {}))
    for key, value in active_part_root.items():
        if key not in active_part_row or active_part_row.get(key) in (None, "", [], {}):
            active_part_row[key] = deepcopy(value)
    return {
        "active_outline_event": deepcopy(outline_event),
        "resource": deepcopy(resources_by_jid.get(resource_jid) or {}),
        "part": active_part_row if part_name else {},
        "resources_by_jid": deepcopy(resources_by_jid),
        "parts_by_name": deepcopy(parts_by_name),
        "resources": resources_root,
        "parts": parts_root,
        "observation_store": deepcopy(session_state.get("observation_store") or {}),
    }


def _projected_outline_state(
    projected_snapshot: dict[str, Any],
    outline_event: dict[str, Any],
) -> dict[str, Any]:
    del projected_snapshot
    expected_end = dict(outline_event.get("expected_end_state") or {})
    projected_outline: dict[str, Any] = {}
    for field_name in _RESOURCE_OUTLINE_STATE_FIELDS + _PART_OUTLINE_STATE_FIELDS:
        if field_name in expected_end:
            projected_outline[field_name] = deepcopy(expected_end.get(field_name))
    return projected_outline


def _active_source_token(outline_event: dict[str, Any]) -> str:
    action_target = dict(outline_event.get("action_target") or {})
    return str(
        _outline_source_ref(outline_event)
        or action_target.get("source_location")
        or dict(outline_event.get("expected_start_state") or {}).get("part_location")
        or ""
    ).strip()


def _active_destination_token(outline_event: dict[str, Any]) -> str:
    action_target = dict(outline_event.get("action_target") or {})
    candidate = str(
        _outline_target_ref(outline_event)
        or action_target.get("target_location")
        or dict(outline_event.get("expected_end_state") or {}).get("part_location")
        or ""
    ).strip()
    if candidate == _outline_resource_jid(outline_event):
        return ""
    return candidate


def _named_pose_names_only(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, dict):
        raw_names = value.keys()
    else:
        raw_names = value or []
    for raw_name in raw_names:
        name = str(raw_name or "").strip()
        if not name or name in names:
            continue
        names.append(name)
    return names


def _active_resource_named_pose_names(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    resource_jid: str,
) -> list[str]:
    snapshot = _resource_snapshot_with_caps(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        resource_jid=resource_jid,
    )
    static_capabilities = dict(snapshot.get("static_capabilities") or {})
    for source in (
        snapshot.get("named_poses"),
        snapshot.get("available_named_poses"),
        static_capabilities.get("named_poses"),
        static_capabilities.get("available_named_poses"),
    ):
        names = _named_pose_names_only(source)
        if names:
            return names
    return []


# ---------------------------------------------------------------------------
# Resource-specific validator wiring (kept; used after authored plan passes
# the trace validator)
# ---------------------------------------------------------------------------


def _primitive_resource_sequence_findings(
    *,
    outline_event: dict[str, Any],
    primitive_steps: list[dict[str, Any]],
    trace_metadata: dict[str, Any],
    start_snapshot: dict[str, Any],
    projected_snapshot: dict[str, Any],
    grounding_context: dict[str, Any],
    primitive_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    resource_type = str(
        dict(start_snapshot.get("resource_core") or {}).get("resource_type")
        or start_snapshot.get("resource_type")
        or ""
    ).strip()
    profile = get_resource_profile(resource_type or "resource")
    validator = getattr(profile, "primitive_sequence_validator", None)
    if not callable(validator):
        return []
    try:
        raw_findings = validator(
            outline_event=deepcopy(outline_event),
            primitive_steps=deepcopy(primitive_steps),
            trace_metadata=deepcopy(trace_metadata),
            start_snapshot=deepcopy(start_snapshot),
            projected_snapshot=deepcopy(projected_snapshot),
            grounding_context=deepcopy(grounding_context),
            primitive_catalog=deepcopy(primitive_catalog),
        )
    except Exception as exc:
        return [
            annotate_validation_finding(
                {
                    "outline_id": str(outline_event.get("outline_id") or "").strip(),
                    "resource_jid": _outline_resource_jid(outline_event),
                    "part_name": _outline_part_name(outline_event) or None,
                    "constraint_owner": "resource",
                    "constraint_family": "primitive_sequence",
                    "constraint_code": "primitive_sequence_validator_error",
                    "reason": f"resource primitive sequence validator failed: {exc}",
                }
            )
        ]
    return [
        annotate_validation_finding(dict(row))
        for row in (raw_findings or [])
        if isinstance(row, dict)
    ]


# ---------------------------------------------------------------------------
# Agentic context retrieval
#
# The LLM sets ``decision = "need_context"`` and lists refs in
# ``context_requests``. We resolve each ref against the live runtime state and
# stash the result in session_state for the next prompt turn.
# ---------------------------------------------------------------------------


def _visible_primitive_catalog_card(
    primitive_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    card: list[dict[str, Any]] = []
    for entry in primitive_catalog or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        card.append(
            {
                "name": name,
                "description": str(
                    entry.get("description") or entry.get("semantic_summary") or ""
                ).strip(),
                "params": deepcopy(entry.get("params") or {}),
                "required_params": [
                    str(item).strip()
                    for item in (entry.get("required_params") or [])
                    if str(item).strip()
                ],
                "preconditions": deepcopy(entry.get("preconditions") or {}),
                "effects": deepcopy(entry.get("effects") or {}),
                "output_schema": deepcopy(entry.get("output_schema") or {}),
                "primitive_kind": str(entry.get("primitive_kind") or "").strip() or None,
            }
        )
    return card


def _visible_primitive_catalog_names(
    primitive_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Minimal catalog view: just ``{name, primitive_kind}`` per entry.

    Full contract cards (params / preconditions / effects / output_schema) are
    retrieved on demand via ``/primitive_contracts/<name>``.
    """
    rows: list[dict[str, Any]] = []
    for entry in primitive_catalog or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        kind = str(entry.get("primitive_kind") or "").strip() or None
        rows.append({"name": name, "primitive_kind": kind})
    return rows


def _compact_active_event_token(outline_event: dict[str, Any] | None) -> dict[str, Any]:
    """Identity + action fingerprint for the active event."""
    if not isinstance(outline_event, dict):
        return {}
    start = dict(outline_event.get("expected_start_state") or {})
    end = dict(outline_event.get("expected_end_state") or {})
    return {
        "outline_id": str(outline_event.get("outline_id") or "").strip(),
        "resource_jid": _outline_resource_jid(outline_event),
        "action": _primitive_action_token(outline_event),
        "part_name": _outline_part_name(outline_event) or None,
        "start_state_id": str(start.get("state_id") or "").strip() or None,
        "end_state_id": str(end.get("state_id") or "").strip() or None,
    }


def _primitive_authoring_event_context(
    outline_event: dict[str, Any] | None,
) -> dict[str, Any]:
    """Prompt-facing active event context with the fields needed for authoring."""
    if not isinstance(outline_event, dict):
        return {}
    row: dict[str, Any] = {
        "outline_id": str(outline_event.get("outline_id") or "").strip(),
        "resource_jid": _outline_resource_jid(outline_event),
        "event_name": str(outline_event.get("event_name") or "").strip() or None,
        "description": _outline_description(outline_event) or None,
        "action_type": str(outline_event.get("action_type") or "").strip() or None,
        "part_name": _outline_part_name(outline_event) or None,
        "target_ref": _outline_target_ref(outline_event) or None,
        "action_target": deepcopy(outline_event.get("action_target") or {}),
        "expected_start_state": deepcopy(outline_event.get("expected_start_state") or {}),
        "expected_end_state": deepcopy(outline_event.get("expected_end_state") or {}),
    }
    candidate_outline_id = str(outline_event.get("candidate_outline_id") or "").strip()
    if candidate_outline_id:
        row["candidate_outline_id"] = candidate_outline_id
    return {key: value for key, value in row.items() if value not in (None, "", [], {})}


def _primitive_action_token(outline_event: dict[str, Any]) -> str:
    raw = str(outline_event.get("event_name") or "").strip()
    if not raw:
        return ""
    first = raw.split()[0].strip().lower()
    if first in {
        "acquire",
        "pick",
        "place",
        "release",
        "reorient",
        "recover",
        "store_and_regrasp",
        "pick_insert",
    }:
        return first
    return raw


def _truncate_for_log(value: Any, limit: int = _LOG_FIELD_TRUNCATE) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _parse_primitive_response_decision(
    raw_decision: str,
    *,
    turn_index: int,
    outline_id: str,
    resource_jid: str,
) -> str:
    return str(raw_decision or "").strip()


def _log_primitive_turn_summary(
    parsed_response: dict[str, Any],
    *,
    turn_index: int,
    outline_id: str,
    resource_jid: str,
) -> None:
    """Emit a single-line INFO summary of the LLM's turn decision."""
    decision = str(parsed_response.get("decision") or "").strip() or "<unset>"
    thought = _truncate_for_log(parsed_response.get("thought"))
    rationale = _truncate_for_log(parsed_response.get("rationale"))
    notes_raw = parsed_response.get("notes") or []
    notes = (
        [
            _truncate_for_log(note, 80)
            for note in notes_raw
            if isinstance(notes_raw, list) and note not in (None, "")
        ]
        if isinstance(notes_raw, list)
        else []
    )
    context_requests = [
        str(r).strip() for r in (parsed_response.get("context_requests") or []) if str(r).strip()
    ]
    primitive_steps = parsed_response.get("primitive_steps") or []
    step_count = len(primitive_steps) if isinstance(primitive_steps, list) else 0
    _logger.info(
        "[primitive-gen outline=%s resource=%s turn=%s] decision=%s steps=%d refs=%d",
        outline_id or "<none>",
        resource_jid or "<none>",
        turn_index,
        decision,
        step_count,
        len(context_requests),
    )
    _logger.debug(
        "[primitive-gen outline=%s resource=%s turn=%s] context_requests=%s thought=%r "
        "rationale=%r notes=%s",
        outline_id or "<none>",
        resource_jid or "<none>",
        turn_index,
        context_requests,
        thought,
        rationale,
        notes,
    )


def _log_primitive_validator_feedback(
    feedback: list[dict[str, Any]],
    *,
    turn_index: int,
    outline_id: str,
    resource_jid: str,
) -> None:
    rows = [dict(row) for row in (feedback or []) if isinstance(row, dict)]
    if not rows:
        return
    codes = _count_named_rows(rows, key="constraint_code")
    _logger.info(
        "[primitive-gen outline=%s resource=%s turn=%s] validator_feedback count=%d codes=%s",
        outline_id or "<none>",
        resource_jid or "<none>",
        turn_index,
        len(rows),
        _counter_summary(codes, limit=5) or "<unset>",
    )
    _logger.debug(
        "[primitive-gen outline=%s resource=%s turn=%s] validator_feedback_rows=%s",
        outline_id or "<none>",
        resource_jid or "<none>",
        turn_index,
        [
            {
                "constraint_code": str(row.get("constraint_code") or "").strip(),
                "reason": _truncate_for_log(row.get("reason")),
            }
            for row in rows
        ],
    )


def _steps_summary_for_memo(primitive_steps: list[dict[str, Any]]) -> list[str]:
    return [
        str(step.get("primitive") or "").strip()
        for step in (primitive_steps or [])
        if isinstance(step, dict) and str(step.get("primitive") or "").strip()
    ]


def _record_primitive_authoring_memo(
    *,
    session_state: dict[str, Any],
    outline_event: dict[str, Any],
    primitive_steps: list[dict[str, Any]],
    accepted_at_turn: int,
) -> None:
    """Append a compact memo row for a newly accepted decomposition.

    The memo is session-scoped: it helps subsequent events of the same
    ``(action, part)`` or ``(action, resource_jid)`` reuse the pattern.
    """
    memo = list(session_state.get("primitive_authoring_memo") or [])
    entry = {
        "outline_id": str(outline_event.get("outline_id") or "").strip(),
        "resource_jid": _outline_resource_jid(outline_event),
        "part": _outline_part_name(outline_event) or None,
        "part_name": _outline_part_name(outline_event) or None,
        "action": _primitive_action_token(outline_event),
        "steps_summary": _steps_summary_for_memo(primitive_steps),
        "accepted_at_turn": int(accepted_at_turn),
    }
    memo.append(entry)
    if len(memo) > _MEMO_MAX_ENTRIES:
        memo = memo[-_MEMO_MAX_ENTRIES:]
    session_state["primitive_authoring_memo"] = memo


def _relevant_authoring_memo(
    session_state: dict[str, Any],
    active_event: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return memo rows matching the active event on (action, part) or (action, resource)."""
    if not isinstance(active_event, dict):
        return []
    action = _primitive_action_token(active_event)
    if not action:
        return []
    part = _outline_part_name(active_event) or None
    resource = _outline_resource_jid(active_event)
    matches: list[dict[str, Any]] = []
    for row in session_state.get("primitive_authoring_memo") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("action") or "").strip() != action:
            continue
        row_part = row.get("part_name")
        row_resource = str(row.get("resource_jid") or "").strip()
        if (part and row_part == part) or (resource and row_resource == resource):
            matches.append(deepcopy(row))
    return matches


def _resolve_context_ref(
    *,
    ref: str,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    outline_event: dict[str, Any],
) -> tuple[Any, str | None]:
    """Resolve a single published ref.

    Supported ref shapes:
      /outline_event                              -> the active outline event
      /outline_event/expected_start_state         -> nested field
      /outline_event/expected_end_state
      /outline_event/action_target
      /resources/<jid>/current_pose               -> live pose from start snapshot
      /resources/<jid>/held_part
      /resources/<jid>/snapshot                   -> full resource snapshot
      /resources/<jid>/static_capabilities
      /parts/<name>/observed_pose                 -> last observation if any
      /parts/<name>/target                        -> destination geometry if grounded
      /parts/<name>/snapshot                      -> full part state
      /primitive_catalog/<resource_jid>           -> visible catalog for that resource
      /primitive_contracts/<primitive_name>       -> catalog entry for one primitive
      /projected_snapshot/<resource_jid>          -> last projected snapshot for this resource
    """
    trimmed = str(ref or "").strip()
    if not trimmed.startswith("/"):
        return None, f"ref must begin with '/': {ref!r}"
    segments = [seg for seg in trimmed.split("/") if seg]
    if not segments:
        return None, "ref cannot be empty"

    head = segments[0]
    rest = segments[1:]

    if head == "outline_event":
        payload = deepcopy(outline_event)
        for key in rest:
            if not isinstance(payload, dict):
                return None, f"cannot descend into {key!r} on non-object"
            if key not in payload:
                return None, f"outline_event has no field {key!r}"
            payload = deepcopy(payload.get(key))
        return payload, None

    if head == "resources":
        if not rest:
            return None, "resources ref requires <resource_jid>"
        resource_jid = rest[0]
        if resource_jid == _outline_resource_jid(outline_event):
            start_snapshot = (
                _primitive_resource_start_views(
                    session_state=session_state,
                    prepared_recovery_request=prepared_recovery_request,
                    outline_event=outline_event,
                ).get("prompt_snapshot")
                or {}
            )
        else:
            start_snapshot = _build_literal_resource_start_snapshot(
                base_snapshot=_resource_snapshot_with_caps(
                    session_state=session_state,
                    prepared_recovery_request=prepared_recovery_request,
                    resource_jid=resource_jid,
                ),
                exact_state={},
            )
        field = rest[1] if len(rest) > 1 else "snapshot"
        if field == "snapshot":
            return deepcopy(start_snapshot), None
        if field in start_snapshot:
            return deepcopy(start_snapshot.get(field)), None
        return None, f"resources/{resource_jid} has no field {field!r}"

    if head == "parts":
        if not rest:
            return None, "parts ref requires <part_name>"
        part_name = rest[0]
        grounding = _primitive_grounding_context(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
            outline_event=outline_event,
        )
        parts_root = dict(grounding.get("parts") or {})
        part_entry = _build_literal_part_start_snapshot(
            base_snapshot=dict(parts_root.get(part_name) or {}),
            outline_event=outline_event,
            part_name=part_name,
        )
        field = rest[1] if len(rest) > 1 else "snapshot"
        if field == "snapshot":
            return deepcopy(part_entry), None
        if field in part_entry:
            return deepcopy(part_entry.get(field)), None
        return None, f"parts/{part_name} has no field {field!r}"

    if head == "primitive_catalog":
        resource_jid = rest[0] if rest else _outline_resource_jid(outline_event)
        catalog = _primitive_catalog_for_resource(prepared_recovery_request, resource_jid)
        return _visible_primitive_catalog_card(catalog), None

    if head == "primitive_contracts":
        if not rest:
            return None, "primitive_contracts ref requires <primitive_name>"
        primitive_name = rest[0]
        resource_jid = _outline_resource_jid(outline_event)
        catalog = _primitive_catalog_for_resource(prepared_recovery_request, resource_jid)
        for entry in _visible_primitive_catalog_card(catalog):
            if entry.get("name") == primitive_name:
                return deepcopy(entry), None
        return None, f"primitive {primitive_name!r} is not in the visible catalog"

    if head == "projected_snapshot":
        resource_jid = rest[0] if rest else _outline_resource_jid(outline_event)
        snapshot = _previous_projected_snapshot_for_resource(session_state, resource_jid)
        return deepcopy(snapshot) if snapshot is not None else {}, None

    if head == "observation_store":
        return deepcopy(session_state.get("observation_store") or {}), None

    if head == "accepted_outline_prefix":
        return deepcopy(list(session_state.get("accepted_outline_prefix") or [])), None

    if head == "remaining_outline_events":
        cursor = int(session_state.get("primitive_generation_cursor") or 0)
        prefix = list(session_state.get("accepted_outline_prefix") or [])
        remaining = prefix[cursor:] if cursor < len(prefix) else []
        return deepcopy(remaining), None

    if head == "safety_rules":
        llm_input = dict(prepared_recovery_request.get("llm_input") or {})
        return deepcopy(list(llm_input.get("loaded_safety_rules") or [])), None

    if head == "capability_decompositions":
        function_name = rest[0] if rest else ""
        resource_jid = _outline_resource_jid(outline_event)
        available_names = _capability_decomposition_names_for_resource(
            prepared_recovery_request=prepared_recovery_request,
            resource_jid=resource_jid,
        )
        payload = _capability_decomposition_for_resource(
            prepared_recovery_request=prepared_recovery_request,
            resource_jid=resource_jid,
            function_name=function_name,
        )
        if function_name and not payload:
            visible_primitive_names = {
                str(row.get("name") or "").strip()
                for row in _visible_primitive_catalog_names(
                    _primitive_catalog_for_resource(prepared_recovery_request, resource_jid)
                )
                if str(row.get("name") or "").strip()
            }
            if function_name in visible_primitive_names:
                available_text = ", ".join(available_names) or "<none>"
                return None, (
                    f"{function_name!r} is a primitive name, not a capability decomposition; "
                    f"use /primitive_contracts/{function_name} for the primitive contract or "
                    f"choose one of the available capability decomposition names: {available_text}"
                )
            return None, (
                f"no capability decomposition registered for function "
                f"{function_name!r} on resource {resource_jid!r}; "
                f"available names: {', '.join(available_names) or '<none>'}"
            )
        return deepcopy(payload), None

    if head == "memo":
        if not rest or rest[0] != "primitive_authoring":
            return None, f"unknown memo ref {trimmed!r}"
        return deepcopy(list(session_state.get("primitive_authoring_memo") or [])), None

    return None, f"unknown ref head {head!r}"


def _serve_context_requests(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    outline_event: dict[str, Any],
    context_requests: list[Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Resolve each requested ref. Returns (served_context, errors)."""
    served: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    for raw_ref in context_requests or []:
        ref = str(raw_ref or "").strip()
        if not ref:
            errors.append({"context_ref": "", "reason": "empty ref"})
            continue
        value, error = _resolve_context_ref(
            ref=ref,
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
            outline_event=outline_event,
        )
        if error:
            errors.append({"context_ref": ref, "reason": error})
            continue
        served[ref] = value
    return served, errors


# ---------------------------------------------------------------------------
# Authored-plan shape validation
# ---------------------------------------------------------------------------


def _validate_authored_plan(
    *,
    parsed_response: dict[str, Any],
    active_event: dict[str, Any],
    visible_catalog: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Shape-validate the authored plan before the trace validator runs.

    Enforces the hard rules of the LLM contract: matching outline/resource ids,
    non-empty primitive_steps, only visible-catalog primitives, and proper step
    shape.
    """
    active_outline_id = str(active_event.get("outline_id") or "").strip()
    active_resource_jid = _outline_resource_jid(active_event)
    response_outline_id = str(parsed_response.get("outline_id") or "").strip()
    response_resource_jid = str(parsed_response.get("resource_jid") or "").strip()
    if response_outline_id != active_outline_id:
        return None, f"outline_id must match the active event {active_outline_id!r}"
    if response_resource_jid != active_resource_jid:
        return None, f"resource_jid must match the active event {active_resource_jid!r}"

    raw_steps = parsed_response.get("primitive_steps") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        return None, "primitive_steps_ready requires non-empty primitive_steps"

    visible_names = {
        str(entry.get("name") or "").strip()
        for entry in (visible_catalog or [])
        if isinstance(entry, dict) and str(entry.get("name") or "").strip()
    }

    steps: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            return None, f"primitive_steps[{index}] must be an object"
        primitive = str(raw.get("primitive") or "").strip()
        if not primitive:
            return None, f"primitive_steps[{index}] is missing 'primitive'"
        if primitive not in visible_names:
            return None, (
                f"primitive_steps[{index}].primitive {primitive!r} is not in the visible "
                "catalog for this resource"
            )
        params = raw.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return None, f"primitive_steps[{index}].params must be an object"
        step: dict[str, Any] = {
            "primitive": primitive,
            "params": deepcopy(params),
        }
        store_as = str(raw.get("store_as") or "").strip()
        if store_as:
            return None, (
                f"primitive_steps[{index}] uses legacy field 'store_as'; "
                "data-producing primitives now publish deterministic event_facts automatically"
            )
        steps.append(step)
    return steps, None


# ---------------------------------------------------------------------------
# Trace validator wrapper (authored plan -> projected snapshot)
# ---------------------------------------------------------------------------


def _validate_single_event_primitive_steps(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    outline_event: dict[str, Any],
    primitive_steps: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    resource_jid = _outline_resource_jid(outline_event)
    start_state_views = _primitive_resource_start_views(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        outline_event=outline_event,
    )
    start_snapshot = dict(start_state_views.get("executor_snapshot") or {})
    start_diagnostics = [
        dict(row) for row in (start_state_views.get("diagnostics") or []) if isinstance(row, dict)
    ]
    primitive_catalog = _primitive_catalog_for_resource(
        prepared_recovery_request,
        resource_jid,
    )
    grounding_context = _primitive_grounding_context(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        outline_event=outline_event,
    )
    trace_result = validate_and_project_steps_with_trace(
        primitive_steps,
        primitive_catalog,
        start_snapshot,
        grounding_context=grounding_context,
    )
    valid = bool(trace_result.get("valid"))
    projected_snapshot = dict(trace_result.get("projected_snapshot") or {})
    projected_outline_state = _projected_outline_state(projected_snapshot, outline_event)
    validation_error = trace_result.get("validation_error")
    per_event_result = {
        "outline_id": str(outline_event.get("outline_id") or "").strip(),
        "resource_jid": resource_jid,
        "start_snapshot": deepcopy(start_snapshot),
        "projected_snapshot": deepcopy(projected_snapshot),
        "projected_outline_state": deepcopy(projected_outline_state),
        "valid": valid,
        "validation_error": validation_error,
        "trace_step_count": len(trace_result.get("step_results") or []),
    }
    if start_diagnostics:
        per_event_result["start_input_diagnostics"] = deepcopy(start_diagnostics)
        feedback = [
            _primitive_feedback_row(
                outline_event=outline_event,
                constraint_code=str(
                    row.get("constraint_code") or "primitive_start_state_contradiction"
                ),
                reason=str(row.get("reason") or "primitive start state is contradictory"),
                finding=row,
            )
            for row in start_diagnostics
        ]
        return per_event_result, feedback
    if not valid:
        feedback = [
            _primitive_feedback_row(
                outline_event=outline_event,
                constraint_code="primitive_validation_failed",
                reason=validation_error or "primitive sequence failed validation",
            )
        ]
        return per_event_result, feedback

    resource_findings = _primitive_resource_sequence_findings(
        outline_event=outline_event,
        primitive_steps=primitive_steps,
        trace_metadata=trace_result,
        start_snapshot=start_snapshot,
        projected_snapshot=projected_snapshot,
        grounding_context=grounding_context,
        primitive_catalog=primitive_catalog,
    )
    if resource_findings:
        per_event_result["resource_validation_findings"] = deepcopy(resource_findings)
        feedback = [
            _primitive_feedback_row(
                outline_event=outline_event,
                constraint_code=str(row.get("constraint_code") or "primitive_sequence_invalid"),
                reason=str(row.get("reason") or "resource primitive sequence validation failed"),
                finding=row,
            )
            for row in resource_findings
        ]
        return per_event_result, feedback

    per_event_result["trace_metadata"] = deepcopy(trace_result)
    return per_event_result, []


def _accepted_program_row(
    *,
    outline_event: dict[str, Any],
    primitive_steps: list[dict[str, Any]],
    projected_snapshot: dict[str, Any],
    projected_outline_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "outline_id": str(outline_event.get("outline_id") or "").strip(),
        "des_event_id": str(outline_event.get("outline_id") or "").strip(),
        "resource_jid": _outline_resource_jid(outline_event),
        "part_name": _outline_part_name(outline_event) or None,
        "event_name": str(outline_event.get("event_name") or "").strip(),
        "description": _outline_description(outline_event),
        "predecessors": [
            str(item).strip()
            for item in (outline_event.get("predecessors") or [])
            if str(item).strip()
        ],
        "primitive_steps": deepcopy(primitive_steps),
        "projected_snapshot": deepcopy(projected_snapshot),
        "projected_outline_state": deepcopy(projected_outline_state),
    }


def _primitive_batch_session_state(
    *,
    assigned_outline_events: list[dict[str, Any]],
    carried_session_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    carried = deepcopy(carried_session_state or {})
    accepted_outline_prefix = [
        deepcopy(row) for row in (assigned_outline_events or []) if isinstance(row, dict)
    ]
    session_state = {
        "current_phase": "primitive_generation",
        "status": "running",
        "turn_index": int(carried.get("turn_index") or 0),
        "final_output_turn_base": int(carried.get("turn_index") or 0),
        "turns": deepcopy(carried.get("turns") or []),
        "accepted_outline_prefix": accepted_outline_prefix,
        "accepted_transition_prefix": deepcopy(accepted_outline_prefix),
        "des_event_sequence": deepcopy(accepted_outline_prefix),
        "transition_trace": deepcopy(accepted_outline_prefix),
        "primitive_generation_turn_index": 0,
        "primitive_generation_cursor": 0,
        "primitive_outline_turn_counters": {},
        "accepted_primitive_program": [],
        "primitive_rejection_feedback": [],
        "primitive_served_context": deepcopy(carried.get("primitive_served_context") or {}),
        "primitive_context_errors": deepcopy(carried.get("primitive_context_errors") or []),
        "primitive_input_diagnostics": [],
        "primitive_event_guard": {},
        "primitive_escalation_diagnostics": [],
        "primitive_authoring_memo": deepcopy(carried.get("primitive_authoring_memo") or []),
        "observation_store": deepcopy(carried.get("observation_store") or {}),
        "base_symbolic_resources": deepcopy(carried.get("base_symbolic_resources") or {}),
        "base_symbolic_parts": deepcopy(carried.get("base_symbolic_parts") or {}),
        "symbolic_resources": deepcopy(carried.get("symbolic_resources") or {}),
        "symbolic_parts": deepcopy(carried.get("symbolic_parts") or {}),
    }
    compact_multi_turn_runtime_session(session_state)
    return session_state


def _write_primitive_subturn_artifact(
    *,
    prepared_recovery_request: dict[str, Any],
    session_state: dict[str, Any],
    assigned_outline_events: list[dict[str, Any]],
    recovery_session_id: str,
    turn_entry: dict[str, Any],
) -> dict[str, str]:
    recovery_debug = dict(prepared_recovery_request.get("recovery_debug") or {})
    per_turn_debug_dir = str(recovery_debug.get("per_turn_debug_dir") or "").strip()
    if not per_turn_debug_dir:
        return {}

    payload = {
        "prepared_recovery_request": deepcopy(prepared_recovery_request),
        "reasoning_mode": "multi_turn",
        "multi_turn_current_turn": {
            "turn_index": int(turn_entry.get("turn_index") or 0),
            "phase": "primitive_generation",
            "primitive_substream_turns": [deepcopy(turn_entry)],
        },
        "primitive_batch_resume_checkpoint": {
            "prepared_recovery_request": deepcopy(prepared_recovery_request),
            "assigned_outline_events": [
                deepcopy(row) for row in (assigned_outline_events or []) if isinstance(row, dict)
            ],
            "session_state": deepcopy(session_state),
            "recovery_session_id": str(recovery_session_id or "").strip(),
            "resource_jid": str(turn_entry.get("resource_jid") or "").strip(),
            "current_turn": deepcopy(turn_entry),
            "final_output_turn_base": int(session_state.get("final_output_turn_base") or 0),
        },
    }
    try:
        artifact_paths = write_recovery_artifacts(
            payload,
            phase_label="multi_turn",
            debug_dir=per_turn_debug_dir,
            write_latest=False,
        )
    except Exception as exc:
        _logger.warning(
            "[MultiTurn] Failed to write immediate primitive subturn artifact: %s",
            exc,
        )
        return {}

    rows_raw = artifact_paths.get("primitive_substream_artifact_paths")
    if not rows_raw:
        return {}
    try:
        rows = json.loads(rows_raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return {}

    artifact_row = {
        "prompt_artifact_path": str(rows[0].get("prompt_artifact_path") or "").strip(),
        "response_artifact_path": str(rows[0].get("response_artifact_path") or "").strip(),
        "primitive_resume_checkpoint_artifact_path": str(
            artifact_paths.get("primitive_resume_checkpoint_artifact_path") or ""
        ).strip(),
        "latest_primitive_resume_checkpoint_artifact_path": str(
            artifact_paths.get("latest_primitive_resume_checkpoint_artifact_path") or ""
        ).strip(),
    }
    if artifact_row["prompt_artifact_path"]:
        turn_entry["prompt_artifact_path"] = artifact_row["prompt_artifact_path"]
    if artifact_row["response_artifact_path"]:
        turn_entry["response_artifact_path"] = artifact_row["response_artifact_path"]
    if artifact_row["primitive_resume_checkpoint_artifact_path"]:
        turn_entry["primitive_resume_checkpoint_artifact_path"] = artifact_row[
            "primitive_resume_checkpoint_artifact_path"
        ]
    if artifact_row["latest_primitive_resume_checkpoint_artifact_path"]:
        turn_entry["latest_primitive_resume_checkpoint_artifact_path"] = artifact_row[
            "latest_primitive_resume_checkpoint_artifact_path"
        ]
    return artifact_row


async def generate_primitive_batch_with_llm_agent(
    *,
    llm_agent: Any,
    prepared_recovery_request: dict[str, Any],
    assigned_outline_events: list[dict[str, Any]],
    recovery_session_id: str = "",
    carried_session_state: dict[str, Any] | None = None,
    session_state: dict[str, Any] | None = None,
    max_turns: int = 24,
) -> dict[str, Any]:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        build_multi_turn_phase_prompt_input,
        multi_turn_phase_response_schema,
        render_multi_turn_phase_prompt,
    )

    ask_llm_structured = getattr(llm_agent, "ask_llm_structured", None)
    if not callable(ask_llm_structured):
        raise RuntimeError(
            "llm_agent.ask_llm_structured is required for primitive batch generation"
        )

    assigned_events = [
        deepcopy(row) for row in (assigned_outline_events or []) if isinstance(row, dict)
    ]
    resource_jids = {
        _outline_resource_jid(row) for row in assigned_events if _outline_resource_jid(row)
    }
    resource_jid = next(iter(resource_jids)) if len(resource_jids) == 1 else ""
    llm_input = dict(prepared_recovery_request.get("llm_input") or {})
    recovery_resources = dict(prepared_recovery_request.get("recovery_resources") or {})
    response_schema = multi_turn_phase_response_schema("primitive_generation")
    system_instructions = ""
    if session_state is not None:
        session_state = deepcopy(session_state)
    else:
        session_state = _primitive_batch_session_state(
            assigned_outline_events=assigned_events,
            carried_session_state=carried_session_state,
        )
    final_output_turn_base = int(
        session_state.get("final_output_turn_base")
        or (carried_session_state or {}).get("turn_index")
        or 0
    )
    session_state["final_output_turn_base"] = final_output_turn_base
    turn_records: list[dict[str, Any]] = []
    final_decision = "need_primitive_revision"

    start_turn_index = int(session_state.get("turn_index") or 0)
    for turn_index in range(
        start_turn_index + 1,
        max(1, int(max_turns or 1)) + 1,
    ):
        session_state["turn_index"] = turn_index
        prompt_input = build_multi_turn_phase_prompt_input(
            phase="primitive_generation",
            llm_input=llm_input,
            session_state=session_state,
            recovery_resources=recovery_resources,
        )
        prompt_input.update(
            build_primitive_generation_prompt_context(
                session_state=session_state,
                prepared_recovery_request=prepared_recovery_request,
            )
        )
        prompt_text = render_multi_turn_phase_prompt(prompt_input)
        raw_response = await ask_llm_structured(
            prompt=prompt_text,
            response_format=response_schema,
            include_agent_instructions=False,
        )
        expected_messages = [{"role": "user", "content": prompt_text}]
        llm_request = deepcopy(
            dict(getattr(llm_agent, "_last_structured_request", {}) or {})
        )
        if llm_request.get("messages") != expected_messages:
            llm_request = {
                "model": str(
                    getattr(llm_agent, "model", "")
                    or getattr(llm_agent, "llm_model", "")
                ).strip(),
                "messages": expected_messages,
                "reasoning_effort": str(
                    getattr(llm_agent, "reasoning_effort", "")
                    or getattr(llm_agent, "llm_reasoning_effort", "")
                ).strip(),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": deepcopy(response_schema),
                },
                "response_source": "unknown",
            }
        parsed_response = deepcopy(raw_response) if isinstance(raw_response, dict) else {}
        decision, turn_entry = await _handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response=parsed_response,
            prepared_recovery_request=prepared_recovery_request,
            planner=None,
        )
        turn_entry["turn_index"] = turn_index
        turn_entry["phase"] = "primitive_generation"
        turn_entry["decision"] = decision
        turn_entry["prompt_input"] = deepcopy(prompt_input)
        turn_entry["prompt_text"] = prompt_text
        turn_entry["llm_raw_response"] = deepcopy(parsed_response)
        turn_entry["llm_request"] = llm_request
        turn_entry["resource_jid"] = resource_jid
        turn_entry["recovery_session_id"] = str(recovery_session_id or "").strip()
        turn_entry["response_schema"] = deepcopy(response_schema)
        turn_entry["system_instructions"] = system_instructions
        raw_response = deepcopy(parsed_response)
        if (
            str(turn_entry.get("decision") or "").strip()
            and not str(raw_response.get("decision") or "").strip()
        ):
            raw_response["decision"] = str(turn_entry.get("decision") or "").strip()
        turn_entry["raw_response"] = raw_response
        _write_primitive_subturn_artifact(
            prepared_recovery_request=prepared_recovery_request,
            session_state=session_state,
            assigned_outline_events=assigned_events,
            recovery_session_id=recovery_session_id,
            turn_entry=turn_entry,
        )
        turn_records.append(deepcopy(turn_entry))
        session_state.setdefault("turns", []).append(deepcopy(turn_entry))
        compact_multi_turn_runtime_session(session_state)
        final_decision = decision
        if decision in {"draft_ready", "primitive_blocked", "primitive_event_stuck"}:
            break

    if final_decision not in {
        "draft_ready",
        "primitive_blocked",
        "primitive_event_stuck",
    }:
        final_decision = "need_primitive_revision"

    if final_decision == "draft_ready":
        from .multi_turn import _append_final_output_turn

        session_state["turn_index"] = final_output_turn_base
        _append_final_output_turn(
            planner=None,
            prepared_recovery_request=prepared_recovery_request,
            session_state=session_state,
            stage="primitive_program_ready",
        )

    return {
        "decision": final_decision,
        "resource_jid": resource_jid,
        "recovery_session_id": str(recovery_session_id or "").strip(),
        "primitive_events": [
            deepcopy(row)
            for row in (session_state.get("accepted_primitive_program") or [])
            if isinstance(row, dict)
        ],
        "feedback": [
            deepcopy(row)
            for row in (session_state.get("primitive_rejection_feedback") or [])
            if isinstance(row, dict)
        ],
        "context_errors": [
            deepcopy(row)
            for row in (session_state.get("primitive_context_errors") or [])
            if isinstance(row, dict)
        ],
        "turns": turn_records,
        "session_state": deepcopy(session_state),
        "final_output": deepcopy(session_state.get("final_output") or {}),
        "recovery_proposal": deepcopy(session_state.get("proposal") or {}),
    }


# ---------------------------------------------------------------------------
# Prompt-context builder (minimal kernel; served context is stashed in
# session_state by the phase handler and carried through)
# ---------------------------------------------------------------------------


_PUBLISHED_REF_SCHEMA: list[dict[str, str]] = [
    {"ref": "/outline_event", "description": "the active outline event (full)"},
    {"ref": "/resources/<jid>/snapshot", "description": "full resource snapshot"},
    {"ref": "/resources/<jid>/current_pose", "description": "current pose for the resource"},
    {"ref": "/resources/<jid>/held_part", "description": "currently held part, if any"},
    {
        "ref": "/resources/<jid>/static_capabilities",
        "description": "named poses, workspace bounds, staging areas",
    },
    {"ref": "/parts/<name>/snapshot", "description": "full part state"},
    {
        "ref": "/parts/<name>/observed_pose",
        "description": "last observed pose for the part, if any",
    },
    {"ref": "/parts/<name>/target", "description": "destination geometry if grounded"},
    {
        "ref": "/primitive_catalog/<jid>",
        "description": "visible primitive catalog for the resource",
    },
    {
        "ref": "/primitive_contracts/<name>",
        "description": "contract card for one visible primitive",
    },
    {
        "ref": "/projected_snapshot/<jid>",
        "description": "last accepted projected snapshot for this resource",
    },
    {"ref": "/observation_store", "description": "session observation store"},
    {"ref": "/accepted_outline_prefix", "description": "full accepted outline prefix (all events)"},
    {"ref": "/remaining_outline_events", "description": "events at/after the current cursor"},
    {"ref": "/safety_rules", "description": "loaded safety rules with raw text"},
    {
        "ref": "/capability_decompositions/<function_name>",
        "description": "primary modeled source for how the active resource composes recovery-visible low-level primitives",
    },
    {
        "ref": "/memo/primitive_authoring",
        "description": "session-scoped memo of previously accepted decompositions",
    },
]


def build_primitive_generation_prompt_context(
    *,
    session_state: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
) -> dict[str, Any]:
    cursor, active_event, accepted_prefix = _active_primitive_outline_event(session_state)
    remaining_events = _missing_primitive_outline_events(session_state)
    context: dict[str, Any] = {
        "primitive_active_outline_event": _primitive_authoring_event_context(active_event),
        "primitive_generation_cursor_state": {
            "active_index": cursor,
            "accepted_outline_count": len(accepted_prefix),
            "remaining_events_count": len(remaining_events),
        },
        "primitive_published_ref_schema": deepcopy(_PUBLISHED_REF_SCHEMA),
        "primitive_served_context": deepcopy(session_state.get("primitive_served_context") or {}),
        "primitive_context_errors": deepcopy(session_state.get("primitive_context_errors") or []),
        "primitive_authoring_memo_summary": _relevant_authoring_memo(session_state, active_event),
    }
    if active_event is None:
        context["primitive_visible_catalog"] = []
        context["primitive_capability_decomposition_names"] = []
        context["primitive_active_resource_named_poses"] = []
        return context
    context["primitive_input_diagnostics"] = deepcopy(
        _primitive_resource_start_views(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
            outline_event=active_event,
        ).get("diagnostics")
        or session_state.get("primitive_input_diagnostics")
        or []
    )
    resource_jid = _outline_resource_jid(active_event)
    context["primitive_visible_catalog"] = _visible_primitive_catalog_names(
        _primitive_catalog_for_resource(prepared_recovery_request, resource_jid)
    )
    context["primitive_capability_decomposition_names"] = (
        _capability_decomposition_names_for_resource(
            prepared_recovery_request=prepared_recovery_request,
            resource_jid=resource_jid,
        )
    )
    context["primitive_active_resource_named_poses"] = _active_resource_named_pose_names(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        resource_jid=resource_jid,
    )
    # Decomposition selection is LLM-driven: the LLM picks from the
    # "Available Capability Decompositions" list via context_requests, so no
    # deterministic keyword shortlist is provided.
    context["primitive_suggested_capability_decompositions"] = []
    return context


def _maybe_escalate_primitive_event(
    *,
    session_state: dict[str, Any],
    turn_entry: dict[str, Any],
    cursor: int,
    active_event: dict[str, Any],
    guard_state: dict[str, Any],
    feedback_rows: list[dict[str, Any]] | None = None,
    context_errors: list[dict[str, Any]] | None = None,
    input_diagnostics: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    if not _primitive_event_should_escalate(guard_state):
        return None

    stuck_reason = _primitive_event_stuck_reason(
        guard_state=guard_state,
        feedback_rows=feedback_rows,
        context_errors=context_errors,
        input_diagnostics=input_diagnostics,
    )
    stuck_feedback = [
        _primitive_feedback_row(
            outline_event=active_event,
            constraint_code="primitive_event_stuck",
            reason=stuck_reason,
        )
    ]
    diagnostics = _record_primitive_escalation_diagnostic(
        session_state,
        cursor=cursor,
        outline_event=active_event,
        guard_state=guard_state,
        feedback_rows=feedback_rows,
        context_errors=context_errors,
        input_diagnostics=input_diagnostics,
        reason=stuck_reason,
    )
    session_state["primitive_rejection_feedback"] = deepcopy(stuck_feedback)
    _clear_primitive_event_guard(session_state)
    turn_entry["primitive_rejection_feedback"] = deepcopy(stuck_feedback)
    turn_entry["primitive_escalation_diagnostics"] = deepcopy(diagnostics)
    session_state["status"] = "paused_after_primitive_stuck"
    _logger.warning(
        "[primitive-gen outline=%s resource=%s cursor=%d] stuck_event "
        "same_signature_streak=%d no_progress_turns=%d blockers=%s",
        str(active_event.get("outline_id") or ""),
        _outline_resource_jid(active_event),
        cursor,
        int(guard_state.get("same_signature_streak") or 0),
        int(guard_state.get("no_progress_turns") or 0),
        _counter_summary(
            _count_named_rows(
                list(guard_state.get("feedback_history") or []),
                key="constraint_code",
            ),
            limit=5,
        )
        or "<none>",
    )
    return "primitive_event_stuck", turn_entry


# ---------------------------------------------------------------------------
# Phase handler
# ---------------------------------------------------------------------------


async def _handle_primitive_generation_phase(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    prepared_recovery_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Handle one event-local turn of primitive generation (LLM-authored plan)."""
    del planner
    turn_index = int(session_state.get("primitive_generation_turn_index") or 0) + 1
    session_state["primitive_generation_turn_index"] = turn_index
    cursor, active_event, accepted_prefix = _active_primitive_outline_event(session_state)
    remaining_events = accepted_prefix[cursor:] if cursor < len(accepted_prefix) else []
    outline_id = str(
        (active_event or {}).get("outline_id") or parsed_response.get("outline_id") or ""
    )
    resource_jid = str(
        _outline_resource_jid(active_event) or parsed_response.get("resource_jid") or ""
    )
    primitive_turn_counters = dict(session_state.get("primitive_outline_turn_counters") or {})
    primitive_local_turn_index = 0
    if outline_id:
        primitive_local_turn_index = int(primitive_turn_counters.get(outline_id) or 0) + 1
        primitive_turn_counters[outline_id] = primitive_local_turn_index
        session_state["primitive_outline_turn_counters"] = primitive_turn_counters
    response_decision = _parse_primitive_response_decision(
        str(parsed_response.get("decision") or "").strip(),
        turn_index=turn_index,
        outline_id=outline_id,
        resource_jid=resource_jid,
    )
    logged_response = dict(parsed_response)
    logged_response["decision"] = response_decision
    _log_primitive_turn_summary(
        logged_response,
        turn_index=turn_index,
        outline_id=outline_id,
        resource_jid=resource_jid,
    )
    turn_entry: dict[str, Any] = {
        "primitive_generation_cursor": cursor,
        "outline_id": outline_id,
        "primitive_local_turn_index": primitive_local_turn_index,
        "active_outline_event": deepcopy(active_event),
        "active_recovery_event": deepcopy(active_event),
        "accepted_transition_prefix": deepcopy(accepted_prefix),
        "des_event_sequence": deepcopy(accepted_prefix),
        "remaining_outline_events": deepcopy(remaining_events),
    }

    if active_event is None:
        if accepted_prefix:
            session_state["primitive_rejection_feedback"] = []
            session_state["primitive_context_errors"] = []
            session_state["primitive_input_diagnostics"] = []
            _clear_primitive_event_guard(session_state)
            session_state["status"] = "paused_after_primitive_generation"
            return "draft_ready", turn_entry
        feedback = [
            _primitive_feedback_row(
                outline_event={},
                constraint_code="outline_prefix_missing",
                reason="primitive generation requires an accepted outline prefix",
            )
        ]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        session_state["primitive_context_errors"] = []
        session_state["primitive_input_diagnostics"] = []
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        session_state["status"] = "paused_after_primitive_blocked"
        return "primitive_blocked", turn_entry

    input_diagnostics = [
        dict(row)
        for row in (
            _primitive_resource_start_views(
                session_state=session_state,
                prepared_recovery_request=prepared_recovery_request,
                outline_event=active_event,
            ).get("diagnostics")
            or []
        )
        if isinstance(row, dict)
    ]
    session_state["primitive_input_diagnostics"] = deepcopy(input_diagnostics)
    turn_entry["primitive_input_diagnostics"] = deepcopy(input_diagnostics)

    turn_entry["primitive_response"] = {
        key: deepcopy(response_decision if key == "decision" else parsed_response.get(key))
        for key in (
            "outline_id",
            "resource_jid",
            "decision",
            "context_requests",
            "primitive_steps",
            "rationale",
            "notes",
        )
        if key in parsed_response
    }
    raw_response_decision = str(parsed_response.get("decision") or "").strip()
    if raw_response_decision and raw_response_decision != response_decision:
        turn_entry["primitive_response"]["legacy_decision"] = raw_response_decision

    # --- primitive_blocked -------------------------------------------------
    if response_decision == "primitive_blocked":
        rationale = str(parsed_response.get("rationale") or "").strip()
        feedback = [
            _primitive_feedback_row(
                outline_event=active_event,
                constraint_code="primitive_blocked",
                reason=rationale
                or (
                    "LLM reported that the active event is blocked by contradictory "
                    "or insufficient primitive-side context"
                ),
            )
        ]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        session_state["primitive_input_diagnostics"] = deepcopy(input_diagnostics)
        _clear_primitive_event_guard(session_state)
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        turn_entry["primitive_served_context"] = deepcopy(
            session_state.get("primitive_served_context") or {}
        )
        turn_entry["primitive_context_errors"] = deepcopy(
            session_state.get("primitive_context_errors") or []
        )
        session_state["status"] = "paused_after_primitive_blocked"
        return "primitive_blocked", turn_entry

    # --- need_primitive_revision -------------------------------------------
    if response_decision == "need_primitive_revision":
        rationale = str(parsed_response.get("rationale") or "").strip()
        if not rationale:
            feedback = [
                _primitive_feedback_row(
                    outline_event=active_event,
                    constraint_code="primitive_schema_violation",
                    reason="need_primitive_revision requires a non-empty rationale naming the contract gap",
                )
            ]
            session_state["primitive_rejection_feedback"] = deepcopy(feedback)
            session_state["primitive_context_errors"] = []
            turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
            guard_state = _record_primitive_no_progress(
                session_state,
                cursor=cursor,
                outline_event=active_event,
                response_decision=response_decision,
                feedback_rows=feedback,
                input_diagnostics=input_diagnostics,
            )
            escalated = _maybe_escalate_primitive_event(
                session_state=session_state,
                turn_entry=turn_entry,
                cursor=cursor,
                active_event=active_event,
                guard_state=guard_state,
                feedback_rows=feedback,
                input_diagnostics=input_diagnostics,
            )
            if escalated is not None:
                return escalated
            session_state["status"] = "paused_after_primitive_turn"
            return "need_primitive_revision", turn_entry
        # LLM-driven decomposition selection: no keyword-suggested decompositions
        # are auto-served. The LLM requests the decomposition it needs via
        # context_requests (decision=need_context); a need_primitive_revision with
        # a rationale is therefore handled directly as a revision below.
        feedback = [
            _primitive_feedback_row(
                outline_event=active_event,
                constraint_code="primitive_revision_requested",
                reason=f"LLM reported primitive-surface gap: {rationale}",
            )
        ]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        session_state["primitive_context_errors"] = []
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        guard_state = _record_primitive_no_progress(
            session_state,
            cursor=cursor,
            outline_event=active_event,
            response_decision=response_decision,
            feedback_rows=feedback,
            input_diagnostics=input_diagnostics,
        )
        escalated = _maybe_escalate_primitive_event(
            session_state=session_state,
            turn_entry=turn_entry,
            cursor=cursor,
            active_event=active_event,
            guard_state=guard_state,
            feedback_rows=feedback,
            input_diagnostics=input_diagnostics,
        )
        if escalated is not None:
            return escalated
        session_state["status"] = "paused_after_primitive_turn"
        return "need_primitive_revision", turn_entry

    # --- need_context (agentic retrieval) ----------------------------------
    if response_decision == "need_context":
        context_requests = [
            str(raw).strip()
            for raw in (parsed_response.get("context_requests") or [])
            if str(raw).strip()
        ]
        if not context_requests:
            feedback = [
                _primitive_feedback_row(
                    outline_event=active_event,
                    constraint_code="primitive_schema_violation",
                    reason="need_context requires a non-empty context_requests list",
                )
            ]
            session_state["primitive_rejection_feedback"] = deepcopy(feedback)
            session_state["primitive_context_errors"] = []
            turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
            guard_state = _record_primitive_no_progress(
                session_state,
                cursor=cursor,
                outline_event=active_event,
                response_decision=response_decision,
                feedback_rows=feedback,
                input_diagnostics=input_diagnostics,
            )
            escalated = _maybe_escalate_primitive_event(
                session_state=session_state,
                turn_entry=turn_entry,
                cursor=cursor,
                active_event=active_event,
                guard_state=guard_state,
                feedback_rows=feedback,
                input_diagnostics=input_diagnostics,
            )
            if escalated is not None:
                return escalated
            session_state["status"] = "paused_after_primitive_turn"
            return "need_primitive_revision", turn_entry
        if parsed_response.get("primitive_steps"):
            feedback = [
                _primitive_feedback_row(
                    outline_event=active_event,
                    constraint_code="primitive_schema_violation",
                    reason="need_context must not include primitive_steps",
                )
            ]
            session_state["primitive_rejection_feedback"] = deepcopy(feedback)
            session_state["primitive_context_errors"] = []
            turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
            guard_state = _record_primitive_no_progress(
                session_state,
                cursor=cursor,
                outline_event=active_event,
                response_decision=response_decision,
                feedback_rows=feedback,
                input_diagnostics=input_diagnostics,
            )
            escalated = _maybe_escalate_primitive_event(
                session_state=session_state,
                turn_entry=turn_entry,
                cursor=cursor,
                active_event=active_event,
                guard_state=guard_state,
                feedback_rows=feedback,
                input_diagnostics=input_diagnostics,
            )
            if escalated is not None:
                return escalated
            session_state["status"] = "paused_after_primitive_turn"
            return "need_primitive_revision", turn_entry
        served, errors = _serve_context_requests(
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
            outline_event=active_event,
            context_requests=context_requests,
        )
        existing = dict(session_state.get("primitive_served_context") or {})
        new_refs = sorted(ref for ref in served if ref not in existing)
        existing.update(served)
        session_state["primitive_served_context"] = deepcopy(existing)
        session_state["primitive_context_errors"] = deepcopy(errors)
        turn_entry["primitive_served_context"] = deepcopy(existing)
        turn_entry["primitive_context_errors"] = deepcopy(errors)
        turn_entry["newly_served_context_refs"] = deepcopy(new_refs)
        session_state["primitive_rejection_feedback"] = []
        if new_refs:
            _reset_primitive_event_guard(
                session_state,
                cursor=cursor,
                outline_event=active_event,
            )
        else:
            guard_state = _record_primitive_no_progress(
                session_state,
                cursor=cursor,
                outline_event=active_event,
                response_decision=response_decision,
                context_errors=errors,
                input_diagnostics=input_diagnostics,
            )
            escalated = _maybe_escalate_primitive_event(
                session_state=session_state,
                turn_entry=turn_entry,
                cursor=cursor,
                active_event=active_event,
                guard_state=guard_state,
                context_errors=errors,
                input_diagnostics=input_diagnostics,
            )
            if escalated is not None:
                return escalated
        session_state["status"] = "paused_after_primitive_turn"
        return "need_context", turn_entry

    # --- primitive_steps_ready ---------------------------------------------
    if response_decision != "primitive_steps_ready":
        feedback = [
            _primitive_feedback_row(
                outline_event=active_event,
                constraint_code="primitive_schema_violation",
                reason=(
                    "decision must be one of: primitive_steps_ready, need_context, "
                    "need_primitive_revision, primitive_blocked"
                ),
            )
        ]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        session_state["primitive_context_errors"] = []
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        guard_state = _record_primitive_no_progress(
            session_state,
            cursor=cursor,
            outline_event=active_event,
            response_decision=response_decision,
            feedback_rows=feedback,
            input_diagnostics=input_diagnostics,
        )
        escalated = _maybe_escalate_primitive_event(
            session_state=session_state,
            turn_entry=turn_entry,
            cursor=cursor,
            active_event=active_event,
            guard_state=guard_state,
            feedback_rows=feedback,
            input_diagnostics=input_diagnostics,
        )
        if escalated is not None:
            return escalated
        session_state["status"] = "paused_after_primitive_turn"
        return "need_primitive_revision", turn_entry

    visible_catalog = _primitive_catalog_for_resource(
        prepared_recovery_request,
        _outline_resource_jid(active_event),
    )
    primitive_steps, shape_error = _validate_authored_plan(
        parsed_response=parsed_response,
        active_event=active_event,
        visible_catalog=visible_catalog,
    )
    if shape_error or primitive_steps is None:
        feedback = [
            _primitive_feedback_row(
                outline_event=active_event,
                constraint_code="primitive_schema_violation",
                reason=shape_error or "authored plan failed shape validation",
            )
        ]
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        session_state["primitive_context_errors"] = []
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        guard_state = _record_primitive_no_progress(
            session_state,
            cursor=cursor,
            outline_event=active_event,
            response_decision=response_decision,
            feedback_rows=feedback,
            input_diagnostics=input_diagnostics,
        )
        escalated = _maybe_escalate_primitive_event(
            session_state=session_state,
            turn_entry=turn_entry,
            cursor=cursor,
            active_event=active_event,
            guard_state=guard_state,
            feedback_rows=feedback,
            input_diagnostics=input_diagnostics,
        )
        if escalated is not None:
            return escalated
        session_state["status"] = "paused_after_primitive_turn"
        return "need_primitive_revision", turn_entry

    per_event_result, feedback = _validate_single_event_primitive_steps(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
        outline_event=active_event,
        primitive_steps=primitive_steps,
    )
    turn_entry["per_event_results"] = [deepcopy(per_event_result)]
    if feedback:
        _log_primitive_validator_feedback(
            feedback,
            turn_index=turn_index,
            outline_id=str(active_event.get("outline_id") or ""),
            resource_jid=_outline_resource_jid(active_event),
        )
        session_state["primitive_rejection_feedback"] = deepcopy(feedback)
        session_state["primitive_context_errors"] = []
        turn_entry["primitive_rejection_feedback"] = deepcopy(feedback)
        guard_state = _record_primitive_no_progress(
            session_state,
            cursor=cursor,
            outline_event=active_event,
            response_decision=response_decision,
            feedback_rows=feedback,
            input_diagnostics=input_diagnostics,
        )
        escalated = _maybe_escalate_primitive_event(
            session_state=session_state,
            turn_entry=turn_entry,
            cursor=cursor,
            active_event=active_event,
            guard_state=guard_state,
            feedback_rows=feedback,
            input_diagnostics=input_diagnostics,
        )
        if escalated is not None:
            return escalated
        session_state["status"] = "paused_after_primitive_turn"
        return "need_primitive_revision", turn_entry

    accepted_row = _accepted_program_row(
        outline_event=active_event,
        primitive_steps=primitive_steps,
        projected_snapshot=dict(per_event_result.get("projected_snapshot") or {}),
        projected_outline_state=dict(per_event_result.get("projected_outline_state") or {}),
    )
    accepted_program = list(session_state.get("accepted_primitive_program") or [])
    accepted_program.append(deepcopy(accepted_row))
    session_state["accepted_primitive_program"] = deepcopy(accepted_program)
    session_state["primitive_rejection_feedback"] = []
    session_state["primitive_context_errors"] = []
    session_state["primitive_escalation_diagnostics"] = []
    session_state["primitive_input_diagnostics"] = []
    session_state["primitive_generation_cursor"] = cursor + 1
    _clear_primitive_event_guard(session_state)
    _record_primitive_authoring_memo(
        session_state=session_state,
        outline_event=active_event,
        primitive_steps=primitive_steps,
        accepted_at_turn=turn_index,
    )
    # Clear served context once the event is accepted; the next event starts fresh.
    session_state["primitive_served_context"] = {}
    turn_entry["accepted_primitive_event"] = deepcopy(accepted_row)
    turn_entry["accepted_primitive_macros"] = [deepcopy(accepted_row)]
    turn_entry["projected_snapshot"] = deepcopy(accepted_row.get("projected_snapshot") or {})
    turn_entry["primitive_steps"] = deepcopy(primitive_steps)

    is_final_event = int(session_state.get("primitive_generation_cursor") or 0) >= len(
        accepted_prefix
    )
    if is_final_event:
        session_state["status"] = "paused_after_primitive_generation"
        return "draft_ready", turn_entry

    session_state["status"] = "paused_after_primitive_turn"
    return "primitive_steps_ready", turn_entry


__all__ = [
    "_active_primitive_outline_event",
    "_missing_primitive_outline_events",
    "_primitive_feedback_row",
    "_primitive_catalog_for_resource",
    "_primitive_resource_sequence_findings",
    "_primitive_start_snapshot",
    "_primitive_grounding_context",
    "_serve_context_requests",
    "_validate_authored_plan",
    "_validate_single_event_primitive_steps",
    "_accepted_program_row",
    "_primitive_batch_session_state",
    "generate_primitive_batch_with_llm_agent",
    "_visible_primitive_catalog_card",
    "_visible_primitive_catalog_names",
    "_compact_active_event_token",
    "_primitive_authoring_event_context",
    "_log_primitive_turn_summary",
    "_record_primitive_authoring_memo",
    "_relevant_authoring_memo",
    "_capability_decomposition_for_resource",
    "_PUBLISHED_REF_SCHEMA",
    "build_primitive_generation_prompt_context",
    "_handle_primitive_generation_phase",
]
