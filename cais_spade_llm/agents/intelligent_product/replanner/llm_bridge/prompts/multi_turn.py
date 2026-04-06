"""Multi-turn prompt builders and response schemas for the active v4 bridge."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.prompts.shared import (
    build_shared_fact_sections,
    json_block,
    prompt_part_facts,
)


_PHASE_TITLES = {
    "grounding": "Grounding Assessment",
    "outline": "Recovery Outline",
    "primitive_generation": "Primitive Generation",
    "finalize": "Finalize Proposal",
}


def _compact_session_state(session_state: dict[str, Any]) -> dict[str, Any]:
    state = dict(session_state or {})
    return {
        "session_id": str(state.get("session_id") or "").strip(),
        "current_phase": str(state.get("current_phase") or "").strip(),
        "turn_index": int(state.get("turn_index") or 0),
        "max_turns": int(state.get("max_turns") or 0),
        "observation_count": int(state.get("observation_count") or 0),
        "observation_fact_count": len(dict(state.get("observation_fact_ledger") or {})),
        "max_observations": int(state.get("max_observations") or 0),
        "max_observe_batch": int(state.get("max_observe_batch") or 0),
        "pruned_action_count": len(list(state.get("pruned_actions") or [])),
        "stop_after_phase": str(state.get("stop_after_phase") or "").strip() or None,
    }


def _compact_outline_tasks(outline_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_tasks: list[dict[str, Any]] = []
    for raw_task in (outline_tasks or []):
        if not isinstance(raw_task, dict):
            continue
        outline_id = str(raw_task.get("outline_id") or "").strip()
        resource_jid = str(raw_task.get("resource_jid") or "").strip()
        if not outline_id and not resource_jid:
            continue
        row: dict[str, Any] = {
            "outline_id": outline_id or None,
            "resource_jid": resource_jid or None,
        }
        part_name = str(raw_task.get("part_name") or "").strip()
        if part_name:
            row["part_name"] = part_name
        depends_on = [
            str(item).strip()
            for item in (raw_task.get("depends_on") or [])
            if str(item).strip()
        ]
        if depends_on:
            row["depends_on"] = depends_on
        compact_tasks.append(row)
    return compact_tasks


def _outline_task_summary(raw_task: dict[str, Any]) -> str:
    if not isinstance(raw_task, dict):
        return ""
    outline_id = str(raw_task.get("outline_id") or "").strip()
    description = str(raw_task.get("description") or "").strip()
    resource_jid = str(raw_task.get("resource_jid") or "").strip()
    part_name = str(raw_task.get("part_name") or "").strip()
    if description:
        prefix = f"{outline_id}: " if outline_id else ""
        return f"{prefix}{description}"
    fallback_bits = [bit for bit in (resource_jid, part_name) if bit]
    if not fallback_bits and outline_id:
        return outline_id
    fallback = " / ".join(fallback_bits)
    if outline_id and fallback:
        return f"{outline_id}: {fallback}"
    return fallback


def _outline_memory_sanitize_reason(reason: Any) -> str:
    text = str(reason or "").strip()
    if not text:
        return ""
    text = re.sub(r"cond_[A-Za-z0-9]+", "<condition>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _outline_memory_reason_task_id(reason: Any) -> str:
    text = str(reason or "").strip()
    if not text:
        return ""
    match = re.match(r"task '([^']+)':", text)
    return str(match.group(1) or "").strip() if match else ""


def _outline_memory_from_session_state(
    session_state: dict[str, Any],
) -> list[dict[str, Any]]:
    memory: list[dict[str, Any]] = []
    for raw_turn in reversed(list(session_state.get("turns") or [])):
        if not isinstance(raw_turn, dict):
            continue
        if str(raw_turn.get("phase") or "").strip().lower() != "outline":
            continue
        turn_index = int(raw_turn.get("turn_index") or 0)
        outline_tasks = [
            deepcopy(row)
            for row in (raw_turn.get("outline_tasks") or [])
            if isinstance(row, dict)
        ]
        accepted_rows_source = [
            deepcopy(row)
            for row in (raw_turn.get("accepted_prefix") or [])
            if isinstance(row, dict)
        ]
        outline_validation = dict(raw_turn.get("outline_validation") or {})
        if (
            not accepted_rows_source
            and str(outline_validation.get("status") or "").strip().lower() == "passed"
        ):
            accepted_rows_source = deepcopy(outline_tasks)

        accepted_rows = [
            _outline_task_summary(row)
            for row in accepted_rows_source
            if _outline_task_summary(row)
        ]
        accepted_ids = {
            str(dict(row).get("outline_id") or "").strip()
            for row in accepted_rows_source
            if str(dict(row).get("outline_id") or "").strip()
        }
        outline_rows_by_id = {
            str(dict(row).get("outline_id") or "").strip(): deepcopy(row)
            for row in outline_tasks
            if isinstance(row, dict) and str(dict(row).get("outline_id") or "").strip()
        }
        rejected_ids: list[str] = []
        for finding in (
            list(raw_turn.get("task_outline_validation_findings") or [])
            + list(raw_turn.get("outline_validation_findings") or [])
        ):
            if not isinstance(finding, dict):
                continue
            task_id = str(finding.get("task_id") or "").strip()
            if task_id and task_id in outline_rows_by_id and task_id not in rejected_ids:
                rejected_ids.append(task_id)
        if not rejected_ids and outline_tasks:
            for row in outline_tasks:
                outline_id = str(dict(row).get("outline_id") or "").strip()
                if outline_id and outline_id not in accepted_ids:
                    rejected_ids.append(outline_id)
                    break

        rejected_rows = [
            _outline_task_summary(outline_rows_by_id[row_id])
            for row_id in rejected_ids
            if row_id in outline_rows_by_id and _outline_task_summary(outline_rows_by_id[row_id])
        ]
        reasons: list[str] = []
        raw_reasons = [
            _outline_memory_sanitize_reason(item)
            for item in (raw_turn.get("validation_violations") or [])
            if _outline_memory_sanitize_reason(item)
        ]
        if rejected_ids:
            for reason in raw_reasons:
                task_id = _outline_memory_reason_task_id(reason)
                if task_id and task_id not in rejected_ids:
                    continue
                if reason not in reasons:
                    reasons.append(reason)
        else:
            reasons = raw_reasons

        row: dict[str, Any] = {"turn": turn_index}
        if accepted_rows:
            row["accepted_rows"] = accepted_rows
        if rejected_rows:
            row["rejected_rows"] = rejected_rows
        if reasons:
            row["reasons"] = reasons
        if len(row) > 1:
            memory.append(row)
    return memory


def _outline_memory_text(memory: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in (memory or []):
        turn_index = int(row.get("turn") or 0)
        lines.append(f"Turn {turn_index}")
        accepted_rows = [
            str(item).strip()
            for item in (row.get("accepted_rows") or [])
            if str(item).strip()
        ]
        if accepted_rows:
            lines.append("Accepted:")
            lines.extend(f"- {item}" for item in accepted_rows)
        rejected_rows = [
            str(item).strip()
            for item in (row.get("rejected_rows") or [])
            if str(item).strip()
        ]
        if rejected_rows:
            lines.append("Rejected:")
            lines.extend(f"- {item}" for item in rejected_rows)
        reasons = [
            str(item).strip()
            for item in (row.get("reasons") or [])
            if str(item).strip()
        ]
        if reasons:
            lines.append("Why:")
            lines.extend(f"- {item}" for item in reasons)
        lines.append("")
    while lines and not str(lines[-1]).strip():
        lines.pop()
    return "\n".join(lines)


def _compact_turn_history(session_state: dict[str, Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for raw_turn in (session_state.get("turns") or []):
        if not isinstance(raw_turn, dict):
            continue
        row: dict[str, Any] = {
            "turn_index": int(raw_turn.get("turn_index") or 0),
            "phase": str(raw_turn.get("phase") or "").strip(),
        }
        raw_response = deepcopy(raw_turn.get("raw_response") or {})
        if str(raw_turn.get("phase") or "").strip().lower() == "grounding":
            if not isinstance(raw_response, dict):
                raw_response = {}
            else:
                if not isinstance(raw_response.get("blocking_reasons"), list):
                    legacy_blocking = raw_response.get("blocking_summary")
                    if isinstance(legacy_blocking, list):
                        raw_response["blocking_reasons"] = deepcopy(legacy_blocking)
                raw_response.pop("blocking_summary", None)
                sanitized_requests: list[dict[str, Any]] = []
                for request in (raw_response.get("observe_requests") or []):
                    if not isinstance(request, dict):
                        continue
                    request_row = deepcopy(request)
                    request_row.pop("resource_jid", None)
                    sanitized_requests.append(request_row)
                if sanitized_requests:
                    raw_response["observe_requests"] = sanitized_requests
        elif str(raw_turn.get("phase") or "").strip().lower() == "outline":
            if not isinstance(raw_response, dict):
                raw_response = {}
            else:
                raw_response = {}
        if raw_response not in ({}, [], "", None):
            row["model_response"] = raw_response
        observation_results = [
            {
                key: deepcopy(value)
                for key, value in dict(observation_result or {}).items()
                if key != "resource_jid"
            }
            for observation_result in (raw_turn.get("observation_results") or [])
            if isinstance(observation_result, dict)
        ]
        if observation_results:
            row["observation_results"] = observation_results
        error = str(raw_turn.get("error") or "").strip()
        if error:
            row["runtime_error"] = error
        runtime_decision = str(raw_turn.get("decision") or "").strip()
        model_decision = ""
        if isinstance(raw_response, dict):
            model_decision = str(raw_response.get("decision") or "").strip()
        if runtime_decision and runtime_decision != model_decision:
            row["runtime_decision"] = runtime_decision
        outline_validation = dict(raw_turn.get("outline_validation") or {})
        validation_status = str(outline_validation.get("status") or "").strip()
        if validation_status:
            row["outline_validation"] = {"status": validation_status}
            validation_reason = str(outline_validation.get("reason") or "").strip()
            if validation_reason:
                row["outline_validation"]["reason"] = validation_reason
        outline_revision_coverage = dict(raw_turn.get("outline_revision_coverage") or {})
        coverage_status = str(outline_revision_coverage.get("status") or "").strip()
        if coverage_status:
            row["outline_revision_coverage"] = {"status": coverage_status}
            coverage_violations = [
                str(item).strip()
                for item in (outline_revision_coverage.get("violations") or [])
                if str(item).strip()
            ]
            if coverage_violations:
                row["outline_revision_coverage"]["violations"] = coverage_violations
        history.append(row)
    return history


def _previous_outline_attempt(session_state: dict[str, Any]) -> dict[str, Any]:
    latest_outline_turn: dict[str, Any] = {}
    for raw_turn in (session_state.get("turns") or []):
        if not isinstance(raw_turn, dict):
            continue
        if str(raw_turn.get("phase") or "").strip().lower() != "outline":
            continue
        latest_outline_turn = raw_turn
    if not latest_outline_turn:
        return {}

    previous_attempt: dict[str, Any] = {}
    runtime_decision = str(latest_outline_turn.get("decision") or "").strip()
    if runtime_decision:
        previous_attempt["runtime_decision"] = runtime_decision
    outline_validation = dict(latest_outline_turn.get("outline_validation") or {})
    if outline_validation:
        previous_attempt["outline_validation"] = deepcopy(outline_validation)
    validation_violations = [
        str(item).strip()
        for item in (latest_outline_turn.get("validation_violations") or [])
        if str(item).strip()
    ]
    if validation_violations:
        previous_attempt["validation_violations"] = validation_violations
    return previous_attempt


def _outline_prompt_resource_facts(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered_rows: list[dict[str, Any]] = []
    for raw_row in (resources or []):
        if not isinstance(raw_row, dict):
            continue
        row = deepcopy(raw_row)
        row.pop("availability", None)
        rendered_rows.append(row)
    return rendered_rows


def _outline_prompt_part_facts(part_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered_rows = prompt_part_facts(part_facts)
    for row in rendered_rows:
        if not isinstance(row, dict):
            continue
        row.pop("nominal_requirement_resource_jid", None)
    return rendered_rows


def _outline_prompt_loaded_safety_rules(
    safety_rules: list[dict[str, Any]],
    obligation_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    compact_rules: list[dict[str, Any]] = []
    for raw_rule in (safety_rules or []):
        if not isinstance(raw_rule, dict):
            continue
        row: dict[str, Any] = {}
        rule_id = str(raw_rule.get("rule_id") or "").strip()
        if rule_id:
            row["rule_id"] = rule_id
        constraint_type = str(raw_rule.get("constraint_type") or "").strip()
        if constraint_type:
            row["constraint_type"] = constraint_type
        if row:
            compact_rules.append(row)
    return {
        "obligation_targets": deepcopy(obligation_targets or []),
        "loaded_safety_rules": compact_rules,
    }


def _outline_prompt_relevant_requirements(
    requirements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact_requirements: list[dict[str, Any]] = []
    for raw_requirement in (requirements or []):
        if not isinstance(raw_requirement, dict):
            continue
        row: dict[str, Any] = {}
        requirement_id = str(raw_requirement.get("requirement_id") or "").strip()
        if requirement_id:
            row["requirement_id"] = requirement_id
        status = str(raw_requirement.get("status") or "").strip()
        if status:
            row["status"] = status
        product = str(raw_requirement.get("product") or "").strip()
        if product:
            row["product"] = product
        pending_task_ids = [
            str(item).strip()
            for item in (raw_requirement.get("pending_task_ids") or [])
            if str(item).strip()
        ]
        if pending_task_ids:
            row["pending_task_ids"] = pending_task_ids
        if row:
            compact_requirements.append(row)
    return compact_requirements


def _outline_prompt_llm_input(llm_input: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(llm_input or {})
    payload.pop("bridge_safety_context", None)
    observed_runtime_state = dict(payload.get("observed_runtime_state") or {})
    payload["observed_runtime_state"] = {
        **observed_runtime_state,
        "resources": _outline_prompt_resource_facts(observed_runtime_state.get("resources") or []),
    }
    payload["part_facts"] = _outline_prompt_part_facts(payload.get("part_facts") or [])
    payload["loaded_safety_rules"] = _outline_prompt_loaded_safety_rules(
        payload.get("loaded_safety_rules") or [],
        payload.get("obligation_targets") or [],
    )["loaded_safety_rules"]
    payload["relevant_assembly_requirements"] = _outline_prompt_relevant_requirements(
        payload.get("relevant_assembly_requirements") or []
    )
    return payload


def _outline_prompt_validation_findings(
    outline_validation_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scalar_fields = (
        "task_id",
        "resource_jid",
        "part_name",
        "constraint_family",
        "constraint_code",
        "constraint_owner",
        "validation_status",
        "pose_source",
        "kind",
        "entity",
        "expected",
        "actual",
        "blocking_rule_id",
        "blocked_nominal_task_id",
        "failed_reason",
        "named_pose",
        "conflicting_part",
        "current_holder_resource_jid",
        "blocked_task_id",
        "blocked_part_name",
        "blocked_resource_jid",
        "rule_id",
    )
    list_fields = (
        "failed_axes",
        "required_dependency_ids",
        "dependency_ids",
        "claimed_task_ids",
        "state_tokens",
    )
    object_fields = ("pose", "workspace_bounds")
    rendered_rows: list[dict[str, Any]] = []
    for raw_row in (outline_validation_findings or []):
        if not isinstance(raw_row, dict):
            continue
        row = dict(raw_row)
        rendered_row: dict[str, Any] = {}
        for field_name in scalar_fields:
            value = row.get(field_name)
            if value not in (None, "", [], {}):
                rendered_row[field_name] = deepcopy(value)
        for field_name in list_fields:
            values = [
                str(item).strip()
                for item in (row.get(field_name) or [])
                if str(item).strip()
            ]
            if values:
                rendered_row[field_name] = values
        for field_name in object_fields:
            value = row.get(field_name)
            if value not in (None, "", [], {}):
                rendered_row[field_name] = deepcopy(value)
        if rendered_row:
            rendered_rows.append(rendered_row)
    return rendered_rows


def _outline_prompt_pruned_actions(
    pruned_actions: list[dict[str, Any]],
) -> list[str]:
    rendered_rows: list[str] = []
    inferred_constraint_codes = {
        "observed_pose_unreachable": "workspace_unreachable",
        "resource_holds_part": "holder_conflict",
        "part_held_by_other": "resource_holds_other_part",
        "required_part_not_held": "required_part_not_held",
        "source_reference_unavailable": "source_reference_unavailable",
        "unsupported_resource_target": "unsupported_resource_target",
        "gripper_closed_without_target_part": "gripper_occupancy_conflict",
        "condition_unmet": "blocker_open",
        "named_pose_unavailable": "named_pose_unavailable",
    }
    for raw_row in (pruned_actions or []):
        if not isinstance(raw_row, dict):
            continue
        if str(raw_row.get("summary") or "").strip():
            rendered_rows.append(str(raw_row.get("summary") or "").strip())
            continue
        action = dict(raw_row.get("action") or {})
        guard = dict(raw_row.get("guard") or {})
        summary = _outline_task_summary(dict(raw_row.get("task") or {}))
        if not summary:
            summary = _outline_task_summary(
                {
                    "outline_id": str(action.get("resource_jid") or "").strip(),
                    "description": str(raw_row.get("reason") or "").strip(),
                }
            )
        constraint_code = str(guard.get("constraint_code") or "").strip()
        if not constraint_code:
            constraint_code = str(
                inferred_constraint_codes.get(str(guard.get("kind") or "").strip(), "")
            ).strip()
        reason = str(raw_row.get("reason") or "").strip()
        if not reason and constraint_code:
            reason = constraint_code.replace("_", " ")
        if summary and reason:
            rendered_rows.append(f"{summary} — {reason}")
        elif summary:
            rendered_rows.append(summary)
        elif reason:
            rendered_rows.append(reason)
    return rendered_rows


_PERSISTENT_PRUNE_CONSTRAINT_CODES = {
    "workspace_unreachable",
    "unsupported_resource_target",
    "named_pose_unavailable",
}


def _outline_prompt_grouped_pruned_actions(
    pruned_actions: list[dict[str, Any]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "temporary": [],
        "persistent": [],
    }
    inferred_constraint_codes = {
        "observed_pose_unreachable": "workspace_unreachable",
        "resource_holds_part": "holder_conflict",
        "part_held_by_other": "resource_holds_other_part",
        "required_part_not_held": "required_part_not_held",
        "source_reference_unavailable": "source_reference_unavailable",
        "unsupported_resource_target": "unsupported_resource_target",
        "gripper_closed_without_target_part": "gripper_occupancy_conflict",
        "condition_unmet": "blocker_open",
        "named_pose_unavailable": "named_pose_unavailable",
    }
    for raw_row in (pruned_actions or []):
        if not isinstance(raw_row, dict):
            continue
        constraint_code = str(dict(raw_row.get("guard") or {}).get("constraint_code") or "").strip()
        if not constraint_code:
            constraint_code = str(
                inferred_constraint_codes.get(
                    str(dict(raw_row.get("guard") or {}).get("kind") or "").strip(),
                    str(dict(raw_row).get("constraint_code") or "").strip(),
                )
            ).strip()
        bucket = (
            "persistent"
            if constraint_code in _PERSISTENT_PRUNE_CONSTRAINT_CODES
            else "temporary"
        )
        summary_rows = _outline_prompt_pruned_actions([raw_row])
        grouped[bucket].extend(summary_rows)
    return {
        key: value
        for key, value in grouped.items()
        if value
    }


def _outline_prompt_blocked_nominal_tasks(
    recovery_gap_state: dict[str, Any],
) -> list[dict[str, Any]]:
    rendered_rows: list[dict[str, Any]] = []
    for raw_row in (recovery_gap_state.get("blocked_nominal_tasks") or []):
        if not isinstance(raw_row, dict):
            continue
        row = deepcopy(raw_row)
        row.pop("blocked_by_condition_ids", None)
        row.pop("blocking_condition_ids", None)
        rendered_rows.append(row)
    return rendered_rows


def _outline_prompt_unmet_continuation_conditions(
    recovery_gap_state: dict[str, Any],
) -> list[dict[str, Any]]:
    rendered_rows: list[dict[str, Any]] = []
    for raw_row in (recovery_gap_state.get("unmet_continuation_conditions") or []):
        if not isinstance(raw_row, dict):
            continue
        row: dict[str, Any] = {}
        for field_name in (
            "kind",
            "entity",
            "expected",
            "actual",
            "part_name",
            "resource_jid",
            "target_location",
            "requirement_id",
        ):
            value = raw_row.get(field_name)
            if value not in (None, "", [], {}):
                row[field_name] = deepcopy(value)
        source_task_id = str(raw_row.get("source_task_id") or "").strip()
        if source_task_id:
            row["blocked_nominal_task_id"] = source_task_id
        blocking_rule_id = str(
            raw_row.get("blocking_rule_id") or raw_row.get("rule_id") or ""
        ).strip()
        if blocking_rule_id:
            row["blocking_rule_id"] = blocking_rule_id
        if row:
            rendered_rows.append(row)
    return rendered_rows


def _outline_prompt_current_blockers(
    llm_input: dict[str, Any],
    recovery_gap_state: dict[str, Any],
) -> list[str]:
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})
    resource_rows_by_jid = {
        str(row.get("resource_jid") or "").strip(): dict(row)
        for row in (observed_runtime_state.get("resources") or [])
        if isinstance(row, dict) and str(row.get("resource_jid") or "").strip()
    }
    part_rows_by_name = {
        str(row.get("part_name") or "").strip(): dict(row)
        for row in (llm_input.get("part_facts") or [])
        if isinstance(row, dict) and str(row.get("part_name") or "").strip()
    }
    blocked_tasks_by_id = {
        str(row.get("id") or "").strip(): dict(row)
        for row in (recovery_gap_state.get("blocked_nominal_tasks") or [])
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    fallback_parts = _outline_fault_event_fallback_parts(llm_input)
    lines: list[str] = []

    for raw_condition in (recovery_gap_state.get("unmet_continuation_conditions") or []):
        if not isinstance(raw_condition, dict):
            continue
        row = {
            "kind": raw_condition.get("kind"),
            "entity": raw_condition.get("entity"),
            "expected": raw_condition.get("expected"),
            "actual": raw_condition.get("actual"),
            "part_name": raw_condition.get("part_name"),
            "resource_jid": raw_condition.get("resource_jid"),
            "blocking_reason": raw_condition.get("blocking_reason"),
        }
        filtered = {k: v for k, v in row.items() if v}
        if filtered:
            lines.append(str(filtered))
            
    for raw_task in (recovery_gap_state.get("blocked_nominal_tasks") or []):
        if not isinstance(raw_task, dict):
            continue
        task_id = str(raw_task.get("id") or "").strip()
        resource_jid = str(raw_task.get("resource") or "").strip()
        part_name = str(raw_task.get("part") or "").strip()
        resource_row = dict(resource_rows_by_jid.get(resource_jid) or {})
        held_part = str(resource_row.get("held_part") or "").strip()
        if task_id and resource_jid and part_name and held_part and held_part != part_name:
            lines.append({
                "constraint": "gripper_occupancy_conflict",
                "resource_jid": resource_jid,
                "held_part": held_part,
                "requested_part": part_name
            })

    return [str(line) for line in lines]


def _outline_fault_event_fallback_parts(llm_input: dict[str, Any]) -> list[str]:
    fault_event = dict(llm_input.get("fault_event") or {})
    return [
        str(item).strip()
        for item in (fault_event.get("affected_part_names") or [])
        if str(item).strip()
    ]


def _outline_extract_blocker_part_names(
    *,
    blocking_reason: str,
    part_rows_by_name: dict[str, dict[str, Any]],
    fallback_parts: list[str],
) -> list[str]:
    preferred_fallback = [
        str(part_name).strip()
        for part_name in fallback_parts
        if str(part_name).strip()
    ]
    if preferred_fallback:
        return preferred_fallback
    blocker_text = str(blocking_reason or "").strip().lower()
    blocker_parts = [
        part_name
        for part_name in part_rows_by_name
        if part_name and part_name.lower() in blocker_text
    ]
    return blocker_parts or preferred_fallback


def _outline_prompt_continuation_clearance_facts(
    llm_input: dict[str, Any],
    recovery_gap_state: dict[str, Any],
) -> list[dict[str, Any]]:
    part_rows_by_name = {
        str(row.get("part_name") or "").strip(): dict(row)
        for row in (llm_input.get("part_facts") or [])
        if isinstance(row, dict) and str(row.get("part_name") or "").strip()
    }
    fallback_parts = _outline_fault_event_fallback_parts(llm_input)
    rendered_rows: list[dict[str, Any]] = []
    for raw_condition in (recovery_gap_state.get("unmet_continuation_conditions") or []):
        if not isinstance(raw_condition, dict):
            continue
        row: dict[str, Any] = {}
        condition_id = str(raw_condition.get("condition_id") or "").strip()
        if condition_id:
            row["condition_id"] = condition_id
        kind = str(raw_condition.get("kind") or "").strip()
        if kind:
            row["kind"] = kind
        blocked_nominal_task_id = str(raw_condition.get("source_task_id") or "").strip()
        if blocked_nominal_task_id:
            row["blocked_nominal_task_id"] = blocked_nominal_task_id
        blocking_rule_id = str(
            raw_condition.get("blocking_rule_id") or raw_condition.get("rule_id") or ""
        ).strip()
        if blocking_rule_id:
            row["blocking_rule_id"] = blocking_rule_id

        if kind == "focused_resource_terminal_state":
            resource_jid = str(
                raw_condition.get("entity") or raw_condition.get("resource_jid") or ""
            ).strip()
            expected_state = str(raw_condition.get("expected") or "").strip()
            if resource_jid and expected_state:
                row["clear_when"] = {
                    "entity_kind": "resource",
                    "entity": resource_jid,
                    "field": "current_state",
                    "equals": expected_state,
                }
        elif kind == "safety_blocked_suffix_task":
            blocker_part_names = _outline_extract_blocker_part_names(
                blocking_reason=str(raw_condition.get("blocking_reason") or "").strip(),
                part_rows_by_name=part_rows_by_name,
                fallback_parts=fallback_parts,
            )
            blocker_part_name = blocker_part_names[0] if blocker_part_names else ""
            part_row = dict(part_rows_by_name.get(blocker_part_name) or {})
            goal_location = str(part_row.get("goal_location") or "").strip()
            satisfies_any = [
                {"field": "current_state", "equals": "placed"},
                {"field": "current_state", "equals": "assembled"},
            ]
            if goal_location:
                satisfies_any.append(
                    {"field": "current_location", "equals": goal_location}
                )
            if blocker_part_name and satisfies_any:
                row["clear_when"] = {
                    "entity_kind": "part",
                    "entity": blocker_part_name,
                    "satisfies_any": satisfies_any,
                }
        if row:
            rendered_rows.append(row)
    return rendered_rows


def _outline_prompt_resource_transition_facts(
    llm_input: dict[str, Any],
) -> list[dict[str, Any]]:
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})
    part_facts = [
        dict(row)
        for row in (llm_input.get("part_facts") or [])
        if isinstance(row, dict)
    ]
    rendered_rows: list[dict[str, Any]] = []
    for raw_resource in (observed_runtime_state.get("resources") or []):
        if not isinstance(raw_resource, dict):
            continue
        resource_row = dict(raw_resource)
        resource_jid = str(resource_row.get("resource_jid") or "").strip()
        if not resource_jid:
            continue
        row: dict[str, Any] = {"resource_jid": resource_jid}
        resource_type = str(resource_row.get("resource_type") or "").strip()
        if resource_type:
            row["resource_type"] = resource_type

        tracked_state_fields = [
            field_name
            for field_name in (
                "current_state",
                "gripper_state",
                "held_part",
                "current_location",
                "current_pose",
            )
            if field_name in resource_row
        ]
        if tracked_state_fields:
            row["tracked_state_fields"] = tracked_state_fields

        valid_reference_families: list[str] = []
        if list(resource_row.get("named_poses") or []) or list(
            resource_row.get("available_named_poses") or []
        ):
            valid_reference_families.append("named_pose")
        if (
            str(resource_row.get("current_location") or "").strip()
            or tracked_state_fields
        ):
            valid_reference_families.append("location")
        if isinstance(resource_row.get("current_pose"), dict) or isinstance(
            resource_row.get("workspace_bounds"), dict
        ):
            valid_reference_families.append("pose")
        if valid_reference_families:
            row["valid_reference_families"] = valid_reference_families

        has_holder_semantics = bool(
            "held_part" in resource_row
            or "gripper_state" in resource_row
            or any(
                str(dict(part_row).get("current_holder_resource_jid") or "").strip()
                == resource_jid
                for part_row in part_facts
            )
        )
        modeled_invariants: list[dict[str, Any]] = []
        modeled_transitions: list[dict[str, Any]] = []
        if has_holder_semantics:
            modeled_invariants.append(
                {
                    "kind": "single_part_occupancy",
                    "field": "held_part",
                    "max_items": 1,
                }
            )
            modeled_transitions.extend(
                [
                    {
                        "kind": "holder_clearance",
                        "pre_state": {"held_part": "<part>"},
                        "post_state": {"held_part": None},
                    },
                    {
                        "kind": "holder_assignment",
                        "pre_state": {"held_part": None},
                        "post_state": {"held_part": "<part>"},
                    },
                ]
            )
        mutable_resource_fields = [
            field_name
            for field_name in ("current_state", "current_location", "current_pose")
            if field_name in tracked_state_fields
        ]
        if mutable_resource_fields:
            modeled_transitions.append(
                {
                    "kind": "resource_state_update",
                    "fields": mutable_resource_fields,
                }
            )
        if modeled_invariants:
            row["modeled_invariants"] = modeled_invariants
        if modeled_transitions:
            row["modeled_transitions"] = modeled_transitions
        rendered_rows.append(row)
    return rendered_rows


def _outline_prompt_blocked_task_dependency_facts(
    recovery_gap_state: dict[str, Any],
    *,
    continuation_clearance_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    clearance_by_id = {
        str(dict(row).get("condition_id") or "").strip(): dict(row)
        for row in (continuation_clearance_facts or [])
        if isinstance(row, dict) and str(dict(row).get("condition_id") or "").strip()
    }
    rendered_rows: list[dict[str, Any]] = []
    for raw_row in (recovery_gap_state.get("blocked_nominal_tasks") or []):
        if not isinstance(raw_row, dict):
            continue
        blocked_by_condition_ids = [
            str(item).strip()
            for item in (
                raw_row.get("blocked_by_condition_ids")
                or raw_row.get("blocking_condition_ids")
                or []
            )
            if str(item).strip()
        ]
        if not blocked_by_condition_ids:
            continue
        row: dict[str, Any] = {
            "task_id": str(raw_row.get("id") or "").strip() or None,
            "resource_jid": str(raw_row.get("resource") or "").strip() or None,
            "part_name": str(raw_row.get("part") or "").strip() or None,
            "blocked_by_condition_ids": blocked_by_condition_ids,
            "clearance_dependencies": [],
        }
        for condition_id in blocked_by_condition_ids:
            dependency = {"condition_id": condition_id}
            matched_condition = clearance_by_id.get(condition_id) or {}
            clear_when = deepcopy(matched_condition.get("clear_when"))
            if clear_when not in (None, "", [], {}):
                dependency["clear_when"] = clear_when
            row["clearance_dependencies"].append(dependency)
        rendered_rows.append(
            {
                key: deepcopy(value)
                for key, value in row.items()
                if value not in (None, "", [], {})
            }
        )
    return rendered_rows


def _outline_prompt_repair_contract(
    *,
    outline_validation_findings: list[dict[str, Any]],
    required_addressed_validation_findings: list[dict[str, Any]],
    pruned_actions: list[dict[str, Any]],
    session_state: dict[str, Any],
) -> dict[str, Any] | None:
    repair_contract: dict[str, Any] = {}
    if outline_validation_findings:
        repair_contract["unresolved_findings"] = deepcopy(outline_validation_findings)
    if required_addressed_validation_findings:
        repair_contract["required_addressed_validation_findings"] = deepcopy(
            required_addressed_validation_findings
        )
    if pruned_actions:
        grouped_pruned_actions = _outline_prompt_grouped_pruned_actions(pruned_actions)
        if grouped_pruned_actions:
            repair_contract["active_pruned_actions"] = grouped_pruned_actions
    accepted_prefix = [
        _outline_task_summary(row)
        for row in (
            session_state.get("accepted_outline_prefix")
            or session_state.get("accepted_prefix")
            or []
        )
        if isinstance(row, dict) and _outline_task_summary(row)
    ]
    if accepted_prefix:
        repair_contract["accepted_prefix"] = accepted_prefix
    outline_revision_coverage = deepcopy(session_state.get("outline_revision_coverage") or {})
    if outline_revision_coverage not in ({}, [], "", None):
        revision_feedback = {
            "status": str(outline_revision_coverage.get("status") or "").strip(),
            "required_addressed_validation_findings": deepcopy(
                outline_revision_coverage.get("required_addressed_validation_findings") or []
            ),
            "provided_addressed_validation_findings": deepcopy(
                outline_revision_coverage.get("provided_addressed_validation_findings") or []
            ),
        }
        if outline_revision_coverage.get("missing_required_refs"):
            revision_feedback["missing_required_refs"] = deepcopy(
                outline_revision_coverage.get("missing_required_refs") or []
            )
        if outline_revision_coverage.get("unexpected_refs"):
            revision_feedback["unexpected_refs"] = deepcopy(
                outline_revision_coverage.get("unexpected_refs") or []
            )
        repair_contract["revision_feedback"] = revision_feedback
    return repair_contract or None


def _build_outline_fact_sections(
    llm_input: dict[str, Any],
    *,
    session_state: dict[str, Any],
    recovery_gap_state: dict[str, Any],
    repair_contract: dict[str, Any] | None,
) -> list[str]:
    payload = dict(llm_input or {})
    observed_runtime_state = dict(payload.get("observed_runtime_state") or {})
    outline_memory = _outline_memory_from_session_state(session_state)
    current_blockers = _outline_prompt_current_blockers(payload, recovery_gap_state)
    safety_section = {
        "obligation_targets": deepcopy(payload.get("obligation_targets") or []),
        "loaded_safety_rules": deepcopy(payload.get("loaded_safety_rules") or []),
    }
    sections: list[str] = [
        "Fault Event",
        json_block(payload.get("fault_event") or {}),
    ]
    if outline_memory:
        sections.extend(["", "Outline Memory", _outline_memory_text(outline_memory)])
    sections.extend(
        [
            "",
            "Resources",
            json_block(observed_runtime_state.get("resources") or []),
            "",
            "Parts",
            json_block(prompt_part_facts(payload.get("part_facts") or [])),
        ]
    )
    sections.extend(
        [
            "",
            "Safety Rules",
            json_block(safety_section),
            "",
            "Assembly Requirements",
            json_block(payload.get("relevant_assembly_requirements") or []),
        ]
    )
    sections.extend(
        [
            "",
            "Blocked Tasks",
            json_block(_outline_prompt_blocked_nominal_tasks(recovery_gap_state)),
            "",
            "Current Blockers",
            json_block(current_blockers),
        ]
    )
    if repair_contract not in ({}, [], "", None):
        sections.extend(["", "Repair Contract", json_block(repair_contract)])
    return sections


def _latest_session_delta(session_state: dict[str, Any]) -> dict[str, Any]:
    turns = [
        dict(raw_turn)
        for raw_turn in (session_state.get("turns") or [])
        if isinstance(raw_turn, dict)
    ]
    if not turns:
        return {}
    latest_turn = turns[-1]
    delta: dict[str, Any] = {
        "turn_index": int(latest_turn.get("turn_index") or 0),
        "phase": str(latest_turn.get("phase") or "").strip(),
    }
    observation_results = [
        {
            key: deepcopy(value)
            for key, value in dict(observation_result or {}).items()
            if key not in {"resource_jid", "store_as"}
        }
        for observation_result in (latest_turn.get("observation_results") or [])
        if isinstance(observation_result, dict)
    ]
    if observation_results:
        delta["observation_results"] = observation_results
    runtime_error = str(latest_turn.get("error") or "").strip()
    if runtime_error:
        delta["runtime_error"] = runtime_error
    if len(delta) <= 2:
        return {}
    return delta


def _grounding_observation_fact_key(
    fact_type: Any,
    entity: Any,
    scope: Any = None,
) -> str:
    payload: dict[str, Any] = {
        "fact_type": str(fact_type or "").strip(),
        "entity": str(entity or "").strip(),
    }
    if scope not in (None, "", [], {}):
        payload["scope"] = deepcopy(scope)
    return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)


def _build_observation_fulfillment_status(
    session_state: dict[str, Any],
    turn_history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Cross-reference prior observe_requests with fulfilled observation facts."""
    observation_fact_ledger = {
        str(fact_key).strip(): dict(row)
        for fact_key, row in dict(session_state.get("observation_fact_ledger") or {}).items()
        if str(fact_key).strip() and isinstance(row, dict)
    }
    prior_requests: list[dict[str, Any]] = []
    for turn in turn_history:
        response = turn.get("model_response") or {}
        for req in response.get("observe_requests") or []:
            fact_type = str(req.get("fact_type") or "").strip()
            entity = str(req.get("entity") or "").strip()
            scope = deepcopy(req.get("scope"))
            fact_key = (
                _grounding_observation_fact_key(fact_type, entity, scope)
                if fact_type and entity
                else ""
            )
            fact_row = dict(observation_fact_ledger.get(fact_key) or {}) if fact_key else {}
            is_fulfilled = bool(
                fact_row
                and str(fact_row.get("validity") or "current").strip().lower() != "stale"
            )
            row = {
                "fact_type": fact_type,
                "entity": entity,
                "requested_at_turn": turn.get("turn_index"),
                "status": "fulfilled" if is_fulfilled else "pending",
            }
            if scope not in (None, "", [], {}):
                row["scope"] = scope
            prior_requests.append({
                key: value
                for key, value in row.items()
            })
    if not prior_requests:
        return None
    unfulfilled = [r for r in prior_requests if r["status"] != "fulfilled"]
    return {
        "prior_requests": prior_requests,
        "fulfilled_count": len(prior_requests) - len(unfulfilled),
        "unfulfilled_count": len(unfulfilled),
    }


def _grounding_prompt_part_facts(part_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered_rows = prompt_part_facts(part_facts)
    for row in rendered_rows:
        if not isinstance(row, dict):
            continue
        row.pop("observed_store_as", None)
        row.pop("observed_aliases", None)
    return rendered_rows


def _build_grounding_fact_sections(
    llm_input: dict[str, Any],
    *,
    session_state: dict[str, Any],
    world_observation_surface: dict[str, Any],
    turn_history: list[dict[str, Any]],
) -> list[str]:
    payload = dict(llm_input or {})
    observed_runtime_state = dict(payload.get("observed_runtime_state") or {})
    safety_section = {
        "obligation_targets": deepcopy(payload.get("obligation_targets") or []),
        "loaded_safety_rules": deepcopy(payload.get("loaded_safety_rules") or []),
    }
    latest_delta = _latest_session_delta(session_state)
    observation_store = deepcopy(session_state.get("observation_store") or {})
    observation_fact_ledger = deepcopy(session_state.get("observation_fact_ledger") or {})
    grounding_contract = deepcopy(session_state.get("grounding_contract") or {})
    sections: list[str] = [
        "Fault Event",
        json_block(payload.get("fault_event") or {}),
    ]
    fulfillment = _build_observation_fulfillment_status(session_state, turn_history)
    if fulfillment is not None:
        sections.extend(
            [
                "",
                "Observation Fulfillment Status",
                json_block(fulfillment),
            ]
        )
    if grounding_contract not in ({}, [], "", None):
        sections.extend(
            [
                "",
                "Grounding Contract",
                json_block(grounding_contract),
            ]
        )
    if latest_delta:
        sections.extend(
            [
                "",
                "Latest Session Delta",
                json_block(latest_delta),
            ]
        )
    sections.extend(
        [
            "",
            "Current Resource Facts",
            json_block(observed_runtime_state.get("resources") or []),
            "",
            "Current Part Facts",
            json_block(_grounding_prompt_part_facts(payload.get("part_facts") or [])),
            "",
            "Loaded Safety Rules",
            json_block(safety_section),
            "",
            "Relevant Assembly Requirements",
            json_block(payload.get("relevant_assembly_requirements") or []),
            "",
            "Modeled Continuation Gap",
            json_block(payload.get("modeled_continuation_gap") or {}),
            "",
            "World Observation Surface",
            json_block(deepcopy(world_observation_surface or {})),
        ]
    )
    return sections


def _grounding_contract() -> dict[str, Any]:
    return {
        "required_fields": [
            "thought",
            "decision",
            "blocking_reasons",
            "grounded_facts",
            "recovery_implications",
        ],
        "decision": ["observe", "grounded"],
        "observe_required_fields": ["observe_reason", "observe_requests"],
        "observe_request": {
            "required_fields": ["fact_type", "entity"],
            "optional_fields": ["scope", "reason"],
            "storage": "Observation results are stored internally by the runtime.",
        },
    }


def _outline_validation_ref(finding: dict[str, Any]) -> dict[str, Any]:
    ref = {
        "task_id": str(finding.get("task_id") or "").strip(),
        "pose_source": str(finding.get("pose_source") or "").strip(),
        "failed_axes": [
            str(item).strip()
            for item in (finding.get("failed_axes") or [])
            if str(item).strip()
        ],
    }
    resource_jid = str(finding.get("resource_jid") or "").strip()
    if resource_jid:
        ref["resource_jid"] = resource_jid
    for field_name in (
        "kind",
        "entity",
        "expected",
        "actual",
        "blocking_rule_id",
        "blocked_nominal_task_id",
        "claimed_task_ids",
    ):
        value = finding.get(field_name)
        if value not in (None, "", [], {}):
            ref[field_name] = deepcopy(value)
    failed_reason = str(finding.get("failed_reason") or "").strip()
    if failed_reason:
        ref["failed_reason"] = failed_reason
    return ref


def _outline_contract() -> dict[str, Any]:
    return {
        "required_fields": ["thought", "addressed_validation_findings", "outline_tasks"],
        "addressed_validation_findings": {
            "required_fields": ["task_id", "pose_source", "failed_axes"],
            "optional_fields": [
                "resource_jid",
                "failed_reason",
                "kind",
                "entity",
                "expected",
                "actual",
                "blocking_rule_id",
                "blocked_nominal_task_id",
                "condition_id",
                "claimed_task_ids",
            ],
            "usage": (
                "Use the exact refs listed in Repair Contract.required_addressed_validation_findings. "
                "Use [] when no outstanding validation findings are present."
            ),
        },
        "outline_task": {
            "required_fields": [
                "outline_id",
                "resource_jid",
                "description",
                "rationale",
                "expected_start_state",
                "expected_end_state",
                "depends_on",
            ],
            "optional_fields": [
                "part_name",
                "action_target",
                "closes_condition_ids",
                "enables_task_ids",
            ],
            "expected_state_fields": [
                "resource_state",
                "resource_location",
                "held_part",
                "part_state",
                "part_location",
                "part_holder_resource_jid",
            ],
        },
        "outline_tasks_usage": (
            "If Repair Contract.accepted_prefix is present, return only the replacement suffix "
            "that follows that accepted prefix. Returning the full outline is still allowed "
            "only when it begins with the same accepted prefix."
        ),
    }


def _primitive_generation_contract() -> dict[str, Any]:
    return {
        "required_fields": ["thought", "decision", "macro_tasks"],
        "decision": ["need_outline_revision", "draft_ready"],
        "macro_task": {
            "required_fields": [
                "resource_jid",
                "macro_name",
                "description",
                "rationale",
                "expected_start_state",
                "task_params",
                "task_metadata",
                "primitive_steps",
            ],
            "optional_fields": ["part_name"],
        },
    }


def _finalize_contract() -> dict[str, Any]:
    return {
        "required_fields": ["thought", "decision", "final_proposal"],
        "decision": [
            "final_ready",
            "need_outline_revision",
            "need_primitive_revision",
        ],
        "final_proposal": {
            "required_fields": ["thought", "primary_obligation", "macro_tasks"],
        },
    }


def _contract_for_phase(phase: str) -> dict[str, Any]:
    normalized = str(phase or "").strip().lower()
    if normalized == "grounding":
        return _grounding_contract()
    if normalized == "outline":
        return _outline_contract()
    if normalized == "primitive_generation":
        return _primitive_generation_contract()
    if normalized == "finalize":
        return _finalize_contract()
    raise ValueError(f"unsupported multi-turn phase: {phase!r}")


def build_multi_turn_phase_prompt_input(
    *,
    phase: str,
    llm_input: dict[str, Any],
    session_state: dict[str, Any],
    world_observation_surface: dict[str, Any] | None = None,
    recovery_gap_state: dict[str, Any] | None = None,
    pruned_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_phase = str(phase or "").strip().lower()
    return {
        "reasoning_mode": "multi_turn",
        "phase": normalized_phase,
        "llm_input": deepcopy(llm_input or {}),
        "session_state": deepcopy(session_state or {}),
        "world_observation_surface": deepcopy(world_observation_surface or {}),
        "recovery_gap_state": deepcopy(recovery_gap_state or {}),
        "pruned_actions": deepcopy(pruned_actions or []),
        "response_contract": _contract_for_phase(normalized_phase),
    }


def render_multi_turn_phase_prompt(prompt_input: dict[str, Any]) -> str:
    payload = deepcopy(prompt_input or {})
    llm_input = dict(payload.get("llm_input") or {})
    session_state = dict(payload.get("session_state") or {})
    phase = str(payload.get("phase") or "").strip().lower()
    response_contract = deepcopy(payload.get("response_contract") or {})
    observation_store = deepcopy(session_state.get("observation_store") or {})
    accepted_outline = deepcopy(session_state.get("accepted_outline"))
    proposal_draft = deepcopy(session_state.get("proposal_draft"))
    turn_history = _compact_turn_history(session_state)
    world_observation_surface = deepcopy(payload.get("world_observation_surface") or {})
    recovery_gap_state = deepcopy(payload.get("recovery_gap_state") or {})
    pruned_actions = deepcopy(payload.get("pruned_actions") or [])
    if phase == "outline":
        llm_input = _outline_prompt_llm_input(llm_input)
    outline_validation_findings = [
        deepcopy(row)
        for row in (session_state.get("outline_validation_findings") or [])
        if isinstance(row, dict)
    ]
    prompt_outline_validation_findings = (
        _outline_prompt_validation_findings(outline_validation_findings)
        if phase == "outline"
        else deepcopy(outline_validation_findings)
    )
    required_addressed_validation_findings = [
        _outline_validation_ref(row) for row in outline_validation_findings
    ]
    repair_contract = (
        _outline_prompt_repair_contract(
            outline_validation_findings=prompt_outline_validation_findings,
            required_addressed_validation_findings=required_addressed_validation_findings,
            pruned_actions=pruned_actions,
            session_state=session_state,
        )
        if phase == "outline"
        else None
    )

    extra_sections: list[tuple[str, Any]] = [
        ("Session State", _compact_session_state(session_state)),
        ("Session Observation Store", observation_store),
    ]
    if phase == "grounding":
        extra_sections.append(
            (
                "World Observation Surface",
                deepcopy(payload.get("world_observation_surface") or {}),
            )
        )
    elif phase == "primitive_generation":
        extra_sections.append(("Accepted Outline", accepted_outline))
    elif phase == "finalize":
        extra_sections.extend(
            [
                ("Accepted Outline", accepted_outline),
                ("Proposal Draft", proposal_draft),
            ]
        )
    if turn_history and phase != "outline":
        extra_sections.append(("Session Turn History", turn_history))

    include_surface = phase == "primitive_generation"
    sections: list[str] = [
        "Task and Role",
        (
            "You are the active replanner for a DES fallback recovery session.\n"
            f"Current phase: {_PHASE_TITLES.get(phase, phase)}.\n"
            "Use the structured context below to produce only the output required for this phase.\n"
            "Do not skip ahead to a later phase unless the current phase contract explicitly allows it."
        ),
        "",
        *(
            _build_grounding_fact_sections(
                llm_input,
                session_state=session_state,
                world_observation_surface=world_observation_surface,
                turn_history=turn_history,
            )
            if phase == "grounding"
            else (
                _build_outline_fact_sections(
                    llm_input,
                    session_state=session_state,
                    recovery_gap_state=recovery_gap_state,
                    repair_contract=repair_contract,
                )
                if phase == "outline"
                else build_shared_fact_sections(
                    llm_input,
                    extra_sections=extra_sections,
                    include_allowed_execution_surface=include_surface,
                )
            )
        ),
        "",
        "Required JSON Response Contract",
        json_block(response_contract),
        "",
        "Hard Constraints",
        (
            "- Use only the listed resources."
            if phase == "outline"
            else "- Use only the listed resources and controller primitives."
        ),
        "- Keep the response inside the current phase purpose and contract.",
        "- Do not invent observations, safety obligations, or grounded locations.",
        "- Do not contradict grounded runtime facts already present in the prompt.",
    ]
    if phase == "outline":
        sections.extend(
            [
                "- addressed_validation_findings must exactly match Repair Contract.required_addressed_validation_findings when that field is present; otherwise use [].",
                "- Each outline row must be one concrete physical recovery task for one listed resource.",
                "- Each outline task must use grounded start and end states consistent with the prompt and predecessor-projected state.",
                "- expected_start_state and expected_end_state may use only: resource_state, resource_location, held_part, part_state, part_location, part_holder_resource_jid.",
                "- Resource-only recovery rows may end in grounded terminal state 'idle' without naming the concrete home primitive; primitive generation will choose the concrete controller action.",
                "- rationale must be one short factual cause-to-effect explanation tied to the row's grounded state change, claimed closes_condition_ids, or claimed enables_task_ids.",
                "- closes_condition_ids may list only currently unmet continuation condition ids that the projected row actually clears.",
                "- enables_task_ids may list only blocked nominal task ids that become unblocked after the projected row.",
                "- depends_on may reference only outline_id values from outline rows in this same response; never use nominal task ids, condition ids, or free-form markers.",
                "- If a task stops holding a part, make the part's resulting holder, location, or pose explicit in grounded state.",
                "- Do not use robot-specific state fields such as gripper_state, current_pose, position, pose, current_location, or current_holder_resource_jid inside outline state objects.",
                "- Do not use abstract blocker states such as blocked/unblocked/safe/unsafe or invented lifecycle tokens such as placed_approached.",
                "- Do not invent any fact, location, observation, or state not already grounded in the prompt.",
                "- If Repair Contract.accepted_prefix is present, return only the replacement suffix after that accepted prefix unless you intentionally repeat the full outline with the same prefix unchanged.",
                "- Actions listed in Repair Contract.active_pruned_actions are blocked in the current state; treat temporary entries as state-dependent and persistent entries as still invalid until their blocker changes.",
                "- Outline is always validated after this phase response; do not emit a phase decision field.",
            ]
        )
    if phase == "grounding":
        sections.extend(
            [
                "- Use only the observation facts listed in World Observation Surface.",
                "- Observation requests must use the exact fact_type/entity schema shown in World Observation Surface and the response contract.",
                "- Do not emit resource_jid in grounding observe requests; runtime resolves observation facts internally.",
                "- Choose \"observe\" only when at least one unresolved observation fact is still needed for recovery grounding.",
                "- Choose \"grounded\" when the prompt already contains the needed observation facts and Observation Fulfillment Status shows no unresolved observation request remains.",
                "- If decision is \"observe\", provide a non-empty observe_reason and list only unresolved observation facts.",
                "- If decision is \"grounded\", leave observe_requests empty.",
                "- grounded_facts must summarize the concrete grounded facts established by the current prompt state and prior observations.",
                "- recovery_implications must summarize the immediate recovery-relevant consequences implied by those facts.",
                "- Do not request observations for facts already marked \"fulfilled\" in Observation Fulfillment Status.",
            ]
        )
    return "\n".join(sections).strip() + "\n"


def multi_turn_phase_response_schema(phase: str) -> dict[str, Any]:
    normalized = str(phase or "").strip().lower()
    if normalized == "grounding":
        return {
            "name": "multi_turn_grounding_response",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "decision": {"type": "string", "enum": ["observe", "grounded"]},
                    "blocking_reasons": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "grounded_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "recovery_implications": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "observe_reason": {"type": "string"},
                    "observe_requests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "fact_type": {"type": "string", "minLength": 1},
                                "entity": {"type": "string", "minLength": 1},
                                "scope": {},
                                "reason": {"type": "string"},
                            },
                            "required": ["fact_type", "entity"],
                        },
                    },
                },
                "required": [
                    "thought",
                    "decision",
                    "blocking_reasons",
                    "grounded_facts",
                    "recovery_implications",
                ],
            },
        }
    if normalized == "outline":
        outline_state_schema = {
            "type": "object",
            "properties": {
                "resource_state": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "resource_location": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "held_part": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "part_state": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "part_location": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "part_holder_resource_jid": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
            },
            "additionalProperties": False,
        }
        return {
            "name": "multi_turn_outline_response",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "addressed_validation_findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string"},
                                "resource_jid": {"type": "string"},
                                "pose_source": {"type": "string"},
                                "failed_axes": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "failed_reason": {"type": "string"},
                                "claimed_task_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["task_id", "pose_source", "failed_axes"],
                        },
                    },
                    "outline_tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "outline_id": {"type": "string"},
                                "resource_jid": {"type": "string"},
                                "description": {"type": "string"},
                                "rationale": {"type": "string"},
                                "part_name": {"type": "string"},
                                "closes_condition_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "enables_task_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "action_target": {
                                    "type": "object",
                                    "properties": {
                                        "target_location": {"type": "string"},
                                        "source_location": {"type": "string"},
                                        "named_pose": {"type": "string"},
                                        "requirement_id": {"type": "string"},
                                    },
                                },
                                "expected_start_state": deepcopy(outline_state_schema),
                                "expected_end_state": deepcopy(outline_state_schema),
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "outline_id",
                                "resource_jid",
                                "description",
                                "rationale",
                                "expected_start_state",
                                "expected_end_state",
                                "depends_on",
                            ],
                        },
                    },
                },
                "required": ["thought", "addressed_validation_findings", "outline_tasks"],
            },
        }
    if normalized == "primitive_generation":
        return {
            "name": "multi_turn_primitive_generation_response",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": [
                            "need_outline_revision",
                            "draft_ready",
                        ],
                    },
                    "macro_tasks": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["thought", "decision", "macro_tasks"],
            },
        }
    if normalized == "finalize":
        return {
            "name": "multi_turn_finalize_response",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": [
                            "final_ready",
                            "need_outline_revision",
                            "need_primitive_revision",
                        ],
                    },
                    "final_proposal": {"type": "object"},
                },
                "required": ["thought", "decision", "final_proposal"],
            },
        }
    raise ValueError(f"unsupported multi-turn phase: {phase!r}")


__all__ = [
    "build_multi_turn_phase_prompt_input",
    "multi_turn_phase_response_schema",
    "render_multi_turn_phase_prompt",
]
