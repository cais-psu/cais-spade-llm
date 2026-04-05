"""Prompt builders for multi-turn v2 bridge — one-task-at-a-time outline."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


# ---------------------------------------------------------------------------
# Phase titles
# ---------------------------------------------------------------------------

_PHASE_TITLES: dict[str, str] = {
    "grounding": "Grounding (Observe)",
    "outline": "Outline (One Task At A Time)",
    "primitive_generation": "Primitive Generation",
    "finalize": "Finalize",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

_OUTLINE_TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "outline_id": {"type": "string"},
        "resource_jid": {"type": "string"},
        "description": {"type": "string"},
        "part_name": {"type": "string"},
        "expected_start_state": {"type": "object"},
        "expected_end_state": {"type": "object"},
    },
    "required": [
        "outline_id",
        "resource_jid",
        "description",
        "expected_start_state",
        "expected_end_state",
    ],
}

_OUTLINE_CANDIDATE_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "resource_jid": {"type": "string"},
        "action_type": {
            "type": "string",
            "enum": ["recover_resource", "acquire_part", "release_part"],
        },
        "part_name": {"type": "string"},
        "target_ref": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["resource_jid", "action_type"],
}


def _outline_incremental_response_schema() -> dict[str, Any]:
    return {
        "name": "multi_turn_v2_outline_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "next_task": deepcopy(_OUTLINE_TASK_SCHEMA),
                "lookahead_tasks": {
                    "type": "array",
                    "items": deepcopy(_OUTLINE_TASK_SCHEMA),
                },
            },
            "required": ["thought", "next_task"],
        },
    }


def _outline_candidates_response_schema() -> dict[str, Any]:
    return {
        "name": "multi_turn_v2_outline_candidates_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "candidate_tasks": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": deepcopy(_OUTLINE_CANDIDATE_ACTION_SCHEMA),
                },
            },
            "required": ["thought", "candidate_tasks"],
        },
    }


def _outline_single_pass_response_schema() -> dict[str, Any]:
    return {
        "name": "multi_turn_v2_outline_single_pass_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "outline_tasks": {
                    "type": "array",
                    "items": deepcopy(_OUTLINE_TASK_SCHEMA),
                },
            },
            "required": ["thought", "outline_tasks"],
        },
    }


def _grounding_response_schema() -> dict[str, Any]:
    return {
        "name": "multi_turn_v2_grounding_response",
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
                            "fact_type": {"type": "string"},
                            "entity": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["fact_type", "entity"],
                    },
                },
            },
            "required": ["thought", "decision"],
        },
    }


def _primitive_generation_response_schema() -> dict[str, Any]:
    return {
        "name": "multi_turn_v2_primitive_generation_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "primitives": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "decision": {"type": "string"},
            },
            "required": ["thought"],
        },
    }


def _finalize_response_schema() -> dict[str, Any]:
    return {
        "name": "multi_turn_v2_finalize_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "decision": {"type": "string"},
            },
            "required": ["thought"],
        },
    }


def multi_turn_v2_phase_response_schema(
    phase: str,
    *,
    outline_mode: str = "incremental",
) -> dict[str, Any]:
    """Return the JSON response schema for the given phase."""
    normalized = phase.strip().lower()
    if normalized == "outline":
        if outline_mode == "single_pass":
            return _outline_single_pass_response_schema()
        if outline_mode == "incremental_candidates_validated":
            return _outline_candidates_response_schema()
        return _outline_incremental_response_schema()
    if normalized == "grounding":
        return _grounding_response_schema()
    if normalized == "primitive_generation":
        return _primitive_generation_response_schema()
    if normalized == "finalize":
        return _finalize_response_schema()
    raise ValueError(f"Unknown phase: {phase!r}")


# ---------------------------------------------------------------------------
# Prompt input builder
# ---------------------------------------------------------------------------


def build_multi_turn_v2_phase_prompt_input(
    *,
    phase: str,
    llm_input: dict[str, Any],
    session_state: dict[str, Any],
    world_observation_surface: dict[str, Any] | None = None,
    recovery_gap_state: dict[str, Any] | None = None,
    pruned_actions: list[dict[str, Any]] | None = None,
    current_recovery_blockers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the structured prompt input payload for a given phase."""
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()
    return {
        "phase": phase,
        "llm_input": deepcopy(llm_input),
        "session_state": deepcopy(session_state),
        "world_observation_surface": deepcopy(world_observation_surface or {}),
        "recovery_gap_state": deepcopy(recovery_gap_state or {}),
        "pruned_actions": deepcopy(pruned_actions or []),
        "current_recovery_blockers": deepcopy(current_recovery_blockers or []),
    }


# ---------------------------------------------------------------------------
# Grounding prompt
# ---------------------------------------------------------------------------


def _compact_json(obj: Any) -> str:
    """JSON with 2-space indent, no markdown fences."""
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _compact_safety_rules(llm_input: dict[str, Any]) -> str:
    """Extract raw text from safety rules."""
    rules = llm_input.get("loaded_safety_rules") or []
    lines: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
        raw_text = str(rule.get("raw_text") or rule.get("summary") or "").strip()
        if rule_id and raw_text:
            lines.append(f"- {rule_id}: {raw_text}")
    return "\n".join(lines) if lines else "(none)"


def _compact_assembly_requirements(llm_input: dict[str, Any]) -> str:
    """Extract summary lines from assembly requirements."""
    reqs = llm_input.get("relevant_assembly_requirements") or []
    lines: list[str] = []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        req_id = str(req.get("requirement_id") or "").strip()
        summary = str(req.get("summary") or "").strip()
        status = str(req.get("status") or "").strip()
        if req_id and summary:
            lines.append(f"- {req_id} [{status}]: {summary}")
    return "\n".join(lines) if lines else "(none)"


def _compact_recovery_objectives(
    llm_input: dict[str, Any],
    *,
    projected_parts: list[dict[str, Any]],
) -> str:
    reqs = llm_input.get("relevant_assembly_requirements") or []
    parts_by_req: dict[str, dict[str, Any]] = {}
    for row in projected_parts:
        if not isinstance(row, dict):
            continue
        req_id = str(row.get("goal_requirement_id") or "").strip()
        if req_id and req_id not in parts_by_req:
            parts_by_req[req_id] = dict(row)

    lines: list[str] = []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        req_id = str(req.get("requirement_id") or "").strip()
        status = str(req.get("status") or "").strip()
        part_row = dict(parts_by_req.get(req_id) or {})
        part_name = str(part_row.get("part_name") or "").strip()
        goal_location = str(part_row.get("goal_location") or "").strip()
        if req_id and part_name and goal_location:
            lines.append(f"- {req_id} [{status}]: restore {part_name} to {goal_location}")
            continue
        summary = str(req.get("summary") or "").strip()
        if req_id and summary:
            lines.append(f"- {req_id} [{status}]: {summary}")
    return "\n".join(lines) if lines else "(none)"


def _outline_rejection_history(session_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return prior rejected outline attempts with their validation findings."""
    history: list[dict[str, Any]] = []
    for turn in (session_state.get("turns") or []):
        if not isinstance(turn, dict):
            continue
        if str(turn.get("phase") or "").strip().lower() != "outline":
            continue
        findings = [
            deepcopy(row) for row in (turn.get("validation_findings") or [])
            if isinstance(row, dict)
        ]
        if not findings:
            continue
        row: dict[str, Any] = {
            "turn_index": int(turn.get("turn_index") or 0),
            "decision": str(turn.get("decision") or "").strip() or None,
            "proposed_next_task": deepcopy(dict(turn.get("next_task") or {})),
            "validation_findings": findings,
        }
        lookahead_tasks = [
            deepcopy(item) for item in (turn.get("lookahead_tasks") or [])
            if isinstance(item, dict)
        ]
        if lookahead_tasks:
            row["lookahead_tasks"] = lookahead_tasks
        history.append(row)
    return history


def _outline_validation_summary(findings: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        task_id = str(finding.get("task_id") or "").strip()
        resource_jid = str(finding.get("resource_jid") or "").strip()
        part_name = str(finding.get("part_name") or "").strip()
        constraint_code = str(finding.get("constraint_code") or "").strip()
        reason = str(finding.get("reason") or "").strip()

        label_parts = [item for item in (task_id, resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "validation finding"
        if constraint_code and reason:
            lines.append(f"- {label} [{constraint_code}]: {reason}")
        elif reason:
            lines.append(f"- {label}: {reason}")
        elif constraint_code:
            lines.append(f"- {label} [{constraint_code}]")
    return "\n".join(lines) if lines else "(none)"


def _effective_task_part_holder(
    state: dict[str, Any],
    *,
    part_name: str,
    resource_jid: str,
) -> str:
    holder = str(
        state.get("part_holder_resource_jid")
        or state.get("current_holder_resource_jid")
        or ""
    ).strip()
    if holder:
        return holder
    held_part = str(state.get("held_part") or "").strip()
    if part_name and held_part == part_name:
        return resource_jid
    return ""


def _structured_task_action_summary(task: dict[str, Any]) -> str:
    action_type = str(task.get("action_type") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    target_ref = str(task.get("target_ref") or "").strip()
    if action_type == "recover_resource":
        if target_ref:
            return f"recover resource via {target_ref}"
        return "recover resource"
    if action_type == "acquire_part":
        if part_name:
            return f"acquire {part_name}"
        return "acquire part"
    if action_type == "release_part":
        if part_name and target_ref:
            return f"release {part_name} to {target_ref}"
        if part_name:
            return f"release {part_name}"
        return "release part"

    resource_jid = str(task.get("resource_jid") or "").strip()
    start_state = dict(task.get("expected_start_state") or {})
    end_state = dict(task.get("expected_end_state") or {})

    start_resource_state = str(start_state.get("resource_state") or "").strip()
    end_resource_state = str(end_state.get("resource_state") or "").strip()
    start_part_state = str(start_state.get("part_state") or "").strip()
    end_part_state = str(end_state.get("part_state") or "").strip()
    start_part_location = str(start_state.get("part_location") or "").strip()
    end_part_location = str(end_state.get("part_location") or "").strip()
    start_holder = _effective_task_part_holder(
        start_state,
        part_name=part_name,
        resource_jid=resource_jid,
    )
    end_holder = _effective_task_part_holder(
        end_state,
        part_name=part_name,
        resource_jid=resource_jid,
    )

    summary_parts: list[str] = []
    if part_name:
        if end_holder == resource_jid and start_holder != resource_jid:
            summary_parts.append(f"acquire {part_name}")
        elif start_holder == resource_jid and end_holder != resource_jid:
            if end_part_location:
                summary_parts.append(f"release {part_name} to {end_part_location}")
            else:
                summary_parts.append(f"release {part_name}")
        elif start_part_location != end_part_location and end_part_location:
            summary_parts.append(f"move {part_name} to {end_part_location}")
        elif start_part_state != end_part_state and end_part_state:
            summary_parts.append(
                f"{part_name} {start_part_state or 'state'} -> {end_part_state}"
            )

    if start_resource_state != end_resource_state and end_resource_state:
        summary_parts.append(
            f"{start_resource_state or 'resource'} -> {end_resource_state}"
        )

    if summary_parts:
        return "; ".join(summary_parts)
    return str(task.get("description") or "").strip() or "state transition"


def _outline_task_sequence_summary(tasks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        outline_id = str(task.get("outline_id") or "").strip()
        resource_jid = str(task.get("resource_jid") or "").strip()
        part_name = str(task.get("part_name") or "").strip()
        label_parts = [item for item in (outline_id, resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "accepted task"
        lines.append(f"- {label} -> {_structured_task_action_summary(task)}")
    return "\n".join(lines) if lines else "(none)"


def _outline_rejection_history_summary(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        turn_index = int(row.get("turn_index") or 0)
        task = dict(row.get("proposed_next_task") or {})
        outline_id = str(task.get("outline_id") or "").strip()
        resource_jid = str(task.get("resource_jid") or "").strip()
        part_name = str(task.get("part_name") or "").strip()
        description = str(task.get("description") or "").strip()
        finding_summary = _outline_validation_summary(
            [item for item in (row.get("validation_findings") or []) if isinstance(item, dict)]
        ).replace("\n- ", "; ")
        finding_summary = finding_summary[2:] if finding_summary.startswith("- ") else finding_summary

        label_parts = [item for item in (outline_id, resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "rejected task"
        sentence = f"- Turn {turn_index}: {label}"
        action_summary = _structured_task_action_summary(task)
        if action_summary:
            sentence += f" -> {action_summary}"
        if finding_summary and finding_summary != "(none)":
            sentence += f". Feedback: {finding_summary}"
        lines.append(sentence)
    return "\n".join(lines) if lines else "(none)"


def _candidate_rejection_feedback_summary(feedback_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in feedback_rows:
        if not isinstance(row, dict):
            continue
        candidate_index = int(row.get("candidate_index") or 0)
        task = dict(row.get("task") or {})
        outline_id = str(task.get("outline_id") or "").strip()
        resource_jid = str(task.get("resource_jid") or "").strip()
        part_name = str(task.get("part_name") or "").strip()
        description = str(task.get("description") or "").strip()
        finding_summary = _outline_validation_summary(
            [item for item in (row.get("validation_findings") or []) if isinstance(item, dict)]
        ).replace("\n- ", "; ")
        finding_summary = finding_summary[2:] if finding_summary.startswith("- ") else finding_summary
        label_parts = [item for item in (outline_id, resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "rejected candidate"
        sentence = f"- Candidate {candidate_index + 1}: {label}"
        action_summary = _structured_task_action_summary(task)
        if action_summary:
            sentence += f" -> {action_summary}"
        if finding_summary and finding_summary != "(none)":
            sentence += f". Feedback: {finding_summary}"
        lines.append(sentence)
    return "\n".join(lines) if lines else "(none)"


def _candidate_rejection_history(session_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all prior rejected candidate batches with validation feedback."""
    history: list[dict[str, Any]] = []
    for turn in (session_state.get("turns") or []):
        if not isinstance(turn, dict):
            continue
        if str(turn.get("phase") or "").strip().lower() != "outline":
            continue
        candidate_evaluations = [
            deepcopy(row)
            for row in (turn.get("candidate_evaluations") or [])
            if isinstance(row, dict) and not bool(row.get("valid"))
        ]
        if not candidate_evaluations:
            continue
        history.append({
            "turn_index": int(turn.get("turn_index") or 0),
            "candidate_evaluations": candidate_evaluations,
        })
    return history


def _candidate_rejection_history_summary(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        turn_index = int(row.get("turn_index") or 0)
        for evaluation in (row.get("candidate_evaluations") or []):
            if not isinstance(evaluation, dict):
                continue
            candidate_index = int(evaluation.get("candidate_index") or 0)
            task = dict(evaluation.get("task") or {})
            outline_id = str(task.get("outline_id") or "").strip()
            resource_jid = str(task.get("resource_jid") or "").strip()
            part_name = str(task.get("part_name") or "").strip()
            description = str(task.get("description") or "").strip()
            finding_summary = _outline_validation_summary(
                [item for item in (evaluation.get("validation_findings") or []) if isinstance(item, dict)]
            ).replace("\n- ", "; ")
            finding_summary = finding_summary[2:] if finding_summary.startswith("- ") else finding_summary
            label_parts = [item for item in (outline_id, resource_jid, part_name) if item]
            label = " / ".join(label_parts) if label_parts else "rejected candidate"
            sentence = f"- Turn {turn_index} Candidate {candidate_index + 1}: {label}"
            action_summary = _structured_task_action_summary(task)
            if action_summary:
                sentence += f" -> {action_summary}"
            if finding_summary and finding_summary != "(none)":
                sentence += f". Feedback: {finding_summary}"
            lines.append(sentence)
    return "\n".join(lines) if lines else "(none)"


def _current_recovery_blockers_summary(blockers: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for row in blockers:
        if not isinstance(row, dict):
            continue
        summary = str(row.get("summary") or "").strip()
        if not summary or summary in seen:
            continue
        seen.add(summary)
        lines.append(f"- {summary}")
    return "\n".join(lines) if lines else "(none)"


_PRUNED_ACTION_DURABLE_CONSTRAINT_CODES = {
    "workspace_unreachable",
    "holder_conflict",
    "required_part_not_held",
    "source_reference_unavailable",
    "part_relocation_without_carrier",
    "safety_rule_violation",
    "blocker_open",
}


def _finding_currently_applicable_for_pruned_actions(
    finding: dict[str, Any],
    *,
    task: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
) -> bool:
    constraint_code = str(finding.get("constraint_code") or "").strip().lower()
    if not constraint_code:
        return False
    if constraint_code not in _PRUNED_ACTION_DURABLE_CONSTRAINT_CODES:
        return False

    resource_jid = str(
        task.get("resource_jid")
        or finding.get("resource_jid")
        or ""
    ).strip()
    part_name = str(
        task.get("part_name")
        or finding.get("part_name")
        or ""
    ).strip()
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    part_row = dict(parts_by_name.get(part_name) or {})

    if constraint_code == "workspace_unreachable":
        current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        current_state = str(part_row.get("current_state") or "").strip().lower()
        current_location = str(part_row.get("current_location") or "").strip()
        if current_holder or current_state in {"held", "in_gripper", "assembled", "placed"}:
            return False
        if current_location:
            return False
        return True

    if constraint_code == "holder_conflict":
        actual_held = str(resource_row.get("held_part") or "").strip()
        current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        return bool(
            (part_name and actual_held and actual_held != part_name)
            or (resource_jid and current_holder and current_holder != resource_jid)
        )

    if constraint_code == "required_part_not_held":
        actual_held = str(resource_row.get("held_part") or "").strip()
        return bool(resource_jid and part_name and actual_held != part_name)

    if constraint_code == "part_relocation_without_carrier":
        actual_held = str(resource_row.get("held_part") or "").strip()
        current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        return bool(resource_jid and part_name and current_holder != resource_jid and actual_held != part_name)

    if constraint_code == "source_reference_unavailable":
        observed_pose = dict(part_row.get("observed_pose") or {})
        current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        current_location = str(part_row.get("current_location") or "").strip()
        return bool(part_name and not current_holder and not current_location and not observed_pose)

    return True


def _history_derived_pruned_actions_summary(
    *,
    pruned_actions: list[dict[str, Any]],
    rejection_history: list[dict[str, Any]],
    candidate_rejection_history: list[dict[str, Any]],
    candidate_rejection_feedback: list[dict[str, Any]],
    projected_resources: list[dict[str, Any]],
    projected_parts: list[dict[str, Any]],
) -> str:
    line_by_key: dict[tuple[Any, ...], str] = {}
    resources_by_jid = {
        str(row.get("resource_jid") or "").strip(): dict(row)
        for row in projected_resources
        if isinstance(row, dict) and str(row.get("resource_jid") or "").strip()
    }
    parts_by_name = {
        str(row.get("part_name") or "").strip(): dict(row)
        for row in projected_parts
        if isinstance(row, dict) and str(row.get("part_name") or "").strip()
    }

    def _add_row(
        *,
        task: dict[str, Any],
        findings: list[dict[str, Any]],
        explicit_summary: str = "",
        explicit_reason: str = "",
    ) -> None:
        applicable_findings = [
            dict(item)
            for item in findings
            if isinstance(item, dict)
            and _finding_currently_applicable_for_pruned_actions(
                dict(item),
                task=task,
                resources_by_jid=resources_by_jid,
                parts_by_name=parts_by_name,
            )
        ]
        if not applicable_findings:
            return

        summary = str(explicit_summary or "").strip()
        reason = str(explicit_reason or "").strip()
        if not summary:
            summary = _structured_task_action_summary(task)
        if not reason:
            codes = [
                str(item.get("constraint_code") or "").strip()
                for item in applicable_findings
                if isinstance(item, dict) and str(item.get("constraint_code") or "").strip()
            ]
            reason = "/".join(dict.fromkeys(codes))

        resource_jid = str(dict(task or {}).get("resource_jid") or "").strip()
        part_name = str(dict(task or {}).get("part_name") or "").strip()
        key = (resource_jid, part_name, summary, reason)

        label_parts = [item for item in (resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "rejected action"
        sentence = f"- {label}"
        if summary:
            sentence += f" -> {summary}"
        if reason:
            sentence += f" [{reason}]"
        if key in line_by_key:
            line_by_key.pop(key, None)
        line_by_key[key] = sentence

    for row in (pruned_actions or []):
        if not isinstance(row, dict):
            continue
        _add_row(
            task=dict(row.get("task") or row.get("action") or {}),
            findings=[dict(row)] if isinstance(row, dict) else [],
            explicit_summary=str(row.get("summary") or "").strip(),
            explicit_reason=str(row.get("reason") or "").strip(),
        )

    for row in (rejection_history or []):
        if not isinstance(row, dict):
            continue
        _add_row(
            task=dict(row.get("proposed_next_task") or {}),
            findings=[item for item in (row.get("validation_findings") or []) if isinstance(item, dict)],
        )

    for row in (candidate_rejection_history or []):
        if not isinstance(row, dict):
            continue
        for evaluation in (row.get("candidate_evaluations") or []):
            if not isinstance(evaluation, dict):
                continue
            _add_row(
                task=dict(evaluation.get("task") or {}),
                findings=[item for item in (evaluation.get("validation_findings") or []) if isinstance(item, dict)],
            )

    for row in (candidate_rejection_feedback or []):
        if not isinstance(row, dict):
            continue
        _add_row(
            task=dict(row.get("task") or {}),
            findings=[item for item in (row.get("validation_findings") or []) if isinstance(item, dict)],
        )

    lines = list(line_by_key.values())[-8:]
    return "\n".join(lines) if lines else "(none)"


def _projected_outline_resources(
    *,
    llm_input: dict[str, Any],
    session_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Render projected resource state from symbolic state when available."""
    symbolic_resources = dict(session_state.get("symbolic_resources") or {})
    if symbolic_resources:
        return [
            deepcopy(row) for row in symbolic_resources.values()
            if isinstance(row, dict)
        ]
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})
    return [
        deepcopy(row) for row in (observed_runtime_state.get("resources") or [])
        if isinstance(row, dict)
    ]


def _projected_outline_parts(
    *,
    llm_input: dict[str, Any],
    session_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Render projected part state from symbolic state when available."""
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})
    base_parts = [
        deepcopy(row) for row in (
            symbolic_parts.values()
            if symbolic_parts
            else (llm_input.get("part_facts") or [])
        )
        if isinstance(row, dict)
    ]

    parts_by_name: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in base_parts:
        part_name = str(row.get("part_name") or "").strip()
        if not part_name:
            continue
        parts_by_name[part_name] = deepcopy(row)
        order.append(part_name)

    for entry in dict(session_state.get("observation_store") or {}).values():
        if not isinstance(entry, dict):
            continue
        part_name = str(entry.get("part_name") or "").strip()
        if not part_name:
            continue
        if part_name not in parts_by_name:
            parts_by_name[part_name] = {"part_name": part_name}
            order.append(part_name)
        part_row = parts_by_name[part_name]
        pose = dict(entry.get("pose") or {})
        if not pose and entry.get("x") is not None:
            pose = {"x": entry.get("x"), "y": entry.get("y"), "z": entry.get("z")}
        if pose:
            part_row["observed_pose"] = deepcopy(pose)
        if part_row.get("current_location") in (None, "") and entry.get("current_location") not in (None, ""):
            part_row["current_location"] = deepcopy(entry.get("current_location"))
        holder = str(entry.get("current_holder_resource_jid") or "").strip()
        if not str(part_row.get("current_holder_resource_jid") or "").strip() and holder:
            part_row["current_holder_resource_jid"] = holder

    return [deepcopy(parts_by_name[name]) for name in order if name in parts_by_name]


def _clean_continuation_gap(llm_input: dict[str, Any]) -> dict[str, Any]:
    """Strip internal IDs and prefixes from modeled continuation gap."""
    raw_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    cleaned: dict[str, Any] = {}

    # Clean pending nominal tasks — remove blocked_by_condition_ids
    pending = []
    for task in (raw_gap.get("pending_nominal_tasks") or []):
        if not isinstance(task, dict):
            continue
        clean_task = {k: v for k, v in task.items() if k != "blocked_by_condition_ids"}
        pending.append(clean_task)
    if pending:
        cleaned["pending_nominal_tasks"] = pending

    # Clean unmet continuation conditions — remove internal fields, strip "focused_" prefix
    conditions = []
    _drop_fields = {"condition_id", "condition_family", "source_task_ids", "role"}
    for cond in (raw_gap.get("unmet_continuation_conditions") or []):
        if not isinstance(cond, dict):
            continue
        clean_cond = {k: v for k, v in cond.items() if k not in _drop_fields}
        kind = str(clean_cond.get("kind") or "").strip()
        if kind.startswith("focused_"):
            clean_cond["kind"] = kind[len("focused_"):]
        conditions.append(clean_cond)
    if conditions:
        cleaned["unmet_continuation_conditions"] = conditions

    resume_ready = raw_gap.get("resume_ready")
    if resume_ready is not None:
        cleaned["resume_ready"] = resume_ready

    return cleaned


def _slim_fault_event(fault_event: dict[str, Any]) -> dict[str, Any]:
    """Drop resource_state_after (redundant with resource facts)."""
    return {k: v for k, v in dict(fault_event or {}).items() if k != "resource_state_after"}


def _slim_resource_facts(resources: list[Any]) -> list[dict[str, Any]]:
    """Keep only fields needed for outline reasoning."""
    _keep = {
        "resource_jid",
        "current_state",
        "current_location",
        "held_part",
        "gripper_state",
    }
    return [
        {k: v for k, v in dict(row).items() if k in _keep}
        for row in resources
        if isinstance(row, dict)
    ]


def _slim_part_facts(parts: list[Any]) -> list[dict[str, Any]]:
    """Keep only fields relevant to grounding decisions."""
    _keep = {
        "part_name", "current_state", "current_location",
        "observed_pose", "current_holder_resource_jid",
        "origin_location", "goal_location", "goal_requirement_id",
    }
    return [
        {k: v for k, v in dict(row).items() if k in _keep}
        for row in parts
        if isinstance(row, dict)
    ]


def _render_grounding_prompt(payload: dict[str, Any]) -> str:
    llm_input = dict(payload.get("llm_input") or {})
    session_state = dict(payload.get("session_state") or {})
    world_observation_surface = dict(payload.get("world_observation_surface") or {})
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})
    observation_store = dict(session_state.get("observation_store") or {})

    sections: list[str] = [
        "Task and Role",
        (
            "You are the active replanner for a DES fallback recovery session.\n"
            "Current phase: Grounding (Observe).\n"
            "Assess the current world state and decide whether you have enough "
            "grounded information to plan a recovery, or whether you need additional "
            "observations first."
        ),
        "",
        "Fault Event",
        _compact_json(_slim_fault_event(llm_input.get("fault_event") or {})),
        "",
        "Current Resource Facts",
        _compact_json(_slim_resource_facts(observed_runtime_state.get("resources") or [])),
        "",
        "Current Part Facts",
        _compact_json(_slim_part_facts(llm_input.get("part_facts") or [])),
        "",
        "Safety Rules",
        _compact_safety_rules(llm_input),
        "",
        "Assembly Requirements",
        _compact_assembly_requirements(llm_input),
        "",
        "Modeled Continuation Gap",
        _compact_json(_clean_continuation_gap(llm_input)),
    ]

    if world_observation_surface:
        sections.extend([
            "",
            "World Observation Surface",
            _compact_json(world_observation_surface),
        ])

    if observation_store:
        sections.extend([
            "",
            "Session Observation Store",
            _compact_json(observation_store),
        ])

    phase_feedback = [
        deepcopy(row) for row in (session_state.get("phase_feedback") or [])
        if isinstance(row, dict) and str(row.get("phase") or "").strip() == "grounding"
    ]
    if phase_feedback:
        sections.extend([
            "",
            "Prior Grounding Feedback",
            _compact_json(phase_feedback),
        ])

    sections.extend([
        "",
        "Decision Rules",
        '- If the current resource facts and part facts provide enough grounded '
        'information to plan recovery, set decision to "grounded".',
        '- When decision is "grounded", leave observe_requests empty.',
        "- Do not request facts already present in the current session observations.",
        '- To request an observation, set decision to "observe" and include '
        "observe_requests with fact_type and entity.",
    ])

    return "\n".join(sections).strip() + "\n"


# ---------------------------------------------------------------------------
# Outline prompt
# ---------------------------------------------------------------------------


def _render_outline_prompt(payload: dict[str, Any]) -> str:
    llm_input = dict(payload.get("llm_input") or {})
    session_state = dict(payload.get("session_state") or {})
    current_recovery_blockers = [
        deepcopy(row)
        for row in (payload.get("current_recovery_blockers") or [])
        if isinstance(row, dict)
    ]
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()
    observation_store = dict(session_state.get("observation_store") or {})
    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    previous_lookahead = list(session_state.get("outline_lookahead") or [])
    pruned_actions = list(session_state.get("pruned_actions") or [])
    outline_validation_findings = list(
        session_state.get("outline_validation_findings") or []
    )
    rejection_history = _outline_rejection_history(session_state)
    projected_resources = _projected_outline_resources(
        llm_input=llm_input, session_state=session_state,
    )
    projected_parts = _projected_outline_parts(
        llm_input=llm_input, session_state=session_state,
    )
    candidate_rejection_feedback = [
        deepcopy(row)
        for row in (session_state.get("candidate_rejection_feedback") or [])
        if isinstance(row, dict)
    ]
    candidate_rejection_history = _candidate_rejection_history(session_state)
    history_pruned_actions = _history_derived_pruned_actions_summary(
        pruned_actions=pruned_actions,
        rejection_history=rejection_history,
        candidate_rejection_history=candidate_rejection_history,
        candidate_rejection_feedback=candidate_rejection_feedback,
        projected_resources=projected_resources,
        projected_parts=projected_parts,
    )

    is_single_pass = outline_mode == "single_pass"
    is_candidate_mode = outline_mode == "incremental_candidates_validated"

    if is_single_pass:
        role_text = (
            "You are the active replanner for a DES fallback recovery session.\n"
            "Current phase: Outline.\n"
            "Propose ALL recovery tasks as an ordered list in outline_tasks.\n"
            "Each task must be one concrete physical recovery action for one resource."
        )
    elif is_candidate_mode:
        role_text = (
            "You are the active replanner for a DES fallback recovery session.\n"
            "Current phase: Outline (Candidate Selection).\n"
            "Propose exactly 3 distinct candidate next tasks from the SAME current state.\n"
            "Each candidate must be a different recovery option, not a wording variant of the same action.\n"
            "A candidate does not need to complete recovery in one step. It may be an "
            "intermediate action that enables later recovery steps.\n"
            "The runtime will validate all candidates and commit at most one of them."
        )
    else:
        role_text = (
            "You are the active replanner for a DES fallback recovery session.\n"
            "Current phase: Outline (One Task At A Time).\n"
            "Propose exactly ONE next recovery task to append after the accepted prefix.\n"
            "You may include optional lookahead_tasks to show your intended remaining "
            "plan, but only next_task will be validated and accepted.\n"
            "When next_task is your LAST recovery task, leave lookahead_tasks empty "
            "to signal that the outline is complete."
        )

    sections: list[str] = [
        "Task and Role",
        role_text,
    ]

    if not is_single_pass and accepted_prefix:
        sections.extend([
            "",
            "Accepted Outline Prefix (keep exactly, do not modify)",
            _outline_task_sequence_summary(accepted_prefix),
        ])

    if is_candidate_mode:
        sections.extend([
            "",
            "Current Recovery Blockers",
            _current_recovery_blockers_summary(current_recovery_blockers),
        ])
    elif outline_validation_findings:
        sections.extend([
            "",
            "Active Validation Findings (still unresolved)",
            _outline_validation_summary(outline_validation_findings),
        ])

    if history_pruned_actions != "(none)":
        sections.extend([
            "",
            "Pruned Actions (derived from rejected history; do not re-propose these)",
            history_pruned_actions,
        ])

    if rejection_history and not is_candidate_mode:
        sections.extend([
            "",
            "Rejected Outline Attempts And Validation Feedback",
            _outline_rejection_history_summary(rejection_history),
        ])

    if is_candidate_mode and candidate_rejection_feedback:
        sections.extend([
            "",
            "Last Rejection Feedback",
            _candidate_rejection_feedback_summary(candidate_rejection_feedback),
        ])

    sections.extend([
        "",
        "Current Resource State",
        _compact_json(_slim_resource_facts(projected_resources)),
        "",
        "Current Part State",
        _compact_json(_slim_part_facts(projected_parts)),
        "Recovery Objectives",
        _compact_recovery_objectives(llm_input, projected_parts=projected_parts),
    ])

    if is_candidate_mode:
        sections.extend([
            "",
            "Action Shape Example (shape only; use actual grounded values from this prompt)",
            """```json
{
  "resource_jid": "RESOURCE_JID",
  "action_type": "recover_resource | acquire_part | release_part",
  "part_name": "PART_NAME or omit",
  "target_ref": "GROUNDED_DESTINATION_REF or omit",
  "description": "optional short text"
}
```""",
        ])
    else:
        sections.extend([
            "",
            "Safety Rules",
            _compact_safety_rules(llm_input),
        ])

    if not is_candidate_mode:
        sections.extend([
            "",
            "Modeled Continuation Gap",
            _compact_json(_clean_continuation_gap(llm_input)),
        ])

    if not is_single_pass and not is_candidate_mode and previous_lookahead:
        sections.extend([
            "",
            "Your Previous Lookahead (non-binding, for context)",
            _compact_json(previous_lookahead),
        ])

    constraints: list[str] = [
        "",
        "Hard Constraints",
        "- Use only the listed resources.",
        "- Do not assume a task is restricted to its nominal resource.",
        "- Each task must be one concrete physical recovery action for one resource.",
        "- Do not change a part or resource location by declaration alone. "
        "Any location change must result from a concrete physical action by the named resource.",
        "- Do not invent observations, locations, or states not grounded in the prompt.",
    ]
    if not is_candidate_mode:
        constraints.extend([
            "- expected_start_state and expected_end_state may use only: "
            "resource_state, held_part, part_state, part_location, part_holder_resource_jid.",
            "- Use flat scalar values in state objects. Do not nest by part name or resource jid.",
        ])

    if is_single_pass:
        constraints.append(
            "- Propose all recovery tasks in outline_tasks as an ordered sequence."
        )
    elif is_candidate_mode:
        constraints.extend([
            "- Propose exactly 3 tasks in candidate_tasks.",
            "- All candidate_tasks must start from the same current state shown in this prompt.",
            "- Make the 3 candidate_tasks meaningfully different recovery options.",
            "- Use only these action_type values: recover_resource, acquire_part, release_part.",
            "- recover_resource: requires resource_jid, must omit part_name, target_ref is optional and may be a grounded location or named pose.",
            "- acquire_part: requires resource_jid and part_name, must omit target_ref.",
            "- release_part: requires resource_jid, part_name, and a grounded destination target_ref.",
            "- Do not include lookahead_tasks in this mode.",
        ])
    else:
        constraints.append("- Propose exactly one task in next_task.")

    sections.extend(constraints)
    return "\n".join(sections).strip() + "\n"


# ---------------------------------------------------------------------------
# Prompt renderer
# ---------------------------------------------------------------------------


def render_multi_turn_v2_phase_prompt(prompt_input: dict[str, Any]) -> str:
    """Render a human-readable prompt string for the current phase."""
    payload = deepcopy(prompt_input or {})
    phase = str(payload.get("phase") or "").strip().lower()

    if phase == "grounding":
        return _render_grounding_prompt(payload)
    if phase == "outline":
        return _render_outline_prompt(payload)

    # Other phases: placeholder until implemented
    phase_title = _PHASE_TITLES.get(phase, phase)
    return f"Phase: {phase_title}\n\n[Prompt content for {phase} — not yet implemented]\n"


__all__ = [
    "build_multi_turn_v2_phase_prompt_input",
    "multi_turn_v2_phase_response_schema",
    "render_multi_turn_v2_phase_prompt",
]
