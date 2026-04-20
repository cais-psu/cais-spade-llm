"""Prompt builders for the active multi-turn bridge mode."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from cais_spade_llm.resources.resource_primitives import (
    filter_synthesis_primitive_catalog,
)
from cais_spade_llm.resources.resource_profile import get_resource_profile


_DEFAULT_CANDIDATE_BOUND = 5


# ---------------------------------------------------------------------------
# Phase titles
# ---------------------------------------------------------------------------

_PHASE_TITLES: dict[str, str] = {
    "grounding": "Observation / State Estimation",
    "outline": "Recovery Event Synthesis",
    "primitive_generation": "Event-to-Primitive Grounding",
    "finalize": "Executable Recovery Trace Finalization",
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
        "event_name": {"type": "string"},
        "part_name": {"type": "string"},
        "target_ref": {"type": "string"},
        "description": {"type": "string"},
        "expected_start_state": {"type": "object"},
        "expected_end_state": {"type": "object"},
    },
    "required": [
        "resource_jid",
        "event_name",
        "description",
        "expected_start_state",
        "expected_end_state",
    ],
}


def _outline_incremental_response_schema() -> dict[str, Any]:
    return {
        "name": "multi_turn_outline_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "next_transition": deepcopy(_OUTLINE_TASK_SCHEMA),
                "transition_suffix": {
                    "type": "array",
                    "items": deepcopy(_OUTLINE_TASK_SCHEMA),
                },
            },
            "required": ["thought", "next_transition"],
        },
    }


def _outline_candidates_response_schema(
    *,
    candidate_bound: int = _DEFAULT_CANDIDATE_BOUND,
) -> dict[str, Any]:
    normalized_bound = max(1, int(candidate_bound or 1))
    return {
        "name": "multi_turn_outline_candidates_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "candidate_events": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": normalized_bound,
                    "items": deepcopy(_OUTLINE_CANDIDATE_ACTION_SCHEMA),
                },
            },
            "required": ["thought", "candidate_events"],
        },
    }


def _outline_single_pass_response_schema() -> dict[str, Any]:
    return {
        "name": "multi_turn_outline_single_pass_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "transition_trace": {
                    "type": "array",
                    "items": deepcopy(_OUTLINE_TASK_SCHEMA),
                },
            },
            "required": ["thought", "transition_trace"],
        },
    }


def _grounding_response_schema() -> dict[str, Any]:
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
        "name": "multi_turn_primitive_generation_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "decision": {
                    "type": "string",
                    "enum": [
                        "primitive_steps_ready",
                        "need_context",
                        "need_primitive_revision",
                        "primitive_blocked",
                    ],
                },
                "outline_id": {"type": "string"},
                "resource_jid": {"type": "string"},
                "context_requests": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "rationale": {"type": "string"},
                "notes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "primitive_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "primitive": {"type": "string"},
                            "params": {"type": "object"},
                        },
                        "required": ["primitive", "params"],
                    },
                },
            },
            "required": [
                "thought",
                "decision",
                "outline_id",
                "resource_jid",
            ],
        },
    }


def _finalize_response_schema() -> dict[str, Any]:
    return {
        "name": "multi_turn_finalize_response",
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


def multi_turn_phase_response_schema(
    phase: str,
    *,
    outline_mode: str = "incremental",
    candidate_bound: int | None = None,
) -> dict[str, Any]:
    """Return the JSON response schema for the given phase."""
    normalized = phase.strip().lower()
    if normalized == "outline":
        if outline_mode == "single_pass":
            return _outline_single_pass_response_schema()
        if outline_mode == "incremental_candidates_validated":
            return _outline_candidates_response_schema(
                candidate_bound=max(1, int(candidate_bound or _DEFAULT_CANDIDATE_BOUND)),
            )
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


def build_multi_turn_phase_prompt_input(
    *,
    phase: str,
    llm_input: dict[str, Any],
    session_state: dict[str, Any],
    bridge_resources: dict[str, Any] | None = None,
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
        "bridge_resources": deepcopy(bridge_resources or {}),
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


_RESOURCE_ACTOR_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_-]*)\s+by\s+([A-Za-z][A-Za-z0-9_.@-]*)"
)


def _compact_safety_rules(
    llm_input: dict[str, Any],
    *,
    neutralize_resource_actors: bool = False,
) -> str:
    """Extract raw text from safety rules."""
    rules = llm_input.get("loaded_safety_rules") or []
    lines: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
        raw_text = str(rule.get("raw_text") or rule.get("summary") or "").strip()
        if neutralize_resource_actors:
            # Candidate mode should expose safety ordering facts, not nominal
            # resource assignments that can bias recovery away from feasible handoff.
            raw_text = _RESOURCE_ACTOR_PATTERN.sub(r"\1", raw_text)
            raw_text = " ".join(raw_text.split())
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
            "proposed_next_transition": deepcopy(dict(turn.get("next_transition") or {})),
            "validation_findings": findings,
        }
        transition_suffix = [
            deepcopy(item)
            for item in (turn.get("transition_suffix") or [])
            if isinstance(item, dict)
        ]
        if transition_suffix:
            row["transition_suffix"] = transition_suffix
        history.append(row)
    return history


_PROMPT_FEEDBACK_RENDER_STYLES = {"des_event_diagnostic", "raw_code"}

_PROMPT_PERSISTENT_FINDING_CODES = {
    "workspace_unreachable",
    "holder_conflict",
    "required_part_not_held",
    "part_relocation_without_carrier",
    "source_reference_unavailable",
}


def _feedback_render_style_token(session_state: dict[str, Any]) -> str:
    return "des_event_diagnostic"


def _finding_constraint_code_token(finding: dict[str, Any]) -> str:
    return str(finding.get("constraint_code") or "").strip().lower()


def _task_target_ref(task: dict[str, Any], finding: dict[str, Any] | None = None) -> str:
    direct_target = str(task.get("target_ref") or "").strip()
    if direct_target:
        return direct_target
    action_target = task.get("action_target")
    if isinstance(action_target, dict):
        target = str(
            action_target.get("target_ref")
            or action_target.get("target_location")
            or action_target.get("location")
            or ""
        ).strip()
        if target:
            return target
    if isinstance(finding, dict):
        evidence = dict(finding.get("evidence") or {})
        return str(
            evidence.get("target_ref")
            or evidence.get("target_location")
            or ""
        ).strip()
    return ""


def _candidate_event_label(
    *,
    task: dict[str, Any],
    finding: dict[str, Any] | None = None,
) -> str:
    outline_id = str(
        task.get("outline_id")
        or task.get("task_id")
        or (finding or {}).get("task_id")
        or ""
    ).strip()
    resource_jid = str(
        task.get("resource_jid")
        or (finding or {}).get("resource_jid")
        or ""
    ).strip()
    part_name = str(
        task.get("part_name")
        or (finding or {}).get("part_name")
        or ""
    ).strip()
    label_parts = [item for item in (outline_id, resource_jid, part_name) if item]
    label = "/".join(label_parts) if label_parts else "anonymous_event"
    target_ref = _task_target_ref(task, finding=finding)
    if target_ref:
        label += f"->{target_ref}"
    return label


def _finding_resources_by_jid(
    projected_resources: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("resource_jid") or "").strip(): dict(row)
        for row in projected_resources
        if isinstance(row, dict) and str(row.get("resource_jid") or "").strip()
    }


def _finding_parts_by_name(
    projected_parts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("part_name") or "").strip(): dict(row)
        for row in projected_parts
        if isinstance(row, dict) and str(row.get("part_name") or "").strip()
    }


def _finding_state_evidence_text(
    *,
    task: dict[str, Any],
    finding: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    evidence = dict(finding.get("evidence") or {})
    resource_jid = str(
        task.get("resource_jid") or finding.get("resource_jid") or ""
    ).strip()
    part_name = str(task.get("part_name") or finding.get("part_name") or "").strip()
    target_ref = _task_target_ref(task, finding=finding)
    resource_row = dict((resources_by_jid or {}).get(resource_jid) or {})
    part_row = dict((parts_by_name or {}).get(part_name) or {})

    facts: list[str] = []
    if resource_jid:
        facts.append(f"resource_jid={resource_jid}")
    if part_name:
        facts.append(f"part_name={part_name}")
    if target_ref:
        facts.append(f"target_ref={target_ref}")

    checked_pose = dict(evidence.get("checked_pose") or finding.get("pose") or {})
    if checked_pose:
        facts.append(f"checked_pose=({_pose_xyz_text(checked_pose)})")

    workspace_bounds = dict(
        evidence.get("workspace_bounds") or finding.get("workspace_bounds") or {}
    )
    if workspace_bounds:
        facts.append(_workspace_capability_hint(workspace_bounds))

    changed_fields = [
        str(field).strip()
        for field in (evidence.get("changed_fields") or [])
        if str(field).strip()
    ]
    if changed_fields:
        facts.append("changed_fields=" + ",".join(changed_fields))

    condition_ids = [
        str(item).strip()
        for item in (
            evidence.get("condition_ids")
            or finding.get("condition_ids")
            or []
        )
        if str(item).strip()
    ]
    if condition_ids:
        facts.append("condition_ids=" + ",".join(condition_ids))

    rule_id = str(finding.get("rule_id") or evidence.get("rule_id") or "").strip()
    if rule_id:
        facts.append(f"rule_id={rule_id}")

    held_part = str(resource_row.get("held_part") or "").strip()
    if held_part:
        facts.append(f"held_part={held_part}")

    current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
    if current_holder:
        facts.append(f"current_holder_resource_jid={current_holder}")

    current_location = str(part_row.get("current_location") or "").strip()
    if current_location:
        facts.append(f"current_location={current_location}")

    return "; ".join(facts) if facts else "projected_state_facts=not_available"


def _des_diagnostic_fields(finding: dict[str, Any]) -> dict[str, str]:
    constraint_code = _finding_constraint_code_token(finding)
    constraint_family = str(finding.get("constraint_family") or "").strip().lower()

    if constraint_code in {
        "unknown_location_token",
        "unknown_named_pose",
        "unknown_state_token",
        "source_reference_unavailable",
        "part_ambiguous",
    }:
        return {
            "event_status": "disabled",
            "diagnosis": "state_estimate_not_grounded",
            "guard_or_condition": (
                "candidate event parameters are not grounded in the current state estimate"
            ),
            "re_enablement": (
                "ground the referenced entity, pose, or location in the current state estimate"
            ),
        }

    if constraint_code in {
        "resource_unbound",
        "part_unbound",
        "part_relocation_without_carrier",
        "required_part_not_held",
        "holder_conflict",
        "missing_release_destination",
        "expected_start_state_mismatch",
    }:
        return {
            "event_status": "disabled",
            "diagnosis": "event_not_enabled",
            "guard_or_condition": (
                "event preconditions are not enabled in the current projected state"
            ),
            "re_enablement": (
                "first satisfy the missing precondition or establish the required "
                "carrier/control relation"
            ),
        }

    if constraint_code == "workspace_unreachable" or constraint_family == "resource_feasibility":
        return {
            "event_status": "disabled",
            "diagnosis": "feasibility_guard_disabled",
            "guard_or_condition": (
                "resource feasibility guard is false for the referenced source/target"
            ),
            "re_enablement": (
                "choose an event whose source/target is feasible for the acting resource"
            ),
        }

    if constraint_code in {
        "blocker_open",
        "dependency_unsatisfied",
        "order_violation",
        "invalid_dependency_reference",
        "claimed_condition_not_currently_unmet",
    } or constraint_family == "sequence":
        return {
            "event_status": "blocked_by_supervisor",
            "diagnosis": "supervisor_ordering_block",
            "guard_or_condition": (
                "projected successor would advance before open continuation conditions are cleared"
            ),
            "re_enablement": (
                "first execute an event that clears the open prerequisite or "
                "continuation condition"
            ),
        }

    if constraint_code == "safety_rule_violation" or constraint_family == "safety":
        return {
            "event_status": "blocked_by_supervisor",
            "diagnosis": "supervisor_safety_block",
            "guard_or_condition": (
                "projected successor is outside the safety-admissible region of the "
                "product automaton"
            ),
            "re_enablement": (
                "choose an event whose projected APs remain supervisor-admissible"
            ),
        }

    if constraint_code in {"no_state_change", "no_blocker_reduction"} or constraint_family == "candidate_selection":
        return {
            "event_status": "nonprogressing",
            "diagnosis": "nonblocking_progress_failure",
            "guard_or_condition": (
                "candidate event does not move the recovery plant toward a marked "
                "continuation-ready state"
            ),
            "re_enablement": (
                "choose an event that changes projected state and closes an open "
                "nonblocking condition"
            ),
        }

    return {
        "event_status": "disabled",
        "diagnosis": "runtime_validation_block",
        "guard_or_condition": (
            "candidate event is blocked by a runtime validation constraint"
        ),
        "re_enablement": (
            "revise the event to satisfy the reported validation condition"
        ),
    }


def _render_des_event_diagnostic(
    *,
    task: dict[str, Any],
    finding: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    diagnostic = _des_diagnostic_fields(finding)
    line = (
        f"- candidate_event={_candidate_event_label(task=task, finding=finding)}"
        f" | event_status={diagnostic['event_status']}"
        f" | diagnosis={diagnostic['diagnosis']}"
        f" | guard_or_condition={diagnostic['guard_or_condition']}"
        f" | state_evidence={_finding_state_evidence_text(task=task, finding=finding, resources_by_jid=resources_by_jid, parts_by_name=parts_by_name)}"
        f" | re_enablement={diagnostic['re_enablement']}"
    )
    if _finding_constraint_code_token(finding) in _PROMPT_PERSISTENT_FINDING_CODES:
        line += (
            " | persistence=diagnosis persists until the relevant projected state facts change"
        )
    return line


def _candidate_diagnostic_signature(
    *,
    task: dict[str, Any],
    finding: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, str, str, str, str, str]:
    """Stable semantic key for feedback de-duplication.

    Candidate ids are deliberately excluded so repeated RECOVERY_SEQ*_N rows
    collapse when they describe the same event/failure under the same state.
    """
    diagnostic = _des_diagnostic_fields(finding)
    resource_jid = str(
        task.get("resource_jid")
        or finding.get("resource_jid")
        or ""
    ).strip()
    part_name = str(task.get("part_name") or finding.get("part_name") or "").strip()
    target_ref = _task_target_ref(task, finding=finding)
    evidence_text = _finding_state_evidence_text(
        task=task,
        finding=finding,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    return (
        resource_jid,
        part_name,
        target_ref,
        diagnostic["event_status"],
        diagnostic["diagnosis"],
        evidence_text,
    )


def _outline_validation_summary(
    findings: list[dict[str, Any]],
    *,
    feedback_render_style: str = "raw_code",
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    if feedback_render_style == "des_event_diagnostic":
        lines: list[str] = []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            task = {
                "outline_id": str(finding.get("task_id") or "").strip(),
                "resource_jid": str(finding.get("resource_jid") or "").strip(),
                "part_name": str(finding.get("part_name") or "").strip(),
            }
            lines.append(
                _render_des_event_diagnostic(
                    task=task,
                    finding=finding,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )
            )
        return "\n".join(lines) if lines else "(none)"

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
    event_name = str(task.get("event_name") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    target_ref = str(task.get("target_ref") or "").strip()
    if event_name:
        summary = event_name
        lower_summary = event_name.lower()
        if part_name and part_name.lower() not in lower_summary:
            summary = f"{summary} {part_name}"
            lower_summary = summary.lower()
        if target_ref and target_ref.lower() not in lower_summary:
            summary = f"{summary} to {target_ref}"
        return summary

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
                summary_parts.append(f"place {part_name} at {end_part_location}")
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


def _outline_task_sequence_summary(
    tasks: list[dict[str, Any]],
    *,
    include_descriptions: bool = True,
) -> str:
    lines: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        outline_id = str(task.get("outline_id") or "").strip()
        resource_jid = str(task.get("resource_jid") or "").strip()
        part_name = str(task.get("part_name") or "").strip()
        label_parts = [item for item in (outline_id, resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "accepted event"
        lines.append(f"- {label} -> {_structured_task_action_summary(task)}")
        description = str(task.get("description") or "").strip()
        if include_descriptions and description:
            lines.append(f"  ({description})")
    return "\n".join(lines) if lines else "(none)"


def _outline_rejection_history_summary(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        turn_index = int(row.get("turn_index") or 0)
        task = dict(row.get("proposed_next_transition") or {})
        outline_id = str(task.get("outline_id") or "").strip()
        resource_jid = str(task.get("resource_jid") or "").strip()
        part_name = str(task.get("part_name") or "").strip()
        description = str(task.get("description") or "").strip()
        finding_summary = _outline_validation_summary(
            [item for item in (row.get("validation_findings") or []) if isinstance(item, dict)]
        ).replace("\n- ", "; ")
        finding_summary = finding_summary[2:] if finding_summary.startswith("- ") else finding_summary

        label_parts = [item for item in (outline_id, resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "rejected event"
        sentence = f"- Turn {turn_index}: {label}"
        action_summary = _structured_task_action_summary(task)
        if action_summary:
            sentence += f" -> {action_summary}"
        if finding_summary and finding_summary != "(none)":
            sentence += f". Feedback: {finding_summary}"
        lines.append(sentence)
    return "\n".join(lines) if lines else "(none)"


def _candidate_rejection_history(session_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rejected candidate batches since the last accepted recovery event."""
    history: list[dict[str, Any]] = []
    for turn in (session_state.get("turns") or []):
        if not isinstance(turn, dict):
            continue
        if str(turn.get("phase") or "").strip().lower() != "outline":
            continue
        if (
            isinstance(turn.get("selected_transition"), dict)
            or isinstance(turn.get("next_transition"), dict)
        ):
            history = []
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


def _candidate_rejection_learning_summary(
    *,
    history: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]],
    feedback_render_style: str = "raw_code",
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Summarize rejected candidate patterns without repeating action labels."""
    if feedback_render_style == "des_event_diagnostic":
        line_by_key: dict[tuple[str, str, str, str, str, str], str] = {}

        def _add_des_rows(*, task: dict[str, Any], findings: list[dict[str, Any]]) -> None:
            for finding in findings:
                if not isinstance(finding, dict):
                    continue
                key = _candidate_diagnostic_signature(
                    task=task,
                    finding=finding,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )
                if key in line_by_key:
                    line_by_key.pop(key, None)
                line_by_key[key] = _render_des_event_diagnostic(
                    task=task,
                    finding=finding,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )

        for row in history:
            if not isinstance(row, dict):
                continue
            for evaluation in (row.get("candidate_evaluations") or []):
                if not isinstance(evaluation, dict):
                    continue
                _add_des_rows(
                    task=dict(evaluation.get("task") or {}),
                    findings=[
                        item
                        for item in (evaluation.get("validation_findings") or [])
                        if isinstance(item, dict)
                    ],
                )

        for row in feedback_rows:
            if not isinstance(row, dict):
                continue
            _add_des_rows(
                task=dict(row.get("task") or {}),
                findings=[
                    item
                    for item in (row.get("validation_findings") or [])
                    if isinstance(item, dict)
                ],
            )

        lines = list(line_by_key.values())[-8:]
        return "\n".join(lines) if lines else "(none)"

    line_by_key: dict[tuple[str, str, str, str, str], str] = {}

    def _target_ref(task: dict[str, Any]) -> str:
        direct_target = str(task.get("target_ref") or "").strip()
        if direct_target:
            return direct_target
        action_target = task.get("action_target")
        if isinstance(action_target, dict):
            return str(
                action_target.get("target_ref")
                or action_target.get("location")
                or ""
            ).strip()
        return ""

    def _add_evaluation(
        *,
        task: dict[str, Any],
        findings: list[dict[str, Any]],
        turn_index: int | None = None,
    ) -> None:
        resource_jid = str(task.get("resource_jid") or "").strip()
        part_name = str(task.get("part_name") or "").strip()
        target_ref = _target_ref(task)
        label_parts = [item for item in (resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "candidate"
        if target_ref:
            label += f" to {target_ref}"
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            constraint_code = str(finding.get("constraint_code") or "").strip()
            reason = str(finding.get("reason") or "").strip()
            if not constraint_code and not reason:
                continue
            key = (resource_jid, part_name, target_ref, constraint_code, reason)
            sentence = f"- {label}"
            if turn_index is not None:
                sentence += f" (turn {turn_index})"
            if constraint_code and reason:
                sentence += f" [{constraint_code}]: {reason}"
            elif constraint_code:
                sentence += f" [{constraint_code}]"
            else:
                sentence += f": {reason}"
            if constraint_code == "workspace_unreachable":
                sentence += " (persists while observed_pose and workspace_bounds are unchanged)"
            if key in line_by_key:
                line_by_key.pop(key, None)
            line_by_key[key] = sentence

    for row in history:
        if not isinstance(row, dict):
            continue
        turn_index = int(row.get("turn_index") or 0)
        for evaluation in (row.get("candidate_evaluations") or []):
            if not isinstance(evaluation, dict):
                continue
            _add_evaluation(
                task=dict(evaluation.get("task") or {}),
                findings=[
                    item
                    for item in (evaluation.get("validation_findings") or [])
                    if isinstance(item, dict)
                ],
                turn_index=turn_index or None,
            )

    for row in feedback_rows:
        if not isinstance(row, dict):
            continue
        _add_evaluation(
            task=dict(row.get("task") or {}),
            findings=[
                item
                for item in (row.get("validation_findings") or [])
                if isinstance(item, dict)
            ],
        )

    lines = list(line_by_key.values())[-8:]
    return "\n".join(lines) if lines else "(none)"


def _candidate_rejection_diagnostic_signatures(
    *,
    history: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]],
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> set[tuple[str, str, str, str, str, str]]:
    signatures: set[tuple[str, str, str, str, str, str]] = set()

    def _add_rows(*, task: dict[str, Any], findings: list[dict[str, Any]]) -> None:
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            signatures.add(
                _candidate_diagnostic_signature(
                    task=task,
                    finding=finding,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )
            )

    for row in history:
        if not isinstance(row, dict):
            continue
        for evaluation in (row.get("candidate_evaluations") or []):
            if not isinstance(evaluation, dict):
                continue
            _add_rows(
                task=dict(evaluation.get("task") or {}),
                findings=[
                    item
                    for item in (evaluation.get("validation_findings") or [])
                    if isinstance(item, dict)
                ],
            )

    for row in feedback_rows:
        if not isinstance(row, dict):
            continue
        _add_rows(
            task=dict(row.get("task") or {}),
            findings=[
                item
                for item in (row.get("validation_findings") or [])
                if isinstance(item, dict)
            ],
        )

    return signatures


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


def _named_pose_tokens(value: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(value, dict):
        tokens.extend(
            str(token).strip()
            for token in value.keys()
            if str(token).strip()
        )
    else:
        tokens.extend(
            str(token).strip()
            for token in (value or [])
            if str(token).strip()
        )
    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped


def _workspace_capability_hint(bounds: dict[str, Any]) -> str:
    if not isinstance(bounds, dict) or not bounds:
        return "workspace not advertised"
    axis_parts: list[str] = []
    for axis in ("x", "y", "z"):
        lo = bounds.get(f"{axis}_min_m")
        hi = bounds.get(f"{axis}_max_m")
        if lo is None and hi is None:
            continue
        lo_text = "?" if lo is None else f"{float(lo):.2f}"
        hi_text = "?" if hi is None else f"{float(hi):.2f}"
        axis_parts.append(f"{axis}[{lo_text},{hi_text}]")
    if not axis_parts:
        return "workspace not advertised"
    return "workspace " + ", ".join(axis_parts)


def _pose_xyz_text(pose: dict[str, Any]) -> str:
    if not isinstance(pose, dict) or not pose:
        return "unknown"
    parts: list[str] = []
    for axis in ("x", "y", "z"):
        coord = _float_or_none(pose.get(axis))
        parts.append(f"{axis}=?" if coord is None else f"{axis}={coord:.2f}")
    return ",".join(parts)


def _resource_current_pose_text(
    *,
    entry: dict[str, Any],
    bridge_snapshot: dict[str, Any],
) -> str:
    bridge_facets = dict(bridge_snapshot.get("resource_facets") or {})
    entry_facets = dict(entry.get("resource_facets") or {})
    bridge_manipulator = dict(bridge_facets.get("manipulator") or {})
    entry_manipulator = dict(entry_facets.get("manipulator") or {})
    for raw_pose in (
        bridge_snapshot.get("current_pose"),
        bridge_manipulator.get("current_pose"),
        entry.get("current_pose"),
        entry_manipulator.get("current_pose"),
    ):
        if isinstance(raw_pose, dict) and raw_pose:
            return _pose_xyz_text(dict(raw_pose))
    return "unknown"


def _resource_capabilities_summary(bridge_resources: dict[str, Any]) -> str:
    if not isinstance(bridge_resources, dict) or not bridge_resources:
        return "(none advertised)"

    lines: list[str] = []
    for resource_jid in sorted(bridge_resources):
        entry = dict(bridge_resources.get(resource_jid) or {})
        bridge_snapshot = dict(entry.get("bridge_snapshot") or {})
        static_capabilities = dict(entry.get("static_capabilities") or {})
        bridge_adapter = dict(
            entry.get("bridge_adapter")
            or bridge_snapshot.get("bridge_adapter")
            or {}
        )

        manipulation = (
            "manipulate parts"
            if bool(bridge_adapter.get("supports_manipulator_pick_place"))
            else (
                "execute bridge actions"
                if bool(bridge_adapter.get("supports_executable_bridge"))
                else "manipulation not advertised"
            )
        )

        named_poses = _named_pose_tokens(
            bridge_snapshot.get("named_poses")
            or static_capabilities.get("named_poses")
            or bridge_snapshot.get("available_named_poses")
            or static_capabilities.get("available_named_poses")
            or []
        )
        named_pose_text = ", ".join(named_poses) if named_poses else "none advertised"
        reachable_locations = [
            str(token).strip()
            for token in (
                static_capabilities.get("reachability")
                or static_capabilities.get("reachable_locations")
                or bridge_snapshot.get("reachability")
                or bridge_snapshot.get("reachable_locations")
                or []
            )
            if str(token).strip()
        ]
        reachable_text = (
            ", ".join(dict.fromkeys(reachable_locations))
            if reachable_locations
            else "none advertised"
        )

        workspace_hint = _workspace_capability_hint(
            dict(
                bridge_snapshot.get("workspace_bounds")
                or static_capabilities.get("workspace_bounds")
                or {}
            )
        )
        current_pose_text = _resource_current_pose_text(
            entry=entry,
            bridge_snapshot=bridge_snapshot,
        )
        lines.append(
            f"- {resource_jid}: {manipulation}; named poses {named_pose_text}; "
            f"reachable locations {reachable_text}; current_pose({current_pose_text}); "
            f"{workspace_hint}"
        )
    return "\n".join(lines) if lines else "(none advertised)"


def _active_primitive_outline_event(
    session_state: dict[str, Any],
) -> tuple[int, dict[str, Any] | None]:
    accepted_prefix = [
        dict(row)
        for row in (session_state.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    cursor = int(session_state.get("primitive_generation_cursor") or 0)
    if cursor < 0:
        cursor = 0
    if cursor >= len(accepted_prefix):
        return cursor, None
    return cursor, deepcopy(accepted_prefix[cursor])


def _primitive_catalog_description_for_prompt(raw_entry: dict[str, Any]) -> str:
    primitive_kind = str(raw_entry.get("primitive_kind") or "").strip()
    preconditions = dict(raw_entry.get("preconditions") or {})
    effects = dict(raw_entry.get("effects") or {})
    summary_parts: list[str] = []
    if primitive_kind:
        summary_parts.append(f"{primitive_kind} primitive")
    if preconditions:
        summary_parts.append(
            "preconditions: " + ", ".join(sorted(str(key) for key in preconditions.keys()))
        )
    if effects:
        summary_parts.append(
            "effects: " + ", ".join(sorted(str(key) for key in effects.keys()))
        )
    if summary_parts:
        return "; ".join(summary_parts)
    return str(
        raw_entry.get("semantic_summary")
        or raw_entry.get("description")
        or ""
    ).strip()


def _slim_primitive_catalog_for_prompt(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slim: list[dict[str, Any]] = []
    for raw_entry in filter_synthesis_primitive_catalog(catalog or []):
        if not isinstance(raw_entry, dict):
            continue
        name = str(raw_entry.get("name") or "").strip()
        if not name:
            continue
        description = _primitive_catalog_description_for_prompt(raw_entry)
        entry: dict[str, Any] = {
            "name": name,
            "primitive_kind": str(raw_entry.get("primitive_kind") or "").strip(),
            "description": description,
            "params": deepcopy(raw_entry.get("params") or {}),
            "required_params": deepcopy(raw_entry.get("required_params") or []),
        }
        preconditions = deepcopy(raw_entry.get("preconditions") or {})
        effects = deepcopy(raw_entry.get("effects") or {})
        if preconditions:
            entry["preconditions"] = preconditions
        if effects:
            entry["effects"] = effects
        resource_type = str(raw_entry.get("resource_type") or "").strip()
        trace_fact_contract = deepcopy(
            raw_entry.get("trace_fact_contract")
            or raw_entry.get("trace_facts")
            or {}
        )
        if not trace_fact_contract and resource_type:
            profile = get_resource_profile(resource_type)
            trace_fact_contract = deepcopy(
                dict(profile.primitive_trace_fact_map or {}).get(name) or {}
            )
        if trace_fact_contract:
            entry["trace_fact_contract"] = trace_fact_contract
        slim.append(entry)
    return slim


def _primitive_signature_card(
    primitive_catalogs_by_resource: dict[str, list[dict[str, Any]]],
) -> str:
    lines: list[str] = []
    for resource_jid in sorted(primitive_catalogs_by_resource.keys()):
        catalog = primitive_catalogs_by_resource.get(resource_jid) or []
        signatures: list[str] = []
        for entry in catalog:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            required = [
                str(item).strip()
                for item in (entry.get("required_params") or [])
                if str(item).strip()
            ]
            if not required:
                params = dict(entry.get("params") or {})
                required = [str(key).strip() for key in params.keys() if str(key).strip()]
            signatures.append(f"{name}({','.join(required)})")
        if signatures:
            lines.append(f"- {resource_jid}: " + "; ".join(signatures))
    return "\n".join(lines).strip() or "(none)"


def _trace_fact_spec_text(spec: Any) -> str:
    if isinstance(spec, dict):
        fact_name = str(spec.get("fact") or "").strip()
        when = str(spec.get("when") or "").strip()
    else:
        fact_name = str(spec or "").strip()
        when = ""
    if not fact_name:
        return ""
    text = f"{fact_name}(part)"
    if when:
        text += f" when {when}"
    return text


def _primitive_trace_fact_contract_card(
    primitive_catalogs_by_resource: dict[str, list[dict[str, Any]]],
) -> str:
    lines: list[str] = []
    for resource_jid in sorted(primitive_catalogs_by_resource.keys()):
        fragments: list[str] = []
        for entry in primitive_catalogs_by_resource.get(resource_jid) or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            contract = dict(entry.get("trace_fact_contract") or {})
            if not name or not contract:
                continue
            establishes = [
                _trace_fact_spec_text(spec)
                for spec in (contract.get("establishes") or [])
            ]
            requires = [
                _trace_fact_spec_text(spec)
                for spec in (contract.get("requires") or [])
            ]
            details: list[str] = []
            if any(establishes):
                details.append(
                    "establishes " + ", ".join(item for item in establishes if item)
                )
            if any(requires):
                details.append(
                    "requires " + ", ".join(item for item in requires if item)
                )
            if details:
                fragments.append(f"{name}: " + "; ".join(details))
        if fragments:
            lines.append(f"- {resource_jid}: " + " | ".join(fragments))
    return "\n".join(lines).strip() or "(none)"


def _active_event_start_facts_for_prompt(
    *,
    resources: list[dict[str, Any]],
    parts: list[dict[str, Any]],
    active_event: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(active_event, dict) or not active_event:
        return deepcopy(resources), deepcopy(parts)

    resource_jid = str(active_event.get("resource_jid") or "").strip()
    part_name = str(active_event.get("part_name") or "").strip()
    expected_start = dict(active_event.get("expected_start_state") or {})
    projected_resources = [deepcopy(row) for row in resources if isinstance(row, dict)]
    projected_parts = [deepcopy(row) for row in parts if isinstance(row, dict)]

    for resource in projected_resources:
        if str(resource.get("resource_jid") or "").strip() != resource_jid:
            continue
        if "resource_state" in expected_start:
            resource["current_state"] = deepcopy(expected_start.get("resource_state"))
        if "resource_location" in expected_start:
            resource["current_location"] = deepcopy(expected_start.get("resource_location"))
        if "held_part" in expected_start:
            resource["held_part"] = deepcopy(expected_start.get("held_part"))
            resource["gripper_state"] = (
                "closed" if expected_start.get("held_part") not in (None, "") else "open"
            )

    for part in projected_parts:
        if str(part.get("part_name") or "").strip() != part_name:
            continue
        if "part_state" in expected_start:
            part["current_state"] = deepcopy(expected_start.get("part_state"))
        if "part_location" in expected_start:
            part["current_location"] = deepcopy(expected_start.get("part_location"))
        if "part_holder_resource_jid" in expected_start:
            part["current_holder_resource_jid"] = deepcopy(
                expected_start.get("part_holder_resource_jid")
            )
    return projected_resources, projected_parts


def _primitive_rejection_feedback_summary(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        outline_id = str(row.get("outline_id") or "").strip()
        resource_jid = str(row.get("resource_jid") or "").strip()
        reason = str(row.get("reason") or row.get("error") or "").strip()
        constraint_code = str(row.get("constraint_code") or "").strip()
        label_parts = [item for item in (outline_id, resource_jid) if item]
        label = " / ".join(label_parts) if label_parts else "primitive candidate"
        if constraint_code and reason:
            lines.append(f"- {label} [{constraint_code}]: {reason}")
        elif constraint_code:
            lines.append(f"- {label} [{constraint_code}]")
        elif reason:
            lines.append(f"- {label}: {reason}")
    return "\n".join(lines) if lines else "(none)"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pose_workspace_relation_text(
    pose: dict[str, Any],
    bounds: dict[str, Any],
) -> str:
    if not isinstance(pose, dict) or not pose:
        return "pose_unknown"
    if not isinstance(bounds, dict) or not bounds:
        return "workspace_bounds_unknown"

    failures: list[str] = []
    checked_axes = 0
    for axis in ("x", "y", "z"):
        coord = _float_or_none(pose.get(axis))
        lo = _float_or_none(bounds.get(f"{axis}_min_m"))
        hi = _float_or_none(bounds.get(f"{axis}_max_m"))
        if coord is None or (lo is None and hi is None):
            continue
        checked_axes += 1
        if lo is not None and coord < lo:
            failures.append(f"{axis}={coord:.2f} < {axis}_min_m={lo:.2f}")
        if hi is not None and coord > hi:
            failures.append(f"{axis}={coord:.2f} > {axis}_max_m={hi:.2f}")
    if failures:
        return "outside workspace (" + "; ".join(failures) + ")"
    if checked_axes == 0:
        return "workspace_bounds_unknown"
    return "inside workspace"


def _observed_part_workspace_facts_summary(
    *,
    llm_input: dict[str, Any],
    projected_resources: list[dict[str, Any]],
    projected_parts: list[dict[str, Any]],
) -> str:
    fault_event = dict(llm_input.get("fault_event") or {})
    affected_parts = {
        str(part_name).strip()
        for part_name in (fault_event.get("affected_part_names") or [])
        if str(part_name).strip()
    }

    lines: list[str] = []
    for part in projected_parts:
        if not isinstance(part, dict):
            continue
        part_name = str(part.get("part_name") or "").strip()
        observed_pose = dict(part.get("observed_pose") or {})
        if not part_name or not observed_pose:
            continue
        if affected_parts and part_name not in affected_parts:
            continue

        resource_relations: list[str] = []
        for resource in projected_resources:
            if not isinstance(resource, dict):
                continue
            resource_jid = str(resource.get("resource_jid") or "").strip()
            if not resource_jid:
                continue
            relation = _pose_workspace_relation_text(
                observed_pose,
                dict(resource.get("workspace_bounds") or {}),
            )
            resource_relations.append(f"{resource_jid}: {relation}")

        if resource_relations:
            lines.append(
                f"- {part_name} observed_pose({_pose_xyz_text(observed_pose)}): "
                + "; ".join(resource_relations)
            )
    return "\n".join(lines) if lines else "(none)"


def _candidate_grounding_target_refs(
    *,
    bridge_resources: dict[str, Any],
    projected_parts: list[dict[str, Any]],
) -> list[str]:
    tokens: list[str] = []

    for resource_jid in sorted(bridge_resources):
        entry = dict(bridge_resources.get(resource_jid) or {})
        bridge_snapshot = dict(entry.get("bridge_snapshot") or {})
        static_capabilities = dict(entry.get("static_capabilities") or {})
        reachable_locations = [
            str(token).strip()
            for token in (
                static_capabilities.get("reachability")
                or static_capabilities.get("reachable_locations")
                or bridge_snapshot.get("reachability")
                or bridge_snapshot.get("reachable_locations")
                or []
            )
            if str(token).strip()
        ]
        for token in reachable_locations:
            if token not in tokens:
                tokens.append(token)

    for part in projected_parts:
        if not isinstance(part, dict):
            continue
        for raw_token in (
            part.get("origin_location"),
            part.get("goal_location"),
            part.get("current_location"),
        ):
            token = str(raw_token or "").strip()
            if not token or token.endswith("_gripper"):
                continue
            if token not in tokens:
                tokens.append(token)

    return tokens


def _candidate_grounding_facts_summary(
    *,
    bridge_resources: dict[str, Any],
    projected_resources: list[dict[str, Any]],
    projected_parts: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    target_refs = _candidate_grounding_target_refs(
        bridge_resources=bridge_resources,
        projected_parts=projected_parts,
    )
    if target_refs:
        lines.append("- grounded_target_refs=" + ", ".join(target_refs))

    resources_by_jid = {
        str(row.get("resource_jid") or "").strip(): dict(row)
        for row in projected_resources
        if isinstance(row, dict) and str(row.get("resource_jid") or "").strip()
    }

    for resource_jid in sorted(bridge_resources):
        entry = dict(bridge_resources.get(resource_jid) or {})
        bridge_snapshot = dict(entry.get("bridge_snapshot") or {})
        static_capabilities = dict(entry.get("static_capabilities") or {})
        named_poses = _named_pose_tokens(
            bridge_snapshot.get("named_poses")
            or static_capabilities.get("named_poses")
            or bridge_snapshot.get("available_named_poses")
            or static_capabilities.get("available_named_poses")
            or []
        )
        reachable_refs = [
            str(token).strip()
            for token in (
                static_capabilities.get("reachability")
                or static_capabilities.get("reachable_locations")
                or bridge_snapshot.get("reachability")
                or bridge_snapshot.get("reachable_locations")
                or []
            )
            if str(token).strip()
        ]
        resource_row = dict(resources_by_jid.get(resource_jid) or {})
        named_pose_text = ", ".join(named_poses) if named_poses else "null"
        reachable_text = (
            ", ".join(dict.fromkeys(reachable_refs))
            if reachable_refs
            else "null"
        )
        held_part = str(resource_row.get("held_part") or "").strip() or "null"
        lines.append(
            f"- resource_jid={resource_jid} | named_poses={named_pose_text} | "
            f"grounded_target_refs={reachable_text} | held_part={held_part}"
        )

    for part in projected_parts:
        if not isinstance(part, dict):
            continue
        part_name = str(part.get("part_name") or "").strip()
        if not part_name:
            continue
        current_holder = str(part.get("current_holder_resource_jid") or "").strip() or "null"
        current_location = str(part.get("current_location") or "").strip() or "null"
        goal_location = str(part.get("goal_location") or "").strip() or "null"
        source_ref = "observed_pose" if dict(part.get("observed_pose") or {}) else "null"
        lines.append(
            f"- part_name={part_name} | current_holder_resource_jid={current_holder} | "
            f"current_location={current_location} | goal_location={goal_location} | "
            f"source_ref={source_ref}"
        )

    return "\n".join(lines) if lines else "(none)"


_PRUNED_ACTION_DURABLE_CONSTRAINT_CODES = {
    "workspace_unreachable",
    "holder_conflict",
    "required_part_not_held",
    "source_reference_unavailable",
    "part_relocation_without_carrier",
    "safety_rule_violation",
    "blocker_open",
    "dependency_unsatisfied",
    "order_violation",
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
    feedback_render_style: str = "raw_code",
    excluded_diagnostic_signatures: set[tuple[str, str, str, str, str, str]] | None = None,
) -> str:
    line_by_key: dict[tuple[Any, ...], str] = {}
    excluded_diagnostic_signatures = set(excluded_diagnostic_signatures or set())
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

        if feedback_render_style == "des_event_diagnostic":
            for finding in applicable_findings:
                signature = _candidate_diagnostic_signature(
                    task=task,
                    finding=finding,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )
                if signature in excluded_diagnostic_signatures:
                    continue
                key = signature
                if key in line_by_key:
                    line_by_key.pop(key, None)
                line_by_key[key] = _render_des_event_diagnostic(
                    task=task,
                    finding=finding,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )
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
        # Show only durable constraint codes (supervisor facts, not hints).
        if reason:
            durable_codes = {
                code.strip()
                for code in reason.split("/")
                if code.strip() in _PRUNED_ACTION_DURABLE_CONSTRAINT_CODES
            }
            if durable_codes:
                sentence += f" [{'/'.join(sorted(durable_codes))}]"
        if key in line_by_key:
            line_by_key.pop(key, None)
        line_by_key[key] = sentence

    for row in (pruned_actions or []):
        if not isinstance(row, dict):
            continue
        _add_row(
            task=dict(row.get("task") or row.get("action") or {}),
            findings=[
                dict(row.get("guard") or row)
            ] if isinstance(row, dict) else [],
            explicit_summary=str(row.get("summary") or "").strip(),
            explicit_reason=str(row.get("reason") or "").strip(),
        )

    for row in (rejection_history or []):
        if not isinstance(row, dict):
            continue
        _add_row(
            task=dict(row.get("proposed_next_transition") or {}),
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
        is_new_part = part_name not in parts_by_name
        if part_name not in parts_by_name:
            parts_by_name[part_name] = {"part_name": part_name}
            order.append(part_name)
        part_row = parts_by_name[part_name]
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
        "current_pose",
        "current_pose_ref",
        "workspace_bounds",
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
            "Current phase: Observation / State Estimation.\n"
            "Assess the current world state and decide whether you have enough "
            "grounded state predicates to synthesize a recovery event sequence, "
            "or whether you need additional observations first."
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
# Primitive-generation prompt
# ---------------------------------------------------------------------------


def _render_primitive_generation_prompt(payload: dict[str, Any]) -> str:
    session_state = dict(payload.get("session_state") or {})
    active_event_token = dict(payload.get("primitive_active_outline_event_token") or {})
    active_event_full = dict(payload.get("primitive_active_outline_event") or {})
    if not active_event_token or not active_event_full:
        _cursor, fallback_active_event = _active_primitive_outline_event(session_state)
        if fallback_active_event:
            from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_primitive_generation import (
                _compact_active_event_token,
                _primitive_authoring_event_context,
            )
            if not active_event_token:
                active_event_token = _compact_active_event_token(fallback_active_event)
            if not active_event_full:
                active_event_full = _primitive_authoring_event_context(fallback_active_event)
    accepted_outline_prefix = [
        dict(row)
        for row in (session_state.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    cursor_state = dict(payload.get("primitive_generation_cursor_state") or {})
    if not cursor_state:
        accepted_prefix_len = len(session_state.get("accepted_outline_prefix") or [])
        cursor_idx = int(session_state.get("primitive_generation_cursor") or 0)
        cursor_state = {
            "active_index": cursor_idx,
            "accepted_outline_count": accepted_prefix_len,
            "remaining_events_count": max(accepted_prefix_len - cursor_idx, 0),
        }

    visible_catalog = [
        dict(row)
        for row in (payload.get("primitive_visible_catalog") or [])
        if isinstance(row, dict)
    ]
    published_ref_schema = [
        dict(row)
        for row in (payload.get("primitive_published_ref_schema") or [])
        if isinstance(row, dict)
    ]
    served_context = dict(payload.get("primitive_served_context") or {})
    context_errors = [
        dict(row)
        for row in (payload.get("primitive_context_errors") or [])
        if isinstance(row, dict)
    ]
    input_diagnostics = [
        dict(row)
        for row in (payload.get("primitive_input_diagnostics") or [])
        if isinstance(row, dict)
    ]
    capability_names = [
        str(name).strip()
        for name in (payload.get("primitive_capability_decomposition_names") or [])
        if str(name).strip()
    ]
    active_resource_named_poses = [
        str(name).strip()
        for name in (payload.get("primitive_active_resource_named_poses") or [])
        if str(name).strip()
    ]
    suggested_capability_names = [
        str(name).strip()
        for name in (payload.get("primitive_suggested_capability_decompositions") or [])
        if str(name).strip()
    ]
    memo_summary = [
        dict(row)
        for row in (payload.get("primitive_authoring_memo_summary") or [])
        if isinstance(row, dict)
    ]

    primitive_feedback = _primitive_rejection_feedback_summary(
        [
            dict(row)
            for row in (session_state.get("primitive_rejection_feedback") or [])
            if isinstance(row, dict)
        ]
    )

    sections: list[str] = [
        "Task and Role",
        (
            "You are the active replanner authoring primitive_steps for a DES "
            "fallback recovery session.\n"
            "Current phase: Event-to-Primitive Authoring (event-local).\n"
            "Author primitive_steps for ONLY the active DES transition, using only "
            "the visible primitive catalog for the active resource. There is no "
            "mechanical composer behind you: if you emit no primitive_steps, no "
            "trace is produced for this event.\n"
            "The active event body and the surrounding accepted outline are "
            "provided inline below so you can reason about intent immediately. "
            "For peripheral facts (poses, contract cards, capability decompositions, "
            "projected snapshots, prior decompositions), emit decision=need_context "
            "with refs in context_requests and the bridge will resolve them in the "
            "next turn under 'Served Context'. Do not guess values. Treat "
            "/capability_decompositions/<function_name> as the primary modeled "
            "source for how this resource composes valid low-level actions. When "
            "capability decompositions are available for the active resource, "
            "retrieve the closest applicable one before authoring primitive_steps. "
            "Adapt the retrieved decomposition to the served context instead of "
            "inventing a fresh sequence. The <function_name> token must be one "
            "of the names listed under 'Available Capability Decompositions'; "
            "primitive names such as grasp_part or release_part belong under "
            "/primitive_contracts/<name>, not /capability_decompositions/<function_name>. "
            "Examples of refs worth pulling: "
            "/primitive_contracts/<name> (full contract card), "
            "/resources/<jid>/current_pose, /parts/<name>/observed_pose, "
            "/projected_snapshot/<jid>, /capability_decompositions/<function_name> "
            "(primary modeled task-function decompositions for this resource), /memo/primitive_authoring "
            "(prior accepted decompositions)."
        ),
        "",
        "Active DES Transition (token)",
        _compact_json(active_event_token),
        "",
        "Active DES Transition (full event)",
        _compact_json(active_event_full) if active_event_full else "(unavailable)",
        "",
        "Transition Authoring Cursor",
        _compact_json(cursor_state),
    ]

    if accepted_outline_prefix:
        sections.extend([
            "",
            "Accepted Outline Summary (context for active event)",
            _outline_task_sequence_summary(
                accepted_outline_prefix,
                include_descriptions=True,
            ),
        ])

    sections.extend([
        "",
        "Visible Primitive Catalog (names only; full contract via /primitive_contracts/<name>)",
        _compact_json(visible_catalog),
        "",
        "Published Ref Schema (for context_requests)",
        _compact_json(published_ref_schema),
    ])

    if capability_names:
        sections.extend([
            "",
            "Available Capability Decompositions (names only; retrieve the closest applicable one before authoring)",
            _compact_json(capability_names),
        ])
    if suggested_capability_names:
        sections.extend([
            "",
            "Suggested Capability Decompositions For This Event",
            _compact_json(suggested_capability_names),
        ])

    if memo_summary:
        sections.extend([
            "",
            "Prior Accepted Decompositions (session-scoped memo; full via /memo/primitive_authoring)",
            _compact_json(memo_summary),
        ])

    if served_context:
        sections.extend([
            "",
            "Served Context (retrieved earlier in this event)",
            _compact_json(served_context),
        ])

    if context_errors:
        sections.extend([
            "",
            "Context Request Errors (previous turn)",
            _compact_json(context_errors),
        ])

    if input_diagnostics:
        sections.extend([
            "",
            "Input Diagnostics",
            _compact_json(input_diagnostics),
        ])

    if primitive_feedback != "(none)":
        sections.extend([
            "",
            "Validator Feedback (previous turn)",
            primitive_feedback,
        ])

    active_resource_jid = str(active_event_token.get("resource_jid") or "").strip()
    sections.extend([
        "",
        "Named Pose Rules",
        _compact_json(active_resource_named_poses),
        "- move_to_named_pose may only use named poses advertised for the active resource.",
        (
            "- Full named-pose details remain retrievable via "
            f"/resources/{active_resource_jid}/static_capabilities."
            if active_resource_jid
            else "- Full named-pose details remain retrievable via /resources/<jid>/static_capabilities."
        ),
    ])

    sections.extend([
        "",
        "Output Contract",
        "- Author primitive_steps for ONLY the active DES transition shown above.",
        "- outline_id and resource_jid MUST exactly match the active transition.",
        "- primitive_steps[*].primitive MUST be a name in Visible Primitive Catalog; hidden primitives (e.g. move_pose, get_current_pose) are rejected.",
        "- Each step MUST include params; bind params to grounded values via {\"context_ref\": \"event_facts.<path>\"} referencing deterministic event-local facts from prior data-producing primitives, or via literal values derived from retrieved poses. Do NOT hardcode pose/offset numeric constants.",
        "- Data-producing primitives publish event_facts automatically: get_current_pose -> event_facts.current_pose; detect_parts(part_name=P) -> event_facts.detected_part.P; compute_pick_targets(part_name=P) -> event_facts.pick_targets.P; compute_place_targets(part_name=P) -> event_facts.place_targets.P.",
        "- Do not include store_as or any per-step alias field. Action primitives do not publish custom event_facts.",
        "- If you need a grounded value (current_pose, observed_pose, contract card, projected snapshot, full event body), emit decision=need_context with refs in context_requests and leave primitive_steps empty; do not guess.",
        "- When capability decompositions are available for the active resource, retrieve the closest applicable /capability_decompositions/<function_name> before authoring primitive_steps.",
        "- /capability_decompositions/<function_name> accepts only names from Available Capability Decompositions. Primitive names like grasp_part and release_part must be retrieved via /primitive_contracts/<name> instead.",
        "- Prefer adapting a retrieved capability decomposition to served context over authoring a novel low-level sequence from scratch.",
        "- Author from scratch only when no retrieved capability decomposition fits; name that gap explicitly in rationale.",
        "- If retrieved context or input diagnostics still contradict the active event, emit decision=primitive_blocked and explain the blocker; do not send the event back to outline from this phase.",
        "- Use decision=primitive_steps_ready only when primitive_steps is non-empty AND every param is either a literal known-safe value, a context_ref to a prior step output, or a value derived from previously served context.",
        "- Use decision=need_primitive_revision with a non-empty rationale when the visible catalog cannot safely satisfy the active event (name the contract gap, e.g. 'no visible primitive establishes part orientation for flipped-SG insert').",
        "- Use decision=primitive_blocked when the active event is blocked by contradictory or insufficient primitive-side context that should pause for operator inspection.",
        "- Do not invent observations, resources, parts, or grounded locations. Do not produce a fragile-but-valid trace.",
    ])
    return "\n".join(sections).strip() + "\n"


# ---------------------------------------------------------------------------
# Outline prompt
# ---------------------------------------------------------------------------


def _render_outline_prompt(payload: dict[str, Any]) -> str:
    llm_input = dict(payload.get("llm_input") or {})
    session_state = dict(payload.get("session_state") or {})
    bridge_resources = dict(payload.get("bridge_resources") or {})
    feedback_render_style = _feedback_render_style_token(session_state)
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()
    observation_store = dict(session_state.get("observation_store") or {})
    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    previous_lookahead = list(session_state.get("outline_lookahead") or [])
    pruned_actions = list(session_state.get("pruned_actions") or [])
    current_recovery_blockers = [
        deepcopy(row)
        for row in (payload.get("current_recovery_blockers") or [])
        if isinstance(row, dict)
    ]
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
    resources_by_jid = _finding_resources_by_jid(projected_resources)
    parts_by_name = _finding_parts_by_name(projected_parts)
    candidate_rejection_feedback = [
        deepcopy(row)
        for row in (session_state.get("candidate_rejection_feedback") or [])
        if isinstance(row, dict)
    ]
    primitive_escalation_diagnostics = [
        deepcopy(row)
        for row in (session_state.get("primitive_escalation_diagnostics") or [])
        if isinstance(row, dict)
    ]
    candidate_rejection_history = _candidate_rejection_history(session_state)
    is_single_pass = outline_mode == "single_pass"
    is_candidate_mode = outline_mode == "incremental_candidates_validated"
    recent_candidate_diagnostic_signatures = (
        _candidate_rejection_diagnostic_signatures(
            history=candidate_rejection_history,
            feedback_rows=candidate_rejection_feedback,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
        )
        if is_candidate_mode and feedback_render_style == "des_event_diagnostic"
        else set()
    )
    history_pruned_actions = _history_derived_pruned_actions_summary(
        pruned_actions=pruned_actions,
        rejection_history=[] if is_candidate_mode else rejection_history,
        candidate_rejection_history=[] if is_candidate_mode else candidate_rejection_history,
        candidate_rejection_feedback=[] if is_candidate_mode else candidate_rejection_feedback,
        projected_resources=projected_resources,
        projected_parts=projected_parts,
        feedback_render_style=feedback_render_style,
        excluded_diagnostic_signatures=recent_candidate_diagnostic_signatures,
    )

    if is_single_pass:
        role_text = (
            "You are the active replanner for a bridge recovery session.\n"
            "Current phase: Recovery Event Synthesis (single pass).\n"
            "Propose ALL recovery events as an ordered transition trace, "
            "serialized in JSON field `transition_trace`.\n"
            "Each row must be one recovery transition: one concrete physical action "
            "for one resource, with source-state and target-state predicates."
        )
    elif is_candidate_mode:
        candidate_bound = int(
            session_state.get("candidate_bound")
            or _DEFAULT_CANDIDATE_BOUND
        )
        role_text = (
            "You are the active replanner for a bridge recovery session.\n"
            "Current phase: Recovery Event Candidate Selection.\n"
            "Choose grounded recovery events enabled by the current symbolic state.\n"
            "Accepted events extend the recovery trace toward marked-state conditions.\n"
            "The runtime supervisor may validate and commit at most one event."
        )
        if candidate_rejection_feedback or candidate_rejection_history:
            role_text += (
                "\nUse listed diagnostics as projected-state evidence and avoid "
                "repeating candidate events that remain disabled, blocked_by_supervisor, "
                "or nonprogressing under unchanged facts."
            )
    else:
        role_text = (
            "You are the active replanner for a DES fallback recovery session.\n"
            "Current phase: Recovery Event Synthesis (one transition at a time).\n"
            "Propose exactly ONE next transition to append after the accepted "
            "transition prefix, serialized in JSON field `next_transition`.\n"
            "You may include an optional transition suffix, serialized in JSON "
            "field `transition_suffix`, to show your intended remaining trace; "
            "only `next_transition` will be validated and accepted.\n"
            "When `next_transition` is your LAST recovery event, leave "
            "`transition_suffix` empty to signal that the recovery event sequence "
            "is complete."
        )

    sections: list[str] = [
        "Task and Role",
        role_text,
    ]

    if not is_single_pass and accepted_prefix:
        sections.extend([
            "",
            "Accepted Transition Prefix (keep exactly, do not modify)",
            _outline_task_sequence_summary(
                accepted_prefix,
                include_descriptions=not is_candidate_mode,
            ),
        ])

    if is_candidate_mode:
        rejected_candidate_summary = _candidate_rejection_learning_summary(
            history=candidate_rejection_history,
            feedback_rows=candidate_rejection_feedback,
            feedback_render_style=feedback_render_style,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
        )
        if rejected_candidate_summary != "(none)":
            sections.extend([
                "",
                "Disabled And Blocked Candidate Events",
                rejected_candidate_summary,
            ])

    if primitive_escalation_diagnostics:
        sections.extend([
            "",
            "Primitive Escalation Diagnostics",
            _compact_json(primitive_escalation_diagnostics),
        ])

    if is_candidate_mode:
        if history_pruned_actions != "(none)":
            sections.extend([
                "",
                "Persistently Disabled Candidate Events",
                history_pruned_actions,
            ])
        sections.extend([
            "",
            f"Enabled Event Candidate Budget: {candidate_bound}",
            "",
            "Open Guard / Marking Conditions",
            _current_recovery_blockers_summary(current_recovery_blockers),
            "",
            "Resource Capabilities",
            _resource_capabilities_summary(bridge_resources),
            "",
            "Grounded Event Facts",
            _candidate_grounding_facts_summary(
                bridge_resources=bridge_resources,
                projected_resources=projected_resources,
                projected_parts=projected_parts,
            ),
            # Experiment: keep raw observed poses and workspace bounds visible,
            # but do not precompute the resource/part workspace relationship.
        ])
    elif outline_validation_findings:
        sections.extend([
            "",
            "Active Transition Diagnostics (still unresolved)",
            _outline_validation_summary(
                outline_validation_findings,
                feedback_render_style=feedback_render_style,
                resources_by_jid=resources_by_jid,
                parts_by_name=parts_by_name,
            ),
        ])

    if not is_candidate_mode and history_pruned_actions != "(none)":
        sections.extend([
            "",
            "Persistently Disabled Candidate Events",
            history_pruned_actions,
        ])

    if rejection_history and not is_candidate_mode:
        sections.extend([
            "",
            "Rejected Transition Attempts And Validation Feedback",
            _outline_rejection_history_summary(rejection_history),
        ])

    sections.extend([
        "",
        "Current DES State",
        "Resources",
        _compact_json(_slim_resource_facts(projected_resources)),
        "",
        "Parts",
        _compact_json(_slim_part_facts(projected_parts)),
        "Marked-State Conditions",
        _compact_recovery_objectives(llm_input, projected_parts=projected_parts),
    ])

    # Safety rules stay visible as facts; assembly requirements are hidden in
    # candidate mode to avoid nominal-resource bias.
    sections.extend([
        "",
        "Safety Rules",
        _compact_safety_rules(
            llm_input,
            neutralize_resource_actors=is_candidate_mode,
        ),
    ])
    if not is_candidate_mode:
        sections.extend([
            "",
            "Assembly Requirements",
            _compact_assembly_requirements(llm_input),
        ])

    if is_candidate_mode:
        sections.extend([
            "",
            "Controllable Event JSON Template",
            """```json
{
  "thought": "WHY THESE EVENTS ARE ENABLED FROM THE CURRENT STATE",
  "candidate_events": [
    {
      "resource_jid": "RESOURCE_JID",
      "event_name": "UNIQUE_DES_EVENT_LABEL",
      "description": "STATE-BASED EVENT INTENT (cite source-state predicates and target-state deltas)",
      "part_name": "OPTIONAL_PART_NAME",
      "target_ref": "OPTIONAL_GROUNDED_TARGET_REF",
      "expected_start_state": {
        "resource_state": "SOURCE_RESOURCE_STATE",
        "held_part": "PART_NAME_OR_NULL",
        "part_state": "SOURCE_PART_STATE_OR_NULL",
        "part_location": "SOURCE_PART_LOCATION_OR_NULL",
        "part_holder_resource_jid": "HOLDER_JID_OR_NULL"
      },
      "expected_end_state": {
        "resource_state": "TARGET_RESOURCE_STATE",
        "held_part": "PART_NAME_OR_NULL",
        "part_state": "TARGET_PART_STATE_OR_NULL",
        "part_location": "TARGET_PART_LOCATION_OR_NULL",
        "part_holder_resource_jid": "HOLDER_JID_OR_NULL"
      }
    }
  ]
}
```""",
        ])

    if not is_candidate_mode:
        sections.extend([
            "",
            "Modeled Continuation Gap (unresolved target predicates)",
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
        "Output Constraints",
    ]
    if is_candidate_mode:
        constraints.extend([
            "- Return JSON with field `candidate_events`.",
            f"- `candidate_events` contains 1 to {candidate_bound} candidate events.",
            "- Each event is one physical action by one listed resource.",
            "- Each event must include resource_jid, event_name, description, expected_start_state, and expected_end_state.",
            "- Use part_name and target_ref only when applicable.",
            "- Use only listed resources, parts, and grounded target refs.",
            f"- Returning fewer than {candidate_bound} rows is valid when fewer grounded candidates are worth proposing.",
        ])
    else:
        constraints.extend([
            "- Use only the listed resources.",
            "- Each transition is one physical action by one resource.",
        ])
    constraints.extend([
        "- expected_start_state is the source-state predicate snapshot; "
        "expected_end_state is the target-state predicate snapshot.",
        "- The description must include guard/feasibility reasoning: why the source state enables the event and what target predicates change.",
        "- expected_start_state and expected_end_state may use only: "
        "resource_state, held_part, part_state, part_location, part_holder_resource_jid.",
        "- Use flat scalar values in state objects.",
        "- Use JSON null (not the string \"-\" or \"null\") for unset scalar state fields. "
        "A `null` rendered in Grounded Event Facts maps to JSON null in your output.",
    ])

    if is_single_pass:
        constraints.append(
            "- Propose all recovery events as an ordered transition trace serialized in JSON field `transition_trace`."
        )
    elif not is_candidate_mode:
        constraints.extend([
            "- Propose exactly one next transition serialized in JSON field `next_transition`.",
            "- Optional remaining transitions are serialized in JSON field `transition_suffix`.",
        ])

    sections.extend(constraints)
    return "\n".join(sections).strip() + "\n"


# ---------------------------------------------------------------------------
# Prompt renderer
# ---------------------------------------------------------------------------


def render_multi_turn_phase_prompt(prompt_input: dict[str, Any]) -> str:
    """Render a human-readable prompt string for the current phase."""
    payload = deepcopy(prompt_input or {})
    phase = str(payload.get("phase") or "").strip().lower()

    if phase == "grounding":
        return _render_grounding_prompt(payload)
    if phase == "outline":
        return _render_outline_prompt(payload)
    if phase == "primitive_generation":
        return _render_primitive_generation_prompt(payload)

    # Other phases: placeholder until implemented
    phase_title = _PHASE_TITLES.get(phase, phase)
    return f"Phase: {phase_title}\n\n[Prompt content for {phase} — not yet implemented]\n"


__all__ = [
    "build_multi_turn_phase_prompt_input",
    "multi_turn_phase_response_schema",
    "render_multi_turn_phase_prompt",
]
