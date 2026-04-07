"""Prompt builders for multi-turn v2 bridge — one-task-at-a-time outline."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
    filter_synthesis_primitive_catalog,
)


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
        "action_name": {"type": "string"},
        "part_name": {"type": "string"},
        "target_ref": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["resource_jid", "action_name", "description"],
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
                "decision": {
                    "type": "string",
                    "enum": [
                        "primitive_event_ready",
                        "need_primitive_revision",
                        "need_outline_revision",
                    ],
                },
                "outline_id": {"type": "string"},
                "resource_jid": {"type": "string"},
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
                "primitive_steps",
            ],
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
    action_name = str(task.get("action_name") or "").strip()
    action_type = str(task.get("action_type") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    target_ref = str(task.get("target_ref") or "").strip()
    if action_name:
        summary = action_name
        lower_summary = action_name.lower()
        if part_name and part_name.lower() not in lower_summary:
            summary = f"{summary} {part_name}"
            lower_summary = summary.lower()
        if target_ref and target_ref.lower() not in lower_summary:
            summary = f"{summary} to {target_ref}"
        return summary
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
        description = str(task.get("description") or "").strip()
        if description:
            lines.append(f"  ({description})")
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


def _candidate_rejection_history(session_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rejected candidate batches since the last accepted outline event."""
    history: list[dict[str, Any]] = []
    for turn in (session_state.get("turns") or []):
        if not isinstance(turn, dict):
            continue
        if str(turn.get("phase") or "").strip().lower() != "outline":
            continue
        if (
            isinstance(turn.get("selected_next_task"), dict)
            or isinstance(turn.get("next_task"), dict)
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
) -> str:
    """Summarize rejected candidate patterns without repeating action labels."""
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


def _slim_primitive_catalog_for_prompt(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slim: list[dict[str, Any]] = []
    for raw_entry in filter_synthesis_primitive_catalog(catalog or []):
        if not isinstance(raw_entry, dict):
            continue
        name = str(raw_entry.get("name") or "").strip()
        if not name:
            continue
        description = str(
            raw_entry.get("description")
            or raw_entry.get("semantic_summary")
            or ""
        ).strip()
        if name == "release_part":
            description = "Release the currently held part."
        elif name == "grasp_part":
            description = "Secure the target part for transport."
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
        slim.append(entry)
    return slim


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
# Primitive-generation prompt
# ---------------------------------------------------------------------------


def _render_primitive_generation_prompt(payload: dict[str, Any]) -> str:
    llm_input = dict(payload.get("llm_input") or {})
    session_state = dict(payload.get("session_state") or {})
    bridge_resources = dict(payload.get("bridge_resources") or {})
    observation_store = dict(session_state.get("observation_store") or {})
    accepted_prefix = [
        dict(row)
        for row in (session_state.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    cursor, active_event = _active_primitive_outline_event(session_state)

    projected_resources = _projected_outline_resources(
        llm_input=llm_input,
        session_state=session_state,
    )
    projected_parts = _projected_outline_parts(
        llm_input=llm_input,
        session_state=session_state,
    )

    active_resource_jid = str(
        dict(active_event or {}).get("resource_jid") or ""
    ).strip()
    active_resource_entry = dict(bridge_resources.get(active_resource_jid) or {})
    primitive_catalog = _slim_primitive_catalog_for_prompt(
        [
            dict(row)
            for row in (active_resource_entry.get("primitive_catalog") or [])
            if isinstance(row, dict)
        ]
    )
    primitive_feedback = _primitive_rejection_feedback_summary(
        [
            dict(row)
            for row in (session_state.get("primitive_rejection_feedback") or [])
            if isinstance(row, dict)
        ]
    )
    active_resources, active_parts = _active_event_start_facts_for_prompt(
        resources=projected_resources,
        parts=projected_parts,
        active_event=active_event,
    )

    sections: list[str] = [
        "Task and Role",
        (
            "You are the active replanner for a DES fallback recovery session.\n"
            "Current phase: Primitive Generation.\n"
            "Synthesize executable controller primitive steps for exactly one "
            "accepted outline event."
        ),
        "",
        "Accepted Outline Context",
        _outline_task_sequence_summary(accepted_prefix),
        "",
        "Active Outline Event",
        _compact_json(active_event or {}),
        "",
        "Primitive Generation Cursor",
        _compact_json({
            "active_index": cursor,
            "accepted_outline_count": len(accepted_prefix),
            "remaining_events_after_this": max(len(accepted_prefix) - cursor - 1, 0),
        }),
    ]

    if primitive_feedback != "(none)":
        sections.extend([
            "",
            "Primitive Rejection Feedback",
            primitive_feedback,
        ])

    sections.extend([
        "",
        "Current Resource State",
        _compact_json(_slim_resource_facts(active_resources)),
        "",
        "Current Part State",
        _compact_json(_slim_part_facts(active_parts)),
        "",
        "Session Observation Store",
        _compact_json(observation_store),
        "",
        "Active Resource Primitive Catalog",
        _compact_json(primitive_catalog),
        "",
        "Safety Rules",
        _compact_safety_rules(llm_input, neutralize_resource_actors=True),
        "",
        "Output Constraints",
        "- Implement only the Active Outline Event.",
        "- Use only primitives listed in Active Resource Primitive Catalog.",
        "- primitive_steps must be ordered controller primitive calls.",
        "- Each primitive step must include primitive and params.",
        "- Do not invent observations, resources, parts, or grounded locations.",
        "- Use decision primitive_event_ready when primitive_steps are ready for validation.",
        "- Use need_primitive_revision only when prior primitive feedback cannot be addressed in the same event.",
        "- Use need_outline_revision only when the accepted outline event is not implementable with the listed primitives.",
    ])
    return "\n".join(sections).strip() + "\n"


# ---------------------------------------------------------------------------
# Outline prompt
# ---------------------------------------------------------------------------


def _render_outline_prompt(payload: dict[str, Any]) -> str:
    llm_input = dict(payload.get("llm_input") or {})
    session_state = dict(payload.get("session_state") or {})
    bridge_resources = dict(payload.get("bridge_resources") or {})
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
        candidate_bound = int(session_state.get("des_candidate_bound") or 3)
        role_text = (
            "You are the active replanner for a DES fallback recovery session.\n"
            "Current phase: Outline (Candidate Event Selection).\n"
            f"Propose up to {candidate_bound} candidate recovery events from "
            "the current world state.\n"
            "Each candidate is one physical action for one resource.\n"
            "The runtime supervisor will validate and commit at most one."
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
        rejected_candidate_summary = _candidate_rejection_learning_summary(
            history=candidate_rejection_history,
            feedback_rows=candidate_rejection_feedback,
        )
        if rejected_candidate_summary != "(none)":
            sections.extend([
                "",
                "Recent Rejected Candidate Feedback",
                rejected_candidate_summary,
            ])

    if is_candidate_mode:
        sections.extend([
            "",
            "Current Recovery Conditions",
            _current_recovery_blockers_summary(current_recovery_blockers),
            "",
            "Resource Capabilities",
            _resource_capabilities_summary(bridge_resources),
            # Experiment: keep raw observed poses and workspace bounds visible,
            # but do not precompute the resource/part workspace relationship.
        ])
    elif outline_validation_findings:
        sections.extend([
            "",
            "Active Validation Findings (still unresolved)",
            _outline_validation_summary(outline_validation_findings),
        ])

    if not is_candidate_mode and history_pruned_actions != "(none)":
        sections.extend([
            "",
            "Rejected Events (do not re-propose)",
            history_pruned_actions,
        ])

    if rejection_history and not is_candidate_mode:
        sections.extend([
            "",
            "Rejected Outline Attempts And Validation Feedback",
            _outline_rejection_history_summary(rejection_history),
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
            "Candidate Shape Example",
            """```json
{
  "resource_jid": "RESOURCE_JID",
  "action_name": "short action label",
  "description": "short description",
  "part_name": "PART_NAME or omit",
  "target_ref": "DESTINATION_REF or omit"
}
```""",
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
        "Output Constraints",
        "- Use only the listed resources.",
        "- Each event is one physical action for one resource.",
    ]
    if not is_candidate_mode:
        constraints.extend([
            "- expected_start_state and expected_end_state may use only: "
            "resource_state, held_part, part_state, part_location, part_holder_resource_jid.",
            "- Use flat scalar values in state objects.",
        ])

    if is_single_pass:
        constraints.append(
            "- Propose all recovery tasks in outline_tasks as an ordered sequence."
        )
    elif is_candidate_mode:
        constraints.extend([
            f"- Propose 1 to {candidate_bound} events in candidate_tasks.",
            "- Each event must include resource_jid, action_name, and description.",
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
    if phase == "primitive_generation":
        return _render_primitive_generation_prompt(payload)

    # Other phases: placeholder until implemented
    phase_title = _PHASE_TITLES.get(phase, phase)
    return f"Phase: {phase_title}\n\n[Prompt content for {phase} — not yet implemented]\n"


__all__ = [
    "build_multi_turn_v2_phase_prompt_input",
    "multi_turn_v2_phase_response_schema",
    "render_multi_turn_v2_phase_prompt",
]
