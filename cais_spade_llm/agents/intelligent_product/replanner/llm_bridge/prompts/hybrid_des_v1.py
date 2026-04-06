"""Prompt builders for hybrid DES bridge mode — LLM as domain author.

The LLM generates a recovery plant automaton (state-transition model)
rather than proposing individual actions.  The DES solver then composes
this plant with safety DFAs and finds the optimal recovery trace.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _compact_safety_rules(llm_input: dict[str, Any]) -> str:
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


def _compact_resource_capabilities(
    bridge_resources: dict[str, Any],
) -> str:
    lines: list[str] = []
    for jid, res in (bridge_resources or {}).items():
        res = dict(res or {})
        caps = dict(res.get("static_capabilities") or res.get("capabilities") or {})
        parts: list[str] = [f"manipulate parts"]
        named_poses = res.get("named_poses") or caps.get("named_poses")
        if isinstance(named_poses, dict):
            parts.append(f"named poses {', '.join(named_poses.keys())}")
        elif isinstance(named_poses, list):
            parts.append(f"named poses {', '.join(str(p) for p in named_poses)}")
        bounds = caps.get("workspace_bounds") or res.get("workspace_bounds") or {}
        if bounds:
            ws_parts: list[str] = []
            for axis in ("x", "y", "z"):
                lo = bounds.get(f"{axis}_min_m")
                hi = bounds.get(f"{axis}_max_m")
                if lo is not None and hi is not None:
                    ws_parts.append(f"{axis}[{lo},{hi}]")
            if ws_parts:
                parts.append(f"workspace {', '.join(ws_parts)}")
        lines.append(f"- {jid}: {'; '.join(parts)}")
    return "\n".join(lines) if lines else "(none)"


def _compact_blockers(
    current_recovery_blockers: list[dict[str, Any]] | None,
) -> str:
    if not current_recovery_blockers:
        return "(none)"
    lines: list[str] = []
    for b in current_recovery_blockers:
        if isinstance(b, dict):
            text = str(b.get("description") or b.get("reason") or b.get("text") or "").strip()
            if text:
                lines.append(f"- {text}")
        elif isinstance(b, str):
            lines.append(f"- {b}")
    return "\n".join(lines) if lines else "(none)"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


def _domain_generation_response_schema() -> dict[str, Any]:
    return {
        "name": "hybrid_des_domain_generation",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": (
                        "Your reasoning about the gap between the current world "
                        "state and the recovery objectives, and what actions are "
                        "needed to bridge it."
                    ),
                },
                "plant": {
                    "type": "object",
                    "description": "Recovery plant automaton.",
                    "properties": {
                        "states": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "All state names in the plant.",
                        },
                        "initial": {
                            "type": "string",
                            "description": "Initial state (matches current world state).",
                        },
                        "marked": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Goal/accepting states (recovery complete).",
                        },
                        "events": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Unique event label.",
                                    },
                                    "from": {
                                        "type": "string",
                                        "description": "Source state.",
                                    },
                                    "to": {
                                        "type": "string",
                                        "description": "Target state.",
                                    },
                                    "resource_jid": {
                                        "type": "string",
                                        "description": "Which resource executes this.",
                                    },
                                    "action_type": {
                                        "type": "string",
                                        "description": (
                                            "Category of physical action. "
                                            "You MUST strictly use one of: "
                                            "'pick_part', 'place_prepare', 'place_part', "
                                            "'release_part', 'move_resource', 'resequence'."
                                        ),
                                    },
                                    "part_name": {
                                        "type": "string",
                                        "description": "Part involved (omit for resource-only).",
                                    },
                                    "target_ref": {
                                        "type": "string",
                                        "description": (
                                            "Destination: named pose, station, "
                                            "or 'observed_pose'."
                                        ),
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Human-readable action description.",
                                    },
                                },
                                "required": [
                                    "name",
                                    "from",
                                    "to",
                                    "resource_jid",
                                    "action_type",
                                    "description",
                                ],
                            },
                        },
                    },
                    "required": ["states", "initial", "marked", "events"],
                },
            },
            "required": ["thought", "plant"],
        },
    }


def hybrid_des_phase_response_schema(phase: str) -> dict[str, Any]:
    """Return the JSON response schema for the given hybrid DES phase."""
    normalized = phase.strip().lower()
    if normalized == "domain_generation":
        return _domain_generation_response_schema()
    raise ValueError(f"Unknown hybrid DES phase: {phase!r}")


# ---------------------------------------------------------------------------
# Domain generation prompt
# ---------------------------------------------------------------------------


def build_hybrid_domain_generation_prompt(
    prompt_input: dict[str, Any],
) -> str:
    """Build the LLM prompt for domain generation phase.

    The LLM must produce a recovery plant automaton — a state-transition
    model defining what recovery actions exist and how they connect.
    """
    llm_input = dict(prompt_input.get("llm_input") or {})
    bridge_resources = dict(prompt_input.get("bridge_resources") or {})
    recovery_gap_state = dict(prompt_input.get("recovery_gap_state") or {})
    current_recovery_blockers = list(prompt_input.get("current_recovery_blockers") or [])

    resource_state = list(recovery_gap_state.get("resource_state") or [])
    part_state = list(recovery_gap_state.get("part_state") or [])

    sections: list[str] = []

    # Task and role
    sections.append(
        "Task and Role\n"
        "You are the recovery domain modeler for a DES-based fallback recovery session.\n"
        "Your job is to generate a recovery plant automaton — a state-transition model\n"
        "defining all physically valid actions that bridge the gap between the current\n"
        "world state and the recovery objectives listed below.\n"
        "\n"
        "The nominal plan has reached a state it cannot continue from. You must reason\n"
        "about what actions are needed to restore progress — these actions were never\n"
        "modeled in the original plan. Analyze the current resource states, part states,\n"
        "and recovery objectives to determine what transitions are possible.\n"
        "\n"
        "The runtime will compose your plant with safety specifications and find\n"
        "the optimal action sequence. You define WHAT is possible; the solver finds HOW."
    )

    # Recovery blockers
    sections.append(
        "Current Recovery Blockers\n"
        + _compact_blockers(current_recovery_blockers)
    )

    # Resource capabilities
    sections.append(
        "Resource Capabilities\n"
        + _compact_resource_capabilities(bridge_resources)
    )

    # Current state
    sections.append("Current Resource State\n" + _compact_json(resource_state))
    sections.append("Current Part State\n" + _compact_json(part_state))

    # Recovery objectives
    sections.append(
        "Recovery Objectives\n"
        + _compact_recovery_objectives(llm_input, projected_parts=part_state)
    )

    # Safety rules (so LLM can reason about ordering constraints)
    safety_text = _compact_safety_rules(llm_input)
    if safety_text != "(none)":
        sections.append("Safety Rules (the solver enforces these; consider them in your design)\n" + safety_text)

    # Available vocabulary
    vocab_lines: list[str] = []
    vocab_lines.append(f"Resources: {', '.join(sorted(bridge_resources.keys()))}")
    part_names = [str(p.get("part_name") or "").strip() for p in part_state if isinstance(p, dict)]
    vocab_lines.append(f"Parts: {', '.join(sorted(set(p for p in part_names if p)))}")
    locations: set[str] = set()
    for p in part_state:
        if not isinstance(p, dict):
            continue
        for f in ("current_location", "origin_location", "goal_location"):
            loc = str(p.get(f) or "").strip()
            if loc:
                locations.add(loc)
    for jid, res in bridge_resources.items():
        res = dict(res or {})
        np = res.get("named_poses")
        if isinstance(np, dict):
            locations.update(np.keys())
        elif isinstance(np, list):
            locations.update(str(x) for x in np)
    vocab_lines.append(f"Known locations: {', '.join(sorted(locations))}")
    vocab_lines.append("Observed poses: use 'observed_pose' as target_ref for parts with observed_pose data")
    
    sections.append(
        "Available Vocabulary\n" 
        + "\n".join(vocab_lines) + "\n\n"
        "Strict Action Types:\n"
        "- pick_part: Acquire a part (requires target_ref).\n"
        "- place_prepare: Move resource to approach/hover over a target destination BEFORE placing.\n"
        "- place_part: Physically place the part at the target_ref.\n"
        "- release_part: Let go of the part.\n"
        "- move_resource: Move a resource to a named pose or location."
    )

    # Plant shape example
    sections.append(
        "Output Shape (produce a plant matching this structure with actual values)\n"
        "```json\n"
        "{\n"
        '  "states": ["s0", "s1", "s2", "..."],\n'
        '  "initial": "s0",\n'
        '  "marked": ["s_goal"],\n'
        '  "events": [\n'
        "    {\n"
        '      "name": "unique_event_label",\n'
        '      "from": "s0",\n'
        '      "to": "s1",\n'
        '      "resource_jid": "RESOURCE_JID",\n'
        '      "action_type": "ACTION_CATEGORY (use your own label)",\n'
        '      "part_name": "PART_NAME (omit for resource-only)",\n'
        '      "target_ref": "DESTINATION (named pose, station, or observed_pose)",\n'
        '      "description": "what this action does"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```"
    )

    # Hard Constraints
    sections.append(
        "Hard Constraints\n"
        "- Use only the listed resources, parts, and locations.\n"
        "- Each event must strictly represent a single physical action by one resource.\n"
        "- You MUST use the exact strict action types.\n"
        "- The initial state must equal the current world state.\n"
        "- Marked states must equal all recovery objectives achieved.\n"
        "- Ordering dependencies must be encoded structurally within the automaton state graph.\n"
        "- Every event name must be unique.\n"
        "- The plant must be strictly deterministic."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Feedback prompt (revision after validation/solving failure)
# ---------------------------------------------------------------------------


def build_hybrid_feedback_prompt(
    prompt_input: dict[str, Any],
    *,
    plant_findings_text: str = "",
    solver_diagnostic_text: str = "",
    feasibility_findings_text: str = "",
    previous_plant_json: str = "",
) -> str:
    """Build a revision prompt when the previous plant was invalid or unsolvable."""
    sections: list[str] = []

    sections.append(
        "Task and Role\n"
        "Your previous recovery plant automaton was rejected. Revise it based on the "
        "feedback below. Generate a complete replacement plant (not a patch)."
    )

    # Multi-turn style diagnostics blocking
    if solver_diagnostic_text:
        sections.append("--- PROPOSED SOLVER TRACE ---\n" + solver_diagnostic_text)

    if feasibility_findings_text or plant_findings_text:
        feedback_parts = []
        if plant_findings_text:
            feedback_parts.append("Plant Structural Findings:\n" + plant_findings_text)
        if feasibility_findings_text:
            feedback_parts.append("Physical / Ordering Feasibility Findings:\n" + feasibility_findings_text)
        sections.append("--- REJECTED VALIDATION FEEDBACK ---\n\n" + "\n\n".join(feedback_parts))

    if previous_plant_json:
        sections.append("--- PREVIOUS REJECTED PLANT ---\n```json\n" + previous_plant_json + "\n```")

    # Re-include the world state context
    base_prompt = build_hybrid_domain_generation_prompt(prompt_input)
    # Extract from "Current Recovery Blockers" onward
    marker = "Current Recovery Blockers"
    idx = base_prompt.find(marker)
    if idx >= 0:
        sections.append("--- CONTEXT ---\n\n" + base_prompt[idx:])

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Prompt input builder
# ---------------------------------------------------------------------------


def build_hybrid_des_prompt_input(
    *,
    phase: str,
    llm_input: dict[str, Any],
    session_state: dict[str, Any],
    bridge_resources: dict[str, Any] | None = None,
    recovery_gap_state: dict[str, Any] | None = None,
    current_recovery_blockers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the structured prompt input payload for a given phase."""
    return {
        "phase": phase,
        "llm_input": deepcopy(llm_input),
        "session_state": deepcopy(session_state),
        "bridge_resources": deepcopy(bridge_resources or {}),
        "recovery_gap_state": deepcopy(recovery_gap_state or {}),
        "current_recovery_blockers": deepcopy(current_recovery_blockers or []),
    }


__all__ = [
    "build_hybrid_des_prompt_input",
    "build_hybrid_domain_generation_prompt",
    "build_hybrid_feedback_prompt",
    "hybrid_des_phase_response_schema",
]
