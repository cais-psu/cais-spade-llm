"""Prompt builders for hybrid DES bridge mode: LLM as domain author.

The LLM generates a recovery plant automaton (state-transition model)
rather than proposing individual actions.  The DES solver then composes
this plant with safety DFAs and finds the optimal recovery trace.

Vocabulary remains data-driven for schemas and deployment overrides, but
the prompt deliberately keeps canonical predicate and marking vocabulary
out of the foreground so hybrid mode stays general across failures.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


# ---------------------------------------------------------------------------
# Domain vocabulary (data-driven prompt configuration)
# ---------------------------------------------------------------------------

# These defaults intentionally stay empty. Deployments may inject vocabulary
# overrides, but hybrid mode should not carry hidden canonical predicate,
# effect, marking, or blocker labels by default.
_DEFAULT_PREDICATE_TEMPLATES: list[dict[str, Any]] = []
_DEFAULT_EFFECT_KINDS: list[dict[str, str]] = []


_DEFAULT_SPECIAL_LOCATION_REFS: dict[str, str] = {
    "observed_pose": (
        "use as location_ref for parts whose only available location is an "
        "observed pose (no named location)"
    ),
}


_DEFAULT_MARKING_TEMPLATE_FRAGMENTS: list[str] = []
_DEFAULT_BLOCKER_GUIDANCE: dict[str, dict[str, Any]] = {}


def build_domain_vocabulary(
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a hybrid-DES domain vocabulary, merged with optional overrides.

    The returned dict is plain data and can be serialized / diffed.  All
    fields are optional at call sites; helpers fall back to defaults.
    """
    vocab: dict[str, Any] = {
        "predicate_templates": deepcopy(_DEFAULT_PREDICATE_TEMPLATES),
        "effect_kinds": deepcopy(_DEFAULT_EFFECT_KINDS),
        "special_location_refs": deepcopy(_DEFAULT_SPECIAL_LOCATION_REFS),
        "marking_template_fragments": deepcopy(_DEFAULT_MARKING_TEMPLATE_FRAGMENTS),
        "blocker_guidance": deepcopy(_DEFAULT_BLOCKER_GUIDANCE),
    }
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key == "special_target_refs" and "special_location_refs" not in (overrides or {}):
            vocab["special_location_refs"] = deepcopy(value)
            vocab["special_target_refs"] = deepcopy(value)
            continue
        vocab[key] = deepcopy(value)
    if "special_target_refs" not in vocab:
        vocab["special_target_refs"] = deepcopy(vocab.get("special_location_refs") or {})
    if "special_location_refs" not in vocab:
        vocab["special_location_refs"] = deepcopy(vocab.get("special_target_refs") or {})
    return vocab


# ---------------------------------------------------------------------------
# Helpers (generic)
# ---------------------------------------------------------------------------


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _resource_workspace_bounds(resource_entry: dict[str, Any]) -> dict[str, Any] | None:
    entry = dict(resource_entry or {})
    caps = dict(entry.get("static_capabilities") or entry.get("capabilities") or {})
    bounds = caps.get("workspace_bounds") or entry.get("workspace_bounds")
    return dict(bounds) if isinstance(bounds, dict) and bounds else None


def _pose_in_bounds(pose: dict[str, Any], bounds: dict[str, Any]) -> bool:
    try:
        x = float(pose["x"])
        y = float(pose["y"])
        z = float(pose["z"])
    except (KeyError, TypeError, ValueError):
        return False
    for axis, value in (("x", x), ("y", y), ("z", z)):
        lo = bounds.get(f"{axis}_min_m")
        hi = bounds.get(f"{axis}_max_m")
        if lo is not None and value < float(lo):
            return False
        if hi is not None and value > float(hi):
            return False
    return True


def _grounded_pose_reachability(
    *,
    part_state: list[dict[str, Any]],
    bridge_resources: dict[str, Any],
) -> str:
    lines: list[str] = []
    for row in part_state:
        if not isinstance(row, dict):
            continue
        part_name = str(row.get("part_name") or "").strip()
        if not part_name:
            continue
        pose = row.get("observed_pose")
        holder = str(row.get("current_holder_resource_jid") or "").strip()
        if holder or not isinstance(pose, dict) or not pose:
            continue
        reachable: list[str] = []
        for jid, resource_entry in (bridge_resources or {}).items():
            bounds = _resource_workspace_bounds(dict(resource_entry or {}))
            if bounds and _pose_in_bounds(pose, bounds):
                reachable.append(str(jid))
        reachable = sorted(jid for jid in reachable if jid)
        if reachable:
            lines.append(f"- part '{part_name}': grounded pose reachable by {', '.join(reachable)}.")
            continue
        lines.append(
            f"- part '{part_name}': grounded pose is outside every listed resource workspace."
        )
    return "\n".join(lines) if lines else "(none)"


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
            text = str(b.get("reason") or b.get("text") or b.get("description") or "").strip()
            if ":" in text:
                label, remainder = text.split(":", 1)
                label_words = label.replace("/", " ").replace("_", " ").replace("-", " ").split()
                if label_words and all(word.isupper() for word in label_words):
                    text = remainder.strip()
            if text:
                lines.append(f"- {text}")
        elif isinstance(b, str):
            lines.append(f"- {b}")
    return "\n".join(lines) if lines else "(none)"


def _allowed_location_refs(
    *,
    part_state: list[dict[str, Any]],
    bridge_resources: dict[str, Any],
    vocab: dict[str, Any],
) -> str:
    refs: set[str] = set()
    for row in part_state:
        if not isinstance(row, dict):
            continue
        for field_name in ("current_location", "origin_location", "goal_location"):
            value = str(row.get(field_name) or "").strip()
            if value:
                refs.add(value)
    for res in (bridge_resources or {}).values():
        res = dict(res or {})
        named_poses = res.get("named_poses")
        if isinstance(named_poses, dict):
            refs.update(str(key) for key in named_poses.keys() if str(key).strip())
        elif isinstance(named_poses, list):
            refs.update(str(item) for item in named_poses if str(item).strip())
    refs.update(
        str(token).strip()
        for token in (
            vocab.get("special_location_refs")
            or vocab.get("special_target_refs")
            or {}
        ).keys()
        if str(token).strip()
    )
    return "\n".join(f"- {ref}" for ref in sorted(refs)) if refs else "(none)"


def _prior_rejections_text(
    *,
    plant_findings_text: str = "",
    solver_diagnostic_text: str = "",
    feasibility_findings_text: str = "",
    persistent_constraint_summary: list[str] | None = None,
    revision_history_summary_text: str = "",
) -> str:
    sections: list[str] = []
    if plant_findings_text:
        sections.append("Plant findings\n" + plant_findings_text)
    if solver_diagnostic_text:
        sections.append("Solver result\n" + solver_diagnostic_text)
    if feasibility_findings_text:
        sections.append("Validation findings\n" + feasibility_findings_text)
    if revision_history_summary_text:
        sections.append("Revision history\n" + revision_history_summary_text)
    rows = [
        str(row).strip()
        for row in (persistent_constraint_summary or [])
        if str(row).strip()
        and not str(row).strip().startswith("state=")
        and "derived_state_labels=" not in str(row)
    ]
    if rows:
        sections.append("Rejected constraints\n" + "\n".join(f"- {row}" for row in rows[-12:]))
    return "\n\n".join(sections) if sections else "(none)"


def _compact_continuation_conditions(
    recovery_gap_state: dict[str, Any],
    current_recovery_blockers: list[dict[str, Any]] | None,
) -> str:
    resource_state = list(recovery_gap_state.get("resource_state") or [])
    part_state = list(recovery_gap_state.get("part_state") or [])
    blockers = list(current_recovery_blockers or [])
    conditions: list[str] = []
    for row in part_state:
        if not isinstance(row, dict):
            continue
        part_name = str(row.get("part_name") or "").strip()
        goal_location = str(row.get("goal_location") or "").strip()
        if part_name and goal_location:
            conditions.append(f"- {part_name} must be at {goal_location}")
    terminal_resources = {
        str(blocker.get("resource_jid") or "").strip()
        for blocker in blockers
        if isinstance(blocker, dict) and blocker.get("kind") == "resource_terminal_state"
    }
    for row in resource_state:
        if not isinstance(row, dict):
            continue
        resource_jid = str(row.get("resource_jid") or "").strip()
        if resource_jid in terminal_resources:
            conditions.append(f"- {resource_jid} must be restored to an available execution state")
    return "\n".join(conditions) if conditions else "(none)"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


def _domain_generation_response_schema(
    vocab: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_vocab = vocab or build_domain_vocabulary()
    special_tokens = [
        str(tok).strip()
        for tok in (
            active_vocab.get("special_location_refs")
            or active_vocab.get("special_target_refs")
            or {}
        ).keys()
        if str(tok).strip()
    ]
    primary_token = special_tokens[0] if special_tokens else ""
    if primary_token:
        location_ref_desc = (
            f"Destination: named pose, station, or '{primary_token}'."
        )
    else:
        location_ref_desc = "Destination: named pose or station."

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
                        "state_metadata": {
                            "type": "object",
                            "description": (
                                "Optional audit metadata keyed by state name. "
                                "Does not change solver semantics."
                            ),
                            "additionalProperties": {
                                "type": "object",
                                "properties": {
                                    "atomic_bindings": {
                                        "type": "object",
                                        "description": (
                                            "Optional audit facts for this state. "
                                            "Use zero or more short labels as keys and boolean "
                                            "true/false values only. These bindings do not "
                                            "change event semantics."
                                        ),
                                        "additionalProperties": {"type": "boolean"},
                                    },
                                    "marking_predicate": {
                                        "type": "string",
                                    },
                                },
                            },
                        },
                        "marked_state_metadata": {
                            "type": "object",
                            "description": (
                                "Optional marked-state audit metadata keyed by state name. "
                                "Does not change solver semantics."
                            ),
                            "additionalProperties": {
                                "type": "object",
                                "properties": {
                                    "marking_predicate": {"type": "string"},
                                },
                            },
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
                                    "part_name": {
                                        "type": "string",
                                        "description": "Part involved (omit for resource-only).",
                                    },
                                    "location_ref": {
                                        "type": "string",
                                        "description": location_ref_desc,
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Human-readable action description.",
                                    },
                                    "projected_effect": {
                                        "type": "object",
                                        "description": (
                                            "Optional symbolic state updates after this event. "
                                            "These after-state facts are the authoritative bridge contract "
                                            "for how this event closes part of the recovery gap."
                                        ),
                                        "properties": {
                                            "resource": {
                                                "type": "object",
                                                "additionalProperties": True,
                                            },
                                            "part": {
                                                "type": "object",
                                                "additionalProperties": True,
                                            },
                                            "resources": {
                                                "type": "object",
                                                "additionalProperties": {
                                                    "type": "object",
                                                    "additionalProperties": True,
                                                },
                                            },
                                            "parts": {
                                                "type": "object",
                                                "additionalProperties": {
                                                    "type": "object",
                                                    "additionalProperties": True,
                                                },
                                            },
                                        },
                                    },
                                },
                                "required": [
                                    "name",
                                    "from",
                                    "to",
                                    "resource_jid",
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


def _primitive_generation_response_schema() -> dict[str, Any]:
    return {
        "name": "hybrid_des_primitive_generation",
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
                        "need_domain_revision",
                    ],
                },
                "event_index": {"type": "integer"},
                "event_name": {"type": "string"},
                "resource_jid": {"type": "string"},
                "primitive_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "primitive": {"type": "string"},
                            "params": {
                                "type": "object",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["primitive", "params"],
                    },
                },
            },
            "required": [
                "thought",
                "decision",
                "event_index",
                "resource_jid",
                "primitive_steps",
            ],
        },
    }


def hybrid_des_phase_response_schema(
    phase: str,
    *,
    vocab: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the JSON response schema for the given hybrid DES phase."""
    normalized = phase.strip().lower()
    if normalized == "domain_generation":
        return _domain_generation_response_schema(vocab)
    if normalized == "primitive_generation":
        return _primitive_generation_response_schema()
    raise ValueError(f"Unknown hybrid DES phase: {phase!r}")


# ---------------------------------------------------------------------------
# Domain generation prompt
# ---------------------------------------------------------------------------


def build_hybrid_domain_generation_prompt(
    prompt_input: dict[str, Any],
) -> str:
    """Build the compact LLM prompt for domain generation phase."""
    llm_input = dict(prompt_input.get("llm_input") or {})
    bridge_resources = dict(prompt_input.get("bridge_resources") or {})
    recovery_gap_state = dict(prompt_input.get("recovery_gap_state") or {})
    current_recovery_blockers = list(prompt_input.get("current_recovery_blockers") or [])
    persistent_constraint_summary = list(prompt_input.get("persistent_constraint_summary") or [])
    revision_summary_text = str(prompt_input.get("revision_history_summary_text") or "").strip()
    vocab = dict(prompt_input.get("domain_vocabulary") or build_domain_vocabulary())

    resource_state = list(recovery_gap_state.get("resource_state") or [])
    part_state = list(recovery_gap_state.get("part_state") or [])
    sections: list[str] = [
        "Task and Role",
        (
            "You are the domain modeler for a DES fallback recovery session.\n"
            "Current phase: Recovery Plant Synthesis.\n"
            "Generate a recovery plant automaton. Each event is one grounded "
            "transition for one resource. The runtime will solve and validate "
            "the plant; do not generate controller primitive steps here."
        ),
        "",
        "Current Resource State",
        _compact_json(resource_state),
        "",
        "Current Part State",
        _compact_json(part_state),
        "",
        "Grounded Pose Reachability",
        _grounded_pose_reachability(
            part_state=part_state,
            bridge_resources=bridge_resources,
        ),
        "",
        "Open Recovery Conditions",
        "\n".join(
            line for line in (
                _compact_continuation_conditions(recovery_gap_state, current_recovery_blockers),
                _compact_blockers(current_recovery_blockers),
                _compact_recovery_objectives(llm_input, projected_parts=part_state),
            )
            if line and line != "(none)"
        ) or "(none)",
        "",
        "Available Resources",
        _compact_resource_capabilities(bridge_resources),
        "",
        "Allowed Location References",
        _allowed_location_refs(
            part_state=part_state,
            bridge_resources=bridge_resources,
            vocab=vocab,
        ),
    ]

    safety_text = _compact_safety_rules(llm_input)
    if safety_text != "(none)":
        sections.extend(["", "Safety Rules", safety_text])

    prior_rejections = _prior_rejections_text(
        persistent_constraint_summary=persistent_constraint_summary,
        revision_history_summary_text=revision_summary_text,
    )
    if prior_rejections != "(none)":
        sections.extend(["", "Prior Rejections", prior_rejections])

    sections.extend([
        "",
        "Output Shape",
        """```json
{
  "states": ["s0", "s1", "s_goal"],
  "initial": "s0",
  "marked": ["s_goal"],
  "state_metadata": {
    "s1": {
      "atomic_bindings": {
        "condition_name_a": true,
        "condition_name_b": true,
        "condition_name_c": false
      }
    }
  },
  "marked_state_metadata": {
    "s_goal": {"marking_predicate": "recovery objectives and resume conditions are satisfied"}
  },
  "events": [
    {
      "name": "unique_event_label",
      "from": "s0",
      "to": "s1",
      "resource_jid": "RESOURCE_JID",
      "part_name": "OPTIONAL_PART_NAME",
      "location_ref": "OPTIONAL_ALLOWED_LOCATION_REF",
      "description": "state transition intent",
      "projected_effect": {
        "resource": {},
        "part": {}
      }
    }
  ]
}
```""",
        "",
        "Output Constraints",
        "- Treat the initial state as the grounded failed world and the marked state as the resume-ready recovery goal.",
        "- Use only listed resources, parts, and grounded location references.",
        "- Each event must be one physical transition by one resource.",
        "- Event names are provenance only. Do not rely on canonical verbs or action taxonomies.",
        "- Use projected_effect to state the after-state facts that this event makes true.",
        "- For events that alter resource or part state rows, projected_effect should contain the successor field values.",
        "- projected_effect may update fields already present in the current resource or part state rows.",
        "- atomic_bindings may contain zero or more boolean audit facts; use only true/false values.",
        "- Atomic bindings and nominal assignment fields are provenance only and do not control event semantics.",
        "- Event feasibility must come from grounded state, reachability, and safety facts, not from label wording.",
        "- If location_ref and pose are both present, they must describe the same grounded anchor.",
        "- Encode required ordering through the state graph.",
        "- Initial state must match the current state facts.",
        "- Marked states must satisfy recovery objectives and resume conditions.",
        "- Event names must be unique.",
        "- The plant must be deterministic.",
    ])

    return "\n".join(sections).strip() + "\n"


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
    persistent_constraint_summary: list[str] | None = None,
    revision_history_summary_text: str = "",
) -> str:
    """Build a revision prompt when the previous plant was invalid or unsolvable."""
    del previous_plant_json
    sections: list[str] = [
        "Task and Role",
        (
            "You are revising a recovery plant for a DES fallback recovery session.\n"
            "Current phase: Recovery Plant Synthesis.\n"
            "Generate a complete replacement plant. Use the rejection data as "
            "validation evidence; do not generate controller primitive steps here."
        ),
        "",
        "Prior Rejections",
        _prior_rejections_text(
            plant_findings_text=plant_findings_text,
            solver_diagnostic_text=solver_diagnostic_text,
            feasibility_findings_text=feasibility_findings_text,
            persistent_constraint_summary=persistent_constraint_summary,
            revision_history_summary_text=revision_history_summary_text,
        ),
    ]

    base_prompt = build_hybrid_domain_generation_prompt(prompt_input)
    marker = "Current Resource State"
    idx = base_prompt.find(marker)
    if idx >= 0:
        sections.extend(["", base_prompt[idx:].strip()])

    return "\n".join(sections).strip() + "\n"


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
    ap_descriptors: list[dict[str, Any]] | None = None,
    persistent_constraint_summary: list[str] | None = None,
    revision_history_summary_text: str = "",
    domain_vocabulary_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the structured prompt input payload for a given phase.

    ``domain_vocabulary_overrides`` lets callers swap predicate names,
    effect_kind names, target-ref tokens, or marking templates without
    editing prompt code.
    """
    vocab = build_domain_vocabulary(overrides=domain_vocabulary_overrides)
    return {
        "phase": phase,
        "llm_input": deepcopy(llm_input),
        "session_state": deepcopy(session_state),
        "bridge_resources": deepcopy(bridge_resources or {}),
        "recovery_gap_state": deepcopy(recovery_gap_state or {}),
        "current_recovery_blockers": deepcopy(current_recovery_blockers or []),
        "ap_descriptors": deepcopy(ap_descriptors or []),
        "persistent_constraint_summary": deepcopy(persistent_constraint_summary or []),
        "revision_history_summary_text": str(revision_history_summary_text or ""),
        "domain_vocabulary": vocab,
    }


__all__ = [
    "build_domain_vocabulary",
    "build_hybrid_des_prompt_input",
    "build_hybrid_domain_generation_prompt",
    "build_hybrid_feedback_prompt",
    "hybrid_des_phase_response_schema",
]
