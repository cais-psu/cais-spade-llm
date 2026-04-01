"""Multi-turn prompt builders and response schemas for the active v4 bridge."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.prompts.shared import (
    build_shared_fact_sections,
    json_block,
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
        "max_observations": int(state.get("max_observations") or 0),
        "max_observe_batch": int(state.get("max_observe_batch") or 0),
    }


def _grounding_contract() -> dict[str, Any]:
    return {
        "required_fields": [
            "thought",
            "decision",
            "blocking_summary",
            "sufficient_grounding",
            "observe_requests",
        ],
        "decision": ["observe", "grounded"],
        "observe_required_fields": ["observe_reason", "observe_requests"],
        "observe_request": {
            "required_fields": ["resource_jid", "primitive", "params", "store_as"],
        },
    }


def _outline_contract() -> dict[str, Any]:
    return {
        "required_fields": ["thought", "decision", "outline_tasks"],
        "decision": ["need_grounding", "outline_ready"],
        "outline_task": {
            "required_fields": [
                "outline_id",
                "resource_jid",
                "macro_name",
                "description",
                "rationale",
                "expected_start_state",
                "expected_end_state",
                "depends_on",
            ],
            "optional_fields": ["part_name"],
        },
    }


def _primitive_generation_contract() -> dict[str, Any]:
    return {
        "required_fields": ["thought", "decision", "macro_tasks"],
        "decision": ["need_grounding", "need_outline_revision", "draft_ready"],
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
            "need_grounding",
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
    observation_surface: dict[str, Any] | None = None,
    compact_resources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_phase = str(phase or "").strip().lower()
    return {
        "reasoning_mode": "multi_turn",
        "phase": normalized_phase,
        "llm_input": deepcopy(llm_input or {}),
        "session_state": deepcopy(session_state or {}),
        "observation_surface": deepcopy(observation_surface or {}),
        "compact_resources": deepcopy(compact_resources or []),
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
    phase_feedback = deepcopy(session_state.get("phase_feedback") or [])

    extra_sections: list[tuple[str, Any]] = [
        ("Session State", _compact_session_state(session_state)),
        ("Session Observation Store", observation_store),
    ]
    if phase_feedback:
        extra_sections.append(("Phase Feedback", phase_feedback))
    if phase == "grounding":
        extra_sections.append(
            ("Observation Surface", deepcopy(payload.get("observation_surface") or {}))
        )
    elif phase == "outline":
        extra_sections.append(
            ("Available Resources", deepcopy(payload.get("compact_resources") or []))
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
        *build_shared_fact_sections(
            llm_input,
            extra_sections=extra_sections,
            include_allowed_execution_surface=include_surface,
        ),
        "",
        "Required JSON Response Contract",
        json_block(response_contract),
        "",
        "Hard Constraints",
        "- Use only the listed resources and controller primitives.",
        "- Keep the response inside the current phase purpose and contract.",
        "- Make the recovery logically connected as a state progression: each state claim, outline task, and primitive step should follow from prior established state and establish the state needed by later work.",
        "- Do not invent observations, safety obligations, or grounded locations.",
        "- Do not contradict grounded runtime facts already present in the prompt.",
        "- Reuse grounded facts and previously derived outputs. If a step produces a reusable output, later steps should consume it. Do not invent concrete grounded values when grounded references or derived outputs are already available.",
    ]
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
                    "blocking_summary": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "sufficient_grounding": {"type": "boolean"},
                    "observe_reason": {"type": "string"},
                    "observe_requests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "resource_jid": {"type": "string"},
                                "primitive": {"type": "string"},
                                "params": {"type": "object"},
                                "store_as": {"type": "string"},
                            },
                            "required": ["resource_jid", "primitive", "params", "store_as"],
                        },
                    },
                },
                "required": [
                    "thought",
                    "decision",
                    "blocking_summary",
                    "sufficient_grounding",
                    "observe_requests",
                ],
            },
        }
    if normalized == "outline":
        return {
            "name": "multi_turn_outline_response",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": ["need_grounding", "outline_ready"],
                    },
                    "outline_tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "outline_id": {"type": "string"},
                                "resource_jid": {"type": "string"},
                                "macro_name": {"type": "string"},
                                "description": {"type": "string"},
                                "rationale": {"type": "string"},
                                "part_name": {"type": "string"},
                                "expected_start_state": {"type": "object"},
                                "expected_end_state": {"type": "object"},
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "outline_id",
                                "resource_jid",
                                "macro_name",
                                "description",
                                "rationale",
                                "expected_start_state",
                                "expected_end_state",
                                "depends_on",
                            ],
                        },
                    },
                },
                "required": ["thought", "decision", "outline_tasks"],
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
                            "need_grounding",
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
                            "need_grounding",
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
