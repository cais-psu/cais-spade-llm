"""Active v4 single-shot bridge prompt builders.

This module owns the pre-LLM prompt artifacts for the active bridge.  It is
intentionally small and local to the active ``llm_bridge`` package so the
prompt-building logic stays easy to read without depending on legacy bridge
prompt code.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


def _json_block(value: Any) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=True)


def _single_shot_response_contract() -> dict[str, Any]:
    return {
        "top_level_required_fields": [
            "primary_obligation",
            "macro_tasks",
        ],
        "primary_obligation": {
            "required_fields": [
                "rule_id",
                "resource_jid",
            ],
        },
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
            "optional_fields": [
                "part_name",
            ],
        },
        "task_metadata": {
            "required_fields": [
                "in_state",
                "out_state",
                "required_context_keys",
                "context_mapping",
                "part_transition",
            ],
        },
        "primitive_step": {
            "required_fields": [
                "primitive",
                "params",
            ],
            "optional_fields": [
                "store_as",
            ],
        },
    }


def build_single_shot_prompt_input(
    *,
    reasoning_mode: str,
    llm_input: dict[str, Any],
    proposal_success_criteria: dict[str, Any],
) -> dict[str, Any]:
    """Build the structured pre-LLM payload for one single-shot bridge turn."""
    return {
        "reasoning_mode": str(reasoning_mode or "single_shot").strip().lower() or "single_shot",
        "llm_input": deepcopy(llm_input or {}),
        "proposal_success_criteria": deepcopy(proposal_success_criteria or {}),
        "response_contract": _single_shot_response_contract(),
    }


def render_single_shot_prompt(prompt_input: dict[str, Any]) -> str:
    """Render the final single-shot prompt string right before LLM handoff."""
    payload = deepcopy(prompt_input or {})
    llm_input = dict(payload.get("llm_input") or {})

    safety_section = {
        "obligation_targets": deepcopy(llm_input.get("obligation_targets") or []),
        "loaded_safety_rules": deepcopy(llm_input.get("loaded_safety_rules") or []),
    }
    execution_surface = deepcopy(llm_input.get("allowed_execution_surface") or {})
    proposal_success_criteria = deepcopy(payload.get("proposal_success_criteria") or {})
    response_contract = deepcopy(payload.get("response_contract") or {})

    sections: list[str] = [
        "Task and Role",
        (
            "You are the active replanner for a DES fallback recovery turn.\n"
            "Use the structured context below to prepare one bridge recovery proposal.\n"
            "Your proposal must restore a state where nominal DES continuation can resume.\n"
            "Macro tasks may span multiple listed resources when coordination is needed."
        ),
        "",
        "Fault Event",
        _json_block(llm_input.get("fault_event") or {}),
        "",
        "Observed Runtime State",
        _json_block(llm_input.get("observed_runtime_state") or {}),
        "",
        "Loaded Safety Rules",
        _json_block(safety_section),
        "",
        "Relevant Assembly Requirements",
        _json_block(llm_input.get("relevant_assembly_requirements") or []),
        "",
        "Modeled Continuation Gap",
        _json_block(llm_input.get("modeled_continuation_gap") or {}),
        "",
        "Proposal Success Criteria",
        _json_block(proposal_success_criteria),
        "",
        "Allowed Execution Surface",
        _json_block(execution_surface),
        "",
        "Required JSON Response Contract",
        _json_block(response_contract),
        "",
        "Hard Constraints",
        "- Use only the listed resources and controller primitives.",
        "- Prefer the smallest coordinated recovery that restores the protected suffix.",
        "- Return exactly one JSON object and no prose.",
        "- Do not invent observations, safety obligations, or grounded locations.",
        "- When the context already gives a grounded reference, reuse that reference instead of inventing a new one.",
    ]
    return "\n".join(sections).strip() + "\n"


__all__ = [
    "build_single_shot_prompt_input",
    "render_single_shot_prompt",
]
