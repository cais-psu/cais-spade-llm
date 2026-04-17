"""Single-shot prompt builders for the active v4 bridge."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


def json_block(value: Any) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=True)


def prompt_part_facts(part_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered_rows: list[dict[str, Any]] = []
    for raw_row in (part_facts or []):
        if not isinstance(raw_row, dict):
            continue
        row = deepcopy(raw_row)
        current_state = str(row.get("current_state") or "").strip().lower()
        if current_state in {"", "unknown"}:
            row.pop("current_state", None)
        row.pop("current_location", None)
        rendered_rows.append(row)
    return rendered_rows


def build_shared_fact_sections(
    llm_input: dict[str, Any],
    *,
    extra_sections: list[tuple[str, Any]] | None = None,
    include_allowed_execution_surface: bool = False,
    allowed_execution_surface: dict[str, Any] | None = None,
) -> list[str]:
    payload = dict(llm_input or {})
    observed_runtime_state = dict(payload.get("observed_runtime_state") or {})
    safety_section = {
        "obligation_targets": deepcopy(payload.get("obligation_targets") or []),
        "loaded_safety_rules": deepcopy(payload.get("loaded_safety_rules") or []),
    }
    sections: list[str] = [
        "Fault Event",
        json_block(payload.get("fault_event") or {}),
        "",
        "Current Resource Facts",
        json_block(observed_runtime_state.get("resources") or []),
        "",
        "Current Part Facts",
        json_block(prompt_part_facts(payload.get("part_facts") or [])),
        "",
        "Loaded Safety Rules",
        json_block(safety_section),
        "",
        "Relevant Assembly Requirements",
        json_block(payload.get("relevant_assembly_requirements") or []),
        "",
        "Modeled Continuation Gap",
        json_block(payload.get("modeled_continuation_gap") or {}),
    ]
    for title, value in extra_sections or []:
        sections.extend(["", str(title or "").strip(), json_block(value)])
    if include_allowed_execution_surface:
        surface = deepcopy(
            allowed_execution_surface
            if allowed_execution_surface is not None
            else payload.get("allowed_execution_surface") or {}
        )
        sections.extend(["", "Allowed Execution Surface", json_block(surface)])
    return sections


def _single_shot_response_contract() -> dict[str, Any]:
    return {
        "top_level_required_fields": [
            "thought",
            "primary_obligation",
            "macro_tasks",
        ],
        "thought": {
            "summary_style": "2-4 factual sentences",
            "must_cover": [
                "current blocker",
                "chosen resource set",
                "why the proposal restores resumability",
            ],
        },
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
            "optional_fields": [],
        },
    }


def build_single_shot_prompt_input(
    *,
    reasoning_mode: str,
    llm_input: dict[str, Any],
) -> dict[str, Any]:
    return {
        "reasoning_mode": str(reasoning_mode or "single_shot").strip().lower() or "single_shot",
        "llm_input": deepcopy(llm_input or {}),
        "response_contract": _single_shot_response_contract(),
    }


def render_single_shot_prompt(prompt_input: dict[str, Any]) -> str:
    payload = deepcopy(prompt_input or {})
    llm_input = dict(payload.get("llm_input") or {})
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
        *build_shared_fact_sections(
            llm_input,
            include_allowed_execution_surface=True,
        ),
        "",
        "Required JSON Response Contract",
        json_block(response_contract),
        "",
        "Hard Constraints",
        "- Use only the listed resources and controller primitives.",
        "- Return a recovery that restores a state where the blocked nominal tasks can run again and leaves the focused failed resource in a resumable state.",
        "- Make the recovery logically connected as a state progression: each macro task and primitive step should be executable from the state established by prior steps and should establish the state needed by later steps.",
        '- Return exactly one JSON object. Begin with a "thought" field containing 2-4 factual sentences about the blocker, chosen resource(s), and why the proposal restores resumability.',
        "- Do not invent observations, safety obligations, or grounded locations.",
        "- Do not contradict grounded runtime facts already present in the prompt.",
        "- Reuse grounded facts and previously derived outputs. If a step produces a reusable output, later steps should consume it. Do not invent concrete grounded values when grounded references or derived outputs are already available.",
    ]
    return "\n".join(sections).strip() + "\n"


__all__ = ["build_single_shot_prompt_input", "render_single_shot_prompt"]
