"""Shared prompt rendering helpers for active v4 bridge modes."""

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


__all__ = ["build_shared_fact_sections", "json_block", "prompt_part_facts"]
