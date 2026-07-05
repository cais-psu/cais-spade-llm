"""Post-outline recovery safety generation helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

_FIXED_STATE_FIELDS = (
    "resource_state",
    "resource_location",
    "held_part",
    "part_state",
    "part_location",
)


def _utc_now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")
    return str(path)


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _fixed_state_surface_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {field: {"type": "string"} for field in _FIXED_STATE_FIELDS},
        "required": list(_FIXED_STATE_FIELDS),
    }


def _outline_event_echo_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "outline_id": {"type": "string"},
            "llm_outline_id": {"type": "string"},
            "event_name": {"type": "string"},
            "resource_jid": {"type": "string"},
            "part_name": {"type": "string"},
            "expected_start_state": _fixed_state_surface_schema(),
            "expected_end_state": _fixed_state_surface_schema(),
            "projected_outline_state": _fixed_state_surface_schema(),
        },
        "required": [
            "outline_id",
            "llm_outline_id",
            "event_name",
            "resource_jid",
            "part_name",
            "expected_start_state",
            "expected_end_state",
            "projected_outline_state",
        ],
    }


def _pending_nominal_task_echo_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "function": {"type": "string"},
            "resource": {"type": "string"},
            "part": {"type": "string"},
            "blocked_by_condition_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "expected_start_state": _fixed_state_surface_schema(),
            "expected_end_state": _fixed_state_surface_schema(),
            "projected_outline_state": _fixed_state_surface_schema(),
            "status": {"type": "string"},
        },
        "required": [
            "id",
            "function",
            "resource",
            "part",
            "blocked_by_condition_ids",
            "expected_start_state",
            "expected_end_state",
            "projected_outline_state",
            "status",
        ],
    }


def _grounding_response_schema() -> dict[str, Any]:
    return {
        "name": "recovery_safety_grounding",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "rule_id": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["selected", "not_involved"],
                            },
                            "reason": {"type": "string"},
                            "selected_recovery_outline_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "rule_id",
                            "status",
                            "reason",
                            "selected_recovery_outline_ids",
                        ],
                    },
                }
            },
            "required": ["rules"],
        },
        "strict": True,
    }


def _summarize_rule(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(rule.get("id") or rule.get("rule_id") or "").strip(),
        "raw_text": str(rule.get("raw_text") or "").strip(),
        "constraint_type": str(rule.get("constraint_type") or "").strip(),
        "process": str(rule.get("process") or "").strip(),
        "product": deepcopy(rule.get("product") or []),
        "resources": deepcopy(rule.get("resources") or []),
        "event": str(rule.get("event") or "").strip(),
        "context": deepcopy(rule.get("context") or {}),
    }


def _compact_state_surface(value: Any) -> dict[str, Any]:
    state = dict(value or {})
    result: dict[str, Any] = {}
    for field in _FIXED_STATE_FIELDS:
        if field not in state:
            continue
        field_value = state.get(field)
        if field_value is None:
            continue
        token = str(field_value).strip()
        if not token:
            continue
        result[field] = token
    return result


def _complete_state_surface(value: Any) -> dict[str, str]:
    state = dict(value or {})
    return {field: str(state.get(field) or "").strip() for field in _FIXED_STATE_FIELDS}


def _brief_outline_event(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "outline_id": str(row.get("outline_id") or "").strip(),
        "resource_jid": str(row.get("resource_jid") or "").strip(),
        "part_name": str(row.get("part_name") or "").strip(),
        "event_name": str(row.get("event_name") or "").strip(),
        "state_tokens": sorted(_accepted_state_tokens(row)),
        "target_ref": str(row.get("target_ref") or "").strip(),
    }


def _brief_pending_nominal_task(row: dict[str, Any]) -> dict[str, Any]:
    tokens = set()
    for state in (
        row.get("expected_start_state"),
        row.get("expected_end_state"),
        row.get("projected_outline_state"),
    ):
        tokens |= {
            f"{field}={value}"
            for field, value in _compact_state_surface(state or {}).items()
            if str(field).strip() and str(value).strip()
        }
    destination_location = str(row.get("destination_location") or "").strip()
    return {
        "id": str(row.get("id") or row.get("task_id") or "").strip(),
        "function": str(row.get("function") or row.get("function_name") or "").strip(),
        "resource": str(row.get("resource") or row.get("resource_jid") or "").strip(),
        "part": str(row.get("part") or row.get("part_name") or "").strip(),
        "status": str(row.get("status") or "").strip(),
        "destination_location": destination_location,
        "state_tokens": sorted(tokens),
    }


def build_recovery_safety_grounding_prompt(payload: dict[str, Any]) -> str:
    rules = [
        _summarize_rule(rule)
        for rule in (payload.get("loaded_safety_rules") or [])
        if isinstance(rule, dict)
    ]
    accepted_outline_prefix = [
        _brief_outline_event(row)
        for row in (payload.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    prompt_payload = {
        "accepted_outline_prefix": accepted_outline_prefix,
        "loaded_safety_rules": rules,
    }
    return (
        "You are selecting which supplied recovery outline rows are relevant to each loaded safety rule.\n"
        "Use the supplied recovery outline row fields and loaded safety rule fields to decide relevance.\n"
        "Copy only supplied outline_id values into selected_recovery_outline_ids.\n"
        "Do not rewrite rule bindings.\n"
        "Do not adapt one resource to another resource.\n"
        "Do not adapt one event surface to another event surface.\n"
        "Do not select nominal task rows.\n"
        "For each rule, return rule_id, status, reason, and selected_recovery_outline_ids.\n"
        "status must be selected or not_involved.\n"
        "selected_recovery_outline_ids must come only from the supplied recovery outline rows.\n"
        "Return JSON only.\n\n"
        f"{json.dumps(prompt_payload, indent=2, ensure_ascii=False)}"
    )


def _normalize_grounded_events(items: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "outline_id": str(item.get("outline_id") or "").strip(),
                "llm_outline_id": str(item.get("llm_outline_id") or "").strip(),
                "event_name": str(item.get("event_name") or "").strip(),
                "resource_jid": str(item.get("resource_jid") or "").strip(),
                "part_name": str(item.get("part_name") or "").strip(),
            }
        )
    return result


def _normalize_grounded_nominal_events(items: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "id": str(item.get("id") or "").strip(),
                "function": str(item.get("function") or "").strip(),
                "resource": str(item.get("resource") or "").strip(),
                "part": str(item.get("part") or "").strip(),
                "blocked_by_condition_ids": [
                    str(token).strip()
                    for token in (item.get("blocked_by_condition_ids") or [])
                    if str(token).strip()
                ],
            }
        )
    return result


def _normalize_grounded_states(items: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").strip()
        value = str(item.get("value") or "").strip()
        if not field or not value:
            continue
        result.append(
            {
                "outline_id": str(item.get("outline_id") or "").strip(),
                "llm_outline_id": str(item.get("llm_outline_id") or "").strip(),
                "resource_jid": str(item.get("resource_jid") or "").strip(),
                "part_name": str(item.get("part_name") or "").strip(),
                "field": field,
                "value": value,
            }
        )
    return result


def _normalize_grounded_nominal_states(items: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").strip()
        value = str(item.get("value") or "").strip()
        if not field or not value:
            continue
        result.append(
            {
                "id": str(item.get("id") or "").strip(),
                "function": str(item.get("function") or "").strip(),
                "resource": str(item.get("resource") or "").strip(),
                "part": str(item.get("part") or "").strip(),
                "field": field,
                "value": value,
                "blocked_by_condition_ids": [
                    str(token).strip()
                    for token in (item.get("blocked_by_condition_ids") or [])
                    if str(token).strip()
                ],
            }
        )
    return result


def _normalize_generated_aps(items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        full = str(item.get("full") or "").strip()
        if kind not in {"ap_event", "ap_state"}:
            continue
        if not full or full in seen:
            continue
        if not full.startswith(("ap_event/", "ap_state/")):
            continue
        if len(full.split("/")) < 6:
            continue
        normalized_item = {
            "kind": kind,
            "full": full,
            "why": str(item.get("why") or "").strip(),
            "source": str(item.get("source") or "").strip(),
            "source_task_ids": [
                str(token).strip()
                for token in (item.get("source_task_ids") or [])
                if str(token).strip()
            ],
            "outline_id": str(item.get("outline_id") or "").strip(),
            "llm_outline_id": str(item.get("llm_outline_id") or "").strip(),
            "id": str(item.get("id") or "").strip(),
            "event_name": str(item.get("event_name") or "").strip(),
            "function": str(item.get("function") or "").strip(),
            "resource_jid": str(item.get("resource_jid") or "").strip(),
            "resource": str(item.get("resource") or "").strip(),
            "part_name": str(item.get("part_name") or "").strip(),
            "part": str(item.get("part") or "").strip(),
            "field": str(item.get("field") or "").strip(),
            "value": str(item.get("value") or "").strip(),
        }
        normalized.append(normalized_item)
        seen.add(full)
    return normalized


def _clone_rule_for_scope(rule: dict[str, Any]) -> dict[str, Any]:
    scoped = deepcopy(rule)
    scoped.pop("dfa_dot", None)
    scoped.pop("bridge_aps", None)
    scoped.pop("aps", None)
    scoped.pop("ltlf", None)
    return scoped


def _grounded_bindings_from_row(
    row: dict[str, Any],
) -> dict[str, Any]:
    recovery_events = _normalize_grounded_events(row.get("grounded_recovery_events") or [])
    nominal_events = _normalize_grounded_nominal_events(row.get("grounded_nominal_events") or [])
    recovery_states = _normalize_grounded_states(row.get("grounded_recovery_states") or [])
    nominal_states = _normalize_grounded_nominal_states(row.get("grounded_nominal_states") or [])
    return {
        "recovery_outline_ids": sorted(
            {
                str(item.get("outline_id") or "").strip()
                for item in recovery_events + recovery_states
                if str(item.get("outline_id") or "").strip()
            }
        ),
        "recovery_llm_outline_ids": sorted(
            {
                str(item.get("llm_outline_id") or "").strip()
                for item in recovery_events + recovery_states
                if str(item.get("llm_outline_id") or "").strip()
            }
        ),
        "recovery_resource_jids": sorted(
            {
                str(item.get("resource_jid") or "").strip()
                for item in recovery_events + recovery_states
                if str(item.get("resource_jid") or "").strip()
            }
        ),
        "recovery_event_names": sorted(
            {
                str(item.get("event_name") or "").strip()
                for item in recovery_events
                if str(item.get("event_name") or "").strip()
            }
        ),
        "recovery_part_names": sorted(
            {
                str(item.get("part_name") or "").strip()
                for item in recovery_events + recovery_states
                if str(item.get("part_name") or "").strip()
            }
        ),
        "recovery_state_tokens": sorted(
            {
                f"{str(item.get('field') or '').strip()}={str(item.get('value') or '').strip()}"
                for item in recovery_states
                if str(item.get("field") or "").strip() and str(item.get("value") or "").strip()
            }
        ),
        "nominal_task_ids": sorted(
            {
                str(item.get("id") or "").strip()
                for item in nominal_events + nominal_states
                if str(item.get("id") or "").strip()
            }
        ),
        "nominal_resources": sorted(
            {
                str(item.get("resource") or "").strip()
                for item in nominal_events + nominal_states
                if str(item.get("resource") or "").strip()
            }
        ),
        "nominal_functions": sorted(
            {
                str(item.get("function") or "").strip()
                for item in nominal_events + nominal_states
                if str(item.get("function") or "").strip()
            }
        ),
        "nominal_parts": sorted(
            {
                str(item.get("part") or "").strip()
                for item in nominal_events + nominal_states
                if str(item.get("part") or "").strip()
            }
        ),
        "nominal_state_tokens": sorted(
            {
                f"{str(item.get('field') or '').strip()}={str(item.get('value') or '').strip()}"
                for item in nominal_states
                if str(item.get("field") or "").strip() and str(item.get("value") or "").strip()
            }
        ),
    }


def _normalize_string_tokens(items: Any) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for item in items or []:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        normalized.append(token)
        seen.add(token)
    return normalized


def _normalize_resource_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    if "@" in token:
        token = token.split("@", 1)[0]
    return token


def _normalize_part_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_destination_token(value: Any) -> str:
    token = str(value or "").strip()
    if "@" in token:
        token = token.split("@", 1)[0]
    return token.lower()


def _rule_constraint_family(source_rule: dict[str, Any]) -> str:
    family = str(source_rule.get("constraint_type") or "").strip()
    if family in {"precedence", "mutex"}:
        return family
    ltlf = str(source_rule.get("ltlf") or "").strip()
    resources = [
        _normalize_resource_token(token)
        for token in (source_rule.get("resources") or [])
        if _normalize_resource_token(token)
    ]
    if len(resources) >= 2 and "G" in ltlf and "!" in ltlf and "&" in ltlf:
        return "mutex"
    if " U " in ltlf or ltlf.startswith("U ") or ltlf.endswith(" U"):
        return "precedence"
    return family


def _parse_state_token(token: Any) -> tuple[str, str]:
    text = str(token or "").strip()
    if "=" not in text:
        return "", ""
    field, value = text.split("=", 1)
    field = str(field or "").strip()
    value = str(value or "").strip()
    if field not in _FIXED_STATE_FIELDS or not value:
        return "", ""
    return field, value


def _row_state_tokens(row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for state in (
        row.get("expected_start_state"),
        row.get("expected_end_state"),
        row.get("projected_outline_state"),
    ):
        tokens |= {
            f"{field}={value}"
            for field, value in _compact_state_surface(state or {}).items()
            if str(field).strip() and str(value).strip()
        }
    return tokens


def _row_completion_state_tokens(row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for state in (
        row.get("expected_end_state"),
        row.get("projected_outline_state"),
    ):
        tokens |= {
            f"{field}={value}"
            for field, value in _compact_state_surface(state or {}).items()
            if str(field).strip() and str(value).strip()
        }
    return tokens


def _row_destination_tokens(row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("target_ref", "destination_location"):
        token = _normalize_destination_token(row.get(key))
        if token:
            tokens.add(token)
    for state in (
        row.get("expected_start_state"),
        row.get("expected_end_state"),
        row.get("projected_outline_state"),
    ):
        state_surface = _compact_state_surface(state or {})
        for field in ("resource_location", "part_location"):
            token = _normalize_destination_token(state_surface.get(field))
            if token:
                tokens.add(token)
    return tokens


def _row_function(row: dict[str, Any]) -> str:
    return str(row.get("function") or row.get("function_name") or "").strip()


def _row_resource(row: dict[str, Any]) -> str:
    return str(row.get("resource") or row.get("resource_jid") or "").strip()


def _row_part(row: dict[str, Any]) -> str:
    return str(row.get("part") or row.get("part_name") or "").strip()


def _tool_rows_for_nominal_task(
    row: dict[str, Any],
    tools_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    function = _row_function(row)
    resource = _normalize_resource_token(_row_resource(row))
    if not function:
        return []
    matches: list[dict[str, Any]] = []
    for tool in tools_catalog:
        if not isinstance(tool, dict):
            continue
        if str(tool.get("function") or "").strip() != function:
            continue
        owner = _normalize_resource_token(tool.get("function_owner_agent"))
        if owner and resource and owner != resource:
            continue
        matches.append(dict(tool))
    return matches


def _nominal_state_rows_from_task_projection(
    selected_task_ids: list[str],
    nominal_by_id: dict[str, dict[str, Any]],
    tools_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for task_id in selected_task_ids:
        nominal_row = dict(nominal_by_id.get(task_id) or {})
        if not nominal_row:
            continue
        state_values: list[str] = []
        for source in (
            nominal_row,
            *_tool_rows_for_nominal_task(nominal_row, tools_catalog),
        ):
            out_state = str(source.get("out_state") or "").strip()
            if not out_state or out_state.lower() == "any":
                continue
            state_values.append(out_state)
        for value in state_values:
            key = (task_id, "resource_state", value)
            if key in seen:
                continue
            rows.append(
                {
                    "id": str(nominal_row.get("id") or "").strip(),
                    "function": _row_function(nominal_row),
                    "resource": _row_resource(nominal_row),
                    "part": _row_part(nominal_row),
                    "field": "resource_state",
                    "value": value,
                    "blocked_by_condition_ids": [
                        str(token_value).strip()
                        for token_value in (nominal_row.get("blocked_by_condition_ids") or [])
                        if str(token_value).strip()
                    ],
                }
            )
            seen.add(key)
    return rows


def _enrich_nominal_rows_with_tool_metadata(
    rows: list[dict[str, Any]],
    tools_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        enriched = dict(row)
        for tool_row in _tool_rows_for_nominal_task(enriched, tools_catalog):
            for key in ("in_state", "out_state", "part_in_state"):
                if str(enriched.get(key) or "").strip():
                    continue
                token = str(tool_row.get(key) or "").strip()
                if token:
                    enriched[key] = token
            if isinstance(tool_row.get("context_mapping"), dict) and not isinstance(
                enriched.get("context_mapping"), dict
            ):
                enriched["context_mapping"] = deepcopy(tool_row.get("context_mapping"))
            if isinstance(tool_row.get("part_transition"), dict) and not isinstance(
                enriched.get("part_transition"), dict
            ):
                enriched["part_transition"] = deepcopy(tool_row.get("part_transition"))
            break
        enriched_rows.append(enriched)
    return enriched_rows


def _normalize_selection_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": str(row.get("rule_id") or "").strip(),
        "status": str(row.get("status") or "").strip().lower(),
        "reason": str(row.get("reason") or "").strip(),
        "selected_recovery_outline_ids": _normalize_string_tokens(
            row.get("selected_recovery_outline_ids") or []
        ),
    }


def _recovery_event_rows_for_ids(
    selected_outline_ids: list[str],
    accepted_by_outline_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for outline_id in selected_outline_ids:
        accepted_row = dict(accepted_by_outline_id.get(outline_id) or {})
        if not accepted_row:
            missing.append(outline_id)
            continue
        rows.append(
            {
                "outline_id": str(accepted_row.get("outline_id") or "").strip(),
                "llm_outline_id": str(accepted_row.get("llm_outline_id") or "").strip(),
                "event_name": str(accepted_row.get("event_name") or "").strip(),
                "resource_jid": str(accepted_row.get("resource_jid") or "").strip(),
                "part_name": str(accepted_row.get("part_name") or "").strip(),
            }
        )
    return rows, missing


def _nominal_event_rows_for_ids(
    selected_task_ids: list[str],
    nominal_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for task_id in selected_task_ids:
        nominal_row = dict(nominal_by_id.get(task_id) or {})
        if not nominal_row:
            missing.append(task_id)
            continue
        rows.append(
            {
                "id": str(nominal_row.get("id") or "").strip(),
                "function": str(
                    nominal_row.get("function") or nominal_row.get("function_name") or ""
                ).strip(),
                "resource": str(
                    nominal_row.get("resource") or nominal_row.get("resource_jid") or ""
                ).strip(),
                "part": str(nominal_row.get("part") or nominal_row.get("part_name") or "").strip(),
                "blocked_by_condition_ids": [
                    str(token).strip()
                    for token in (nominal_row.get("blocked_by_condition_ids") or [])
                    if str(token).strip()
                ],
            }
        )
    return rows, missing


def _selected_recovery_states(
    selected_outline_ids: list[str],
    selected_tokens: list[str],
    accepted_by_outline_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    matched_tokens: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for outline_id in selected_outline_ids:
        accepted_row = dict(accepted_by_outline_id.get(outline_id) or {})
        if not accepted_row:
            continue
        accepted_tokens = _row_completion_state_tokens(accepted_row)
        for token in selected_tokens:
            if token not in accepted_tokens:
                continue
            field, value = _parse_state_token(token)
            if not field or (outline_id, token) in seen:
                continue
            rows.append(
                {
                    "outline_id": str(accepted_row.get("outline_id") or "").strip(),
                    "llm_outline_id": str(accepted_row.get("llm_outline_id") or "").strip(),
                    "resource_jid": str(accepted_row.get("resource_jid") or "").strip(),
                    "part_name": str(accepted_row.get("part_name") or "").strip(),
                    "field": field,
                    "value": value,
                }
            )
            matched_tokens.add(token)
            seen.add((outline_id, token))
    unmatched = [token for token in selected_tokens if token not in matched_tokens]
    return rows, unmatched


def _selected_nominal_states(
    selected_task_ids: list[str],
    selected_tokens: list[str],
    nominal_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    matched_tokens: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for task_id in selected_task_ids:
        nominal_row = dict(nominal_by_id.get(task_id) or {})
        if not nominal_row:
            continue
        task_tokens = _row_state_tokens(nominal_row)
        for token in selected_tokens:
            if token not in task_tokens:
                continue
            field, value = _parse_state_token(token)
            if not field or (task_id, token) in seen:
                continue
            rows.append(
                {
                    "id": str(nominal_row.get("id") or "").strip(),
                    "function": str(
                        nominal_row.get("function") or nominal_row.get("function_name") or ""
                    ).strip(),
                    "resource": str(
                        nominal_row.get("resource") or nominal_row.get("resource_jid") or ""
                    ).strip(),
                    "part": str(
                        nominal_row.get("part") or nominal_row.get("part_name") or ""
                    ).strip(),
                    "field": field,
                    "value": value,
                    "blocked_by_condition_ids": [
                        str(token_value).strip()
                        for token_value in (nominal_row.get("blocked_by_condition_ids") or [])
                        if str(token_value).strip()
                    ],
                }
            )
            matched_tokens.add(token)
            seen.add((task_id, token))
    unmatched = [token for token in selected_tokens if token not in matched_tokens]
    return rows, unmatched


def _rule_destination(rule: dict[str, Any]) -> str:
    return _normalize_destination_token(dict(rule.get("context") or {}).get("destination"))


def _recovery_row_matches_precedence_side(
    rule: dict[str, Any],
    row: dict[str, Any],
    *,
    destination_required: str,
) -> bool:
    resources = [
        _normalize_resource_token(token)
        for token in (rule.get("resources") or [])
        if _normalize_resource_token(token)
    ]
    products = [
        _normalize_part_token(token)
        for token in (rule.get("product") or [])
        if _normalize_part_token(token)
    ]
    if len(resources) < 1 or len(products) < 1:
        return False
    if _normalize_resource_token(row.get("resource_jid")) != resources[0]:
        return False
    if _normalize_part_token(row.get("part_name")) != products[0]:
        return False
    if destination_required and destination_required not in _row_destination_tokens(row):
        return False
    return True


def _nominal_row_matches_precedence_side(
    rule: dict[str, Any],
    row: dict[str, Any],
    *,
    destination_required: str,
) -> bool:
    resources = [
        _normalize_resource_token(token)
        for token in (rule.get("resources") or [])
        if _normalize_resource_token(token)
    ]
    products = [
        _normalize_part_token(token)
        for token in (rule.get("product") or [])
        if _normalize_part_token(token)
    ]
    if len(resources) < 2 or len(products) < 2:
        return False
    if _normalize_resource_token(row.get("resource")) != resources[1]:
        return False
    if _normalize_part_token(row.get("part")) != products[1]:
        return False
    expected_function = str(rule.get("event") or "").strip()
    if expected_function and str(row.get("function") or "").strip() != expected_function:
        return False
    if destination_required and destination_required not in _row_destination_tokens(row):
        return False
    return True


def _recovery_event_ap(rule: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    process = str(rule.get("process") or "assembly").strip().lower() or "assembly"
    product = _normalize_part_token(row.get("part_name")) or "any"
    resource = _normalize_resource_token(row.get("resource_jid")) or "any"
    outline_id = str(row.get("outline_id") or "").strip()
    return {
        "kind": "ap_event",
        "full": f"ap_event/{process}/{product}/{resource}/execute_recovery_macro/outline_id={outline_id}",
        "why": f"selected recovery outline {outline_id}",
        "source": "recovery",
        "source_task_ids": [outline_id] if outline_id else [],
        "outline_id": outline_id,
        "llm_outline_id": str(row.get("llm_outline_id") or "").strip(),
        "id": "",
        "event_name": "",
        "function": "execute_recovery_macro",
        "resource_jid": str(row.get("resource_jid") or "").strip(),
        "resource": "",
        "part_name": str(row.get("part_name") or "").strip(),
        "part": "",
        "field": "",
        "value": "",
    }


def _nominal_event_ap(rule: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    process = str(rule.get("process") or "assembly").strip().lower() or "assembly"
    product = _normalize_part_token(row.get("part")) or "any"
    resource = _normalize_resource_token(row.get("resource")) or "any"
    task_id = str(row.get("id") or "").strip()
    function = str(row.get("function") or "").strip()
    return {
        "kind": "ap_event",
        "full": f"ap_event/{process}/{product}/{resource}/{function}/task_id={task_id}",
        "why": f"selected nominal task {task_id}",
        "source": "nominal",
        "source_task_ids": [task_id] if task_id else [],
        "outline_id": "",
        "llm_outline_id": "",
        "id": task_id,
        "event_name": "",
        "function": function,
        "resource_jid": "",
        "resource": str(row.get("resource") or "").strip(),
        "part_name": "",
        "part": str(row.get("part") or "").strip(),
        "field": "",
        "value": "",
    }


def _context_token_from_pairs(pairs: list[tuple[str, str]]) -> str:
    tokens = [
        f"{str(key).strip()}={str(value).strip()}"
        for key, value in pairs
        if str(key).strip() and str(value).strip()
    ]
    return "&".join(tokens) if tokens else "any"


def _recovery_state_ap(
    rule: dict[str, Any],
    row: dict[str, Any],
    *,
    destination_required: str = "",
) -> dict[str, Any]:
    process = str(rule.get("process") or "assembly").strip().lower() or "assembly"
    product = _normalize_part_token(row.get("part_name")) or "any"
    resource = _normalize_resource_token(row.get("resource_jid")) or "any"
    outline_id = str(row.get("outline_id") or "").strip()
    token = f"{str(row.get('field') or '').strip()}={str(row.get('value') or '').strip()}"
    state_symbol = str(row.get("value") or "").strip() or token
    context = _context_token_from_pairs(
        [
            ("destination", destination_required),
            ("outline_id", outline_id),
        ]
    )
    return {
        "kind": "ap_state",
        "full": f"ap_state/{process}/{product}/{resource}/{state_symbol}/{context}",
        "why": f"selected recovery state token {token} for {outline_id}",
        "source": "recovery",
        "source_task_ids": [outline_id] if outline_id else [],
        "outline_id": outline_id,
        "llm_outline_id": str(row.get("llm_outline_id") or "").strip(),
        "id": "",
        "event_name": "",
        "function": "",
        "resource_jid": str(row.get("resource_jid") or "").strip(),
        "resource": "",
        "part_name": str(row.get("part_name") or "").strip(),
        "part": "",
        "field": str(row.get("field") or "").strip(),
        "value": str(row.get("value") or "").strip(),
    }


def _nominal_state_ap(
    rule: dict[str, Any],
    row: dict[str, Any],
    *,
    destination_required: str = "",
) -> dict[str, Any]:
    process = str(rule.get("process") or "assembly").strip().lower() or "assembly"
    product = _normalize_part_token(row.get("part")) or "any"
    resource = _normalize_resource_token(row.get("resource")) or "any"
    task_id = str(row.get("id") or "").strip()
    token = f"{str(row.get('field') or '').strip()}={str(row.get('value') or '').strip()}"
    state_symbol = str(row.get("value") or "").strip() or token
    context = _context_token_from_pairs(
        [
            ("destination", destination_required),
            ("task_id", task_id),
        ]
    )
    return {
        "kind": "ap_state",
        "full": f"ap_state/{process}/{product}/{resource}/{state_symbol}/{context}",
        "why": f"selected nominal state token {token} for {task_id}",
        "source": "nominal",
        "source_task_ids": [task_id] if task_id else [],
        "outline_id": "",
        "llm_outline_id": "",
        "id": task_id,
        "event_name": "",
        "function": str(row.get("function") or "").strip(),
        "resource_jid": "",
        "resource": str(row.get("resource") or "").strip(),
        "part_name": "",
        "part": str(row.get("part") or "").strip(),
        "field": str(row.get("field") or "").strip(),
        "value": str(row.get("value") or "").strip(),
    }


def _dedupe_ap_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        full = str(item.get("full") or "").strip()
        if not full or full in seen:
            continue
        out.append(item)
        seen.add(full)
    return out


def _rule_state_symbols(rule: dict[str, Any]) -> set[str]:
    symbols: set[str] = set()
    for source_key in ("aps", "bridge_aps"):
        for ap in rule.get(source_key) or []:
            if not isinstance(ap, dict):
                continue
            full = str(ap.get("full") or "").strip()
            parts = full.split("/")
            if len(parts) < 6 or parts[0] != "ap_state":
                continue
            symbol = str(parts[4] or "").strip().lower()
            if symbol:
                symbols.add(symbol)
            for pair in str(parts[5] or "").split("&"):
                key, sep, value = pair.partition("=")
                if sep and key.strip() == "symbol" and value.strip():
                    symbols.add(value.strip().lower())
    return symbols


def _state_rows_relevant_to_rule(
    *,
    state_rows: list[dict[str, Any]],
    source_rows_by_id: dict[str, dict[str, Any]],
    source_id_key: str,
    state_symbols: set[str],
    destination_required: str,
    require_exact_destination: bool = True,
) -> list[dict[str, Any]]:
    if not state_symbols:
        return []
    relevant: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in state_rows:
        source_id = str(row.get(source_id_key) or "").strip()
        source_row = dict(source_rows_by_id.get(source_id) or {})
        if destination_required:
            destination_tokens = _row_destination_tokens(source_row)
            if require_exact_destination:
                if destination_required not in destination_tokens:
                    continue
            elif not destination_tokens:
                continue
        value = str(row.get("value") or "").strip().lower()
        if value not in state_symbols:
            continue
        key = (
            source_id,
            str(row.get("field") or "").strip(),
            str(row.get("value") or "").strip(),
        )
        if key in seen:
            continue
        relevant.append(row)
        seen.add(key)
    return relevant


def _recovery_state_rows_relevant_to_rule(
    *,
    state_rows: list[dict[str, Any]],
    accepted_by_outline_id: dict[str, dict[str, Any]],
    destination_required: str,
) -> list[dict[str, Any]]:
    relevant: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in state_rows:
        outline_id = str(row.get("outline_id") or "").strip()
        accepted_row = dict(accepted_by_outline_id.get(outline_id) or {})
        if not accepted_row:
            continue
        if destination_required and destination_required not in _row_destination_tokens(
            accepted_row
        ):
            continue
        token = f"{str(row.get('field') or '').strip()}={str(row.get('value') or '').strip()}"
        if token not in _row_completion_state_tokens(accepted_row):
            continue
        key = (
            outline_id,
            str(row.get("field") or "").strip(),
            str(row.get("value") or "").strip(),
        )
        if key in seen:
            continue
        relevant.append(row)
        seen.add(key)
    return relevant


def _state_tokens_for_recovery_outline_ids(
    selected_outline_ids: list[str],
    accepted_by_outline_id: dict[str, dict[str, Any]],
) -> list[str]:
    tokens: set[str] = set()
    for outline_id in selected_outline_ids:
        accepted_row = dict(accepted_by_outline_id.get(outline_id) or {})
        if not accepted_row:
            continue
        tokens |= _row_completion_state_tokens(accepted_row)
    return sorted(tokens)


def _state_tokens_for_nominal_task_ids(
    selected_task_ids: list[str],
    nominal_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    tokens: set[str] = set()
    for task_id in selected_task_ids:
        nominal_row = dict(nominal_by_id.get(task_id) or {})
        if not nominal_row:
            continue
        tokens |= _row_state_tokens(nominal_row)
    return sorted(tokens)


def _matched_recovery_events_for_family(
    *,
    source_rule: dict[str, Any],
    family: str,
    recovery_events: list[dict[str, Any]],
    accepted_by_outline_id: dict[str, dict[str, Any]],
    destination_required: str,
) -> list[dict[str, Any]]:
    if family == "precedence":
        return [
            row
            for row in recovery_events
            if _recovery_row_matches_precedence_side(
                source_rule,
                dict(accepted_by_outline_id.get(str(row.get("outline_id") or "").strip()) or {}),
                destination_required=destination_required,
            )
        ]

    if family == "mutex":
        rule_resources = {
            _normalize_resource_token(token)
            for token in (source_rule.get("resources") or [])
            if _normalize_resource_token(token)
        }
        return [
            row
            for row in recovery_events
            if _normalize_resource_token(row.get("resource_jid")) in rule_resources
            and (
                not destination_required
                or destination_required
                in _row_destination_tokens(
                    dict(accepted_by_outline_id.get(str(row.get("outline_id") or "").strip()) or {})
                )
            )
        ]

    return []


def _nominal_task_ids_for_recovery_selection(
    *,
    source_rule: dict[str, Any],
    family: str,
    recovery_events: list[dict[str, Any]],
    accepted_by_outline_id: dict[str, dict[str, Any]],
    nominal_by_id: dict[str, dict[str, Any]],
    destination_required: str,
) -> list[str]:
    matched_recovery_events = _matched_recovery_events_for_family(
        source_rule=source_rule,
        family=family,
        recovery_events=recovery_events,
        accepted_by_outline_id=accepted_by_outline_id,
        destination_required=destination_required,
    )
    if not matched_recovery_events:
        return []

    selected: list[str] = []
    seen: set[str] = set()

    if family == "precedence":
        for task_id, row in nominal_by_id.items():
            if not _nominal_row_matches_precedence_side(
                source_rule,
                row,
                destination_required=destination_required,
            ):
                continue
            if task_id and task_id not in seen:
                selected.append(task_id)
                seen.add(task_id)
        return selected

    if family == "mutex":
        rule_resources = {
            _normalize_resource_token(token)
            for token in (source_rule.get("resources") or [])
            if _normalize_resource_token(token)
        }
        recovery_resources = {
            _normalize_resource_token(row.get("resource_jid"))
            for row in matched_recovery_events
            if _normalize_resource_token(row.get("resource_jid"))
        }
        for task_id, row in nominal_by_id.items():
            row_resource = _normalize_resource_token(row.get("resource") or row.get("resource_jid"))
            if not row_resource or row_resource not in rule_resources:
                continue
            if row_resource in recovery_resources:
                continue
            if destination_required and not _row_destination_tokens(row):
                continue
            if task_id and task_id not in seen:
                selected.append(task_id)
                seen.add(task_id)
        return selected

    return selected


def _grounding_response_from_rule_results(
    all_rule_results: list[dict[str, Any]],
) -> dict[str, Any]:
    rules: list[dict[str, Any]] = []
    for row in all_rule_results:
        if not isinstance(row, dict):
            continue
        rules.append(
            {
                "rule_id": str(row.get("rule_id") or "").strip(),
                "status": str(row.get("llm_status") or "").strip(),
                "reason": str(row.get("llm_reason") or "").strip(),
                "selected_recovery_outline_ids": deepcopy(
                    row.get("selected_recovery_outline_ids") or []
                ),
                "selected_recovery_state_tokens": deepcopy(
                    row.get("selected_recovery_state_tokens") or []
                ),
                "selected_nominal_task_ids": deepcopy(row.get("selected_nominal_task_ids") or []),
                "selected_nominal_state_tokens": deepcopy(
                    row.get("selected_nominal_state_tokens") or []
                ),
                "deterministic_status": str(row.get("status") or "").strip(),
                "failure_reason": str(row.get("failure_reason") or "").strip(),
            }
        )
    return {"rules": rules}


def _deterministic_rule_result_from_selection(
    *,
    source_rule: dict[str, Any],
    selection_row: dict[str, Any],
    accepted_by_outline_id: dict[str, dict[str, Any]],
    nominal_by_id: dict[str, dict[str, Any]],
    tools_catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized = _normalize_selection_row(selection_row)
    selected_recovery_outline_ids = list(normalized["selected_recovery_outline_ids"])
    recovery_events, missing_recovery_ids = _recovery_event_rows_for_ids(
        selected_recovery_outline_ids,
        accepted_by_outline_id,
    )
    family = _rule_constraint_family(source_rule)
    destination_required = _rule_destination(source_rule)
    selected_nominal_task_ids = _nominal_task_ids_for_recovery_selection(
        source_rule=source_rule,
        family=family,
        recovery_events=recovery_events,
        accepted_by_outline_id=accepted_by_outline_id,
        nominal_by_id=nominal_by_id,
        destination_required=destination_required,
    )
    nominal_events, missing_nominal_ids = _nominal_event_rows_for_ids(
        selected_nominal_task_ids,
        nominal_by_id,
    )
    selected_recovery_state_tokens = _state_tokens_for_recovery_outline_ids(
        selected_recovery_outline_ids,
        accepted_by_outline_id,
    )
    explicit_nominal_state_tokens = _state_tokens_for_nominal_task_ids(
        selected_nominal_task_ids,
        nominal_by_id,
    )
    recovery_states, unmatched_recovery_tokens = _selected_recovery_states(
        selected_recovery_outline_ids,
        selected_recovery_state_tokens,
        accepted_by_outline_id,
    )
    nominal_states, unmatched_nominal_tokens = _selected_nominal_states(
        selected_nominal_task_ids,
        explicit_nominal_state_tokens,
        nominal_by_id,
    )
    projected_nominal_states = _nominal_state_rows_from_task_projection(
        selected_nominal_task_ids,
        nominal_by_id,
        tools_catalog,
    )
    selected_nominal_state_tokens = sorted(
        {
            *explicit_nominal_state_tokens,
            *[
                f"{str(row.get('field') or '').strip()}={str(row.get('value') or '').strip()}"
                for row in projected_nominal_states
                if str(row.get("field") or "").strip() and str(row.get("value") or "").strip()
            ],
        }
    )
    nominal_states = [*nominal_states, *projected_nominal_states]

    rule_result = {
        "rule_id": str(source_rule.get("id") or source_rule.get("rule_id") or "").strip(),
        "llm_status": normalized["status"],
        "llm_reason": normalized["reason"],
        "selected_recovery_outline_ids": selected_recovery_outline_ids,
        "selected_recovery_state_tokens": selected_recovery_state_tokens,
        "selected_nominal_task_ids": selected_nominal_task_ids,
        "selected_nominal_state_tokens": selected_nominal_state_tokens,
        "grounded_bindings": {},
        "grounded_recovery_events": [],
        "grounded_nominal_events": [],
        "grounded_recovery_states": [],
        "grounded_nominal_states": [],
        "recovery_side_aps": [],
        "nominal_side_aps": [],
        "aps": [],
        "ltlf": "",
        "status": "not_involved",
        "failure_reason": "",
    }

    if normalized["status"] == "not_involved":
        rule_result["failure_reason"] = normalized["reason"]
        return rule_result

    if (
        missing_recovery_ids
        or missing_nominal_ids
        or unmatched_recovery_tokens
        or unmatched_nominal_tokens
    ):
        details = []
        if missing_recovery_ids:
            details.append(f"unknown recovery outline ids: {missing_recovery_ids}")
        if missing_nominal_ids:
            details.append(f"unknown nominal task ids: {missing_nominal_ids}")
        if unmatched_recovery_tokens:
            details.append(f"unknown recovery state tokens: {unmatched_recovery_tokens}")
        if unmatched_nominal_tokens:
            details.append(f"unknown nominal state tokens: {unmatched_nominal_tokens}")
        rule_result["status"] = "ungroundable"
        rule_result["failure_reason"] = "; ".join(details)
        return rule_result

    if family == "precedence":
        matched_recovery_events = _matched_recovery_events_for_family(
            source_rule=source_rule,
            family=family,
            recovery_events=recovery_events,
            accepted_by_outline_id=accepted_by_outline_id,
            destination_required=destination_required,
        )
        matched_nominal_events = [
            row
            for row in nominal_events
            if _nominal_row_matches_precedence_side(
                source_rule,
                dict(nominal_by_id.get(str(row.get("id") or "").strip()) or {}),
                destination_required=destination_required,
            )
        ]
        if not matched_recovery_events or not matched_nominal_events:
            rule_result["status"] = "not_involved"
            rule_result["failure_reason"] = (
                "selected rows do not satisfy the loaded concrete precedence binding"
            )
            return rule_result
        primary_recovery_event = matched_recovery_events[0]
        primary_nominal_event = matched_nominal_events[0]
        grounded_recovery_states = [
            row
            for row in recovery_states
            if str(row.get("outline_id") or "").strip()
            == str(primary_recovery_event.get("outline_id") or "").strip()
        ]
        aps = _dedupe_ap_rows(
            [
                _recovery_event_ap(source_rule, primary_recovery_event),
                _nominal_event_ap(source_rule, primary_nominal_event),
            ]
        )
        rule_result["status"] = "grounded"
        rule_result["grounded_recovery_events"] = [primary_recovery_event]
        rule_result["grounded_nominal_events"] = [primary_nominal_event]
        rule_result["grounded_recovery_states"] = grounded_recovery_states
        rule_result["grounded_nominal_states"] = []
        rule_result["aps"] = aps
        rule_result["grounded_bindings"] = _grounded_bindings_from_row(rule_result)
        recovery_formula_aps = [
            str(row.get("full") or "").strip()
            for row in [_recovery_event_ap(source_rule, primary_recovery_event)]
            if str(row.get("full") or "").strip()
        ]
        nominal_formula_aps = [
            str(row.get("full") or "").strip()
            for row in [_nominal_event_ap(source_rule, primary_nominal_event)]
            if str(row.get("full") or "").strip()
        ]
        if recovery_formula_aps and nominal_formula_aps:
            rule_result["ltlf"] = (
                f"(!({' | '.join(nominal_formula_aps)}) U ({' | '.join(recovery_formula_aps)}))"
            )
        return rule_result

    if family == "mutex":
        rule_resources = {
            _normalize_resource_token(token)
            for token in (source_rule.get("resources") or [])
            if _normalize_resource_token(token)
        }
        matched_recovery_events = _matched_recovery_events_for_family(
            source_rule=source_rule,
            family=family,
            recovery_events=recovery_events,
            accepted_by_outline_id=accepted_by_outline_id,
            destination_required=destination_required,
        )
        matched_nominal_events = [
            row
            for row in nominal_events
            if _normalize_resource_token(row.get("resource")) in rule_resources
            and (
                not destination_required
                or bool(
                    _row_destination_tokens(
                        dict(nominal_by_id.get(str(row.get("id") or "").strip()) or {})
                    )
                )
            )
        ]
        if not matched_recovery_events or not matched_nominal_events:
            rule_result["status"] = "not_involved"
            rule_result["failure_reason"] = (
                "selected rows do not provide both sides of the loaded mutual exclusion binding"
            )
            return rule_result
        primary_recovery_event = matched_recovery_events[0]
        primary_nominal_event = next(
            (
                row
                for row in matched_nominal_events
                if _normalize_resource_token(row.get("resource"))
                != _normalize_resource_token(primary_recovery_event.get("resource_jid"))
            ),
            {},
        )
        if not primary_nominal_event:
            rule_result["status"] = "not_involved"
            rule_result["failure_reason"] = (
                "selected rows do not provide distinct resources for the loaded mutual exclusion binding"
            )
            return rule_result
        grounded_recovery_states = [
            row
            for row in recovery_states
            if str(row.get("outline_id") or "").strip()
            == str(primary_recovery_event.get("outline_id") or "").strip()
        ]
        nominal_side_event_ids = {
            str(row.get("id") or "").strip()
            for row in matched_nominal_events
            if _normalize_resource_token(row.get("resource"))
            != _normalize_resource_token(primary_recovery_event.get("resource_jid"))
        }
        grounded_nominal_states = [
            row
            for row in nominal_states
            if str(row.get("id") or "").strip() in nominal_side_event_ids
        ]
        state_symbols = _rule_state_symbols(source_rule)
        relevant_recovery_states = _recovery_state_rows_relevant_to_rule(
            state_rows=grounded_recovery_states,
            accepted_by_outline_id=accepted_by_outline_id,
            destination_required=destination_required,
        )
        relevant_nominal_states = _state_rows_relevant_to_rule(
            state_rows=grounded_nominal_states,
            source_rows_by_id=nominal_by_id,
            source_id_key="id",
            state_symbols=state_symbols,
            destination_required=destination_required,
            require_exact_destination=False,
        )
        recovery_side_aps = _dedupe_ap_rows(
            [
                _recovery_event_ap(source_rule, primary_recovery_event),
                *[
                    _recovery_state_ap(
                        source_rule,
                        row,
                        destination_required=destination_required,
                    )
                    for row in relevant_recovery_states
                ],
            ]
        )
        nominal_side_aps = _dedupe_ap_rows(
            [
                _nominal_event_ap(source_rule, primary_nominal_event),
                *[
                    _nominal_state_ap(
                        source_rule,
                        row,
                        destination_required=destination_required,
                    )
                    for row in relevant_nominal_states
                ],
            ]
        )
        aps: list[dict[str, Any]] = _dedupe_ap_rows(recovery_side_aps + nominal_side_aps)
        rule_result["status"] = "grounded"
        rule_result["grounded_recovery_events"] = [primary_recovery_event]
        rule_result["grounded_nominal_events"] = [primary_nominal_event]
        rule_result["grounded_recovery_states"] = relevant_recovery_states
        rule_result["grounded_nominal_states"] = relevant_nominal_states
        rule_result["recovery_side_aps"] = [
            str(row.get("full") or "").strip()
            for row in recovery_side_aps
            if str(row.get("full") or "").strip()
        ]
        rule_result["nominal_side_aps"] = [
            str(row.get("full") or "").strip()
            for row in nominal_side_aps
            if str(row.get("full") or "").strip()
        ]
        rule_result["aps"] = aps
        if rule_result["recovery_side_aps"] and rule_result["nominal_side_aps"]:
            rule_result["ltlf"] = (
                f"G !(({' | '.join(rule_result['recovery_side_aps'])}) "
                f"& ({' | '.join(rule_result['nominal_side_aps'])}))"
            )
        rule_result["grounded_bindings"] = _grounded_bindings_from_row(rule_result)
        return rule_result

    rule_result["status"] = "not_involved"
    rule_result["failure_reason"] = "unsupported deterministic recovery safety family"
    return rule_result


def _normalize_grounded_bindings(items: Any) -> dict[str, Any]:
    payload = dict(items or {})
    return {
        "recovery_outline_ids": _normalize_string_tokens(payload.get("recovery_outline_ids") or []),
        "recovery_llm_outline_ids": _normalize_string_tokens(
            payload.get("recovery_llm_outline_ids") or []
        ),
        "recovery_resource_jids": _normalize_string_tokens(
            payload.get("recovery_resource_jids") or []
        ),
        "recovery_event_names": _normalize_string_tokens(payload.get("recovery_event_names") or []),
        "recovery_part_names": _normalize_string_tokens(payload.get("recovery_part_names") or []),
        "recovery_state_tokens": _normalize_string_tokens(
            payload.get("recovery_state_tokens") or []
        ),
        "nominal_task_ids": _normalize_string_tokens(payload.get("nominal_task_ids") or []),
        "nominal_resources": _normalize_string_tokens(payload.get("nominal_resources") or []),
        "nominal_functions": _normalize_string_tokens(payload.get("nominal_functions") or []),
        "nominal_parts": _normalize_string_tokens(payload.get("nominal_parts") or []),
        "nominal_state_tokens": _normalize_string_tokens(payload.get("nominal_state_tokens") or []),
    }


def _accepted_state_tokens(row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for state in (
        row.get("expected_start_state"),
        row.get("expected_end_state"),
        row.get("projected_outline_state"),
    ):
        state_surface = _compact_state_surface(state or {})
        for field, value in state_surface.items():
            tokens.add(f"{field}={value}")
    return tokens


def _validate_grounded_rule_result(
    *,
    source_rule: dict[str, Any],
    rule_result: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    accepted_by_outline_id = {
        str(row.get("outline_id") or "").strip(): dict(row)
        for row in (payload.get("accepted_outline_prefix") or [])
        if isinstance(row, dict) and str(row.get("outline_id") or "").strip()
    }
    pending_by_id = {
        str(row.get("id") or row.get("task_id") or "").strip(): dict(row)
        for row in (payload.get("pending_nominal_tasks") or [])
        if isinstance(row, dict) and str(row.get("id") or row.get("task_id") or "").strip()
    }
    recovery_events = [
        dict(item)
        for item in (rule_result.get("grounded_recovery_events") or [])
        if isinstance(item, dict)
    ]
    nominal_events = [
        dict(item)
        for item in (rule_result.get("grounded_nominal_events") or [])
        if isinstance(item, dict)
    ]
    recovery_states = [
        dict(item)
        for item in (rule_result.get("grounded_recovery_states") or [])
        if isinstance(item, dict)
    ]
    nominal_states = [
        dict(item)
        for item in (rule_result.get("grounded_nominal_states") or [])
        if isinstance(item, dict)
    ]
    normalized_aps = [
        dict(item) for item in (rule_result.get("generated_aps") or []) if isinstance(item, dict)
    ]
    source_rule_ap_fulls = {
        str(item.get("full") or "").strip()
        for item in (source_rule.get("aps") or [])
        if isinstance(item, dict) and str(item.get("full") or "").strip()
    }

    for item in recovery_events:
        outline_id = str(item.get("outline_id") or "").strip()
        accepted_row = dict(accepted_by_outline_id.get(outline_id) or {})
        if not accepted_row:
            return (
                f"grounded recovery event references unknown outline_id {outline_id or '<missing>'}"
            )
        for key in ("llm_outline_id", "event_name", "resource_jid", "part_name"):
            token = str(item.get(key) or "").strip()
            accepted_token = str(accepted_row.get(key) or "").strip()
            if token != accepted_token:
                return (
                    f"grounded recovery event for {outline_id} does not preserve accepted {key}: "
                    f"{token or '<missing>'} != {accepted_token or '<missing>'}"
                )

    for item in recovery_states:
        outline_id = str(item.get("outline_id") or "").strip()
        accepted_row = dict(accepted_by_outline_id.get(outline_id) or {})
        if not accepted_row:
            return (
                f"grounded recovery state references unknown outline_id {outline_id or '<missing>'}"
            )
        token = f"{str(item.get('field') or '').strip()}={str(item.get('value') or '').strip()}"
        if token not in _row_completion_state_tokens(accepted_row):
            return f"grounded recovery state for {outline_id} does not preserve accepted completion token {token or '<missing>'}"

    for item in nominal_events:
        task_id = str(item.get("id") or "").strip()
        pending_row = dict(pending_by_id.get(task_id) or {})
        if not pending_row:
            return f"grounded nominal event references unknown task id {task_id or '<missing>'}"
        expected_values = {
            "function": str(
                pending_row.get("function") or pending_row.get("function_name") or ""
            ).strip(),
            "resource": str(
                pending_row.get("resource") or pending_row.get("resource_jid") or ""
            ).strip(),
            "part": str(pending_row.get("part") or pending_row.get("part_name") or "").strip(),
        }
        for key, expected_value in expected_values.items():
            token = str(item.get(key) or "").strip()
            if token != expected_value:
                return (
                    f"grounded nominal event for {task_id} does not preserve pending {key}: "
                    f"{token or '<missing>'} != {expected_value or '<missing>'}"
                )

    for item in nominal_states:
        task_id = str(item.get("id") or "").strip()
        pending_row = dict(pending_by_id.get(task_id) or {})
        if not pending_row:
            return f"grounded nominal state references unknown task id {task_id or '<missing>'}"
        token = f"{str(item.get('field') or '').strip()}={str(item.get('value') or '').strip()}"
        pending_tokens: set[str] = set()
        for state in (
            pending_row.get("expected_start_state"),
            pending_row.get("expected_end_state"),
            pending_row.get("projected_outline_state"),
        ):
            pending_surface = _compact_state_surface(state or {})
            for field, value in pending_surface.items():
                pending_tokens.add(f"{field}={value}")
        if pending_tokens and token not in pending_tokens:
            return f"grounded nominal state for {task_id} does not preserve pending token {token or '<missing>'}"

    for item in normalized_aps:
        full = str(item.get("full") or "").strip()
        if full in source_rule_ap_fulls:
            return f"generated AP copies source rule AP {full}"
        source = str(item.get("source") or "").strip()
        if source == "recovery":
            outline_id = str(item.get("outline_id") or "").strip()
            accepted_row = dict(accepted_by_outline_id.get(outline_id) or {})
            if not accepted_row:
                return f"generated recovery AP references unknown outline_id {outline_id or '<missing>'}"
            kind = str(item.get("kind") or "").strip()
            if kind == "ap_event" and (
                str(item.get("event_name") or "").strip()
                != str(accepted_row.get("event_name") or "").strip()
            ):
                return f"generated recovery event AP for {outline_id} does not preserve accepted event_name"
            if kind == "ap_state":
                token = (
                    f"{str(item.get('field') or '').strip()}={str(item.get('value') or '').strip()}"
                )
                if token not in _row_completion_state_tokens(accepted_row):
                    return f"generated recovery state AP for {outline_id} does not preserve accepted completion token {token or '<missing>'}"
        if source == "nominal":
            task_id = str(item.get("id") or "").strip()
            if task_id not in pending_by_id:
                return f"generated nominal AP references unknown task id {task_id or '<missing>'}"

    recovery_side_present = bool(recovery_events or recovery_states) or any(
        str(item.get("source") or "").strip() == "recovery" for item in normalized_aps
    )
    if not recovery_side_present:
        return "grounded rule omitted recovery-side binding"

    return ""


async def generate_recovery_safety_bundle(
    controller_agent: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    recovery_safety_dir_raw = str(
        payload.get("recovery_safety_dir")
        or payload.get("recovery_safery_dir")
        or payload.get("recovery_plan_dir")
        or ""
    ).strip()
    if not recovery_safety_dir_raw:
        raise RuntimeError("recovery safety generation requires recovery_safety_dir")
    recovery_safety_dir = Path(recovery_safety_dir_raw)
    recovery_plan_dir = recovery_safety_dir
    recovery_safery_dir = recovery_safety_dir

    recovery_safety_dir.mkdir(parents=True, exist_ok=True)
    tools_catalog = [
        deepcopy(row)
        for row in (
            payload.get("tools_catalog") or getattr(controller_agent, "tools_catalog", []) or []
        )
        if isinstance(row, dict)
    ]

    snapshot = {
        "accepted_outline_prefix": deepcopy(payload.get("accepted_outline_prefix") or []),
        "projected_outline_state": deepcopy(payload.get("projected_outline_state") or {}),
        "pending_nominal_tasks": deepcopy(payload.get("pending_nominal_tasks") or []),
        "pending_nominal_task_ids": deepcopy(payload.get("pending_nominal_task_ids") or []),
        "nominal_candidate_tasks": deepcopy(payload.get("nominal_candidate_tasks") or []),
        "nominal_candidate_task_ids": deepcopy(payload.get("nominal_candidate_task_ids") or []),
        "loaded_safety_rules": deepcopy(payload.get("loaded_safety_rules") or []),
        "bridge_safety_context": deepcopy(payload.get("bridge_safety_context") or {}),
        "tools_catalog": deepcopy(tools_catalog),
        "recovery_safety_scope_id": str(payload.get("recovery_safety_scope_id") or "").strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    snapshot_path = _write_json(
        recovery_safety_dir / "recovery_safety_input_snapshot.json",
        snapshot,
    )

    prompt = build_recovery_safety_grounding_prompt(payload)
    prompt_name = f"recovery_safety_grounding_prompt_{_utc_now_token()}.txt"
    prompt_path = _write_text(recovery_safety_dir / prompt_name, prompt)
    latest_prompt_path = _write_text(
        recovery_safety_dir / "recovery_safety_grounding_prompt_latest.txt",
        prompt,
    )

    grounding = await controller_agent.ask_llm_structured(
        prompt,
        response_format=_grounding_response_schema(),
    )
    llm_response_name = f"recovery_safety_grounding_llm_response_{_utc_now_token()}.json"
    llm_response_path = _write_json(
        recovery_safety_dir / llm_response_name,
        grounding,
    )
    latest_llm_response_path = _write_json(
        recovery_safety_dir / "recovery_safety_grounding_llm_response_latest.json",
        grounding,
    )
    response_name = f"recovery_safety_grounding_response_{_utc_now_token()}.json"
    response_path = str(recovery_safety_dir / response_name)
    latest_response_path = str(
        recovery_safety_dir / "recovery_safety_grounding_response_latest.json"
    )

    loaded_rules = [
        deepcopy(rule)
        for rule in (payload.get("loaded_safety_rules") or [])
        if isinstance(rule, dict)
    ]
    rules_by_id = {
        str(rule.get("id") or rule.get("rule_id") or "").strip(): rule
        for rule in loaded_rules
        if str(rule.get("id") or rule.get("rule_id") or "").strip()
    }
    selection_rows = [
        deepcopy(row) for row in (grounding.get("rules") or []) if isinstance(row, dict)
    ]
    accepted_by_outline_id = {
        str(row.get("outline_id") or "").strip(): dict(row)
        for row in (payload.get("accepted_outline_prefix") or [])
        if isinstance(row, dict) and str(row.get("outline_id") or "").strip()
    }
    nominal_candidate_tasks = [
        deepcopy(row)
        for row in (
            payload.get("nominal_candidate_tasks") or payload.get("pending_nominal_tasks") or []
        )
        if isinstance(row, dict)
    ]
    nominal_candidate_tasks = _enrich_nominal_rows_with_tool_metadata(
        nominal_candidate_tasks,
        tools_catalog,
    )
    nominal_by_id = {
        str(row.get("id") or row.get("task_id") or "").strip(): dict(row)
        for row in nominal_candidate_tasks
        if str(row.get("id") or row.get("task_id") or "").strip()
    }

    scoped_rules: list[dict[str, Any]] = []
    logic_raw: dict[str, dict[str, Any]] = {}
    ungroundable: list[dict[str, Any]] = []
    grounded_rule_ids: list[str] = []
    all_rule_results: list[dict[str, Any]] = []

    seen_rule_ids: set[str] = set()
    for row in selection_rows:
        rule_id = str(row.get("rule_id") or "").strip()
        if not rule_id or rule_id not in rules_by_id:
            continue
        source_rule = deepcopy(rules_by_id[rule_id])
        rule_result = _deterministic_rule_result_from_selection(
            source_rule=source_rule,
            selection_row=row,
            accepted_by_outline_id=accepted_by_outline_id,
            nominal_by_id=nominal_by_id,
            tools_catalog=tools_catalog,
        )
        seen_rule_ids.add(rule_id)
        if str(rule_result.get("status") or "").strip() == "not_involved":
            all_rule_results.append(rule_result)
            continue
        if str(rule_result.get("status") or "").strip() == "ungroundable":
            ungroundable.append(
                {
                    "rule_id": rule_id,
                    "failure_reason": str(
                        rule_result.get("failure_reason") or "rule could not be grounded"
                    ).strip(),
                }
            )
            all_rule_results.append(rule_result)
            continue
        aps = [
            str(item.get("full") or "").strip()
            for item in (rule_result.get("aps") or [])
            if str(item.get("full") or "").strip()
        ]
        if not aps:
            failure_reason = "deterministic grounding produced no APs"
            rule_result["status"] = "ungroundable"
            rule_result["failure_reason"] = failure_reason
            ungroundable.append(
                {
                    "rule_id": rule_id,
                    "failure_reason": failure_reason,
                }
            )
            all_rule_results.append(rule_result)
            continue
        deterministic_ltlf = str(rule_result.get("ltlf") or "").strip()
        if not deterministic_ltlf:
            failure_reason = (
                f"grounded rule {rule_id} did not provide ltlf; "
                "recovery safety generation does not synthesize temporal formulas"
            )
            ungroundable.append(
                {
                    "rule_id": rule_id,
                    "failure_reason": failure_reason,
                }
            )
            rule_result["failure_reason"] = failure_reason
            all_rule_results.append(rule_result)
            continue
        rule = _clone_rule_for_scope(source_rule)
        rule["source_rule_id"] = rule_id
        rule["source_rule"] = source_rule
        rule["selected_recovery_outline_ids"] = deepcopy(
            rule_result.get("selected_recovery_outline_ids") or []
        )
        rule["selected_recovery_state_tokens"] = deepcopy(
            rule_result.get("selected_recovery_state_tokens") or []
        )
        rule["selected_nominal_task_ids"] = deepcopy(
            rule_result.get("selected_nominal_task_ids") or []
        )
        rule["selected_nominal_state_tokens"] = deepcopy(
            rule_result.get("selected_nominal_state_tokens") or []
        )
        rule["grounded_bindings"] = deepcopy(rule_result.get("grounded_bindings") or {})
        rule["grounded_recovery_events"] = deepcopy(
            rule_result.get("grounded_recovery_events") or []
        )
        rule["grounded_nominal_events"] = deepcopy(rule_result.get("grounded_nominal_events") or [])
        rule["grounded_recovery_states"] = deepcopy(
            rule_result.get("grounded_recovery_states") or []
        )
        rule["grounded_nominal_states"] = deepcopy(rule_result.get("grounded_nominal_states") or [])
        rule["recovery_side_aps"] = deepcopy(rule_result.get("recovery_side_aps") or [])
        rule["nominal_side_aps"] = deepcopy(rule_result.get("nominal_side_aps") or [])

        scoped_rules.append(rule)
        grounded_rule_ids.append(rule_id)
        logic_raw[rule_id] = {
            "aps": aps,
            "ap_details": deepcopy(rule_result.get("aps") or []),
            "ltlf": deterministic_ltlf,
        }
        rule_result["ltlf"] = deterministic_ltlf
        all_rule_results.append(rule_result)

    for rule_id in rules_by_id:
        if rule_id in seen_rule_ids:
            continue
        all_rule_results.append(
            {
                "rule_id": rule_id,
                "llm_status": "not_involved",
                "llm_reason": "selector omitted this rule",
                "selected_recovery_outline_ids": [],
                "selected_recovery_state_tokens": [],
                "selected_nominal_task_ids": [],
                "selected_nominal_state_tokens": [],
                "grounded_bindings": {},
                "grounded_recovery_events": [],
                "grounded_nominal_events": [],
                "grounded_recovery_states": [],
                "grounded_nominal_states": [],
                "recovery_side_aps": [],
                "nominal_side_aps": [],
                "aps": [],
                "ltlf": "",
                "status": "not_involved",
                "failure_reason": "selector omitted this rule",
            }
        )

    grounding_response = _grounding_response_from_rule_results(all_rule_results)
    response_path = _write_json(recovery_safety_dir / response_name, grounding_response)
    latest_response_path = _write_json(
        recovery_safety_dir / "recovery_safety_grounding_response_latest.json",
        grounding_response,
    )

    if ungroundable:
        failure_payload = {
            "ok": False,
            "recovery_safety_scope_id": str(payload.get("recovery_safety_scope_id") or "").strip(),
            "recovery_safety_status": "failed",
            "recovery_safety_dir": str(recovery_safety_dir),
            "recovery_plan_dir": str(recovery_plan_dir),
            "recovery_safery_dir": str(recovery_safery_dir),
            "recovery_safety_logic_json": "",
            "dfa_dot_files": [],
            "rule_ids": grounded_rule_ids,
            "accepted_outline_prefix": deepcopy(payload.get("accepted_outline_prefix") or []),
            "pending_nominal_tasks": deepcopy(payload.get("pending_nominal_tasks") or []),
            "nominal_candidate_tasks": deepcopy(nominal_candidate_tasks),
            "all_rule_results": deepcopy(all_rule_results),
            "ungroundable_rules": ungroundable,
            "failure_reason": "one or more safety rules could not be grounded for the active recovery scope",
            "snapshot_artifact_path": snapshot_path,
            "grounding_prompt_artifact_path": prompt_path,
            "latest_grounding_prompt_artifact_path": latest_prompt_path,
            "grounding_llm_response_artifact_path": llm_response_path,
            "latest_grounding_llm_response_artifact_path": latest_llm_response_path,
            "grounding_response_artifact_path": response_path,
            "latest_grounding_response_artifact_path": latest_response_path,
        }
        _write_json(
            recovery_safety_dir / "recovery_safety_generation_result.json",
            failure_payload,
        )
        return failure_payload

    safety_logic = SafetyLogic(
        controller_agent,
        recovery_safety_dir / "recovery_safety_logic_source.txt",
    )
    safety_logic.rules = scoped_rules
    safety_logic.logic_raw = logic_raw
    safety_logic._apply_labels_into_rules()
    safety_logic.global_safety_spec = safety_logic._combine_safety_rules()
    safety_logic_json = recovery_safety_dir / "cca_safety_logic.json"
    safety_logic.save(safety_logic_json)
    dfa_map = safety_logic.build_dfas_per_rule(recovery_safety_dir)

    dfa_dot_files = [
        str((recovery_safety_dir / f"{rule_id}_dfa.dot").resolve())
        for rule_id in grounded_rule_ids
        if (recovery_safety_dir / f"{rule_id}_dfa.dot").exists()
    ]
    result = {
        "ok": True,
        "recovery_safety_scope_id": str(payload.get("recovery_safety_scope_id") or "").strip(),
        "recovery_safety_status": "ready",
        "recovery_safety_dir": str(recovery_safety_dir),
        "recovery_plan_dir": str(recovery_plan_dir),
        "recovery_safery_dir": str(recovery_safery_dir),
        "recovery_safety_logic_json": str(safety_logic_json.resolve()),
        "dfa_dot_files": dfa_dot_files,
        "rule_ids": grounded_rule_ids,
        "accepted_outline_prefix": deepcopy(payload.get("accepted_outline_prefix") or []),
        "pending_nominal_tasks": deepcopy(payload.get("pending_nominal_tasks") or []),
        "nominal_candidate_tasks": deepcopy(nominal_candidate_tasks),
        "all_rule_results": deepcopy(all_rule_results),
        "ungroundable_rules": [],
        "failure_reason": "",
        "snapshot_artifact_path": snapshot_path,
        "grounding_prompt_artifact_path": prompt_path,
        "latest_grounding_prompt_artifact_path": latest_prompt_path,
        "grounding_llm_response_artifact_path": llm_response_path,
        "latest_grounding_llm_response_artifact_path": latest_llm_response_path,
        "grounding_response_artifact_path": response_path,
        "latest_grounding_response_artifact_path": latest_response_path,
        "rules": deepcopy(safety_logic.rules),
        "rule_dfas": deepcopy(dfa_map),
    }
    _write_json(recovery_safety_dir / "recovery_safety_generation_result.json", result)
    return result
