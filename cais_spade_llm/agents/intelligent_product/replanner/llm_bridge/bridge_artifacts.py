"""Bridge prompt artifact writer shared by dry-run and live runtime paths."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BRIDGE_RUNTIME_DATA_DIR = Path(
    "cais_spade_llm/agents/intelligent_product/replanner/llm_bridge/runtime_data"
)
DEFAULT_BRIDGE_DEBUG_DIR = DEFAULT_BRIDGE_RUNTIME_DATA_DIR
_RUNTIME_MULTI_TURN_FULL_TURN_WINDOW = 1


def _resolve_debug_dir(debug_dir: str | Path | None) -> Path:
    if debug_dir is None:
        return DEFAULT_BRIDGE_DEBUG_DIR
    candidate = Path(debug_dir)
    return candidate if str(candidate).strip() else DEFAULT_BRIDGE_DEBUG_DIR


def _multi_turn_phase_directory(
    target_dir: Path,
    *,
    phase: str,
) -> Path:
    phase_token = _artifact_token(phase, fallback="")
    if phase_token in {"grounding", "outline", "final_output"}:
        return target_dir / "recovery_outline"
    if phase_token == "primitive_generation":
        return target_dir / "recovery_primitves"
    return target_dir


def _artifact_token(value: Any, *, fallback: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return fallback
    token = "".join(
        ch if ch.isalnum() else "_"
        for ch in raw
    ).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or fallback


def _artifact_case_preserving_token(value: Any, *, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    token = "".join(
        ch if ch.isalnum() else "_"
        for ch in raw
    ).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or fallback


def _bridge_debug_payload(payload: dict[str, Any]) -> dict[str, Any]:
    bridge_debug = payload.get("bridge_debug")
    if isinstance(bridge_debug, dict):
        return bridge_debug
    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        return dict(prepared_bridge_request.get("bridge_debug") or {})
    return {}


def _artifact_task_action_summary(task: dict[str, Any]) -> str:
    description = str(task.get("description") or "").strip()
    if description:
        return description
    event_name = str(task.get("event_name") or task.get("name") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    location_ref = str(task.get("location_ref") or task.get("target_ref") or "").strip()
    if event_name:
        qualifiers = [item for item in (part_name, location_ref) if item]
        if qualifiers:
            return f"{event_name} ({' -> '.join(qualifiers)})"
        return event_name
    if part_name and location_ref:
        return f"{part_name} -> {location_ref}"
    return part_name or location_ref or "state transition"


def _artifact_outline_sequence_summary(tasks: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        outline_id = str(task.get("outline_id") or "").strip()
        resource_jid = str(task.get("resource_jid") or "").strip()
        part_name = str(task.get("part_name") or "").strip()
        label_parts = [item for item in (outline_id, resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "accepted task"
        lines.append(f"{label} -> {_artifact_task_action_summary(task)}")
    return lines


def _multi_turn_artifact_context(payload: dict[str, Any]) -> dict[str, Any]:
    bridge_debug = _bridge_debug_payload(payload)
    multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
    latest_turn = _latest_multi_turn_turn(payload)
    turn_index = int(
        latest_turn.get("turn_index")
        or multi_turn_session.get("turn_index")
        or 0
    )
    phase = _artifact_token(
        latest_turn.get("phase") or multi_turn_session.get("current_phase"),
        fallback="grounding",
    )
    session_id = _artifact_token(
        multi_turn_session.get("session_id"),
        fallback="session",
    )
    return {
        "turn_index": max(turn_index, 0),
        "phase": phase,
        "session_id": session_id,
    }


def _latest_multi_turn_turn(payload: dict[str, Any]) -> dict[str, Any]:
    current_turn = payload.get("multi_turn_current_turn")
    if isinstance(current_turn, dict) and current_turn:
        return dict(current_turn)

    bridge_debug = _bridge_debug_payload(payload)
    multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
    turns = list(multi_turn_session.get("turns") or [])
    if turns:
        return dict(turns[-1] or {})

    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            return dict(turns[-1] or {})
    return {}


def _extract_prompt_text(payload: dict[str, Any]) -> str:
    latest_turn = _latest_multi_turn_turn(payload)
    if str(latest_turn.get("prompt_text") or "").strip():
        return str(latest_turn.get("prompt_text") or "")

    bridge_debug = payload.get("bridge_debug")
    if isinstance(bridge_debug, dict):
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            if str(latest_turn.get("prompt_text") or "").strip():
                return str(latest_turn.get("prompt_text") or "")

    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            if str(latest_turn.get("prompt_text") or "").strip():
                return str(latest_turn.get("prompt_text") or "")
    return ""


def _extract_report_text(payload: dict[str, Any]) -> str:
    latest_turn = _latest_multi_turn_turn(payload)
    if str(latest_turn.get("report_text") or "").strip():
        return str(latest_turn.get("report_text") or "")

    bridge_debug = payload.get("bridge_debug")
    if isinstance(bridge_debug, dict):
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            if str(latest_turn.get("report_text") or "").strip():
                return str(latest_turn.get("report_text") or "")

    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            if str(latest_turn.get("report_text") or "").strip():
                return str(latest_turn.get("report_text") or "")
    return ""


def _extract_raw_response(payload: dict[str, Any]) -> str:
    raw_response = payload.get("raw_response")
    if raw_response not in (None, "", [], {}):
        if isinstance(raw_response, str):
            return raw_response
        return json.dumps(raw_response, indent=2, default=str, ensure_ascii=True)

    latest_turn = _latest_multi_turn_turn(payload)
    latest_response = latest_turn.get("raw_response")
    if latest_response not in (None, "", [], {}):
        latest_response = _compact_multi_turn_response_artifact(latest_response)
        if isinstance(latest_response, str):
            return latest_response
        return json.dumps(
            latest_response,
            indent=2,
            default=str,
            ensure_ascii=True,
        )

    bridge_debug = payload.get("bridge_debug")
    if isinstance(bridge_debug, dict):
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            latest_response = latest_turn.get("raw_response")
            if latest_response not in (None, "", [], {}):
                latest_response = _compact_multi_turn_response_artifact(latest_response)
                if isinstance(latest_response, str):
                    return latest_response
                return json.dumps(
                    latest_response,
                    indent=2,
                    default=str,
                    ensure_ascii=True,
                )

    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            latest_response = latest_turn.get("raw_response")
            if latest_response not in (None, "", [], {}):
                latest_response = _compact_multi_turn_response_artifact(latest_response)
                if isinstance(latest_response, str):
                    return latest_response
                return json.dumps(
                    latest_response,
                    indent=2,
                    default=str,
                    ensure_ascii=True,
                )
    return ""


def _extract_phase_result(payload: dict[str, Any]) -> str:
    latest_turn = _latest_multi_turn_turn(payload)
    result = latest_turn.get("phase_result")
    if result not in (None, "", [], {}):
        if isinstance(result, str):
            return result
        return json.dumps(result, indent=2, default=str, ensure_ascii=True)

    bridge_debug = payload.get("bridge_debug")
    if isinstance(bridge_debug, dict):
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            result = latest_turn.get("phase_result")
            if result not in (None, "", [], {}):
                if isinstance(result, str):
                    return result
                return json.dumps(result, indent=2, default=str, ensure_ascii=True)

    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            result = latest_turn.get("phase_result")
            if result not in (None, "", [], {}):
                if isinstance(result, str):
                    return result
                return json.dumps(result, indent=2, default=str, ensure_ascii=True)
    return ""


def _compact_artifact_finding(finding: Any) -> dict[str, Any]:
    if not isinstance(finding, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in (
        "constraint_code",
        "constraint_family",
        "constraint_owner",
        "resource_jid",
        "part_name",
        "task_id",
        "reason",
    ):
        value = finding.get(key)
        if value in (None, "", [], {}):
            continue
        compact[key] = deepcopy(value)
    return compact


def _compact_artifact_feedback_rows(rows: Any) -> list[dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if "constraint_code" in row and not (
            row.get("validation_findings") or row.get("findings")
        ):
            compact_finding = _compact_artifact_finding(row)
            if compact_finding:
                compact_rows.append(compact_finding)
            continue
        compact_row: dict[str, Any] = {
            "candidate_index": int(row.get("candidate_index") or 0),
        }
        task = row.get("task")
        if isinstance(task, dict):
            compact_row["task"] = deepcopy(task)
        findings = [
            compact_finding
            for compact_finding in (
                _compact_artifact_finding(item)
                for item in (row.get("validation_findings") or row.get("findings") or [])
            )
            if compact_finding
        ]
        if findings:
            compact_row["findings"] = findings
            compact_row["constraint_codes"] = sorted({
                str(item.get("constraint_code") or "")
                for item in findings
                if str(item.get("constraint_code") or "").strip()
            })
        compact_rows.append(compact_row)
    return compact_rows


def _compact_artifact_candidate_evaluations(rows: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        task = row.get("validated_task")
        if not isinstance(task, dict):
            task = row.get("task")
        if not isinstance(task, dict):
            task = row.get("surface_task")
        summary: dict[str, Any] = {
            "candidate_index": int(row.get("candidate_index") or 0),
            "valid": bool(row.get("valid")),
        }
        if row.get("pruned_match") is not None:
            summary["pruned_match"] = bool(row.get("pruned_match"))
        if isinstance(task, dict):
            summary["task"] = deepcopy(task)
        findings = [
            compact_finding
            for compact_finding in (
                _compact_artifact_finding(item)
                for item in (row.get("validation_findings") or [])
            )
            if compact_finding
        ]
        if findings:
            summary["findings"] = findings
            summary["constraint_codes"] = sorted({
                str(item.get("constraint_code") or "")
                for item in findings
                if str(item.get("constraint_code") or "").strip()
            })
        progress_detail = row.get("progress_detail")
        if isinstance(progress_detail, dict) and progress_detail:
            summary["progress_detail"] = deepcopy(progress_detail)
        summaries.append(summary)
    return summaries


def _compact_runtime_candidate_evaluations(rows: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        summary: dict[str, Any] = {}
        for key in ("candidate_index", "valid", "pruned_match"):
            value = row.get(key)
            if value in (None, "", [], {}):
                continue
            summary[key] = deepcopy(value)
        task = row.get("task")
        if not isinstance(task, dict):
            task = row.get("validated_task")
        if not isinstance(task, dict):
            task = row.get("surface_task")
        if isinstance(task, dict) and task:
            summary["task"] = deepcopy(task)
        findings = [
            deepcopy(item)
            for item in (row.get("validation_findings") or [])
            if isinstance(item, dict)
        ]
        if findings:
            summary["validation_findings"] = findings
        progress_detail = row.get("progress_detail")
        if isinstance(progress_detail, dict) and progress_detail:
            summary["progress_detail"] = deepcopy(progress_detail)
        summaries.append(summary)
    return summaries


def _compact_multi_turn_response_artifact(response: Any) -> Any:
    if not isinstance(response, dict):
        return response
    compact = deepcopy(response)
    if "transition_trace" in compact:
        compact.pop("des_event_sequence", None)
        compact.pop("accepted_transition_prefix", None)
    evaluations = compact.pop("candidate_evaluations", None)
    if isinstance(evaluations, list):
        compact["candidate_evaluation_summary"] = (
            _compact_artifact_candidate_evaluations(evaluations)
        )
    rejection_feedback = compact.get("candidate_rejection_feedback")
    if isinstance(rejection_feedback, list):
        compact["candidate_rejection_feedback"] = _compact_artifact_feedback_rows(
            rejection_feedback
        )
    transition_validation = compact.get("transition_validation")
    if isinstance(transition_validation, dict):
        compact_validation: dict[str, Any] = {}
        for key in ("status", "selected_candidate_index"):
            if transition_validation.get(key) not in (None, "", [], {}):
                compact_validation[key] = deepcopy(transition_validation.get(key))
        findings = transition_validation.get("findings")
        if isinstance(findings, list) and findings:
            compact_validation["findings"] = _compact_artifact_feedback_rows(findings)
        compact["transition_validation"] = compact_validation
    return compact


def _compact_multi_turn_session_transcript(session: dict[str, Any]) -> dict[str, Any]:
    compact_session = deepcopy(session)
    compact_turns: list[dict[str, Any]] = []
    for turn in compact_session.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        compact_turn = deepcopy(turn)
        compact_turn.pop("prompt_text", None)
        compact_turn.pop("prompt_input", None)
        if "transition_trace" in compact_turn:
            compact_turn.pop("des_event_sequence", None)
            compact_turn.pop("accepted_transition_prefix", None)
        evaluations = compact_turn.pop("candidate_evaluations", None)
        if isinstance(evaluations, list):
            compact_turn["candidate_evaluation_summary"] = (
                _compact_artifact_candidate_evaluations(evaluations)
            )
        rejection_feedback = compact_turn.get("candidate_rejection_feedback")
        if isinstance(rejection_feedback, list):
            compact_turn["candidate_rejection_feedback"] = (
                _compact_artifact_feedback_rows(rejection_feedback)
            )
        transition_validation = compact_turn.get("transition_validation")
        if isinstance(transition_validation, dict):
            compact_turn["transition_validation"] = (
                _compact_multi_turn_response_artifact({
                    "transition_validation": transition_validation,
                }).get("transition_validation", {})
            )
        raw_response = compact_turn.get("raw_response")
        if isinstance(raw_response, dict):
            compact_turn["raw_response"] = _compact_multi_turn_response_artifact(
                raw_response
            )
        primitive_substream_turns = compact_turn.get("primitive_substream_turns")
        if isinstance(primitive_substream_turns, list):
            compact_turn["primitive_substream_turns"] = [
                {
                    "outline_id": str(row.get("outline_id") or "").strip(),
                    "resource_jid": str(row.get("resource_jid") or "").strip(),
                    "primitive_local_turn_index": int(
                        row.get("primitive_local_turn_index") or 0
                    ),
                    "decision": str(row.get("decision") or "").strip(),
                }
                for row in primitive_substream_turns
                if isinstance(row, dict)
            ]
        compact_turns.append(compact_turn)
    compact_session["turns"] = compact_turns
    return compact_session


def _compact_multi_turn_runtime_turn(turn: dict[str, Any]) -> dict[str, Any]:
    compact_turn: dict[str, Any] = {
        "turn_index": int(turn.get("turn_index") or 0),
        "phase": str(turn.get("phase") or "").strip(),
        "decision": str(turn.get("decision") or "").strip(),
    }
    for key in (
        "status",
        "outline_id",
        "resource_jid",
        "primitive_local_turn_index",
        "final_output_stage",
        "prompt_artifact_path",
        "response_artifact_path",
    ):
        value = turn.get(key)
        if value in (None, "", [], {}):
            continue
        compact_turn[key] = deepcopy(value)

    phase = str(turn.get("phase") or "").strip().lower()
    if phase == "outline":
        next_transition = turn.get("next_transition")
        if isinstance(next_transition, dict) and next_transition:
            compact_turn["next_transition"] = deepcopy(next_transition)
            compact_turn["proposed_next_transition"] = deepcopy(next_transition)
        selected_transition = turn.get("selected_transition")
        if isinstance(selected_transition, dict) and selected_transition:
            compact_turn["selected_transition"] = deepcopy(selected_transition)
        transition_suffix = [
            deepcopy(item)
            for item in (turn.get("transition_suffix") or [])
            if isinstance(item, dict)
        ]
        if transition_suffix:
            compact_turn["transition_suffix"] = transition_suffix
        validation_findings = [
            deepcopy(item)
            for item in (turn.get("validation_findings") or [])
            if isinstance(item, dict)
        ]
        if validation_findings:
            compact_turn["validation_findings"] = validation_findings
        candidate_evaluations = _compact_runtime_candidate_evaluations(
            turn.get("candidate_evaluations") or []
        )
        if candidate_evaluations:
            compact_turn["candidate_evaluations"] = candidate_evaluations
    elif phase == "primitive_generation":
        primitive_substream_turns = turn.get("primitive_substream_turns")
        if isinstance(primitive_substream_turns, list):
            compact_turn["primitive_substream_turns"] = [
                {
                    "outline_id": str(row.get("outline_id") or "").strip(),
                    "resource_jid": str(row.get("resource_jid") or "").strip(),
                    "primitive_local_turn_index": int(
                        row.get("primitive_local_turn_index") or 0
                    ),
                    "decision": str(row.get("decision") or "").strip(),
                }
                for row in primitive_substream_turns
                if isinstance(row, dict)
            ]

    raw_response = turn.get("raw_response")
    if isinstance(raw_response, dict) and phase != "final_output":
        compact_turn["raw_response"] = _compact_multi_turn_response_artifact(
            raw_response
        )
    return compact_turn


def compact_multi_turn_runtime_session(
    session: dict[str, Any],
    *,
    keep_full_turns: int = _RUNTIME_MULTI_TURN_FULL_TURN_WINDOW,
) -> dict[str, Any]:
    if not isinstance(session, dict):
        return {}
    turns = [row for row in (session.get("turns") or []) if isinstance(row, dict)]
    if not turns:
        session["turns"] = []
        return session
    full_turn_window = max(0, int(keep_full_turns or 0))
    full_turn_start = max(len(turns) - full_turn_window, 0)
    compact_turns: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if index >= full_turn_start:
            compact_turns.append(deepcopy(turn))
            continue
        compact_turns.append(_compact_multi_turn_runtime_turn(turn))
    session["turns"] = compact_turns
    return session


def _extract_session_transcript(payload: dict[str, Any]) -> str:
    bridge_debug = payload.get("bridge_debug")
    if not isinstance(bridge_debug, dict):
        prepared_bridge_request = payload.get("prepared_bridge_request")
        if isinstance(prepared_bridge_request, dict):
            bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        else:
            bridge_debug = {}

    session = dict(bridge_debug.get("multi_turn_session") or {})
    if session and list(session.get("turns") or []):
        session = _compact_multi_turn_session_transcript(session)
        accepted_prefix = [
            deepcopy(row)
            for row in (session.get("accepted_outline_prefix") or [])
            if isinstance(row, dict)
        ]
        artifact_session = {
            "final_accepted_outline_summary": _artifact_outline_sequence_summary(accepted_prefix),
            **session,
        }
        return json.dumps(
            artifact_session,
            indent=2,
            default=str,
            ensure_ascii=True,
        )
    return ""


def _resolve_reasoning_mode(payload: dict[str, Any]) -> str:
    direct_mode = str(payload.get("reasoning_mode") or "").strip().lower()
    if direct_mode:
        return direct_mode
    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        session_mode = str(bridge_session.get("reasoning_mode") or "").strip().lower()
        if session_mode:
            return session_mode
    return "multi_turn"


def _should_write_resume_checkpoints(payload: dict[str, Any]) -> bool:
    direct_flag = payload.get("write_resume_checkpoints")
    if isinstance(direct_flag, bool):
        return direct_flag
    bridge_debug = _bridge_debug_payload(payload)
    bridge_flag = bridge_debug.get("write_resume_checkpoints")
    if isinstance(bridge_flag, bool):
        return bridge_flag
    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        bridge_flag = bridge_debug.get("write_resume_checkpoints")
        if isinstance(bridge_flag, bool):
            return bridge_flag
    return False


def _primitive_substream_prompt_text(turn: dict[str, Any]) -> str:
    system_instructions = str(turn.get("system_instructions") or "").strip()
    response_schema = turn.get("response_schema")
    prompt_text = str(turn.get("prompt_text") or "").strip()
    lines = [
        f"Outline ID: {str(turn.get('outline_id') or '').strip() or '-'}",
        f"Resource JID: {str(turn.get('resource_jid') or '').strip() or '-'}",
        (
            "Primitive Local Turn: "
            f"{int(turn.get('primitive_local_turn_index') or 0)}"
        ),
        "",
        "System Instructions",
        system_instructions or "_None_",
        "",
        "Response Schema",
        "```json",
        json.dumps(response_schema if isinstance(response_schema, dict) else {}, indent=2, ensure_ascii=True),
        "```",
        "",
        "Prompt",
        "```",
        prompt_text,
        "```",
        "",
    ]
    return "\n".join(lines)


def _primitive_substream_response_text(turn: dict[str, Any]) -> str:
    response_payload = turn.get("raw_response")
    if not isinstance(response_payload, dict):
        response_payload = deepcopy(turn.get("llm_raw_response") or {})
    if not isinstance(response_payload, dict):
        response_payload = {}
    decision = str(turn.get("decision") or "").strip()
    if decision and not str(response_payload.get("decision") or "").strip():
        response_payload["decision"] = decision
    response_artifact = {
        "outline_id": str(turn.get("outline_id") or "").strip(),
        "resource_jid": str(turn.get("resource_jid") or "").strip(),
        "primitive_local_turn_index": int(turn.get("primitive_local_turn_index") or 0),
        "decision": decision,
        "response": response_payload,
    }
    return json.dumps(response_artifact, indent=2, ensure_ascii=True, default=str)


def _extract_primitive_substream_turns(payload: dict[str, Any]) -> list[dict[str, Any]]:
    latest_turn = _latest_multi_turn_turn(payload)
    raw_turns = latest_turn.get("primitive_substream_turns")
    if not isinstance(raw_turns, list):
        return []
    return [
        deepcopy(turn)
        for turn in raw_turns
        if isinstance(turn, dict)
    ]


def _write_multi_turn_primitive_substream_artifacts(
    payload: dict[str, Any],
    *,
    target_dir: Path,
    timestamp: str,
) -> list[dict[str, str]]:
    artifact_rows: list[dict[str, str]] = []
    for turn in _extract_primitive_substream_turns(payload):
        outline_id = _artifact_case_preserving_token(
            turn.get("outline_id"),
            fallback="RECOVERY_SEQ",
        )
        local_turn_index = max(1, int(turn.get("primitive_local_turn_index") or 1))
        existing_prompt_path = str(turn.get("prompt_artifact_path") or "").strip()
        existing_response_path = str(turn.get("response_artifact_path") or "").strip()
        if existing_prompt_path and existing_response_path:
            prompt_path = Path(existing_prompt_path)
            response_path = Path(existing_response_path)
            if prompt_path.exists() and response_path.exists():
                artifact_rows.append({
                    "outline_id": str(turn.get("outline_id") or "").strip(),
                    "resource_jid": str(turn.get("resource_jid") or "").strip(),
                    "primitive_local_turn_index": str(local_turn_index),
                    "prompt_artifact_path": str(prompt_path),
                    "response_artifact_path": str(response_path),
                })
                continue
        prompt_path = target_dir / (
            f"multi_turn_{outline_id}_primitive_generation_"
            f"turn{local_turn_index:02d}_prompt_{timestamp}.txt"
        )
        response_path = target_dir / (
            f"multi_turn_{outline_id}_primitive_generation_"
            f"turn{local_turn_index:02d}_response_{timestamp}.txt"
        )
        prompt_path.write_text(
            _primitive_substream_prompt_text(turn),
            encoding="utf-8",
        )
        response_path.write_text(
            _primitive_substream_response_text(turn),
            encoding="utf-8",
        )
        artifact_rows.append({
            "outline_id": str(turn.get("outline_id") or "").strip(),
            "resource_jid": str(turn.get("resource_jid") or "").strip(),
            "primitive_local_turn_index": str(local_turn_index),
            "prompt_artifact_path": str(prompt_path),
            "response_artifact_path": str(response_path),
        })
    return artifact_rows


def _json_safe_resume_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_resume_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_resume_value(item) for item in value]
    if isinstance(value, set):
        return [
            _json_safe_resume_value(item)
            for item in sorted(value, key=lambda row: str(row))
        ]
    return str(value)


def _write_resume_checkpoint_artifact(
    *,
    checkpoint_payload: dict[str, Any] | None,
    target_dir: Path,
    artifact_name: str,
    latest_artifact_name: str,
) -> dict[str, str]:
    if not isinstance(checkpoint_payload, dict) or not checkpoint_payload:
        return {}
    serialized_payload = json.dumps(
        _json_safe_resume_value(checkpoint_payload),
        indent=2,
        ensure_ascii=True,
        sort_keys=True,
    )
    artifact_path = target_dir / artifact_name
    artifact_path.write_text(serialized_payload, encoding="utf-8")
    latest_artifact_path = target_dir / latest_artifact_name
    latest_artifact_path.write_text(serialized_payload, encoding="utf-8")
    return {
        "resume_checkpoint_artifact_path": str(artifact_path),
        "latest_resume_checkpoint_artifact_path": str(latest_artifact_path),
    }


def _multi_turn_resume_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    prepared_bridge_request = payload.get("prepared_bridge_request")
    if not isinstance(prepared_bridge_request, dict):
        return None
    session_state = payload.get("multi_turn_session_result")
    if not isinstance(session_state, dict):
        session_state = prepared_bridge_request.get("multi_turn_session_state")
    if not isinstance(session_state, dict) or not session_state:
        return None
    current_turn = payload.get("multi_turn_current_turn")
    if not isinstance(current_turn, dict):
        current_turn = {}
    return {
        "kind": "multi_turn_resume_checkpoint",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_bridge_request": deepcopy(prepared_bridge_request),
        "session_state": deepcopy(session_state),
        "current_turn": deepcopy(current_turn),
    }


def _primitive_batch_resume_checkpoint_payload(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    checkpoint_payload = payload.get("primitive_batch_resume_checkpoint")
    if not isinstance(checkpoint_payload, dict):
        return None
    if not isinstance(checkpoint_payload.get("prepared_bridge_request"), dict):
        return None
    if not isinstance(checkpoint_payload.get("assigned_outline_events"), list):
        return None
    if not isinstance(checkpoint_payload.get("session_state"), dict):
        return None
    normalized_checkpoint = {
        "kind": "primitive_batch_resume_checkpoint",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_bridge_request": deepcopy(
            checkpoint_payload.get("prepared_bridge_request") or {}
        ),
        "assigned_outline_events": [
            deepcopy(row)
            for row in (checkpoint_payload.get("assigned_outline_events") or [])
            if isinstance(row, dict)
        ],
        "session_state": deepcopy(checkpoint_payload.get("session_state") or {}),
        "bridge_session_id": str(checkpoint_payload.get("bridge_session_id") or "").strip(),
        "resource_jid": str(checkpoint_payload.get("resource_jid") or "").strip(),
    }
    if "final_output_turn_base" in checkpoint_payload:
        normalized_checkpoint["final_output_turn_base"] = int(
            checkpoint_payload.get("final_output_turn_base") or 0
        )
    current_turn = checkpoint_payload.get("current_turn")
    if isinstance(current_turn, dict) and current_turn:
        normalized_checkpoint["current_turn"] = deepcopy(current_turn)
    return normalized_checkpoint


def write_bridge_artifacts(
    payload: dict[str, Any],
    *,
    phase_label: str,
    debug_dir: str | Path | None = None,
    write_latest: bool = False,
    filename_prefix: str | None = None,
    write_session_transcript: bool | None = None,
    write_phase_prompt_response: bool = True,
) -> dict[str, str]:
    """Write bridge prompt/response artifacts."""
    del phase_label, filename_prefix
    normalized_payload = payload if isinstance(payload, dict) else {"payload": payload}
    base_target_dir = _resolve_debug_dir(debug_dir)
    base_target_dir.mkdir(parents=True, exist_ok=True)

    reasoning_mode = _resolve_reasoning_mode(normalized_payload)
    if write_session_transcript is None:
        write_session_transcript = reasoning_mode != "multi_turn"
    write_latest = bool(write_latest) and reasoning_mode != "multi_turn"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if reasoning_mode == "multi_turn":
        multi_turn_ctx = _multi_turn_artifact_context(normalized_payload)
        target_dir = _multi_turn_phase_directory(
            base_target_dir,
            phase=str(multi_turn_ctx["phase"] or ""),
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        suppress_phase_prompt_response = multi_turn_ctx["phase"] == "primitive_generation"
        prompt_artifact_name = (
            f"multi_turn_turn{int(multi_turn_ctx['turn_index']):02d}_"
            f"{multi_turn_ctx['phase']}_prompt_{timestamp}.txt"
        )
        response_artifact_name = (
            f"multi_turn_turn{int(multi_turn_ctx['turn_index']):02d}_"
            f"{multi_turn_ctx['phase']}_response_{timestamp}.txt"
        )
        latest_prompt_artifact_name = (
            f"multi_turn_turn{int(multi_turn_ctx['turn_index']):02d}_"
            f"{multi_turn_ctx['phase']}_prompt_latest.txt"
        )
        latest_response_artifact_name = (
            f"multi_turn_turn{int(multi_turn_ctx['turn_index']):02d}_"
            f"{multi_turn_ctx['phase']}_response_latest.txt"
        )
        session_transcript_artifact_name = (
            f"multi_turn_session_{multi_turn_ctx['session_id']}_{timestamp}.txt"
        )
        latest_session_transcript_artifact_name = (
            f"multi_turn_session_{multi_turn_ctx['session_id']}_latest.txt"
        )
        resume_checkpoint_artifact_name = (
            f"multi_turn_turn{int(multi_turn_ctx['turn_index']):02d}_"
            f"{multi_turn_ctx['phase']}_resume_checkpoint_{timestamp}.json"
        )
        latest_resume_checkpoint_artifact_name = (
            "multi_turn_resume_checkpoint_latest.json"
        )
    else:
        target_dir = base_target_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        suppress_phase_prompt_response = False
        prompt_artifact_name = f"{reasoning_mode}_prompt_{timestamp}.txt"
        response_artifact_name = f"{reasoning_mode}_response_{timestamp}.txt"
        latest_prompt_artifact_name = f"{reasoning_mode}_prompt_latest.txt"
        latest_response_artifact_name = f"{reasoning_mode}_response_latest.txt"
        session_transcript_artifact_name = f"{reasoning_mode}_session_{timestamp}.txt"
        latest_session_transcript_artifact_name = f"{reasoning_mode}_session_latest.txt"
        resume_checkpoint_artifact_name = ""
        latest_resume_checkpoint_artifact_name = ""

    prompt_text = _extract_prompt_text(normalized_payload)
    report_text = _extract_report_text(normalized_payload)
    is_report_artifact = not str(prompt_text or "").strip() and bool(
        str(report_text or "").strip()
    )
    primary_text = report_text if is_report_artifact else prompt_text
    primary_artifact_name = (
        prompt_artifact_name.replace("_prompt_", "_report_")
        if is_report_artifact
        else prompt_artifact_name
    )
    latest_primary_artifact_name = (
        latest_prompt_artifact_name.replace("_prompt_", "_report_")
        if is_report_artifact
        else latest_prompt_artifact_name
    )

    artifact_paths: dict[str, str] = {}
    if reasoning_mode == "multi_turn" and _should_write_resume_checkpoints(normalized_payload):
        checkpoint_paths = _write_resume_checkpoint_artifact(
            checkpoint_payload=_multi_turn_resume_checkpoint_payload(normalized_payload),
            target_dir=target_dir,
            artifact_name=resume_checkpoint_artifact_name,
            latest_artifact_name=latest_resume_checkpoint_artifact_name,
        )
        artifact_paths.update(checkpoint_paths)
        primitive_batch_checkpoint = _primitive_batch_resume_checkpoint_payload(
            normalized_payload
        )
        if primitive_batch_checkpoint is not None:
            assigned_outline_events = list(
                primitive_batch_checkpoint.get("assigned_outline_events") or []
            )
            assigned_outline_id = ""
            if assigned_outline_events and isinstance(assigned_outline_events[0], dict):
                assigned_outline_id = str(
                    assigned_outline_events[0].get("outline_id") or ""
                ).strip()
            outline_id = _artifact_case_preserving_token(
                primitive_batch_checkpoint.get("current_turn", {}).get("outline_id")
                or assigned_outline_id,
                fallback="RECOVERY_SEQ",
            )
            local_turn_index = max(
                1,
                int(
                    primitive_batch_checkpoint.get("current_turn", {}).get(
                        "primitive_local_turn_index"
                    )
                    or primitive_batch_checkpoint.get("session_state", {}).get("turn_index")
                    or 1
                ),
            )
            primitive_checkpoint_paths = _write_resume_checkpoint_artifact(
                checkpoint_payload=primitive_batch_checkpoint,
                target_dir=target_dir,
                artifact_name=(
                    f"multi_turn_{outline_id}_primitive_generation_"
                    f"turn{local_turn_index:02d}_resume_checkpoint_{timestamp}.json"
                ),
                latest_artifact_name=(
                    f"multi_turn_{outline_id}_primitive_generation_"
                    "resume_checkpoint_latest.json"
                ),
            )
            if primitive_checkpoint_paths:
                artifact_paths["primitive_resume_checkpoint_artifact_path"] = (
                    primitive_checkpoint_paths["resume_checkpoint_artifact_path"]
                )
                artifact_paths["latest_primitive_resume_checkpoint_artifact_path"] = (
                    primitive_checkpoint_paths["latest_resume_checkpoint_artifact_path"]
                )
    if write_phase_prompt_response and suppress_phase_prompt_response and reasoning_mode == "multi_turn":
        primitive_substream_artifacts = _write_multi_turn_primitive_substream_artifacts(
            normalized_payload,
            target_dir=target_dir,
            timestamp=timestamp,
        )
        if primitive_substream_artifacts:
            artifact_paths["primitive_substream_artifact_paths"] = json.dumps(
                primitive_substream_artifacts,
                ensure_ascii=True,
            )
    if write_phase_prompt_response and not suppress_phase_prompt_response and str(primary_text or "").strip():
        primary_artifact_path = target_dir / primary_artifact_name
        primary_artifact_path.write_text(
            primary_text,
            encoding="utf-8",
        )
        if is_report_artifact:
            artifact_paths["report_artifact_path"] = str(primary_artifact_path)
        else:
            artifact_paths["prompt_artifact_path"] = str(primary_artifact_path)

    result_text = _extract_phase_result(normalized_payload) if is_report_artifact else ""
    raw_response = "" if is_report_artifact else _extract_raw_response(normalized_payload)
    secondary_text = result_text or raw_response
    secondary_artifact_name = (
        response_artifact_name.replace("_response_", "_result_").replace(".txt", ".json")
        if is_report_artifact
        else response_artifact_name
    )
    latest_secondary_artifact_name = (
        latest_response_artifact_name.replace("_response_", "_result_").replace(".txt", ".json")
        if is_report_artifact
        else latest_response_artifact_name
    )
    if write_phase_prompt_response and not suppress_phase_prompt_response and secondary_text:
        secondary_artifact_path = target_dir / secondary_artifact_name
        secondary_artifact_path.write_text(
            secondary_text,
            encoding="utf-8",
        )
        if is_report_artifact:
            artifact_paths["result_artifact_path"] = str(secondary_artifact_path)
        else:
            artifact_paths["response_artifact_path"] = str(secondary_artifact_path)
    session_transcript = _extract_session_transcript(normalized_payload)
    if write_session_transcript and session_transcript:
        session_transcript_artifact_path = target_dir / session_transcript_artifact_name
        session_transcript_artifact_path.write_text(
            session_transcript,
            encoding="utf-8",
        )
        artifact_paths["session_transcript_artifact_path"] = str(
            session_transcript_artifact_path
        )
    if write_latest:
        if write_phase_prompt_response and not suppress_phase_prompt_response and str(primary_text or "").strip():
            latest_primary_artifact_path = target_dir / latest_primary_artifact_name
            latest_primary_artifact_path.write_text(
                primary_text,
                encoding="utf-8",
            )
            if is_report_artifact:
                artifact_paths["latest_report_artifact_path"] = str(
                    latest_primary_artifact_path
                )
            else:
                artifact_paths["latest_prompt_artifact_path"] = str(
                    latest_primary_artifact_path
                )
        if write_phase_prompt_response and not suppress_phase_prompt_response and secondary_text:
            latest_secondary_artifact_path = target_dir / latest_secondary_artifact_name
            latest_secondary_artifact_path.write_text(
                secondary_text,
                encoding="utf-8",
            )
            if is_report_artifact:
                artifact_paths["latest_result_artifact_path"] = str(
                    latest_secondary_artifact_path
                )
            else:
                artifact_paths["latest_response_artifact_path"] = str(
                    latest_secondary_artifact_path
                )
        if write_session_transcript and session_transcript:
            latest_session_transcript_artifact_path = (
                target_dir / latest_session_transcript_artifact_name
            )
            latest_session_transcript_artifact_path.write_text(
                session_transcript,
                encoding="utf-8",
            )
            artifact_paths["latest_session_transcript_artifact_path"] = str(
                latest_session_transcript_artifact_path
            )
    return artifact_paths


__all__ = [
    "DEFAULT_BRIDGE_RUNTIME_DATA_DIR",
    "DEFAULT_BRIDGE_DEBUG_DIR",
    "compact_multi_turn_runtime_session",
    "write_bridge_artifacts",
]
