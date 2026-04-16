"""Bridge prompt artifact writer shared by dry-run and live runtime paths."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_BRIDGE_DEBUG_DIR = Path("cais_spade_llm/monitor/debug")


def _resolve_debug_dir(debug_dir: str | Path | None) -> Path:
    if debug_dir is None:
        return DEFAULT_BRIDGE_DEBUG_DIR
    candidate = Path(debug_dir)
    return candidate if str(candidate).strip() else DEFAULT_BRIDGE_DEBUG_DIR


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
    action_name = str(task.get("action_name") or task.get("name") or "").strip()
    part_name = str(task.get("part_name") or "").strip()
    location_ref = str(task.get("location_ref") or task.get("target_ref") or "").strip()
    if action_name:
        qualifiers = [item for item in (part_name, location_ref) if item]
        if qualifiers:
            return f"{action_name} ({' -> '.join(qualifiers)})"
        return action_name
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
    turns = list(multi_turn_session.get("turns") or [])
    latest_turn = dict(turns[-1] or {}) if turns else {}
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


def _des_artifact_context(payload: dict[str, Any], *, session_key: str) -> dict[str, Any]:
    bridge_debug = _bridge_debug_payload(payload)
    session = dict(bridge_debug.get(session_key) or {})
    turns = list(session.get("turns") or [])
    latest_turn = dict(turns[-1] or {}) if turns else {}
    turn_index = int(
        latest_turn.get("turn_index")
        or session.get("turn_index")
        or 0
    )
    phase = _artifact_token(
        latest_turn.get("phase") or session.get("current_phase"),
        fallback="grounding",
    )
    session_id = _artifact_token(
        session.get("session_id") or session.get("des_engine"),
        fallback="session",
    )
    return {
        "turn_index": max(turn_index, 0),
        "phase": phase,
        "session_id": session_id,
    }


def _extract_prompt_text(payload: dict[str, Any]) -> str:
    prompt_text = payload.get("single_shot_prompt_text")
    if str(prompt_text or "").strip():
        return str(prompt_text)

    bridge_debug = payload.get("bridge_debug")
    if isinstance(bridge_debug, dict):
        for session_key in ("hybrid_session", "procedural_session"):
            session = dict(bridge_debug.get(session_key) or {})
            turns = list(session.get("turns") or [])
            if turns:
                latest_turn = dict(turns[-1] or {})
                if str(latest_turn.get("prompt_text") or "").strip():
                    return str(latest_turn.get("prompt_text") or "")
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            if str(latest_turn.get("prompt_text") or "").strip():
                return str(latest_turn.get("prompt_text") or "")

    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        prompt_text = prepared_bridge_request.get("single_shot_prompt_text")
        if str(prompt_text or "").strip():
            return str(prompt_text)
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        for session_key in ("hybrid_session", "procedural_session"):
            session = dict(bridge_debug.get(session_key) or {})
            turns = list(session.get("turns") or [])
            if turns:
                latest_turn = dict(turns[-1] or {})
                if str(latest_turn.get("prompt_text") or "").strip():
                    return str(latest_turn.get("prompt_text") or "")
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            if str(latest_turn.get("prompt_text") or "").strip():
                return str(latest_turn.get("prompt_text") or "")
    return ""


def _extract_report_text(payload: dict[str, Any]) -> str:
    bridge_debug = payload.get("bridge_debug")
    if isinstance(bridge_debug, dict):
        for session_key in ("hybrid_session", "procedural_session"):
            session = dict(bridge_debug.get(session_key) or {})
            turns = list(session.get("turns") or [])
            if turns:
                latest_turn = dict(turns[-1] or {})
                if str(latest_turn.get("report_text") or "").strip():
                    return str(latest_turn.get("report_text") or "")
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            if str(latest_turn.get("report_text") or "").strip():
                return str(latest_turn.get("report_text") or "")

    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        for session_key in ("hybrid_session", "procedural_session"):
            session = dict(bridge_debug.get(session_key) or {})
            turns = list(session.get("turns") or [])
            if turns:
                latest_turn = dict(turns[-1] or {})
                if str(latest_turn.get("report_text") or "").strip():
                    return str(latest_turn.get("report_text") or "")
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

    bridge_debug = payload.get("bridge_debug")
    if isinstance(bridge_debug, dict):
        single_shot_turn = dict(bridge_debug.get("single_shot_turn") or {})
        if str(single_shot_turn.get("raw_response") or "").strip():
            return str(single_shot_turn.get("raw_response") or "")
        for session_key in ("hybrid_session", "procedural_session"):
            session = dict(bridge_debug.get(session_key) or {})
            turns = list(session.get("turns") or [])
            if turns:
                latest_turn = dict(turns[-1] or {})
                latest_response = latest_turn.get("raw_response")
                if latest_response not in (None, "", [], {}):
                    if isinstance(latest_response, str):
                        return latest_response
                    return json.dumps(
                        latest_response,
                        indent=2,
                        default=str,
                        ensure_ascii=True,
                    )
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
        single_shot_turn = dict(bridge_debug.get("single_shot_turn") or {})
        if str(single_shot_turn.get("raw_response") or "").strip():
            return str(single_shot_turn.get("raw_response") or "")
        for session_key in ("hybrid_session", "procedural_session"):
            session = dict(bridge_debug.get(session_key) or {})
            turns = list(session.get("turns") or [])
            if turns:
                latest_turn = dict(turns[-1] or {})
                latest_response = latest_turn.get("raw_response")
                if latest_response not in (None, "", [], {}):
                    if isinstance(latest_response, str):
                        return latest_response
                    return json.dumps(
                        latest_response,
                        indent=2,
                        default=str,
                        ensure_ascii=True,
                    )
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
    bridge_debug = payload.get("bridge_debug")
    if isinstance(bridge_debug, dict):
        for session_key in ("hybrid_session", "procedural_session"):
            session = dict(bridge_debug.get(session_key) or {})
            turns = list(session.get("turns") or [])
            if turns:
                latest_turn = dict(turns[-1] or {})
                result = latest_turn.get("phase_result")
                if result not in (None, "", [], {}):
                    if isinstance(result, str):
                        return result
                    return json.dumps(result, indent=2, default=str, ensure_ascii=True)
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
        for session_key in ("hybrid_session", "procedural_session"):
            session = dict(bridge_debug.get(session_key) or {})
            turns = list(session.get("turns") or [])
            if turns:
                latest_turn = dict(turns[-1] or {})
                result = latest_turn.get("phase_result")
                if result not in (None, "", [], {}):
                    if isinstance(result, str):
                        return result
                    return json.dumps(result, indent=2, default=str, ensure_ascii=True)
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
        task = row.get("normalized_task")
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


def _compact_multi_turn_response_artifact(response: Any) -> Any:
    if not isinstance(response, dict):
        return response
    compact = deepcopy(response)
    compact.pop("candidate_transitions", None)
    compact.pop("candidate_tasks", None)
    compact.pop("candidate_recovery_events", None)
    if "selected_transition" in compact:
        compact.pop("selected_next_task", None)
        compact.pop("selected_recovery_event", None)
    if "accepted_transition_prefix" in compact:
        compact.pop("des_event_sequence", None)
        compact.pop("transition_trace", None)
    if "outline_tasks" in compact:
        compact.pop("accepted_transition_prefix", None)
        compact.pop("des_event_sequence", None)
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
        compact_turn.pop("candidate_transitions", None)
        compact_turn.pop("candidate_tasks", None)
        compact_turn.pop("candidate_recovery_events", None)
        if "accepted_transition_prefix" in compact_turn:
            compact_turn.pop("des_event_sequence", None)
            compact_turn.pop("transition_trace", None)
        if "outline_tasks" in compact_turn:
            compact_turn.pop("accepted_transition_prefix", None)
            compact_turn.pop("des_event_sequence", None)
        if "selected_transition" in compact_turn:
            compact_turn.pop("selected_next_task", None)
            compact_turn.pop("selected_recovery_event", None)
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
        compact_turns.append(compact_turn)
    compact_session["turns"] = compact_turns
    return compact_session


def _extract_session_transcript(payload: dict[str, Any]) -> str:
    bridge_debug = payload.get("bridge_debug")
    if not isinstance(bridge_debug, dict):
        prepared_bridge_request = payload.get("prepared_bridge_request")
        if isinstance(prepared_bridge_request, dict):
            bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        else:
            bridge_debug = {}

    for session_key in ("hybrid_session", "procedural_session", "multi_turn_session"):
        session = dict(bridge_debug.get(session_key) or {})
        if not session or not list(session.get("turns") or []):
            continue
        if session_key == "multi_turn_session":
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
    return "hybrid"


def write_bridge_artifacts(
    payload: dict[str, Any],
    *,
    phase_label: str,
    debug_dir: str | Path | None = None,
    write_latest: bool = False,
    filename_prefix: str | None = None,
    write_session_transcript: bool | None = None,
) -> dict[str, str]:
    """Write bridge prompt/response artifacts."""
    del phase_label, filename_prefix
    normalized_payload = payload if isinstance(payload, dict) else {"payload": payload}
    target_dir = _resolve_debug_dir(debug_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    reasoning_mode = _resolve_reasoning_mode(normalized_payload)
    if write_session_transcript is None:
        write_session_transcript = reasoning_mode != "multi_turn"
    write_latest = bool(write_latest) and reasoning_mode != "multi_turn"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if reasoning_mode == "multi_turn":
        multi_turn_ctx = _multi_turn_artifact_context(normalized_payload)
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
    elif reasoning_mode == "hybrid":
        hybrid_ctx = _des_artifact_context(normalized_payload, session_key="hybrid_session")
        prompt_artifact_name = (
            f"hybrid_attempt{int(hybrid_ctx['turn_index']):02d}_"
            f"{hybrid_ctx['phase']}_prompt_{timestamp}.txt"
        )
        response_artifact_name = (
            f"hybrid_attempt{int(hybrid_ctx['turn_index']):02d}_"
            f"{hybrid_ctx['phase']}_response_{timestamp}.txt"
        )
        latest_prompt_artifact_name = (
            f"hybrid_attempt{int(hybrid_ctx['turn_index']):02d}_"
            f"{hybrid_ctx['phase']}_prompt_latest.txt"
        )
        latest_response_artifact_name = (
            f"hybrid_attempt{int(hybrid_ctx['turn_index']):02d}_"
            f"{hybrid_ctx['phase']}_response_latest.txt"
        )
        session_transcript_artifact_name = (
            f"hybrid_session_{timestamp}.txt"
        )
        latest_session_transcript_artifact_name = (
            f"hybrid_session_latest.txt"
        )
    elif reasoning_mode in {"procedural", "procedural_des_v1"}:
        des_ctx = _des_artifact_context(normalized_payload, session_key="procedural_session")
        prompt_artifact_name = (
            f"procedural_attempt{int(des_ctx['turn_index']):02d}_"
            f"{des_ctx['phase']}_prompt_{timestamp}.txt"
        )
        response_artifact_name = (
            f"procedural_attempt{int(des_ctx['turn_index']):02d}_"
            f"{des_ctx['phase']}_response_{timestamp}.txt"
        )
        latest_prompt_artifact_name = (
            f"procedural_attempt{int(des_ctx['turn_index']):02d}_"
            f"{des_ctx['phase']}_prompt_latest.txt"
        )
        latest_response_artifact_name = (
            f"procedural_attempt{int(des_ctx['turn_index']):02d}_"
            f"{des_ctx['phase']}_response_latest.txt"
        )
        session_transcript_artifact_name = f"procedural_session_{timestamp}.txt"
        latest_session_transcript_artifact_name = "procedural_session_latest.txt"
    else:
        prompt_artifact_name = f"{reasoning_mode}_prompt_{timestamp}.txt"
        response_artifact_name = f"{reasoning_mode}_response_{timestamp}.txt"
        latest_prompt_artifact_name = f"{reasoning_mode}_prompt_latest.txt"
        latest_response_artifact_name = f"{reasoning_mode}_response_latest.txt"
        session_transcript_artifact_name = f"{reasoning_mode}_session_{timestamp}.txt"
        latest_session_transcript_artifact_name = f"{reasoning_mode}_session_latest.txt"

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
    if str(primary_text or "").strip():
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
    if secondary_text:
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
        if str(primary_text or "").strip():
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
        if secondary_text:
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


__all__ = ["DEFAULT_BRIDGE_DEBUG_DIR", "write_bridge_artifacts"]
