"""Bridge prompt artifact writer shared by dry-run and live runtime paths."""

from __future__ import annotations

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


def _extract_prompt_text(payload: dict[str, Any]) -> str:
    prompt_text = payload.get("single_shot_prompt_text")
    if str(prompt_text or "").strip():
        return str(prompt_text)

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
        prompt_text = prepared_bridge_request.get("single_shot_prompt_text")
        if str(prompt_text or "").strip():
            return str(prompt_text)
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            if str(latest_turn.get("prompt_text") or "").strip():
                return str(latest_turn.get("prompt_text") or "")
    return ""


def _extract_raw_response(payload: dict[str, Any]) -> str:
    raw_response = payload.get("raw_response")
    if str(raw_response or "").strip():
        return str(raw_response)

    bridge_debug = payload.get("bridge_debug")
    if isinstance(bridge_debug, dict):
        single_shot_turn = dict(bridge_debug.get("single_shot_turn") or {})
        if str(single_shot_turn.get("raw_response") or "").strip():
            return str(single_shot_turn.get("raw_response") or "")
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
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

    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        single_shot_turn = dict(bridge_debug.get("single_shot_turn") or {})
        if str(single_shot_turn.get("raw_response") or "").strip():
            return str(single_shot_turn.get("raw_response") or "")
        multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
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
    return ""


def _extract_session_transcript(payload: dict[str, Any]) -> str:
    bridge_debug = payload.get("bridge_debug")
    if not isinstance(bridge_debug, dict):
        prepared_bridge_request = payload.get("prepared_bridge_request")
        if isinstance(prepared_bridge_request, dict):
            bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        else:
            bridge_debug = {}

    multi_turn_session = dict(bridge_debug.get("multi_turn_session") or {})
    if not multi_turn_session:
        return ""
    if not list(multi_turn_session.get("turns") or []):
        return ""
    return json.dumps(
        multi_turn_session,
        indent=2,
        default=str,
        ensure_ascii=True,
    )


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
    return "single_shot"


def write_bridge_artifacts(
    payload: dict[str, Any],
    *,
    phase_label: str,
    debug_dir: str | Path | None = None,
    write_latest: bool = False,
    filename_prefix: str | None = None,
) -> dict[str, str]:
    """Write timestamped prompt/response artifacts and optional ``latest`` aliases."""
    del phase_label, filename_prefix
    normalized_payload = payload if isinstance(payload, dict) else {"payload": payload}
    target_dir = _resolve_debug_dir(debug_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    reasoning_mode = _resolve_reasoning_mode(normalized_payload)
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
    else:
        prompt_artifact_name = f"{reasoning_mode}_prompt_{timestamp}.txt"
        response_artifact_name = f"{reasoning_mode}_response_{timestamp}.txt"
        latest_prompt_artifact_name = f"{reasoning_mode}_prompt_latest.txt"
        latest_response_artifact_name = f"{reasoning_mode}_response_latest.txt"
        session_transcript_artifact_name = f"{reasoning_mode}_session_{timestamp}.txt"
        latest_session_transcript_artifact_name = f"{reasoning_mode}_session_latest.txt"

    prompt_artifact_path = target_dir / prompt_artifact_name
    prompt_text = _extract_prompt_text(normalized_payload)
    prompt_artifact_path.write_text(
        prompt_text,
        encoding="utf-8",
    )

    artifact_paths = {"prompt_artifact_path": str(prompt_artifact_path)}
    raw_response = _extract_raw_response(normalized_payload)
    if raw_response:
        response_artifact_path = target_dir / response_artifact_name
        response_artifact_path.write_text(
            raw_response,
            encoding="utf-8",
        )
        artifact_paths["response_artifact_path"] = str(response_artifact_path)
    session_transcript = _extract_session_transcript(normalized_payload)
    if session_transcript:
        session_transcript_artifact_path = target_dir / session_transcript_artifact_name
        session_transcript_artifact_path.write_text(
            session_transcript,
            encoding="utf-8",
        )
        artifact_paths["session_transcript_artifact_path"] = str(
            session_transcript_artifact_path
        )
    if write_latest:
        latest_prompt_artifact_path = target_dir / latest_prompt_artifact_name
        latest_prompt_artifact_path.write_text(
            prompt_text,
            encoding="utf-8",
        )
        artifact_paths["latest_prompt_artifact_path"] = str(latest_prompt_artifact_path)
        if raw_response:
            latest_response_artifact_path = target_dir / latest_response_artifact_name
            latest_response_artifact_path.write_text(
                raw_response,
                encoding="utf-8",
            )
            artifact_paths["latest_response_artifact_path"] = str(
                latest_response_artifact_path
            )
        if session_transcript:
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
