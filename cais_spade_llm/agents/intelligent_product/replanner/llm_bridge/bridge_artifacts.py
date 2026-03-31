"""Bridge prompt artifact writer shared by dry-run and live runtime paths."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BRIDGE_DEBUG_DIR = Path("cais_spade_llm/monitor/debug")


def _resolve_debug_dir(debug_dir: str | Path | None) -> Path:
    if debug_dir is None:
        return DEFAULT_BRIDGE_DEBUG_DIR
    candidate = Path(debug_dir)
    return candidate if str(candidate).strip() else DEFAULT_BRIDGE_DEBUG_DIR


def _extract_prompt_text(payload: dict[str, Any]) -> str:
    prompt_text = payload.get("single_shot_prompt_text")
    if str(prompt_text or "").strip():
        return str(prompt_text)
    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        return str(prepared_bridge_request.get("single_shot_prompt_text") or "")
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

    prepared_bridge_request = payload.get("prepared_bridge_request")
    if isinstance(prepared_bridge_request, dict):
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        single_shot_turn = dict(bridge_debug.get("single_shot_turn") or {})
        if str(single_shot_turn.get("raw_response") or "").strip():
            return str(single_shot_turn.get("raw_response") or "")
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
    prompt_artifact_path = target_dir / f"{reasoning_mode}_prompt_{timestamp}.txt"
    prompt_text = _extract_prompt_text(normalized_payload)
    prompt_artifact_path.write_text(
        prompt_text,
        encoding="utf-8",
    )

    artifact_paths = {"prompt_artifact_path": str(prompt_artifact_path)}
    raw_response = _extract_raw_response(normalized_payload)
    if raw_response:
        response_artifact_path = target_dir / f"{reasoning_mode}_response_{timestamp}.txt"
        response_artifact_path.write_text(
            raw_response,
            encoding="utf-8",
        )
        artifact_paths["response_artifact_path"] = str(response_artifact_path)
    if write_latest:
        latest_prompt_artifact_path = target_dir / f"{reasoning_mode}_prompt_latest.txt"
        latest_prompt_artifact_path.write_text(
            prompt_text,
            encoding="utf-8",
        )
        artifact_paths["latest_prompt_artifact_path"] = str(latest_prompt_artifact_path)
        if raw_response:
            latest_response_artifact_path = target_dir / f"{reasoning_mode}_response_latest.txt"
            latest_response_artifact_path.write_text(
                raw_response,
                encoding="utf-8",
            )
            artifact_paths["latest_response_artifact_path"] = str(
                latest_response_artifact_path
            )
    return artifact_paths


__all__ = ["DEFAULT_BRIDGE_DEBUG_DIR", "write_bridge_artifacts"]
