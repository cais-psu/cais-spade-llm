"""Shared message helpers for agent-owned recovery outline validation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

RECOVERY_OUTLINE_PHYSICAL_VALIDATE = "recovery_outline_physical_validate"
RECOVERY_OUTLINE_PHYSICAL_VALIDATED = "recovery_outline_physical_validated"
RECOVERY_OUTLINE_SAFETY_VALIDATE = "recovery_outline_safety_validate"
RECOVERY_OUTLINE_SAFETY_VALIDATED = "recovery_outline_safety_validated"


def recovery_validation_fingerprint(value: Any) -> str:
    """Return a stable fingerprint without changing any formal-system token."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recovery_validation_stage(  # noqa: PLR0913
    *,
    validation_category: str,
    validator_role: str,
    validator_jid: str,
    status: str,
    findings: list[dict[str, Any]] | None = None,
    request_id: str = "",
    latency_ms: float | None = None,
    state_fingerprint: str = "",
    snapshot_fingerprint: str = "",
    mocked: bool = False,
) -> dict[str, Any]:
    """Build one auditable validation-stage record."""
    row: dict[str, Any] = {
        "validation_category": str(validation_category or "").strip(),
        "validator_role": str(validator_role or "").strip(),
        "validator_jid": str(validator_jid or "").strip(),
        "request_id": str(request_id or "").strip(),
        "status": str(status or "").strip(),
        "findings": [deepcopy(item) for item in (findings or []) if isinstance(item, dict)],
        "latency_ms": round(max(0.0, float(latency_ms or 0.0)), 3),
        "state_fingerprint": str(state_fingerprint or "").strip(),
        "snapshot_fingerprint": str(snapshot_fingerprint or "").strip(),
        "mocked": bool(mocked),
    }
    return row


def recovery_validation_reply_matches(
    payload: dict[str, Any],
    *,
    request_id: str,
    recovery_session_id: str,
    turn_index: int,
    state_fingerprint: str,
) -> bool:
    """Return whether a reply belongs to the active validation request."""
    try:
        reply_turn_index = int(payload.get("turn_index") or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        str(payload.get("request_id") or "").strip() == str(request_id or "").strip()
        and str(payload.get("recovery_session_id") or "").strip()
        == str(recovery_session_id or "").strip()
        and reply_turn_index == int(turn_index or 0)
        and str(payload.get("state_fingerprint") or "").strip()
        == str(state_fingerprint or "").strip()
    )


__all__ = [
    "RECOVERY_OUTLINE_PHYSICAL_VALIDATE",
    "RECOVERY_OUTLINE_PHYSICAL_VALIDATED",
    "RECOVERY_OUTLINE_SAFETY_VALIDATE",
    "RECOVERY_OUTLINE_SAFETY_VALIDATED",
    "recovery_validation_fingerprint",
    "recovery_validation_reply_matches",
    "recovery_validation_stage",
]
