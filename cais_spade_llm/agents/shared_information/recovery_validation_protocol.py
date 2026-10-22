"""Shared message helpers for agent-owned recovery outline validation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


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
    evidence: dict[str, Any] | None = None,
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
    if evidence is not None:
        row["evidence"] = deepcopy(evidence)
        unavailable = evidence.get("feasibility_status") == "NEEDS_CONTEXT" or any(
            item.get("constraint_code") in {"resource_validation_unavailable", "safety_validation_unavailable"}
            for item in row["findings"]
        )
        row["outcome"] = "unresolved" if unavailable else row["status"]
        row["reason"] = evidence.get("reason") or "; ".join(
            str(item["reason"]) for item in row["findings"] if item.get("reason")
        )
    return row


__all__ = [
    "recovery_validation_fingerprint",
    "recovery_validation_stage",
]
