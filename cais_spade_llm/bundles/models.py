"""Shared types and helpers for verified bundle persistence."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BUNDLE_STATUS_DRAFT = "draft"
BUNDLE_STATUS_VERIFIED = "verified"
BUNDLE_STATUS_STALE = "stale"
BUNDLE_STATUS_INVALID = "invalid"
BUNDLE_PLAN_GENERATION_MODE_RUNTIME_ONLY_SAFETY_FALLBACK = "runtime_only_safety_fallback"
BUNDLE_VALIDATION_FALLBACK_TRIGGER_GLOBAL_FSA_COMPILE_FAILED = "global_fsa_compile_failed"

INDEX_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    p = Path(path)
    return sha256_bytes(p.read_bytes())


def slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value))
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-_").lower() or "bundle"


def atomic_json_write(path: Path | str, payload: Any) -> None:
    """Write JSON atomically to reduce partial-index corruption risk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, p)


def is_runtime_only_safety_fallback_manifest(manifest: Any) -> bool:
    if not isinstance(manifest, dict):
        return False
    if (
        str(manifest.get("plan_generation_mode", "")).strip()
        == BUNDLE_PLAN_GENERATION_MODE_RUNTIME_ONLY_SAFETY_FALLBACK
    ):
        return True
    validation = manifest.get("validation_summary", {})
    if not isinstance(validation, dict):
        return False
    if (
        str(validation.get("stop_reason", "")).strip()
        == BUNDLE_PLAN_GENERATION_MODE_RUNTIME_ONLY_SAFETY_FALLBACK
    ):
        return True
    return bool(validation.get("offline_validation_skipped", False))
