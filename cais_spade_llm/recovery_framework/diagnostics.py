"""Isolated recovery generation and validation jobs using saved inputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def read_object(path: Path) -> dict:
    """Read a JSON object without changing its identifiers."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def write_record(path: Path, value: dict) -> None:
    """Atomically publish a diagnostic record for polling clients."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    temporary.replace(path)


