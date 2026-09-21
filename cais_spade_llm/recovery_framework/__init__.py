"""Recovery-framework configuration and recorded Gazebo evidence helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = ROOT / "cais_spade_llm/initialization/recovery_framework_gazebo.json"
PRODUCT_PATH = (
    ROOT / "cais_spade_llm/initialization/products/assembly_board-v1-recovery-framework.json"
)


def fingerprint(value: object) -> str:
    """Hash exact JSON values independently of object member ordering."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def read_json(path: str | Path) -> dict:
    """Read one JSON object without invoking runtime or configuration writes."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value
