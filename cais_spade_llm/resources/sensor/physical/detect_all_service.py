from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

SNAPSHOT_PATH = Path("/tmp/cais_physical_perception.json")
MAXIMUM_SNAPSHOT_AGE_SEC = 10.0
MAXIMUM_DETECTION_AGE_SEC = 10.0


def detect_all() -> dict[str, dict[str, Any]]:
    """
    Direct physical perception entrypoint for all currently visible parts.

    Coordinates are world-frame metres. This compatibility client reads only
    validated output written atomically by ``realsense_roboflow_node``.
    """
    if not SNAPSHOT_PATH.is_file():
        return {}
    try:
        payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    now = time.time()
    if now - float(payload.get("updated_at", 0.0)) > MAXIMUM_SNAPSHOT_AGE_SEC:
        return {}
    if str(payload.get("last_error") or "").strip():
        return {}
    rows = payload.get("detections", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row["part_name"]): dict(row)
        for row in rows
        if isinstance(row, dict)
        and row.get("part_name")
        and now - float(row.get("captured_at", 0.0)) <= MAXIMUM_DETECTION_AGE_SEC
    }
