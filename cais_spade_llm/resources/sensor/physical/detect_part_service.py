from __future__ import annotations

from typing import Any

from .detect_all_service import detect_all
from .roboflow_detector import UNSUPPORTED_PARTS


def detect_part(part_name: str) -> dict[str, Any] | None:
    """
    Direct physical perception entrypoint for a single part.

    Coordinates are world-frame metres. Unsupported LG remains blocked.
    """
    target = str(part_name or "").strip().upper()
    if target in UNSUPPORTED_PARTS:
        return None
    return detect_all().get(target)
