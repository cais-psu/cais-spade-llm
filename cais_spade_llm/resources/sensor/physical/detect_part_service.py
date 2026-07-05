from __future__ import annotations

from typing import Any


def detect_part(part_name: str) -> dict[str, Any] | None:
    """
    Direct physical perception entrypoint for a single part.

    Contract:
      - Return None when not detected.
      - Return dict containing at least x,y,z in mm when detected.
      - Optional keys: confidence, camera_id, orientation, model_name.
    """
    _ = str(part_name or "").strip().upper()
    # TODO: implement YOLO + depth + calibration pipeline.
    return None
