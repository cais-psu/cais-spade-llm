from __future__ import annotations

from typing import Any


def detect_all() -> dict[str, dict[str, Any]]:
    """
    Direct physical perception entrypoint for all currently visible parts.

    Contract:
      - Return mapping: part_name -> dict with at least x,y,z in mm.
      - Optional keys per part: confidence, camera_id, orientation, model_name.
    """
    # TODO: implement YOLO + depth fusion + multi-camera association.
    return {}
