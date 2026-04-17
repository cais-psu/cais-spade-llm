"""Helpers for resolving symbolic placement destinations into concrete geometry."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]


def has_place_geometry_fields(geometry: Mapping[str, Any] | None) -> bool:
    """Return True when the dict already contains enough place-target geometry."""
    if not isinstance(geometry, Mapping):
        return False
    slot_xy = geometry.get("slot_xy")
    board_center = geometry.get("board_center")
    slot_floor_z = geometry.get("slot_floor_z_m")
    return (
        isinstance(slot_xy, (list, tuple))
        and len(slot_xy) >= 2
        and isinstance(board_center, Mapping)
        and slot_floor_z is not None
    )


def destination_token_from_place_inputs(
    *,
    destination_location: str = "",
    product_geometry: Mapping[str, Any] | None = None,
) -> str:
    """Extract a symbolic destination token from primitive params."""
    direct_token = _normalize_destination_token(destination_location)
    if direct_token:
        return direct_token

    geometry = dict(product_geometry) if isinstance(product_geometry, Mapping) else {}
    for key in ("destination_location", "destination", "location", "fixture_name"):
        token = _normalize_destination_token(geometry.get(key))
        if token:
            return token
    return ""


def resolve_place_geometry(
    *,
    part_name: str,
    destination_location: str = "",
    product_geometry: Mapping[str, Any] | None = None,
    execution_mode: str = "simulation",
) -> dict[str, Any]:
    """Return explicit place geometry, resolving symbolic destinations when possible."""
    geometry = dict(product_geometry) if isinstance(product_geometry, Mapping) else {}
    if has_place_geometry_fields(geometry):
        return geometry

    token = destination_token_from_place_inputs(
        destination_location=destination_location,
        product_geometry=geometry,
    )
    if not token or not str(part_name or "").strip():
        return geometry

    resolved = _load_geometry_for_destination_part(
        token=token,
        part_name=str(part_name or "").strip(),
        execution_mode=execution_mode,
    )
    if not resolved:
        return geometry

    merged = dict(resolved)
    merged.update(geometry)
    return merged


@lru_cache(maxsize=16)
def _load_geometry_for_destination_part(
    *,
    token: str,
    part_name: str,
    execution_mode: str,
) -> dict[str, Any]:
    geometry_doc = _load_geometry_doc_for_destination(
        token=token,
        execution_mode=execution_mode,
    )
    if not geometry_doc:
        return {}

    board = dict(geometry_doc.get("assembly_board") or {})
    parts = dict(geometry_doc.get("parts") or {})
    slot_xy = dict(board.get("slots") or {}).get(part_name)
    if not isinstance(slot_xy, (list, tuple)) or len(slot_xy) < 2:
        return {}

    return {
        "slot_xy": list(slot_xy[:2]),
        "part_height_m": dict(parts.get("heights_m") or {}).get(part_name),
        "model_name": dict(parts.get("model_map") or {}).get(part_name),
        "slot_floor_z_m": board.get("slot_floor_z_m"),
        "board_center": board.get("center") or {},
    }


@lru_cache(maxsize=32)
def _load_geometry_doc_for_destination(
    *,
    token: str,
    execution_mode: str,
) -> dict[str, Any]:
    for geometry_path in (
        _product_geometry_path_for_token(token),
        _direct_geometry_path_for_token(token),
    ):
        if geometry_path is None or not geometry_path.exists():
            continue
        try:
            payload = json.loads(geometry_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        env_key = "real" if str(execution_mode or "").strip().lower() == "physical" else "gazebo"
        geometry_doc = payload.get(env_key)
        if not isinstance(geometry_doc, dict):
            geometry_doc = payload
        if isinstance(geometry_doc, dict):
            return dict(geometry_doc)
    return {}


def _product_geometry_path_for_token(token: str) -> Path | None:
    product_meta = _load_product_meta(token)
    if not product_meta:
        return None
    geometry_relpath = str(product_meta.get("product_geometry_file") or "").strip()
    if not geometry_relpath:
        return None
    return _REPO_ROOT / geometry_relpath


def _direct_geometry_path_for_token(token: str) -> Path:
    # Dry-run staging anchors such as printers may have geometry mocks without
    # being registered as full product manifests.
    return (
        _PACKAGE_ROOT
        / "specification"
        / "products"
        / "geometry"
        / f"{token}.json"
    )


@lru_cache(maxsize=16)
def _load_product_meta(token: str) -> dict[str, Any]:
    init_path = _PACKAGE_ROOT / "initialization" / "products" / f"{token}.json"
    if not init_path.exists():
        return {}

    try:
        payload = json.loads(init_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    raw_meta = payload.get(token)
    if isinstance(raw_meta, dict):
        return dict(raw_meta)

    if len(payload) == 1:
        only_value = next(iter(payload.values()))
        if isinstance(only_value, dict):
            return dict(only_value)
    return {}


def _normalize_destination_token(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    if "@" in token:
        token = token.split("@", 1)[0].strip()
    return token
