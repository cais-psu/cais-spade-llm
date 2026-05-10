"""Product-order parsing, validation, and dry-run helpers."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ProductOrder:
    """Validated product order payload with selected parts resolved from geometry."""

    payload: dict[str, Any]
    selected_parts: list[str]


_REQUIRED_FIELDS = ("product", "product_jid", "quantity", "objective")


def load_product_order_file(path: str | Path) -> dict[str, Any]:
    order_path = Path(str(path))
    with order_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("product order JSON must be an object")
    return payload


def geometry_slot_names(product_geometry: Mapping[str, Any] | None) -> list[str]:
    board = dict((product_geometry or {}).get("assembly_board") or {})
    slots = board.get("slots") or {}
    if not isinstance(slots, Mapping):
        return []
    return [str(part_name) for part_name in slots.keys()]


def validate_product_order(
    payload: Mapping[str, Any],
    product_geometry: Mapping[str, Any] | None,
) -> ProductOrder:
    if not isinstance(payload, Mapping):
        raise ValueError("product order JSON must be an object")

    if "constraints" in payload:
        raise ValueError("product order must not include constraints; put constraints in safety files")

    missing = [field for field in _REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError("product order missing required field(s): " + ", ".join(missing))

    out = dict(payload)
    for field in ("product", "product_jid", "objective"):
        value = out.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"product order field '{field}' must be a non-empty string")
        out[field] = value.strip()

    quantity = out.get("quantity")
    if isinstance(quantity, bool):
        raise ValueError("product order field 'quantity' must be a positive integer")
    try:
        quantity_int = int(quantity)
    except (TypeError, ValueError) as exc:
        raise ValueError("product order field 'quantity' must be a positive integer") from exc
    if quantity_int <= 0:
        raise ValueError("product order field 'quantity' must be a positive integer")
    out["quantity"] = quantity_int

    slots = geometry_slot_names(product_geometry)
    if not slots:
        raise ValueError("product geometry has no assembly_board slots")

    raw_parts = out.get("parts", "all")
    if raw_parts is None or raw_parts == "" or raw_parts == "all":
        selected_parts = list(slots)
        out["parts"] = "all"
    elif isinstance(raw_parts, list):
        if not raw_parts:
            raise ValueError("product order parts list must not be empty")
        selected_parts = []
        for item in raw_parts:
            part_name = str(item or "").strip()
            if not part_name:
                raise ValueError("product order parts list contains an empty part")
            selected_parts.append(part_name)
        unknown = [part for part in selected_parts if part not in slots]
        if unknown:
            raise ValueError("unknown product order part(s): " + ", ".join(unknown))
        out["parts"] = selected_parts
    else:
        raise ValueError("product order field 'parts' must be omitted, 'all', or a list")

    return ProductOrder(payload=out, selected_parts=selected_parts)


def part_place_geometry(
    part_name: str,
    product_geometry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    geometry = dict(product_geometry or {})
    board = dict(geometry.get("assembly_board") or {})
    parts = dict(geometry.get("parts") or {})
    slots = dict(board.get("slots") or {})
    slot_xy = slots.get(part_name)
    if not isinstance(slot_xy, (list, tuple)) or len(slot_xy) < 2:
        return {}
    return {
        "slot_xy": list(slot_xy[:2]),
        "part_height_m": dict(parts.get("heights_m") or {}).get(part_name),
        "model_name": dict(parts.get("model_map") or {}).get(part_name),
        "slot_floor_z_m": board.get("slot_floor_z_m"),
        "board_center": deepcopy(board.get("center") or {}),
        "target_reference": deepcopy(geometry.get("target_reference") or {}),
    }


def derive_ordering_constraints_from_safety(
    safety_text: str,
    selected_parts: list[str],
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    selected = set(selected_parts)
    pattern = re.compile(
        r"\b(?P<before>[A-Za-z0-9_-]+)\b\s+must\s+be\s+placed\s+before\s+\b(?P<after>[A-Za-z0-9_-]+)\b",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(str(safety_text or "")):
        before = str(match.group("before") or "").strip()
        after = str(match.group("after") or "").strip()
        if not before or not after:
            continue
        if before not in selected or after not in selected:
            continue
        constraints.append(
            {
                "type": "place_before",
                "before": before,
                "after": after,
                "before_event": f"place_insert({before}).done",
                "after_event": f"place_insert({after}).start",
                "raw_text": match.group(0),
            }
        )
    return constraints
