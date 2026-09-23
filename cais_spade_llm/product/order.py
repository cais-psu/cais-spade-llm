"""Product-order parsing, validation, and dry-run helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cais_spade_llm.product.stl_geometry import actual_mg_stl_geometry_for_part


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
    *,
    require_process_requirements: bool = False,
) -> ProductOrder:
    """Validate an order, requiring explicit results for environmental matching."""
    if not isinstance(payload, Mapping):
        raise ValueError("product order JSON must be an object")

    if "constraints" in payload:
        raise ValueError(
            "product order must not include constraints; put constraints in safety files"
        )

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

    if "completion_conditions" in out:
        validate_completion_conditions(out["completion_conditions"])
        if len(selected_parts) != 1:
            raise ValueError("completion_conditions currently require one selected part")

    if "machine_resource" in out:
        if (
            len(selected_parts) != 1
            or "processPlan" not in out
            or "completion_conditions" in out
            or not isinstance(out["machine_resource"], str)
            or not out["machine_resource"]
        ):
            raise ValueError("machine_resource requires a one-part processPlan order")

    _validate_order_processes(out, selected_parts, product_geometry, require_process_requirements)

    return ProductOrder(payload=out, selected_parts=selected_parts)


def _validate_order_processes(
    order: dict, parts: list[str], geometry: Mapping, required: bool
) -> None:
    if "processPlan" in order and "requirements" in order:
        raise ValueError("Use processPlan or historical requirements, not both")
    if "processPlan" in order:
        validate_process_plan(order["processPlan"], parts, geometry)
    elif "requirements" in order:
        validate_process_requirements(order.get("requirements"), parts, geometry)
    elif required:
        raise ValueError("processPlan must declare ordered steps for every selected part")


def _validate_process_requirement(requirement: Any) -> None:
    if (
        not isinstance(requirement, dict)
        or set(requirement) not in ({"process"}, {"process", "result"})
        or not isinstance(requirement.get("process"), str)
        or not requirement["process"]
    ):
        raise ValueError("Process requirements need an exact process and optional result")
    if "result" in requirement and (
        not isinstance(requirement["result"], str) or not requirement["result"]
    ):
        raise ValueError("Process result must be a nonempty string")
    if requirement["process"] == "trim" and "result" not in requirement:
        raise ValueError("trim requires an explicit result")


def validate_process_plan(plan: Any, parts: list[str], geometry: Mapping) -> None:
    """Validate ordered process steps, resolving assembly features from geometry.

    Each step is a conjunction. Process names and results remain exact symbols;
    resource declarations determine which requirements can be fulfilled.
    """
    if not isinstance(plan, dict) or any(part not in plan for part in parts):
        raise ValueError("processPlan must declare ordered steps for every selected part")
    targets = geometry.get("parts", {}).get("assembly_target_map", {})
    for part, steps in plan.items():
        if part not in geometry_slot_names(geometry) or not isinstance(steps, list) or not steps:
            raise ValueError("processPlan requires known product parts and nonempty steps")
        seen = []
        for index, step in enumerate(steps):
            if not isinstance(step, dict) or set(step) != {"processesToComplete"}:
                raise ValueError("Each processPlan step must contain only processesToComplete")
            processes = step["processesToComplete"]
            if not isinstance(processes, list) or not processes:
                raise ValueError("processesToComplete must be a nonempty list")
            for requirement in processes:
                _validate_process_requirement(requirement)
                if requirement["process"] == "assembly":
                    if set(requirement) != {"process"} or index != len(steps) - 1:
                        raise ValueError("assembly must occur in the final step without a result")
                    if not isinstance(targets.get(part), str) or not targets[part]:
                        raise ValueError("assembly requires the exact feature in product geometry")
                if requirement in seen:
                    raise ValueError("Repeated process requirements need distinct results")
                seen.append(requirement)
        if {"process": "assembly"} not in steps[-1]["processesToComplete"]:
            raise ValueError("Component processPlan must finish with assembly")


def validate_process_requirements(requirements: Any, parts: list[str], geometry: Mapping) -> None:
    """Validate ordered product results without assigning resources or tasks."""
    if not isinstance(requirements, dict) or any(part not in requirements for part in parts):
        raise ValueError("requirements must declare ordered results for every selected part")
    targets = geometry.get("parts", {}).get("assembly_target_map", {})
    for part, sequence in requirements.items():
        if part not in geometry_slot_names(geometry) or not isinstance(sequence, list) or not sequence:
            raise ValueError("requirements must reference known product parts and nonempty result sequences")
        for result in sequence:
            if not isinstance(result, dict):
                raise ValueError("Each product requirement must be a result object")
            if result.get("process") == "trim":
                if set(result) != {"process", "result"} or not isinstance(result["result"], str) or not result["result"]:
                    raise ValueError("trim requires an explicit result")
            elif result.get("process") == "print_part":
                if set(result) != {"process"}:
                    raise ValueError("print_part requirement has unsupported fields")
            elif result.get("state") == "assembled":
                if set(result) != {"state", "target"} or not result.get("target") or result["target"] != targets.get(part):
                    raise ValueError("assembled requires the exact product geometry target")
            else:
                raise ValueError("Unsupported product result; declare a supported process or assembled target")
        if sequence[-1].get("state") != "assembled":
            raise ValueError("Component requirements must finish at their assembled target")


def validate_completion_conditions(conditions: Any, models: Mapping | None = None) -> dict:
    """Validate a conjunction of exact resource valuation fields and values.

    Args:
        conditions: Resource IDs mapped to state fields and expected values.
        models: Optional descriptors for checking exact identifiers and domains.

    Returns:
        An independent copy of the validated conditions.
    """
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("completion_conditions must be a nonempty resource map")
    for resource_id, fields in conditions.items():
        if not isinstance(resource_id, str) or not isinstance(fields, dict) or not fields:
            raise ValueError("completion_conditions require exact resource IDs and state fields")
        if models is not None and resource_id not in models:
            raise ValueError(f"Unknown completion_conditions resource: {resource_id}")
        for field, expected in fields.items():
            if not isinstance(field, str) or type(expected) not in (str, bool, int, float, type(None)):
                raise ValueError("completion_conditions require state fields and scalar values")
            if models is None:
                continue
            variables = models[resource_id]["state_variables"]
            if field not in variables:
                raise ValueError(f"Unknown completion_conditions field: {resource_id}.{field}")
            if not any(type(expected) is type(value) and expected == value
                       for value in variables[field]["domain"]):
                raise ValueError(f"Invalid completion_conditions value: {resource_id}.{field}")
    return deepcopy(conditions)


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
    result = {
        "slot_xy": list(slot_xy[:2]),
        "part_height_m": dict(parts.get("heights_m") or {}).get(part_name),
        "model_name": dict(parts.get("model_map") or {}).get(part_name),
        "slot_floor_z_m": board.get("slot_floor_z_m"),
        "board_center": deepcopy(board.get("center") or {}),
        "target_reference": deepcopy(geometry.get("target_reference") or {}),
    }
    result.update(actual_mg_stl_geometry_for_part(part_name, parts))
    return result


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
