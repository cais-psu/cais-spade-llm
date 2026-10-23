"""Product-owned nominal state projected from acknowledged resource transitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypedDict


class ProductState(TypedDict):
    """A part's state: where it is, its unfinished process, and its completed-process ledger.

    As in the semiconductor framework's ``ProductState``, a part is its location and its process record.
    How it is held follows from the location (a resource holding it, a place, or the product it is
    assembled on), so no separate handling state is stored.
    """

    location: str | None
    current_process: dict | None
    last_task: str | None
    processCompleted: list[dict]


def nominal_product_states(
    models: dict,
    valuation: dict,
    previous: dict[str, ProductState] | None = None,
) -> dict[str, ProductState]:
    """Derive product locations from resource custody, preserving process history.

    Configured completed printer outputs seed process facts only at initialization;
    they do not manufacture acknowledged task records.
    """
    product = models["Exit"]["assignments"]["completed_product"]
    parts = [
        *models["Conveyor"]["assignments"]["nominal_parts"],
        *models["3D Printing Station"]["assignments"]["supported_products"],
        product,
    ]
    states = {
        part: ProductState(
            location=None,
            current_process=deepcopy((previous or {}).get(part, {}).get("current_process")),
            last_task=(previous or {}).get(part, {}).get("last_task"),
            processCompleted=list((previous or {}).get(part, {}).get("processCompleted", [])),
        )
        for part in parts
    }

    def locate(part: str, location: str) -> None:
        states[part]["location"] = location

    for rid, values in valuation.items():
        for field, value in values.items():
            if field.startswith(("inventory.", "output.")) and value is True:
                part = field.split(".", 1)[1]
                locate(part, rid)
                if field.startswith("output.") and previous is None:
                    states[part]["processCompleted"] = [{"process": "print_part"}]
            elif field.startswith("assembled.") and value is True:
                locate(field.split(".", 1)[1], product)
            elif field in {"held_part", "part_name"} and value is not None:
                locate(value, rid)
            elif field == "staging_part" and value is not None:
                locate(value, f"{rid} staging tray")
            elif field.startswith("zone_") and field.endswith("_part") and value is not None:
                locate(value, rid)
            elif rid == "Conveyor" and field.startswith("part_location.") and value is not None:
                locate(field.split(".", 1)[1], rid)
    if states[product]["location"] is None:
        locate(product, valuation["Exit"]["product_location"])
    return states


def project_product_states(
    models: dict,
    before: dict,
    after: dict,
    previous: dict[str, ProductState],
    task: dict[str, Any],
    event: dict,
) -> tuple[dict[str, ProductState], list[str]]:
    """Project affected product states without mutating product or resource owners."""
    states = nominal_product_states(models, after, previous)
    parameters = task["parameters"]
    affected = {part for part in states if states[part] != previous[part]}
    for parameter in ("part_name", "delivered_part"):
        if parameters.get(parameter) in states:
            affected.add(parameters[parameter])
    actor = task["resource_id"]
    if before[actor].get("held_part") is not None:
        affected.add(before[actor]["held_part"])
    for rid, values in after.items():
        for field, value in values.items():
            if field.startswith(("part_location.", "part_order.")) and before[rid][field] != value:
                affected.add(field.split(".", 1)[1])
    part = parameters.get("part_name")
    if part in states:
        for process in event["product_effects"]["processCompleted"]:
            if process not in states[part]["processCompleted"]:
                states[part]["processCompleted"].append(process)
    for part in affected:
        states[part]["last_task"] = task["task_id"]
    return deepcopy(states), [part for part in states if part in affected]
