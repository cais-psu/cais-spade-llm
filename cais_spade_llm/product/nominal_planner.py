"""Deterministic nominal planning from ResourceAgent capability descriptors."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from itertools import combinations_with_replacement
from typing import Any

from cais_spade_llm.resources.nominal_context import NominalResourceContext
from cais_spade_llm.resources.nominal_conveyor import CONVEYOR_LOCATIONS, conveyor_parts


def nominal_requirement(models: dict, part_name: str) -> dict:
    """Bind a product requirement to exact resource-owned completion facts."""
    product = models["Exit"]["assignments"]["completed_product"]
    if part_name == product:
        return {"Exit": {"part_name": product, "resource_state": "occupied"}}
    for rid, model in models.items():
        if f"assembled.{part_name}" in model["state_variables"]:
            return {rid: {f"assembled.{part_name}": True, "resource_state": "idle"}}
    return {}


def nominal_goal(models: dict, valuation: dict, part_name: str, conditions: dict | None = None) -> bool:
    """Check component assembly or acknowledged completed-product delivery."""
    requirement = conditions if conditions is not None else nominal_requirement(models, part_name)
    return bool(requirement) and all(
        valuation[rid][field] == expected
        for rid, conditions in requirement.items()
        for field, expected in conditions.items()
    )


def _templates(models: dict, part_name: str) -> list[dict]:
    templates = []
    for rid, model in models.items():
        for event in model["events"]:
            if event["parameter_bindings"]["resource_id"]["equals"] != rid:
                continue
            params: dict[str, Any] = {}
            for field, rule in event["parameter_bindings"].items():
                if "equals" in rule:
                    params[field] = rule["equals"]
                elif field in {"next_locations", "delivered_part"}:
                    continue
                elif "from_assignment" in rule:
                    assignment = rule["from_assignment"]
                    allowed = models[assignment["resource_id"]]["assignments"][assignment["field"]]
                    if part_name not in allowed:
                        break
                    params[field] = part_name
                else:
                    break
            else:
                if "part_name" in params and params["part_name"] != part_name:
                    continue
                if (
                    event["event_name"] == "move_home"
                    and part_name not in (model["state_variables"]["held_part"]["domain"])
                ):
                    continue
                templates.append(
                    {
                        "resource_id": rid,
                        "event_id": event["event_id"],
                        "event_name": event["event_name"],
                        "parameters": params,
                    }
                )
    return templates


def _candidates(models: dict, valuation: dict, templates: list[dict]):
    for task in templates:
        if task["event_name"] != "advance_conveyor":
            yield task
            continue
        resident = conveyor_parts(models["Conveyor"], valuation["Conveyor"])
        if not resident:
            continue
        # A missing delivered_part binding denotes the parameterized handoff variant.
        delivered = None if "delivered_part" in task["parameters"] else resident[0]
        remaining = [part for part in resident if part != delivered]
        for indices in combinations_with_replacement(
            range(len(CONVEYOR_LOCATIONS)), len(remaining)
        ):
            locations = {
                part: CONVEYOR_LOCATIONS[index]
                for part, index in zip(remaining, reversed(indices), strict=True)
            }
            yield {
                **task,
                "parameters": {
                    **task["parameters"],
                    "delivered_part": delivered,
                    "next_locations": locations,
                },
            }


def search_nominal_part(
    resources: dict[str, NominalResourceContext],
    valuation: dict,
    part_name: str,
    *,
    max_search_states: int = 50_000,
    completion_conditions: dict | None = None,
) -> dict:
    """Search one component goal without changing any resource or product state.

    Every candidate contains explicit simulated completion assumptions. They are
    planning inputs, not evidence that a physical task has completed.
    """
    if type(max_search_states) is not int or max_search_states < 1:
        raise ValueError("max_search_states must be a positive integer")
    models = {rid: resource.nominal_des_model() for rid, resource in resources.items()}
    templates = _templates(models, part_name)

    def key(state: dict) -> tuple:
        return tuple(
            tuple(state[rid][field] for field in models[rid]["state_variables"]) for rid in models
        )

    queue = deque([(valuation, [])])
    visited = {key(valuation)}
    expanded = 0
    while queue:
        if expanded >= max_search_states:
            return {
                "status": "budget_exhausted",
                "part_name": part_name,
                "expanded": expanded,
                "tasks": [],
            }
        before, trace = queue.popleft()
        expanded += 1
        if nominal_goal(models, before, part_name, completion_conditions):
            return {
                "status": "planned",
                "part_name": part_name,
                "expanded": expanded,
                "tasks": deepcopy(trace),
            }
        for task in _candidates(models, before, templates):
            try:
                after = resources[task["resource_id"]].validate_nominal_event(models, before, task)
            except ValueError:
                continue
            signature = key(after)
            if signature in visited:
                continue
            visited.add(signature)
            queue.append((after, [*trace, task]))
    return {"status": "blocked", "part_name": part_name, "expanded": expanded, "tasks": []}


def plan_nominal_order(context: Any, *, max_search_states: int = 50_000) -> dict:
    """Request resource capabilities and select the first feasible component plan."""
    valuation = context.snapshot()
    models = context.models
    goals = list(context.selected_parts)
    conditions = context.product_order.get("completion_conditions")
    if conditions is None and set(goals) == set(context.component_parts):
        goals.append(context.product_name)
    requests = []
    for part in goals:
        if nominal_goal(models, valuation, part, conditions):
            continue
        request = {
            "part_name": part,
            "revision": context.revision,
            "resources": list(context.resources),
        }
        result = search_nominal_part(
            context.resources, valuation, part, max_search_states=max_search_states,
            completion_conditions=conditions,
        )
        requests.append({**request, "status": result["status"], "expanded": result["expanded"]})
        if result["status"] == "planned":
            return {**result, "requests": requests}
    if not requests:
        return {"status": "completed", "tasks": [], "requests": []}
    status = (
        "budget_exhausted"
        if any(row["status"] == "budget_exhausted" for row in requests)
        else "blocked"
    )
    return {"status": status, "tasks": [], "requests": requests}
