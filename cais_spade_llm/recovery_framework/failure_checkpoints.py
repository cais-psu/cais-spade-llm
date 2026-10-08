"""Observed checkpoints for the four simulation failure scenarios."""

from __future__ import annotations

from copy import deepcopy

CHECKPOINTS = {
    "Conveyor breakdown": "after_M1_pick_before_release",
    "ur5e-1 breakdown": "after_M1_processing_before_pick",
    "Machining breakdown during part processing": "during_processing_halfway",
    "Part slippage": "during_place_lowering",
}


SUPPORTED_CHECKPOINTS = {scenario: (name,) for scenario, name in CHECKPOINTS.items()}
SUPPORTED_CHECKPOINTS["Part slippage"] = (
    "during_place_lowering", "after_both_pickups_before_place",
)


def acknowledged_pickup(context, resource_id: str, part: str) -> dict | None:
    """Require the latest robot acknowledgement to establish its current custody."""
    actor = context.resources[resource_id]
    if (actor.valuation.get("held_part") != part
            or actor.valuation.get("resource_state") != "picked"
            or context.part_tracker.get(part, {}).get("location") != resource_id):
        return None
    for record in reversed(context.transitions):
        task = record["acknowledgement"]
        if task["resource_id"] != resource_id:
            continue
        if (task.get("run_id") != context.run_id
                or task["event_name"] != "pick_grasp" or task.get("evidence") != "resource"
                or task["parameters"].get("part_name") != part
                or not record.get("observations")):
            return None
        return record
    return None


def _slippage_pickups(context, configuration: dict) -> dict | None:
    other = configuration["additional_condition"]
    records = {}
    for robot, part in ((configuration["resource_id"], configuration["part_name"]),
                        (other["resource_id"], other["part_name"])):
        record = acknowledged_pickup(context, robot, part)
        if record is None:
            return None
        records[robot] = deepcopy(record)
    return {"part_name": configuration["part_name"], "custodian": configuration["resource_id"],
            "other_custodian": other["resource_id"], "other_part_name": other["part_name"],
            "pickups": records}


def placement_checkpoint(context, configuration: dict, task: dict) -> dict | None:
    """Require both pickups and the selected, still-unacknowledged placement task.

    Args:
        context: Current run's observed state and pending tasks.
        configuration: Selected Part slippage configuration.
        task: Placement task still pending, or retained after Stop cancelled it.

    Returns:
        Observed pickup custody, without asserting a placement interruption.
    """
    if (configuration.get("scenario") != "Part slippage"
            or configuration.get("checkpoint") != "during_place_lowering"
            or task.get("run_id") != context.run_id
            or task.get("resource_id") != configuration["resource_id"]
            or task.get("event_name") != "place_approach"
            or task.get("parameters", {}).get("part_name") != configuration["part_name"]):
        return None
    evidence = _slippage_pickups(context, configuration)
    if evidence is None or any(
        pending.get("resource_id") in evidence["pickups"] and pending != task
        for pending in context.pending_tasks.values()
    ):
        return None
    return evidence


def checkpoint(context, configuration: dict) -> dict | None:
    """Return observed fault evidence without projecting a successful task."""
    scenario = configuration["scenario"]
    if scenario == "Conveyor breakdown":
        part = context.resources["ur5e-1"].valuation.get("held_part")
        record = acknowledged_pickup(context, "ur5e-1", part) if part else None
        if record is None:
            return None
        task = record["acknowledgement"]
        machine = record["before"].get("M1", {})
        if (task["parameters"].get("origin_resource_location") != "M1"
                or machine.get("resource_state") != "completed"
                or machine.get("part_name") != part):
            return None
        return {"task_id": task["task_id"], "part_name": part, "custodian": "ur5e-1",
                "source": "M1", "pickup_observations": deepcopy(record["observations"])}
    if scenario == "ur5e-1 breakdown":
        machine = context.resources["M1"].valuation
        part = machine.get("part_name")
        if (not part or machine.get("resource_state") != "completed"
                or context.part_tracker.get(part, {}).get("location") != "M1"):
            return None
        for record in reversed(context.transitions):
            task = record["acknowledgement"]
            if task["resource_id"] != "M1":
                continue
            if (task.get("run_id") == context.run_id
                    and task["event_name"] == "machine_part" and task.get("evidence") == "resource"
                    and task["parameters"].get("part_name") == part and record.get("observations")):
                return {"task_id": task["task_id"], "part_name": part, "custodian": "M1",
                        "processing_observations": deepcopy(record["observations"])}
            return None
    if scenario == "Part slippage" and configuration.get("checkpoint") == "after_both_pickups_before_place":
        evidence = _slippage_pickups(context, configuration)
        if evidence is None or any(
            task["resource_id"] in evidence["pickups"] for task in context.pending_tasks.values()
        ):
            return None
        return evidence
    return None


def holds_task(context, configuration: dict | None, task: dict) -> bool:
    """Hold selected custody while allowing the configured placement to lower."""
    if not configuration or configuration["scenario"] != "Part slippage":
        return False
    other = configuration["additional_condition"]
    parts = {configuration["resource_id"]: configuration["part_name"],
             other["resource_id"]: other["part_name"]}
    rid = task["resource_id"]
    if rid not in parts or acknowledged_pickup(context, rid, parts[rid]) is None:
        return False
    if configuration.get("checkpoint") == "during_place_lowering" and rid == configuration["resource_id"]:
        return (task.get("event_name") != "place_approach"
                or task.get("parameters", {}).get("part_name") != parts[rid]
                or acknowledged_pickup(context, other["resource_id"], other["part_name"]) is None)
    return True
