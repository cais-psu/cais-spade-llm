"""Observed checkpoints for the four simulation failure scenarios."""

from __future__ import annotations

from copy import deepcopy

CHECKPOINTS = {
    "Conveyor breakdown": "after_M1_pick_before_release",
    "ur5e-1 breakdown": "after_M1_processing_before_pick",
    "Machining breakdown during part processing": "during_processing_halfway",
    "Part slippage": "after_both_pickups_before_place",
}


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


def checkpoint(context, configuration: dict) -> dict | None:
    """Return observed fault evidence without projecting a successful task."""
    scenario = configuration["scenario"]
    rid = configuration["resource_id"]
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
    if scenario == "Part slippage":
        other = configuration["additional_condition"]
        records = {}
        for robot, part in ((rid, configuration["part_name"]),
                            (other["resource_id"], other["part_name"])):
            record = acknowledged_pickup(context, robot, part)
            if record is None:
                return None
            records[robot] = deepcopy(record)
        if any(task["resource_id"] in records for task in context.pending_tasks.values()):
            return None
        return {"part_name": configuration["part_name"], "custodian": rid,
                "other_custodian": other["resource_id"], "other_part_name": other["part_name"],
                "pickups": records}
    return None


def holds_task(context, configuration: dict | None, task: dict) -> bool:
    """Hold selected pickup states until the joint checkpoint exists."""
    if not configuration or configuration["scenario"] != "Part slippage":
        return False
    other = configuration["additional_condition"]
    parts = {configuration["resource_id"]: configuration["part_name"],
             other["resource_id"]: other["part_name"]}
    rid = task["resource_id"]
    return rid in parts and acknowledged_pickup(context, rid, parts[rid]) is not None
