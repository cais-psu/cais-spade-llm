from __future__ import annotations

"""Observed checkpoints for the four simulation failure scenarios."""

from copy import deepcopy

CHECKPOINTS = {
    "Conveyor breakdown": "after_M1_pick_before_release",
    "ur5e-1 breakdown": "after_M1_processing_before_pick",
    "Machining breakdown during part processing": "during_processing_halfway",
    "Part slippage": "during_place_lowering",
}


SUPPORTED_CHECKPOINTS = {scenario: (name,) for scenario, name in CHECKPOINTS.items()}
SUPPORTED_CHECKPOINTS["Machining breakdown during part processing"] = (
    "during_processing_halfway", "during_processing",
)
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
    """Return observed fault evidence without projecting a successful task.

    Args:
        context: Current run's resource state and acknowledgements.
        configuration: Exact selected failure and any additional pickup.

    Returns:
        Recorded checkpoint evidence, or None while required observations are missing.
    """
    scenario = configuration["scenario"]
    if scenario == "Conveyor breakdown":
        part = context.resources["ur5e-1"].valuation.get("held_part")
        if configuration.get("part_name") is not None and part != configuration["part_name"]:
            return None
        record = acknowledged_pickup(context, "ur5e-1", part) if part else None
        if record is None:
            return None
        task = record["acknowledgement"]
        machine = record["before"].get("M1", {})
        if (task["parameters"].get("origin_resource_location") != "M1"
                or machine.get("resource_state") != "completed"
                or machine.get("part_name") != part):
            return None
        evidence = {"task_id": task["task_id"], "part_name": part, "custodian": "ur5e-1",
                    "source": "M1", "pickup_observations": deepcopy(record["observations"])}
        other = configuration.get("additional_condition")
        if other is not None:
            peer = acknowledged_pickup(context, other["resource_id"], other["part_name"])
            if (peer is None or peer["acknowledgement"]["parameters"].get("origin_resource_location")
                    != "3D Printing Station"):
                return None
            if any(pending.get("resource_id") in {"ur5e-1", other["resource_id"]}
                   for pending in context.pending_tasks.values()):
                return None
            evidence.update(other_custodian=other["resource_id"], other_part_name=other["part_name"],
                            pickups={"ur5e-1": deepcopy(record), other["resource_id"]: deepcopy(peer)})
        return evidence
    if scenario == "ur5e-1 breakdown":
        machine = context.resources["M1"].valuation
        robot = context.resources["ur5e-1"].valuation
        part = machine.get("part_name")
        if (not part or machine.get("resource_state") != "completed"
                or context.part_tracker.get(part, {}).get("location") != "M1"
                or robot.get("resource_state") != "at_pick" or robot.get("held_part") is not None
                or any(task.get("resource_id") == "ur5e-1" for task in context.pending_tasks.values())):
            return None
        records = {}
        for rid, name in (("M1", "machine_part"), ("ur5e-1", "pick_approach")):
            record = next((row for row in reversed(context.transitions)
                           if row["acknowledgement"]["resource_id"] == rid), None)
            task = record["acknowledgement"] if record is not None else {}
            if (task.get("run_id") != context.run_id or task.get("event_name") != name
                    or task.get("evidence") != "resource"
                    or task.get("parameters", {}).get("part_name") != part
                    or not record.get("observations")):
                return None
            records[name] = record
        approach = records["pick_approach"]
        if (approach["acknowledgement"]["parameters"].get("origin_resource_location") != "M1"
                or approach.get("before", {}).get("M1", {}).get("resource_state") != "completed"
                or approach["before"]["M1"].get("part_name") != part):
            return None
        return {"task_id": approach["acknowledgement"]["task_id"], "part_name": part, "custodian": "M1",
                "processing_observations": deepcopy(records["machine_part"]["observations"]),
                "machine_part": deepcopy(records["machine_part"]), "pick_approach": deepcopy(approach)}
    if scenario == "Part slippage" and configuration.get("checkpoint") == "after_both_pickups_before_place":
        evidence = _slippage_pickups(context, configuration)
        if evidence is None or any(
            task["resource_id"] in evidence["pickups"] for task in context.pending_tasks.values()
        ):
            return None
        return evidence
    return None


def holds_task(context, configuration: dict | None, task: dict) -> bool:
    """Hold selected custody while allowing the configured placement to lower.

    Args:
        context: Current run's observed resource state and acknowledgements.
        configuration: Selected failure, or None when injection is disabled.
        task: Nominal task being considered for dispatch.

    Returns:
        Whether the armed checkpoint must retain the task's resource custody.
    """
    if (configuration and configuration["scenario"] == "Conveyor breakdown"
            and configuration.get("additional_condition") is not None):
        other = configuration["additional_condition"]
        parts = {"ur5e-1": configuration["part_name"], other["resource_id"]: other["part_name"]}
        rid = task["resource_id"]
        return rid in parts and acknowledged_pickup(context, rid, parts[rid]) is not None
    if (configuration and configuration["scenario"] == "Machining breakdown during part processing"
            and configuration.get("checkpoint") == "during_processing" and task["event_name"] == "machine_part"):
        machines = context.inputs["scene"]["machines"]
        return any(context.resources[row["resource_id"]].valuation.get("resource_state") != "loaded"
                   or context.resources[row["handling_robot"]].valuation.get("held_part") is not None
                   for row in machines)
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
