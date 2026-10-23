from __future__ import annotations

"""Tentative symbolic bookkeeping for the existing multi-turn diagnostic mode."""

import math
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)

_CONDITION_FIELDS = {
    ("resource", "current_state"): "resource_state", ("resource", "location"): "resource_location",
    ("resource", "held_part"): "held_part", ("part", "location"): "part_location",
}
# How the prompt and the lookups show a value an accepted event changed without stating it.
NOT_STATED = "not stated"


def _task_nodes(prepared: dict) -> dict:
    return {node["id"]: node for node in prepared.get("saved_pending_nodes", []) if node.get("id")}


def _suffixes(prepared: dict, role: str) -> list[dict]:
    return [row for row in prepared.get("marked_reentry_context", {}).get("pending_suffixes", [])
            if row.get("role") == role]


def replaced_tasks(prepared: dict) -> set[str]:
    """Read replacement obligations from the captured handoff, without a route."""
    return {identifier for row in _suffixes(prepared, "recovery_replaced") for identifier in row.get("pending_task_ids", [])}


def llm_selects_next_event(session: dict) -> bool:
    """Return whether the LLM chooses each event: always in a draft, which no validator checks."""
    return session.get("outline_diagnostic_bypass") is True


def recorded_conditions(prepared: dict, key: str = "marked_reentry_conditions") -> list[dict]:
    """Read the handoff conditions recorded with the failure capture."""
    return [row for row in prepared.get("marked_reentry_context", {}).get(key, []) if isinstance(row, dict)]


def _condition_id(row: dict) -> str:
    return "handoff_condition:" + "|".join(str(row.get(key)) for key in ("role", "entity_kind", "entity", "field"))


def condition_holds(row: dict, session: dict, prepared: dict) -> bool:
    """Compare one recorded handoff condition with the tentative symbolic state.

    Args:
        row: Recorded condition with entity_kind, entity, field and expected value.
        session: Tentative planning state.
        prepared: Recorded failure context.

    Returns:
        Whether the projected field exactly equals the recorded expected value.
    """
    from . import multi_turn_outline_generation as comparison

    resources, parts = comparison._projected_outline_validation_context(
        session_state=session, prepared_recovery_request=prepared)
    kind = row.get("entity_kind")
    state = (resources if kind == "resource" else parts).get(row.get("entity"), {})
    if kind == "part" and row.get("field") == "state":
        # Older captures record a task contract's part label; a part stores none, so it is checked by
        # who holds the part, as capture preparation now does.
        from ..recovery_session import _contract_part_state_holds

        holder = next((jid for jid, values in resources.items() if values.get("held_part") == row.get("entity")), "")
        held = row.get("entity") if holder else ""
        return bool(state) and _contract_part_state_holds(str(row.get("expected") or ""), str(row.get("entity")), held)
    name = _CONDITION_FIELDS.get((kind, row.get("field")), row.get("field"))
    if not state or name in session.get("diagnostic_unknown_owner_fields", {}).get(row.get("entity"), []):
        return False
    return comparison._exact_state_field_value(state, name) == row.get("expected")


def _inside(pose: dict, bounds: list) -> bool:
    """Return whether a pose lies in region bounds on every axis the region bounds.

    A pose that does not state one of those axes, such as a planar (x, y, yaw) pose against a
    height bound, cannot be said to lie inside. An axis the region leaves unbounded is not tested.
    """
    if not all(isinstance(pose.get(axis), (int, float)) for axis in "xy"):
        return False
    return all(isinstance(pose.get(axis), (int, float)) and low <= pose[axis] <= high
               for axis, (low, high) in zip("xyz", bounds, strict=False))


def prompt_context(session: dict, prepared: dict, fault: dict, *, outline: bool) -> dict:
    """Collect the recorded handoff facts that a diagnostic prompt states.

    Args:
        session: Tentative planning state.
        prepared: Recorded failure context.
        fault: The prompt's fault event.
        outline: Whether condition values are compared with the projected state.

    Returns:
        Handoff conditions (with ``holds`` on outline turns), compact pending nominal tasks,
        the interrupted event, and the
        diagnostic refs that have recorded data.
    """
    from .multi_turn_context import diagnostic_refs

    conditions = [{**{key: row.get(key) for key in ("role", "entity_kind", "entity", "field", "expected")},
                   **({"holds": condition_holds(row, session, prepared)} if outline else {})}
                  for row in entry_conditions(prepared)]
    roles = {identifier: suffix["role"] for role in ("recovery_replaced", "resume_suffix")
             for suffix in _suffixes(prepared, role)
             for identifier in [*suffix.get("pending_task_ids", []), suffix.get("entry_task_id")] if identifier}
    tasks = []
    for node in prepared.get("saved_pending_nodes", []):
        params = node.get("params", {})
        row = {"id": node.get("id"), "role": roles.get(node.get("id")),
               "resource_jid": node.get("resource_jid") or prepared.get("observation_jids", {}).get(params.get("resource_id")),
               "function_name": node.get("function_name"),
               **{key: params[key] for key in ("part_name", "destination_location") if key in params}}
        tasks.append({key: value for key, value in row.items() if value is not None})
    interrupted = ({key: fault[key] for key in ("blocked_at_function", "focused_resource_jid")}
                   if fault.get("blocked_at_function") and fault.get("focused_resource_jid") else {})
    return {"conditions": conditions,
            "pending_tasks": tasks,
            "interrupted": interrupted, "refs": diagnostic_refs(prepared)}


def _envelope_relation(pose: dict, reach: dict) -> dict | None:
    """State where a pose lies against one declared envelope; the reader judges whether it fits."""
    origin = reach.get("origin_pose") or {}
    values = (pose.get("x"), pose.get("y"), pose.get("z"), origin.get("x"), origin.get("y"),
              reach.get("max_xy_radius_m"), reach.get("z_min_m"), reach.get("z_max_m"))
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    return {"xy_distance_m": round(math.hypot(pose["x"] - origin["x"], pose["y"] - origin["y"]), 3),
            "max_xy_radius_m": reach["max_xy_radius_m"], "z_m": round(pose["z"], 3),
            "z_range_m": [reach["z_min_m"], reach["z_max_m"]], "tolerance_m": reach.get("tolerance_m", 0.0)}


def pose_relations(pose: dict, regions: dict, locations: dict[str, set[str]], envelopes: dict[str, dict],
                   movers: set[str] = frozenset()) -> dict:
    """Relate one recorded pose to declared regions, locations and pose envelopes.

    Every resource is treated alike: it relates through the locations or the envelope
    its declarations record. One whose declarations do not bound the pose is reported as
    not evaluated, including a resource whose events move it to coordinates, since its
    reach follows where it goes. Nothing here selects a resource or judges reach.

    Args:
        pose: World-frame position with x, y and z.
        regions: Region name to its recorded record with ``bounds``.
        locations: Every resource JID to its declared location tokens, possibly empty.
        envelopes: Resource JID to its declared ``gripper_reach``.
        movers: Resources with a declared event that moves them to coordinates.

    Returns:
        Containing regions, resources declaring a location that owns one of them, the pose's
        distance and height beside every declared envelope, and resources not evaluated.
    """
    from cais_spade_llm.recovery_framework.failure_grounding import _regions_for

    containing = {name: row["bounds"] for name, row in regions.items()
                  if isinstance(row, dict) and row.get("bounds") and _inside(pose, row["bounds"])}
    by_location = {jid: sorted(name for name in names if _regions_for(name, containing))
                   for jid, names in sorted(locations.items())}
    located = {jid: names for jid, names in by_location.items() if names}
    return {"containing_regions": sorted(containing), "by_declared_location": located,
            "envelopes": {jid: relation for jid, reach in sorted(envelopes.items())
                          if (relation := _envelope_relation(pose, reach)) is not None},
            "not_evaluated": sorted(jid for jid, names in locations.items()
                                    if jid not in envelopes and jid not in located and (not names or jid in movers))}


def _declared_event(model: dict, node: dict) -> dict | None:
    """Return the declared event a recorded task runs, matching its fixed parameter bindings."""
    params = node.get("params", {})
    return next((event for event in model.get("events", []) if event.get("event_name") == node.get("function_name")
                 and not any(name in params and "equals" in binding and params[name] != binding["equals"]
                             for name, binding in event.get("parameter_bindings", {}).items() if isinstance(binding, dict))),
                None)


def _task_part(node: dict, prepared: dict) -> str | None:
    """Return the part a recorded task handles: its ``part_name``, else a known part another parameter names.

    Parameters naming places are skipped, since a place may share a part's name.
    """
    params = node.get("params", {})
    if params.get("part_name"):
        return params["part_name"]
    known = {row.get("part_name") for row in prepared.get("llm_input", {}).get("part_facts", [])} | {
        other.get("params", {}).get("part_name") for other in prepared.get("saved_pending_nodes", [])}
    for key, value in params.items():
        if key in ("resource_id", *_LOCATION_PARAMS):
            continue
        for item in value if isinstance(value, (dict, list)) else [value]:
            if isinstance(item, str) and item in known - {None}:
                return item
    return None


def resume_entries(prepared: dict) -> list[tuple[dict, str]]:
    """Return the resumed tasks nominal work restarts at, with their suffixes.

    A part needs to be ready only for its next remaining step: of the resumed entries that handle the
    same part, only the earliest in the recorded plan order counts, because nominal work brings the part
    to its later steps. Entries that handle no part all count.
    """
    nodes = _task_nodes(prepared)
    order = {node["id"]: index for index, node in enumerate(prepared.get("saved_pending_nodes", [])) if node.get("id")}
    entries = []
    for suffix in _suffixes(prepared, "resume_suffix"):
        identifier = suffix.get("entry_task_id") or next(iter(suffix.get("pending_task_ids", [])), None)
        if nodes.get(identifier):
            entries.append((suffix, identifier))
    parts = {identifier: _task_part(nodes[identifier], prepared) for _, identifier in entries}
    earliest: dict[str, str] = {}
    for _, identifier in entries:
        part = parts[identifier]
        if part and order.get(identifier, len(order)) < order.get(earliest.get(part), len(order) + 1):
            earliest[part] = identifier
    return [(suffix, identifier) for suffix, identifier in entries
            if earliest.get(parts[identifier], identifier) == identifier]


def entry_conditions(prepared: dict, key: str = "marked_reentry_conditions") -> list[dict]:
    """Return the recorded handoff conditions, keeping a resumed task's only when nominal work restarts there."""
    kept = {identifier for _, identifier in resume_entries(prepared)}
    return [row for row in recorded_conditions(prepared, key)
            if row.get("role") != "resume_suffix" or not row.get("source_task_id") or row["source_task_id"] in kept]


def nominal_rows(session: dict, prepared: dict) -> list[dict]:
    """Bind the resumed entries nominal work restarts at to recorded nominal declarations."""
    nodes = _task_nodes(prepared)
    rows = []
    for suffix, identifier in resume_entries(prepared):
        node = nodes[identifier]
        params = node.get("params", {})
        jid = suffix.get("resource_jid") or prepared.get("observation_jids", {}).get(params.get("resource_id"))
        event = _declared_event(session.get("recovery_des_models", {}).get(jid, {}), node)
        if event is not None:
            rows.append({"event_id": "nominal_entry:" + identifier,
                "task": {"task_id": identifier, "resource_jid": jid, "part_name": params.get("part_name"),
                         "function_name": node["function_name"], "params": deepcopy(params)},
                "ra_event": deepcopy(event), "tool": {}})
    return rows


def unbound_resume_entries(session: dict, prepared: dict) -> list[dict]:
    """State the resumed entries no declared event binds, whose continuation the handoff cannot check.

    nominal_rows leaves such an entry out, so without this note a handoff could read complete while a
    part's next nominal step was never checked.
    """
    nodes = _task_nodes(prepared)
    bound = {row["task"]["task_id"] for row in nominal_rows(session, prepared)}
    unbound = []
    for suffix, identifier in resume_entries(prepared):
        if identifier in bound:
            continue
        node = nodes[identifier]
        jid = suffix.get("resource_jid") or prepared.get("observation_jids", {}).get(node.get("params", {}).get("resource_id"))
        unbound.append({"task_id": identifier, "reason": f"no declared event of {jid} binds the resumed "
                        f"{node['function_name']}; whether nominal work continues there is not checked"})
    return unbound


def _fixed(condition: Any, params: dict) -> Any:
    if not isinstance(condition, dict):
        return condition
    if "equals" in condition:
        return condition["equals"]
    if "set" in condition:
        return condition["set"]
    if "set_from_param" in condition:
        return params.get(condition["set_from_param"])
    return None


def _event_changes(task: dict, session: dict) -> list[dict]:
    """List the variables an event's end state sets, each on its resource or part.

    A recovery event's values are the LLM's. A nominal step's come from its declared model, so only
    the public variables it actually changes are listed.
    """
    variables = session.get("recovery_des_models", {}).get(task["resource_jid"], {}).get("state_variables", {})
    by_llm = task.get("candidate_source") != "nominal_connection"
    start = task.get("expected_start_state") or {}
    return [{"entity": task["part_name"] if variables.get(name, {}).get("scope") == "part" and task.get("part_name")
             else task["resource_jid"], "variable": name, "value": deepcopy(value), "by": "LLM" if by_llm else "model"}
            for name, value in task["expected_end_state"].items()
            if by_llm or (not name.startswith(("task_ctx.", "_")) and not variables.get(name, {}).get("private")
                          and value != start.get(name))]


def _place_of(pose: dict, prepared: dict) -> str | None:
    """Return the named place whose recorded area contains a pose, or None when none does."""
    from .multi_turn_outline_facts import _containing_regions, _declared_locations

    places = {place for entry in prepared.get("recovery_resources", {}).values() for place in _declared_locations(entry)}
    return next((region.split(".")[0] for region in _containing_regions(pose, prepared)
                 if region.split(".")[0] in places), None)


def _derived(entity: str, variable: str, value: Any, reason: str) -> dict:
    return {"entity": entity, "variable": variable, "value": deepcopy(value), "by": "code", "reason": reason}


def _locate_by_pose(task: dict, session: dict, prepared: dict) -> list[dict]:
    """Keep a resource's projected location consistent with the pose its event states."""
    pose = task["expected_end_state"].get("pose")
    if not isinstance(pose, dict):
        return []
    # A stated pose is where the resource is now; position checks read it as its pose.
    row = session.setdefault("symbolic_resources", {}).setdefault(task["resource_jid"], {})
    row.update(current_pose=deepcopy(pose), current_pose_source="projected_state")
    if "resource_location" in task["expected_end_state"]:
        return []
    place = _place_of(pose, prepared)
    if row.get("resource_location") == place:
        return []
    row.update(resource_location=place, current_location=place)
    row.setdefault("occupancy", {})["location"] = place
    return [_derived(task["resource_jid"], "resource_location", place,
                     f"the stated pose is inside {place}" if place else "no named place's recorded area contains the stated pose")]


def _apply_neighbor_end_state(task: dict, session: dict) -> list[dict]:
    """Apply the changes the event states for other resources its part's move affects.

    A recovery event's values are the LLM's; a nominal step's come from those resources' own copies of it.
    """
    from . import multi_turn

    by = "model" if task.get("candidate_source") == "nominal_connection" else "LLM"
    changes, owner_unknown = [], session.setdefault("diagnostic_unknown_owner_fields", {})
    part = session.get("symbolic_parts", {}).get(task.get("part_name"), {})
    for jid, values in (task.get("neighbor_end_state") or {}).items():
        multi_turn._apply_task_effects_to_symbolic_state({"resource_jid": jid, "expected_end_state": values}, session)
        # A stated holding change on that resource is also the part's: who holds it must agree on both sides.
        if "held_part" in values and part:
            holds = values["held_part"] == task["part_name"]
            for key in ("part_holder_resource_jid", "current_holder_resource_jid"):
                if holds or part.get(key) == jid:
                    part[key] = jid if holds else None
        if jid in owner_unknown:
            owner_unknown[jid] = [name for name in owner_unknown[jid] if name not in values]
        changes += [{"entity": jid, "variable": name, "value": deepcopy(value), "by": by}
                    for name, value in values.items()]
    return changes


def _release_holders(task: dict, session: dict, prepared: dict, after: Any) -> list[dict]:
    """Clear the holding of a part the event places elsewhere, on its former holder and on the part.

    A resource the event states as taking the part (in its own end state, or in neighbor_end_state for
    another resource) is its new holder, whatever location the end state gives the part. Code never
    overrides a held_part the event states, so only holders the event does not mention are released.
    """
    from .multi_turn_outline_facts import resource_names

    part_name, names, derived = task["part_name"], resource_names(prepared), []
    stated = {task["resource_jid"]: task["expected_end_state"], **(task.get("neighbor_end_state") or {})}
    takers = sorted(jid for jid, values in stated.items() if values.get("held_part") == part_name)
    reason = (f"{part_name} is now held by {', '.join(names.get(jid, jid) for jid in takers)}" if takers
              else f"{part_name} was placed elsewhere")
    # A part the event places elsewhere is no longer held, even when the end state omits held_part:
    # neither its former holder nor the part itself may still record the holding.
    for jid, row in session.get("symbolic_resources", {}).items():
        if (row.get("held_part") == part_name and after not in {jid, names.get(jid)}
                and "held_part" not in stated.get(jid, {})):
            row["held_part"] = None
            derived.append(_derived(jid, "held_part", None, reason))
    part_row = session.get("symbolic_parts", {}).get(part_name, {})
    holders = [key for key in ("part_holder_resource_jid", "current_holder_resource_jid")
               if part_row.get(key) and part_row[key] not in takers
               and after not in {part_row[key], names.get(part_row[key])}]
    for key in holders:
        part_row[key] = None
    if holders:
        derived.append(_derived(part_name, "held_by", None, "it was placed elsewhere"))
    return derived


def _clear_pose_of_unheld_part(task: dict, session: dict) -> None:
    """Keep the stated pose on the event's part only while the event's resource holds that part.

    A stated pose is the position of the event's resource itself. A part that resource does not hold
    after the event is at a position no event states, so its projected pose is unknown.
    """
    part = session.get("symbolic_parts", {}).get(task.get("part_name"))
    held = session.get("symbolic_resources", {}).get(task["resource_jid"], {}).get("held_part")
    if part and "pose" in task["expected_end_state"] and held != task["part_name"]:
        part["projected_pose"] = None


def machine_jobs(jid: str, session: dict, prepared: dict) -> list[dict]:
    """Return a resource's job records as the accepted events leave them, oldest first.

    They start from the jobs the resource recorded at the failure. A job stays recorded when its part
    leaves, so a paused job that resumes after reload is still listed while another part is processed.

    Args:
        jid: The resource.
        session: Tentative planning state.
        prepared: Recorded failure context holding the recorded jobs.

    Returns:
        Each job's part, process, result, status and, while no event changed it, its recorded processing
        fraction.
    """
    if jid in session.get("diagnostic_machine_jobs", {}):
        return session["diagnostic_machine_jobs"][jid]
    observations = (prepared.get("recovery_resources", {}).get(jid, {}).get("static_capabilities") or {}).get(
        "processing_observations") or {}
    return [{"part_name": row["part_name"], "process": row.get("process"), "result": row.get("result"),
             "status": row.get("job_state"), "processing_fraction": row.get("processing_fraction")}
            for row in (observations.get("jobs") or {}).values() if row.get("part_name")]


def _finish_completed_process(task: dict, session: dict) -> list[dict]:
    """End the part's unfinished process when the event states the ledger entry that completes it.

    A process cannot be both unfinished and completed; the model's completion event clears it the same
    way. An event that states the part's current process itself is applied as stated.
    """
    part_name = task.get("part_name")
    part = session.get("symbolic_parts", {}).get(part_name, {})
    current = part.get("current_process")
    if not isinstance(current, dict) or "current_process" in task["expected_end_state"]:
        return []
    done = {key: current.get(key) for key in ("process", "result") if current.get(key) is not None}
    if done not in (part.get("processCompleted") or []):
        return []
    part["current_process"] = None
    return [_derived(part_name, "current_process", None, f"the stated processCompleted completes {done}")]


def _project_machine_jobs(task: dict, session: dict, prepared: dict) -> None:
    """Keep the machine's job record of the event's part in step with the part's stated current process.

    The part carries its unfinished process with the machine that holds the job, so a stated status
    changes that machine's job of the part and leaves its other jobs as they are. A stated completion
    (``current_process`` none) leaves the records as they are.
    """
    part = task.get("part_name")
    process = session.get("symbolic_parts", {}).get(part, {}).get("current_process")
    if not part or "current_process" not in task["expected_end_state"] or not isinstance(process, dict):
        return
    jid = process.get("machine") or task["resource_jid"]
    jobs = session.setdefault("diagnostic_machine_jobs", {}).setdefault(jid, deepcopy(machine_jobs(jid, session, prepared)))
    job = next((row for row in jobs if row["part_name"] == part), None)
    if job is None:
        job = {"part_name": part}
        jobs.append(job)
    job.update({key: deepcopy(process[key]) for key in ("process", "result") if process.get(key) is not None})
    job.update(status=deepcopy(process.get("status")), processing_fraction=None)


def _place_records(location: Any, part_name: str, session: dict, prepared: dict) -> list[tuple[str, list[str], bool]]:
    """Return the resources that record parts at a location, each with the variables that may record this part.

    A location that is itself a resource records them in its state and in each variable that can hold
    the part: its declared values name the part, it records the part now, or it is the part's own entry
    such as ``part_location.<part>``. A location inside a resource, such as a machine's staging tray, is
    recorded only by the owner variables its declared events set there. The flag tells the two apart.
    """
    from .multi_turn_outline_facts import declared_record_places

    jid = prepared.get("observation_jids", {}).get(location)
    if jid:
        variables = session.get("recovery_des_models", {}).get(jid, {}).get("state_variables", {})
        resource = session.get("symbolic_resources", {}).get(jid, {})
        return [(jid, [name for name, row in variables.items()
                       if not row.get("private") and row.get("scope") != "part" and not name.startswith(("task_ctx.", "_"))
                       and (name == "resource_state" or part_name in (row.get("domain") or [])
                            or resource.get(name) == part_name or name.endswith(f".{part_name}"))], True)]
    records = []
    for owner in prepared.get("recovery_resources", {}):
        names = [name for name, place in declared_record_places(owner, prepared).items() if place == location]
        if names:
            records.append((owner, names, False))
    return records


def _mark_places_unknown(task: dict, session: dict, prepared: dict, before: Any, after: Any) -> list[dict]:
    """Mark unknown what a place the part left or entered records about it, unless the event states it."""
    part_name, derived, stated = task.get("part_name"), [], task.get("neighbor_end_state") or {}
    owner_unknown = session.setdefault("diagnostic_unknown_owner_fields", {})
    for location in (before, after):
        for jid, names, whole_place in _place_records(location, part_name, session, prepared):
            if jid == task["resource_jid"]:
                continue
            # The place's own record changes. When the event states that change, its other values stay
            # as they are; otherwise each variable that may record the part becomes unknown rather than
            # guessed. A place inside a resource leaves the rest of its owner's state as it is.
            unknown = [] if jid in stated else names
            cleared = ([*unknown, *(["current_state"] if "resource_state" in unknown else []), "active_job"]
                       if whole_place else unknown)
            resource = session.get("symbolic_resources", {}).get(jid, {})
            for field in cleared:
                resource.pop(field, None)
            owner_unknown[jid] = list(dict.fromkeys([*owner_unknown.get(jid, []), *unknown,
                                                     *(["active_job"] if whole_place else [])]))
            moved = "arrived there" if location == after else "left"
            derived += [_derived(jid, name, "unknown", f"{part_name} {moved}; its own change was not stated")
                        for name in unknown]
    return derived


def _carry_owner_bookkeeping(task: dict, session: dict) -> None:
    """Carry an owner's fixed private bookkeeping when the end state matches a declared event's public effects."""
    model = session.get("recovery_des_models", {}).get(task["resource_jid"], {})
    end = task["expected_end_state"]
    # Owner-managed fixed bookkeeping can be carried as a tentative consequence
    # of an authored public outcome, without backward/forward or motion checks.
    for event in model.get("events", []):
        updates = event.get("updates", {})
        public = {key: value for key, value in updates.items()
                  if key in end and key != "active_job" and isinstance(value, dict) and "set" in value}
        if public and all(end[key] == value["set"] for key, value in public.items()):
            private = updates.get("active_job", {})
            if "set" in private:
                session["symbolic_resources"][task["resource_jid"]]["active_job"] = deepcopy(private["set"])


def _completed_effects(event: dict, params: dict) -> list[dict]:
    """Return the ledger entries an event appends to its part's processCompleted, resolved with params."""
    return [{name: _fixed(value, params) for name, value in row.items()}
            for row in (event.get("product_effects") or {}).get("processCompleted") or [] if isinstance(row, dict)]


def _meets(end: dict, name: str, value: Any) -> bool:
    """Compare one outcome value; a ledger outcome holds when the stated record contains its entries."""
    if name == "processCompleted":
        return isinstance(end.get(name), list) and all(row in end[name] for row in value)
    return end.get(name) == value


def _declared_outcome(event: dict, params: dict, variables: dict) -> dict:
    """Return the fixed outcome a declared event gives a recorded task: its part's, else its fixed effects."""
    # The part's declared outcome decides, so a resource may reach it under a new state name: where the
    # part ends up and the processes it completes. Work that declares no part outcome is compared on its
    # fixed declared effects.
    outcome = {name: _fixed(value, params) for name, value in event.get("updates", {}).items()
               if variables.get(name, {}).get("scope") == "part" and not variables[name].get("private")
               and _fixed(value, params) is not None}
    if completed := _completed_effects(event, params):
        outcome["processCompleted"] = completed
    return outcome or {name: value["set"] for name, value in event.get("updates", {}).items()
                       if name != "active_job" and isinstance(value, dict) and "set" in value}


def _met_event(model: dict, node: dict, end: dict) -> dict | None:
    """Return the declared event of an interrupted task whose outcome the end state meets, if any."""
    params, variables = node.get("params", {}), model.get("state_variables", {})
    for event in model.get("events", []):
        if event.get("event_name") != node.get("function_name"):
            continue
        bindings = event.get("parameter_bindings", {})
        if any(name in bindings and _fixed(bindings[name], params) != params[name]
               for name in ("part_name", "process", "result") if name in params):
            continue
        outcome = _declared_outcome(event, params, variables)
        if outcome and all(_meets(end, name, value) for name, value in outcome.items()):
            return event
    return None


def _claim_interrupted_work(task: dict, session: dict, prepared: dict) -> None:
    """Claim interrupted work whose outcome the event's end state now meets.

    The outcome is the task's declared outcome, or, for a task its own model declares no event for,
    its part at the task's destination. Claims are kept for the handoff record; they are not values
    of any resource or part, so they are not listed among the event's condition changes.
    """
    part_name = task.get("part_name")
    model = session.get("recovery_des_models", {}).get(task["resource_jid"], {})
    nodes = _task_nodes(prepared)
    owners = {node.get("id"): jid for jid, node in _replaced_work(prepared)}
    claimed = session.get("diagnostic_replaced_task_claims", {})
    # Moved work states its outcome through the handover conditions, so it is never claimed.
    moved = {identifier for suffix in _suffixes(prepared, "recovery_replaced") if suffix.get(_MOVED)
             for identifier in suffix.get("pending_task_ids", [])}
    for identifier in replaced_tasks(prepared) - moved:
        node = nodes.get(identifier, {})
        params = node.get("params", {})
        if not node or params.get("part_name") != part_name or identifier in claimed:
            continue
        destination = destination_outcome(owners.get(identifier), node, prepared, session)
        # Without a declared event only the part resting at the destination completes the task, not
        # an acting resource's event that happens to share its name.
        event = None if destination else _met_event(model, node, task["expected_end_state"])
        if event is None and not (destination and _at_destination(part_name, destination["part_location"], session, prepared)):
            continue
        session.setdefault("diagnostic_replaced_task_claims", {})[identifier] = {
            "outline_id": task["outline_id"], "resource_jid": task["resource_jid"],
            "basis": ("unvalidated proposed outcome matching recorded task effects" if event
                      else "unvalidated proposed part location matching the recorded task destination")}
        if event is None:
            continue
        product = session.get("symbolic_parts", {}).get(part_name, {})
        for key, values in event.get("product_effects", {}).items():
            if not isinstance(values, list):
                continue
            claims = [{name: _fixed(value, params) for name, value in row.items()} for row in values]
            existing = product.setdefault(key, [])
            if isinstance(existing, list):
                existing.extend(row for row in claims if row not in existing)


def apply_tentative_effects(task: dict, session: dict, prepared: dict) -> None:
    """Apply an accepted event to the projected state and record who set each changed variable.

    The event's end state, and what it states in ``neighbor_end_state`` for other resources its part's
    move changes, are applied as written. Code then derives only consequences of them:
    - a stated pose without a location change sets the location to the named place containing it;
    - a part placed elsewhere loses its holder, unless the event states who takes it;
    - the event's part keeps the stated pose as its projected pose only while the event's resource
      holds it, since that pose is the resource's own;
    - a place a part leaves or enters gets unknown values for its state and its variables that can
      hold the part, unless the event states how that place changed;
    - a stated ledger entry that completes the part's unfinished process ends that process, as the
      model's own completion does, and a stated process status changes the job record of the machine
      that holds the part's job;
    - interrupted work whose outcome now holds (its declared outcome, or its part resting at the
      task's destination) is claimed.
    Every changed variable is recorded on the event in ``condition_changes`` with ``by`` set to
    ``LLM``, ``model`` (a nominal step) or ``code`` and, for code, the reason.
    """
    from . import multi_turn

    part_name = task.get("part_name")
    before = session.get("symbolic_parts", {}).get(part_name, {}).get("part_location")
    multi_turn._apply_task_effects_to_symbolic_state(task, session)
    changes = [*_event_changes(task, session), *_apply_neighbor_end_state(task, session),
               *_locate_by_pose(task, session, prepared)]
    changes += _finish_completed_process(task, session)
    _project_machine_jobs(task, session, prepared)
    unknown = session.get("diagnostic_unknown_owner_fields", {}).get(task["resource_jid"], [])
    session.setdefault("diagnostic_unknown_owner_fields", {})[task["resource_jid"]] = [
        name for name in unknown if name not in task["expected_end_state"]]
    after = session.get("symbolic_parts", {}).get(part_name, {}).get("part_location")
    if part_name and "part_location" in task["expected_end_state"]:
        changes += _release_holders(task, session, prepared, after)
    _clear_pose_of_unheld_part(task, session)
    if before != after:
        changes += _mark_places_unknown(task, session, prepared, before, after)
    _carry_owner_bookkeeping(task, session)
    _claim_interrupted_work(task, session, prepared)
    task["condition_changes"] = changes


def nominal_entry_ids(session: dict, prepared: dict) -> set[str]:
    """Compare captured entry conditions with tentative facts; geometry stays unknown."""
    from . import multi_turn_outline_generation as comparison

    resources, parts = comparison._projected_outline_validation_context(
        session_state=session, prepared_recovery_request=prepared)
    enabled = set()
    assumptions = []
    for row in comparison._nominal_reentry_event_rows(session_state=session, prepared_recovery_request=prepared):
        task = row.get("task", {})
        jid, part_name = task.get("resource_jid"), task.get("part_name")
        resource, part = resources.get(jid, {}), parts.get(part_name, {})
        event = row.get("ra_event", {})
        source = event.get("capability_transition", {}).get("source", {})
        # A part must actually be proposed at the recorded entry location. Unknown
        # controller fields cannot supply that missing public handoff condition.
        location = source.get("part_location")
        if part_name and location is not None and part.get("part_location") != location:
            continue
        params = task.get("params", {})
        matches = True
        variables = session.get("recovery_des_models", {}).get(jid, {}).get("state_variables", {})
        for name, guard in event.get("guards", {}).items():
            if name == "active_job" or name.startswith(("task_ctx.", "_")):
                assumptions.append({"task_id": task.get("task_id"), "field": name, "reason": "owner bookkeeping not checked"})
                continue
            state = part if variables.get(name, {}).get("scope") == "part" else resource
            if name not in state or name in session.get("diagnostic_unknown_owner_fields", {}).get(jid, []):
                assumptions.append({"task_id": task.get("task_id"), "field": name, "reason": "owner state unknown after tentative transfer"})
                continue
            if not comparison._recovery_event_guard_is_satisfied(event=event, condition=guard,
                    actual=comparison._exact_state_field_value(state, name), resource_jid=jid,
                    part_name=part_name or "", resource_row=resource, part_row=part, parameters=params):
                matches = False
                break
        # Another resource may declare its own half of the resumed task's joint event, such as where its part must be.
        neighbors = _neighbor_guards(jid, event, part_name or "", session, prepared, resources, part, params) if matches else None
        if neighbors is not None:
            assumptions += [{"task_id": task.get("task_id"), **note} for note in neighbors]
            enabled.add(row["event_id"])
    session["diagnostic_handoff_assumptions"] = assumptions
    return enabled


_MOVED = "resumed_work_needs_faulted_resource"


def _needs_faulted(params: dict, faulted: set[str], prepared: dict) -> bool:
    """Return whether a task names a faulted resource as its actor or as a place it uses."""
    jids = prepared.get("observation_jids", {})
    return any(jids.get(params.get(key), params.get(key)) in faulted for key in ("resource_id", *_LOCATION_PARAMS))


def faulted_replaced_work(session: dict, prepared: dict) -> set[str]:
    """Return the replaced tasks that need a resource still reported faulted.

    Normal production cannot run them, so only the recovery replaces them. Replaced tasks that need
    no faulted resource stay normal work, since nominal work may redo them (see _nominal_successors).

    Args:
        session: Tentative planning state, used to find the faulted resources.
        prepared: Recorded failure context holding the replaced work.

    Returns:
        The identifiers of those replaced tasks.
    """
    faulted = faulted_resources(session, prepared)
    return {node["id"] for _, node in _replaced_work(prepared)
            if _needs_faulted(node.get("params", {}), faulted, prepared)}


def resume_after_faults(prepared: dict, session: dict) -> list[str]:
    """Move resumed tasks that need a faulted resource into the replaced work.

    A failure blocks every task that needs the failed resource, as its actor or as a place the task
    uses, not only the tasks that resource performs. Each resumed suffix keeps its tasks from the
    first one that needs no faulted resource; the blocked ones become replaced work, and the recorded
    handoff conditions of a dropped entry are removed. Only each resource's own task list is read;
    dependencies across resources stay future work.

    Args:
        prepared: Recorded failure context, updated in place.
        session: Planning state at the failure, used to find faulted resources.

    Returns:
        The identifiers of the resumed tasks that became replaced work.
    """
    faulted = faulted_resources(session, prepared)
    context = prepared.get("marked_reentry_context") or {}
    nodes = _task_nodes(prepared)
    moved: list[str] = []
    suffixes = []
    for suffix in context.get("pending_suffixes", []):
        tasks = (list(suffix.get("pending_task_ids") or [suffix.get("entry_task_id")])
                 if suffix.get("role") == "resume_suffix" else [])
        blocked = next((index for index, identifier in enumerate(tasks)
                        if not _needs_faulted(nodes.get(identifier, {}).get("params", {}), faulted, prepared)), len(tasks))
        if not blocked:
            suffixes.append(suffix)
            continue
        moved += tasks[:blocked]
        suffixes.append({"role": "recovery_replaced", "resource_jid": suffix.get("resource_jid"),
                         "parts": deepcopy(suffix.get("parts", [])), "pending_task_ids": tasks[:blocked], _MOVED: True})
        if tasks[blocked:]:
            # The old entry's recorded requirements describe the blocked task, so they are not carried over.
            first = nodes.get(tasks[blocked], {})
            suffixes.append({**{key: value for key, value in suffix.items() if not key.startswith(("entry_", "required_"))},
                             "entry_task_id": tasks[blocked], "entry_function_name": first.get("function_name"),
                             "entry_part_name": first.get("params", {}).get("part_name"), "pending_task_ids": tasks[blocked:]})
    if moved:
        context["pending_suffixes"] = suffixes
        for key in ("marked_reentry_conditions", "unmet_reentry_conditions"):
            context[key] = [row for row in context.get(key, []) if row.get("source_task_id") not in moved]
    return moved


def declare_derived_conditions(prepared: dict, primitive_catalogs: dict[str, list]) -> dict[str, list[str]]:
    """Declare the condition variables a resource's declarations imply but its DES model lacks.

    A resource's conditions are the variables its model and its primitives declare (paper
    assumption A.2). Two are added by the same rule for every resource:
    - ``pose``: the position a primitive's declared ``base_pose`` effect sets (a resource that moves its
      base, which that pose then locates), else its declared ``current_pose`` effect, with that effect's
      fields; failing both, the free numeric inputs of a declared event that sets its location;
    - ``processCompleted``: a part's process record, on a resource with a declared event whose
      product effect adds to it.
    Each part's recorded process record is seeded into its part facts, so the projected state
    starts from it.

    Args:
        prepared: Recorded failure context, updated in place.
        primitive_catalogs: Each resource's recorded primitive declarations.

    Returns:
        The variables added per resource.
    """
    from .multi_turn_outline_facts import _coordinate_parameters

    added: dict[str, list[str]] = {}
    for jid, entry in prepared.get("recovery_resources", {}).items():
        model = entry.setdefault("recovery_des_model", {})
        variables = model.setdefault("state_variables", {})
        if "pose" not in variables:
            effects = [(effect, rule) for row in primitive_catalogs.get(jid) or [] if isinstance(row, dict)
                       for effect, rule in (row.get("effects") or {}).items() if isinstance(rule, dict)]
            fields, source, locates = None, "", False
            for name, locating in (("base_pose", True), ("current_pose", False)):
                fields = next((list(value) for effect, rule in effects if effect == name for value in rule.values()
                               if isinstance(value, list) and value and all(isinstance(field, str) for field in value)), None)
                if fields:
                    source, locates = f"the {name} effect of its primitives", locating
                    break
            if not fields:
                fields, source = _coordinate_parameters(model), "the coordinate inputs of its declared events"
                locates = bool(fields)
            if fields:
                variables["pose"] = {"scope": "resource", "type": "object", "fields": fields,
                                     "description": f"position ({', '.join(fields)}), from {source}"}
                if locates:
                    variables["pose"]["locates_resource"] = True
                added.setdefault(jid, []).append("pose")
        # Declared events default to an empty processCompleted effect, which leaves the record unchanged.
        if "processCompleted" not in variables and any(
                (event.get("product_effects") or {}).get("processCompleted") for event in model.get("events", [])):
            variables["processCompleted"] = {
                "scope": "part", "type": "array",
                "items": {"type": "object", "properties": {"process": {"type": "string"},
                                                           "result": {"type": ["string", "null"]}}},
                "description": "the part's process record: each process with its result, from declared product effects"}
            added.setdefault(jid, []).append("processCompleted")
    sources = (prepared.get("grounding_context", {}).get("parts", {}), prepared.get("part_tracker", {}))
    for row in prepared.get("llm_input", {}).get("part_facts", []):
        record = next((source.get(row.get("part_name"), {}).get("processCompleted") for source in sources
                       if isinstance(source.get(row.get("part_name"), {}).get("processCompleted"), list)), None)
        if "processCompleted" not in row and record is not None:
            row["processCompleted"] = deepcopy(record)
    return added


def handover_conditions(prepared: dict) -> list[dict]:
    """Return what the resumed work needs of the parts whose resumed tasks became replaced work.

    Moved work states no outcome of its own that recovery should reproduce (its recorded outcome
    needs the faulted resource), so it is complete when each of its parts is as the resumed work
    needs it at its next remaining step (see resume_entries): the recorded part conditions of that step,
    and its declared guards that name the part on its own resource, such as a station slot.
    """
    nodes = _task_nodes(prepared)
    parts = {part for suffix in _suffixes(prepared, "recovery_replaced") if suffix.get(_MOVED)
             for part in [*suffix.get("parts", []),
                          *(nodes.get(identifier, {}).get("params", {}).get("part_name")
                            for identifier in suffix.get("pending_task_ids", []))] if part}
    rows = [row for row in entry_conditions(prepared)
            if row.get("role") == "resume_suffix" and row.get("entity_kind") == "part" and row.get("entity") in parts]
    if not parts:
        return rows
    stated = {(row.get("entity"), row.get("field")) for row in entry_conditions(prepared)}
    for suffix, identifier in resume_entries(prepared):
        node = nodes[identifier]
        params = node.get("params", {})
        jid = suffix.get("resource_jid") or prepared.get("observation_jids", {}).get(params.get("resource_id"))
        model = prepared.get("recovery_resources", {}).get(jid, {}).get("recovery_des_model") or {}
        variables = model.get("state_variables", {})
        for field, guard in ((_declared_event(model, node) or {}).get("guards") or {}).items():
            expected = params.get(guard.get("equals_from_param")) if isinstance(guard, dict) else None
            if (expected in parts and variables.get(field, {}).get("scope", "resource") == "resource"
                    and (jid, field) not in stated):
                stated.add((jid, field))
                rows.append({"kind": "resume_entry_declared_guard", "role": "resume_suffix", "entity_kind": "resource",
                             "entity": jid, "field": field, "expected": expected, "source_task_id": identifier})
    return rows


def declared_part_effects(jid: str | None, task: dict, prepared: dict, session: dict) -> dict:
    """Return the part effects a task's declared event sets, resolved with the task's parameters."""
    params = task.get("params", {})
    model = (session.get("recovery_des_models", {}).get(jid)
             or prepared.get("recovery_resources", {}).get(jid, {}).get("recovery_des_model") or {})
    variables = model.get("state_variables", {})
    for event in model.get("events", []):
        bindings = event.get("parameter_bindings", {})
        if event.get("event_name") != task.get("function_name") or any(
                key in bindings and _fixed(bindings[key], params) != params[key]
                for key in ("part_name", "process", "result") if key in params):
            continue
        effects = {key: _fixed(value, params) for key, value in event.get("updates", {}).items()
                   if variables.get(key, {}).get("scope") == "part" and not variables[key].get("private")
                   and _fixed(value, params) is not None}
        if completed := _completed_effects(event, params):
            effects["processCompleted"] = completed
        return effects
    return {}


def destination_outcome(jid: str | None, task: dict, prepared: dict, session: dict) -> dict:
    """Return a task's part at the task's destination, when its resource's model declares no such event.

    A recorded task the model cannot describe still names where it takes its part, so that is its
    outcome: the goal states it and the handoff checks it, instead of assuming the task completes.
    """
    params = task.get("params", {})
    model = (session.get("recovery_des_models", {}).get(jid)
             or prepared.get("recovery_resources", {}).get(jid, {}).get("recovery_des_model") or {})
    destination, part = params.get("destination_location"), params.get("part_name")
    if (not isinstance(destination, str) or not isinstance(part, str)
            or any(event.get("event_name") == task.get("function_name") for event in model.get("events", []))):
        return {}
    return {"part_location": destination}


def _at_destination(part: str, destination: str, session: dict, prepared: dict) -> bool:
    """Return whether a part rests at a destination, named or by JID, with no resource holding it."""
    place = {destination, prepared.get("observation_jids", {}).get(destination, destination)}
    return (session.get("symbolic_parts", {}).get(part, {}).get("part_location") in place
            and not any(row.get("held_part") == part for row in session.get("symbolic_resources", {}).values()))


def _replaced_work(prepared: dict, *, include_moved: bool = True) -> list[tuple[str | None, dict]]:
    """Return each interrupted task still to be done, in recorded order, with its resource."""
    nodes = _task_nodes(prepared)
    rows = []
    for suffix in _suffixes(prepared, "recovery_replaced"):
        if suffix.get(_MOVED) and not include_moved:
            continue
        for identifier in suffix.get("pending_task_ids", []):
            node = nodes.get(identifier, {"id": identifier, "params": {}})
            jid = suffix.get("resource_jid") or prepared.get("observation_jids", {}).get(node.get("params", {}).get("resource_id"))
            rows.append((jid, node))
    return rows


def _replaced_finals(session: dict, prepared: dict) -> list[tuple[str, bool]]:
    """Return each replaced group's final task, and whether its outcome can be checked.

    Replaced tasks are grouped by part, or by resource for part-free work. Only each group's final
    task counts, because its outcome makes the intermediate owner states unnecessary. An outcome can
    be checked when the task's event declares effects, or, for a task its model declares no event for,
    by its part reaching the task's destination.
    """
    nodes = _task_nodes(prepared)
    finals: dict[tuple, tuple[str, str | None]] = {}
    # Moved work's recorded outcome needs the faulted resource; handover_conditions state its outcome.
    for jid, node in _replaced_work(prepared, include_moved=False):
        finals[(jid, node.get("params", {}).get("part_name"))] = (node["id"], jid)
    rows = []
    handover = bool(handover_conditions(prepared))
    for identifier, jid in finals.values():
        # With a handover, work that declares no part outcome is complete once the handover holds.
        if handover and not declared_part_effects(jid, nodes.get(identifier, {}), prepared, session):
            continue
        function = nodes.get(identifier, {}).get("function_name")
        declared = bool(destination_outcome(jid, nodes.get(identifier, {}), prepared, session)) or any(
            event.get("event_name") == function and any(
                isinstance(value, dict) and "set" in value and not name.startswith(("task_ctx.", "_"))
                and name != "active_job" for name, value in event.get("updates", {}).items())
            for event in session.get("recovery_des_models", {}).get(jid, {}).get("events", []))
        rows.append((identifier, declared))
    return rows


def open_replaced_work(session: dict, prepared: dict) -> tuple[set[str], list[dict]]:
    """Return unfinished replaced work that the recorded handoff conditions do not state.

    A final replaced task whose event declares effects must be claimed by an accepted event; one checked
    by its part's destination is open while the part does not rest there; one without either is
    reported as an assumption.

    Args:
        session: Tentative planning state with replaced-task claims.
        prepared: Recorded failure context.

    Returns:
        Open final task identifiers, and assumption rows for final tasks that can be checked neither way.
    """
    if any(row.get("role") == "recovery_replaced" for row in recorded_conditions(prepared)):
        return set(), []
    claims = set(session.get("diagnostic_replaced_task_claims", {}))
    nodes, owners = _task_nodes(prepared), {node.get("id"): jid for jid, node in _replaced_work(prepared)}
    finals = _replaced_finals(session, prepared)
    open_work = set()
    for identifier, declared in finals:
        node = nodes.get(identifier, {})
        destination = destination_outcome(owners.get(identifier), node, prepared, session) if declared else {}
        # A destination is a place the part must still rest at, so a later move away reopens the work.
        if destination and not _at_destination(node["params"]["part_name"], destination["part_location"], session, prepared) or declared and not destination and identifier not in claims:
            open_work.add(identifier)
    return (open_work, [{"task_id": identifier, "reason": "replaced work declares no effect; completion assumed"}
                        for identifier, declared in finals if not declared])


def goal_reached(session: dict, prepared: dict) -> bool | None:
    """Return whether the interrupted work's outcome holds, or None when nothing recorded can show it.

    Recorded replaced-work conditions decide when the capture has them; otherwise each final replaced
    task's outcome must hold: its declared effects claimed, or its part resting at the task's
    destination. Work that can be checked neither way cannot be checked.
    """
    replaced = [row for row in recorded_conditions(prepared) if row.get("role") == "recovery_replaced"]
    if replaced:
        return all(condition_holds(row, session, prepared) for row in replaced)
    handover = handover_conditions(prepared)
    if not handover and not any(declared for _, declared in _replaced_finals(session, prepared)):
        return None
    return (all(condition_holds(row, session, prepared) for row in handover)
            and not open_replaced_work(session, prepared)[0])


_LOCATION_PARAMS = ("destination_location", "origin_resource_location", "target_resource", "source_resource")
_CONNECTION_MAX_STEPS = 6


def faulted_resources(session: dict, prepared: dict) -> set[str]:
    """Return resources a grounded signal reports unavailable or in a fault state not yet left.

    The rule reads each signal's recorded values, not its kind, so any detection rule that reports a
    resource's availability or state counts the same way. A reported state is left only for another
    value the resource's model declares; a new recovery state name does not hand it back to nominal work.
    """
    from .multi_turn_outline_facts import _fault, _jid, _signal_values

    faulted = set()
    for signal in _fault(prepared)["signals"]:
        values, jid = _signal_values(signal), _jid(signal.get("resource_id"), prepared)
        row = session.get("symbolic_resources", {}).get(jid, {})
        reported, state = values.get("resource_state"), row.get("resource_state") or row.get("current_state")
        domain = (session.get("recovery_des_models", {}).get(jid, {}).get("state_variables", {})
                  .get("resource_state", {}).get("domain"))
        if (values.get("available") is False and row.get("availability") == "unavailable") or (
                reported and (state == reported or (isinstance(domain, list) and state not in domain))):
            faulted.add(jid)
    return faulted


def _faulted_resumed_work(session: dict, prepared: dict) -> list[str]:
    """State resumed nominal tasks that would need a resource still reported faulted."""
    faulted = faulted_resources(session, prepared)
    reasons = []
    for row in nominal_rows(session, prepared):
        task, jids = row["task"], prepared.get("observation_jids", {})
        if task["resource_jid"] in faulted:
            reasons.append(f"{task['resource_jid']} must do the resumed {task['function_name']} but is still reported faulted")
        for key in _LOCATION_PARAMS:
            if jids.get(task["params"].get(key)) in faulted:
                reasons.append(f"the resumed {task['function_name']} of {task['resource_jid']} uses "
                               f"{task['params'][key]}, which is still reported faulted")
    return reasons


def _branch(state: dict) -> dict:
    """Copy the fields a tentative event changes; declarations stay shared."""
    working = dict(state)
    for key in ("symbolic_resources", "symbolic_parts", "diagnostic_unknown_owner_fields",
                "diagnostic_replaced_task_claims", "diagnostic_machine_jobs"):
        working[key] = deepcopy(state.get(key, {}))
    working["accepted_outline_prefix"] = list(state.get("accepted_outline_prefix", []))
    return working


def _declaring_neighbors(actors: set[str], prepared: dict) -> set[str]:
    """Return the other resources whose own declared events name one of ``actors`` as their actor."""
    from .multi_turn_outline_facts import _environment_model_events, resource_names

    names = resource_names(prepared)
    acting = {names.get(jid, jid) for jid in actors}
    return {owner for owner, entry in prepared.get("recovery_resources", {}).items() if owner not in actors
            and any((event.get("parameter_bindings", {}).get("resource_id") or {}).get("equals") in acting
                    for event in _environment_model_events(entry))}


def _connection_scope(session: dict, prepared: dict) -> tuple[set[str], set[str]]:
    """Return the resources and parts named by what still blocks the handoff, minus faulted resources.

    A resource whose own declared events constrain an event of one of those resources joins them, so
    its own events can prepare that event, such as moving a part to the position it is picked from.
    """
    unmet = [row for row in entry_conditions(prepared) if not condition_holds(row, session, prepared)]
    actors = {row["entity"] for row in unmet if row.get("entity_kind") == "resource"}
    parts = {row["entity"] for row in unmet if row.get("entity_kind") == "part"}
    entries = nominal_entry_ids(dict(session), prepared)
    for row in nominal_rows(session, prepared):
        if row["event_id"] not in entries:
            actors.add(row["task"]["resource_jid"])
            parts.add(row["task"].get("part_name"))
    if goal_reached(session, prepared) is not True:
        # Nominal work may still finish the interrupted outcome itself.
        for jid, node in _replaced_work(prepared):
            actors.add(jid)
            parts.add(node.get("params", {}).get("part_name"))
    faulted = faulted_resources(session, prepared)
    actors = {jid for jid in actors if jid} - faulted
    return actors | (_declaring_neighbors(actors, prepared) - faulted), {name for name in parts if name}


def _public_guards_hold(event: dict, variables: dict, jid: str, part: str, resource_row: dict, part_row: dict,
                        params: dict | None = None) -> bool:
    """Check an event's public guards; private bookkeeping guards are the owner's, as in nominal_entry_ids."""
    from . import multi_turn_outline_generation as generation

    return all(name.startswith(("task_ctx.", "_")) or variables.get(name, {}).get("private")
               or generation._recovery_event_guard_is_satisfied(
                   event=event, condition=condition, resource_jid=jid, part_name=part,
                   resource_row=resource_row, part_row=part_row, parameters=params,
                   actual=generation._exact_state_field_value(
                       part_row if variables.get(name, {}).get("scope") == "part" else resource_row, name))
               for name, condition in event.get("guards", {}).items())


def _parameters_resolve(event: dict, jid: str, part: str, resource_row: dict, part_row: dict,
                        params: dict | None = None) -> bool:
    """Return whether every parameter an update reads has a value, so a nominal event acts only at declared places."""
    from . import multi_turn_outline_generation as generation

    def has_value(name: str) -> bool:
        found, value = generation._recovery_event_parameter_value(
            event=event, parameter_name=name, resource_jid=jid, part_name=part,
            resource_row=resource_row, part_row=part_row, parameters=params)
        return found and value is not None

    for update in event.get("updates", {}).values():
        if not isinstance(update, dict):
            continue
        names = [update["set_from_param"]] if update.get("set_from_param") else list(update.get("set_from_param_any_of") or [])
        if names and not any(has_value(name) for name in names):
            return False
    return True


def _bookkeeping_assumptions(event: dict, variables: dict, task: dict) -> list[dict]:
    """State the guards on a nominal step's own bookkeeping, which the connection search does not check.

    _public_guards_hold leaves them to the owner, as nominal_entry_ids does, so each is recorded
    rather than read as checked.
    """
    return [{"field": name, "reason": f"the nominal {task['event_name']} of {task['resource_jid']} needs its own "
             f"bookkeeping {name}"}
            for name in event.get("guards", {}) if name.startswith(("task_ctx.", "_")) or variables.get(name, {}).get("private")]


def _same_place_and_part(declared: dict, event: dict, jid: str, part: str, resource_row: dict, part_row: dict,
                         params: dict | None, names: dict[str, str]) -> bool:
    """Return whether another resource's declared event fixes the place and part ``event`` resolves to.

    Only the part and location parameters are compared, and only those ``event`` resolves.
    """
    from . import multi_turn_outline_generation as generation

    for name in ("part_name", *_LOCATION_PARAMS):
        binding = declared.get("parameter_bindings", {}).get(name)
        if not isinstance(binding, dict) or "equals" not in binding:
            continue
        found, value = generation._recovery_event_parameter_value(
            event=event, parameter_name=name, resource_jid=jid, part_name=part,
            resource_row=resource_row, part_row=part_row, parameters=params)
        if found and binding["equals"] != names.get(value, value):
            return False
    return True


def _owner_allows(declared: dict, owner: str, jid: str, part: str, state: dict, resources: dict,
                  part_row: dict, params: dict | None) -> list[dict] | None:
    """Check one declared event's guards on its owner's projected variables.

    Returns:
        The guards assumed (owner bookkeeping, undeclared, or left not stated by an accepted event),
        or None when a guard fails.
    """
    from . import multi_turn_outline_generation as generation

    variables = state.get("recovery_des_models", {}).get(owner, {}).get("state_variables", {})
    unstated = state.get("diagnostic_unknown_owner_fields", {}).get(owner, [])
    assumed = []
    for template, condition in declared.get("guards", {}).items():
        name = template.replace("{part_name}", part)
        if name not in variables or variables[name].get("private") or name.startswith(("task_ctx.", "_")) or name in unstated:
            assumed.append({"field": name, "resource_jid": owner, "reason": f"the nominal {declared['event_name']} of {jid} needs "
                            f"{name} of {owner}" + ("; an accepted event left it not stated" if name in unstated else "")})
        elif not generation._recovery_event_guard_is_satisfied(
                event=declared, condition=condition, resource_jid=jid, part_name=part,
                resource_row=resources.get(jid, {}), part_row=part_row, parameters=params,
                actual=generation._exact_state_field_value(resources.get(owner, {}), name)):
            return None
    return assumed


def _neighbor_rows(jid: str, event: dict, part: str, state: dict, prepared: dict, resources: dict,
                   part_row: dict, params: dict | None = None) -> list[tuple[str, dict, list[dict]]] | None:
    """Find other resources' own copies of one event of ``jid`` whose guards hold on the projected state.

    A resource's declared events may name another resource as the actor of a joint event, such as the
    position a part must be at to be picked there. Such an event constrains ``jid``'s event of the same
    name at the same place and part. For each resource that declares such events, the guards on its own
    variables must hold on the projected state for at least one of them; the one needing the fewest
    assumptions is chosen.

    Args:
        jid: The acting resource.
        event: Its declared event.
        part: The part the event handles, or "".
        state: Tentative planning state.
        prepared: Recorded failure context with each resource's declared events.
        resources: Projected resource rows.
        part_row: Projected row of ``part``.
        params: Recorded task parameters, when the event runs a recorded task.

    Returns:
        Each such resource with its chosen copy and the assumptions it needs, or None when another
        resource's declared events do not allow the event.
    """
    from .multi_turn_outline_facts import _environment_model_events, resource_names

    names = resource_names(prepared)
    chosen = []
    for owner, entry in prepared.get("recovery_resources", {}).items():
        declared = [row for row in _environment_model_events(entry)
                    if owner != jid and row.get("event_name") == event.get("event_name")
                    and (row.get("parameter_bindings", {}).get("resource_id") or {}).get("equals") == names.get(jid, jid)
                    and _same_place_and_part(row, event, jid, part, resources.get(jid, {}), part_row, params, names)]
        allowed = [(row, notes) for row in declared if (notes := _owner_allows(
            row, owner, jid, part, state, resources, part_row, params)) is not None]
        if declared and not allowed:
            return None
        if allowed:
            chosen.append((owner, *min(allowed, key=lambda item: len(item[1]))))
    return chosen


def _neighbor_guards(jid: str, event: dict, part: str, state: dict, prepared: dict, resources: dict,
                     part_row: dict, params: dict | None = None) -> list[dict] | None:
    """Return the assumptions other resources' copies of one event of ``jid`` need, or None when one forbids it."""
    chosen = _neighbor_rows(jid, event, part, state, prepared, resources, part_row, params)
    return None if chosen is None else [note for _, _, notes in chosen for note in notes]


def _neighbor_end_state(chosen: list[tuple[str, dict, list[dict]]], jid: str, event: dict, part: str,
                        resource_row: dict, part_row: dict, params: dict | None) -> dict:
    """Bind the updates other resources' own copies of a nominal step declare, keyed by resource.

    A joint event changes every participant, each by its own copy, so a nominal step changes those
    resources as their models declare.
    """
    from . import multi_turn_outline_generation as generation

    changes = {}
    for owner, row, _ in chosen:
        values = {}
        for template, update in row.get("updates", {}).items():
            if not isinstance(update, dict):
                continue
            if "set" in update:
                value = update["set"]
            elif "set_from_param" in update:
                found, value = generation._recovery_event_parameter_value(
                    event=event, parameter_name=update["set_from_param"], resource_jid=jid, part_name=part,
                    resource_row=resource_row, part_row=part_row, parameters=params)
                if not found:
                    continue
            else:
                continue
            values[template.replace("{part_name}", part)] = deepcopy(value)
        if values:
            changes[owner] = values
    return changes



def _bound_parameters(event: dict, part: str) -> dict:
    """Return the parameter values a declared event fixes, with its part."""
    params = {name: binding["equals"] for name, binding in (event.get("parameter_bindings") or {}).items()
              if isinstance(binding, dict) and "equals" in binding}
    return {**params, "part_name": part} if part else params


def _with_product_effects(task: dict, event: dict, part_row: dict, params: dict) -> dict | None:
    """Give a nominal step the part effects its model declares, or None when its part guard fails.

    A nominal step changes its part's process record as the runtime model does: it appends its
    ledger entries and updates the part's unfinished process. Its guard on the part's own state, such
    as a reloaded job resuming only on the machine that paused it, must hold.
    """
    from cais_spade_llm.resources.environment_models import _check_product_guards, current_process_after

    if not task.get("part_name"):
        return task
    try:
        _check_product_guards(event, part_row, params)
        process = (current_process_after(event, part_row.get("current_process"), params)
                   if isinstance((event.get("product_effects") or {}).get("current_process"), dict) else None)
    except (ValueError, KeyError):
        return None
    end = task["expected_end_state"]
    if completed := _completed_effects(event, params):
        record = list(part_row.get("processCompleted") or [])
        end["processCompleted"] = record + [row for row in completed if row not in record]
    if isinstance((event.get("product_effects") or {}).get("current_process"), dict):
        end["current_process"] = process
    return task

def _recorded_task(jid: str, node: dict, state: dict, prepared: dict, resources: dict, part_rows: dict,
                   depth: int) -> dict | None:
    """Instantiate an interrupted task with its recorded parameters when its public and neighbor guards hold."""
    from . import multi_turn_outline_generation as generation

    model = state.get("recovery_des_models", {}).get(jid, {})
    variables, params = model.get("state_variables", {}), node.get("params", {})
    part = params.get("part_name") or ""
    resource_row, part_row = resources.get(jid, {}), part_rows.get(part, {}) if part else {}
    event = _declared_event(model, node)
    if (event is None or not _public_guards_hold(event, variables, jid, part, resource_row, part_row, params)
            or not _parameters_resolve(event, jid, part, resource_row, part_row, params)
            or (chosen := _neighbor_rows(jid, event, part, state, prepared, resources, part_row, params)) is None):
        return None
    task = generation._recovery_event_instance_task(
        outline_id=f"NOMINAL_CONNECTION_{depth + 1}", resource_jid=jid, event=event,
        resource_row=resource_row, part_name=part, part_row=part_row, state_variables=variables)
    if task is None:
        return None
    for name, update in event.get("updates", {}).items():
        if isinstance(update, dict) and update.get("set_from_param") in params:
            task["expected_end_state"][name] = deepcopy(params[update["set_from_param"]])
    task.update(candidate_source="nominal_connection", task_id=node["id"],
                diagnostic_assumptions=[*_bookkeeping_assumptions(event, variables, task),
                                        *(note for _, _, notes in chosen for note in notes)])
    if neighbor := _neighbor_end_state(chosen, jid, event, part, resource_row, part_row, params):
        task["neighbor_end_state"] = neighbor
    return _with_product_effects(task, event, part_row, {**_bound_parameters(event, part), **params})


def _nominal_successors(state: dict, prepared: dict, actors: set[str], parts: set[str], depth: int) -> list[dict]:
    """Instantiate every declared nominal event of the given actors whose public guards hold."""
    from . import multi_turn_outline_generation as generation

    resources, part_rows = generation._projected_outline_validation_context(
        session_state=state, prepared_recovery_request=prepared)
    faulted, jids = faulted_resources(state, prepared), prepared.get("observation_jids", {})
    tasks = []
    for jid in sorted(actors):
        model = state.get("recovery_des_models", {}).get(jid, {})
        variables = model.get("state_variables", {})
        resource_row = resources.get(jid, {})
        # Observed events, such as a job's completion, continue the plant as commanded ones do.
        for event in model.get("events", []):
            bindings = (sorted(parts) if generation._recovery_event_requires_part_binding(
                event=event, state_variables=variables) else [""])
            for part in bindings:
                part_row = part_rows.get(part, {}) if part else {}
                # An event declared for one part never acts on another.
                if (part and (event.get("parameter_bindings", {}).get("part_name") or {}).get("equals", part) != part
                        or not _public_guards_hold(event, variables, jid, part, resource_row, part_row)
                        or not _parameters_resolve(event, jid, part, resource_row, part_row)
                        or (chosen := _neighbor_rows(jid, event, part, state, prepared, resources, part_row)) is None):
                    continue
                task = generation._recovery_event_instance_task(
                    outline_id=f"NOMINAL_CONNECTION_{depth + 1}", resource_jid=jid, event=event,
                    resource_row=resource_row, part_name=part, part_row=part_row, state_variables=variables)
                end = (task or {}).get("expected_end_state", {})
                if task is not None and not any(jids.get(end.get(field), end.get(field)) in faulted
                                                for field in ("part_location", "resource_location")):
                    task.update(candidate_source="nominal_connection",
                                diagnostic_assumptions=[*_bookkeeping_assumptions(event, variables, task),
                                                        *(note for _, _, notes in chosen for note in notes)])
                    if neighbor := _neighbor_end_state(chosen, jid, event, part, resource_row, part_row, None):
                        task["neighbor_end_state"] = neighbor
                    if (task := _with_product_effects(task, event, part_row, _bound_parameters(event, part))) is not None:
                        tasks.append(task)
    # The interrupted tasks keep their recorded parameters, so nominal work can redo them.
    for jid, node in _replaced_work(prepared):
        if (jid in actors and not _needs_faulted(node.get("params", {}), faulted, prepared)
                and (task := _recorded_task(jid, node, state, prepared, resources, part_rows, depth)) is not None):
            tasks.append(task)
    return tasks


def _comparison_state_fingerprint(session: dict) -> str:
    """Compare modeled facts without counting generated record identities as progress."""
    ignored = {"source", "provenance", "captured_at", "recorded_at", "revision", "source_fingerprint",
               "snapshot_fingerprint", "fingerprint", "outline_id", "command_id", "candidate_source",
               "current_pose_captured_at", "current_pose_source", "field_sources", "commands", "history"}

    def facts(value):
        if isinstance(value, dict):
            return {key: facts(item) for key, item in value.items() if key not in ignored}
        if isinstance(value, list):
            return [facts(item) for item in value]
        return value

    physical = session.get("projected_physical_evidence") or {}
    return recovery_validation_fingerprint(facts({
        "resources": session.get("symbolic_resources"), "parts": session.get("symbolic_parts"),
        "monitor": {key: {field: value for field, value in record.items()
                          if field in {"rules_fingerprint", "states", "pending_rule_ids", "activation"}}
                    for key, record in session.get("outline_monitor_context", {}).items()},
        "snapshot": physical.get("snapshot"), "geometry": physical.get("geometry"),
        "processing_observations": physical.get("processing_observations"),
        "environment_valuation": physical.get("environment_valuation"),
        "resource_models": physical.get("resource_models"),
        "unresolved_processing": physical.get("unresolved_processing"),
    }))


def nominal_connection(session: dict, prepared: dict, *, max_steps: int = _CONNECTION_MAX_STEPS) -> list[dict] | None:
    """Search declared nominal events that bring the projected state to the next nominal task.

    A breadth-first search over at most ``max_steps`` events, using only the resources and parts named
    by what still blocks the handoff, and the interrupted work while its outcome does not hold; its
    recorded tasks may be redone with their recorded parameters. Faulted resources never act, and no
    connection is offered while the resumed work needs one.

    Args:
        session: Tentative planning state after the accepted recovery events.
        prepared: Recorded failure context.
        max_steps: The longest connection considered.

    Returns:
        The connecting nominal tasks, an empty list when the handoff already holds, or None.
    """
    if handoff(dict(session), prepared)["complete"]:
        return []
    if _faulted_resumed_work(session, prepared):
        return None
    actors, parts = _connection_scope(session, prepared)
    frontier = [(_branch(session), [])]
    seen = {_comparison_state_fingerprint(session)}
    while frontier:
        state, steps = frontier.pop(0)
        if len(steps) >= max_steps:
            continue
        for task in _nominal_successors(state, prepared, actors, parts, len(steps)):
            working = _branch(state)
            apply_tentative_effects(task, working, prepared)
            working["accepted_outline_prefix"].append(deepcopy(task))
            identity = _comparison_state_fingerprint(working)
            if identity in seen:
                continue
            seen.add(identity)
            if handoff(dict(working), prepared)["complete"]:
                return [*steps, task]
            frontier.append((working, [*steps, task]))
    return None


def connection_blockers(session: dict, prepared: dict) -> list[str]:
    """State why nominal work cannot take over, without listing the states its next task needs."""
    reasons = _faulted_resumed_work(session, prepared)
    if reasons:
        return reasons
    return [f"no sequence of up to {_CONNECTION_MAX_STEPS} declared nominal events reaches its next task"]


def _connected_handoff(session: dict, prepared: dict, connection: list[dict]) -> dict:
    """Describe the handoff reached by appending the nominal connection to the accepted events."""
    working = _branch(session)
    for task in connection:
        apply_tentative_effects(task, working, prepared)
        working["accepted_outline_prefix"].append(deepcopy(task))
    session["diagnostic_nominal_connection"] = deepcopy(connection)
    steps = []
    for task in connection:
        step = {key: deepcopy(task.get(key)) for key in ("outline_id", "event_name", "resource_jid", "part_name",
                                                          "candidate_source", "task_id") if task.get(key) is not None}
        step["expected_end_state"] = {change["variable"]: change["value"] for change in _event_changes(task, session)}
        steps.append(step)
    proposed = handoff(working, prepared)
    # What the search assumed for its steps is part of what the handoff assumes.
    proposed["assumptions"] += [{"outline_id": task["outline_id"], **note}
                                for task in connection for note in task.get("diagnostic_assumptions", [])]
    return {**proposed, "nominal_connection": steps}


def _unconnected_reasons(session: dict, prepared: dict) -> list[str]:
    """State why nominal work cannot take over, for the record of a declared recovery."""
    if goal_reached(session, prepared) is not False:
        return connection_blockers(session, prepared)
    faulted = faulted_resources(session, prepared)
    return [f"the outcome stated in Recovery Goal does not hold, and no sequence of up to {_CONNECTION_MAX_STEPS} "
            "declared nominal events produces it", *_faulted_resumed_work(session, prepared),
            *(f"{jid} did the interrupted work and is still reported faulted"
              for jid in sorted({jid for jid, _ in _replaced_work(prepared)} & faulted))]


def declare_recovery_complete(*, session: dict, prepared: dict, response: dict) -> tuple[str, dict]:
    """End the recovery on the LLM's declaration and record whether nominal work can take over.

    Whether the recovery is complete is the LLM's decision: code neither refuses nor answers it. It
    searches the declared nominal events for a connection and records the result, or why none exists,
    for review.

    Args:
        session: Tentative planning state, ended by the declaration.
        prepared: Recorded failure context.
        response: The LLM's ``recovery_complete`` reply.

    Returns:
        ``outline_ready`` with the recorded handoff.
    """
    connection = nominal_connection(session, prepared)
    proposed = (_connected_handoff(session, prepared, connection) if connection is not None
                else handoff(dict(session), prepared))
    proposed.update(declared_complete_by_llm=True,
                    not_connected_reasons=[] if connection is not None else _unconnected_reasons(session, prepared))
    session.update(diagnostic_nominal_handoff=proposed, status="ready_for_primitive_generation")
    return "outline_ready", {"diagnostic_nominal_handoff": proposed,
                             "transition_validation": {"status": "not_checked"}, "execution_authorized": False}


def handoff(session: dict, prepared: dict) -> dict:
    """Describe a proposed nominal handoff without asserting feasibility or safety."""
    from . import multi_turn_outline_generation as comparison

    required = replaced_tasks(prepared)
    remaining = required - set(session.get("diagnostic_replaced_task_claims", {}))
    rows = comparison._nominal_reentry_event_rows(session_state=session, prepared_recovery_request=prepared)
    entries = nominal_entry_ids(session, prepared)
    expected = {row["event_id"] for row in rows}
    recorded = bool(recorded_conditions(prepared))
    unmet = [deepcopy(row) for row in entry_conditions(prepared) if not condition_holds(row, session, prepared)]
    open_work, undeclared = open_replaced_work(session, prepared) if recorded else (set(), [])
    # Recorded handoff conditions state what the interrupted work must leave
    # true; its intermediate owner states need not be reproduced. When they
    # only describe the resumed work, the replaced work's final outcome is
    # still required.
    conditions_met = not unmet and not open_work if recorded else bool(required or expected) and not remaining
    ready = conditions_met and bool(expected) and expected <= entries
    return {"complete": ready, "conditions_met": ready, "unmet_handoff_conditions": unmet,
        "remaining_replaced_task_ids": sorted(remaining), "open_replaced_task_ids": sorted(open_work),
        "proposed_nominal_entry_ids": sorted(entries), "required_nominal_entry_ids": sorted(expected),
        "assumptions": [*deepcopy(session.get("diagnostic_handoff_assumptions", [])), *undeclared,
                        *unbound_resume_entries(session, prepared)],
        "basis": "recorded handoff conditions on tentative symbolic state; complete nominal continuation not checked",
        "execution_authorized": False}


def unresolved_conditions(session: dict, prepared: dict) -> set[str]:
    """Return open recovery obligations used by candidate comparison.

    Conditions already true at the failure are required again at handoff but are
    not obligations, so a held part may be set down temporarily.
    """
    if recorded_conditions(prepared):
        return {_condition_id(row) for row in entry_conditions(prepared, "unmet_reentry_conditions")
                if not condition_holds(row, session, prepared)} | {
            "replaced_work:" + identifier for identifier in open_replaced_work(session, prepared)[0]}
    state = handoff(session, prepared)
    return {"recorded_task:" + name for name in state["remaining_replaced_task_ids"]} | {
        "nominal_entry:" + name for name in set(state["required_nominal_entry_ids"]) - set(state["proposed_nominal_entry_ids"])}


def event_record(event: dict, session: dict, prepared: dict) -> dict:
    """Return the event as the LLM states it, with the projected values it starts from.

    Nothing about the event is checked. Its start state holds the current projected value of each
    variable its end state names, where the projected state has one.

    Args:
        event: One candidate event from the LLM's reply.
        session: Projected planning state before the event.
        prepared: Recorded failure context.

    Returns:
        The event record that is applied and kept in the outline.
    """
    from . import multi_turn

    jid, part = event.get("resource_jid"), event.get("part_name")
    resources, parts = multi_turn._projected_outline_validation_context(
        session_state=session, prepared_recovery_request=prepared)
    model = (session.get("recovery_des_models") or {}).get(jid) or (
        (prepared.get("recovery_resources") or {}).get(jid) or {}).get("recovery_des_model") or {}
    variables = model.get("state_variables") or {}
    start = {}
    for name in sorted(event.get("expected_end_state") or {}):
        scope = variables.get(name, {}).get("scope") or (
            "part" if name in {"part_location", "current_process", "processCompleted"} else "resource")
        row = parts.get(part, {}) if scope == "part" else resources.get(jid, {})
        known, value = multi_turn._projected_state_value(row=row, field_name=name, scope=scope)
        if known:
            start[name] = value
    record = {"outline_id": event.get("outline_id"), "event_name": event.get("event_name"), "resource_jid": jid,
              "expected_start_state": start, "expected_end_state": deepcopy(event.get("expected_end_state")),
              "rationale": event.get("rationale")}
    if "part_name" in event:
        record["part_name"] = part
    if event.get("neighbor_end_state"):
        record["neighbor_end_state"] = deepcopy(event["neighbor_end_state"])
    return record


async def handle_event(*, session: dict, prepared: dict, response: dict, planner: Any) -> tuple[str, dict]:
    """Apply the candidate event the LLM chose, as it states it; nothing checks it.

    The LLM alone ends the recovery, by declaring it complete.
    """
    from . import multi_turn

    task = multi_turn._commit_selected_candidate_task(task=event_record(response["next_transition"], session, prepared),
        sequence_index=multi_turn._next_recovery_sequence_index(session))
    working = deepcopy(session)
    apply_tentative_effects(task, working, prepared)
    working["derived_resource_snapshots"] = deepcopy(working.get("symbolic_resources", {}))
    working.setdefault("accepted_outline_prefix", []).append(deepcopy(task))
    session.update(working)
    multi_turn._sync_des_recovery_aliases(session)
    proposed = handoff(session, prepared)
    session["diagnostic_nominal_handoff"] = proposed
    session["status"] = "paused_after_outline_turn"
    return "need_next_task", {
        "next_transition": deepcopy(task), "transition_validation": {"status": "not_checked"},
        "diagnostic_nominal_handoff": proposed, "execution_authorized": False}


def populate_comparison(evaluations: list[dict], session: dict, prepared: dict) -> None:
    """Supply the existing comparator with symbolic enabledness and no agent calls."""
    from . import multi_turn_outline_generation as comparison

    def enabled(state):
        return {row["event_id"] for row in comparison._symbolically_enabled_recovery_event_instances(
            session_state=state, prepared_recovery_request=prepared,
            unresolved_condition_ids=unresolved_conditions(state, prepared))}

    before, nominal_before = enabled(session), nominal_entry_ids(session, prepared)
    for row in evaluations:
        state = comparison._candidate_projected_session(session_state=session, evaluation=row)
        row.update(agent_filtered_enabledness=True, diagnostic_validation_bypass=True,
            admissible_recovery_enabled_event_ids_before=sorted(before),
            admissible_recovery_enabled_event_ids_after=sorted(enabled(state)),
            admissible_nominal_reentry_event_ids_before=sorted(nominal_before),
            admissible_nominal_reentry_event_ids=sorted(nominal_entry_ids(state, prepared)),
            cca_admissible_goal_recovery_event_ids_before=[], cca_admissible_goal_recovery_event_ids_after=[],
            comparison_evidence_basis="unvalidated proposed symbolic effects")
