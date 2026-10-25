from __future__ import annotations

"""Failure-scoped outline prompt facts, stated by resource name.

Every section is built from the recorded capture (grounded signals, pending
tasks, declared fields and events) with the same rules for every resource type,
so no scenario or resource type needs its own wording.
"""

import re
from copy import deepcopy
from typing import Any

# Matches the grounding settings' location parameters (execution_monitoring).
_LOCATION_PARAMS = ("destination_location", "origin_resource_location", "target_resource", "source_resource")
# Owner bookkeeping, as already treated in multi_turn_diagnostic.
_OWNER_BOOKKEEPING = {"active_job"}
_PART_FIELDS = {"part_location", "current_process"}


def resource_names(prepared: dict) -> dict[str, str]:
    """Return each resource JID's recorded name; a JID without one keeps itself."""
    names = {jid: name for name, jid in prepared.get("observation_jids", {}).items() if isinstance(jid, str)}
    return {jid: names.get(jid, jid) for jid in prepared.get("recovery_resources", {})} | names


def with_resource_names(text: str, names: dict[str, str]) -> str:
    """Replace whole JID tokens with resource names in rendered prompt text."""
    for jid in sorted((jid for jid, name in names.items() if jid != name), key=len, reverse=True):
        text = re.sub(rf"(?<![\w@.-]){re.escape(jid)}(?![\w@.-])", names[jid], text)
    return text


def _named_domain(values: list, names: dict[str, str]) -> list:
    """State a declared domain by resource name, each value once, keeping its first place."""
    domain = []
    for value in values:
        named = names.get(value, value) if isinstance(value, str) else value
        if named not in domain:
            domain.append(named)
    return domain


def with_named_domains(served: dict, names: dict[str, str]) -> dict:
    """Return lookup replies whose declared domains name each resource once.

    A domain may list a resource both by JID and by its recorded name. The prompt names both alike,
    so after naming they are one value.

    Args:
        served: Lookup replies by reference.
        names: Resource JID to recorded name.

    Returns:
        A copy whose state_variables replies state their domains by name without repeats.
    """
    result = dict(served)
    for ref, fact in served.items():
        if ref.endswith("/state_variables") and isinstance(fact.get("value"), dict):
            result[ref] = {**fact, "value": {
                name: {**row, "domain": _named_domain(row["domain"], names)}
                if isinstance(row, dict) and isinstance(row.get("domain"), list) else row
                for name, row in fact["value"].items()}}
    return result


def schema_without_params(schema: Any) -> Any:
    """Return a response schema whose outline events carry no ``params``; positions are conditions."""
    if isinstance(schema, list):
        return [schema_without_params(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: schema_without_params(value) for key, value in schema.items()}
    properties = result.get("properties")
    if isinstance(properties, dict) and "expected_end_state" in properties and "params" in properties:
        result["properties"] = {key: value for key, value in properties.items() if key != "params"}
    return result


_NEIGHBOR_END_STATE_SCHEMA = {
    "type": "object", "additionalProperties": {"type": "object"},
    "description": ("Changes to other resources' variables caused by moving the event's part, by resource name: "
                    "{\"<resource name>\": {\"<variable>\": <value>}}. Those resources only record the change; the "
                    "event's resource is the one that acts."),
}


def schema_with_neighbor_end_state(schema: Any) -> Any:
    """Return a response schema whose outline events may also state what moving their part changes elsewhere.

    Args:
        schema: Outline response schema.

    Returns:
        A copy whose outline event offers the optional ``neighbor_end_state`` after ``expected_end_state``.
    """
    if isinstance(schema, list):
        return [schema_with_neighbor_end_state(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: schema_with_neighbor_end_state(value) for key, value in schema.items()}
    properties = result.get("properties")
    if isinstance(properties, dict) and "event_name" in properties and "expected_end_state" in properties:
        result["properties"] = {}
        for name, value in properties.items():
            result["properties"][name] = value
            if name == "expected_end_state":
                result["properties"]["neighbor_end_state"] = deepcopy(_NEIGHBOR_END_STATE_SCHEMA)
    return result


def schema_with_resource_names(schema: Any, names: dict[str, str]) -> Any:
    """Return a response schema whose events name their resource by ``resource``.

    Args:
        schema: Outline response schema keyed by ``resource_jid``.
        names: Resource JID to recorded name.

    Returns:
        A copy with ``resource_jid`` properties and requirements renamed to ``resource`` and any
        JID enum values replaced by names.
    """
    if isinstance(schema, list):
        return [schema_with_resource_names(item, names) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict) and "resource_jid" in value:
            value = {("resource" if name == "resource_jid" else name): item for name, item in value.items()}
            if isinstance(value["resource"], dict) and isinstance(value["resource"].get("enum"), list):
                value["resource"] = {**value["resource"],
                                     "enum": [names.get(item, item) for item in value["resource"]["enum"]]}
        elif key == "required" and isinstance(value, list):
            value = ["resource" if item == "resource_jid" else item for item in value]
        result[key] = schema_with_resource_names(value, names)
    return result


def _jid(value: Any, prepared: dict) -> Any:
    return prepared.get("observation_jids", {}).get(value, value) if isinstance(value, str) else value


def response_with_jids(response: dict, prepared: dict) -> dict:
    """Map resource names in an outline response back to the JIDs the runtime validates.

    Args:
        response: LLM response with ``candidate_events`` or ``context_requests``.
        prepared: Recorded failure context holding ``observation_jids``.

    Returns:
        A copy whose events carry ``resource_jid`` and whose ``neighbor_end_state`` is keyed by JID.
        When the event's resource takes hold of the part, a ``part_location`` naming that resource
        becomes its JID; location tokens such as a station that shares a resource's name are kept
        exactly.
    """
    mapped = dict(response)
    events = []
    for event in response.get("candidate_events") or []:
        if not isinstance(event, dict):
            events.append(event)
            continue
        row = dict(event)
        name = row.pop("resource", None) or row.get("resource_jid")
        row["resource_jid"] = _jid(name, prepared)
        end = row.get("expected_end_state")
        if (isinstance(end, dict) and name is not None and end.get("part_location") == name
                and row.get("part_name") and end.get("held_part") == row["part_name"]):
            row["expected_end_state"] = {**end, "part_location": row["resource_jid"]}
        if isinstance(row.get("neighbor_end_state"), dict):
            row["neighbor_end_state"] = {_jid(place, prepared): values for place, values in row["neighbor_end_state"].items()}
        events.append(row)
    if "candidate_events" in response:
        mapped["candidate_events"] = events
    if isinstance(response.get("context_requests"), list):
        mapped["context_requests"] = [_ref_with_jid(ref, prepared) for ref in response["context_requests"]]
    return mapped


def _ref_with_jid(ref: Any, prepared: dict) -> Any:
    if not isinstance(ref, str):
        return ref
    segments = ref.split("/")
    if len(segments) >= 3 and segments[1] == "resources":
        segments[2] = _jid(segments[2], prepared)
    return "/".join(segments)


def _pose(pose: Any) -> str:
    """State a position as (x, y, z), or (x, y, yaw …) for a planar pose; empty when x or y is missing."""
    if not isinstance(pose, dict) or not all(isinstance(pose.get(axis), (int, float)) for axis in "xy"):
        return ""
    if isinstance(pose.get("z"), (int, float)):
        return f"({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f})"
    yaw = f", yaw {pose['yaw']:.3f}" if isinstance(pose.get("yaw"), (int, float)) else ""
    return f"({pose['x']:.3f}, {pose['y']:.3f}{yaw})"


def _containing_regions(pose: Any, prepared: dict) -> list[str]:
    from .multi_turn_diagnostic import _inside

    if not _pose(pose):
        return []
    return sorted(name for name, row in prepared.get("diagnostic_regions", {}).items()
                  if isinstance(row, dict) and row.get("bounds") and _inside(pose, row["bounds"]))


def _fault(prepared: dict) -> dict:
    """Return the grounded fault event, or the prompt fault event for older captures."""
    grounded = prepared.get("code_grounding", {}).get("fault_event") or {}
    if grounded.get("signals") is not None:
        return {"signals": grounded["signals"], "task_id": grounded.get("failed_task_id"),
                "function": grounded.get("failed_function_name"),
                "resource": grounded.get("failed_resource_jid")}
    fault = prepared.get("llm_input", {}).get("fault_event", {})
    anchor = prepared.get("failure_anchor", {})
    return {"signals": fault.get("detected_signals") or [],
            "task_id": fault.get("blocked_at_task_id") or anchor.get("failed_task_id"),
            "function": fault.get("blocked_at_function") or anchor.get("failed_function_name"),
            "resource": fault.get("focused_resource_jid") or anchor.get("failed_resource_jid")}


def _signal_values(signal: dict) -> dict:
    meta = {"rule", "signal", "resource_id", "task_id", "part_name", "evidence", "values"}
    return {**{key: value for key, value in signal.items() if key not in meta}, **(signal.get("values") or {})}


def _signal_line(signal: dict) -> str:
    """State one grounded signal by its own name, subjects and recorded values.

    No wording depends on the kind of signal, so a signal from a detection rule that is new
    to this prompt is stated just as completely as the known ones.
    """
    subjects = [signal.get("resource_id"), signal.get("part_name"),
                f"task {signal['task_id']}" if signal.get("task_id") else None]
    values = ", ".join(f"{key} {_value_text(value)}" for key, value in _signal_values(signal).items())
    name = signal.get("signal") or signal.get("rule") or "signal"
    subject_text = ", ".join(str(item) for item in subjects if item)
    return f"- {name}" + (f" ({subject_text})" if subject_text else "") + (f": {values}" if values else "")


def _affected_part_names(prepared: dict, fault: dict) -> set[str]:
    """Return the parts named by the grounded signals, the failure record and the interrupted task."""
    names = {row.get("part_name") for row in fault["signals"] if row.get("part_name")}
    names |= set(prepared.get("llm_input", {}).get("fault_event", {}).get("affected_part_names", []))
    names |= {row.get("part_name") for row in prepared.get("failure_anchor", {}).get("affected_parts", [])}
    names.add(_task_nodes(prepared).get(fault["task_id"], {}).get("params", {}).get("part_name"))
    return {name for name in names if isinstance(name, str) and name}


def _recorded_part(prepared: dict, name: str) -> dict:
    """Return a part's grounded state at the failure: location, unfinished process and observed pose."""
    for source in (prepared.get("code_grounding", {}).get("parts", {}), prepared.get("grounding_context", {}).get("parts", {})):
        if isinstance(source.get(name), dict):
            return source[name]
    fact = next((row for row in prepared.get("llm_input", {}).get("part_facts", []) if row.get("part_name") == name), {})
    return {"location": fact.get("current_location"), "current_process": fact.get("current_process"),
            "observed_pose": fact.get("observed_pose")}


def _process_text(entry: dict, status: str, names: dict[str, str]) -> str:
    """State one process: its name, its result or other recorded values, and its status.

    A completed ledger entry reads ``trim → square completed``; the unfinished process also names the
    machine that holds its job: ``trim → square paused on M1``.
    """
    text = str(entry.get("process"))
    if entry.get("result") not in (None, ""):
        text += f" → {entry['result']}"
    others = [str(value) for key, value in entry.items()
              if key not in {"process", "result", "status", "machine"} and value not in (None, "")]
    machine = entry.get("machine")
    return " ".join([text, *others, status]) + (f" on {names.get(machine, machine)}" if machine else "")


def _processes_text(current: Any, record: Any, names: dict[str, str]) -> str:
    """State a part's unfinished process and its completed ledger, or nothing when it has neither."""
    items = [_process_text(current, str(current.get("status") or "unfinished"), names)] if isinstance(current, dict) else []
    items += [_process_text(item, "completed", names) for item in record or []
              if isinstance(item, dict) and item.get("process")]
    return f"processes: {', '.join(items)}" if items else ""


def _area_text(pose: Any, prepared: dict) -> str:
    """Name the recorded areas that contain a pose; state nothing when none does."""
    regions = _containing_regions(pose, prepared)
    return f"inside {', '.join(regions)}" if regions else ""


def _part_detail(name: str, prepared: dict, names: dict[str, str]) -> str:
    part = _recorded_part(prepared, name)
    location, pose = part.get("location"), _pose(part.get("observed_pose"))
    holder = next((names.get(jid, jid) for jid, entry in prepared.get("recovery_resources", {}).items()
                   if entry.get("recovery_snapshot", {}).get("held_part") == name), None)
    # The part's facts are where it is and its process record; the measured facts describe the rest.
    details = [f"held by {holder}" if holder else f"at {names.get(location, location)}" if location
               else "not held, not at any station",
               f"measured at {pose}" if pose else "", _area_text(part.get("observed_pose"), prepared)]
    processes = _processes_text(part.get("current_process"), [], names)
    return f"- {name}: {', '.join(item for item in details if item)}" + (f"; {processes}" if processes else "")


def failure_lines(prepared: dict, session: dict) -> list[str]:
    """State the interrupted task, each grounded detection signal and the recorded failure details.

    Args:
        prepared: Recorded failure context.
        session: Tentative planning state, which holds the declared resource models.

    Returns:
        Failure section lines. The interrupted task comes from the failure record, so it is stated
        even when no controller goal report exists. Each affected part is stated with its recorded
        place or measured pose, and each involved resource with its measured pose. Which resources
        can act there is left for the LLM to look up and reason out.
    """
    names = resource_names(prepared)
    fault = _fault(prepared)
    lines, signals = [], list(fault["signals"])
    task = _task_nodes(prepared).get(fault["task_id"], {})
    task_params = task.get("params", {})
    owner = task_params.get("resource_id") or names.get(fault["resource"], fault["resource"])
    if fault["function"]:
        # The interrupted task belongs to its own resource, which may differ from the faulted one.
        details = [_params_text([task])] if task and _params_text([task]) else []
        part = task_params.get("part_name")
        lines.append(f"- {owner} did not complete {fault['function']}" + (f" of {part}" if part else "")
                     + (f" ({'; '.join(details)})" if details else ""))
    lines.extend(_signal_line(row) for row in signals)
    affected = sorted(_affected_part_names(prepared, fault))
    lines.extend(_part_detail(name, prepared, names) for name in affected)
    involved = {_jid(owner, prepared), fault["resource"]} | {
        _jid(row.get("resource_id"), prepared) for row in fault["signals"] if row.get("resource_id")}
    for jid in sorted(jid for jid in involved if isinstance(jid, str)):
        pose = prepared.get("recovery_resources", {}).get(jid, {}).get("recovery_snapshot", {}).get("current_pose")
        if _pose(pose):
            area = _area_text(pose, prepared)
            lines.append(f"- {names.get(jid, jid)}: measured pose {_pose(pose)}" + (f", {area}" if area else ""))
    return lines


def _task_nodes(prepared: dict) -> dict:
    return {node["id"]: node for node in prepared.get("saved_pending_nodes", []) if node.get("id")}


def _value_text(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_value_text(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "(" + ", ".join(_value_text(item) for item in value) + ")"
    if isinstance(value, bool) or value is None:
        return {True: "true", False: "false", None: "none"}[value]
    if isinstance(value, float):
        return f"{round(value, 3):g}"
    return str(value)


def _params_text(tasks: list[dict]) -> str:
    """State the tasks' public parameters once; a value that changes between tasks is shown in order."""
    values: dict[str, list[str]] = {}
    for task in tasks:
        for key, value in task.get("params", {}).items():
            if key in {"resource_id", "part_name"} or isinstance(value, bool) or value in (None, "", {}, []):
                continue
            text = _value_text(value)
            if not values.setdefault(key, []) or values[key][-1] != text:
                values[key].append(text)
    return ", ".join(f"{key} {' → '.join(items)}" for key, items in values.items())


def _suffix_rows(prepared: dict, role: str) -> list[tuple[str | None, list[dict], dict]]:
    nodes = _task_nodes(prepared)
    rows = []
    for suffix in prepared.get("marked_reentry_context", {}).get("pending_suffixes", []):
        if suffix.get("role") != role:
            continue
        identifiers = suffix.get("pending_task_ids") or [suffix.get("entry_task_id")]
        tasks = [nodes[identifier] for identifier in identifiers if identifier in nodes]
        jid = suffix.get("resource_jid") or next(
            (_jid(task.get("params", {}).get("resource_id"), prepared) for task in tasks), None)
        rows.append((jid, tasks, suffix))
    return rows


def _normal_work(jid: str | None, task: dict, prepared: dict, session: dict, names: dict[str, str]) -> str:
    """State the recorded task that normally produces a goal, by its resource's name.

    Each resource the task needs, as its actor or as a place it names (as in
    ``multi_turn_diagnostic._needs_faulted``), is also stated with its current value while a grounded
    signal still reports it faulted: "unavailable" when it is, else its state. Nothing is said about
    what a resource can still do.

    Args:
        jid: The resource that does the task.
        task: The recorded task.
        prepared: Recorded failure context.
        session: Tentative planning state.
        names: Resource JID to recorded name.

    Returns:
        `` (normally <resource>'s <function>; <resource> <value>)``, or an empty string for a task
        without a function name.
    """
    from .multi_turn_diagnostic import faulted_resources

    if not jid or not task.get("function_name"):
        return ""
    faulted, params = faulted_resources(session, prepared), task.get("params", {})
    needed = dict.fromkeys([jid, *(_jid(params[key], prepared) for key in _LOCATION_PARAMS if isinstance(params.get(key), str))])
    reported = []
    for other in needed:
        row = session.get("symbolic_resources", {}).get(other, {})
        value = ("unavailable" if row.get("availability") == "unavailable"
                 else row.get("resource_state") or row.get("current_state"))
        if other in faulted and value:
            reported.append(f"{names.get(other, other)} {value}")
    return f" (normally {names.get(jid, jid)}'s {task['function_name']}" + "".join(f"; {item}" for item in reported) + ")"


def _part_normal_work(part: Any, prepared: dict, session: dict, names: dict[str, str], *, include_moved: bool) -> str:
    """State the interrupted task that normally brings a part to its goal.

    That is the last replaced task, in recorded plan order (the order ``resume_entries`` uses), that
    handles the part.
    """
    from .multi_turn_diagnostic import _replaced_work, _task_part

    order = {node.get("id"): index for index, node in enumerate(prepared.get("saved_pending_nodes", []))}
    rows = [(jid, node) for jid, node in _replaced_work(prepared, include_moved=include_moved)
            if isinstance(part, str) and _task_part(node, prepared) == part]
    if not rows:
        return ""
    jid, node = max(rows, key=lambda row: order.get(row[1].get("id"), -1))
    return _normal_work(jid, node, prepared, session, names)


def _work_outcome(jid: str, tasks: list[dict], prepared: dict, session: dict, names: dict[str, str]) -> str:
    """State the outcome interrupted work was producing, and the recorded task that normally produces it.

    The final task's declared part effects, resolved with its parameters, state the outcome when
    the model declares them. A final task the model declares no event for states its part at the
    task's destination. Otherwise the tasks' public parameters are stated as recorded.
    """
    from .multi_turn_diagnostic import declared_part_effects, destination_outcome

    params = tasks[-1].get("params", {})
    effects = (declared_part_effects(jid, tasks[-1], prepared, session)
               or destination_outcome(jid, tasks[-1], prepared, session))
    part = next((task["params"]["part_name"] for task in tasks if task.get("params", {}).get("part_name")), None)
    if effects:
        text = ", ".join(", ".join(_process_text(item, "completed", names) for item in value) if key == "processCompleted"
                         else f"{key} {names.get(value, value) if isinstance(value, str) else _value_text(value)}"
                         for key, value in effects.items())
        text += f" (target {params['target']})" if params.get("target") else ""
    else:
        text = _params_text(tasks)
    return f"- {part or names.get(jid, jid)}: {text}" + _normal_work(jid, tasks[-1], prepared, session, names)


def goal_lines(prepared: dict, session: dict) -> list[str]:
    """State the outcome the interrupted work was producing.

    Each outcome also names the recorded task that normally produces it, with the current reported
    fault of each resource that task needs (see ``_normal_work``).

    Neighboring nominal work is not stated as a goal; it reaches the LLM through the safety rules and
    pending-task references. Whether normal production can take over is never told to the LLM, which
    decides completion itself.

    Args:
        prepared: Recorded failure context.
        session: Tentative planning state.

    Returns:
        Recovery Goal section lines.
    """
    from .multi_turn_diagnostic import _MOVED, declared_part_effects, handover_conditions, recorded_conditions

    names = resource_names(prepared)
    replaced = [row for row in recorded_conditions(prepared) if row.get("role") == "recovery_replaced"]
    handover = handover_conditions(prepared)
    outcomes: dict[str, list[str]] = {}
    parts: dict[str, Any] = {}
    for row in replaced or handover:
        outcomes.setdefault(row["entity"], []).append(f"{row['field']} {names.get(row['expected'], row['expected'])}")
        # A resource's condition, such as a slot, names the part it waits for.
        parts.setdefault(row["entity"], row["entity"] if row.get("entity_kind") == "part" else row.get("expected"))
    # Moved work's outcome is the handover (see _replaced_finals), so only handover rows may name it.
    lines = [f"- {names.get(entity, entity)}: {', '.join(items)}"
             + _part_normal_work(parts[entity], prepared, session, names, include_moved=not replaced)
             for entity, items in outcomes.items()]
    if not replaced:
        # Once resumed work needing a faulted resource is replaced, replaced work that declares no
        # part outcome is complete when the parts are as the remaining resumed work needs them.
        lines += [_work_outcome(jid, tasks, prepared, session, names)
                  for jid, tasks, suffix in _suffix_rows(prepared, "recovery_replaced")
                  if tasks and not suffix.get(_MOVED)
                  and not (handover and not declared_part_effects(jid, tasks[-1], prepared, session))]
    if not lines:
        # Without interrupted work, the goal is the expected value each signal recorded for its part.
        for signal in _fault(prepared)["signals"]:
            for key, value in _signal_values(signal).items():
                if signal.get("part_name") and key.startswith("expected_") and value not in (None, ""):
                    lines.append(f"- {signal['part_name']}: {key.removeprefix('expected_')} {_value_text(value)}")
    return lines


def _declared_locations(entry: dict) -> list[str]:
    from .multi_turn_context import _resource_location_domain

    for source in ("static_capabilities", "recovery_adapter", "recovery_snapshot"):
        values = (entry.get(source) or {}).get("reachability")
        if isinstance(values, list) and values:
            return [value for value in values if isinstance(value, str)]
    return _resource_location_domain(entry)


def _coordinate_parameters(model: dict) -> list[str]:
    """Return the coordinates a resource moves to.

    These are the fields of its declared pose when that pose locates the resource (a base it moves),
    or else the free numeric parameters of a declared event that sets ``resource_location``.
    """
    pose = model.get("state_variables", {}).get("pose", {})
    if pose.get("locates_resource") and pose.get("fields"):
        return list(pose["fields"])
    for event in model.get("events", []):
        if "resource_location" not in event.get("updates", {}):
            continue
        free = [name for name, binding in event.get("parameter_bindings", {}).items()
                if isinstance(binding, dict) and binding.get("required") is True and binding.get("type") == "number"]
        if free:
            return free
    return []


def _environment_model_events(entry: dict) -> list[dict]:
    """Return a resource's own declared environment model events: the draft's copy, or its owner evidence."""
    if "environment_model_events" in entry:
        return entry["environment_model_events"]
    evidence = entry.get("resource_evidence") or {}
    return ((evidence.get("environment_models") or {}).get(evidence.get("resource_id")) or {}).get("events", [])


def _placement_positions(place: str, resource: str, prepared: dict, names: dict[str, str]) -> list[str]:
    """Return where a place's own declared events put a part that ``resource`` places there.

    A place that keeps each part's position (a per-part variable) declares the value a placement by
    ``resource`` sets, such as the loading position a robot uses on a shared belt.
    """
    place_jid = next((jid for jid, name in names.items() if name == place), None)
    positions = []
    for event in _environment_model_events(prepared.get("recovery_resources", {}).get(place_jid, {})):
        bindings = event.get("parameter_bindings", {})
        if ((bindings.get("resource_id") or {}).get("equals") != resource
                or (bindings.get("destination_location") or {}).get("equals") != place):
            continue
        for field, update in event.get("updates", {}).items():
            value = update.get("set") if isinstance(update, dict) else None
            if "{part_name}" in field and isinstance(value, str) and value not in positions:
                positions.append(value)
    return positions


def declared_record_places(jid: str, prepared: dict) -> dict[str, str]:
    """Return, by variable, the places inside a resource where its declared events record a part.

    When another resource places a part at such a place, or takes one from it, the owner's variable
    records it; a machine's staging tray, for example, sets the machine's staging_part. Places named
    like the resource itself are its own, not places inside it.

    Args:
        jid: The owning resource.
        prepared: Recorded failure context holding the resource's declared events.

    Returns:
        The place each recording variable belongs to.
    """
    own = resource_names(prepared).get(jid, jid)
    places: dict[str, str] = {}
    for event in _environment_model_events(prepared.get("recovery_resources", {}).get(jid, {})):
        bindings = event.get("parameter_bindings", {})
        if (bindings.get("resource_id") or {}).get("equals") == own:
            continue
        for key in ("destination_location", "origin_resource_location"):
            place = (bindings.get(key) or {}).get("equals")
            if isinstance(place, str) and place != own:
                for name, update in (event.get("updates") or {}).items():
                    if isinstance(update, dict) and update.get("set_from_param") == "part_name":
                        places.setdefault(name, place)
    return places


def _advance_order(model: dict, resource: str) -> list[str]:
    """Return the positions a resource's own declared events advance parts through, in order.

    A belt's advance event moves every part along a per-part position family, in the family's declared
    order; a zoned transport's events each move a part from one zone variable to the next. Only the
    resource's own events count: an event another resource performs there, such as a robot loading a
    machine, is that resource's motion.
    """
    variables, steps = model.get("state_variables", {}), {}
    for event in model.get("events", []):
        if (event.get("parameter_bindings", {}).get("resource_id") or {}).get("equals") != resource:
            continue
        for family in event.get("collection_effects") or {}:
            if family.endswith(".{part_name}"):
                prefix = family[:-len("{part_name}")]
                domain = next((row.get("domain") or [] for name, row in variables.items() if name.startswith(prefix)), [])
                if positions := [value for value in domain if value is not None]:
                    return positions
        updates, guards = event.get("updates") or {}, event.get("guards") or {}
        sources = [name for name, rule in updates.items() if rule == {"set": None}
                   and (guards.get(name) or {}).get("equals_from_param") == "part_name"]
        targets = [name for name, rule in updates.items() if rule == {"set_from_param": "part_name"}]
        # Taking a part into held_part is a grasp, not transport.
        if len(sources) == 1 and len(targets) == 1 and "held_part" not in (sources[0], targets[0]):
            steps[sources[0]] = targets[0]
    starts = [name for name in steps if name not in steps.values()]
    if len(starts) != 1:
        return []
    order = starts
    while order[-1] in steps and len(order) <= len(steps):
        order = [*order, steps[order[-1]]]
    return order


def _capabilities(jid: str, prepared: dict, session: dict) -> str:
    """State a resource's declared capabilities, the same way for every resource.

    These are the resource capabilities of the paper's knowledge base, read from the resource's
    DES declarations:
    - reachable locations: the places its events can take it to (its declared ``reachability`` or
      ``resource_location`` domain);
    - places parts on a place at a position: for a reachable place that keeps each part's position,
      where its own declared events put a part this resource places there. It is a separate item so
      the place name stays exactly as declared;
    - workspace bounds: a declared reach area (``gripper_reach``), detailed by lookup; it constrains
      event parameters like the paper's capability constraint;
    - moves to any position: a declared event that sets its location from free numeric parameters;
    - moves parts in one direction: the positions its own declared events advance parts through, in order;
    - processes: the declared ``process`` values of its events, with their ``result`` values;
    - job control: its declared job-control flags by name and value, such as whether a paused job
      resumes after its part is reloaded.
    A resource that declares none of these has no capabilities entry.
    """
    entry = prepared.get("recovery_resources", {}).get(jid, {})
    model = session.get("recovery_des_models", {}).get(jid) or entry.get("recovery_des_model") or {}
    items = []
    if locations := _declared_locations(entry):
        items.append("reachable locations " + ", ".join(locations))
        names = resource_names(prepared)
        resource = names.get(jid, jid)
        for location in locations:
            if positions := _placement_positions(location, resource, prepared, names):
                items.append(f"places parts on {location} at {', '.join(positions)}")
    if jid in prepared.get("diagnostic_reach", {}):
        items.append("workspace bounds")
    if coordinates := _coordinate_parameters(model):
        items.append(f"moves to any position ({', '.join(coordinates)})")
    if positions := _advance_order(model, resource_names(prepared).get(jid, jid)):
        items.append("moves parts in one direction: " + " → ".join(positions))
    processes: dict[str, set[str]] = {}
    for event in model.get("events", []):
        bindings = event.get("parameter_bindings", {})
        process, result = (bindings.get("process") or {}).get("equals"), (bindings.get("result") or {}).get("equals")
        if isinstance(process, str):
            processes.setdefault(process, set()).update([result] if isinstance(result, str) else [])
    items += [f"{process} ({', '.join(sorted(results))})" if results else process
              for process, results in sorted(processes.items())]
    job_control = (entry.get("static_capabilities") or {}).get("job_control") or {}
    if flags := [f"{name} {_value_text(value)}" for name, value in job_control.items() if isinstance(value, bool)]:
        items.append("job control: " + ", ".join(flags))
    return "capabilities: " + "; ".join(items) if items else ""


def _displayed_fields(jid: str, session: dict) -> dict:
    """Return the values a resource line displays: declared public fields that hold a value, and its part records.

    A variable that records parts (held_part, one whose declared values name parts, or a part's own entry
    such as part_location.<part>) is shown even when empty, so the record a part move changes is in view.
    A value an accepted event changed without stating it reads "not stated".
    """
    from .multi_turn_diagnostic import NOT_STATED

    row = session.get("symbolic_resources", {}).get(jid, {})
    parts = set(session.get("symbolic_parts", {}))
    not_stated = set(session.get("diagnostic_unknown_owner_fields", {}).get(jid, []))
    shown = {}
    for name, declaration in session.get("recovery_des_models", {}).get(jid, {}).get("state_variables", {}).items():
        if (declaration.get("private") or name.startswith(("task_ctx.", "_")) or name in _OWNER_BOOKKEEPING
                or name in _PART_FIELDS or declaration.get("scope") == "part" or name in {"resource_state", "pose"}):
            continue
        records_parts = (name == "held_part" or name.rpartition(".")[2] in parts
                         or bool(parts.intersection(value for value in declaration.get("domain") or [] if isinstance(value, str))))
        if name in not_stated:
            shown[name] = NOT_STATED
        elif row.get(name) is not None or records_parts:
            shown[name] = row.get(name)
    return shown


def _shown_values(resource_jids: list[str], part_names: list[str], session: dict) -> dict:
    """Return the values the Resources and Parts rows display, so context replies repeat none of them."""
    resources = {}
    for jid in resource_jids:
        row = session.get("symbolic_resources", {}).get(jid, {})
        values = {"resource_state": row.get("resource_state") or row.get("current_state"), **_displayed_fields(jid, session)}
        if _pose(row.get("current_pose")):
            values["current_pose"] = row["current_pose"]
        resources[jid] = values
    parts = {name: {key: row[key] for key in ("part_location", "part_holder_resource_jid", "observed_pose",
                                              "current_process", "processCompleted")
                    if key in row}
             for name in part_names for row in [session.get("symbolic_parts", {}).get(name, {})]}
    return {"resources": resources, "parts": parts}


def _placements(jid: str, session: dict) -> list[str]:
    """Return how each accepted event that moved a resource placed it, in order: by "pose" or by "location"."""
    placements = []
    for task in session.get("accepted_outline_prefix", []):
        end = task.get("expected_end_state") or {}
        if task.get("resource_jid") == jid and ("pose" in end or "resource_location" in end):
            placements.append("pose" if "pose" in end else "location")
    return placements


def current_values(jid: str, declarations: dict, session: dict) -> dict:
    """Return the current value of each resource-scoped variable, as the Resources section states it.

    Args:
        jid: The resource.
        declarations: Its public condition variables.
        session: Tentative planning state after the accepted events.

    Returns:
        Each resource-scoped variable's projected value. A value an accepted event changed without
        stating it is "not stated". ``pose`` is declared from the resource's current pose, so it reads
        that pose in its declared fields; once an event moves the resource by a location change
        alone, its pose is not stated.
    """
    from . import multi_turn_outline_generation as generation
    from .multi_turn_diagnostic import NOT_STATED

    row = session.get("symbolic_resources", {}).get(jid, {})
    not_stated = set(session.get("diagnostic_unknown_owner_fields", {}).get(jid, []))
    if _placements(jid, session)[-1:] == ["location"]:
        not_stated.add("pose")
    values = {}
    for name, declaration in declarations.items():
        if declaration.get("scope", "resource") != "resource":
            continue
        if name in not_stated:
            values[name] = NOT_STATED
        elif name == "pose":
            pose = row.get("current_pose") or {}
            values[name] = {field: pose[field] for field in declaration.get("fields", []) if field in pose} or None
        else:
            values[name] = deepcopy(generation._exact_state_field_value(row, name))
    return values


def _resource_row(jid: str, prepared: dict, session: dict, names: dict[str, str]) -> str:
    """State one resource on one line with the same kinds of facts as every other resource."""
    entry = prepared.get("recovery_resources", {}).get(jid, {})
    row = session.get("symbolic_resources", {}).get(jid, {})
    declarations = session.get("recovery_des_models", {}).get(jid, {}).get("state_variables", {})
    places, parts = declared_record_places(jid, prepared), set(session.get("symbolic_parts", {}))
    items, families = [], {}
    for field, value in _displayed_fields(jid, session).items():
        # Every field is shown under its variable name, the name an event states it by.
        if field == "held_part":
            items.append(f"held_part {names.get(value, value) if value else 'none'}")
        elif field == "resource_location":
            items.append(f"resource_location {names.get(value, value)}")
        elif "." in field:
            prefix, key = field.split(".", 1)
            # A per-part family shows only its set entries, or none.
            entries = families.setdefault(prefix, [])
            if value is not None and value is not False:
                entries.append(key if value is True else f"{key} {_value_text(value)}")
        else:
            items.append(f"{field} {_value_text(value)}" + (f" ({places[field]})" if field in places else ""))
    for prefix, keys in families.items():
        values = list(dict.fromkeys(value for name, declaration in declarations.items() if name.startswith(prefix + ".")
                                    for value in declaration.get("domain") or [] if isinstance(value, str) and value not in parts))
        items.append(f"{prefix}.<part name>: {', '.join(keys) or 'none'}" + (f" (values {', '.join(values)})" if values else ""))
    if _pose(row.get("current_pose")):
        # The latest event that moved the resource decides: a stated pose is where it is, a location
        # change leaves the last pose behind, which an earlier event stated or the capture measured.
        placements = _placements(jid, session)
        area = _area_text(row["current_pose"], prepared)
        label = "pose"
        if placements[-1:] == ["location"]:
            label = "last stated pose" if "pose" in placements else "last measured pose"
        items.append(f"{label} {_pose(row['current_pose'])}" + (f", {area}" if area else ""))
    from .multi_turn_diagnostic import machine_jobs

    # Each job by its part, so a paused job stays in view while another part is processed.
    if jobs := machine_jobs(jid, session, prepared):
        items.append("jobs: " + ", ".join(
            f"{job['part_name']} "
            + (_process_text({"process": job["process"], "result": job.get("result")}, _value_text(job.get("status")), names)
               if job.get("process")
               else _value_text(job.get("status")))
            + (f" (processing_fraction {_value_text(job['processing_fraction'])})"
               if job.get("processing_fraction") is not None else "") for job in jobs))
    if capabilities := _capabilities(jid, prepared, session):
        items.append(capabilities)
    unstated = "resource_state" in session.get("diagnostic_unknown_owner_fields", {}).get(jid, [])
    # A resource whose model declares no resource_state has no state word; its variables are its condition.
    status = [row.get("resource_state") or row.get("current_state") or ("state not stated" if unstated else "state not recorded")
              ] if "resource_state" in declarations else []
    if row.get("availability") == "unavailable":
        status.append("unavailable")
    head = f"{names.get(jid, jid)} ({entry.get('resource_type') or 'resource'})" + ("  " + ", ".join(status) if status else "")
    return head + ("; " + "; ".join(items) if items else "")


def _part_text(name: str, prepared: dict, session: dict, names: dict[str, str]) -> str:
    row = session.get("symbolic_parts", {}).get(name, {})
    holder = row.get("part_holder_resource_jid") or row.get("current_holder_resource_jid")
    location = row.get("part_location")
    pose = row.get("observed_pose") if row.get("observed_pose_current") is not False else None
    details = [f"held by {names.get(holder, holder)}" if holder else f"at {names.get(location, location)}" if location
               else "not held, not at any station",
               f"measured at {_pose(pose)}" if _pose(pose) else "", _area_text(pose, prepared)]
    text = ", ".join(item for item in details if item)
    processes = _processes_text(row.get("current_process"), row.get("processCompleted"), names)
    return text + (f"; {processes}" if processes else "")


def part_rows(prepared: dict, session: dict, names: dict[str, str]) -> list[str]:
    """State every part; parts with the same state and place share one line."""
    groups: dict[str, list[str]] = {}
    for name in sorted(session.get("symbolic_parts", {})):
        groups.setdefault(_part_text(name, prepared, session, names), []).append(name)
    return [f"{', '.join(members)}  {text}" for text, members in groups.items()]


def _ref_lines(prepared: dict) -> list[str]:
    """List the fact references, each described by what it returns."""
    from .multi_turn_context import diagnostic_refs

    recorded = {ref for ref, _ in diagnostic_refs(prepared)}
    relation = ("which areas contain its measured position (one point, not its full extent), which declared places "
                "contain it, and how far it lies from each resource's workspace bounds")
    rows = [("/resources/<name>/snapshot", "every recorded value of a resource; once your events change it, its "
             "expected values after them"),
            ("/resources/<name>/state_variables", "its condition variables, their allowed values, and current values"),
            ("/resources/<name>/pending_tasks", "its remaining planned tasks"),
            ("/resources/<name>/static_capabilities", "named poses, staging areas and process options"),
            *([("/resources/<name>/pose_relations", relation)] if "/resources/<jid>/pose_relations" in recorded else []),
            ("/parts/<name>/snapshot", "every recorded value of a part"),
            ("/parts/<name>/state_variables", "its condition variables and current values"),
            ("/parts/<name>/target", "where the part is assembled"),
            *([("/parts/<name>/pose_relations", relation)] if "/parts/<name>/pose_relations" in recorded else []),
            *([("/regions", "area bounds in world coordinates")] if "/regions/<name>" in recorded else []),
            *([("/geometry", "station poses and sizes, robot bases and floor extent")] if "/geometry/<key>" in recorded else [])]
    return [f"- {ref}: {text}" for ref, text in rows]


def next_task_lines(prepared: dict, session: dict) -> list[str]:
    """State the planned tasks normal production carries out next: the ones the handoff requires.

    They are the recorded tasks nominal work resumes at (``multi_turn_diagnostic.nominal_rows``), stated
    by resource, function, part and public parameters, the same way for every scenario. Their start
    conditions are not stated, since normal production may run its own steps before them.
    """
    from .multi_turn_diagnostic import nominal_rows

    names = resource_names(prepared)
    lines = []
    for row in nominal_rows(session, prepared):
        task = row["task"]
        text = f"- {names.get(task['resource_jid'], task['resource_jid'])}: {task['function_name']}"
        text += f" of {task['part_name']}" if task.get("part_name") else ""
        params = _params_text([task])
        lines.append(text + (f" ({params})" if params else ""))
    return lines


def outline_template(prepared: dict, session: dict) -> dict[str, Any]:
    """Build the outline prompt sections from recorded facts, stating every resource and part alike.

    Args:
        prepared: Recorded failure context (the capture's prepared recovery request).
        session: Tentative planning state after the accepted events.

    Returns:
        Lines for the Failure, Recovery Goal (with normal production's next tasks), Resources and Parts
        sections, the fact references,
        the resource names, and the displayed values that context replies must not repeat.
    """
    names = resource_names(prepared)
    resource_jids = sorted(prepared.get("recovery_resources", {}), key=lambda jid: names.get(jid, jid))
    part_names = sorted(session.get("symbolic_parts", {}))
    return {"failure": failure_lines(prepared, session), "goal": goal_lines(prepared, session),
            "next_tasks": next_task_lines(prepared, session),
            "resources": [_resource_row(jid, prepared, session, names) for jid in resource_jids],
            "parts": part_rows(prepared, session, names), "refs": _ref_lines(prepared),
            "resource_jids": resource_jids, "part_names": part_names, "resource_names": names,
            "shown": _shown_values(resource_jids, part_names, session)}
