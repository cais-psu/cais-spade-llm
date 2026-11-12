"""Add an accepted recovery's events to the nominal resource models after the recovery has run.

Supplementary demonstration only: "Model growth after recovery" in PAPER_FRAMEWORK.md, and the one narrow exception
to "No hidden rules in recovery events" in AGENTS.md. Outline and composition checks never use this module.

After an accepted recovery has run, each of its accepted events whose resource is still in production becomes a
commanded activity of that resource, so the nominal scheduler can use it again for later parts:

- The acting resource's side is the event as the LLM stated it: its stated start state as guards, its stated end
  state as updates, its name, and the steps the accepted composition gave it. The event's own ``part_name`` becomes
  the part parameter, which the scheduler binds on each use as it does for every nominal event. Nothing else of the
  stated event changes.
- Every other resource the event changes keeps its own declared side. Its model already holds the entries for a part
  put at or taken from its place by another resource; the one whose effects give the change the event stated, and
  whose conditions held when the accepted event ran, is copied unchanged, together with the other resources that take
  part in it. A value the event stated that this entry does not declare, such as a belt order, comes from that
  resource's own model on each use and is listed as not copied.

No resource, place or part is named here: everything comes from the accepted records and the declared models.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any

from cais_spade_llm.resources.environment_models import (
    PART_PARAMETER,
    validate_environment_composition,
)
from cais_spade_llm.resources.function_contracts import validate_function_contracts
from cais_spade_llm.resources.nominal_des import FINISH_SUFFIX

logger = logging.getLogger(__name__)

_MISSING = object()


class GrowthRefusal(ValueError):
    """An accepted event that cannot be added to the nominal models, with the fact that stops it."""


def resource_names(observations: dict) -> dict[str, str]:
    """Return each recovery jid's resource id from a capture's jid table.

    Args:
        observations: The capture's saved observations.

    Returns:
        ``{jid: resource id}``.
    """
    return {jid: rid for rid, jid in observations["jids"].items()}


def states_before_events(observations: dict, events: list[dict], names: dict[str, str]) -> list[dict]:
    """Return every resource's recorded values before each accepted event.

    The capture's values at the failure, with each earlier event's stated end values written in, as the outline
    projection states them. Only fields the capture records are written.

    Args:
        observations: The capture's saved observations.
        events: The accepted outline events, in order.
        names: ``{jid: resource id}``.

    Returns:
        One ``{resource id: values}`` per event, before that event.
    """
    values = deepcopy(observations["failure"]["resource_values_before"])
    states = []
    for event in events:
        states.append(deepcopy(values))
        for rid, stated in _stated_changes(event, names).items():
            for field, value in stated.items():
                if field in values.get(rid, {}):
                    values[rid][field] = _named(value, names)
    return states


def grow_models(
    models: dict[str, dict],
    composition: dict,
    names: dict[str, str],
    states: list[dict],
    in_production: set[str],
) -> dict:
    """Add the accepted events of resources in production to the models, in place.

    Args:
        models: Every resource's runtime model; the same dict objects are updated.
        composition: The accepted composition record (``accepted_outline_prefix`` and ``accepted_primitive_program``).
        names: ``{jid: resource id}``.
        states: Every resource's recorded values before each accepted event (``states_before_events``).
        in_production: The resources still in production.

    Returns:
        ``{"added": [...], "left_out": [...]}``: each added event with its event ids, copied entries and stated
        values not copied; each event left out with the reason.

    Raises:
        ValueError: The grown models fail the composition or function contract validation.
    """
    programs = {row["outline_id"]: row for row in composition.get("accepted_primitive_program") or []}
    grown = deepcopy(models)
    report: dict[str, list] = {"added": [], "left_out": []}
    # Fields each resource had a change stated for by an earlier accepted event, added or not.
    changed_before: dict[str, set[str]] = {}
    for index, event in enumerate(composition["accepted_outline_prefix"]):
        actor = names.get(event["resource_jid"], event["resource_jid"])
        row = {"outline_id": event["outline_id"], "event_name": event["event_name"], "resource_id": actor,
               "part_name": event.get("part_name")}
        earlier = set(changed_before.get(actor, ()))
        for rid, stated in _stated_changes(event, names).items():
            changed_before.setdefault(rid, set()).update(stated)
        if actor not in in_production:
            report["left_out"].append({**row, "reason": f"{actor} is not in production"})
            continue
        try:
            entries, details = _activity(grown, event, programs.get(event["outline_id"]), actor, names,
                                         states[index], composition.get("source_outline_draft"), earlier)
        except GrowthRefusal as exc:
            report["left_out"].append({**row, "reason": str(exc)})
            continue
        running = event["event_name"]
        domain = grown[actor]["state_variables"]["resource_state"]["domain"]
        if running not in domain:
            domain.append(running)
        for rid, rows in entries.items():
            grown[rid]["events"].extend(rows)
        report["added"].append({**row, **details})
    _complete(grown)
    for rid, model in grown.items():
        models[rid].clear()
        models[rid].update(model)
    return report


def exclude_reported_faults(runtime) -> list[str]:
    """Take every resource with a fault report out of production.

    The scheduler does not read fault reports. This uses its existing switch, as a device fault does
    (``ConveyorFault.trigger``): the resource joins ``unavailable_resources`` and its executors are cleared.

    Args:
        runtime: The live ``EnvironmentRuntime``.

    Returns:
        The resources taken out.
    """
    context = runtime.context
    with context.admission_lock:
        excluded = sorted(rid for rid in context.fault_reports if rid in context.resources)
        for rid in excluded:
            context.unavailable_resources.add(rid)
            context.resources[rid].executors.clear()
    return excluded


def grow_runtime(runtime, composition: dict, observations: dict) -> dict:
    """Grow a stopped runtime's models with an accepted recovery's events and bind their executors.

    Args:
        runtime: The live ``EnvironmentRuntime``, stopped, with no pending task.
        composition: The accepted composition record.
        observations: The capture the accepted outline was made from (its jids and failure values).

    Returns:
        The growth report, with the executors bound.

    Raises:
        RuntimeError: The runtime is running, or a pending task involves a resource whose model would change.
    """
    context = runtime.context
    names = resource_names(observations)
    events = composition["accepted_outline_prefix"]
    with context.admission_lock:
        if not runtime.stopped:
            raise RuntimeError("Grow the models only while the run is stopped")
        in_production = set(context.models) - set(context.unavailable_resources)
        grown = {rid: deepcopy(model) for rid, model in context.models.items()}
        report = grow_models(grown, composition, names, states_before_events(observations, events, names),
                             in_production)
        changed = {rid for added in report["added"] for rid in added["participants"]}
        # A pending task was prepared against its participants' model revisions; growing those would strand it.
        busy = sorted(task["task_id"] for task in context.pending_tasks.values()
                      if changed & set(context._task_participants(task)))
        if busy:
            raise RuntimeError(f"Pending tasks involve resources whose models would change: {busy}")
        for rid, model in grown.items():
            context.models[rid].clear()
            context.models[rid].update(model)
        for rid in changed:
            context.resources[rid].revision += 1
        context.revision += 1
    report["executors"] = bind_executors(runtime, composition, report)
    logger.info("Model growth added %s and left out %s", [row["event_name"] for row in report["added"]],
                [(row["event_name"], row["reason"]) for row in report["left_out"]])
    return report


def bind_executors(runtime, composition: dict, report: dict) -> list[dict]:
    """Bind an executor that runs the accepted composition's steps to each added event.

    The steps run on the acting resource's own primitive interface, whichever resource it is
    (``_primitive_interface``). An added event of a resource whose agent offers none is reported without an executor.

    Args:
        runtime: The live ``EnvironmentRuntime``.
        composition: The accepted composition record.
        report: ``grow_models``' report.

    Returns:
        One row per added event: its name, resource and whether an executor was bound, with the reason if not.
    """
    programs = {row["outline_id"]: row for row in composition["accepted_primitive_program"]}
    agents = {agent.agent_name: agent for agent in runtime.resource_agents}
    rows = []
    for added in report["added"]:
        rid, name = added["resource_id"], added["event_name"]
        interface = _primitive_interface(agents.get(rid))
        if interface is None:
            rows.append({"event_name": name, "resource_id": rid, "bound": False,
                         "reason": f"{rid}'s agent offers no interface that runs one primitive of a composed program"})
            continue
        program = programs[added["outline_id"]]
        runtime.context.resources[rid].bind_executor(
            name, _composed_executor(interface, added, program), _completed, validate_start=_ready)
        rows.append({"event_name": name, "resource_id": rid, "bound": True, "interface": type(interface).__name__})
    return rows


class _WorkerPrimitives:
    """Primitives of an agent that sends each one to its own worker with the task's parameters."""

    def __init__(self, agent) -> None:
        self.agent = agent

    def begin(self, task: dict, bindings: dict) -> None:
        """Name the task's part and places on every request of this program."""
        self.agent.kmr_primitives.task_parameters = bindings

    def end(self) -> None:
        """Clear the task's parameters after the program."""
        self.agent.kmr_primitives.task_parameters = None

    async def run(self, primitive: str, params: dict) -> dict:
        """Run one primitive and return its output with ``success``."""
        method = getattr(self.agent.kmr_primitives, primitive, None)
        if method is None:
            return {"success": False, "message": f"the resource has no primitive {primitive}"}
        return await method(**params)

    def attachment(self) -> dict | None:
        """Return the resource's current attachment evidence, or None."""
        return getattr(self.agent, "workflow_custody", None)


class _ControllerPrimitives:
    """Primitives of an agent that runs each one on its own controller, one motion at a time."""

    def __init__(self, agent) -> None:
        self.agent = agent

    def begin(self, task: dict, bindings: dict) -> None:
        """Name the task and its part on the controller, as the resource's nominal executor does."""
        controller = self.agent._controller
        controller._simulation_task_id = task["task_id"]
        controller._simulation_task_run_id = task.get("run_id")
        controller._simulation_task_part = bindings.get("part_name")
        controller._simulation_fault_request = None
        controller._simulation_fault_evidence = None

    def end(self) -> None:
        """Nothing is kept on the controller after the program."""

    async def run(self, primitive: str, params: dict) -> dict:
        """Run one primitive and return its output with ``success``."""
        lock = self.agent._robot_motion_lock
        if not lock.acquire(blocking=False):
            return {"success": False, "message": "the resource already has an active motion"}
        try:
            return await self.agent._execute_primitive(primitive, params)
        finally:
            lock.release()

    def attachment(self) -> dict | None:
        """This interface reports no attachment evidence; a program that reads it is refused at that step."""
        return None


def _primitive_interface(agent):
    """Return how an agent runs one primitive of a composed program, or None when it offers no such interface."""
    if agent is None:
        return None
    if getattr(agent, "kmr_primitives", None) is not None:
        return _WorkerPrimitives(agent)
    if callable(getattr(agent, "_execute_primitive", None)) and getattr(agent, "_controller", None) is not None:
        return _ControllerPrimitives(agent)
    return None


def composed_step_parameters(program: dict, index: int, part: str | None, live: dict, evidence: dict | None) -> dict:
    """Return one composed step's parameters for this use of the event.

    Numbers the accepted composition resolved stay as it saved them, except the values that execution itself
    produces: a reference to an event fact written by one of the program's own calculation steps takes that step's
    output from this run, and a reference to the acting resource's attachment evidence
    (``/resources/<jid>/resource_evidence/attachment/...``) takes the resource's current attachment. A literal equal
    to the event's own part names this use's part. References are read in both notations the composition accepts
    (``/a/b`` and ``a.b``).

    Args:
        program: The accepted composition's row for the event.
        index: The step's index.
        part: This use's part, or None for an event without a part.
        live: ``{event fact path: output}`` of this run's calculation steps so far.
        evidence: The acting resource's current attachment evidence, or None.

    Returns:
        Keyword arguments for the primitive.

    Raises:
        ValueError: A referenced live value is not available.
    """
    authored = program["primitive_steps"][index].get("params") or {}
    saved = program["composition_validation"]["trace"][index].get("resolved_params") or {}
    facts = _calculated_facts(program)
    attachment = _attachment_tokens(program)

    def value(written: Any, resolved: Any) -> Any:
        if isinstance(written, dict) and isinstance(written.get("context_ref"), str):
            reference = written["context_ref"]
            tokens = _reference_tokens(reference)
            for fact, path in facts:
                if tokens[:len(fact)] == fact:
                    if path not in live:
                        raise ValueError(f"{reference} is read before its calculation step ran")
                    return _walk(live[path], tokens[len(fact):], reference)
            if tokens[:len(attachment)] == attachment:
                if not evidence:
                    raise ValueError(f"{reference} has no current attachment evidence")
                return _walk(evidence, tokens[len(attachment):], reference)
            return deepcopy(resolved)
        if isinstance(written, dict) and isinstance(resolved, dict):
            return {key: value(written.get(key), item) for key, item in resolved.items()}
        if isinstance(written, list) and isinstance(resolved, list) and len(written) == len(resolved):
            return [value(item, saved_item) for item, saved_item in zip(written, resolved, strict=True)]
        if part is not None and program.get("part_name") is not None and written == program["part_name"]:
            return part
        return deepcopy(resolved)

    return {name: value(authored.get(name), resolved) for name, resolved in saved.items()}


def unreadable_part_facts(program: dict) -> list[str]:
    """Return the references of a program that name its part and cannot be read again for another part.

    A reference to one of the program's own calculation outputs or to the acting resource's attachment evidence is
    read again on each use. Any other reference that names the event's part was resolved once, for that part, from
    the facts served to the composition; an added event has no way to read it for the part of a later use.
    """
    part = program.get("part_name")
    if part is None:
        return []
    facts, attachment = _calculated_facts(program), _attachment_tokens(program)
    found = []
    for step in program["primitive_steps"]:
        for reference in _references(step.get("params")):
            tokens = _reference_tokens(reference)
            if (part in tokens and tokens[:len(attachment)] != attachment
                    and not any(tokens[:len(fact)] == fact for fact, _ in facts) and reference not in found):
                found.append(reference)
    return found


def _composed_executor(interface, added: dict, program: dict):
    steps = program["primitive_steps"]
    trace = program["composition_validation"]["trace"]
    if [step["primitive"] for step in steps] != [row["primitive"] for row in trace]:
        raise ValueError(f"{added['outline_id']}: the saved trace does not match the composed steps")

    async def execute(task: dict) -> dict:
        part = task["parameters"].get("part_name")
        bindings = {key: value for key, value in (("part_name", part),
                                                   *(added["places"].items())) if isinstance(value, str)}
        live: dict = {}
        rows: list[dict] = []
        started = time.time()

        def evidence(reason: str | None = None) -> dict:
            observations = {"primitive_results": rows, "started_at_unix": started, "completed_at_unix": time.time()}
            if reason is None:
                observations["custody"] = deepcopy(interface.attachment())
            return {"status": "completed" if reason is None else "failed", "resource_id": task["resource_id"],
                    "event_name": task["event_name"], "task_id": task["task_id"], "observations": observations,
                    **({} if reason is None else {"reason": reason})}

        interface.begin(task, bindings)
        try:
            for index, step in enumerate(steps):
                try:
                    params = composed_step_parameters(program, index, part, live, interface.attachment())
                except ValueError as exc:
                    rows.append({"index": index, "primitive": step["primitive"], "success": False,
                                 "message": str(exc)})
                    return evidence(f"step {index}: {exc}")
                began = time.time()
                output = await interface.run(step["primitive"], params)
                rows.append({"index": index, "primitive": step["primitive"], "params": params,
                             "seconds": round(time.time() - began, 1),
                             "success": output.get("success") is True, "message": output.get("message")})
                if trace[index].get("event_fact_path"):
                    live[trace[index]["event_fact_path"]] = {
                        key: deepcopy(item) for key, item in output.items() if key not in {"observations"}}
                if output.get("success") is not True:
                    return evidence(f"step {index} {step['primitive']}: {output.get('message')}")
        finally:
            interface.end()
        return evidence()

    return execute


def _completed(task: dict, evidence: dict) -> bool:
    rows = (evidence.get("observations") or {}).get("primitive_results")
    return (evidence.get("status") == "completed" and evidence.get("task_id") == task["task_id"]
            and evidence.get("resource_id") == task["resource_id"]
            and evidence.get("event_name") == task["event_name"]
            and isinstance(rows, list) and bool(rows) and all(row.get("success") is True for row in rows))


async def _ready(task: dict, valuation: dict, geometry: dict) -> bool:
    return bool(task.get("task_id") and valuation and geometry is not None)


def _activity(models: dict, event: dict, program: dict | None, actor: str, names: dict[str, str],
              before: dict, source_draft: str | None, earlier: set[str]) -> tuple[dict[str, list[dict]], dict]:
    """Build the added activity's entries, or refuse the event with the fact that stops it.

    ``earlier`` holds the actor's fields that an earlier accepted event stated a change for.
    """
    if program is None:
        raise GrowthRefusal("the accepted composition has no program for it")
    if unreadable := unreadable_part_facts(program):
        raise GrowthRefusal(f"its program reads {unreadable}, facts of its part that an added event cannot read "
                            "again for another part")
    part = event.get("part_name")
    start_state = event.get("expected_start_state") or {}
    end_state = event.get("expected_end_state") or {}
    declared = models[actor]["state_variables"]
    if "domain" not in declared.get("resource_state", {}):
        raise GrowthRefusal(f"{actor} declares no resource_state domain for its running state")
    running = event["event_name"]
    _check_addable(models, event, actor, earlier)
    own_start = {_templated(field, part): value for field, value in start_state.items()}
    own_end = {_templated(field, part): value for field, value in end_state.items()}
    guards = {field: _rule(value, part, names, "equals") for field, value in own_start.items() if field in declared}
    updates = {field: _rule(value, part, names, "set") for field, value in own_end.items() if field in declared}
    if "resource_state" not in updates:
        raise GrowthRefusal("its stated end state has no resource_state, so its finish would not end the activity")
    for field, rule in [*guards.items(), *updates.items()]:
        _check_domain(models[actor], field, rule)
    origin = _named(start_state.get("part_location"), names) if part is not None else None
    destination = _named(end_state.get("part_location"), names) if part is not None else None
    sides = [_declared_side(models, owner, stated, part, actor, (origin, destination), before, names)
             for owner, stated in _neighbor_changes(event, names).items()]
    start_copies: dict[str, dict] = {}
    finish_copies: dict[str, dict] = {}
    places: dict[str, str] = {}
    for side in sides:
        places[side["binding"]] = side["place"]
        for target, copies in ((start_copies, side["start"]), (finish_copies, side["finish"])):
            for rid, copy in copies.items():
                if rid in target and target[rid] != copy:
                    raise GrowthRefusal(f"two declared sides disagree for {rid}")
                target[rid] = copy
    bindings = {"resource_id": {"equals": actor}}
    if part is not None:
        bindings["part_name"] = deepcopy(PART_PARAMETER)
    bindings.update({binding: {"equals": place} for binding, place in places.items()})
    if part is not None:
        capability = ({"part_location": origin}, {"part_location": destination})
    else:
        capability = ({"resource_id": actor, "resource_state": start_state.get("resource_state", "any")},
                      {"resource_id": actor, "resource_state": end_state["resource_state"]})
    start_id = 1 + max(entry["event_id"] for model in models.values() for entry in model["events"])
    finish_id = start_id + 1
    notes = (f"Accepted recovery event {event['outline_id']} of {source_draft}, added after the recovery ran "
             "(model growth, supplementary demonstration).")
    source = {"outline_id": event["outline_id"], "source_outline_draft": source_draft}
    start_participants = [actor, *(rid for rid in start_copies if rid != actor)]
    finish_participants = [actor, *(rid for rid in finish_copies if rid != actor)]
    shared = {"event_name": running, "observable": True, "parameter_bindings": bindings,
              "product_effects": {"processCompleted": []}, "notes": notes, "recovery_source": source}
    start_entry = {**shared, "event_id": start_id, "controllable": True, "finish_event_id": finish_id,
                   "capability_transition": {"source": capability[0], "target": deepcopy(capability[0])},
                   "participants": start_participants, "function_name": running, "program_status": "implemented",
                   "completion_evidence": {}}
    finish_entry = {**shared, "event_id": finish_id, "event_name": running + FINISH_SUFFIX, "controllable": False,
                    "start_event_id": start_id,
                    "capability_transition": {"source": capability[0], "target": capability[1]},
                    "participants": finish_participants}
    program_steps = {"steps": deepcopy(program["primitive_steps"]),
                     "entry_state": start_state.get("resource_state", "any"),
                     "success_state": end_state["resource_state"]}
    entries: dict[str, list[dict]] = {}
    entries.setdefault(actor, []).append(deepcopy({
        **start_entry, "guards": guards, "updates": {"resource_state": {"set": running}},
        "program": program_steps, "program_key": running}))
    for rid, copy in start_copies.items():
        entries.setdefault(rid, []).append(deepcopy({**start_entry, **_owned(copy)}))
    finish_guards = {field: rule for field, rule in guards.items() if field in updates}
    finish_guards["resource_state"] = {"equals": running}
    entries[actor].append(deepcopy({**finish_entry, "guards": finish_guards, "updates": updates}))
    for rid, copy in finish_copies.items():
        entries.setdefault(rid, []).append(deepcopy({**finish_entry, **_owned(copy)}))
    details = {
        "event_ids": [start_id, finish_id],
        "participants": sorted({*start_participants, *finish_participants}),
        "places": places,
        "copied_from": {side["owner"]: side["copied_from"] for side in sides},
        "not_copied": {side["owner"]: side["not_copied"] for side in sides if side["not_copied"]},
        "stated_outside_the_model": sorted((set(own_start) | set(own_end)) - set(declared) - {"part_location"}),
    }
    return entries, details


def _check_addable(models: dict, event: dict, actor: str, earlier: set[str]) -> None:
    """Refuse an event whose added activity would not be the accepted event, or would clash with the models.

    Args:
        models: Every resource's model, with the events this growth has added so far.
        event: The accepted outline event.
        actor: The resource that performs it.
        earlier: The actor's fields that an earlier accepted event stated a change for.

    Raises:
        GrowthRefusal: The name is taken, it states values of its part the added event would not carry, a per-part
            value cannot be bound to its part, or its stated start leaves out what an earlier event changed.
    """
    running, part = event["event_name"], event.get("part_name")
    start_state = event.get("expected_start_state") or {}
    stated = {*start_state, *(event.get("expected_end_state") or {})}
    declared = models[actor]["state_variables"]
    # The runtime finds executors, feasibility rules and reservations by event name, whatever the resource.
    named = {entry["event_name"] for model in models.values() for entry in model["events"]}
    if running in named or running + FINISH_SUFFIX in named:
        raise GrowthRefusal(f"its name {running!r} already names an event of the models, and the runtime binds "
                            "executors and rules by event name")
    if running in declared["resource_state"]["domain"]:
        raise GrowthRefusal(f"its name {running!r} is already a resource_state value of {actor}")
    part_values = {field for model in models.values() for entry in model["events"]
                   for field in (*(entry.get("product_guards") or {}), *(entry.get("product_effects") or {}))}
    if uncarried := sorted(stated & part_values):
        raise GrowthRefusal(f"it states {uncarried} of its part, which the added event would not carry")
    if unbound := sorted(field for field in stated if _templated(field, part) not in declared
                         and "." in field and field.split(".", 1)[0] + ".{part_name}" in declared):
        raise GrowthRefusal(f"it states {unbound} of {actor}, a per-part value the added event cannot bind to its part")
    own_start = {_templated(field, part) for field in start_state}
    if unstated := sorted(field for field in earlier if field in declared and field not in own_start):
        raise GrowthRefusal(f"its stated start leaves out {unstated} of {actor}, which an earlier accepted event "
                            "changed; on its own it would be offered where the recovery never ran it")


def _declared_side(models: dict, owner: str, stated: dict, part: str | None, actor: str,
                   route: tuple[str | None, str | None], before: dict, names: dict[str, str]) -> dict:
    """Return the owner's own declared entries for the change the event states for it.

    ``route`` is where the event's part starts and ends, as the event states them.
    """
    origin, destination = route
    if owner not in models:
        raise GrowthRefusal(f"it states a change for {owner}, which has no model")
    if part is not None and origin not in (None, actor) and destination == actor:
        binding, place = "origin_resource_location", origin
    elif part is not None and origin == actor and destination not in (None, actor):
        binding, place = "destination_location", destination
    else:
        raise GrowthRefusal(f"it states a change for {owner} without moving its part to or from a place")
    wanted = {_templated(field, part): _named(value, names) for field, value in stated.items()}
    matches = []
    for finish in models[owner]["events"]:
        bindings = finish["parameter_bindings"]
        # Only the owner's own declared entries count, not copies this or an earlier growth added.
        if ("start_event_id" not in finish or "recovery_source" in finish
                or (bindings.get(binding) or {}).get("equals") != place
                or bindings["resource_id"]["equals"] == owner):
            continue
        given = {field: _given(rule, part) for field, rule in finish["updates"].items()}
        agreeing = [field for field in wanted if field in given and _same(given[field], wanted[field])]
        differing = [field for field in wanted if field in given and not _same(given[field], wanted[field])]
        start = _entry(models[owner], finish["start_event_id"])
        if agreeing and not differing and _holds(start["guards"], before.get(owner, {}), part):
            matches.append((start, finish, sorted(set(wanted) - set(given))))
    if not matches:
        raise GrowthRefusal(f"no entry of {owner}'s own model at {place} gives the change it states ({wanted}) "
                            "and held when the accepted event ran")
    if len(matches) > 1:
        ids = [[start["event_id"], finish["event_id"]] for start, finish, _ in matches]
        raise GrowthRefusal(f"several entries of {owner}'s own model at {place} give the change it states: {ids}")
    start, finish, not_copied = matches[0]
    nominal_actor = start["parameter_bindings"]["resource_id"]["equals"]
    return {
        "owner": owner, "binding": binding, "place": place,
        "copied_from": [start["event_id"], finish["event_id"]],
        "not_copied": {field: wanted[field] for field in not_copied},
        "start": {rid: _entry(models[rid], start["event_id"]) for rid in start["participants"]
                  if rid not in (nominal_actor, actor)},
        "finish": {rid: _entry(models[rid], finish["event_id"]) for rid in finish["participants"]
                   if rid not in (nominal_actor, actor)},
    }


def _complete(models: dict) -> None:
    """Recompute the alphabets, participants and neighbors as ``build_environment_models`` does, then validate."""
    for model in models.values():
        model["local_event_alphabet"] = list(dict.fromkeys(e["event_name"] for e in model["events"]))
        model["controllable_event_alphabet"] = list(
            dict.fromkeys(e["event_name"] for e in model["events"] if e.get("controllable") is True))
        model["observable_event_alphabet"] = list(
            dict.fromkeys(e["event_name"] for e in model["events"] if e.get("observable") is True))
    holders: dict[int, list[str]] = {}
    for rid, model in models.items():
        for event in model["events"]:
            holders.setdefault(event["event_id"], [])
            if rid not in holders[event["event_id"]]:
                holders[event["event_id"]].append(rid)
    for model in models.values():
        for event in model["events"]:
            event["participants"] = [rid for rid in models if rid in holders[event["event_id"]]]
    for model in models.values():
        model["neighbors"] = sorted({rid for event in model["events"] for rid in event["participants"]
                                     if rid != model["resource_id"]})
    validate_environment_composition(models)
    validate_function_contracts(models)


def _stated_changes(event: dict, names: dict[str, str]) -> dict[str, dict]:
    changes = {names.get(event["resource_jid"], event["resource_jid"]): dict(event.get("expected_end_state") or {})}
    for rid, stated in _neighbor_changes(event, names).items():
        changes.setdefault(rid, {}).update(stated)
    return changes


def _neighbor_changes(event: dict, names: dict[str, str]) -> dict[str, dict]:
    return {names.get(key, key): dict(stated) for key, stated in (event.get("neighbor_end_state") or {}).items()}


def _named(value: Any, names: dict[str, str]) -> Any:
    return names.get(value, value) if isinstance(value, str) else value


def _rule(value: Any, part: str | None, names: dict[str, str], operator: str) -> dict:
    value = _named(value, names)
    if part is not None and value == part:
        return {f"{operator}_from_param": "part_name"}
    return {operator: deepcopy(value)}


def _check_domain(model: dict, field: str, rule: dict) -> None:
    (operator, value), = rule.items()
    domain = model["state_variables"][field].get("domain")
    if operator.endswith("_from_param") or domain is None:
        return
    if not any(type(item) is type(value) and item == value for item in domain):
        raise GrowthRefusal(f"{model['resource_id']}.{field} = {value!r} is outside its declared domain")


def _templated(field: str, part: str | None) -> str:
    if part is not None and field.endswith("." + part):
        return field[: -len(part)] + "{part_name}"
    return field


def _given(rule: dict, part: str | None) -> Any:
    if set(rule) == {"set"}:
        return rule["set"]
    if rule == {"set_from_param": "part_name"} and part is not None:
        return part
    return _MISSING


def _same(left: Any, right: Any) -> bool:
    return left is not _MISSING and type(left) is type(right) and left == right


def _holds(guards: dict, values: dict, part: str | None) -> bool:
    for field, guard in guards.items():
        bound = field.replace("{part_name}", part) if part is not None else field
        (operator, expected), = guard.items()
        if operator.endswith("_from_param"):
            if expected != "part_name" or part is None:
                return False
            expected = part
        actual = values.get(bound, _MISSING)
        if actual is _MISSING:
            return False
        equal = type(actual) is type(expected) and actual == expected
        if (operator.startswith("not_equals") and equal) or (operator.startswith("equals") and not equal):
            return False
    return True


def _entry(model: dict, event_id: int) -> dict:
    return next(entry for entry in model["events"] if entry["event_id"] == event_id)


def _owned(copy: dict) -> dict:
    """Return a participant's own side of a declared entry: what is not shared between copies."""
    return {key: deepcopy(copy[key]) for key in ("guards", "updates", "collection_guards", "collection_effects")
            if key in copy}


def _references(value: Any) -> list[str]:
    """Return every reference an authored parameter value holds."""
    if isinstance(value, dict):
        if isinstance(value.get("context_ref"), str):
            return [value["context_ref"]]
        return [reference for item in value.values() for reference in _references(item)]
    if isinstance(value, list):
        return [reference for item in value for reference in _references(item)]
    return []


def _reference_tokens(reference: str) -> list[str]:
    """Split a reference as ``resolve_composition_value`` does: by ``/`` when it starts with one, else by ``.``."""
    return reference[1:].split("/") if reference.startswith("/") else reference.split(".")


def _calculated_facts(program: dict) -> list[tuple[list[str], str]]:
    """Return the event facts the program's own calculation steps write, as reference tokens with their path."""
    return [(row["event_fact_path"].split(".", 2), row["event_fact_path"])
            for row in program["composition_validation"]["trace"] if row.get("event_fact_path")]


def _attachment_tokens(program: dict) -> list[str]:
    return ["resources", program["resource_jid"], "resource_evidence", "attachment"]


def _walk(value: Any, tokens: list[str], reference: str) -> Any:
    for token in tokens:
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and token.isdigit() and int(token) < len(value):
            value = value[int(token)]
        else:
            raise ValueError(f"{reference} is not in this run's output")
    return deepcopy(value)
