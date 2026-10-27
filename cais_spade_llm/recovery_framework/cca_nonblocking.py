from __future__ import annotations

"""CCA nonblocking validation of a declared recovery: Eq. 6 of the recovery paper.

When the LLM declares the recovery complete, the CCA asks whether normal production can carry on
from the state the accepted recovery events leave. Normal production is the runtime resource models,
the ones Eq. 13 admission and the product agents use. They describe each resource's capabilities,
not the scenario's routes, so a recovery may leave a part off its usual route when some resource can
carry it on from there. The CCA composes the parts with work left and the resources their
continuations use, starting from the projected end state, and accepts when that state lies in the
composition's nonblocking region: whatever completions occur, every such part can still reach its
remaining requirement and every resource it uses can still return to rest.

The composition is searched to the end, with no step bound. Its state budget only protects the shared
machine: running out of it is an error, never a verdict. No specification monitors are composed: the
selected predefined specifications use physical propositions the composition cannot evaluate, and
the runtime CCA's monitor for them is empty too. Resources are checked through the parts with work
left; when no part has work left, nothing is composed.

A refusal states only what normal production cannot continue (user decision, October 26): a resource
left in a state its model does not have, a part that cannot reach its remaining requirement from
where it is, a resource that cannot carry on from where it is, or parts that cannot finish together.
"""

import json
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.central_controller.local_composition import (
    AnalysisLimit,
    Budget,
    IncompleteModel,
    analyze_state,
)
from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor
from cais_spade_llm.product.environment import EnvironmentProductContext
from cais_spade_llm.recovery_framework.environment_composition import EnvironmentPlant
from cais_spade_llm.resources.environment_models import (
    _process_effects,
    matches_requirement,
    owners,
    product_state_from_custody,
)
from cais_spade_llm.resources.nominal_conveyor import CONVEYOR_LOCATIONS
from cais_spade_llm.resources.nominal_des import FINISH_SUFFIX

NONBLOCKING, BLOCKING = "NONBLOCKING", "BLOCKING"
# The search is exhaustive; these limits protect the shared machine and never decide the verdict.
SEARCH_BUDGET = {"max_states": 200_000, "seconds": 900.0}
# The events the runtime binds an executor to (workflow_execution.bind_environment_executors).
_ROBOT_EVENTS = frozenset({"pick_approach", "pick_grasp", "place_approach", "place_insert", "place_release",
                           "move_home"})
_KMR_EVENTS = frozenset({"pick_approach", "pick_part", "move_to_resource", "place_approach", "place_release"})
_WORKER_EVENTS = frozenset({"machine_part", "complete_job", "select_program", "start_job", "pause_job",
                            "resume_job", "stop_job", "advance_conveyor", "advance_part"})
_STATE_FIELDS = ("held_part", "approached_part", "part_name", "staging_part", "resource_location")


def capture_inputs(observations: dict) -> dict:
    """Return the capture facts the check rebuilds the runtime models from.

    Args:
        observations: Saved failure capture, as read for an outline draft.

    Returns:
        The recorded runtime inputs (scene, order and geometry), the resources normal production may
        use, each resource's JID and the resources the capture reports unavailable.
    """
    return {"inputs": deepcopy(observations["inputs"]),
            "permitted_resources": list(observations["nominal_permitted_resources"]),
            "jids": dict(observations["jids"]),
            "unavailable_resources": sorted(observations.get("unavailable_resources") or [])}


def check_completion(session: dict, prepared: dict, capture: dict) -> dict:
    """Judge whether normal production can carry on from the state the accepted events leave.

    Args:
        session: Tentative planning state when the LLM declares the recovery complete.
        prepared: Recorded failure context.
        capture: The capture facts from :func:`capture_inputs`.

    Returns:
        ``nonblocking_status`` (NONBLOCKING or BLOCKING), the refusal ``reasons`` (empty when
        accepted), the ``blocked`` resources and parts, and the search ``evidence``.

    Raises:
        RuntimeError: The composition could not be built or did not finish within its budget.
    """
    context, invalid = _end_context(session, prepared, capture)
    if invalid:
        return {"nonblocking_status": BLOCKING, "reasons": [reason for _, reason, _ in invalid],
                "blocked": sorted({name for name, _, _ in invalid if name}),
                "evidence": {"invalid_state": [detail for _, _, detail in invalid]}}
    parts = _parts_with_work_left(session, prepared, context)
    if not parts:
        return {"nonblocking_status": NONBLOCKING, "reasons": [], "blocked": [],
                "evidence": {"parts": [], "explored_states": 0}}
    checker = OnlineSafetyMonitor({}, [])
    plant = EnvironmentPlant(context.calculation_snapshot(), checker, capture["jids"],
                             {part: context.requirements[part][-1] for part in parts})
    budget = Budget(**SEARCH_BUDGET)
    try:
        scope = plant.select_parts(set(parts), checker, budget)
        analysis = analyze_state(plant, checker, {}, scope, budget=budget)
    except AnalysisLimit as exc:
        raise RuntimeError(f"The CCA nonblocking check did not finish within its search budget ({exc})") from exc
    except IncompleteModel as exc:
        raise RuntimeError(f"The CCA nonblocking check could not compose the runtime models ({exc})") from exc
    evidence = {"parts": parts, "explored_states": len(analysis.nodes), "seconds": analysis.elapsed,
                "included_resources": sorted(scope.resources), "included_parts": sorted(scope.products),
                "completion_conditions": deepcopy(scope.goals),
                "missing_capabilities": _missing_capabilities(plant.missing_capabilities, context),
                "specification_monitors": "none composed"}
    if analysis.status == "nonblocking":
        return {"nonblocking_status": NONBLOCKING, "reasons": [], "blocked": [], "evidence": evidence}
    reasons, blocked = _blocked(context, parts, scope.goals, [node.state for node in analysis.nodes.values()])
    return {"nonblocking_status": BLOCKING, "reasons": reasons, "blocked": blocked, "evidence": evidence}


def refusal_findings(check: dict[str, Any]) -> list[dict]:
    """Return the one outline validation finding that refuses the declared completion."""
    return [{"phase": "outline", "decision": "recovery_complete", "constraint_code": "cca_nonblocking",
             "validation_category": "CCA", "validation_status": check["nonblocking_status"],
             "reason": "; ".join(check["reasons"])}]


def _end_context(session: dict, prepared: dict, capture: dict) -> tuple[EnvironmentProductContext, list]:
    """Rebuild the runtime models and set them to the state the accepted events leave.

    Each resource takes the projected values its model declares; values no event stated keep the
    grounded failure-time values. Bookkeeping the model derives and recovery events do not state is
    derived as the model does: a robot an event acted on keeps only the destination it is positioned
    at, and a belt's order follows its parts' positions unless an event stated it. Each part takes its
    projected location and process record, plus the declared outcome of every planned task the
    recovery stood in for, since a robot's recovery event records no process result. Faulted
    resources are unavailable. A value an event left unknown is reported before anything else.

    Returns:
        The context, and each way the state is not one the models have, as (name, reason, detail).
    """
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_diagnostic import (
        faulted_resources,
    )

    context = EnvironmentProductContext(**capture["inputs"], permitted_resources=capture["permitted_resources"])
    names = {jid: rid for rid, jid in capture["jids"].items()}
    grounded = prepared["code_grounding"]["resources"]
    unknown = session.get("diagnostic_unknown_owner_fields", {})
    robots = {row["resource_id"] for row in context.inputs["scene"]["robots"]}
    invalid = []
    for rid, resource in context.resources.items():
        jid = capture["jids"][rid]
        row = session["symbolic_resources"][jid]
        base = session.get("base_symbolic_resources", {}).get(jid, row)
        acted = row != base
        values = {}
        for field in resource.model["current_valuation"]:
            if field in unknown.get(jid, []):
                invalid.append((rid, f"{rid} is in a state normal production does not have ({field} not stated)",
                                f"{rid}.{field} not stated"))
                values[field] = deepcopy(grounded[rid].get(field))
            elif field in row:
                values[field] = names.get(row[field], row[field]) if isinstance(row[field], str) else deepcopy(row[field])
            elif field.startswith("task_ctx.") and acted:
                values[field] = None
            else:
                values[field] = deepcopy(grounded[rid][field])
        # place_approach leaves a robot positioned at the place it set as its destination.
        if acted and values.get("resource_state") == "positioned" and "task_ctx.destination_location" in values:
            values["task_ctx.destination_location"] = values.get("resource_location")
        orders = [field for field in values if field.startswith("part_order.")]
        if orders and all(row.get(field) == base.get(field) for field in orders):
            _derive_belt_order(values)
        resource.valuation = values
        events = _ROBOT_EVENTS if rid in robots else _KMR_EVENTS if rid == "KMR" else _WORKER_EVENTS
        for event_name in sorted(events & set(resource.model["local_event_alphabet"])):
            resource.bind_executor(event_name, _not_executed, _not_executed, validate_start=_not_executed)
    for part, state in context.part_tracker.items():
        row = session["symbolic_parts"][part]
        process = deepcopy(row.get("current_process"))
        if isinstance(process, dict) and process.get("machine") in names:
            process["machine"] = names[process["machine"]]
        location = row.get("part_location")
        state.update(location=names.get(location, location), current_process=process,
                     processCompleted=deepcopy(row.get("processCompleted") or []))
    _apply_claimed_outcomes(session, prepared, context)
    context.unavailable_resources = set(capture["unavailable_resources"]) | {
        names[jid] for jid in faulted_resources(session, prepared) if jid in names}
    if invalid:
        # A state with values no event stated cannot be checked for coherence; report only those.
        return context, invalid
    valuation = context.snapshot()
    errors: list[tuple[str | None, str]] = []
    custody = owners(context.models, valuation, context.part_tracker, context.product_name, errors=errors)
    for rid in dict.fromkeys(rid for rid, _ in errors if rid):
        messages = [message for name, message in errors if name == rid]
        # A value outside its domain is quoted, so a stated "none" is not read as no value.
        odd = {message.removeprefix(f"Invalid resource value: {rid}.") for message in messages
               if message.startswith(f"Invalid resource value: {rid}.")}
        invalid.append((rid, f"{rid} is in a state normal production does not have "
                             f"({_state_text(valuation[rid], odd)})", "; ".join(messages)))
    invalid += [(None, f"the state the events leave is not one normal production has ({message})", message)
                for rid, message in errors if not rid]
    for part, owner in custody.items():
        if owner == context.product_name:
            continue
        location = context.part_tracker[part].get("location")
        expected = product_state_from_custody(part, owner, valuation[owner], context.product_name)["location"]
        if location != expected:
            invalid.append((part, f"{part} is recorded at {location} while {owner} holds it",
                            f"{part}: location {location}, custody {expected}"))
    return context, invalid


def _derive_belt_order(values: dict) -> None:
    """Rank the parts on a belt from their positions, the leading (most downstream) part first.

    This is the model's ``part_order``; parts sharing an intermediate region keep their recorded
    relative order. A position the belt does not have is left for the coherence check to report.
    """
    resident = [field.removeprefix("part_location.") for field, value in values.items()
                if field.startswith("part_location.") and value in CONVEYOR_LOCATIONS]
    recorded = {part: values.get(f"part_order.{part}") for part in resident}
    ordered = sorted(resident, key=lambda part: (
        -CONVEYOR_LOCATIONS.index(values[f"part_location.{part}"]),
        recorded[part] if type(recorded[part]) is int else len(resident), part))
    for field in values:
        if field.startswith("part_order."):
            part = field.removeprefix("part_order.")
            values[field] = ordered.index(part) if part in ordered else None


def _not_executed(*_args, **_kwargs) -> None:
    """Stand in for a controller: the composition only needs to know which events can be dispatched."""
    return None


def _apply_claimed_outcomes(session: dict, prepared: dict, context: EnvironmentProductContext) -> None:
    """Add the declared process results of each planned task the recovery claimed it completed."""
    nodes = {node.get("id"): node for node in prepared.get("saved_pending_nodes", [])}
    for identifier in session.get("diagnostic_replaced_task_claims", {}):
        node = nodes.get(identifier, {})
        params = node.get("params", {})
        rid, part = params.get("resource_id"), params.get("part_name")
        if rid not in context.models or part not in context.part_tracker:
            continue
        done = context.part_tracker[part]["processCompleted"]
        for event in context.models[rid]["events"]:
            if event["event_name"] not in {node.get("function_name"), str(node.get("function_name")) + FINISH_SUFFIX}:
                continue
            if any(name in params and params[name] != binding["equals"]
                   for name, binding in event["parameter_bindings"].items()
                   if isinstance(binding, dict) and "equals" in binding):
                continue
            try:
                results = _process_effects(event, params)
            except KeyError:
                # The recorded task does not name the value its result takes.
                continue
            done.extend(result for result in results if result not in done)


def _missing_capabilities(entries: set[str], context: EnvironmentProductContext) -> list[str]:
    """Keep the plant's gaps worth review: a finish has no executor because its start dispatches it."""
    names = {(rid, str(event["event_id"])): event["event_name"]
             for rid, model in context.models.items() for event in model["events"]}
    return sorted(entry for entry in entries if not (
        entry.endswith(":executor")
        and names.get(tuple(entry.split(":")[:2]), "").endswith(FINISH_SUFFIX)))


def _parts_with_work_left(session: dict, prepared: dict, context: EnvironmentProductContext) -> list[str]:
    """Return the order's parts the recovery touched or the remaining plan names, whose work is unfinished."""
    base, end = session.get("base_symbolic_parts", {}), session.get("symbolic_parts", {})
    planned = {node.get("params", {}).get("part_name") for node in prepared.get("saved_pending_nodes", [])}
    touched = {part for part, row in end.items() if _part_facts(row) != _part_facts(base.get(part, {}))}
    return sorted(part for part in (planned | touched) & set(context.requirements)
                  if not matches_requirement(context.part_tracker[part], context.requirements[part][-1]))


def _part_facts(row: dict) -> tuple:
    return row.get("part_location"), row.get("current_process"), row.get("processCompleted") or []


def _blocked(context: EnvironmentProductContext, parts: list[str], goals: list[dict],
             reached: list[dict]) -> tuple[list[str], list[str]]:
    """Name what normal production cannot finish: parts and resources no reached state completes."""
    reasons, blocked = [], []
    for goal in goals:
        part = goal.get("part_name")
        if part in parts and not any(matches_requirement(state["products"][part], goal["operation"])
                                     for state in reached):
            reasons.append(f"{part} cannot reach its remaining requirement ({_requirement_text(context, part)}) "
                           f"from {_place_text(context, part)}")
            blocked.append(part)
    for goal in goals:
        rid = goal.get("resource_id")
        if rid and not any(_released(state["resources"][rid], goal["release"]) for state in reached):
            reasons.append(f"{rid} cannot carry on normal production from where it is "
                           f"({_state_text(context.resources[rid].valuation)})")
            blocked.append(rid)
    if not reasons:
        reasons.append(f"normal production cannot finish {' and '.join(parts)} together from where they are")
        blocked = list(parts)
    return reasons, blocked


def _released(values: dict, conditions: list[dict]) -> bool:
    return any(all(values.get(key) == expected["equals"] for key, expected in condition.items())
               for condition in conditions)


def _requirement_text(context: EnvironmentProductContext, part: str) -> str:
    """State a part's unfinished processPlan steps in order, e.g. ``trim → circle, then assembly at T``."""
    steps = [step for step in context.requirements[part]
             if not matches_requirement(context.part_tracker[part], step)]
    return ", then ".join(" and ".join(_process_text(item) for item in step.get("processesToComplete", [step]))
                          for step in steps)


def _process_text(item: dict) -> str:
    if item.get("process") == "assembly" and item.get("target"):
        return f"assembly at {item['target']}"
    if item.get("result"):
        return f"{item['process']} → {item['result']}"
    return str(item.get("process", item))


def _place_text(context: EnvironmentProductContext, part: str) -> str:
    """Name where a part is, with its belt position or buffer zone when it rests on one."""
    location = context.part_tracker[part].get("location")
    if location is None:
        return "no named place"
    values = context.resources[location].valuation if location in context.resources else {}
    detail = values.get(f"part_location.{part}") or next(
        (field.removesuffix("_part") for field, value in values.items()
         if field.startswith("zone_") and value == part), None)
    return f"{location} {detail}" if detail else str(location)


def _state_text(values: dict, quoted: set[str] = frozenset()) -> str:
    """State a resource's control state the way the outline prompt does: ``positioned; held_part none``.

    A belt or buffer also states the parts it carries, by position, order or zone. Fields in ``quoted``
    hold a value the model does not have; they are always shown, with the value as JSON.
    """
    def text(field: str) -> str:
        value = values[field]
        return json.dumps(value) if field in quoted else "none" if value is None else str(value)

    shown = [field for field in values if field in _STATE_FIELDS or field in quoted or (
        field.startswith(("part_location.", "part_order.", "zone_")) and values[field] is not None)]
    state = "resource_state"
    return "; ".join([text(state) if state in quoted else str(values.get(state)),
                      *(f"{field} {text(field)}" for field in shown if field != state)])
