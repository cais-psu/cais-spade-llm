from __future__ import annotations

"""Tentative symbolic bookkeeping for the existing multi-turn diagnostic mode."""

import math
import re
from copy import deepcopy
from typing import Any

_CONDITION_FIELDS = {
    ("resource", "current_state"): "resource_state", ("resource", "location"): "resource_location",
    ("resource", "held_part"): "held_part", ("part", "state"): "part_state", ("part", "location"): "part_location",
}
_SHARED_AREA_FORMULA = re.compile(r"^G\s*!\s*\(\s*(\w+)\s*&\s*(\w+)\s*\)$")


def _task_nodes(prepared: dict) -> dict:
    return {node["id"]: node for node in prepared.get("saved_pending_nodes", []) if node.get("id")}


def _suffixes(prepared: dict, role: str) -> list[dict]:
    return [row for row in prepared.get("marked_reentry_context", {}).get("pending_suffixes", [])
            if row.get("role") == role]


def replaced_tasks(prepared: dict) -> set[str]:
    """Read replacement obligations from the captured handoff, without a route."""
    return {identifier for row in _suffixes(prepared, "recovery_replaced") for identifier in row.get("pending_task_ids", [])}


def llm_selects_next_event(session: dict) -> bool:
    """Return whether no outline validator is enabled, so the LLM chooses each event."""
    from cais_spade_llm.recovery_framework.outline_checks import selected_checks

    return session.get("outline_diagnostic_bypass") is True and not any(selected_checks(session).values())


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
    name = _CONDITION_FIELDS.get((kind, row.get("field")), row.get("field"))
    if not state or name in session.get("diagnostic_unknown_owner_fields", {}).get(row.get("entity"), []):
        return False
    return comparison._exact_state_field_value(state, name) == row.get("expected")


def custody_findings(task: dict, session: dict) -> list[dict]:
    """Reject handling a part held by another resource, or a second part.

    A task-level event may combine pick and place. Only resources that declare
    ``held_part`` are checked; geometry, reach and feasibility remain unchecked.
    """
    part_name, jid = task.get("part_name"), task["resource_jid"]
    if not part_name or "held_part" not in session.get("recovery_des_models", {}).get(jid, {}).get("state_variables", {}):
        return []
    resources = session.get("symbolic_resources", {})
    holder = next((other for other, row in resources.items() if row.get("held_part") == part_name), None)
    held = resources.get(jid, {}).get("held_part")
    before = session.get("symbolic_parts", {}).get(part_name, {}).get("part_location")
    end = task.get("expected_end_state", {})
    handles = end.get("held_part") == part_name or ("part_location" in end and end["part_location"] != before)
    reasons = []
    if handles and holder not in {None, jid}:
        reasons.append(f"{part_name} is held by {holder}")
    if handles and held not in {None, part_name}:
        reasons.append(f"{jid} already holds {held}")
    return [{"constraint_owner": "resource", "constraint_code": "custody_discontinuity", "reason": reason}
            for reason in reasons]


def _inside(pose: dict, bounds: list) -> bool:
    return all(isinstance(pose.get(axis), (int, float)) and low <= pose[axis] <= high
               for axis, (low, high) in zip("xyz", bounds))


def handoff_conflicts(session: dict, prepared: dict) -> list[dict]:
    """Check selected shared-area rules for the state in which nominal work resumes.

    A resource without a proposed event keeps its measured occupancy; a moved
    resource occupies a region only through its projected resource_location.
    Other formulas are not evaluated here.
    """
    from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind

    from . import multi_turn_outline_generation as comparison

    resources, _ = comparison._projected_outline_validation_context(session_state=session, prepared_recovery_request=prepared)
    moved = {task.get("resource_jid") for task in session.get("accepted_outline_prefix", [])}
    entering = {}
    for row in nominal_rows(session, prepared):
        params = row["task"]["params"]
        for location in {params.get("destination_location"), params.get("origin_resource_location")} - {None}:
            entering.setdefault(location, set()).add(row["task"]["resource_jid"])
    conflicts = []
    for record in prepared.get("diagnostic_safety_specifications", []):
        document = record["document"]
        for rule in document["catalog"]["specifications"]:
            labels = _SHARED_AREA_FORMULA.match(rule.get("formula", "").strip())
            aps = {ap["label"]: ap for ap in rule.get("aps", [])}
            if not labels or any(label not in aps or physical_ap_kind(aps[label]) != "resource_region"
                                 for label in labels.groups()):
                continue
            for scope in document.get("requirement_scopes", []):
                region = scope.get("region")
                if scope.get("specification") != rule["id"] or not region:
                    continue
                bounds = prepared.get("diagnostic_regions", {}).get(region, {}).get("bounds")
                occupants = set(entering.get(region, set()))
                for jid, row in resources.items():
                    pose = prepared.get("recovery_resources", {}).get(jid, {}).get("recovery_snapshot", {}).get("current_pose") or {}
                    if comparison._exact_state_field_value(row, "resource_location") == region or (
                            jid not in moved and bounds and _inside(pose, bounds)):
                        occupants.add(jid)
                if len(occupants) > 1:
                    conflicts.append({"rule_id": rule["id"], "region": region, "resource_jids": sorted(occupants),
                                      "reason": f"{rule['id']}: {', '.join(sorted(occupants))} would occupy {region} when nominal work resumes"})
    return conflicts


def prompt_context(session: dict, prepared: dict, fault: dict, *, outline: bool) -> dict:
    """Collect the recorded handoff facts that a diagnostic prompt states.

    Args:
        session: Tentative planning state.
        prepared: Recorded failure context.
        fault: The prompt's fault event.
        outline: Whether condition values are compared with the projected state.

    Returns:
        Selected checks, handoff conditions (with ``holds`` on outline turns), blocker reasons once
        every condition holds, compact pending nominal tasks, the interrupted event, and the
        diagnostic refs that have recorded data.
    """
    from cais_spade_llm.recovery_framework.outline_checks import selected_checks

    from .multi_turn_context import diagnostic_refs

    conditions = [{**{key: row.get(key) for key in ("role", "entity_kind", "entity", "field", "expected")},
                   **({"holds": condition_holds(row, session, prepared)} if outline else {})}
                  for row in recorded_conditions(prepared)]
    blocked = (handoff_conflicts(session, prepared)
               if outline and conditions and all(row["holds"] for row in conditions) else [])
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
    return {"checks": selected_checks(session), "conditions": conditions,
            "blocked": [row["reason"] for row in blocked], "pending_tasks": tasks,
            "interrupted": interrupted, "refs": diagnostic_refs(prepared)}


def _within_reach(pose: dict, reach: dict) -> bool:
    origin = reach.get("origin_pose") or {}
    tolerance = reach.get("tolerance_m", 0.0)
    values = (pose.get("x"), pose.get("y"), pose.get("z"), origin.get("x"), origin.get("y"),
              reach.get("max_xy_radius_m"), reach.get("z_min_m"), reach.get("z_max_m"), tolerance)
    if not all(isinstance(value, (int, float)) for value in values):
        return False
    return (math.hypot(pose["x"] - origin["x"], pose["y"] - origin["y"]) <= reach["max_xy_radius_m"] + tolerance
            and reach["z_min_m"] - tolerance <= pose["z"] <= reach["z_max_m"] + tolerance)


def pose_relations(pose: dict, regions: dict, locations: dict[str, set[str]], envelopes: dict[str, dict]) -> dict:
    """Relate one recorded pose to declared regions, locations and pose envelopes.

    Every resource is treated alike: it relates through the locations or the envelope
    its declarations record, and one that records neither is reported as not
    evaluated. Nothing here selects a resource or judges whether an action is feasible.

    Args:
        pose: World-frame position with x, y and z.
        regions: Region name to its recorded record with ``bounds``.
        locations: Every resource JID to its declared location tokens, possibly empty.
        envelopes: Resource JID to its declared ``gripper_reach``.

    Returns:
        Containing regions, resources declaring a location that owns one of them, resources
        whose envelope contains the pose, and resources declaring neither.
    """
    from cais_spade_llm.recovery_framework.failure_grounding import _regions_for

    containing = {name: row["bounds"] for name, row in regions.items()
                  if isinstance(row, dict) and row.get("bounds") and _inside(pose, row["bounds"])}
    by_location = {jid: sorted(name for name in names if _regions_for(name, containing))
                   for jid, names in sorted(locations.items())}
    return {"containing_regions": sorted(containing),
            "by_declared_location": {jid: names for jid, names in by_location.items() if names},
            "by_declared_envelope": sorted(jid for jid, reach in envelopes.items() if _within_reach(pose, reach)),
            "not_evaluated": sorted(jid for jid, names in locations.items() if not names and jid not in envelopes)}


def nominal_rows(session: dict, prepared: dict) -> list[dict]:
    """Bind saved resume_suffix entries to recorded nominal declarations."""
    nodes = _task_nodes(prepared)
    rows = []
    for suffix in _suffixes(prepared, "resume_suffix"):
        identifier = suffix.get("entry_task_id")
        if not identifier:
            identifier = next(iter(suffix.get("pending_task_ids", [])), None)
        node = nodes.get(identifier, {})
        if not node:
            continue
        params = node.get("params", {})
        jid = suffix.get("resource_jid") or prepared.get("observation_jids", {}).get(params.get("resource_id"))
        model = session.get("recovery_des_models", {}).get(jid, {})
        for event in model.get("events", []):
            if event.get("event_name") != node.get("function_name"):
                continue
            if any(name in params and "equals" in binding and params[name] != binding["equals"]
                   for name, binding in event.get("parameter_bindings", {}).items() if isinstance(binding, dict)):
                continue
            rows.append({"event_id": "nominal_entry:" + identifier,
                "task": {"task_id": identifier, "resource_jid": jid, "part_name": params.get("part_name"),
                         "function_name": node["function_name"], "params": deepcopy(params)},
                "ra_event": deepcopy(event), "tool": {}})
            break
    return rows


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


def apply_tentative_effects(task: dict, session: dict, prepared: dict) -> None:
    """Apply explicit claims and record declaration-based completion as assumptions."""
    from . import multi_turn

    part_name = task.get("part_name")
    before = session.get("symbolic_parts", {}).get(part_name, {}).get("part_location")
    multi_turn._apply_task_effects_to_symbolic_state(task, session)
    unknown = session.get("diagnostic_unknown_owner_fields", {}).get(task["resource_jid"], [])
    session.setdefault("diagnostic_unknown_owner_fields", {})[task["resource_jid"]] = [name for name in unknown if name not in task["expected_end_state"]]
    after = session.get("symbolic_parts", {}).get(part_name, {}).get("part_location")
    if before != after:
        for location in (before, after):
            jid = prepared.get("observation_jids", {}).get(location)
            if jid and jid != task["resource_jid"]:
                resource = session.get("symbolic_resources", {}).get(jid, {})
                for field in ("part_name", "resource_state", "current_state", "active_job"):
                    resource.pop(field, None)
                session.setdefault("diagnostic_unknown_owner_fields", {})[jid] = ["part_name", "resource_state", "active_job"]
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
    nodes = _task_nodes(prepared)
    for identifier in replaced_tasks(prepared):
        node = nodes.get(identifier, {})
        params = node.get("params", {})
        if not node or params.get("part_name") != part_name:
            continue
        for event in model.get("events", []):
            if event.get("event_name") != node.get("function_name"):
                continue
            bindings = event.get("parameter_bindings", {})
            if any(name in bindings and _fixed(bindings[name], params) != params[name]
                   for name in ("part_name", "process", "result") if name in params):
                continue
            updates = {name: value for name, value in event.get("updates", {}).items()
                       if name != "active_job" and isinstance(value, dict) and "set" in value}
            if not updates or not all(end.get(name) == value["set"] for name, value in updates.items()):
                continue
            completed = session.setdefault("diagnostic_replaced_task_claims", {})
            completed[identifier] = {"outline_id": task["outline_id"], "resource_jid": task["resource_jid"],
                "basis": "unvalidated proposed outcome matching recorded task effects"}
            product = session.get("symbolic_parts", {}).get(part_name, {})
            for key, values in event.get("product_effects", {}).items():
                if not isinstance(values, list):
                    continue
                claims = [{name: _fixed(value, params) for name, value in row.items()} for row in values]
                existing = product.setdefault(key, [])
                if isinstance(existing, list):
                    existing.extend(row for row in claims if row not in existing)
            break


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
        if matches:
            enabled.add(row["event_id"])
    session["diagnostic_handoff_assumptions"] = assumptions
    return enabled


def handoff(session: dict, prepared: dict) -> dict:
    """Describe a proposed nominal handoff without asserting feasibility or safety."""
    from . import multi_turn_outline_generation as comparison

    required = replaced_tasks(prepared)
    remaining = required - set(session.get("diagnostic_replaced_task_claims", {}))
    rows = comparison._nominal_reentry_event_rows(session_state=session, prepared_recovery_request=prepared)
    entries = nominal_entry_ids(session, prepared)
    expected = {row["event_id"] for row in rows}
    recorded = recorded_conditions(prepared)
    unmet = [deepcopy(row) for row in recorded if not condition_holds(row, session, prepared)]
    # Recorded handoff conditions state what the interrupted work must leave
    # true; its intermediate owner states need not be reproduced.
    conditions_met = not unmet if recorded else bool(required or expected) and not remaining
    ready = conditions_met and bool(expected) and expected <= entries
    conflicts = handoff_conflicts(session, prepared) if ready else []
    return {"complete": ready and not conflicts, "conditions_met": ready,
        "unmet_handoff_conditions": unmet, "safety_conflicts": conflicts,
        "remaining_replaced_task_ids": sorted(remaining), "proposed_nominal_entry_ids": sorted(entries),
        "required_nominal_entry_ids": sorted(expected), "assumptions": deepcopy(session.get("diagnostic_handoff_assumptions", [])),
        "basis": ("recorded handoff conditions and shared-area rules on tentative symbolic state; "
                  "complete nominal continuation not checked"), "execution_authorized": False}


def unresolved_conditions(session: dict, prepared: dict) -> set[str]:
    """Return open recovery obligations used by candidate comparison.

    Conditions already true at the failure are required again at handoff but are
    not obligations, so a held part may be set down temporarily.
    """
    if recorded_conditions(prepared):
        return {_condition_id(row) for row in recorded_conditions(prepared, "unmet_reentry_conditions")
                if not condition_holds(row, session, prepared)}
    state = handoff(session, prepared)
    return {"recorded_task:" + name for name in state["remaining_replaced_task_ids"]} | {
        "nominal_entry:" + name for name in set(state["required_nominal_entry_ids"]) - set(state["proposed_nominal_entry_ids"])}


async def handle_event(*, session: dict, prepared: dict, response: dict, planner: Any) -> tuple[str, dict]:
    """Handle one candidate with only the requested outline checks enabled."""
    from . import multi_turn
    from cais_spade_llm.recovery_framework.outline_checks import selected_checks, validate_partial_event

    task, findings = multi_turn._derive_candidate_outline_task(candidate_task=response["next_transition"],
        session_state=session, prepared_recovery_request=prepared)
    if findings or task is None:
        return "need_revision", {"validation_findings": findings, "schema_valid": False}
    task = multi_turn._commit_selected_candidate_task(task=task,
        sequence_index=multi_turn._next_recovery_sequence_index(session))
    findings = custody_findings(task, session)
    if findings:
        return "need_revision", {"validation_findings": findings, "schema_valid": False}
    audit = {}
    try:
        checks, ra, cca = await validate_partial_event(task=task, session=session, prepared=prepared,
                                                      planner=planner, turn=audit)
    except (KeyError, TypeError, ValueError) as exc:
        checks, ra, cca = {}, {}, {}
        findings = [{"constraint_owner": "planner", "constraint_code": "outline_validation_unavailable", "reason": str(exc)}]
    else:
        findings = [{"constraint_owner": "cca" if name == "cca" else "resource",
            "constraint_code": "outline_validation_unavailable" if status == "NEEDS_CONTEXT" else "outline_validation_rejected",
            "reason": (cca.get("reason") if name == "cca" else
                       ra.get("geometry_evidence", {}).get("reason") or "; ".join(ra.get("geometry_evidence", {}).get("unresolved", []))
                       if name == "physical_evidence" else ra.get("reason")) or "Required outline check did not pass",
            "validation_status": status}
            for name, status in checks.items() if status not in {"passed", "not_checked", "not_selected"}]
    if findings:
        missing = any(row.get("validation_status") == "NEEDS_CONTEXT" for row in findings) or not checks
        session.update(status="NEEDS_CONTEXT" if missing else "validation_rejected",
                       outline_validation_findings=findings, outline_check_results=checks)
        return "need_context" if missing else "need_revision", {**audit, "validation_findings": findings,
            "schema_valid": True, "transition_validation": {"status": "unresolved" if missing else "rejected"},
            "checks": checks, "execution_authorized": False}
    working = deepcopy(session)
    apply_tentative_effects(task, working, prepared)
    if selected_checks(session)["check_physical_evidence"]:
        from .multi_turn_validated_event import _project

        try:
            _project(task, ra, cca or {"monitor_context_after": session.get("outline_monitor_context", {})}, working)
        except ValueError as exc:
            finding = {"constraint_owner": "resource", "constraint_code": "outline_validation_unavailable", "reason": str(exc)}
            session.update(status="NEEDS_CONTEXT", outline_validation_findings=[finding])
            return "need_context", {**audit, "schema_valid": True, "validation_findings": [finding], "checks": checks}
    else:
        working["derived_resource_snapshots"] = deepcopy(working.get("symbolic_resources", {}))
        if cca.get("is_safe") is True:
            working["outline_monitor_context"] = deepcopy(cca.get("monitor_context_after") or {})
            if cca.get("state_only_projection"):
                working["projected_state_evidence"] = deepcopy(cca["state_only_projection"])
    proposed = handoff(working, prepared)
    if proposed["complete"]:
        from cais_spade_llm.recovery_framework.outline_checks import validate_partial_handoff

        handoff_checks, findings = await validate_partial_handoff(session=working, prepared=prepared, planner=planner)
        if findings:
            session.update(status="NEEDS_CONTEXT" if any(row.get("validation_status") == "NEEDS_CONTEXT" for row in findings)
                           else "validation_rejected", outline_validation_findings=findings,
                           outline_check_results=checks)
            return "need_context" if session["status"] == "NEEDS_CONTEXT" else "need_revision", {
                **audit, "schema_valid": True, "checks": checks, "validation_findings": findings,
                "nominal_handoff_checks": handoff_checks, "transition_validation": {"status": "unresolved"}}
        audit["nominal_handoff_checks"] = handoff_checks
    working.setdefault("accepted_outline_prefix", []).append(deepcopy(task))
    working["outline_check_results"] = checks
    session.update(working)
    multi_turn._sync_des_recovery_aliases(session)
    proposed = handoff(session, prepared)
    session["diagnostic_nominal_handoff"] = proposed
    if proposed["conditions_met"] and proposed["safety_conflicts"]:
        session.setdefault("diagnostic_handoff_safety_blocks", []).append(
            {"outline_id": task["outline_id"], "conflicts": deepcopy(proposed["safety_conflicts"])})
    session["status"] = "ready_for_primitive_generation" if proposed["complete"] else "paused_after_outline_turn"
    return ("outline_ready" if proposed["complete"] else "need_next_task"), {
        **audit, "schema_valid": True, "next_transition": deepcopy(task), "checks": checks,
        "transition_validation": {"status": "not_checked"},
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
