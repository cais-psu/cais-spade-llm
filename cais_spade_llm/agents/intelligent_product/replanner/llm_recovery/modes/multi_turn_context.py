from __future__ import annotations

"""Select and retrieve public recovery facts without changing validator context."""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)

_RESOURCE_FIELDS = {
    "resource_jid", "resource_type", "current_state", "resource_state", "held_part",
    "held_part_location", "current_location", "resource_location", "current_pose",
    "current_pose_captured_at", "current_pose_source", "projected_pose", "last_observed_pose", "gripper_state", "controller_ready", "named_poses", "availability",
}
# Recorded fields a projected resource snapshot keeps beside the resource's declared variables.
_PROJECTED_SNAPSHOT_FIELDS = {
    "resource_jid", "resource_type", "controller_ready", "named_poses", "availability", "current_pose",
    "current_pose_source", "current_pose_captured_at", "projected_pose", "last_observed_pose",
}
_PART_FIELDS = {
    "part_name", "current_location", "part_location", "current_process",
    "current_holder_resource_jid", "part_holder_resource_jid", "observed_pose",
    "origin_location", "location", "location_basis", "processCompleted",
    "observed_pose_current", "last_observed_pose", "projected_pose",
}
_CAPABILITY_FIELDS = {
    "resource_type", "reachability", "reachable_locations", "staging_areas",
    "named_poses", "supports_executable_recovery",
    "supports_manipulator_pick_place", "capabilities", "availability",
    "process_capabilities", "program_options", "current_configuration",
    "job_control", "processing_observations",
}
_PRIVATE_CONTEXT_FIELDS = {
    "evaluator_data", "reference_recovery", "successful_outlines", "fixture_comments",
    "nominal_sequence", "route_hints", "delivery_bindings", "primitive_support",
    "execution_primitive_catalog", "primitive_catalog", "private_owner_records",
    "operator_metadata", "operator_feedback", "recovery_feedback",
    "gripper_reach", "additional_bindings", "max_xy_radius_m", "z_min_m", "z_max_m",
}


def check_context_revision(prepared: dict, planner: Any) -> None:
    """Reject a response computed from a replaced live observation context."""
    if prepared.get("offline_recovery"):
        return
    context = getattr(getattr(getattr(planner, "product_agent", None), "environment_runtime", None), "context", None)
    revision = prepared.get("source_context_revision")
    if context is not None and revision is not None and context.revision != revision:
        raise ValueError("NEEDS_CONTEXT: live observations changed; regenerate recovery from the current capture")


async def _observe_resource_pose(prepared: dict, session: dict, jid: str, planner: Any) -> dict:
    from cais_spade_llm.resources.resource_profile import resource_snapshot_field_value

    from .multi_turn import _resource_agent_map

    if jid not in prepared.get("recovery_resources", {}):
        raise ValueError(f"NEEDS_CONTEXT: unknown resource {jid}")
    if session.get("observation_count", 0) >= session.get("max_observations", 0):
        raise ValueError("NEEDS_CONTEXT: observation budget exhausted")
    owner = _resource_agent_map(planner).get(jid)
    snapshot_fn = getattr(owner, "get_recovery_snapshot", None)
    if not callable(snapshot_fn):
        raise ValueError(f"NEEDS_CONTEXT: {jid} observation interface is unavailable")
    snapshot = await asyncio.to_thread(snapshot_fn)
    pose = resource_snapshot_field_value(snapshot, "current_pose")
    captured_at = resource_snapshot_field_value(snapshot, "current_pose_captured_at")
    # A configured position without controller feedback is not a fresh pose.
    if not pose or captured_at is None:
        raise ValueError(f"NEEDS_CONTEXT: {jid} fresh current_pose is unavailable")
    row = {"resource_jid": jid, "interface": "get_recovery_snapshot", "output": deepcopy(pose),
           "recorded_at": captured_at}
    for key in ("symbolic_resources", "base_symbolic_resources"):
        session.setdefault(key, {}).setdefault(jid, {"resource_jid": jid}).update(
            current_pose=deepcopy(pose), current_pose_captured_at=captured_at)
    session.setdefault("observation_history", []).append(row)
    session["observation_count"] = session.get("observation_count", 0) + 1
    return deepcopy(row)


async def _observe_missing(prepared: dict, session: dict, ref: str, planner: Any) -> dict:
    from .multi_turn import _handle_grounding_phase

    if prepared.get("offline_recovery") or session.get("accepted_outline_prefix"):
        raise ValueError(f"NEEDS_CONTEXT: {ref} requires a new observation capture")
    segments = ref.split("/")
    request = None
    if len(segments) == 4 and segments[1] == "parts" and segments[3] == "observed_pose":
        if segments[2] not in _views(prepared, session)[1]:
            raise ValueError(f"NEEDS_CONTEXT: unknown part {segments[2]}")
        request = {"fact_type": "part_pose", "entity": segments[2]}
    elif len(segments) == 4 and segments[1] == "resources" and segments[3] == "current_pose":
        return await _observe_resource_pose(prepared, session, segments[2], planner)
    if request is None:
        raise ValueError(f"NEEDS_CONTEXT: {ref} is not available")
    count = len(session.get("observation_history", []))
    _, evidence = await _handle_grounding_phase(session_state=session, prepared_recovery_request=deepcopy(prepared),
        planner=planner, parsed_response={"decision": "observe", "observe_requests": [request]})
    if len(session.get("observation_history", [])) == count:
        raise ValueError(evidence.get("error") or f"NEEDS_CONTEXT: {ref} observation was not obtained")
    row = session["observation_history"][-1]
    row["recorded_at"] = datetime.now(timezone.utc).isoformat()
    return deepcopy(row)


def _fields(value: dict, names: set[str]) -> dict:
    return {key: _public_fact_value(item) for key, item in value.items()
            if key in names and key not in _PRIVATE_CONTEXT_FIELDS}


def _public_fact_value(value: Any) -> Any:
    """Retain factual values while omitting private annotations at any depth."""
    if isinstance(value, dict):
        return {key: _public_fact_value(item) for key, item in value.items()
                if key not in _PRIVATE_CONTEXT_FIELDS}
    if isinstance(value, list):
        return [_public_fact_value(item) for item in value]
    return deepcopy(value)


def _views(prepared: dict, session: dict) -> tuple[dict, dict]:
    from .multi_turn_prompts import _projected_outline_parts, _projected_outline_resources

    resources = {row["resource_jid"]: row for row in _projected_outline_resources(
        llm_input=prepared.get("llm_input", {}), session_state=session)}
    parts = {row["part_name"]: row for row in _projected_outline_parts(
        llm_input=prepared.get("llm_input", {}), session_state=session)}
    physical = session.get("projected_physical_evidence") or {}
    for rid, job in physical.get("unresolved_processing", {}).items():
        row = resources.get(physical.get("jids", {}).get(rid), {})
        for field in ("active_job", "job_state", "resource_state", "current_state"):
            row.pop(field, None)
        part = parts.get(job.get("part"), {})
        for field in ("processCompleted", "processCompleted_evidence"):
            part.pop(field, None)
        part["processCompleted_complete"] = False
    return resources, parts


def _part_names(value: Any) -> set[str]:
    if isinstance(value, list):
        return set().union(*(_part_names(row) for row in value)) if value else set()
    if not isinstance(value, dict):
        return set()
    names = {value[key] for key in ("part_name", "held_part") if isinstance(value.get(key), str)}
    if value.get("entity_kind") == "part" and isinstance(value.get("entity"), str):
        names.add(value["entity"])
    for key in ("params", "parameters", "conditions", "goal_conditions", "expected_end_state"):
        names.update(_part_names(value.get(key)))
    return names


def _task_locations(task: dict) -> set[str]:
    params = task.get("params", task.get("parameters", {}))
    return {params[key] for key in ("destination_location", "origin_resource_location", "target_location",
                                   "source_resource", "target_resource")
            if isinstance(params.get(key), str)}


def _public_resource_capabilities(entry: dict) -> dict:
    declarations = (entry.get("recovery_des_model") or {}).get("state_variables", {})
    fields = _CAPABILITY_FIELDS - {name for name, row in declarations.items() if row.get("private")}
    recorded: dict[str, dict] = {}
    for source in ("static_capabilities", "recovery_adapter", "recovery_snapshot"):
        for name, value in _fields(entry.get(source) or {}, fields).items():
            recorded.setdefault(name, {})[source] = value
    result, field_sources, conflicts = {}, {}, {}
    for name, values in recorded.items():
        first = next(iter(values.values()))
        if all(type(value) is type(first) and value == first for value in values.values()):
            result[name] = deepcopy(first)
            field_sources[name] = list(values)
        else:
            conflicts[name] = deepcopy(values)
    if field_sources:
        result["field_sources"] = field_sources
    if conflicts:
        result["conflicts"] = conflicts
    return result


def _recorded_locations(entry: dict) -> set[str]:
    locations = set()
    for source in ("static_capabilities", "recovery_adapter", "recovery_snapshot"):
        for field in ("reachability", "reachable_locations"):
            values = (entry.get(source) or {}).get(field)
            if isinstance(values, list):
                locations.update(value for value in values if isinstance(value, str))
    return locations


def _resource_location_domain(entry: dict) -> list[str]:
    """Return the public declared ``resource_location`` domain without null."""
    declaration = (entry.get("recovery_des_model") or {}).get("state_variables", {}).get("resource_location") or {}
    if declaration.get("private"):
        return []
    return [value for value in declaration.get("domain") or [] if isinstance(value, str)]


def _public_capability_evidence(entry: dict) -> list[dict]:
    records = []
    for record in entry.get('capability_evidence') or []:
        row = _fields(record, {'resource_location', 'held_part', 'recorded_execution',
            'geometric_validation', 'ref', 'run_id', 'revision', 'source', 'source_fingerprint',
            'current_proposal_validated', 'conflicts', 'missing'})
        row['scope'] = _fields(record.get('scope') or {}, {'part_name', 'scene_fingerprint', 'scene_asset_fingerprint',
            'launch_id', 'attached', 'grasp_transform', 'carrying_arm_configuration'})
        row['checks'] = [_fields(check, {'success', 'checked_states', 'checked_samples', 'sample_step_m',
            'collision_checks_bypassed', 'allowed_relative_error_m', 'relative_error_m', 'ref'}) for check in record.get('checks', [])]
        row['measurements'] = [_fields(measurement, {'ref', 'base_pose', 'current_pose'})
                               for measurement in record.get('measurements', [])]
        records.append(row)
    return records


def resource_index(prepared: dict) -> list[dict]:
    """List declared alternatives without selecting a recovery action."""
    rows = []
    for jid, entry in sorted(prepared.get("recovery_resources", {}).items()):
        caps = _public_resource_capabilities(entry)
        detail_fields = {"staging_areas", "named_poses", "capabilities",
                         "process_capabilities", "program_options", "current_configuration", "processing_observations", "job_control"}
        row = {"resource_jid": jid, "resource_type": entry.get("resource_type"),
               **{name: value for name, value in caps.items() if name not in detail_fields | {"field_sources", "conflicts"}}}
        identities = [rid for rid, identity in prepared.get("observation_jids", {}).items() if identity == jid]
        if len(identities) == 1:
            row["resource_id"] = identities[0]
        available = sorted(detail_fields.intersection(caps) | detail_fields.intersection(caps.get("conflicts", {})))
        if available:
            row["available_capability_fields"] = available
        if caps.get("conflicts"):
            row["conflicts"] = deepcopy(caps["conflicts"])
        from cais_spade_llm.resources.environment_recovery import public_evaluation_support

        support = _public_fact_value(public_evaluation_support(entry))
        if support:
            row["evaluation_support"] = support
        evidence = _public_capability_evidence(entry)
        if evidence:
            summaries = {}
            for record in evidence:
                if not record.get('recorded_execution') and not record.get('conflicts'):
                    continue
                summary = {key: deepcopy(record[key]) for key in (
                    "resource_location", "held_part", "recorded_execution", "geometric_validation", "conflicts"
                ) if key in record}
                key = recovery_validation_fingerprint(summary)
                summaries.setdefault(key, {**summary, 'refs': []})['refs'].append(record['ref'])
            row["capability_evidence"] = list(summaries.values())
            row["capability_evidence_ref"] = f"/resources/{jid}/capability_evidence"
        rows.append(row)
    return rows


def select_outline_context(prepared: dict, session: dict) -> dict:  # noqa: C901, PLR0912 - explicit dependency closure
    """Include observed failure, goal, custody and pending-work dependencies."""
    resources, parts = _views(prepared, session)
    entries = prepared.get("recovery_resources", {})
    fault = prepared.get("llm_input", {}).get("fault_event", {})
    reasons: dict[str, list[str]] = {}
    selected_resources, selected_parts = set(), set()

    def include(kind: str, name: str, reason: str) -> None:
        available, selected = (resources, selected_resources) if kind == "resources" else (parts, selected_parts)
        if name not in available:
            return
        selected.add(name)
        rows = reasons.setdefault(f"/{kind}/{name}", [])
        if reason not in rows:
            rows.append(reason)

    include("resources", prepared.get("ra_jid", ""), "failed_resource_jid")
    include("resources", fault.get("focused_resource_jid", ""), "fault_event")
    for name in fault.get("affected_part_names", []):
        include("parts", name, "affected_part_names")
    for row in prepared.get("failure_anchor", {}).get("affected_parts", []):
        include("parts", row.get("part_name", ""), "affected_parts")
    for name in prepared.get("P_id", []):
        include("parts", name, "P_id")
    for name in _part_names(prepared.get("llm_input", {}).get("goal_conditions", [])):
        include("parts", name, "goal_conditions")
    if session.get("outline_diagnostic_bypass"):
        from .multi_turn_diagnostic import recorded_conditions

        # Diagnostic recovery ends at the recorded handoff, so its entities are the goal.
        for row in recorded_conditions(prepared):
            include("resources" if row.get("entity_kind") == "resource" else "parts", str(row.get("entity") or ""),
                    "handoff_conditions")
    for event in session.get("accepted_outline_prefix", []):
        include("resources", event.get("resource_jid", ""), "accepted_outline_prefix")
        for name in _part_names(event):
            include("parts", name, "accepted_outline_prefix")
    for ref in session.get("outline_requested_refs", []):
        segments = ref.split("/")
        if len(segments) >= 3 and segments[1] in {"resources", "parts"}:
            include(segments[1], segments[2], "context_requests")
    if not selected_resources and not selected_parts:
        for jid in resources:
            include("resources", jid, "failure relationships unavailable")
        for name in parts:
            include("parts", name, "failure relationships unavailable")

    # Reach a fixed point over custody and retained work. No proposed recovery
    # sequence or event ranking participates in choosing these facts.
    previous = None
    while previous != (selected_resources, selected_parts):
        previous = (set(selected_resources), set(selected_parts))
        task_ids = {task.get("id") for jid in selected_resources
                    for task in entries.get(jid, {}).get("pending_tasks", []) if task.get("id") is not None}
        task_links = {link for jid in selected_resources for task in entries.get(jid, {}).get("pending_tasks", [])
                      for field in ("predecessors", "successors") for link in task.get(field, []) if isinstance(link, (str, int))}
        locations = set().union(*(_task_locations(task) for jid in selected_resources
                                  for task in entries.get(jid, {}).get("pending_tasks", [])))
        locations.update(parts[name]["part_location"] for name in selected_parts
                         if isinstance(parts[name].get("part_location"), str))
        for jid, row in sorted(resources.items()):
            if _recorded_locations(entries.get(jid, {})) & locations:
                include("resources", jid, "recorded locations intersect part or pending-task locations")
            held = row.get("held_part")
            if held in selected_parts:
                include("resources", jid, "held_part")
            if jid in selected_resources and isinstance(held, str):
                include("parts", held, "held_part")
            for task in entries.get(jid, {}).get("pending_tasks", []):
                names = _part_names(task)
                links = {link for field in ("predecessors", "successors") for link in task.get(field, []) if isinstance(link, (str, int))}
                if (jid in selected_resources or names & selected_parts or _task_locations(task) & locations
                        or task.get("id") in task_links or links & task_ids):
                    include("resources", jid, "pending_tasks")
                    for name in names:
                        include("parts", name, "pending_tasks")
        for name in list(selected_parts):
            row = parts[name]
            for field in ("part_holder_resource_jid", "current_holder_resource_jid", "part_location", "current_location"):
                if isinstance(row.get(field), str):
                    include("resources", row[field], field)
    return {"resource_jids": sorted(selected_resources), "part_names": sorted(selected_parts),
            "inclusion_reasons": reasons, "resource_index": resource_index(prepared)}


def public_resource_snapshot(prepared: dict, session: dict, jid: str) -> dict:
    """Serve a resource's public declarations and state, excluding internal task and model fields.

    In a validation-off draft, a resource that an accepted event changed keeps only its declared
    variables and its identity, availability and pose records. Events update declared variables
    only, so its other recorded values describe the failure and would contradict its projected state.

    Args:
        prepared: Prepared recovery request holding the recorded resources.
        session: Planning session holding the projected state and the accepted events.
        jid: Resource to describe.

    Returns:
        The resource's public fields.

    Raises:
        ValueError: ``NEEDS_CONTEXT`` when the resource is unknown.
    """
    entries = prepared.get("recovery_resources", {})
    if jid not in entries:
        raise ValueError(f"NEEDS_CONTEXT: unknown resource {jid}")
    entry = entries[jid]
    fields, declared = set(_RESOURCE_FIELDS), set()
    for name, declaration in (entry.get("recovery_des_model") or {}).get("state_variables", {}).items():
        if declaration.get("private") or name.startswith(("task_ctx.", "_")):
            fields.discard(name)
        else:
            fields.add(name)
            if declaration.get("scope", "resource") == "resource":
                declared.add(name)
    resources, _ = _views(prepared, session)
    from .multi_turn_diagnostic import NOT_STATED

    # A value an accepted event changed without stating it is not stated, not its captured value.
    projected = resources.setdefault(jid, {})
    for name in session.get("diagnostic_unknown_owner_fields", {}).get(jid, []):
        projected[name] = NOT_STATED
        if name == "resource_state":
            projected["current_state"] = NOT_STATED
    changed = {name for task in session.get("accepted_outline_prefix", [])
               for name in (task.get("resource_jid"),
                            *(change.get("entity") for change in task.get("condition_changes") or []))}
    if session.get("outline_diagnostic_bypass") and jid in changed:
        fields &= declared | _PROJECTED_SNAPSHOT_FIELDS
    if session.get("validation_contract_version") == 2 and jid in session.get("derived_resource_snapshots", {}):
        if session.get("public_safety_context_version") != 1:
            return _fields(resources.get(jid, {}), fields)
        # New public context keeps projected values distinct from the unchanged
        # captured row. No state token is renamed or assigned through an alias.
        row = resources.get(jid, {})
        snapshot = _fields(row, fields & (declared | _PROJECTED_SNAPSHOT_FIELDS))
        sources = {key: "projected_state" if key in declared else "saved_observations" for key in snapshot}
        physical = session.get("projected_physical_evidence") or {}
        identities = [rid for rid, value in physical.get("jids", {}).items() if value == jid]
        if len(identities) == 1:
            observed = physical.get("snapshot", {}).get("resources", {}).get(identities[0], {})
            for key in fields & {"held_part", "gripper_state", "current_pose"} & observed.keys():
                snapshot[key] = deepcopy(observed[key])
                sources[key] = "projected_state"
            if "current_pose" in snapshot and sources["current_pose"] == "projected_state":
                snapshot["current_pose_source"] = "projected_state"
        if snapshot.get("current_pose_source") == "projected_state":
            sources["current_pose"] = "projected_state"
            snapshot.pop("current_pose_captured_at", None)
            sources.pop("current_pose_captured_at", None)
        if "projected_pose" in snapshot:
            sources["projected_pose"] = "projected_state"
        snapshot["field_sources"] = {key: source for key, source in sources.items() if key in snapshot}
        return snapshot
    result = {**entry.get("recovery_snapshot", {}), **resources.get(jid, {})}
    return _fields(result, fields)


def public_part_snapshot(prepared: dict, session: dict, part_name: str) -> dict:
    """Expose recorded part facts permitted by public part declarations."""
    _, parts = _views(prepared, session)
    if part_name not in parts:
        raise ValueError(f"NEEDS_CONTEXT: unknown part {part_name}")
    fields, private, declared = set(_PART_FIELDS), set(), set()
    for entry in prepared.get("recovery_resources", {}).values():
        for name, declaration in (entry.get("recovery_des_model") or {}).get("state_variables", {}).items():
            if declaration.get("scope") != "part":
                continue
            if declaration.get("private") or name.startswith(("task_ctx.", "_")):
                private.add(name)
            else:
                fields.add(name)
                declared.add(name)
    aliases = {"part_location": {"current_location", "location"},
               "part_holder_resource_jid": {"current_holder_resource_jid", "holder_resource_jid"}}
    for name in private:
        fields.difference_update({name, *aliases.get(name, ())})
    if session.get("validation_contract_version") == 2 and (
        parts[part_name].get("observed_pose_current") is False
        or any(row.get("part_name") == part_name for row in session.get("accepted_outline_prefix", []))
    ):
        if session.get("public_safety_context_version") != 1:
            return _fields(parts[part_name], fields)
        retained = {"part_name", "observed_pose", "last_observed_pose", "projected_pose", "observed_pose_current"}
        snapshot = _fields(parts[part_name], fields & (declared | retained))
        snapshot["field_sources"] = {key: "projected_state" if key in declared or key == "projected_pose"
            else "saved_observations" for key in snapshot}
        if snapshot.get("observed_pose_current") is False:
            snapshot["field_sources"]["observed_pose"] = "unavailable_after_projected_transition"
        return snapshot
    entry = prepared.get("grounding_context", {}).get("parts", {}).get(part_name, {})
    recorded = prepared.get("part_tracker", {}).get(part_name, {})
    snapshot = _fields({**recorded, **entry, **parts[part_name]}, fields)
    if ("observed_pose" in fields and not snapshot.get("observed_pose")
            and snapshot.get("observed_pose_current") is not False):
        snapshot["observed_pose"] = entry.get("observed_pose") or recorded.get("observed_pose")
    physical = session.get("projected_physical_evidence") or {}
    if any(job.get("part") == part_name for job in physical.get("unresolved_processing", {}).values()):
        snapshot.pop("processCompleted", None)
    return snapshot


_CONDITION_KEYS = {"scope", "domain", "domain_complete", "type", "description", "fields", "items"}
# How the listed values of a condition variable are read; shown with every variables lookup.
_CONDITION_VALUES_RULE = ("A variable whose values name places, parts or positions must name a known one; for "
                          "any other variable the domain lists the values normal production uses (domain_complete "
                          "refers to normal production), and new values are allowed.")


def _public_resource_state_variables(entry: dict) -> dict:
    return {name: _fields(row, _CONDITION_KEYS) for name, row in
            (entry.get("recovery_des_model") or {}).get("state_variables", {}).items()
            if not row.get("private") and not name.startswith(("task_ctx.", "_"))}


def _public_part_state_variables(prepared: dict, snapshot: dict) -> dict:
    """Return a part's condition variables: every part-scoped variable a resource declares.

    Each variable carries the part's current value. It does not list the declaring resources: every
    resource declares a part's location, which would read as if a machine could move a part.
    """
    variables: dict[str, dict] = {}
    for _, entry in sorted(prepared.get("recovery_resources", {}).items()):
        for name, row in _public_resource_state_variables(entry).items():
            if row.get("scope") != "part":
                continue
            merged = variables.setdefault(name, {**{key: deepcopy(value) for key, value in row.items() if key != "domain"},
                                                 "domain": []})
            merged["domain"].extend(value for value in row.get("domain") or [] if value not in merged["domain"])
    for name, row in variables.items():
        row["current_value"] = deepcopy(snapshot.get(name))
    return variables


def _held_part_interpretation(entry: dict, declarations: dict) -> dict:
    held_part = declarations.get("held_part")
    if not isinstance(held_part, dict) or held_part.get("scope", "resource") != "resource" or "resource_state" not in declarations:
        return {}
    guards, updates = [], []
    for event in (entry.get("recovery_des_model") or {}).get("events", []):
        if not isinstance(event, dict) or event.get("private") or not isinstance(event.get("event_name"), str):
            continue
        event_updates = event.get("updates") or {}
        state_update = event_updates.get("resource_state")
        if not isinstance(state_update, dict) or not {"set", "set_from_param"}.intersection(state_update):
            continue
        held_update = event_updates.get("held_part")
        held_guard = (event.get("guards") or {}).get("held_part") or {}
        if held_update is None and not held_guard.get("private") and held_guard.get("equals_from_param") == "part_name":
            guard = {"equals_from_param": "part_name"}
            if guard not in guards:
                guards.append(guard)
        elif (isinstance(held_update, dict) and not held_update.get("private") and "set" not in held_update
              and held_update.get("set_from_param") == "part_name"):
            update = {"set_from_param": "part_name"}
            if update not in updates:
                updates.append(update)
    if not guards and not updates:
        return {}
    rules = ["These bindings apply only when that event matches the proposed successor."]
    if updates:
        rules.append("A declared set_from_param update may set held_part to grounded part_name.")
    if guards:
        rules.append("When held_part is preserved, an equals_from_param=part_name guard permits that exact grounded value outside the recorded domain.")
    rules.append("Other guards, bindings and domain checks still apply; this does not establish action acceptance or physical feasibility.")
    return {
        "source": "runtime_validation",
        "source_function": "cais_spade_llm.resources.recovery_feasibility._successor_bindings",
        "field": "held_part",
        "declared_part_name_guards": guards,
        "declared_part_name_updates": updates,
        "rule": " ".join(rules),
    }


def _public_pending_tasks(prepared: dict, session: dict, entry: dict) -> list[dict]:
    """Return a resource's remaining planned tasks with their public fields.

    In a validation-off draft, replaced work that needs a resource still reported faulted is left
    out: normal production cannot run it, so only the recovery replaces it.
    """
    replaced = set()
    if session.get("outline_diagnostic_bypass"):
        from .multi_turn_diagnostic import faulted_replaced_work

        replaced = faulted_replaced_work(session, prepared)
    return [{**_fields(task, {"id", "function_name", "status", "predecessors", "successors"}),
             "params": _fields(task.get("params", {}), {"part_name", "destination_location", "origin_resource_location"})}
            for task in entry.get("pending_tasks", []) if task.get("id") not in replaced]


def _resource_fact(prepared: dict, session: dict, jid: str, field: str, ref: str) -> tuple[Any, str]:
    entry = prepared.get("recovery_resources", {}).get(jid, {})
    snapshot = public_resource_snapshot(prepared, session, jid)
    if field == "snapshot":
        value = snapshot
    elif field == "static_capabilities":
        from .multi_turn_outline_facts import job_control_text

        value = _public_resource_capabilities(entry)
        # The lookup states job control in the same words as the resource's line, not by its flags' names.
        if "job_control" in value:
            value["job_control"] = job_control_text(value["job_control"]) or value["job_control"]
        physical = session.get("projected_physical_evidence") or {}
        if (physical.get("elapsed_time_obligations")
                or any(physical.get("jids", {}).get(rid) == jid for rid in physical.get("unresolved_processing", {}))):
            value.pop("processing_observations", None)
    elif field == "capability_evidence":
        value = _public_capability_evidence(entry)
    elif field == "pending_tasks":
        value = _public_pending_tasks(prepared, session, entry)
    elif field == "state_variables":
        value = _public_resource_state_variables(entry)
        if session.get("outline_diagnostic_bypass"):
            from .multi_turn_outline_facts import current_values

            # Validated prompts promise declarations only; drafts add current values, as parts do.
            for name, current in current_values(jid, value, session).items():
                value[name]["current_value"] = current
    elif field in snapshot:
        value = snapshot[field]
    else:
        reason = " is not established after the accepted projection" if (
            session.get("public_safety_context_version") == 1 and jid in session.get("derived_resource_snapshots", {})
        ) else " is not available"
        raise ValueError(f"NEEDS_CONTEXT: {ref}{reason}")
    source = "declaration" if field in {"static_capabilities", "state_variables"} else (
        "projected_state" if session.get("accepted_outline_prefix") and field not in {"current_pose", "current_pose_captured_at", "pending_tasks"}
        else "saved_observations")
    if field == "current_pose" and snapshot.get("current_pose_source") == "projected_state":
        source = "projected_state"
    if session.get("public_safety_context_version") == 1 and field not in {"static_capabilities", "state_variables", "pending_tasks"}:
        sources = snapshot.get("field_sources", {})
        if field == "snapshot":
            source = "projected_state" if "projected_state" in sources.values() else "saved_observations"
        else:
            source = sources.get(field, "saved_observations")
        if field == "current_pose" and snapshot.get("current_pose_source") == "projected_state":
            source = "projected_state"
    return value, source


# Diagnostic-mode refs: (ref, what it answers, prepared keys that record its data).
_DIAGNOSTIC_REFS = (
    ("/safety_rules/<id>", "full requirement and source of a listed safety rule (or /safety_rules for all)",
     ("prompt_safety_rules",)),
    ("/regions/<name>", "region bounds in the world frame (or /regions for all)", ("diagnostic_regions",)),
    ("/geometry/<key>", "station poses and sizes, robot and mobile bases, floor extent and obstacles (or /geometry for all)",
     ("diagnostic_scene_geometry",)),
    ("/resources/<jid>/gripper_reach", "declared pick envelope for a part located only by its observed_pose",
     ("diagnostic_reach",)),
    ("/resources/<jid>/primitives", "callable primitive signatures", ("diagnostic_primitives",)),
    ("/parts/<name>/pose_relations",
     "regions containing its observed_pose, and resources declaring that location or a pose envelope around it",
     ("diagnostic_regions", "diagnostic_reach")),
    ("/resources/<jid>/pose_relations", "the same relations for its current_pose", ("diagnostic_regions", "diagnostic_reach")),
)
_DIAGNOSTIC_FIELDS = {("resources", "gripper_reach"), ("resources", "primitives"),
                      ("resources", "pose_relations"), ("parts", "pose_relations")}
_REGION_FRAME = "Bounds as [[x_min, x_max], [y_min, y_max], [z_min, z_max]] in the world frame, also used by current_pose and observed_pose."
_GEOMETRY_FRAME = (
    "world_pose as [x, y, z, roll, pitch, yaw] in the world frame; size, length, width and enclosure "
    "dimensions along each station's own axes; base_pose_xy_yaw and floor extent and obstacles, as "
    "[x_min, x_max, y_min, y_max], in the world frame; arm_mount_xyz in the mobile base frame."
)
_REACH_RULE = (
    "A pose is within gripper_reach when its xy distance from origin_pose is at most max_xy_radius_m + "
    "tolerance_m and its z lies within [z_min_m - tolerance_m, z_max_m + tolerance_m]. It bounds picking "
    "a part located only by its observed_pose; station access follows each robot's reachability."
)
_POSE_RELATION_BASIS = (
    "regions: recorded region bounds that contain the measured position (one point, not the full extent); "
    "locations: declared reachability, reachable_locations or "
    "resource_location domain, where a region belongs to location L when it is named L or L.<sub>; "
    "envelopes: the pose beside each declared work envelope, which contains it when xy_distance_m is at most "
    "max_xy_radius_m + tolerance_m and z_m lies within z_range_m widened by tolerance_m; not_evaluated: "
    "resources whose declarations do not bound the pose, because they declare no envelope and either no "
    "location or an event that moves them to coordinates."
)


def is_diagnostic_ref(ref: str) -> bool:
    """Return whether ``ref`` has the shape of a diagnostic-mode reference."""
    segments = ref.split("/")
    return (segments[1:2] in (["safety_rules"], ["regions"], ["geometry"]) and len(segments) <= 3
            or len(segments) == 4 and (segments[1], segments[3]) in _DIAGNOSTIC_FIELDS)


def diagnostic_refs(prepared: dict) -> list[tuple[str, str]]:
    """List the diagnostic-mode refs whose data the capture records, with what each answers."""
    return [(ref, answers) for ref, answers, keys in _DIAGNOSTIC_REFS if any(prepared.get(key) for key in keys)]


def _pose_relations_fact(prepared: dict, session: dict, kind: str, name: str) -> dict:
    from .multi_turn_diagnostic import pose_relations
    from .multi_turn_outline_facts import _coordinate_parameters

    field = "observed_pose" if kind == "parts" else "current_pose"
    pose = resolve_outline_context(prepared, session, f"/{kind}/{name}/{field}")["value"]
    locations = {jid: _recorded_locations(entry) or set(_resource_location_domain(entry))
                 for jid, entry in prepared.get("recovery_resources", {}).items()}
    movers = {jid for jid, model in session.get("recovery_des_models", {}).items() if _coordinate_parameters(model)}
    return {field: pose, **pose_relations(pose, prepared.get("diagnostic_regions", {}), locations,
                                          prepared.get("diagnostic_reach", {}), movers)}


def _diagnostic_fact(prepared: dict, session: dict, ref: str) -> dict:
    """Serve a diagnostic-mode declaration, or a pose relation derived from declarations."""
    if not is_diagnostic_ref(ref):
        raise ValueError(f"NEEDS_CONTEXT: {ref} is not recorded")
    segments = ref.split("/")
    source, interpretation = "saved_context", None
    if len(segments) == 4 and segments[3] == "pose_relations":
        value, source = _pose_relations_fact(prepared, session, segments[1], segments[2]), "model_calculation"
    elif len(segments) == 4 and segments[3] == "gripper_reach":
        # Diagnostic mode states these declarations; validated runs keep them private.
        value, interpretation = deepcopy(prepared.get("diagnostic_reach", {}).get(segments[2])), _REACH_RULE
    elif len(segments) == 4:
        from .multi_turn_prompts import _primitive_signature_card

        catalog = prepared.get("diagnostic_primitives", {}).get(segments[2])
        value = (_primitive_signature_card({segments[2]: catalog}, include_optional=True).removeprefix(f"- {segments[2]}: ")
                 if catalog else None)
        interpretation = "name(parameters; ? marks optional ones)"
    else:
        value = {"safety_rules": prepared.get("prompt_safety_rules", []), "regions": prepared.get("diagnostic_regions", {}),
                 "geometry": prepared.get("diagnostic_scene_geometry", {})}[segments[1]]
        if len(segments) == 3:
            value = (next((row for row in value if row.get("id") == segments[2]), None) if isinstance(value, list)
                     else value.get(segments[2]))
        value = None if value is None else _public_fact_value(value)
        interpretation = {"regions": _REGION_FRAME, "geometry": _GEOMETRY_FRAME}.get(segments[1])
    if value is None:
        raise ValueError(f"NEEDS_CONTEXT: {ref} is not recorded")
    fact = {"value": value, "source": source, "source_run_id": prepared.get("source_observation_run_id"),
            "captured_at": prepared.get("source_observation_captured_at")}
    if interpretation:
        fact["interpretation"] = interpretation
    if source == "model_calculation":
        fact["basis"] = _POSE_RELATION_BASIS
    return fact


def resolve_outline_context(prepared: dict, session: dict, ref: str) -> dict:  # noqa: C901, PLR0912 - explicit public reference allowlist
    """Resolve an allowlisted fact reference with recorded provenance."""
    if session.get("safety_rules_to_llm") == "hidden" and ref.split("/")[1:2] == ["safety_rules"]:
        raise ValueError("Unsupported context reference")
    if session.get("outline_diagnostic_bypass") and (
            ref.split("/")[1:2] in (["safety_rules"], ["regions"], ["geometry"]) or is_diagnostic_ref(ref)):
        return _diagnostic_fact(prepared, session, ref)
    if ref.split("/")[1:2] in (["safety_rules"], ["regions"]):
        raise ValueError("Unsupported context reference")
    resources, parts = _views(prepared, session)
    segments = ref.split("/")
    if not ref.startswith("/") or any(not segment for segment in segments[1:]):
        raise ValueError("Unsupported context reference")
    if segments[1] in {"task_contracts", "primitive_catalog", "primitive_contracts"} or (
        len(segments) == 4 and segments[1] == "resources" and segments[3] in {
            "task_contracts", "primitive_catalog", "primitive_contracts", "execution_primitive_catalog",
        }
    ):
        raise ValueError("Unsupported outline context reference: use public resource or part facts")
    entry, source = {}, "saved_context"
    if ref == "/resources":
        value = resource_index(prepared)
    elif ref == "/parts":
        value = [{"part_name": name} for name in sorted(parts)]
    elif len(segments) == 4 and segments[1] == "resources":
        value, source = _resource_fact(prepared, session, segments[2], segments[3], ref)
    elif len(segments) == 4 and segments[1] == "parts":
        name, field = segments[2:]
        if name not in parts:
            raise ValueError(f"NEEDS_CONTEXT: unknown part {name}")
        entry = prepared.get("grounding_context", {}).get("parts", {}).get(name, {})
        snapshot = (public_part_snapshot(prepared, session, name) if session.get("validation_contract_version") == 2
                    else _fields({**prepared.get("part_tracker", {}).get(name, {}), **entry, **parts[name]}, _PART_FIELDS))
        # An empty prepared field does not erase a measured pose unless a
        # projected part transition made it stale.
        if (session.get("validation_contract_version") != 2 and not snapshot.get("observed_pose")
                and snapshot.get("observed_pose_current") is not False):
            snapshot["observed_pose"] = (entry.get("observed_pose")
                                          or prepared.get("part_tracker", {}).get(name, {}).get("observed_pose"))
        if field == "target":
            value = entry.get("target")
            source = "configuration"
        elif field == "state_variables":
            value = _public_part_state_variables(prepared, snapshot)
            source = "declaration"
        elif field == "snapshot":
            value = snapshot
            source = "projected_state" if session.get("accepted_outline_prefix") else "saved_observations"
        elif field in snapshot:
            value = snapshot[field]
            source = "saved_observations" if field in {"observed_pose", "last_observed_pose"} else "projected_state" if session.get("accepted_outline_prefix") else "saved_observations"
        else:
            raise ValueError(f"NEEDS_CONTEXT: {ref} is not available")
        if session.get("public_safety_context_version") == 1 and field not in {"target", "state_variables"}:
            sources = snapshot.get("field_sources", {})
            if field == "snapshot":
                source = "projected_state" if "projected_state" in sources.values() else "saved_observations"
            else:
                source = sources.get(field, "saved_observations")
    else:
        raise ValueError("Unsupported context reference")
    if segments[-1] == "observed_pose" and len(segments) == 4 and parts.get(segments[2], {}).get("observed_pose_current") is False:
        raise ValueError(f"NEEDS_CONTEXT: {ref} requires a new observation after the projected part transition")
    if value is None and segments[-1] in {"target", "observed_pose", "current_pose"}:
        raise ValueError(f"NEEDS_CONTEXT: {ref} is not recorded")
    observation = session.get("outline_observed_context", {}).get(ref, {})
    captured_at = prepared.get("source_observation_captured_at", prepared.get("prepared_at_utc"))
    if (observation and observation.get("value") == value and not (
            session.get("public_safety_context_version") == 1 and source == "projected_state")):
        source, captured_at = "resource_observation", observation["captured_at"]
    fact = {"value": deepcopy(value), "source": source,
            "captured_at": captured_at,
            "source_run_id": prepared.get("source_observation_run_id"),
            "revision": recovery_validation_fingerprint({"resources": resources, "parts": parts}),
            "accepted_trace_length": len(session.get("accepted_outline_prefix", []))}
    if session.get("public_safety_context_version") == 1 and source == "projected_state":
        fact.update(captured_at=None, source_observation_captured_at=captured_at)
    if len(segments) == 4 and segments[3] == "state_variables" and session.get("outline_diagnostic_bypass"):
        fact["values_rule"] = _CONDITION_VALUES_RULE
    if len(segments) == 4 and segments[1] == "resources" and segments[3] == "state_variables":
        interpretation = _held_part_interpretation(prepared["recovery_resources"][segments[2]], value)
        if interpretation:
            fact["interpretation"] = {
                "source": "runtime_validation", "field": "held_part",
                "rule": "held_part.domain records custody, not a closed set of grounded part_name values. Resource validation and physical feasibility remain required.",
            }
        if not value:
            fact["missing_context"] = f"NEEDS_CONTEXT: public state_variables are not recorded for {segments[2]}"
    if segments[-1] == "snapshot" and source == "projected_state":
        fact["field_sources"] = deepcopy(value.get("field_sources", {}))
        for key in ("observed_pose", "current_pose", "current_pose_captured_at"):
            if key in value:
                fact["field_sources"].setdefault(key, "saved_observations")
        if value.get("observed_pose_current") is False:
            fact["field_sources"].update(observed_pose="unavailable_after_projected_transition",
                                         last_observed_pose="saved_observations", projected_pose="projected_state")
        if value.get("current_pose_source") == "projected_state":
            fact["field_sources"]["current_pose"] = "projected_state"
    return fact


def scope_prompt_input(payload: dict, prepared: dict, session: dict) -> dict:
    """Filter a detached presentation copy; leave PA/RA validation untouched."""
    scope = select_outline_context(prepared, session)
    payload = deepcopy(payload)
    resources, parts = set(scope["resource_jids"]), set(scope["part_names"])
    llm_input, state = payload["llm_input"], payload["session_state"]
    llm_input.setdefault("observed_runtime_state", {})["resources"] = [row for row in
        llm_input.get("observed_runtime_state", {}).get("resources", []) if row.get("resource_jid") in resources]
    llm_input["part_facts"] = [row for row in llm_input.get("part_facts", []) if row.get("part_name") in parts]
    for key in ("symbolic_resources", "base_symbolic_resources", "recovery_des_models"):
        state[key] = {jid: row for jid, row in state.get(key, {}).items() if jid in resources}
    for jid, row in state.get("symbolic_resources", {}).items():
        snapshot = prepared.get("recovery_resources", {}).get(jid, {}).get("recovery_snapshot", {})
        if row.get("current_pose") and row.get("current_pose") == snapshot.get("current_pose"):
            row.setdefault("current_pose_captured_at", snapshot.get("current_pose_captured_at"))
    for key in ("symbolic_parts", "base_symbolic_parts"):
        state[key] = {name: row for name, row in state.get(key, {}).items() if name in parts}
    payload["recovery_resources"] = {jid: row for jid, row in payload["recovery_resources"].items() if jid in resources}
    payload["outline_context"] = scope
    payload["outline_resource_fields"] = {jid: resolve_outline_context(prepared, session, f"/resources/{jid}/state_variables")
                                          for jid in sorted(resources)}
    # Automatic inclusion has no effect on LLM-requested retrieval history.
    payload["outline_resource_capabilities"] = {
        jid: resolve_outline_context(prepared, session, f"/resources/{jid}/static_capabilities")
        for jid in sorted(resources)}
    for row in scope["resource_index"]:
        fact = payload["outline_resource_capabilities"].get(row["resource_jid"])
        if fact:
            details = {"named_poses", "staging_areas", "program_options", "processing_observations"}
            values = {key: deepcopy(value) for key, value in fact["value"].items() if key not in details}
            values["field_sources"] = {key: value for key, value in values.get("field_sources", {}).items() if key not in details}
            row.update(values)
            row["available_capability_fields"] = sorted(details.intersection(fact["value"]))
    if session.get("outline_diagnostic_bypass"):
        from .multi_turn_diagnostic import llm_selects_next_event

        unvalidated = llm_selects_next_event(session)
        for row in scope["resource_index"]:
            if unvalidated:
                row.pop("field_sources", None)
                row.pop("evaluation_support", None)
            # A resource without recorded locations still declares where it can be,
            # which is how a robot's reachability is derived in the first place.
            entry = prepared.get("recovery_resources", {}).get(row["resource_jid"], {})
            if not _recorded_locations(entry) and (domain := _resource_location_domain(entry)):
                row["resource_location_domain"] = domain
    payload["outline_observation_source"] = {
        "source_run_id": prepared.get("source_observation_run_id"),
        "captured_at": prepared.get("source_observation_captured_at", prepared.get("prepared_at_utc")),
        "source_context_revision": prepared.get("source_context_revision"),
    }
    payload["outline_interrupted_task"] = _fields(prepared.get("failure_anchor", {}), {
        "failed_task_id", "failed_function_name", "failed_resource_jid",
    })
    served, errors = {}, []
    for ref in session.get("outline_requested_refs", []):
        try:
            served[ref] = resolve_outline_context(prepared, session, ref)
        except ValueError as exc:
            errors.append({"ref": ref, "error": str(exc)})
    payload["outline_served_context"] = served
    for error in session.get("outline_context_errors", []):
        if error not in errors:
            errors.append(deepcopy(error))
    payload["outline_context_errors"] = errors
    return payload


def _remember_outline_context_results(prepared: dict, session: dict, entry: dict) -> bool:
    resources, parts = _views(prepared, session)
    revision = recovery_validation_fingerprint({"resources": resources, "parts": parts})
    trace_length = len(session.get("accepted_outline_prefix", []))
    seen = session.setdefault("outline_context_seen_results", {})
    results = [(ref, "fact", fact) for ref, fact in entry["served_context"].items()]
    results.extend((error["ref"], "error", error) for error in entry["context_errors"])
    new_information = False
    for ref, kind, result in results:
        state_key = recovery_validation_fingerprint({
            "revision": result.get("revision", revision),
            "accepted_trace_length": result.get("accepted_trace_length", trace_length),
        })
        fingerprints = seen.setdefault(state_key, {}).setdefault(ref, [])
        fingerprint = recovery_validation_fingerprint({"kind": kind, "result": result})
        if fingerprint not in fingerprints:
            fingerprints.append(fingerprint)
            new_information = True
    return new_information


_PROPOSAL_FIELDS = ("candidate_events", "candidate_traces", "next_transition", "transition_trace", "transition_suffix",
                    "selected_candidate_index", "events")


def _context_request_error(response: dict) -> str | None:
    """Name what makes a need_context reply unusable, or return None when it can be served."""
    refs = response.get("context_requests")
    if not isinstance(refs, list) or not refs:
        return "need_context requires context_requests with 1 to 8 references"
    if len(refs) > 8:
        return f"need_context allows at most 8 references; {len(refs)} were requested"
    malformed = [str(ref) for ref in refs if not isinstance(ref, str) or not ref.startswith("/")]
    if malformed:
        return "each context request must be a reference starting with /: " + ", ".join(malformed)
    proposals = [key for key in _PROPOSAL_FIELDS if key in response]
    if proposals:
        return "need_context cannot also carry " + ", ".join(proposals) + "; propose events in a later reply"
    if not isinstance(response.get("thought"), str):
        return "need_context requires thought as a string"
    return None


async def handle_outline_context(*, prepared: dict, session: dict, response: dict, planner: Any) -> tuple[str, dict]:
    """Serve one retrieval turn without committing or selecting candidate events."""
    check_context_revision(prepared, planner)
    refs = response.get("context_requests")
    entry: dict = {"context_requests": deepcopy(refs), "served_context": {}, "context_errors": []}
    new_information: bool | None = None
    error = _context_request_error(response)
    if error:
        entry["context_errors"] = [{"error": error}]
    else:
        for ref in dict.fromkeys(refs):
            try:
                entry["served_context"][ref] = resolve_outline_context(prepared, session, ref)
            except ValueError as exc:
                segments = ref.split("/")
                observable_pose = (len(segments) == 4
                    and ((segments[1] == "parts" and segments[3] == "observed_pose")
                         or (segments[1] == "resources" and segments[3] == "current_pose")))
                if not observable_pose:
                    entry["context_errors"].append({"ref": ref, "error": str(exc)})
                    continue
                try:
                    observation = await _observe_missing(prepared, session, ref, planner)
                    value = resolve_outline_context(prepared, session, ref)
                    value.update(source="resource_observation", captured_at=observation["recorded_at"])
                    session.setdefault("outline_observed_context", {})[ref] = deepcopy(value)
                    entry["served_context"][ref] = value
                    entry.setdefault("observations", []).append(observation)
                except (ValueError, LookupError, RuntimeError, OSError) as observation_error:
                    entry["context_errors"].append({"ref": ref, "error": str(exc), "observation_error": str(observation_error)})
        requested = session.setdefault("outline_requested_refs", [])
        for ref in entry["served_context"]:
            if ref not in requested:
                requested.append(ref)
        new_information = _remember_outline_context_results(prepared, session, entry)
    signature = recovery_validation_fingerprint({"requests": refs, "served": entry["served_context"], "errors": entry["context_errors"]})
    if new_information is False or (
        new_information is None and signature == session.get("outline_context_request_fingerprint")
    ):
        # The repeat is answered, not ended on: the next prompt says it gave nothing new, and the turn
        # budget bounds further repeats.
        entry["context_errors"].append({"error": "NEEDS_CONTEXT: repeated context request produced no new information"})
    session["outline_context_request_fingerprint"] = signature
    session["outline_context_errors"] = deepcopy(entry["context_errors"])
    entry["context_selection"] = select_outline_context(prepared, session)
    return "need_context", entry


