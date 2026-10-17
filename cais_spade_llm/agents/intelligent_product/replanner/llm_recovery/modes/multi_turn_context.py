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
_PART_FIELDS = {
    "part_name", "current_state", "part_state", "current_location", "part_location",
    "current_holder_resource_jid", "part_holder_resource_jid", "observed_pose",
    "origin_location", "state", "location", "location_basis", "processCompleted",
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
    invalidated = session.get("projected_physical_evidence", {}).get("invalidated_geometry", {})
    if invalidated:
        from cais_spade_llm.recovery_framework.outline_evidence_policy import remove_invalidated_pose

        physical = session["projected_physical_evidence"]
        for rid in invalidated.get("resources", {}):
            row = resources.get(physical.get("jids", {}).get(rid))
            if row is not None:
                remove_invalidated_pose(row)
        for part in invalidated.get("parts", {}):
            if part in parts:
                remove_invalidated_pose(parts[part])
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
    """Serve public declarations and state, excluding internal task/model fields."""
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
    if session.get("validation_contract_version") == 2 and jid in session.get("derived_resource_snapshots", {}):
        if session.get("public_safety_context_version") != 1:
            return _fields(resources.get(jid, {}), fields)
        # New public context keeps projected values distinct from the unchanged
        # captured row. No state token is renamed or assigned through an alias.
        row = resources.get(jid, {})
        retained = {"resource_jid", "resource_type", "controller_ready", "named_poses", "availability",
                    "current_pose", "current_pose_source", "current_pose_captured_at",
                    "projected_pose", "last_observed_pose"}
        snapshot = _fields(row, fields & (declared | retained))
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
    physical = session.get("projected_physical_evidence") or {}
    if any(physical.get("jids", {}).get(rid) == jid
           for rid in physical.get("invalidated_geometry", {}).get("resources", {})):
        from cais_spade_llm.recovery_framework.outline_evidence_policy import remove_invalidated_pose

        remove_invalidated_pose(result)
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
    aliases = {"part_state": {"current_state", "state"},
               "part_location": {"current_location", "location"},
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


def _public_resource_state_variables(entry: dict) -> dict:
    return {name: _fields(row, {"scope", "domain", "domain_complete", "type", "description"}) for name, row in
            (entry.get("recovery_des_model") or {}).get("state_variables", {}).items()
            if not row.get("private") and not name.startswith(("task_ctx.", "_"))}


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


def _task_contract_events(entry: dict, ref: str) -> list[dict]:
    model = entry.get("recovery_des_model") or {}
    events = model.get("events")
    if not isinstance(events, list):
        raise ValueError(f"NEEDS_CONTEXT: saved task declarations are not recorded for {ref}")
    return [row for row in events if isinstance(row, dict) and not row.get("private")
            and isinstance(row.get("event_name"), str)]


def _public_task_value(value: Any, private_fields: set[str]) -> bool:
    if isinstance(value, str):
        return not (value.startswith(("task_ctx.", "_")) or value in private_fields)
    if isinstance(value, dict):
        allowed = {"$arg", "$step", "$state", "$add", "$sub", "$mul", "$div", "path"}
        return (set(value) <= allowed
                and all(_public_task_value(item, private_fields) for item in value.values()))
    if isinstance(value, list):
        return all(_public_task_value(item, private_fields) for item in value)
    return value is None or isinstance(value, (bool, int, float))


def _public_task_contract(entry: dict, event: dict) -> dict:
    declarations = (entry.get("recovery_des_model") or {}).get("state_variables", {})
    private_fields = {name for name, row in declarations.items()
                      if row.get("private") or name.startswith(("task_ctx.", "_"))}
    public_fields = set(declarations) - private_fields
    result = {"event_name": event["event_name"]}
    omitted = {section for section in ("guards", "updates", "primitive_support") if section not in event}
    for section, operators in (("guards", {"equals", "not_equals", "equals_from_param", "not_equals_from_param", "exists"}),
                               ("updates", {"set", "set_from_param"})):
        result[section] = {}
        for field, condition in event.get(section, {}).items():
            if (field in public_fields and isinstance(condition, dict) and set(condition) <= operators
                    and all(_public_task_value(item, private_fields) for item in condition.values())):
                result[section][field] = deepcopy(condition)
            else:
                omitted.add(section)
    bindings = {}
    for name, binding in event.get("parameter_bindings", {}).items():
        if (_public_task_value(name, private_fields) and isinstance(binding, dict)
                and set(binding) <= {"location_type"}
                and all(_public_task_value(item, private_fields) for item in binding.values())):
            bindings[name] = deepcopy(binding)
        else:
            omitted.add("parameter_bindings")
    if bindings:
        result["parameter_bindings"] = bindings
    result.update(_fields(event, {"requires_part_binding"}))
    catalog = {row["name"]: row for row in entry.get("execution_primitive_catalog", []) if "name" in row}
    support = []
    for step in event.get("primitive_support", []):
        if not isinstance(step, dict) or not isinstance(step.get("primitive"), str):
            omitted.add("primitive_support")
            continue
        row = {"primitive": step["primitive"]}
        params = {}
        declared_params = catalog.get(step["primitive"], {}).get("params", {})
        for name, value in step.get("params", {}).items():
            if (name in declared_params and _public_task_value(name, private_fields)
                    and not (isinstance(declared_params[name], dict) and declared_params[name].get("private"))
                    and _public_task_value(value, private_fields)):
                params[name] = deepcopy(value)
            else:
                omitted.add("primitive_support")
        if params:
            row["params"] = params
        if step.get("when"):
            omitted.add("primitive_support")
        support.append(row)
    if "primitive_support" in event:
        result["primitive_support"] = support
    result["partial"] = bool(omitted)
    if omitted:
        result["omitted_sections"] = sorted(omitted)
        result["note"] = "Private or unsupported declarations are omitted; this is not a complete task contract."
    return result


def _resource_fact(prepared: dict, session: dict, jid: str, field: str, ref: str) -> tuple[Any, str]:
    entry = prepared.get("recovery_resources", {}).get(jid, {})
    snapshot = public_resource_snapshot(prepared, session, jid)
    if field == "snapshot":
        value = snapshot
    elif field == "static_capabilities":
        value = _public_resource_capabilities(entry)
        physical = session.get("projected_physical_evidence") or {}
        if (physical.get("elapsed_time_obligations")
                or any(physical.get("jids", {}).get(rid) == jid for rid in physical.get("unresolved_processing", {}))):
            value.pop("processing_observations", None)
    elif field == "capability_evidence":
        value = _public_capability_evidence(entry)
    elif field == "pending_tasks":
        value = [{**_fields(task, {"id", "function_name", "status", "predecessors", "successors"}),
                  "params": _fields(task.get("params", {}), {"part_name", "destination_location", "origin_resource_location"})}
                 for task in entry.get("pending_tasks", [])]
    elif field == "state_variables":
        value = _public_resource_state_variables(entry)
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
    "regions: recorded region bounds; locations: declared reachability, reachable_locations or "
    "resource_location domain, where a region belongs to location L when it is named L or L.<sub>; "
    "envelopes: declared gripper_reach. not_evaluated resources declare neither."
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

    field = "observed_pose" if kind == "parts" else "current_pose"
    pose = resolve_outline_context(prepared, session, f"/{kind}/{name}/{field}")["value"]
    locations = {jid: _recorded_locations(entry) or set(_resource_location_domain(entry))
                 for jid, entry in prepared.get("recovery_resources", {}).items()}
    return {field: pose, **pose_relations(pose, prepared.get("diagnostic_regions", {}), locations,
                                          prepared.get("diagnostic_reach", {}))}


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
    if session.get("outline_diagnostic_bypass") and (
            ref.split("/")[1:2] in (["safety_rules"], ["regions"], ["geometry"]) or is_diagnostic_ref(ref)):
        return _diagnostic_fact(prepared, session, ref)
    if ref.split("/")[1:2] in (["safety_rules"], ["regions"]):
        if session.get("public_safety_context_version") != 1:
            raise ValueError("Unsupported context reference")
        return public_safety_context_fact(prepared, session, ref)
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
        elif field == "snapshot":
            value = snapshot
            source = "projected_state" if session.get("accepted_outline_prefix") else "saved_observations"
        elif field in snapshot:
            value = snapshot[field]
            source = "saved_observations" if field in {"observed_pose", "last_observed_pose"} else "projected_state" if session.get("accepted_outline_prefix") else "saved_observations"
        else:
            raise ValueError(f"NEEDS_CONTEXT: {ref} is not available")
        if session.get("public_safety_context_version") == 1 and field != "target":
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
    if session.get("public_safety_context_version") == 1 and not session.get("outline_diagnostic_bypass"):
        safety_context = public_outline_safety_context(prepared, session)
        payload["outline_safety_context"] = compact_public_safety_context(
            safety_context, resource_jids=resources, part_names=parts)
        payload["outline_safety_feedback_bindings"] = safety_feedback_bindings(
            safety_context, session.get("outline_validation_findings") or [])
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


async def handle_outline_context(*, prepared: dict, session: dict, response: dict, planner: Any) -> tuple[str, dict]:
    """Serve one retrieval turn without committing or selecting candidate events."""
    check_context_revision(prepared, planner)
    refs = response.get("context_requests")
    entry: dict = {"context_requests": deepcopy(refs), "served_context": {}, "context_errors": []}
    new_information: bool | None = None
    if (not isinstance(refs, list) or not refs or len(refs) > 8
            or any(not isinstance(ref, str) or not ref.startswith("/") for ref in refs)
            or any(key in response for key in ("candidate_events", "candidate_traces", "next_transition", "transition_trace", "transition_suffix", "selected_candidate_index"))):
        entry["context_errors"] = [{"error": "need_context requires 1 to 8 references and no candidate events"}]
    elif not isinstance(response.get("thought"), str):
        entry["context_errors"] = [{"error": "need_context requires thought as a string"}]
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
        session["status"] = "NEEDS_CONTEXT"
        entry["context_errors"].append({"error": "NEEDS_CONTEXT: repeated context request produced no new information"})
    session["outline_context_request_fingerprint"] = signature
    session["outline_context_errors"] = deepcopy(entry["context_errors"])
    entry["context_selection"] = select_outline_context(prepared, session)
    return "need_context", entry


def public_outline_safety_context(prepared: dict, session: dict) -> dict:
    """Calculate endpoint occupancy using only saved or accepted physical facts."""
    from cais_spade_llm.agents.central_controller.offline_safety_grounding import _rules
    from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _catalog, _ap_binding
    from cais_spade_llm.resources.recovery_geometry import _checkpoint

    accepted = session.get("accepted_outline_prefix") or []
    physical = session.get("projected_physical_evidence")
    if physical is None and not accepted:
        physical = prepared.get("physical_evidence")
    physical = physical if isinstance(physical, dict) else {}
    snapshot = physical.get("snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    geometry = physical.get("geometry") or {}
    models = physical.get("resource_models") or {}
    # Event contracts and earlier preview records do not define this endpoint's
    # geometry and must not recursively enter either public facts or their hash.
    configuration = {rid: model.get("configuration") for rid, model in models.items()}
    source = {
        "source": "model_calculation",
        "basis": "projected_state" if accepted or physical.get("provenance", {}).get("kind") == "projected" else "saved_observations",
        "captured_at": prepared.get("source_observation_captured_at", prepared.get("prepared_at_utc")),
        "source_run_id": prepared.get("source_observation_run_id"),
        "source_context_revision": prepared.get("source_context_revision"),
        "accepted_trace_length": len(accepted),
        "coverage": "endpoint",
        "model_fingerprint": recovery_validation_fingerprint(configuration),
        "geometry_fingerprint": recovery_validation_fingerprint(geometry),
        "snapshot_fingerprint": recovery_validation_fingerprint(snapshot),
        "resource_identity_fingerprint": recovery_validation_fingerprint(physical.get("jids")),
        "specifications_fingerprint": recovery_validation_fingerprint(prepared.get("outline_safety_specifications") or []),
        "execution_authorized": False,
    }
    source["revision"] = recovery_validation_fingerprint({key: source[key] for key in (
        "basis", "source_context_revision", "accepted_trace_length", "model_fingerprint",
        "geometry_fingerprint", "snapshot_fingerprint", "resource_identity_fingerprint", "specifications_fingerprint")})
    result = {"specifications": [], "instances": [], "regions": [], "current_occupancy": {}, "source": source,
              "missing_context": []}
    resources = snapshot.get("resources") or {}
    jids = physical.get("jids") or {}
    selected_regions, selected_resources, selected_parts = set(), set(), set()
    for record in prepared.get("outline_safety_specifications") or []:
        document = record.get("document") or {}
        catalog = document.get("catalog") or {}
        raw = catalog.get("specifications") or []
        scopes = document.get("requirement_scopes") or []
        try:
            definitions = _catalog(deepcopy(catalog), structured_groundings=True)
        except (KeyError, TypeError, ValueError, IndexError):
            result["missing_context"].append({"ref": "/safety_rules", "source_sha256": record.get("source_sha256"),
                "reason": "NEEDS_CONTEXT: selected safety declarations do not match the supported public schema"})
            continue
        for definition in raw:
            # Exact captured public declarations, not compiled monitor internals.
            result["specifications"].append({
                **{key: deepcopy(definition[key]) for key in ("id", "requirement", "formula", "aps") if key in definition},
                "requirement_scopes": [_public_safety_scope(scope, definition["aps"]) for scope in scopes
                    if scope.get("specification") == definition.get("id")],
                "source_path": record.get("source_path"), "source_sha256": record.get("source_sha256"),
            })
        try:
            state_required = any(physical_ap_kind(ap) in {None, "loading_reserved_by"}
                                 for definition in definitions.values() for ap in definition["aps"])
            recorded = physical.get("environment_models") or {}
            if state_required:
                if not isinstance(recorded, dict) or set(recorded) != set(jids):
                    raise ValueError("Complete recorded owner state declarations are unavailable")
                if any(not isinstance(model.get("state_variables"), dict) for model in recorded.values()):
                    raise ValueError("Recorded owner state declarations are unavailable")
            # Pairwise scope selection also reads custody and motion declarations.
            # They remain private inputs to the same binding code used by CCA.
            declared_models = {rid: {"state_variables": deepcopy(recorded.get(rid, {}).get("state_variables", {})),
                **({"configuration": deepcopy(models[rid]["configuration"])}
                   if models.get(rid, {}).get("configuration") else {})} for rid in jids}
            source["owner_state_models_fingerprint"] = recovery_validation_fingerprint(declared_models)
            source["revision"] = recovery_validation_fingerprint({key: value for key, value in source.items()
                                                                  if key != "revision"})
            rules, _ = _rules(definitions, scopes, declared_models, geometry)
        except (KeyError, TypeError, ValueError, IndexError):
            result["missing_context"].append({"ref": "/safety_rules", "source_sha256": record.get("source_sha256"),
                "reason": "NEEDS_CONTEXT: selected AP bindings require complete recorded resource, part, and region declarations"})
            continue
        for rule in rules:
            aps = []
            for ap in rule["aps"]:
                if physical_ap_kind(ap) is None:
                    declared = rule["ap_groundings"][ap["label"]]
                    public = {"resource": declared["resource_id"], "process": declared["process"],
                              "part": declared["product"]}
                    aps.append({"label": ap["label"], "meaning": ap["meaning"], "binding": public})
                    selected_resources.add(public["resource"])
                    continue
                binding = _ap_binding(rule, ap)
                # This is deliberately a physical binding allowlist, never an
                # arbitrary path to private evaluator/model fields.
                public = {key: deepcopy(value) for key, value in binding.items()
                          if key in {"resource", "receiving_resource", "region", "part", "process", "target", "result", "capacity"}}
                aps.append({"label": ap["label"], "meaning": ap["meaning"], "binding": public})
                if "region" in public:
                    selected_regions.add(public["region"])
                for field in ("resource", "receiving_resource"):
                    if field in public:
                        selected_resources.add(public[field])
                if "part" in public:
                    selected_parts.add(public["part"])
                if physical_ap_kind(ap) == "capacity_exceeded":
                    selected_parts.update(geometry.get("parts", {}))
            result["instances"].append({"id": rule["id"], "rule_id": rule["rule_id"], "aps": aps,
                "source_path": record.get("source_path"), "source_sha256": record.get("source_sha256")})
    result["regions"] = sorted(selected_regions)
    result["resource_bindings"] = [{"resource": rid, **({"resource_jid": jids[rid]} if rid in jids else {})}
                                    for rid in sorted(selected_resources)]
    for region in sorted(selected_regions):
        result["current_occupancy"][region] = {"resources": {rid: None for rid in sorted(selected_resources)},
                                              "parts": {name: None for name in sorted(selected_parts)}}
    if not selected_regions:
        return result
    try:
        if not snapshot or set(resources) != set(jids):
            raise ValueError("Recorded resource identities are incomplete")
        observed = _checkpoint(snapshot, geometry, models, index=len(accepted))
        for region, target in result["current_occupancy"].items():
            target["resources"] = {rid: deepcopy(observed["occupancy_possibilities"][region][rid])
                                    for rid in target["resources"]}
            target["parts"] = {name: deepcopy(observed["part_occupancy_possibilities"][region][name])
                                for name in target["parts"]}
    except (KeyError, TypeError, ValueError, IndexError):
        result["missing_context"].append({"ref": "/regions", "reason":
            "NEEDS_CONTEXT: current occupancy requires recorded configuration, robot/tool/carried-part geometry, custody, and matching coordinate frames"})
    return result


def public_safety_context_fact(prepared: dict, session: dict, ref: str) -> dict:
    """Resolve only the listed safety/region references; no live interfaces."""
    segments = ref.split("/")
    if not ref.startswith("/") or any(not segment for segment in segments[1:]):
        raise ValueError("Unsupported context reference")
    supported = (ref in {"/safety_rules", "/regions"} or
        len(segments) == 3 and segments[1] == "safety_rules" and bool(segments[2]) or
        len(segments) == 4 and segments[1] == "safety_rules" and bool(segments[2]) and segments[3] == "instances" or
        len(segments) == 4 and segments[1] == "regions" and bool(segments[2]) and segments[3] == "occupancy")
    if not supported:
        raise ValueError("Unsupported context reference")
    context = public_outline_safety_context(prepared, session)
    metadata = deepcopy(context["source"])
    if ref == "/safety_rules":
        value = context["specifications"]
        metadata.update(source="declaration", coverage="selected specifications")
    elif segments[1] == "safety_rules":
        name = segments[2]
        selected = [row for row in context["specifications"] if row["id"] == name]
        if not selected:
            raise ValueError("NEEDS_CONTEXT: unknown selected safety specification")
        value = selected if len(segments) == 3 else [row for row in context["instances"] if row["id"] == name]
        if not value:
            raise ValueError("NEEDS_CONTEXT: selected safety AP bindings are unavailable")
        metadata.update(source="declaration", coverage="selected specifications")
    elif ref == "/regions":
        value = context["regions"]
    else:
        region = segments[2]
        if region not in context["current_occupancy"]:
            raise ValueError("NEEDS_CONTEXT: unknown selected safety region")
        value = context["current_occupancy"][region]
    return {"value": deepcopy(value), **metadata,
            **({"missing_context": deepcopy(context["missing_context"])} if context["missing_context"] else {})}


def compact_public_safety_context(context: dict, *, resource_jids: set[str] | None = None,
                                  part_names: set[str] | None = None) -> dict:
    """Keep declarations and endpoint facts once; full instances stay retrievable."""
    result = {key: deepcopy(context[key]) for key in (
        "resource_bindings", "regions", "current_occupancy", "source", "missing_context") if key in context}
    if resource_jids is not None:
        selected = {row["resource"] for row in context.get("resource_bindings", [])
                    if row.get("resource_jid") in resource_jids}
        for region in result.get("current_occupancy", {}).values():
            region["resources"] = {rid: value for rid, value in region.get("resources", {}).items()
                                   if rid in selected}
    if part_names is not None:
        for region in result.get("current_occupancy", {}).values():
            region["parts"] = {part: value for part, value in region.get("parts", {}).items()
                               if part in part_names}
    result["specifications"] = [{**{key: deepcopy(value) for key, value in row.items() if key != "aps"},
        "aps": [{"label": ap["label"], "meaning": ap["meaning"]} for ap in row["aps"]]}
        for row in context["specifications"]]
    return result


def safety_feedback_bindings(context: dict, findings: list) -> list:
    """Expose exact rule instances named in the recorded public feedback."""
    result = []
    for row in findings:
        for check in (row.get("evidence") or {}).get("safety_checks", []):
            rule_id = check.get("counterexample", {}).get("rule_id")
            matching = [instance for instance in context["instances"] if instance["rule_id"] == rule_id
                        and ("source_path" not in check or instance["source_path"] == check["source_path"])]
            if len(matching) == 1 and matching[0] not in result:
                result.append(deepcopy(matching[0]))
    return result


def _public_safety_scope(scope: dict, aps: list) -> dict:
    """Keep declared physical bindings without private structured evaluators."""
    fields = {"specification", "resource", "receiving_resource", "region", "part", "process", "target", "result"}
    public = {key: value for key, value in scope.items() if key in fields and isinstance(value, str)}
    if type(scope.get("capacity")) is int:
        public["capacity"] = scope["capacity"]
    bindings = scope.get("physical_ap_bindings")
    if isinstance(bindings, dict):
        labels = {ap["label"] for ap in aps}
        public["physical_ap_bindings"] = {
            label: {key: value for key, value in binding.items() if key in fields and isinstance(value, str)}
            for label, binding in bindings.items() if label in labels and isinstance(binding, dict)}
        for label, binding in bindings.items():
            if label in public["physical_ap_bindings"] and type(binding.get("capacity")) is int:
                public["physical_ap_bindings"][label]["capacity"] = binding["capacity"]
    return public
