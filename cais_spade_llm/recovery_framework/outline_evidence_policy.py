from __future__ import annotations

"""Predicate-dependent evidence for marked recovery and nominal candidates."""

from copy import deepcopy
from fractions import Fraction

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint as fingerprint

OUTLINE_EVIDENCE_POLICY_VERSION = 2
PROCESS_KINDS = {"process_target_completed", "process_result_completed"}
SPATIAL_KINDS = {"resource_region", "part_region_entry"}
POSE_FIELDS = {
    "current_pose", "observed_pose", "projected_pose", "current_pose_source", "current_pose_captured_at",
    "base_pose", "joint_positions", "joint_state", "configuration", "continuous_motion_configuration",
    "grasp_transform", "attachment", "tcp_observation", "component_bounds", "world_bounds",
    "controller_hold", "base_hold_evidence", "hold_evidence", "controller_observations",
}


def policy_enabled(payload: dict) -> bool:
    """Recognize an explicit policy marker without upgrading saved sessions."""
    return payload.get("outline_evidence_policy_version") in (1, OUTLINE_EVIDENCE_POLICY_VERSION)


def evidence_binding(*, task: dict, snapshot: dict, model: dict, physical_input: dict,
                     specifications: list | None) -> dict:
    """Bind requirements to exact source facts, including specification contents."""
    return {"proposal": fingerprint(task), "starting_snapshot": fingerprint(snapshot),
            "owner_model": fingerprint(model), "physical_evidence": fingerprint(physical_input.get("physical_evidence")),
            "owner_evidence": fingerprint(physical_input.get("resource_evidence")),
            "specifications": fingerprint(specifications),
            "outline_evidence_policy_version": physical_input.get("outline_evidence_policy_version", 1),
            **({"endpoint_evidence_policy_version": physical_input["physical_evidence"]["endpoint_evidence_policy_version"]}
               if "endpoint_evidence_policy_version" in (physical_input.get("physical_evidence") or {}) else {})}


def event_evidence_requirements(*, task: dict, snapshot: dict, model: dict, physical_input: dict,
                                specifications: list | None) -> dict:
    """Resolve selected AP dependencies using the existing exact grounding rules."""
    from cais_spade_llm.agents.central_controller.offline_safety_grounding import _rules
    from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _ap_binding, _catalog

    physical = physical_input.get("physical_evidence") or {}
    owner = physical_input.get("resource_evidence") or {}
    population = physical.get("jids", {})
    models = physical.get("environment_models") or owner.get("environment_models") or {rid: {} for rid in population}
    rows, unresolved = [], []
    if not isinstance(specifications, list):
        unresolved.append("Selected safety specifications were not captured")
    for record in specifications or []:
        try:
            document = record["document"]
            rules, _ = _rules(_catalog(document["catalog"], structured_groundings=True),
                              document.get("requirement_scopes", []), models, physical.get("geometry") or {})
            for rule in rules:
                for ap in rule["aps"]:
                    kind = physical_ap_kind(ap)
                    binding = _ap_binding(rule, ap)
                    rows.append({"source_sha256": record["source_sha256"], "rule_id": rule["rule_id"],
                        "label": ap["label"], "kind": kind, "binding": deepcopy(binding),
                        "coverage": "boundaries_and_interval",
                        "dependencies": (["exact_completion_ledger", "relevant_job_progression"] if kind in PROCESS_KINDS
                            else ["bound_region", "complete_resource_population", "robot_and_tool_geometry", "carried_parts"]
                            if kind in SPATIAL_KINDS else ["legacy_owner_and_physical_evidence"])})
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            unresolved.append(str(exc))
    if any(row["kind"] in SPATIAL_KINDS for row in rows):
        if (set(models) != set(population)
                or set(physical.get("snapshot", {}).get("resources", {})) != set(population)
                or set(physical.get("geometry", {}).get("resources", {})) != set(population)):
            unresolved.append("Spatial evidence requires the complete declared resource population")
    owner_sources = []
    if model.get("events") and model.get("state_variables"):
        from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union

        support = derive_primitive_union(task=task, recovery_des_model=model, recovery_snapshot=snapshot)
        owner_sources = [{"declaration_fingerprint": fingerprint(row),
                          "parameter_bindings": deepcopy(row.get("parameter_bindings", {}))}
                         for row in support.get("matching_transitions", [])]
    selected_parts = {row["binding"]["part"] for row in rows if row["kind"] in PROCESS_KINDS}
    jobs = []
    for rid, record in physical.get("processing_observations", {}).items():
        for identity, job in record.get("jobs", {}).items():
            if job.get("job_state") == "running":
                jobs.append({"resource": rid, "job_id": identity, "part": job.get("part_name"),
                    "relevant": job.get("part_name") in selected_parts,
                    "source_fingerprint": fingerprint(job)})
    conservative = bool(unresolved or any(row["kind"] not in PROCESS_KINDS | SPATIAL_KINDS for row in rows))
    reasons = []
    if conservative:
        reasons.append("Unsupported predicate dependencies require owner facts; trajectory fallback is disabled"
            if physical.get("endpoint_evidence_policy_version") in (1, 2)
            else "Unknown dependencies retain the conservative owner projection path")
    if any(row["kind"] in SPATIAL_KINDS for row in rows):
        reasons.append("Selected spatial APs require supported occupancy throughout the event")
    if any(row["kind"] in PROCESS_KINDS for row in rows):
        reasons.append("Selected processing APs require exact completion ledgers and relevant progression")
    if task.get("required_task_contract"):
        reasons.append("Exact nominal task, parameters, steps, and effect evidence remain independently required")
    result = {"binding": evidence_binding(task=task, snapshot=snapshot, model=model,
                physical_input=physical_input, specifications=specifications), "aps": rows,
        "resource_population": sorted(population), "interacting_jobs": jobs,
        "owner_declarations": owner_sources,
        "conservative": conservative, "unresolved": unresolved, "reasons": reasons,
        "independent_checks": ["owner_successor_effects", "resource_capability"] +
            (["exact_nominal_task_contract"] if task.get("required_task_contract") else []),
        "execution_authorized": False}
    result["fingerprint"] = fingerprint(result)
    return result


def remove_invalidated_pose(row: dict) -> None:
    """Remove active geometry aliases while preserving a clearly historical pose."""
    if row.get("current_pose") is not None:
        row.setdefault("last_observed_pose", deepcopy(row["current_pose"]))
    elif row.get("observed_pose") is not None:
        row.setdefault("last_observed_pose", deepcopy(row["observed_pose"]))
    for key in POSE_FIELDS:
        row.pop(key, None)
    for key in ("resource_core", "snapshot", "state", "recorded_snapshot", "observation_state"):
        if isinstance(row.get(key), dict):
            remove_invalidated_pose(row[key])
    row["observed_pose_current"] = False


def invalidate_geometry(physical: dict, resources: set[str], *, binding: dict) -> None:
    """Retire affected observations and caches rather than recapturing their aliases."""
    if not resources:
        return
    from cais_spade_llm.resources.recovery_geometry import retain_part_support_geometry

    retain_part_support_geometry(physical)
    parts = {row.get("held_part") for rid, row in physical.get("snapshot", {}).get("resources", {}).items()
             if rid in resources and row.get("held_part") is not None}
    for name, row in physical.get("snapshot", {}).get("parts", {}).items():
        support = physical.get("stationary_part_evidence", {}).get(name, {})
        if (support.get("status") != "supported" or support.get("current_pose") != row.get("current_pose")
                or not support.get("source")):
            parts.add(name)
    history = {"binding": deepcopy(binding), "resources": {}, "parts": {}}
    invalidated = physical.setdefault("invalidated_geometry", {"resources": {}, "parts": {}})
    for rid in sorted(resources):
        history["resources"][rid] = {}
        for collection in ("snapshot", "geometry"):
            row = physical.get(collection, {}).get("resources", {}).get(rid)
            if isinstance(row, dict):
                history["resources"][rid][collection] = deepcopy(row)
                remove_invalidated_pose(row)
        for collection in ("resources", "resource_models"):
            row = physical.get(collection, {}).get(rid)
            if isinstance(row, dict):
                history["resources"][rid][collection] = deepcopy(row)
                remove_invalidated_pose(row)
                if collection == "resources":
                    for key in ("idle", "model_static", "controller_hold_model", "stationary_contract"):
                        row.pop(key, None)
        invalidated["resources"][rid] = deepcopy(binding)
    for part in sorted(parts):
        row = physical.get("snapshot", {}).get("parts", {}).get(part)
        if isinstance(row, dict):
            history["parts"][part] = deepcopy(row)
            remove_invalidated_pose(row)
        physical.get("stationary_part_evidence", {}).pop(part, None)
        invalidated["parts"][part] = deepcopy(binding)
    # These are observations of the old scene, never supplementary observations
    # of a predicted successor. Keeping them active permits stale fallback.
    for key in ("physics_evidence", "planning_scene", "recorded_geometry_supplements", "machine_access_observations",
                "event_evidence", "step_evidence", "rejected_event_evidence", "incomplete_event_evidence",
                "rejected_capability_evidence"):
        if key in physical:
            history[key] = physical.pop(key)
    physical.setdefault("invalidated_evidence_history", []).append(history)


def _processing_invariant(physical: dict, required: dict) -> None:
    selected = {row["binding"]["part"] for row in required["aps"] if row["kind"] in PROCESS_KINDS}
    for job in required["interacting_jobs"]:
        if job["relevant"]:
            raise ValueError("Required processing progression is unresolved: " + job["resource"] + "/jobs/" + job["job_id"])
    for rid, values in physical.get("environment_valuation", {}).items():
        # A running/unknown owner cannot be replaced by the absence of a job record.
        if values.get("part_name") in selected and (
            values.get("resource_state") == "processing" or values.get("active_job") is not None
            or values.get("job_state") == "running"
        ):
            raise ValueError("Required processing progression is unresolved: " + rid)


def sanitize_validation_input(physical_input: dict, jid: str) -> None:
    """Prevent captured owner aliases from filling facts invalidated by projection."""
    physical = physical_input.get("physical_evidence") or {}
    invalidated = physical.get("invalidated_geometry", {})
    ids = [rid for rid, value in physical.get("jids", {}).items() if value == jid]
    evidence = physical_input.get("resource_evidence") or {}
    if any(rid in invalidated.get("resources", {}) for rid in ids):
        remove_invalidated_pose(physical_input["projected_snapshot"])
        remove_invalidated_pose(evidence)
        evidence.get("state", {}).pop("_task_ctx", None)
        for key in ("pick_geometry", "destination_geometry", "step_outputs"):
            evidence.pop(key, None)
        physical_input.pop("parameter_witnesses", None)
        physical_input.pop("outline_event_evidence", None)
    for part in invalidated.get("parts", {}):
        for collection in ("observed_parts", "parts"):
            if isinstance(evidence.get(collection, {}).get(part), dict):
                remove_invalidated_pose(evidence[collection][part])
    if invalidated:
        evidence["invalidated_geometry"] = deepcopy(invalidated)
    if any(rid in physical.get("unresolved_processing", {}) for rid in ids):
        for row in (physical_input["projected_snapshot"], evidence.get("snapshot", {}), evidence.get("state", {})):
            for field in ("active_job", "job_state", "resource_state", "current_state"):
                row.pop(field, None)
        evidence.pop("processing_observations", None)
        if "environment_valuation" in physical:
            evidence["environment_valuation"] = deepcopy(physical["environment_valuation"])
        physical_input.pop("parameter_witnesses", None)


def invalidate_session_facts(session: dict, physical: dict) -> None:
    """Apply tombstones to symbolic, derived, and retrieval views of accepted facts."""
    invalidated = physical.get("invalidated_geometry", {})
    for rid in invalidated.get("resources", {}):
        jid = physical.get("jids", {}).get(rid)
        for collection in ("symbolic_resources", "derived_resource_snapshots"):
            rows = session.setdefault(collection, {})
            if jid in rows:
                remove_invalidated_pose(rows[jid])
    for part in invalidated.get("parts", {}):
        row = session.get("symbolic_parts", {}).get(part)
        if isinstance(row, dict):
            remove_invalidated_pose(row)
    for rid, job in physical.get("unresolved_processing", {}).items():
        jid = physical.get("jids", {}).get(rid)
        for collection in ("symbolic_resources", "derived_resource_snapshots"):
            row = session.setdefault(collection, {}).setdefault(jid,
                deepcopy(physical.get("snapshot", {}).get("resources", {}).get(rid, {})))
            for field in ("active_job", "job_state", "resource_state", "current_state"):
                row.pop(field, None)
        part = session.get("symbolic_parts", {}).get(job.get("part"), {})
        for field in ("processCompleted", "processCompleted_evidence"):
            part.pop(field, None)
        part["processCompleted_complete"] = False


def project_selected_successor(*, task: dict, snapshot: dict, model: dict, physical_input: dict,
                               requirements: dict) -> dict | None:
    """Project supported effects with an invariant ledger, or defer to the owner.

    None retains the full owner path for unknown effects or independent physical
    checks. No event name, destination, proposed end-state, or primitive union
    is itself accepted as a physical observation.
    """
    from cais_spade_llm.resources.environment_recovery import declared_environment_successors

    if physical_input.get("physical_evidence", {}).get("endpoint_evidence_policy_version") in (1, 2):
        from .outline_endpoint_evidence import project_bounded_successor

        return project_bounded_successor(task=task, snapshot=snapshot, model=model,
            physical_input=physical_input, requirements=requirements)

    if requirements["conservative"] or any(row["kind"] not in PROCESS_KINDS for row in requirements["aps"]):
        return None
    evidence = physical_input.get("resource_evidence") or {}
    if not evidence.get("environment_models") or task.get("required_task_contract"):
        return None
    candidates, _ = declared_environment_successors(task=task, recovery_snapshot=snapshot,
        recovery_des_model=model, physical_input=physical_input)
    unique = {fingerprint(row): row for row in candidates}
    if len(unique) != 1:
        return None
    declared = deepcopy(next(iter(unique.values())))
    rid = evidence["resource_id"]
    event = next(row for row in evidence["environment_models"][rid]["events"]
                 if row["event_id"] == declared["task"]["event_id"])
    # These primitive implementations change configuration, not completion
    # ledgers or custody. Any other operation retains its existing effect checks.
    motion_ops = {"move_base", "move_joints", "move_cartesian", "move_relative", "move_to_named_pose", "rotate_joint"}
    steps = event.get("program", {}).get("steps", [])
    if not steps or any(row["op"] not in motion_ops for row in steps) or event.get("product_effects") != {"processCompleted": []}:
        return None
    return _movement_effects(task=task, snapshot=snapshot, model=model, physical_input=physical_input,
        requirements=requirements, declared=declared, event=event)


def _movement_effects(*, task: dict, snapshot: dict, model: dict, physical_input: dict,
                     requirements: dict, declared: dict, event: dict,
                     calculated_endpoint: dict | None = None) -> dict | None:
    """Keep owner effects and elapsed-time obligations separate from geometry."""
    from cais_spade_llm.resources.recovery_geometry import outline_event_binding

    evidence = physical_input["resource_evidence"]
    rid = evidence["resource_id"]
    physical = physical_input.get("physical_evidence") or {}
    before = physical.get("environment_valuation", evidence["environment_valuation"])
    after, products = declared["environment_valuation"], declared["environment_products"]
    if any(field in snapshot and (field not in before[rid] or fingerprint(snapshot[field]) != fingerprint(before[rid][field]))
           for field, declaration in model.get("state_variables", {}).items()
           if declaration.get("scope", "resource") == "resource"):
        return None
    for owner in before:
        if before[owner].get("held_part") != after[owner].get("held_part"):
            if (calculated_endpoint is None or owner != rid
                    or calculated_endpoint["snapshot"]["resources"][owner].get("held_part") != after[owner].get("held_part")):
                return None
    for name, value in products.items():
        if value.get("processCompleted") != physical.get("products", evidence["environment_products"])[name].get("processCompleted"):
            return None
    part = task.get("part_name")
    part_updates = {field: deepcopy(products[part][key]) for field, key in
        (("part_state", "state"), ("part_location", "location"), ("processCompleted", "processCompleted"))
        if part in products and key in products[part]}
    part_updates.update(deepcopy(declared.get("public_part_updates", {})))
    for field, value in task["expected_end_state"].items():
        scope = model.get("state_variables", {}).get(field, {}).get("scope")
        supported = part_updates if scope == "part" else after[rid]
        if field not in supported or fingerprint(supported[field]) != fingerprint(value):
            return None
    binding = outline_event_binding(task, snapshot, model, physical)
    successor = deepcopy(physical)
    successor["environment_valuation"] = deepcopy(after)
    successor["products"] = deepcopy(products)
    projected = successor.setdefault("snapshot", {})
    for owner, values in after.items():
        projected.setdefault("resources", {}).setdefault(owner, {}).update(deepcopy(values))
    for name, values in products.items():
        projected.setdefault("parts", {}).setdefault(name, {}).update(deepcopy(values))
    moving = set(event["participants"])
    for peer in event['participants']:
        if peer == rid or physical.get('resources', {}).get(peer, {}).get('model_static') is not True:
            continue
        declarations = [row for row in physical['environment_models'][peer]['events']
                        if row['event_id'] == event['event_id']]
        if (len(declarations) == 1 and not declarations[0].get('collection_effects')
                and declarations[0].get('parameter_bindings', {}).get('resource_id') == {'equals': rid}
                and all(field.startswith('inventory.') or field in {'resource_state', 'part_name'}
                        for field in declarations[0].get('updates', {}))):
            # Passive custody participation does not move the recorded static
            # body. Keep its geometry while updating part observations separately.
            moving.discard(peer)
    # Unknown elapsed time can invalidate other moving observations too.
    for owner, record in physical.get("resources", {}).items():
        if record.get("model_static") is not True:
            moving.add(owner)
    invalidate_geometry(successor, moving, binding=binding)
    for job in requirements["interacting_jobs"]:
        successor.get("processing_observations", {}).pop(job["resource"], None)
        successor.setdefault("unresolved_processing", {})[job["resource"]] = deepcopy(job)
        for collection in (successor.get("environment_valuation", {}), projected.get("resources", {})):
            for field in ("active_job", "job_state", "resource_state"):
                collection.get(job["resource"], {}).pop(field, None)
        row = projected.get("parts", {}).get(job["part"], {})
        row["processCompleted_complete"] = False
        row.pop("processCompleted_evidence", None)
        successor.get("products", {}).get(job["part"], {}).pop("processCompleted", None)
    retired_clocks = {}
    for owner, record in successor.get("processing_observations", {}).items():
        if "simulation_clock_sec" in record:
            retired_clocks[owner] = record.pop("simulation_clock_sec")
    if retired_clocks:
        successor.setdefault("elapsed_time_obligations", []).append({
            "binding": deepcopy(binding), "last_observed_clocks": retired_clocks,
            "reason": "Unknown movement duration; retained inactive job facts do not establish a new clock"})
    unresolved = []
    try:
        _processing_invariant(physical, requirements)
        if physical.get("unresolved_processing"):
            selected = {row["binding"]["part"] for row in requirements["aps"] if row["kind"] in PROCESS_KINDS}
            if any(job.get("part") in selected for job in physical["unresolved_processing"].values()):
                raise ValueError("Required processing progression was invalidated by an earlier event")
    except ValueError as exc:
        unresolved.append(str(exc))
    updates = {owner: {key: deepcopy(value) for key, value in values.items()
                       if fingerprint(value) != fingerprint(before[owner].get(key))}
               for owner, values in after.items()}
    projected_owner = {**deepcopy(snapshot), **deepcopy(after[rid])}
    remove_invalidated_pose(projected_owner)
    successor["provenance"] = {"kind": "projected", "source": "owner_declared_successor",
        "binding": binding, "contract_fingerprint": fingerprint(event)}
    # Event coordinates are not elapsed seconds. The invariant covers any
    # nonnegative duration; unrelated unknown job progress is explicitly retired.
    observations = []
    for index, source in ((0, physical.get("snapshot", {})), (0.5, physical.get("snapshot", {})), (1, projected)):
        observations.append({**deepcopy(source), "phase": "between" if index == 0.5 else "at",
            "time": index, "time_exact": str(index), "time_basis": "event_order_not_elapsed_seconds",
            "owner_completion_applied": index == 1})
    return {"status": "needs_context" if unresolved else "projected", "interval_complete": False,
        "projection_kind": "owner_effects_with_predicate_invariants",
        "elapsed_time": {"status": "unknown", "seconds": None},
        "owner_successor": declared, "owner_program_fingerprint": fingerprint(event),
        "projected_resource_snapshot": projected_owner, "resource_valuation_updates": updates,
        "part_updates": part_updates, "projected_snapshot": deepcopy(projected),
        "successor_evidence": successor, "geometry": deepcopy(successor.get("geometry", {})),
        "resource_models": deepcopy(successor.get("resource_models", {})), "observations": observations,
        "unresolved": unresolved, "requires_projection": False, "requires_processing_progression": bool(unresolved),
        "context_requests": ["/physical_evidence/processing_observations"] if unresolved else [],
        "projection_reasons": [("Owner effects preserve completion ledgers; relevant peer progression remains unresolved"
                               if unresolved else "Owner effects preserve selected completion predicates for any elapsed duration"),
                               "Affected geometry is invalidated; no motion interval is claimed"],
        "command_sent": False, "execution_authorized": False}


def _continuous_coverage(observations: list[dict]) -> bool:
    intervals = []
    for row in observations:
        if row.get("phase") != "between":
            continue
        bounds, cells = row.get("continuous_interval"), row.get("continuous_cells")
        if not bounds or not cells:
            return False
        start, end = map(Fraction, bounds)
        position = start
        for cell in cells:
            a, b = map(Fraction, cell["interval"])
            if a != position or not a < b <= end:
                return False
            position = b
        if position != end:
            return False
        intervals.append((start, end))
    start = Fraction(str(observations[0]["time"]))
    end = Fraction(str(observations[-1]["time"]))
    if start == end:
        return False
    position = start
    for a, b in sorted(intervals):
        if a != position:
            return False
        position = b
    return position == end


def evidence_coverage(*, requirements: dict, projection: dict) -> dict:
    """Recalculate coverage from supplied evidence rather than claimed flags."""
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _complete_target_ledger

    observations = projection.get("observations") or []
    invariant = projection.get("projection_kind") == "owner_effects_with_predicate_invariants"
    full = projection.get("interval_complete") is True
    interval = bool(observations) and _continuous_coverage(observations) if not invariant else False
    instantaneous = (full and bool(observations) and observations[0]["time"] == observations[-1]["time"]
                     and projection.get("owner_interval_contract", {}).get("kind") == "resource_owned_simulation_job_control")
    rows = []
    for requirement in requirements["aps"]:
        kind, binding = requirement["kind"], requirement["binding"]
        covered = projection.get("status") == "projected" and bool(observations)
        reason = "Conservative owner interval evidence"
        try:
            if kind in PROCESS_KINDS:
                ledgers = [_complete_target_ledger(row["parts"][binding["part"]]) for row in observations]
                covered = covered and (full or interval or invariant and all(row == ledgers[0] for row in ledgers))
                reason = "Exact completion ledgers and owner-supported progression or invariance"
            elif kind in SPATIAL_KINDS:
                field = "occupancy_possibilities" if kind == "resource_region" else "part_occupancy_possibilities"
                subject = binding["resource"] if kind == "resource_region" else binding["part"]
                for observation in observations:
                    entries = [observation, *observation.get("continuous_cells", [])]
                    for entry in entries:
                        values = entry[field][binding["region"]][subject]
                        if not isinstance(values, list) or not values or any(type(value) is not bool for value in values):
                            raise ValueError("Occupancy requires explicit Boolean alternatives")
                covered = covered and (interval or instantaneous)
                reason = "Supported occupancy envelopes cover the required boundaries and interval"
            else:
                covered = covered and full
        except (KeyError, TypeError, ValueError, IndexError):
            covered = False
        if not covered:
            reason = ("Missing exact completion ledgers or supported progression/invariance" if kind in PROCESS_KINDS
                      else "Missing supported occupancy coverage for the required boundaries or interval" if kind in SPATIAL_KINDS
                      else "Missing conservative owner interval evidence")
        rows.append({**deepcopy(requirement), "status": "covered" if covered else "missing", "reason": reason})
    complete = (projection.get("status") == "projected" and bool(projection.get("successor_evidence"))
                and not requirements["unresolved"] and all(row["status"] == "covered" for row in rows)
                and (not requirements["conservative"] or full))
    result = {"requirements_fingerprint": requirements["fingerprint"], "binding": deepcopy(requirements["binding"]),
        "aps": rows, "status": "covered" if complete else "missing",
        "observations_fingerprint": fingerprint(observations), "elapsed_time": deepcopy(projection.get("elapsed_time")),
        "successor_fingerprint": fingerprint(projection.get("successor_evidence")),
        "execution_authorized": False}
    result["fingerprint"] = fingerprint(result)
    return result


def monitor_evidence_record(projection: dict) -> dict:
    """Retain compact predicate evidence that can be independently replayed."""
    required = deepcopy(projection["evidence_requirements"])
    record = {key: deepcopy(projection[key]) for key in (
        "status", "observations", "projection_kind", "interval_complete", "elapsed_time", "owner_interval_contract",
        "endpoint_evidence_policy_version")
        if key in projection}
    record["successor_evidence"] = {"fingerprint": fingerprint(projection.get("successor_evidence"))}
    return {"requirements": required, "projection": record,
            "coverage": evidence_coverage(requirements=required, projection=record)}


def check_monitor_evidence(context: dict) -> None:
    """Check retained coverage and observation continuity before monitor replay."""
    records = context.get("evidence_history")
    if not isinstance(records, list) or not records:
        raise ValueError("Selected evidence history is missing")
    joined = []
    for record in records:
        projection = record["projection"]
        expected = evidence_coverage(requirements=record["requirements"], projection=projection)
        if record["coverage"] != expected or expected["status"] != "covered":
            raise ValueError("Retained evidence coverage cannot be reproduced")
        observations = deepcopy(projection["observations"])
        joined.extend(observations[1:] if joined else observations)
    actual = deepcopy(context.get("observations"))
    for row in actual or []:
        row.pop("rule_cells", None)
    if actual != joined:
        raise ValueError("Retained predicate observations differ from their coverage history")


def check_evidence_policy(*, task: dict, snapshot: dict, model: dict, physical_input: dict,
                          specifications: list | None, projection: dict) -> dict:
    """Independently check RA requirements, semantic effects, and per-AP coverage."""
    if physical_input.get("physical_evidence", {}).get("endpoint_evidence_policy_version") in (1, 2):
        physical_input = deepcopy(physical_input)
        physical_input.setdefault("projected_snapshot", deepcopy(snapshot))
        sanitize_validation_input(physical_input, task.get("resource_jid"))
    if ("projected_snapshot" in physical_input and physical_input["projected_snapshot"] != snapshot
            or physical_input.get("base_model_fingerprint") not in (None, fingerprint(model))):
        raise ValueError("Evidence requirements refer to a different starting state or owner model")
    requirements = event_evidence_requirements(task=task, snapshot=snapshot, model=model,
        physical_input=physical_input, specifications=specifications)
    if projection.get("evidence_requirements") != requirements:
        raise ValueError("Evidence requirements differ from the selected specifications or starting facts")
    bounded = physical_input.get("physical_evidence", {}).get("endpoint_evidence_policy_version") in (1, 2)
    if bounded or projection.get("projection_kind") == "owner_effects_with_predicate_invariants":
        calculated = project_selected_successor(task=task, snapshot=snapshot, model=model,
            physical_input=physical_input, requirements=requirements)
        if calculated is None or any(projection.get(key) != value for key, value in calculated.items()):
            raise ValueError("Owner successor effects or predicate invariants cannot be independently reproduced")
    else:
        from cais_spade_llm.resources.machine.machine_jobs import missing_processing_projection

        missing = missing_processing_projection(physical_input.get("physical_evidence") or {}, projection)
        if missing:
            raise ValueError("Required processing progression is unresolved: " + ", ".join(missing))
    from cais_spade_llm.recovery_framework.recovery_observations import projection_capture_compatibility

    compatibility = projection_capture_compatibility(
        physical={**(physical_input.get("physical_evidence") or {}), "outline_safety_specifications": specifications},
        task=task, resource_evidence=physical_input.get("resource_evidence") or {},
                projection={} if bounded or projection.get("projection_kind") == "owner_effects_with_predicate_invariants" else projection)
    if compatibility["status"] == "NEEDS_CONTEXT":
        raise ValueError("Required captured facts are incompatible with the selected predicate evidence")
    coverage = evidence_coverage(requirements=requirements, projection=projection)
    if projection.get("evidence_coverage") != coverage:
        raise ValueError("Evidence coverage differs from independently calculated coverage")
    if coverage["status"] != "covered":
        raise ValueError("; ".join(projection.get("unresolved") or ["Selected predicates lack required evidence coverage"]))
    return coverage


def predicate_checkpoint(observation: dict, rules: list[dict]) -> dict:
    """Extract only required facts when comparing consecutive monitor boundaries."""
    from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _ap_binding

    result = {}
    for rule in rules:
        for ap in rule["aps"]:
            binding, kind = _ap_binding(rule, ap), physical_ap_kind(ap)
            key = rule["rule_id"] + "/" + ap["label"]
            if kind in PROCESS_KINDS:
                part = observation["parts"][binding["part"]]
                result[key] = {field: deepcopy(part.get(field)) for field in ("processCompleted", "processCompleted_complete")}
            elif kind in SPATIAL_KINDS:
                field = "occupancy_possibilities" if kind == "resource_region" else "part_occupancy_possibilities"
                subject = binding["resource"] if kind == "resource_region" else binding["part"]
                result[key] = deepcopy(observation[field][binding["region"]][subject])
            else:
                return {field: deepcopy(observation.get(field)) for field in ("resources", "parts", "frame", "state_snapshot")}
    return result


def predicate_geometry_identity(geometry: dict, rules: list[dict]) -> dict:
    """Bind retained history to the regions and targets that define its APs."""
    from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _ap_binding

    result = {"regions": {}, "targets": {}}
    for rule in rules:
        for ap in rule["aps"]:
            binding = _ap_binding(rule, ap)
            if "region" in binding:
                result["regions"][binding["region"]] = deepcopy(geometry["regions"][binding["region"]])
            if physical_ap_kind(ap) == "process_target_completed":
                result["targets"][binding["part"]] = deepcopy(geometry["parts"][binding["part"]].get("target"))
    return result
