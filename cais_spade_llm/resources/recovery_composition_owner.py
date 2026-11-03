from __future__ import annotations

"""Bind primitive composition validation to resource-owned evidence and evaluators."""

from copy import deepcopy
from typing import Any

from cais_spade_llm.resources.recovery_composition import validate_recovery_primitive_composition
from cais_spade_llm.resources.resource_primitives import filter_recovery_primitive_catalog


def validate_owner_composition(owner: Any, *, task: dict, primitive_steps: list,
                               prepared: dict, recovery_snapshot: dict,
                               part_context: dict, grounding_context: dict) -> dict:
    """Validate an exact event program without reading or controlling equipment.

    Args:
        owner: Responsible live RA or detached saved resource.
        task: Immutable accepted outline event.
        primitive_steps: Authored program to validate.
        prepared: Bound models and recorded evidence for this attempt.
        recovery_snapshot: State at the accepted composition prefix.
        part_context: Product facts at the same prefix.
        grounding_context: Recorded values available to parameter references.

    Returns:
        Bound RA composition evidence with execution authorization disabled.
    """
    jid = task["resource_jid"]
    if str(owner.jid) != jid:
        raise ValueError("Composition was assigned to a different resource owner")
    entry = prepared.get("recovery_resources", {}).get(jid, {})
    catalog = filter_recovery_primitive_catalog(
        entry.get("execution_primitive_catalog") or [], resource_type=entry.get("resource_type", ""))
    evidence = deepcopy(grounding_context.get("resources", {}).get(jid, {}).get("resource_evidence")
                        or entry.get("resource_evidence") or {})
    model = entry.get("recovery_des_model") or {}
    physical = deepcopy(prepared.get("physical_evidence") or {})
    for resource_jid, current in grounding_context.get("resources", {}).items():
        resource_id = current.get("resource_id") or next(
            (name for name, value in physical.get("jids", {}).items() if value == resource_jid), None)
        declarations = physical.get("environment_models", {}).get(resource_id, {}).get("state_variables", {})
        if resource_id in physical.get("environment_valuation", {}):
            physical["environment_valuation"][resource_id].update(
                {key: deepcopy(current[key]) for key in declarations if key in current})
    for name, current in grounding_context.get("parts", {}).items():
        if name in physical.get("products", {}):
            physical["products"][name].update(deepcopy(current))
    physical_input = {"resource_evidence": evidence, "physical_evidence": physical,
                      "part_context": deepcopy(part_context), "primitive_support_semantics": "successor_primitive_union"}
    prepare = getattr(owner, "prepare_recovery_primitive_parameters", None)
    if callable(prepare):
        preparation = prepare(task=task, recovery_des_model=model,
            recovery_snapshot=recovery_snapshot, physical_input=physical_input)
        evidence = deepcopy(preparation.get("calculated_resource_evidence", evidence))

    hold, advance, access, picks = None, None, None, {}

    def capability(**values) -> dict:
        if hold is not None and values["primitive"] == "compute_place_targets":
            hold(values["recovery_snapshot"])
        if access is not None and values["primitive"] in {"start_job", "resume_job"}:
            refusal = access(values["params"])
            if refusal:
                return refusal
        evaluator = getattr(owner, "check_recovery_primitive_feasibility", None)
        if not callable(evaluator):
            return {"feasibility_status": "NEEDS_CONTEXT", "reason": "Owner capability evaluator is missing"}
        checked = evaluator(**values, task=deepcopy(task), grounded_action={},
                            resource_evidence=deepcopy(evidence), physical_evidence=deepcopy(physical))
        # A job command the owner accepts leaves its job record; the program's later steps start from it.
        record = (checked.get("evidence") or {}).get("projected_processing_observations")
        if checked.get("feasibility_status") == "FEASIBLE" and isinstance(record, dict):
            evidence["processing_observations"] = deepcopy(record)
        if advance is not None and checked.get("feasibility_status") == "FEASIBLE":
            refusal = advance(primitive=values["primitive"], params=values["params"],
                              recovery_snapshot=values["recovery_snapshot"], checked=checked)
            if refusal:
                return {"feasibility_status": "INFEASIBLE", "allowed": False, "reason": refusal}
        return checked

    preview, completion = None, None
    if entry.get("resource_type") == "robot":
        from cais_spade_llm.resources.robot.recovery_composition import (
            hold_picked_part, preview_composition_output, project_composition_completion,
        )

        def hold(snapshot: dict) -> None:
            # A place calculated after the event's own grasp reads the part as that grasp holds it.
            hold_picked_part(evidence, task=task, recovery_des_model=model, recovery_snapshot=snapshot,
                             pick_target=picks.get(task.get("part_name")))

        def preview(**values) -> dict:
            if values["primitive"] == "compute_place_targets":
                hold(values["recovery_snapshot"])
            output = preview_composition_output(**values, task=task, resource_evidence=evidence,
                                                physical_evidence=physical)
            if values["primitive"] == "compute_pick_targets" and output.get("status") == "calculated":
                picks[task.get("part_name")] = deepcopy(output["output"])
            return output

        def completion(**values) -> dict:
            return project_composition_completion(**values, primitive_catalog=catalog,
                resource_evidence=evidence, physical_evidence=physical,
                locate_pose=lambda pose, stated=None: _locate(pose, prepared, jid, stated))
    elif entry.get("resource_type") == "kmr":
        from cais_spade_llm.recovery_framework.kmr_composition import (
            advance_kmr_program, preview_kmr_output, project_kmr_completion,
        )

        def advance(**values) -> str | None:
            return advance_kmr_program(evidence, picks, task=task, **values)

        def preview(**values) -> dict:
            return preview_kmr_output(picks, task=task, **values)

        def completion(**values) -> dict:
            return project_kmr_completion(**values, targets=picks, resource_evidence=evidence, physical_evidence=physical)
    elif isinstance(((evidence.get("inputs") or {}).get("scene") or {}).get(evidence.get("resource_id"), {}).get(
            "simulation_transport"), dict):
        from cais_spade_llm.recovery_framework.transport_composition import (
            advance_transport_program, preview_transport_output, project_transport_completion,
        )

        def advance(**values) -> str | None:
            return advance_transport_program(evidence, picks, task=task, **values)

        def preview(**values) -> dict:
            return preview_transport_output(picks, **values)

        def completion(**values) -> dict:
            return project_transport_completion(**values, targets=picks, resource_evidence=evidence)
    elif isinstance(evidence.get("processing_observations"), dict) and _scene_machine(evidence) is not None:
        from cais_spade_llm.resources.environment_recovery import composed_machine_access

        def access(params: dict) -> dict | None:
            # A job's access is worked out here, where the state of the other resources is, and read by the machine's check.
            job = evidence["processing_observations"].get("jobs", {}).get(params.get("job_id"), {})
            part = params.get("part_name", job.get("part_name"))
            if part is None:
                # An unknown job is the machine's own refusal, which names it.
                return None
            try:
                observation = composed_machine_access(evidence, physical, grounding_context, part)
            except (KeyError, TypeError, ValueError) as exc:
                missing = not isinstance(exc, ValueError) or str(exc).startswith("NEEDS_CONTEXT")
                return {"feasibility_status": "NEEDS_CONTEXT" if missing else "INFEASIBLE", "allowed": False,
                        "reason": str(exc)}
            physical.setdefault("machine_access_observations", {})[evidence["resource_id"]] = observation
            return None

        def completion(*, task: dict, trace: list, projected_snapshot: dict, part_context: dict, **_unused) -> dict:
            # Later events start from the job record the program leaves, awaited completions included.
            try:
                evidence["processing_observations"] = _await_job_completions(evidence, trace, task.get("part_name"))
            except (KeyError, TypeError, ValueError) as exc:
                missing = not isinstance(exc, ValueError) or str(exc).startswith("NEEDS_CONTEXT")
                return {"feasibility_status": "NEEDS_CONTEXT" if missing else "INFEASIBLE", "reason": str(exc)}
            return {"feasibility_status": "FEASIBLE", "projected_snapshot": projected_snapshot,
                    "projected_part_context": part_context, "resource_evidence": deepcopy(evidence)}

    # A fact the owner's primitives declare but the composition prefix lacks starts from the owner's recorded
    # evidence, which its own capability check already reads; a recorded fact never replaces a projected one.
    recorded = evidence.get("snapshot") if isinstance(evidence.get("snapshot"), dict) else {}
    recorded_facts = {field: deepcopy(recorded[field]) for row in catalog if isinstance(row, dict)
                      for key in ("preconditions", "capability_constraints", "effects")
                      for field in (row.get(key) or {}) if field not in recovery_snapshot and field in recorded}
    return validate_recovery_primitive_composition(task=task, primitive_steps=primitive_steps,
        recovery_des_model=model, recovery_snapshot=recovery_snapshot, part_context=part_context,
        grounding_context=grounding_context, primitive_catalog=catalog,
        validate_primitive=capability, preview_primitive=preview, project_completion=completion,
        captured_snapshot=entry.get("recovery_snapshot"), recorded_facts=recorded_facts)


def _gripper_point(pose: dict, prepared: dict, jid: str) -> dict | None:
    """Return where a robot's gripper is at a pose of its controlled link, the inverse of the RA's ``reference_pose``.

    Returns:
        The point, or None when the recorded tool offset or the pose's orientation is missing.
    """
    from cais_spade_llm.recovery_framework.geometry import rotate

    physical = prepared.get("physical_evidence") or {}
    rid = next((name for name, value in physical.get("jids", {}).items() if value == jid), None)
    offset = (physical.get("resources", {}).get(rid) or {}).get("tcp_offset_z")
    values = [pose.get(axis) for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    if not all(isinstance(value, (int, float)) for value in (offset, *values)):
        return None
    along = rotate(values[3:], [0.0, 0.0, abs(offset)])
    return {axis: values[index] + along[index] for index, axis in enumerate("xyz")}


def place_holding(point: dict, prepared: dict, stated: Any = None) -> str | None:
    """Name the place whose recorded area holds a gripper point, where the RA reaches an area.

    An area belongs to the body whose surface it is, else to the place its name starts with, as the RA names a
    place's areas (``ra_feasibility._Check.areas``), so a tray beside a machine is its own place. A stated place
    holds when one of its areas does; otherwise the first declared place with such an area, or None.
    """
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_outline_facts import (
        _containing_regions,
        _declared_locations,
    )

    regions = prepared.get("diagnostic_regions", {})
    holding = [str((regions[name].get("source") or {}).get("surface_collision") or "").split("::")[0]
               or name.split(".")[0] for name in _containing_regions(point, prepared)]
    if stated in holding:
        return stated
    places = {place for entry in prepared.get("recovery_resources", {}).values() for place in _declared_locations(entry)}
    return next((place for place in holding if place in places), None)


def _locate(pose: dict, prepared: dict, jid: str, stated: str | None = None) -> str | None:
    """Name the place a robot is at: the one whose recorded area holds its gripper."""
    point = _gripper_point(pose, prepared, jid)
    return None if point is None else place_holding(point, prepared, stated)


def _scene_machine(evidence: dict) -> dict | None:
    machines = (evidence.get("inputs") or {}).get("scene", {}).get("machines") or []
    return next((row for row in machines if row.get("resource_id") == evidence.get("resource_id")), None)


def _await_job_completions(evidence: dict, trace: list, part: str | None) -> dict:
    """Finish the active job at each awaited ``complete_job``, as the machine's own record does."""
    from cais_spade_llm.resources.machine.machine_jobs import project_job_completion

    record = deepcopy(evidence["processing_observations"])
    for row in trace:
        if row.get("controllable") is not False or row.get("primitive") != "complete_job":
            continue
        job = record["jobs"].get(record.get("active_job"))
        if job is None or (part is not None and job.get("part_name") != part):
            raise ValueError(f"NEEDS_CONTEXT: the machine's job record has no active job of {part}")
        result = (row.get("resolved_params") or {}).get("result", job.get("result"))
        if result != job.get("result"):
            raise ValueError(f"The awaited job of {job.get('part_name')} completes with result {job.get('result')!r}, "
                             f"not {result!r}")
        # Over the job's remaining processing time, under the workholding observations it last recorded.
        record = project_job_completion(_scene_machine(evidence), record,
            job["processing_time_sec"] - job["processing_active_sec"], job.get("observations") or {})
    return record
