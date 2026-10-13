from __future__ import annotations

"""Agent-owned outline validation using frozen facts and projected evidence."""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)

OUTLINE_VALIDATION_CONTRACT_VERSION = 2


def rejected_outline_capability_projection(*, record: dict, task: dict, recovery_snapshot: dict,
                                          recovery_des_model: dict, physical_evidence: dict,
                                          resource_evidence: dict, static_capabilities: dict) -> dict:
    """Recalculate an owner target rejection from the same recorded facts."""
    from cais_spade_llm.resources.recovery_geometry import outline_event_binding
    from cais_spade_llm.resources.resource_safety_preparation import prepare_outline_event_preview_request

    binding = outline_event_binding(task, recovery_snapshot, recovery_des_model, physical_evidence)
    if (record.get("status") != "rejected" or record.get("validation_status") != "rejected"
            or record.get("physical_capability_rejected") is not True or record.get("task") != task
            or record.get("event_record_binding") != binding
            or record.get("resource_evidence_fingerprint") != recovery_validation_fingerprint(resource_evidence)
            or record.get("static_capabilities_fingerprint") != recovery_validation_fingerprint(static_capabilities)):
        raise ValueError("Rejected owner capability does not match the current proposal and recorded facts")
    checked = prepare_outline_event_preview_request(task=task, recovery_snapshot=recovery_snapshot,
        recovery_des_model=recovery_des_model, physical_evidence=physical_evidence,
        resource_evidence=resource_evidence, static_capabilities=static_capabilities)
    for key in ("status", "validation_status", "physical_capability_rejected", "capability_result",
                "event_record_binding", "resource_evidence_fingerprint", "static_capabilities_fingerprint",
                "motion_requests", "task"):
        if key not in checked or record.get(key) != checked[key]:
            raise ValueError(f"Rejected owner capability cannot be reproduced: {key}")
    return {"status": "rejected", "validation_status": "rejected", "physical_capability_rejected": True,
        "reason": checked["reason"], "capability_result": deepcopy(checked["capability_result"]),
        "capability_source": {"kind": "recorded", "ref": f"/resources/{task['resource_jid']}/static_capabilities",
                              "capture": deepcopy(resource_evidence.get("source") or {})},
        "observations": [], "interval_complete": False, "context_requests": [], "binding": binding,
        "scope": "The calculated owner motion targets and recorded capability limits",
        "command_sent": False, "execution_authorized": False}


def incomplete_outline_motion_projection(*, record: dict, task: dict, recovery_snapshot: dict,
                                         recovery_des_model: dict, physical_evidence: dict,
                                         resource_evidence: dict) -> dict:
    """Retain an incomplete supplied motion without inferring impossibility.

    A complete recorded solver query can inform a different proposal. Missing
    inputs, unavailable solvers, canceled work, and deadlines cannot do so.
    """
    from cais_spade_llm.resources.recovery_geometry import outline_event_binding
    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record, validate_preview_request

    binding = outline_event_binding(task, recovery_snapshot, recovery_des_model, physical_evidence)
    original = record.get("task") or {}
    motion = record.get("incomplete_motion") or {}
    request = motion.get("request") or {}
    if (record.get("status") != "NEEDS_CONTEXT" or record.get("validation_status") != "NEEDS_CONTEXT"
            or record.get("physical_motion_unresolved") is not True
            or record.get("event_record_binding") != binding
            or outline_event_binding(original, recovery_snapshot, recovery_des_model, physical_evidence) != binding
            or record.get("resource_evidence_fingerprint") != recovery_validation_fingerprint(resource_evidence)
            or motion.get("source") != "PreviewRecoveryMotion" or motion.get("status") != "NEEDS_CONTEXT"
            or motion.get("sampled_collision_rejected") is not False or motion.get("path_complete") is not False
            or motion.get("command_sent") is not False or motion.get("execution_authorized") is not False
            or request.get("proposal_fingerprint") != recovery_validation_fingerprint(original)):
        raise ValueError("Incomplete owner motion does not match the current conditions and recorded inputs")
    validate_preview_request(request)
    checked = preview_result_record(request, motion)
    diagnostic = checked.get("cartesian_failure_evidence") or {}
    if (request.get("mode") != "cartesian" or diagnostic.get("complete_inputs") is not True
            or diagnostic.get("failure_kind") not in {"ik_search_failed", "joint_jump"}
            or type(motion.get("fraction")) not in (float, int) or not 0 <= motion["fraction"] < 1):
        raise ValueError("Incomplete motion has no complete recorded solver query permitting proposal revision")
    matches = [rid for rid, jid in physical_evidence.get("jids", {}).items() if jid == task["resource_jid"]]
    if matches != [request["resource_id"]]:
        raise ValueError("Incomplete owner motion identifies another resource")
    policy = {key: deepcopy(request.get(key)) for key in (
        "group_name", "link_name", "frame_id", "mode", "max_translation_step", "max_rotation_step",
        "max_joint_step", "velocity_scaling", "acceleration_scaling", "timeout_seconds")}
    reason = (f"The supplied Cartesian motion did not produce a complete path "
              f"(fraction {motion['fraction']}, {diagnostic['failure_kind']}). "
              "This does not establish that the proposed transition is infeasible.")
    summary = {"source": "PreviewRecoveryMotion", "source_ref": "incomplete_motion.cartesian_diagnostics_json",
        "failure_kind": diagnostic["failure_kind"], "fraction": motion["fraction"]}
    failed = [row for row in diagnostic.get("ik_queries", []) if row.get("success") is False]
    if failed:
        summary["failed_query"] = {key: failed[-1][key] for key in ("index", "error_code", "timeout_seconds")
                                   if type(failed[-1].get(key)) in (int, float) and math.isfinite(failed[-1][key])}
    last = diagnostic["waypoints"][-1].get("last_accepted_state", {}).get("joint_state", {})
    positions = dict(zip(last.get("name", []), last.get("position", [])))
    bounded = [{**{key: row[key] for key in ("name", "min_position", "max_position")},
                "position": positions[row["name"]]} for row in diagnostic.get("joint_bounds", [])
        if row.get("position_bounded") is True and row.get("name") in positions
        and all(type(value) in (int, float) and math.isfinite(value)
                for value in (row.get("min_position"), row.get("max_position"), positions[row["name"]]))]
    if bounded:
        summary["last_accepted_nearest_bound"] = min(bounded, key=lambda row:
            min(abs(row["position"] - row["min_position"]), abs(row["max_position"] - row["position"])))
    return {"status": "needs_context", "validation_status": "NEEDS_CONTEXT", "physical_motion_unresolved": True,
        "reason": reason, "observations": [], "interval_complete": False, "context_requests": [],
        "incomplete_motion": deepcopy(checked), "binding": binding,
        "solver_summary": summary,
        "revision_identity": recovery_validation_fingerprint({"binding": binding, "policy": policy,
            "resource_evidence_fingerprint": record["resource_evidence_fingerprint"]}),
        "scope": "The supplied owner motion; feasibility of other motions remains unresolved",
        "command_sent": False, "execution_authorized": False}


def rejected_outline_motion_projection(*, record: dict, task: dict, recovery_snapshot: dict,
                                       recovery_des_model: dict, physical_evidence: dict) -> dict:
    """Verify a rejected supplied motion without declaring every motion infeasible."""
    from cais_spade_llm.resources.recovery_geometry import outline_event_binding
    from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record

    binding = outline_event_binding(task, recovery_snapshot, recovery_des_model, physical_evidence)
    if record.get("physical_endpoint_rejected") is True:
        from cais_spade_llm.resources.resource_safety_preparation import outline_event_preview_progress

        descriptor = record.get("endpoint_replay_preparation") or {}
        motions, requests = record.get("motion_results"), record.get("native_requests")
        endpoint, index = record.get("nominal_snap_result"), record.get("rejected_stage_index")
        if (record.get("status") != "rejected" or record.get("validation_status") != "rejected"
                or record.get("physical_motion_rejected") is not True or record.get("task") != task
                or record.get("event_record_binding") != binding or descriptor.get("status") != "ready"
                or descriptor.get("event_record", {}).get("binding") != binding
                or descriptor.get("event_record", {}).get("intervals") != []
                or type(index) is not int or index < 0 or descriptor.get("custody_motion_index") != index
                or not isinstance(motions, list) or not isinstance(requests, list)
                or len(motions) != index or len(requests) != len(motions)
                or any(row.get("status") != "PREVIEW_COMPLETE" or row.get("sampled_collision_rejected") is True for row in motions)
                or not isinstance(endpoint, dict) or endpoint.get("status") != "SCENE_ENDPOINT_REJECTED"):
            raise ValueError("Endpoint rejection has stale or incomplete proposal evidence")
        rid = descriptor.get("resource_id")
        matches = [name for name, jid in physical_evidence.get("jids", {}).items() if jid == task["resource_jid"]]
        contract = physical_evidence.get("resource_models", {}).get(rid, {}).get("event_projection_contracts", {}).get(
            descriptor["event_record"].get("contract_id"))
        if (matches != [rid] or not isinstance(contract, dict)
                or descriptor["event_record"].get("contract_fingerprint") != recovery_validation_fingerprint(contract)):
            raise ValueError("Endpoint rejection has changed owner contract evidence")
        preview_result_record(endpoint.get("request") or {}, endpoint)
        prepared = {**deepcopy(descriptor), "task": deepcopy(task), "recovery_snapshot": deepcopy(recovery_snapshot),
            "recovery_des_model": deepcopy(recovery_des_model), "physical_evidence": deepcopy(physical_evidence),
            "native_requests": deepcopy(requests), "nominal_snap_result": deepcopy(endpoint)}
        checked = outline_event_preview_progress(prepared_preview=prepared, motion_results=motions)
        for field in ("status", "validation_status", "physical_motion_rejected", "physical_endpoint_rejected",
                      "rejected_stage_index", "event_record_binding", "endpoint_rejection", "nominal_snap_result", "task"):
            if field not in checked or checked[field] != record.get(field):
                raise ValueError("Endpoint rejection cannot be reproduced: " + field)
        return {"status": "rejected", "validation_status": "rejected", "physical_motion_rejected": True,
            "physical_endpoint_rejected": True, "reason": checked["reason"],
            "endpoint_rejection": deepcopy(checked["endpoint_rejection"]), "binding": binding,
            "observations": [], "interval_complete": False, "context_requests": [],
            "scope": "The supplied owner endpoint; alternative endpoints have not been exhausted",
            "command_sent": False, "execution_authorized": False}
    if record.get("continuous_mating_rejected") is True:
        from cais_spade_llm.resources.resource_safety_preparation import outline_event_preview_progress

        descriptor = record.get("mating_replay_preparation") or {}
        motions = record.get("motion_results")
        requests = record.get("native_requests")
        index = record.get("rejected_stage_index")
        if (record.get("status") != "rejected" or record.get("validation_status") != "rejected"
                or record.get("physical_motion_rejected") is not True or record.get("task") != task
                or record.get("event_record_binding") != binding or descriptor.get("status") != "ready"
                or descriptor.get("event_record", {}).get("binding") != binding
                or descriptor.get("event_record", {}).get("intervals") != []
                or type(index) is not int or index < 0 or not isinstance(motions, list)
                or not isinstance(requests, list) or len(motions) != index + 1 or len(requests) != len(motions)
                or record.get("rejected_motion") != motions[-1]
                or any(row.get("status") != "PREVIEW_COMPLETE" or row.get("sampled_collision_rejected") is True
                       for row in motions)):
            raise ValueError("Continuous mating rejection has stale or incomplete proposal evidence")
        rid = descriptor.get("resource_id")
        matches = [name for name, jid in physical_evidence.get("jids", {}).items() if jid == task["resource_jid"]]
        contract = physical_evidence.get("resource_models", {}).get(rid, {}).get("event_projection_contracts", {}).get(
            descriptor["event_record"].get("contract_id"))
        if (matches != [rid] or not isinstance(contract, dict)
                or descriptor["event_record"].get("contract_fingerprint") != recovery_validation_fingerprint(contract)):
            raise ValueError("Continuous mating rejection has changed owner contract evidence")
        prepared = {**deepcopy(descriptor), "task": deepcopy(task), "recovery_snapshot": deepcopy(recovery_snapshot),
            "recovery_des_model": deepcopy(recovery_des_model), "physical_evidence": deepcopy(physical_evidence),
            "native_requests": deepcopy(requests)}
        checked = outline_event_preview_progress(prepared_preview=prepared, motion_results=motions)
        for field in ("status", "validation_status", "physical_motion_rejected", "continuous_mating_rejected",
                      "rejected_stage_index", "event_record_binding", "mating_rejection", "rejected_motion", "task"):
            if field not in checked or checked[field] != record.get(field):
                raise ValueError("Continuous mating rejection cannot be reproduced: " + field)
        return {"status": "rejected", "validation_status": "rejected", "physical_motion_rejected": True,
            "continuous_mating_rejected": True, "reason": checked["reason"],
            "mating_rejection": deepcopy(checked["mating_rejection"]),
            "rejected_motion": deepcopy(checked["rejected_motion"]), "binding": binding,
            "observations": [], "interval_complete": False, "context_requests": [],
            "scope": "The supplied owner motion; alternative motions have not been exhausted",
            "command_sent": False, "execution_authorized": False}
    motion = record.get("rejected_motion") or {}
    request = motion.get("request") or {}
    if (record.get("status") != "rejected" or record.get("validation_status") != "rejected"
            or record.get("physical_motion_rejected") is not True or record.get("task") != task
            or record.get("event_record_binding") != binding
            or motion.get("source") != "PreviewRecoveryMotion"
            or motion.get("sampled_collision_rejected") is not True
            or motion.get("status") == "PREVIEW_COMPLETE"
            or motion.get("command_sent") is not False or motion.get("execution_authorized") is not False
            or request.get("proposal_fingerprint") != recovery_validation_fingerprint(task)
            or not all(request.get(field) for field in (
                "resource_id", "capture_fingerprint", "proposal_fingerprint", "scene_fingerprint", "model_fingerprint"))):
        raise ValueError("Rejected owner motion does not match the current proposal and physical evidence")
    matches = [rid for rid, jid in physical_evidence.get("jids", {}).items() if jid == task["resource_jid"]]
    if matches != [request["resource_id"]]:
        raise ValueError("Rejected owner motion identifies another resource")
    preview_result_record(request, motion)
    return {"status": "rejected", "validation_status": "rejected", "physical_motion_rejected": True,
        "reason": record.get("reason") or "The checked owner motion intersects recorded geometry",
        "observations": [], "interval_complete": False, "context_requests": [],
        "rejected_motion": deepcopy(motion), "binding": binding,
        "scope": "The supplied owner motion; alternative motions have not been exhausted",
        "command_sent": False, "execution_authorized": False}


def uses_outline_evidence(payload: dict) -> bool:
    """Recognize new attempts without upgrading historical sessions."""
    return payload.get("validation_contract_version") == OUTLINE_VALIDATION_CONTRACT_VERSION


def load_outline_specifications(paths: list[str], root: Path) -> list[dict]:
    """Freeze selected specifications without altering identifiers or AP meanings."""
    records = []
    for source in dict.fromkeys(paths):
        path = Path(source)
        try:
            content = (path if path.is_absolute() else root / path).read_bytes()
            document = json.loads(content)
        except (OSError, ValueError) as exc:
            raise ValueError(f"NEEDS_CONTEXT: selected safety specification {source}: {exc}") from exc
        if not isinstance(document, dict) or not isinstance(document.get("catalog", {}).get("specifications"), list):
            raise ValueError(f"NEEDS_CONTEXT: invalid selected safety specification {source}")
        records.append({"source_path": source, "source_sha256": hashlib.sha256(content).hexdigest(),
                        "document": document})
    return records


def outline_evidence_requirements(prepared: dict) -> dict:
    """Record which saved declarations and physical facts selected APs require."""
    from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind

    fields = {
        "resource_region": ["geometry.resources", "geometry.regions", "snapshot.resources",
                            "resource_models", "geometry.parts", "snapshot.parts"],
        "part_region_entry": ["geometry.parts", "geometry.regions", "snapshot.parts", "monitor_history"],
        "process_target_completed": ["snapshot.parts.processCompleted", "snapshot.parts.processCompleted_evidence", "geometry.parts.target"],
        "process_result_completed": ["snapshot.parts.processCompleted", "snapshot.parts.processCompleted_evidence"],
    }
    rules = []
    for record in prepared.get("outline_safety_specifications", []):
        for rule in record["document"]["catalog"]["specifications"]:
            aps = []
            for ap in rule.get("aps", []):
                try:
                    kind = physical_ap_kind(ap)
                except (ValueError, KeyError, TypeError):
                    kind = None
                aps.append({"label": ap.get("label"), "definition": deepcopy(ap),
                            "required_fields": fields.get(kind, []),
                            "status": "evidence_required" if kind in fields else "NEEDS_CONTEXT"})
            rules.append({"source_path": record["source_path"], "source_sha256": record["source_sha256"],
                          "id": rule["id"], "aps": aps})
    resources = {jid: {"model_fingerprint": recovery_validation_fingerprint(entry.get("recovery_des_model")),
                        "state_variables": deepcopy((entry.get("recovery_des_model") or {}).get("state_variables")),
                        "primitive_catalog_recorded": isinstance((entry.get("recovery_des_model") or {}).get("primitive_catalog"), list)}
                 for jid, entry in prepared.get("recovery_resources", {}).items()}
    return {"rules": rules, "resources": resources,
            "physical_evidence_fingerprint": recovery_validation_fingerprint(prepared.get("physical_evidence")),
            "motion_evidence": ("Required separately for the proposed event; unordered primitive support "
                                "does not establish a motion interval"),
            "execution_authorized": False}


def validate_resource_outline_payload(owner: Any, payload: dict, *, offline: bool = False) -> dict:
    """Validate resource support separately from evidence needed by CCA."""
    from cais_spade_llm.resources.recovery_feasibility import validate_recovery_transition
    from cais_spade_llm.resources.recovery_geometry import project_outline_evidence

    snapshot = deepcopy(owner.get_recovery_snapshot())
    model = deepcopy(owner.recovery_des_model(snapshot=snapshot))
    semantics = payload.get("primitive_support_semantics")
    results = []
    for candidate in payload.get("candidates", []):
        task = deepcopy(candidate["task"])
        physical = candidate.get("physical_input") or {}
        projected = deepcopy(physical.get("projected_snapshot") or snapshot)
        part = deepcopy(physical.get("part_context") or {})
        if task.get("resource_jid") != str(owner.jid):
            result = {"allowed": False, "feasibility_status": "INFEASIBLE",
                      "reason": "Proposed resource does not match the validating ResourceAgent"}
        elif physical.get("base_snapshot_fingerprint") not in (None, recovery_validation_fingerprint(snapshot)):
            result = {"allowed": False, "feasibility_status": "NEEDS_CONTEXT",
                      "reason": "Resource observations changed after capture"}
        elif physical.get("base_model_fingerprint") not in (None, recovery_validation_fingerprint(model)):
            result = {"allowed": False, "feasibility_status": "NEEDS_CONTEXT",
                      "reason": "Resource declarations changed after capture"}
        elif semantics == "successor_primitive_union":
            evaluator = getattr(owner, "validate_recovery_primitive_support", None)
            if not callable(evaluator):
                result = {"allowed": False, "feasibility_status": "NEEDS_CONTEXT",
                          "reason": "Resource-owned primitive capability evaluator is unavailable"}
            else:
                calculation_model = model
                if model.get("resource_id") == "Conveyor" and model.get("source_model_fingerprint"):
                    calculation_model = deepcopy(owner.recovery_des_model(snapshot=projected))
                if calculation_model.get("source_model_fingerprint") != model.get("source_model_fingerprint"):
                    result = {"allowed": False, "feasibility_status": "NEEDS_CONTEXT",
                              "reason": "Resource collection declarations changed after capture"}
                else:
                    result = evaluator(
                        task=task, recovery_des_model=calculation_model, recovery_snapshot=projected,
                        physical_input={**deepcopy(physical), "primitive_support_semantics": semantics})
                    if calculation_model is not model:
                        result["successor_calculation"] = {
                            "source_model_fingerprint": model["source_model_fingerprint"],
                            "snapshot_fingerprint": recovery_validation_fingerprint(projected),
                            "calculation_model_fingerprint": recovery_validation_fingerprint(calculation_model)}
        elif semantics is not None:
            result = {"allowed": False, "feasibility_status": "NEEDS_CONTEXT",
                      "reason": "Unsupported primitive_support_semantics"}
        else:
            result = validate_recovery_transition(task=task, recovery_des_model=model,
                                                 recovery_snapshot=projected, part_context=part,
                                                 parameter_evidence=physical.get("parameter_evidence"),
                                                 required_task_contract=task.get("required_task_contract"))
        result = deepcopy(result)
        result.update(candidate_index=candidate.get("candidate_index", 0),
                      event_id=candidate.get("event_id", task.get("outline_id")),
                      candidate_fingerprint=recovery_validation_fingerprint(task),
                      validation_contract_version=OUTLINE_VALIDATION_CONTRACT_VERSION,
                      execution_authorized=False, planning_only=True)
        if semantics is not None:
            result["primitive_support_semantics"] = semantics
        if result.get("allowed") is True and semantics == "successor_primitive_union":
            # This owner interface accepts event evidence, never the unordered
            # primitive union. A missing physical projection cannot undo RA's
            # independently established capability result.
            projector = getattr(owner, "project_recovery_outline_event", None)
            if callable(projector):
                result["geometry_evidence"] = deepcopy(projector(
                    task=deepcopy(task), recovery_snapshot=deepcopy(projected),
                    recovery_des_model=deepcopy(model), physical_input=deepcopy(physical)))
            else:
                result["geometry_evidence"] = {
                    "status": "needs_context", "interval_complete": False,
                    "unresolved": ["Resource-owned event projection is unavailable"],
                    "context_requests": ["/physical_evidence", f"/resources/{owner.jid}/outline_event_evidence"],
                    "requires_projection": True, "execution_authorized": False,
                }
            from cais_spade_llm.resources.machine.machine_jobs import missing_processing_projection
            missing_processing = missing_processing_projection(physical.get("physical_evidence") or {},
                                                                 result["geometry_evidence"])
            if missing_processing:
                result["geometry_evidence"].update(status="needs_context", interval_complete=False,
                    unresolved=["Processing over the proposed interval is unresolved: " + ", ".join(missing_processing)],
                    context_requests=["/physical_evidence/processing_observations"], requires_projection=True)
            if result["geometry_evidence"].get("status") != "projected":
                from cais_spade_llm.resources.recovery_geometry import outline_event_binding

                scene = physical.get("physical_evidence") or {}
                binding = outline_event_binding(task, projected, model, scene)
                capability_rejections = [row for row in scene.get("rejected_capability_evidence", [])
                    if row.get("event_record_binding") == binding and row.get("task") == task]
                rejected = [row for row in scene.get("rejected_event_evidence", [])
                            if row.get("event_record_binding") == binding and row.get("task") == task]
                if capability_rejections:
                    result["geometry_evidence"] = rejected_outline_capability_projection(
                        record=capability_rejections[-1], task=task, recovery_snapshot=projected,
                        recovery_des_model=model, physical_evidence=scene,
                        resource_evidence=owner.get_recovery_resource_evidence(),
                        static_capabilities=owner.static_capabilities)
                elif rejected:
                    result["geometry_evidence"] = rejected_outline_motion_projection(record=rejected[-1], task=task,
                        recovery_snapshot=projected, recovery_des_model=model, physical_evidence=scene)
                elif scene.get("incomplete_event_evidence"):
                    facts = owner.get_recovery_resource_evidence()
                    incomplete = [row for row in scene.get("incomplete_event_evidence", [])
                        if row.get("event_record_binding") == binding
                        and row.get("resource_evidence_fingerprint") == recovery_validation_fingerprint(facts)]
                    if incomplete:
                        result["geometry_evidence"] = incomplete_outline_motion_projection(
                            record=incomplete[-1], task=task, recovery_snapshot=projected,
                            recovery_des_model=model, physical_evidence=scene, resource_evidence=facts)
            from cais_spade_llm.recovery_framework.recovery_observations import projection_capture_compatibility

            geometry = result["geometry_evidence"]
            compatibility = projection_capture_compatibility(
                physical=physical.get("physical_evidence") or {}, task=task,
                resource_evidence=physical.get("resource_evidence") or {}, projection=geometry)
            geometry["capture_compatibility"] = compatibility
            if compatibility["status"] == "NEEDS_CONTEXT" and geometry.get("status") != "rejected":
                missing = [row for row in compatibility["checks"] if row["status"] == "NEEDS_CONTEXT"]
                geometry.update(status="needs_context", interval_complete=False,
                    unresolved=list(dict.fromkeys([*geometry.get("unresolved", []), *(row["reason"] for row in missing)])),
                    context_requests=list(dict.fromkeys([*geometry.get("context_requests", []),
                        *(ref for row in missing for ref in row.get("context_requests", []))])),
                    requires_projection=True, execution_authorized=False)
        elif result.get("allowed") is True:
            evidence = deepcopy(physical.get("physical_evidence") or {})
            captured = deepcopy(evidence)
            evidence["step_evidence"] = {}
            owners = [rid for rid, jid in evidence.get("jids", {}).items() if jid == str(owner.jid)]
            if len(owners) == 1:
                from cais_spade_llm.resources.resource_safety_preparation import (
                    prepare_recovery_outline_preview,
                )

                rid = owners[0]
                descriptor = evidence.get("resource_models", {}).get(rid, {}).get("primitive_model")
                if descriptor:
                    preview = prepare_recovery_outline_preview(
                        resource_id=rid, primitive_support=result["primitive_support"],
                        projected_start=evidence.get("snapshot", {}).get("resources", {}).get(rid),
                        isolated_scene=captured, primitive_model=descriptor)
                    result["owner_preview"] = deepcopy(preview)
                    if preview.get("status") == "projected" and preview.get("step_evidence") is not None:
                        evidence["step_evidence"] = deepcopy(preview["step_evidence"])
            result["geometry_evidence"] = project_outline_evidence(
                task=task, primitive_support=result["primitive_support"], evidence=evidence)
            geometry = result["geometry_evidence"]
            if geometry.get("status") != "projected":
                result.update(allowed=False, feasibility_status="NEEDS_CONTEXT",
                              constraint_code="resource_validation_unavailable",
                              reason="; ".join(geometry.get("unresolved") or ["Projected geometry is unavailable"]),
                              context_requests=deepcopy(geometry.get("context_requests") or []),
                              requires_projection=bool(geometry.get("requires_projection")))
        elif result.get("feasibility_status") == "NEEDS_CONTEXT":
            result.setdefault("context_requests", [f"/resources/{owner.jid}/recovery_des_model",
                                                    f"/resources/{owner.jid}/recovery_snapshot"])
        result.setdefault("findings", [] if result.get("allowed") else [{
            "validation_category": "transition_feasibility", "constraint_owner": "resource",
            "constraint_code": "resource_validation_unavailable" if result.get("feasibility_status") == "NEEDS_CONTEXT"
            else "transition_feasibility_rejected", "reason": result.get("reason", ""),
        }])
        results.append(result)
    return {"validator_jid": str(owner.jid), "snapshot": snapshot,
            "snapshot_fingerprint": recovery_validation_fingerprint(snapshot),
            "recovery_des_model": model,
            "recovery_des_model_fingerprint": model.get("descriptor_fingerprint"),
            "evidence_source": "saved_observations" if offline else "resource",
            "results": results, "execution_authorized": False}


def validate_safety_outline_payload(payload: dict, *, validator_jid: str = "cca") -> dict:
    """Evaluate each frozen rule separately without granting execution permission."""
    from cais_spade_llm.agents.central_controller.outline_evidence_safety import (
        validate_outline_evidence_safety,
    )

    results = []
    for candidate in payload.get("candidates", []):
        safety = candidate.get("safety_input") or {}
        documents = safety.get("specifications")
        before = deepcopy(safety.get("monitor_context") or {})
        checks, after = [], deepcopy(before)
        projection = safety.get("projection") or {}
        if projection.get("status") == "rejected" and projection.get("owner_effect_rejected") is True:
            results.append({"candidate_index": candidate.get("candidate_index", 0),
                "event_id": candidate.get("event_id"), "is_safe": False, "validation_status": "rejected",
                "owner_effect_rejected": True, "formal_safety_status": "not_checked", "checks": [],
                "monitor_context_before": before, "monitor_context_after": deepcopy(before),
                "candidate_fingerprint": safety.get("candidate_fingerprint"),
                "reason": projection["reason"], "owner_effect_result": deepcopy(projection["owner_effect_result"]),
                "execution_authorized": False, "planning_only": True})
            continue
        if projection.get("status") == "rejected" and projection.get("physical_capability_rejected") is True:
            results.append({"candidate_index": candidate.get("candidate_index", 0),
                "event_id": candidate.get("event_id"), "is_safe": False, "validation_status": "rejected",
                "physical_capability_rejected": True, "formal_safety_status": "not_checked", "checks": [],
                "monitor_context_before": before, "monitor_context_after": deepcopy(before),
                "candidate_fingerprint": safety.get("candidate_fingerprint"),
                "reason": projection["reason"], "capability_result": deepcopy(projection["capability_result"]),
                "capability_source": deepcopy(projection["capability_source"]),
                "execution_authorized": False, "planning_only": True})
            continue
        if projection.get("physical_motion_unresolved") is True:
            results.append({"candidate_index": candidate.get("candidate_index", 0),
                "event_id": candidate.get("event_id"), "is_safe": False, "validation_status": "NEEDS_CONTEXT",
                "physical_motion_unresolved": True, "formal_safety_status": "not_checked", "checks": [],
                "monitor_context_before": before, "monitor_context_after": deepcopy(before),
                "candidate_fingerprint": safety.get("candidate_fingerprint"),
                "revision_identity": projection["revision_identity"], "reason": projection["reason"],
                "solver_summary": deepcopy(projection["solver_summary"]),
                "execution_authorized": False, "planning_only": True})
            continue
        if projection.get("status") == "rejected" and projection.get("physical_motion_rejected") is True:
            results.append({"candidate_index": candidate.get("candidate_index", 0),
                "event_id": candidate.get("event_id"), "is_safe": False, "validation_status": "rejected",
                "physical_motion_rejected": True, "formal_safety_status": "not_checked", "checks": [],
                "monitor_context_before": before, "monitor_context_after": deepcopy(before),
                "candidate_fingerprint": safety.get("candidate_fingerprint"),
                "reason": projection.get("reason"), "execution_authorized": False, "planning_only": True})
            if projection.get("continuous_mating_rejected") is True:
                results[-1].update(continuous_mating_rejected=True,
                    mating_rejection=deepcopy(projection["mating_rejection"]))
            if projection.get("physical_endpoint_rejected") is True:
                results[-1].update(physical_endpoint_rejected=True,
                    endpoint_rejection=deepcopy(projection["endpoint_rejection"]))
            continue
        if not isinstance(documents, list):
            checks.append({"is_safe": False, "validation_status": "NEEDS_CONTEXT",
                           "reason": "Selected safety specifications were not captured"})
        for record in documents or []:
            document = record["document"]
            key = record["source_sha256"]
            history = safety.get("monitor_history") or {}
            if history.get("history_error") or (history.get("physical_monitor") or {}).get("invalid_reason"):
                checks.append({"is_safe": False, "validation_status": "NEEDS_CONTEXT",
                               "reason": "Captured CCA monitor history is invalid", "source_path": record["source_path"]})
                continue
            if history.get("active") and key not in before:
                checks.append({"is_safe": False, "validation_status": "NEEDS_CONTEXT",
                               "reason": "Active CCA history requires a compatible recorded outline context; prospective reset is forbidden",
                               "source_path": record["source_path"]})
                continue
            result = validate_outline_evidence_safety(
                projection=deepcopy(safety.get("projection") or {}),
                catalog=deepcopy(document["catalog"]),
                requirement_scopes=deepcopy(document.get("requirement_scopes") or []),
                monitor_context=before.get(key), prospective_activation=key not in before,
            )
            checks.append({**deepcopy(result), "source_path": record["source_path"], "source_sha256": key})
            if result.get("is_safe") is True:
                after[key] = deepcopy(result.get("monitor_context") or result.get("monitor_context_after") or {})
        safe = isinstance(documents, list) and all(row.get("is_safe") is True for row in checks)
        rejected = any(row.get("validation_status") == "INFEASIBLE" for row in checks)
        results.append({"candidate_index": candidate.get("candidate_index", 0),
                        "event_id": candidate.get("event_id"), "is_safe": safe,
                        "validation_status": "passed" if safe and documents else "not_selected" if safe else "rejected" if rejected else "unresolved",
                        "checks": checks, "monitor_context_before": before,
                        "monitor_context_after": after if safe else before,
                        "candidate_fingerprint": safety.get("candidate_fingerprint"),
                        "reason": "; ".join(row.get("reason", "") for row in checks if not row.get("is_safe")),
                        "execution_authorized": False, "planning_only": True})
    return {"validator_jid": validator_jid, "results": results, "execution_authorized": False}
