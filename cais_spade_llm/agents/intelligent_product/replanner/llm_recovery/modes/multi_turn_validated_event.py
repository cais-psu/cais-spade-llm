from __future__ import annotations

"""LLM events followed by independent RA and CCA validation."""

import asyncio
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.recovery_framework.outline_validation import OUTLINE_VALIDATION_CONTRACT_VERSION


def _payload(session: dict, task: dict) -> dict:
    payload = {"validation_contract_version": OUTLINE_VALIDATION_CONTRACT_VERSION,
            "recovery_session_id": session.get("session_id", ""),
            "turn_index": session.get("turn_index", 0),
            "state_fingerprint": recovery_validation_fingerprint({
                "resources": session.get("symbolic_resources"), "parts": session.get("symbolic_parts"),
                "evidence": session.get("projected_physical_evidence"),
                "validation_resource_context": session.get("validation_resource_context"),
                "primitive_support_semantics": session.get("primitive_support_semantics"), "task": task})}
    if "primitive_support_semantics" in session:
        payload["primitive_support_semantics"] = session["primitive_support_semantics"]
    if "outline_evidence_policy_version" in session:
        payload["outline_evidence_policy_version"] = session["outline_evidence_policy_version"]
        payload["state_fingerprint"] = recovery_validation_fingerprint({
            "legacy_state": payload["state_fingerprint"],
            "outline_evidence_policy_version": session["outline_evidence_policy_version"],
            "specifications": session.get("outline_evidence_specifications_fingerprint")})
    return payload


def _finding(reason: str, *, missing: bool = False, owner: str = "resource") -> dict:
    return {"constraint_owner": owner,
            "constraint_code": ("resource_validation_unavailable" if owner == "resource" else "safety_validation_unavailable")
            if missing else "transition_feasibility_rejected" if owner == "resource" else "safety_rule_violation",
            "reason": reason}


def _allow_incomplete_motion_revision(session: dict, identities: list[str]) -> bool:
    """Allow new feedback once per supplied motion conditions and evidence."""
    if not identities or any(not isinstance(identity, str) or not identity for identity in identities):
        return False
    seen = session.setdefault("incomplete_motion_revision_identities", [])
    if all(identity in seen for identity in identities):
        return False
    seen.extend(identity for identity in identities if identity not in seen)
    return True


def _missing_candidate_fields(task: dict, verdict: dict) -> list[str]:
    """Expose only unbound public arguments identified by resource validation."""
    if verdict.get("allowed") is not False or verdict.get("feasibility_status") != "NEEDS_CONTEXT":
        return []
    fields = set()
    for missing in verdict.get("missing_parameters") or []:
        for expression in (missing.get("unresolved_parameters") or {}).values():
            if expression == {"$arg": "part_name"} and task.get("part_name") is None:
                fields.add("part_name")
    return sorted(fields)


def _physical_fact_fingerprint(evidence: dict | None) -> str:
    """Compare captured facts for retrieval progress, preserving original records."""
    evidence = evidence or {}
    snapshot = deepcopy(evidence.get("snapshot") or {})
    for part in snapshot.get("parts", {}).values():
        completion = part.get("processCompleted_evidence")
        if isinstance(completion, dict):
            part["processCompleted_evidence"] = {
                key: value for key, value in completion.items()
                if key not in {"checkpoint", "source_kind", "captured_at", "provenance"}}
    geometry = deepcopy(evidence.get("geometry") or {})
    for shapes in (geometry.get("resources", {}), geometry.get("parts", {})):
        for shape in shapes.values():
            source = shape.get("source")
            if isinstance(source, dict):
                shape["source"] = {key: value for key, value in source.items()
                                   if key not in {"simulation_time", "simulation_stamp", "observed_monotonic", "captured_at"}}
    identities = {rid: {key: deepcopy(row[key]) for key in (
        "launch_id", "controller_identity", "attachment", "frame", "custody_complete", "model_static") if key in row}
        for rid, row in evidence.get("resources", {}).items()}
    return recovery_validation_fingerprint({
        "snapshot": snapshot, "geometry": geometry, "resource_models": evidence.get("resource_models"),
        "jids": evidence.get("jids"), "resources": identities, "unresolved": evidence.get("unresolved"),
        "event_evidence": evidence.get("event_evidence"),
        "rejected_event_evidence": evidence.get("rejected_event_evidence"),
        "rejected_capability_evidence": evidence.get("rejected_capability_evidence"),
        "incomplete_event_evidence": evidence.get("incomplete_event_evidence"),
        "processing_observations": evidence.get("processing_observations"),
        "processing_capture_compatibility": evidence.get("processing_capture_compatibility"),
        "recorded_geometry_supplements": evidence.get("recorded_geometry_supplements"),
        "invalidated_geometry": evidence.get("invalidated_geometry"),
        "unresolved_processing": evidence.get("unresolved_processing"),
    })


def _validation_resource_input(prepared: dict, session: dict, jid: str) -> dict:
    """Read the frozen resource facts and separately retained validator evidence."""
    recorded = prepared.get("recovery_resources", {}).get(jid, {})
    retrieved = session.get("validation_resource_context", {}).get(jid, {})
    return {field: deepcopy(retrieved[field] if field in retrieved else recorded.get(field))
            for field in ("recovery_des_model", "recovery_snapshot", "parameter_witnesses",
                          "resource_evidence", "outline_event_evidence", "static_capabilities")}


def _validation_fact_fingerprint(prepared: dict, session: dict) -> str:
    if session.get("primitive_support_semantics") != "successor_primitive_union":
        return _physical_fact_fingerprint(session.get("projected_physical_evidence"))
    resources = {jid: _validation_resource_input(prepared, session, jid)
                 for jid in prepared.get("recovery_resources", {})}
    return recovery_validation_fingerprint({
        "resources": resources,
        "physical_evidence": _physical_fact_fingerprint(session.get("projected_physical_evidence")),
    })


async def _retrieve(prepared: dict, session: dict, planner: Any, requests: list, turn: dict) -> bool:
    """Retrieve supported validator references into the next validation inputs."""
    from .multi_turn_context import check_context_revision

    if session.get("primitive_support_semantics") != "successor_primitive_union":
        return await _retrieve_legacy(prepared, session, planner, requests, turn)
    check_context_revision(prepared, planner)
    before = _validation_fact_fingerprint(prepared, session)
    row = {"context_requests": deepcopy(requests), "returned_facts": {}, "missing_facts": []}
    turn.setdefault("validation_context_exchanges", []).append(row)
    supported = {f"/resources/{jid}/{field}": (jid, field)
                 for jid in prepared.get("recovery_resources", {})
                 for field in ("recovery_des_model", "recovery_snapshot", "parameter_witnesses",
                               "resource_evidence", "outline_event_evidence", "static_capabilities")}
    for ref in dict.fromkeys(ref for ref in requests if isinstance(ref, str)):
        if ref == "/physical_evidence":
            continue
        if ref.startswith("/physical_evidence/"):
            path = ref.removeprefix("/physical_evidence/").split("/")
            allowed = {"processing_observations", "recorded_geometry_supplements"}
            if path[0] not in allowed or len(path) not in (1, 2) or any(not key for key in path):
                row["missing_facts"].append({"ref": ref, "reason": "Unsupported validator context reference"})
                continue
            physical = session.setdefault("projected_physical_evidence", {})
            predicted = bool(session.get("accepted_outline_prefix") or physical.get("provenance", {}).get("kind") == "projected")
            if path[0] not in physical and not predicted:
                saved = prepared.get("physical_evidence") or {}
                if path[0] in saved:
                    physical[path[0]] = deepcopy(saved[path[0]])
            value = physical
            try:
                for key in path:
                    value = value[key]
            except (KeyError, TypeError):
                row["missing_facts"].append({"ref": ref, "reason": "Required recorded evidence is unavailable"})
                continue
            row["returned_facts"][ref] = {
                "source": "projected" if predicted else "saved_observations",
                "revision": prepared.get("observation_revision"),
                "captured_at": prepared.get("source_observation_captured_at"),
                "fingerprint": recovery_validation_fingerprint(value), "value": deepcopy(value)}
            continue
        nested = next((base for base in supported if base.endswith(("/resource_evidence", "/static_capabilities"))
                       and ref.startswith(base + "/")), None)
        if ref not in supported and nested is None:
            row["missing_facts"].append({"ref": ref, "reason": "Unsupported validator context reference"})
            continue
        jid, field = supported[nested or ref]
        recorded = prepared["recovery_resources"][jid]
        value = recorded.get(field)
        if nested is not None:
            path = ref[len(nested) + 1:].split("/")
            allowed_roots = ({"gripper_reach"} if field == "static_capabilities" else
                {"snapshot", "state", "controller_config", "gripper_config", "motion_config",
                 "attachment", "inputs", "source", "sources", "conflicts", "resource_type", "frame",
                 "controller_identity", "custody_complete", "execution_mode", "destination_geometry", "step_outputs",
                 "observed_parts", "pick_geometry", "pick_policy", "tcp_observation"})
            if path[0] not in allowed_roots:
                row["missing_facts"].append({"ref": ref, "reason": "Unsupported resource evidence field"})
                continue
            full_value = deepcopy(value)
            try:
                for key in path:
                    value = value[key]
            except (KeyError, TypeError):
                row["missing_facts"].append({"ref": ref, "reason": "Required recorded evidence is unavailable"})
                continue
        if value is None:
            if nested is None and field not in recorded:
                row["missing_facts"].append({"ref": ref, "reason": "Required recorded evidence is unavailable"})
                continue
        session.setdefault("validation_resource_context", {}).setdefault(jid, {})[field] = deepcopy(full_value if nested else value)
        row["returned_facts"][ref] = {
            "source": "saved_observations", "revision": prepared.get("observation_revision"),
            "captured_at": prepared.get("source_observation_captured_at"),
            "fingerprint": recovery_validation_fingerprint(value), "value": deepcopy(value),
        }
    if "/physical_evidence" in requests:
        # Retrieval cannot replace a predicted successor with an older capture.
        saved = prepared.get("physical_evidence") or {}
        projected = session.get("projected_physical_evidence") or {}
        if not session.get("accepted_outline_prefix") and projected.get("provenance", {}).get("kind") != "projected" and saved:
            session["projected_physical_evidence"] = deepcopy(saved)
            row["returned_facts"]["/physical_evidence"] = {"source": "saved_observations", "value": deepcopy(saved)}
        else:
            row["missing_facts"].append({"ref": "/physical_evidence",
                "reason": ("Additional supported owner bounds or facts are required; trajectory fallback is disabled"
                    if saved.get("endpoint_evidence_policy_version") in (1, 2) else
                    "Matching projected evidence is unavailable; an explicit Gazebo preview may supply it")})
    check_context_revision(prepared, planner)
    return before != _validation_fact_fingerprint(prepared, session)


async def _retrieve_legacy(prepared: dict, session: dict, planner: Any, requests: list, turn: dict) -> bool:
    """Retrieve validator facts without mixing fresh measurements into predictions."""
    from .multi_turn_context import check_context_revision

    check_context_revision(prepared, planner)
    row = {"context_requests": deepcopy(requests), "returned_facts": {}, "missing_facts": []}
    turn.setdefault("validation_context_exchanges", []).append(row)
    before = _physical_fact_fingerprint(session.get("projected_physical_evidence"))
    for jid, resource in prepared.get("recovery_resources", {}).items():
        for field in ("recovery_des_model", "recovery_snapshot"):
            ref = f"/resources/{jid}/{field}"
            if ref in requests and resource.get(field):
                value = (session.get("derived_resource_snapshots", {}).get(jid, resource[field])
                         if field == "recovery_snapshot" else resource[field])
                row["returned_facts"][ref] = {"source": "saved_observations",
                    "revision": prepared.get("observation_revision"),
                    "fingerprint": recovery_validation_fingerprint(value), "value": deepcopy(value)}
    if "/physical_evidence" in requests:
        projected = (session.get("accepted_outline_prefix") or session.get("derived_resource_snapshots")
                     or session.get("projected_physical_evidence", {}).get("provenance", {}).get("kind") == "projected")
        if not projected:
            saved = prepared.get("physical_evidence") or {}
            if saved:
                session["projected_physical_evidence"] = deepcopy(saved)
                row["returned_facts"]["/physical_evidence"] = {"source": "saved_observations", "value": deepcopy(saved)}
            if (not prepared.get("offline_recovery")
                    and before == _physical_fact_fingerprint(session.get("projected_physical_evidence"))
                    and session.get("observation_count", 0) < session.get("max_observations", 0)):
                runtime = getattr(planner.product_agent, "environment_runtime", None)
                if runtime is not None:
                    from cais_spade_llm.recovery_framework.recovery_observations import (
                        capture_recovery_physical_evidence,
                    )

                    saved = await asyncio.to_thread(capture_recovery_physical_evidence, runtime)
                    check_context_revision(prepared, planner)
                    session["observation_count"] = session.get("observation_count", 0) + 1
                    session["projected_physical_evidence"] = deepcopy(saved)
                    row["returned_facts"]["/physical_evidence"] = {"source": "resource_observation", "value": deepcopy(saved)}
        else:
            row["missing_facts"].append({"ref": "/physical_evidence", "reason":
                "Fresh initial observations cannot establish a projected state after accepted events"})
    for ref in requests:
        if ref not in row["returned_facts"]:
            row["missing_facts"].append({"ref": ref, "reason": "Required evidence is unavailable"})
    return before != _physical_fact_fingerprint(session.get("projected_physical_evidence"))


async def validate_event(*, task: dict, session: dict, prepared: dict, planner: Any, turn: dict) -> tuple[dict, dict]:
    """Ask the assigned RA, then CCA after physical validation can proceed."""
    from .multi_turn_context import check_context_revision

    jid = task["resource_jid"]
    if "projected_physical_evidence" not in session:
        session["projected_physical_evidence"] = deepcopy(prepared.get("physical_evidence") or {})
    if session.get("outline_evidence_policy_version") in (1, 2):
        session["outline_evidence_specifications_fingerprint"] = recovery_validation_fingerprint(
            prepared.get("outline_safety_specifications"))
    seen = set()
    while True:
        check_context_revision(prepared, planner)
        resource = _validation_resource_input(prepared, session, jid)
        saved = resource.get("recovery_snapshot") or {}
        physical = {
            "projected_snapshot": deepcopy(session.get("derived_resource_snapshots", {}).get(jid, saved)),
            "base_snapshot_fingerprint": recovery_validation_fingerprint(saved) if saved else None,
            "base_model_fingerprint": recovery_validation_fingerprint(resource.get("recovery_des_model"))
            if resource.get("recovery_des_model") else None,
            "part_context": deepcopy(session.get("symbolic_parts", {}).get(task.get("part_name"), {})),
            "physical_evidence": deepcopy(session["projected_physical_evidence"]),
        }
        if session.get("primitive_support_semantics") == "successor_primitive_union":
            physical["parameter_witnesses"] = deepcopy(resource.get("parameter_witnesses") or [])
            physical["resource_evidence"] = deepcopy(resource.get("resource_evidence") or {})
            if resource.get("outline_event_evidence") is not None:
                physical["outline_event_evidence"] = deepcopy(resource["outline_event_evidence"])
        if session.get("outline_evidence_policy_version") in (1, 2):
            from cais_spade_llm.recovery_framework.outline_evidence_policy import sanitize_validation_input

            physical["outline_evidence_policy_version"] = session["outline_evidence_policy_version"]
            physical["outline_safety_specifications"] = deepcopy(prepared.get("outline_safety_specifications"))
            sanitize_validation_input(physical, jid)
        request = {**_payload(session, task), "candidates": [{"candidate_index": 0,
                   "event_id": task["outline_id"], "task": deepcopy(task), "physical_input": physical}]}
        response = await planner.product_agent.request_recovery_outline_physical_validation(resource_jid=jid, payload=request)
        check_context_revision(prepared, planner)
        rows = response.get("results") or []
        if not rows and response.get("reason"):
            raise ValueError(f"NEEDS_CONTEXT: RA validation: {response['reason']}")
        if response.get("validator_jid") != jid or len(rows) != 1:
            raise ValueError("NEEDS_CONTEXT: RA validation identity or result count differs")
        ra = deepcopy(rows[0])
        validation_model = deepcopy(response.get("recovery_des_model") or resource.get("recovery_des_model") or {})
        if ra.get("additional_binding_record"):
            record = ra["additional_binding_record"]
            calculated = ra.get("calculation_model") or {}
            calculated_input = ra.get("calculation_physical_input") or {}
            from cais_spade_llm.recovery_framework.kmr_calculation import validate_calculation_records

            validate_calculation_records(calculated_input.get("physical_evidence") or {},
                session_id=session.get("session_id", ""), inputs=(resource.get("resource_evidence") or {}).get("inputs", {}))
            if (record.get("task") != task or record.get("source_model_fingerprint") != recovery_validation_fingerprint(validation_model)
                    or record.get("snapshot_fingerprint") != recovery_validation_fingerprint(physical["projected_snapshot"])
                    or record.get("calculation_model_fingerprint") != recovery_validation_fingerprint(calculated)):
                raise ValueError("NEEDS_CONTEXT: owner additional binding calculation differs from the proposal")
            validation_model, physical = deepcopy(calculated), deepcopy(calculated_input)
        if ra.get("candidate_fingerprint") != recovery_validation_fingerprint(task):
            raise ValueError("NEEDS_CONTEXT: RA reply belongs to another proposed event")
        semantics = session.get("primitive_support_semantics")
        if semantics is not None and (ra.get("primitive_support_semantics") != semantics or
                (ra.get("primitive_support") and ra["primitive_support"].get("semantics") != semantics)):
            raise ValueError("NEEDS_CONTEXT: RA reply uses different primitive_support_semantics")
        turn.setdefault("validation_stages", []).append({"validator_role": "ResourceAgent",
            "validator_jid": jid, "state_fingerprint": request["state_fingerprint"], "evidence": deepcopy(ra)})
        if ra.get("allowed") is not True:
            requests = ra.get("context_requests") or []
            signature = recovery_validation_fingerprint({"requests": requests,
                "evidence": _validation_fact_fingerprint(prepared, session)})
            if (ra.get("feasibility_status") == "NEEDS_CONTEXT" and requests and signature not in seen
                    and len(seen) < int(session.get("max_turns") or 16)):
                seen.add(signature)
                if await _retrieve(prepared, session, planner, requests, turn):
                    continue
            return ra, {}
        projection = ra.get("geometry_evidence") or {}
        physical_incomplete = projection.get("status") == "needs_context"
        if (physical_incomplete or projection.get("physical_motion_unresolved") is True or
                projection.get("status") == "rejected" and any(projection.get(flag) is True for flag in (
                    "owner_effect_rejected", "physical_capability_rejected", "physical_motion_rejected"))):
            # Preserve the transaction result without sending a resource failure
            # to CCA or representing it as a selected safety-rule violation.
            before = deepcopy(session.get("outline_monitor_context") or {})
            verdict = {"candidate_index": 0, "event_id": task["outline_id"], "is_safe": False,
                "validation_status": "NEEDS_CONTEXT" if physical_incomplete or projection.get("physical_motion_unresolved") else "rejected",
                "formal_safety_status": "not_checked", "constraint_owner": "resource", "checks": [],
                "reason": projection.get("reason") or "; ".join(projection.get("unresolved") or []),
                "monitor_context_before": before, "monitor_context_after": deepcopy(before),
                "candidate_fingerprint": recovery_validation_fingerprint(task),
                "execution_authorized": False, "planning_only": True}
            verdict.update({key: deepcopy(projection[key]) for key in (
                "owner_effect_rejected", "owner_effect_result", "physical_capability_rejected",
                "capability_result", "capability_source", "physical_motion_unresolved", "revision_identity",
                "solver_summary", "physical_motion_rejected", "continuous_mating_rejected", "mating_rejection",
                "physical_endpoint_rejected", "endpoint_rejection") if key in projection})
            turn["validation_stages"].append({"validator_role": "CentralControllerAgent",
                "status": "not_checked", "request_sent": False, "state_fingerprint": request["state_fingerprint"],
                "evidence": deepcopy(verdict)})
            return ra, verdict
        safety_input = {"task": deepcopy(task), "candidate_fingerprint": recovery_validation_fingerprint(task),
                        "projection": deepcopy(projection),
                        "specifications": deepcopy(prepared.get("outline_safety_specifications")),
                        "monitor_history": deepcopy(prepared.get("monitor_history") or {}),
                        "monitor_context": deepcopy(session.get("outline_monitor_context") or {})}
        if session.get("outline_evidence_policy_version") in (1, 2):
            safety_input["evidence_input"] = {"task": deepcopy(task),
                "snapshot": deepcopy(physical["projected_snapshot"]), "model": validation_model,
                "physical_input": deepcopy(physical)}
        safety_request = {**_payload(session, task), "candidates": [{"candidate_index": 0,
                          "event_id": task["outline_id"], "safety_input": safety_input}]}
        response = await planner.product_agent.request_recovery_outline_safety_validation(payload=safety_request)
        check_context_revision(prepared, planner)
        rows = response.get("results") or []
        if not rows and response.get("reason"):
            raise ValueError(f"NEEDS_CONTEXT: CCA validation: {response['reason']}")
        if len(rows) != 1 or rows[0].get("candidate_fingerprint") != recovery_validation_fingerprint(task):
            raise ValueError("NEEDS_CONTEXT: CCA reply belongs to another proposed event")
        cca = deepcopy(rows[0])
        turn["validation_stages"].append({"validator_role": "CentralControllerAgent",
            "validator_jid": response.get("validator_jid"), "state_fingerprint": safety_request["state_fingerprint"],
            "evidence": deepcopy(cca)})
        if (cca.get("is_safe") is True or cca.get("validation_status") in {"rejected", "INFEASIBLE"}
                or cca.get("physical_motion_unresolved") is True):
            return ra, cca
        requests = list(projection.get("context_requests") or [])
        requests.extend(ref for check in cca.get("checks", []) for ref in check.get("context_requests", []))
        signature = recovery_validation_fingerprint({"requests": requests,
            "evidence": _validation_fact_fingerprint(prepared, session)})
        if not requests or signature in seen or len(seen) >= int(session.get("max_turns") or 16):
            return ra, cca
        seen.add(signature)
        if not await _retrieve(prepared, session, planner, requests, turn):
            return ra, cca


def _project(task: dict, ra: dict, cca: dict, session: dict) -> None:
    geometry = ra.get("geometry_evidence") or {}
    successor = deepcopy(geometry.get("successor_evidence") or {})
    if geometry.get("status") != "projected" or not successor:
        raise ValueError("NEEDS_CONTEXT: accepted transition requires its projected physical successor")
    support = ra["primitive_support"]
    independent = support.get("semantics") == "successor_primitive_union"
    snapshot = (geometry.get("projected_resource_snapshot") if independent else
                support.get("projected_snapshot") or support.get("derived_snapshot"))
    if not isinstance(snapshot, dict):
        raise ValueError("NEEDS_CONTEXT: RA reply has no derived resource snapshot")
    jid, part = task["resource_jid"], task.get("part_name")
    resource = session.setdefault("symbolic_resources", {}).setdefault(jid, {})
    resource.update(deepcopy(snapshot))
    if independent:
        for rid, values in geometry.get("resource_valuation_updates", {}).items():
            owner_jid = successor.get("jids", {}).get(rid)
            if owner_jid is None or not values:
                continue
            session.setdefault("symbolic_resources", {}).setdefault(owner_jid, {}).update(deepcopy(values))
            session.setdefault("derived_resource_snapshots", {}).setdefault(owner_jid, {}).update(deepcopy(values))
    if part:
        part_row = session.setdefault("symbolic_parts", {}).setdefault(part, {})
        part_row.update(deepcopy(geometry.get("part_updates") if independent else
                                support.get("projected_part_context") or {}))
    # Missing derived facts must not fall back to old aliases in captured rows.
    aliases = {"resource_state": ("current_state",), "resource_location": ("current_location",),
               "part_state": ("current_state", "state"), "part_location": ("current_location", "location"),
               "part_holder_resource_jid": ("current_holder_resource_jid", "holder_resource_jid")}
    for scope, fields in support.get("terminal_state", {}).get("unknown", {}).items():
        target = resource if scope == "resource" else session.get("symbolic_parts", {}).get(part, {})
        for field in fields:
            for key in (field, *aliases.get(field, ())):
                target.pop(key, None)
    if independent:
        obligations = session.setdefault("outline_projection_obligations", {})
        obligations[task["outline_id"]] = deepcopy(geometry.get("deferred_valuation_fields") or [])
        for field in geometry.get("deferred_valuation_fields") or []:
            # A proposed product condition is still an obligation. In particular,
            # releasing custody cannot preserve the old in_gripper symbolic state
            # or turn the proposed ready value into a proven effect.
            if field in (geometry.get("deferred_part_fields") or []):
                target = session.get("symbolic_parts", {}).get(part, {})
                for key in (field, *aliases.get(field, ())):
                    target.pop(key, None)
    previous = session.get("projected_physical_evidence", {}).get("snapshot", {})
    for rid, row in successor.get("snapshot", {}).get("resources", {}).items():
        resource_jid = successor.get("jids", {}).get(rid)
        target = session.get("symbolic_resources", {}).get(resource_jid)
        if target is None or row.get("current_pose") is None:
            continue
        if row.get("current_pose") != previous.get("resources", {}).get(rid, {}).get("current_pose"):
            if target.get("current_pose") is not None and target.get("current_pose_source") != "projected_state":
                target["last_observed_pose"] = deepcopy(target["current_pose"])
            target.update(current_pose=deepcopy(row["current_pose"]), projected_pose=deepcopy(row["current_pose"]),
                          current_pose_source="projected_state")
            target.pop("current_pose_captured_at", None)
            if resource_jid == jid:
                snapshot = {**snapshot, "current_pose": deepcopy(row["current_pose"]), "current_pose_source": "projected_state"}
                snapshot.pop("current_pose_captured_at", None)
    session.setdefault("derived_resource_snapshots", {})[jid] = deepcopy(snapshot)
    for name, row in successor.get("snapshot", {}).get("parts", {}).items():
        target = session.get("symbolic_parts", {}).get(name)
        old = previous.get("parts", {}).get(name, {})
        old_holders = {rid for rid, value in previous.get("resources", {}).items() if value.get("held_part") == name}
        new_holders = {rid for rid, value in successor.get("snapshot", {}).get("resources", {}).items()
                       if value.get("held_part") == name}
        changed = row.get("current_pose") != old.get("current_pose") or new_holders != old_holders
        if changed and target is not None and row.get("current_pose") is not None:
            target["projected_pose"] = deepcopy(row["current_pose"])
            target["observed_pose_current"] = False
            if target.get("observed_pose") is not None:
                target["last_observed_pose"] = target.pop("observed_pose")
    session["projected_physical_evidence"] = successor
    if session.get("outline_evidence_policy_version") in (1, 2):
        from cais_spade_llm.recovery_framework.outline_evidence_policy import invalidate_session_facts

        invalidate_session_facts(session, successor)
    session["outline_monitor_context"] = deepcopy(cca.get("monitor_context_after") or {})


async def handle_validated_event(*, session: dict, prepared: dict, response: dict, planner: Any) -> tuple[str, dict]:
    """Commit exactly one authored event after independent agent validation."""
    from cais_spade_llm.recovery_framework.kmr_recovery import _cartesian_joint_calculation_cache

    if _cartesian_joint_calculation_cache.get() is not None:
        return await _handle_validated_event(session=session, prepared=prepared, response=response, planner=planner)
    token = _cartesian_joint_calculation_cache.set({})
    try:
        return await _handle_validated_event(session=session, prepared=prepared, response=response, planner=planner)
    finally:
        _cartesian_joint_calculation_cache.reset(token)


async def _handle_validated_event(*, session: dict, prepared: dict, response: dict, planner: Any) -> tuple[str, dict]:
    from . import multi_turn as shared
    from .multi_turn_context import check_context_revision, handle_outline_context

    check_context_revision(prepared, planner)
    if response.get("decision") == "need_context":
        return await handle_outline_context(prepared=prepared, session=session, response=response, planner=planner)
    event = response.get("next_transition")
    turn = {"next_transition": deepcopy(event), "validation_contract_version": OUTLINE_VALIDATION_CONTRACT_VERSION}
    if "primitive_support_semantics" in session:
        turn["primitive_support_semantics"] = session["primitive_support_semantics"]
    if "outline_evidence_policy_version" in session:
        turn["outline_evidence_policy_version"] = session["outline_evidence_policy_version"]
    error = None
    if (set(response) - {"thought", "next_transition", "decision", "context_requests"}
            or response.get("decision") is not None or response.get("context_requests", []) != []):
        error = "Return one next_transition or need_context without proposed events"
    elif not isinstance(response.get("thought"), str) or not isinstance(event, dict):
        error = "The response requires thought and one next_transition object"
    elif any(row.get("outline_id") == event.get("outline_id") for row in session.get("accepted_outline_prefix", [])):
        error = "outline_id was already accepted"
    if error:
        session["outline_validation_findings"] = [_finding(error)]
        session["status"] = "paused_after_outline_turn"
        return "need_revision", {**turn, "validation_findings": deepcopy(session["outline_validation_findings"])}
    task, findings = shared._derive_candidate_outline_task(candidate_task=event, session_state=session,
                                                          prepared_recovery_request=prepared)
    if findings:
        session["outline_validation_findings"] = findings
        session["status"] = "paused_after_outline_turn"
        return "need_revision", {**turn, "validation_findings": findings}
    working = deepcopy({key: value for key, value in session.items()
        if key not in {"turns", "nominal_continuation_evidence", "pending_outline_validation"}})
    try:
        ra, cca = await validate_event(task=task, session=working, prepared=prepared, planner=planner, turn=turn)
        if not ra.get("allowed") or not cca.get("is_safe"):
            upstream = (cca.get("formal_safety_status") == "not_checked" and (
                cca.get("constraint_owner") == "resource" or any(cca.get(flag) is True for flag in (
                "owner_effect_rejected", "physical_capability_rejected", "physical_motion_rejected",
                "physical_motion_unresolved"))))
            owner = "resource" if not ra.get("allowed") or upstream else "cca"
            verdict = ra if not ra.get("allowed") else cca
            missing = (verdict.get("feasibility_status") == "NEEDS_CONTEXT" or
                       verdict.get("validation_status") in {"unresolved", "NEEDS_CONTEXT", "unavailable"})
            from .multi_turn_prompts import _outline_safety_checks_for_prompt

            finding = _finding(verdict.get("reason") or "Required validation did not pass", missing=missing, owner=owner)
            missing_fields = _missing_candidate_fields(task, ra) if owner == "resource" else []
            if missing_fields:
                finding["evidence"] = {"missing_candidate_fields": missing_fields}
            if owner == "resource" and ra.get("constraint_code") == "unsupported_successor_condition":
                finding["constraint_code"] = ra["constraint_code"]
                finding["evidence"] = {"mismatches": deepcopy(ra.get("evidence", {}).get("mismatches", []))}
                if isinstance(ra.get("evidence", {}).get("condition_support"), dict):
                    finding["evidence"]["condition_support"] = deepcopy(ra["evidence"]["condition_support"])
            if owner == "resource" and isinstance(ra.get("evidence", {}).get("capability_result"), dict):
                finding.setdefault("evidence", {})["capability_result"] = deepcopy(ra["evidence"]["capability_result"])
            if owner == "resource":
                conflicts = []
                for primitive in ra.get("primitive_support", {}).get("primitives", []):
                    for attempt in primitive.get("parameter_attempts", []):
                        witness = attempt.get("parameter_witness") or {}
                        bindings = witness.get("parameter_bindings", {})
                        params = witness.get("params", {})
                        for field in attempt.get("evidence", {}).get("conflicting_parameters", []):
                            if field in bindings and field in params:
                                row = {"field": field, "expected": deepcopy(bindings[field]),
                                       "actual": deepcopy(params[field])}
                                if row not in conflicts:
                                    conflicts.append(row)
                if conflicts:
                    finding.setdefault("evidence", {})["parameter_conflicts"] = conflicts
                projection = ra.get("geometry_evidence", {})
                failures = [reason for row in [projection, *projection.get("owner_alternatives", [])]
                            for reason in row.get("unresolved", [])]
                if "Bounded KMR endpoint search found no independently verified joint witness; reachability is unresolved" in failures:
                    finding.setdefault("evidence", {})["solver_summary"] = {
                        "source": "resource-owned endpoint calculation", "failure_kind": "bounded_search_failed",
                        "source_ref": f"/resources/{task['resource_jid']}/resource_evidence"}
            if verdict.get("physical_motion_rejected") is True:
                finding["constraint_code"] = ("continuous_mating_rejected"
                    if verdict.get("continuous_mating_rejected") is True else "physical_endpoint_rejected"
                    if verdict.get("physical_endpoint_rejected") is True else "physical_motion_rejected")
            if verdict.get("physical_capability_rejected") is True:
                finding["constraint_code"] = "physical_capability_rejected"
                finding["constraint_owner"] = "resource"
            if verdict.get("owner_effect_rejected") is True:
                finding["constraint_code"] = "owner_effect_rejected"
                finding["constraint_owner"] = "resource"
            if verdict.get("physical_motion_unresolved") is True:
                finding["constraint_code"] = "physical_motion_unresolved"
            finding.update({key: deepcopy(task[key]) for key in (
                "outline_id", "event_name", "resource_jid", "part_name", "expected_start_state", "expected_end_state"
            ) if key in task})
            finding.update(feasibility_status=ra.get("feasibility_status"),
                           validation_status=cca.get("validation_status"))
            if cca:
                finding.setdefault("evidence", {})["safety_checks"] = _outline_safety_checks_for_prompt(cca.get("checks") or [])
                if cca.get("continuous_mating_rejected") is True:
                    proof = cca["mating_rejection"]
                    counterexample = proof["counterexample"]
                    finding["evidence"]["mating_counterexample"] = {key: deepcopy(counterexample[key]) for key in
                        ("segment", "time_from_start", "fraction", "violated_conditions") if key in counterexample}
                if cca.get("physical_endpoint_rejected") is True:
                    finding["evidence"]["endpoint_counterexample"] = deepcopy(cca["endpoint_rejection"])
                if cca.get("physical_motion_unresolved") is True:
                    finding["evidence"]["solver_summary"] = deepcopy(cca["solver_summary"])
                if cca.get("physical_capability_rejected") is True:
                    finding["evidence"].update({key: deepcopy(cca[key])
                        for key in ("capability_result", "capability_source")})
                if cca.get("owner_effect_rejected") is True:
                    finding["evidence"]["owner_effect_result"] = deepcopy(cca["owner_effect_result"])
            findings = [finding]
            session["outline_validation_findings"] = findings
            if missing_fields:
                identity = recovery_validation_fingerprint({
                    "resource_jid": task["resource_jid"], "missing_candidate_fields": missing_fields,
                    "expected_start_state": task.get("expected_start_state"),
                    "expected_end_state": task.get("expected_end_state"),
                    "descriptor_fingerprint": ra.get("primitive_support", {}).get("descriptor_fingerprint"),
                    "snapshot_fingerprint": ra.get("primitive_support", {}).get("snapshot_fingerprint")})
                seen = session.setdefault("incomplete_parameter_revision_identities", [])
                if identity not in seen:
                    seen.append(identity)
                    session["status"] = "paused_after_outline_turn"
                    session.pop("pending_outline_validation", None)
                    turn.update(validation_findings=findings,
                        transition_validation={"status": "unresolved", "validation_status": "NEEDS_CONTEXT",
                            "findings": findings, "proposal_revision_available": True, "execution_authorized": False})
                    return "need_revision", turn
            if verdict.get("physical_motion_unresolved") is True:
                identity = verdict["revision_identity"]
                if _allow_incomplete_motion_revision(session, [identity]):
                    session["status"] = "paused_after_outline_turn"
                    session.pop("pending_outline_validation", None)
                    turn.update(validation_findings=findings,
                        transition_validation={"status": "unresolved", "validation_status": "NEEDS_CONTEXT",
                            "findings": findings, "proposal_revision_available": True, "execution_authorized": False})
                    return "need_revision", turn
                finding["reason"] += " The same conditions and evidence have already been returned for revision."
            session["status"] = "NEEDS_CONTEXT" if missing else "paused_after_outline_turn"
            if missing:
                session["pending_outline_validation"] = {"kind": "event", "response": deepcopy(response),
                    "task": deepcopy(task), "source_turn_index": session.get("turn_index", 0)}
            else:
                session.pop("pending_outline_validation", None)
                history = session.setdefault("candidate_rejection_feedback", [])
                if finding not in history:
                    history.append(deepcopy(finding))
            turn.update(validation_findings=findings, transition_validation={"status": "unresolved" if missing else "rejected", "findings": findings})
            return "need_context" if missing else "need_revision", turn
        _project(task, ra, cca, working)
    except ValueError as exc:
        if not str(exc).startswith("NEEDS_CONTEXT:"):
            raise
        findings = [_finding(str(exc), missing=True)]
        session["outline_validation_findings"] = findings
        session["status"] = "NEEDS_CONTEXT"
        session["pending_outline_validation"] = {"kind": "event", "response": deepcopy(response),
            "task": deepcopy(task), "source_turn_index": session.get("turn_index", 0)}
        return "need_context", {**turn, "validation_findings": findings}
    check_context_revision(prepared, planner)
    working.setdefault("accepted_outline_prefix", []).append({**deepcopy(task), "candidate_source": "llm"})
    from .multi_turn_nominal_evidence import record_nominal_task_completions

    record_nominal_task_completions(task, ra, cca, working, prepared)
    working["outline_validation_findings"] = []
    working["candidate_rejection_feedback"] = []
    working["outline_lookahead"] = []
    working["status"] = "paused_after_outline_turn"
    working.pop("pending_outline_validation", None)
    turn["transition_validation"] = {"status": "passed", "findings": [], "execution_authorized": False}
    session.pop("pending_outline_validation", None)
    session.update(working)
    shared._sync_des_recovery_aliases(session, turn_entry=turn, transition_validation=turn["transition_validation"])
    from .multi_turn_nominal_evidence import validate_nominal_continuation, nominal_continuation_findings

    try:
        continuation = await validate_nominal_continuation(session=session, prepared=prepared, planner=planner)
    except ValueError as exc:
        if not str(exc).startswith("NEEDS_CONTEXT:"):
            raise
        continuation = {"complete": False, "status": "NEEDS_CONTEXT", "reason": str(exc)}
    session["nominal_continuation_evidence"] = deepcopy(continuation)
    turn["nominal_continuation_evidence"] = deepcopy(continuation)
    session["outline_validation_findings"] = nominal_continuation_findings(continuation, prepared)
    complete = continuation.get("complete") is True
    if not complete and continuation.get("status") == "NEEDS_CONTEXT":
        if _allow_incomplete_motion_revision(session, continuation.get("proposal_revision_identities") or []):
            session["status"] = "paused_after_outline_turn"
            session.pop("pending_outline_validation", None)
            return "need_revision", turn
        session["status"] = "NEEDS_CONTEXT"
        pending_task = next((row for row in continuation.get("failures", []) if row.get("task")), {})
        session["pending_outline_validation"] = {"kind": "nominal", "source_turn_index": session.get("turn_index", 0),
            **{key: deepcopy(pending_task[key]) for key in ("task", "projected_physical_evidence", "recovery_snapshot") if key in pending_task}}
        return "need_context", turn
    session["status"] = "ready_for_primitive_generation" if complete else "paused_after_outline_turn"
    return "outline_ready" if complete else "need_next_task", turn


def _event_state_fingerprint(session: dict, event: dict, current: str) -> str:
    """Let explicit event arguments distinguish candidates that leave the modeled state unchanged.

    A candidate reaching any other state is identified by that state alone, so
    returning to an earlier state is still a repeat.
    """
    state = _comparison_state_fingerprint(session)
    params = event.get("params")
    return recovery_validation_fingerprint({"state": state, "params": params}) if params and state == current else state


def _declares_params(session: dict, prepared: dict, event: dict) -> bool:
    """Return whether every explicit argument is one the resource's own events declare."""
    jid = event.get("resource_jid")
    model = (session.get("recovery_des_models") or {}).get(jid) or (
        (prepared.get("recovery_resources") or {}).get(jid) or {}).get("recovery_des_model") or {}
    declared = {name for row in model.get("events", []) for name in row.get("parameter_bindings", {})}
    params = event.get("params")
    return isinstance(params, dict) and bool(params) and set(params) <= declared


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


async def handle_validated_candidates(*, session: dict, prepared: dict, response: dict, planner: Any) -> tuple[str, dict]:
    """Validate alternatives independently and commit one progressing supported successor.

    Args:
        session: Accepted planning state, updated only for the selected candidate.
        prepared: Recorded context shared by the candidate comparisons.
        response: LLM-authored candidate_events or a public context request.
        planner: Resource and CCA validation transport.

    Returns:
        Continuation decision and the candidate comparison audit.
    """
    if session.get("outline_diagnostic_bypass"):
        return await _handle_validated_candidates(session=session, prepared=prepared, response=response, planner=planner)

    from cais_spade_llm.recovery_framework.kmr_recovery import _cartesian_joint_calculation_cache

    # Only pure solver calculations are shared within this batch. Every owner
    # rechecks the returned FK, limits and proposal evidence independently.
    token = _cartesian_joint_calculation_cache.set({})
    try:
        return await _handle_validated_candidates(session=session, prepared=prepared, response=response, planner=planner)
    finally:
        _cartesian_joint_calculation_cache.reset(token)


async def _handle_validated_candidates(*, session: dict, prepared: dict, response: dict, planner: Any) -> tuple[str, dict]:
    from . import multi_turn_outline_generation as comparison
    from .multi_turn_context import check_context_revision, handle_outline_context

    check_context_revision(prepared, planner)
    if response.get("decision") == "need_context":
        return await handle_outline_context(prepared=prepared, session=session, response=response, planner=planner)
    events = response.get("candidate_events")
    if (set(response) - {"thought", "candidate_events", "decision", "context_requests"}
            or response.get("decision") is not None or response.get("context_requests", []) != []
            or not isinstance(response.get("thought"), str) or not isinstance(events, list)
            or not 1 <= len(events) <= 5 or any(not isinstance(event, dict) for event in events)
            or len({event.get("outline_id") for event in events if isinstance(event.get("outline_id"), str)}) != len(events)):
        finding = {"constraint_owner": "planner", "constraint_code": "invalid_candidate_response",
                   "reason": "Return thought and one to five candidate_events with distinct outline_id values, or need_context."}
        session.update(status="paused_after_outline_turn", outline_validation_findings=[finding])
        return "need_revision", {"validation_findings": [finding]}
    base = {key: value for key, value in session.items()
            if key not in {"turns", "nominal_continuation_evidence", "pending_outline_validation"}}
    initial = _comparison_state_fingerprint(session)
    visited = set(session.get("accepted_state_fingerprints", [])) | {initial}
    sequences, evaluations, branches = [], [], {}
    diagnostic = session.get("outline_diagnostic_bypass") is True
    if diagnostic:
        from .multi_turn_diagnostic import handle_event
    for index, event in enumerate(events):
        working, context = deepcopy(base), deepcopy(prepared)
        event_handler = handle_event if diagnostic else handle_validated_event
        decision, audit = await event_handler(session=working, prepared=context,
            response={"thought": response["thought"], "next_transition": deepcopy(event)}, planner=planner)
        valid = (audit.get("schema_valid") is True and audit.get("transition_validation", {}).get("status") == "not_checked"
                 if diagnostic else audit.get("transition_validation", {}).get("status") == "passed")
        identity = ((_event_state_fingerprint(working, event, initial) if diagnostic
                     else _comparison_state_fingerprint(working)) if valid else None)
        sequences.append({"candidate_index": index, "surface_events": [deepcopy(event)]})
        evaluation = {"candidate_index": index, "task": deepcopy(event), "valid": valid,
            "explored_params": diagnostic and _declares_params(base, prepared, event),
            "validation_audit": audit, "decision": decision, "repeated_state": valid and identity in visited,
            "projected_symbolic_resources": deepcopy(working.get("symbolic_resources", {})),
            "projected_symbolic_parts": deepcopy(working.get("symbolic_parts", {})),
            "safety_dfa_states_before": deepcopy(session.get("projected_safety_dfa_states", {})),
            "safety_dfa_states_after": deepcopy(working.get("projected_safety_dfa_states", {})),
            "projected_validation_state": working if valid else {},
            "validated_events": [deepcopy(audit["next_transition"])] if diagnostic and valid else [],
            "diagnostic_handoff_complete": diagnostic and working.get("diagnostic_nominal_handoff", {}).get("complete") is True,
            "nominal_continuation_complete": working.get("nominal_continuation_evidence", {}).get("complete") is True}
        evaluations.append(evaluation)
        branches[index] = (working, decision, audit, identity)
    # Comparison rejection is not a change to the independent RA/CCA results.
    eligible = [row for row in evaluations if row["valid"] and not row["repeated_state"]]
    complete = [row for row in eligible if row["diagnostic_handoff_complete"]] if diagnostic else [row for row in eligible if row["nominal_continuation_complete"]]
    selected = None
    if complete:
        selected = min(complete, key=lambda row: comparison._candidate_effect_identifier(sequences[row["candidate_index"]]))
        selected["selection_status"] = "proposed_nominal_handoff" if diagnostic else "nominal_continuation_established"
    elif eligible:
        await comparison._populate_agent_filtered_enabledness(candidate_evaluations=eligible,
            session_state=deepcopy(base), prepared_recovery_request=deepcopy(prepared), planner=planner)
        preferred = comparison._apply_neurosymbolic_comparison(candidate_sequences=sequences,
            candidate_evaluations=eligible, session_state=base, prepared_recovery_request=prepared)
        selected = min((row for row in eligible if row.get("candidate_id") in preferred),
                       key=lambda row: row["candidate_id"], default=None)
    for row in evaluations:
        row.pop("projected_validation_state", None)
        if row["repeated_state"]:
            row["selection_status"] = "excluded_repeated_state"
    audit = {"candidate_events": deepcopy(events), "candidate_evaluations": evaluations,
             "validation_contract_version": OUTLINE_VALIDATION_CONTRACT_VERSION,
             "outline_diagnostic_bypass": diagnostic,
             "execution_authorized": False}
    if selected is None:
        findings = [finding for row in evaluations
                    for finding in row["validation_audit"].get("validation_findings", [])]
        findings.append({"constraint_owner": "planner", "constraint_code": "no_progressing_candidate",
            "reason": "No candidate established a new supported state that reduces unresolved obligations or enables further work."})
        session.update(status="paused_after_outline_turn", outline_validation_findings=findings)
        if diagnostic:
            unresolved = [row for row in evaluations if row["decision"] == "need_context"]
            rejected = [row for row in evaluations if row["validation_audit"].get("checks") and not row["valid"]]
            if unresolved or rejected:
                session["status"] = "NEEDS_CONTEXT" if unresolved else "validation_rejected"
                session["outline_check_results"] = deepcopy((unresolved or rejected)[0]["validation_audit"].get("checks", {}))
                audit["validation_findings"] = findings
                return "need_context" if unresolved else "need_revision", audit
            same = session.get("selection_repeated_failure_fingerprint") == initial
            session["selection_repeated_failure_count"] = session.get("selection_repeated_failure_count", 0) + 1 if same else 1
            session["selection_repeated_failure_fingerprint"] = initial
            if session["selection_repeated_failure_count"] >= session.get("selection_repeated_failure_limit", 3):
                session["status"] = "selection_unresolved"
            audit["validation_findings"] = findings
            return "need_revision", audit
        feedback = {recovery_validation_fingerprint(row): deepcopy(row)
                    for working, _, _, _ in branches.values()
                    for row in working.get("candidate_rejection_feedback", [])}
        session["candidate_rejection_feedback"] = [feedback[key] for key in sorted(feedback)]
        unresolved = [row for row in evaluations if row["decision"] == "need_context"]
        revision_available = False
        if unresolved and not eligible and session.get("outline_evidence_policy_version") == 2:
            identity = recovery_validation_fingerprint(sorted(recovery_validation_fingerprint(
                {"conditions": {key: row["task"].get(key) for key in
                    ("resource_jid", "part_name", "expected_start_state", "expected_end_state")},
                 "findings": [{key: finding.get(key) for key in ("constraint_owner", "constraint_code", "reason")}
                    for finding in row["validation_audit"].get("validation_findings", [])]})
                for row in unresolved))
            seen = session.setdefault("incomplete_candidate_revision_identities", [])
            if identity not in seen:
                seen.append(identity)
                revision_available = True
                # The unresolved verdict remains in the audit and prompt. A
                # different proposal may avoid facts that this one requires.
                feedback.update({recovery_validation_fingerprint(row): deepcopy(row) for row in findings[:-1]})
                session["candidate_rejection_feedback"] = [feedback[key] for key in sorted(feedback)]
        if unresolved and not eligible:
            if revision_available:
                session.pop("pending_outline_validation", None)
            else:
                session["pending_outline_validation"] = {"kind": "candidates", "response": deepcopy(response),
                    "source_turn_index": session.get("turn_index")}
                session["status"] = "NEEDS_CONTEXT"
        else:
            session.pop("pending_outline_validation", None)
        audit["validation_findings"] = findings
        audit["proposal_revision_available"] = revision_available
        return ("need_context" if unresolved and not eligible and not revision_available else "need_revision"), audit
    working, decision, selected_audit, identity = branches[selected["candidate_index"]]
    accepted = [initial, identity]
    params = events[selected["candidate_index"]].get("params")
    if diagnostic and params:
        # Remember the arguments with the state they produced, so the same arguments are a repeat there.
        accepted.append(recovery_validation_fingerprint({"state": _comparison_state_fingerprint(working), "params": params}))
    working["accepted_state_fingerprints"] = [*dict.fromkeys([*session.get("accepted_state_fingerprints", []), *accepted])]
    session.pop("pending_outline_validation", None)
    session.update(working)
    audit.update(selected_audit)
    audit["selected_candidate_index"] = selected["candidate_index"]
    return decision, audit


async def retry_pending_event(*, prepared: dict, session: dict, planner: Any) -> tuple[str, dict]:
    """Retry recorded validation after explicit evidence preparation, without LLM calls."""
    from .multi_turn_context import check_context_revision

    check_context_revision(prepared, planner)
    pending = session.get("pending_outline_validation")
    if not isinstance(pending, dict):
        raise ValueError("No recorded outline proposal is awaiting validation")
    pending = deepcopy({key: value for key, value in pending.items()
                        if key in {"kind", "response", "source_turn_index"}})
    if pending.get("kind") == "event":
        decision, turn = await handle_validated_event(session=session, prepared=prepared,
                                                      response=pending["response"], planner=planner)
    elif pending.get("kind") == "candidates":
        decision, turn = await handle_validated_candidates(session=session, prepared=prepared,
            response=pending["response"], planner=planner)
    elif pending.get("kind") == "nominal":
        from .multi_turn_nominal_evidence import validate_nominal_continuation, nominal_continuation_findings

        result = await validate_nominal_continuation(session=session, prepared=prepared, planner=planner)
        session["nominal_continuation_evidence"] = deepcopy(result)
        session["outline_validation_findings"] = nominal_continuation_findings(result, prepared)
        turn = {"nominal_continuation_evidence": deepcopy(result)}
        if result.get("complete") is True:
            session["status"] = "ready_for_primitive_generation"
            session["outline_validation_findings"] = []
            session.pop("pending_outline_validation", None)
            decision = "outline_ready"
        else:
            if _allow_incomplete_motion_revision(session, result.get("proposal_revision_identities") or []):
                session["status"] = "paused_after_outline_turn"
                session.pop("pending_outline_validation", None)
                turn.update(validation_only=True, source_turn_index=pending.get("source_turn_index"), request_sent=False)
                return "need_revision", turn
            session["status"] = "NEEDS_CONTEXT" if result.get("status") == "NEEDS_CONTEXT" else "paused_after_outline_turn"
            decision = "need_context" if result.get("status") == "NEEDS_CONTEXT" else "need_next_task"
            if result.get("status") == "NEEDS_CONTEXT":
                pending_task = next((row for row in result.get("failures", []) if row.get("task")), {})
                session["pending_outline_validation"] = {"kind": "nominal", "source_turn_index": pending.get("source_turn_index"),
                    **{key: deepcopy(pending_task[key]) for key in ("task", "projected_physical_evidence", "recovery_snapshot") if key in pending_task}}
            else:
                session.pop("pending_outline_validation", None)
    else:
        raise ValueError("Unsupported pending outline validation")
    turn.update(validation_only=True, source_turn_index=pending.get("source_turn_index"), request_sent=False)
    return decision, turn
