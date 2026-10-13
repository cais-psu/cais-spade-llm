from __future__ import annotations

"""One LLM event followed by independent RA and CCA validation."""

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
        # New sessions acquire live supplementary evidence only through the
        # explicit UI preview action. Retrieval itself remains saved-facts-only.
        saved = prepared.get("physical_evidence") or {}
        projected = session.get("projected_physical_evidence") or {}
        if not session.get("accepted_outline_prefix") and projected.get("provenance", {}).get("kind") != "projected" and saved:
            session["projected_physical_evidence"] = deepcopy(saved)
            row["returned_facts"]["/physical_evidence"] = {"source": "saved_observations", "value": deepcopy(saved)}
        else:
            row["missing_facts"].append({"ref": "/physical_evidence",
                "reason": "Matching projected evidence is unavailable; an explicit Gazebo preview may supply it"})
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
    """Ask the assigned RA and CCA without accepting a proposed transition."""
    from .multi_turn_context import check_context_revision

    jid = task["resource_jid"]
    if "projected_physical_evidence" not in session:
        session["projected_physical_evidence"] = deepcopy(prepared.get("physical_evidence") or {})
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
        safety_input = {"task": deepcopy(task), "candidate_fingerprint": recovery_validation_fingerprint(task),
                        "projection": deepcopy(projection),
                        "specifications": deepcopy(prepared.get("outline_safety_specifications")),
                        "monitor_history": deepcopy(prepared.get("monitor_history") or {}),
                        "monitor_context": deepcopy(session.get("outline_monitor_context") or {})}
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
    session["outline_monitor_context"] = deepcopy(cca.get("monitor_context_after") or {})


async def handle_validated_event(*, session: dict, prepared: dict, response: dict, planner: Any) -> tuple[str, dict]:
    """Commit exactly one authored event after independent agent validation."""
    from . import multi_turn as shared
    from .multi_turn_context import check_context_revision, handle_outline_context

    check_context_revision(prepared, planner)
    if response.get("decision") == "need_context":
        return await handle_outline_context(prepared=prepared, session=session, response=response, planner=planner)
    event = response.get("next_transition")
    turn = {"next_transition": deepcopy(event), "validation_contract_version": OUTLINE_VALIDATION_CONTRACT_VERSION}
    if "primitive_support_semantics" in session:
        turn["primitive_support_semantics"] = session["primitive_support_semantics"]
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
            owner = "resource" if not ra.get("allowed") else "cca"
            verdict = ra if owner == "resource" else cca
            missing = (verdict.get("feasibility_status") == "NEEDS_CONTEXT" or
                       verdict.get("validation_status") in {"unresolved", "NEEDS_CONTEXT", "unavailable"})
            from .multi_turn_prompts import _outline_safety_checks_for_prompt

            finding = _finding(verdict.get("reason") or "Required validation did not pass", missing=missing, owner=owner)
            if owner == "resource" and ra.get("constraint_code") == "unsupported_successor_condition":
                finding["constraint_code"] = ra["constraint_code"]
                finding["evidence"] = {"mismatches": deepcopy(ra.get("evidence", {}).get("mismatches", []))}
            if owner == "resource" and isinstance(ra.get("evidence", {}).get("capability_result"), dict):
                finding.setdefault("evidence", {})["capability_result"] = deepcopy(ra["evidence"]["capability_result"])
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
