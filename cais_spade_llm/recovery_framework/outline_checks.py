from __future__ import annotations

"""Requested outline checks inside the existing saved-observation pipeline."""

from copy import deepcopy
from fractions import Fraction
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint as fingerprint,
)

CHECK_FIELDS = ("check_ra_feasibility", "check_physical_evidence", "check_cca_safety")


def selected_checks(session: dict) -> dict[str, bool]:
    """Read explicit choices while preserving the legacy diagnostic flag.

    Args:
        session: Existing recovery_session or multi_turn session values.

    Returns:
        The three requested check booleans.
    """
    default = not session.get("outline_diagnostic_bypass", False)
    result = {field: session.get(field, default) for field in CHECK_FIELDS}
    if any(type(value) is not bool for value in result.values()):
        raise ValueError("Outline check selections must be booleans")
    return result


def state_only_projection(*, task: dict, snapshot: dict, model: dict, physical_input: dict,
                          specifications: list | None, monitor_context: dict) -> dict:
    """Derive process-ledger truth from owner models without spatial calculations.

    Args:
        task: Exact proposed event.
        snapshot: Current owner values.
        model: Recorded owner declarations.
        physical_input: Preserved captured facts and owner evidence.
        specifications: Exact selected safety documents.
        monitor_context: Previously checked CCA history.

    Returns:
        A supported state trace, or NEEDS_CONTEXT when its dependencies are missing.
    """
    from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind
    from cais_spade_llm.recovery_framework.outline_evidence_policy import PROCESS_KINDS
    from cais_spade_llm.resources.environment_recovery import declared_environment_successors

    result = {"status": "needs_context", "state_only": True, "interval_complete": False,
              "unresolved": [], "context_requests": [], "execution_authorized": False}
    kinds = {physical_ap_kind(ap) for record in specifications or []
             for rule in record["document"]["catalog"]["specifications"] for ap in rule["aps"]}
    if not isinstance(specifications, list):
        result["unresolved"] = ["Selected safety specifications were not captured"]
        return result
    if kinds - PROCESS_KINDS:
        result.update(unresolved=["Selected safety predicates need occupancy or owner interval evidence; physical evidence is bypassed"],
                      context_requests=["/physical_evidence"])
        return result
    evidence = physical_input.get("resource_evidence") or {}
    captured = physical_input.get("physical_evidence") or {}
    initial = captured.get("snapshot") or {}
    if not evidence.get("environment_models") or not initial.get("parts"):
        result["unresolved"] = ["CCA needs recorded owner declarations and complete process ledgers"]
        return result
    if any(job.get("job_state") == "running" for row in captured.get("processing_observations", {}).values()
           for job in row.get("jobs", {}).values()):
        result.update(unresolved=["Running machining jobs require supported progression over the proposed event"],
                      context_requests=["/physical_evidence/processing_observations"])
        return result
    if any(row.get("processCompleted_complete") is not True
           or row.get("processCompleted_evidence", {}).get("complete") is not True
           for row in initial["parts"].values()):
        result["unresolved"] = ["CCA process predicates require complete recorded completion ledgers"]
        return result
    try:
        declared, unresolved = declared_environment_successors(task=task, recovery_snapshot=snapshot,
            recovery_des_model=model, physical_input={**physical_input, "outline_evidence_policy_version": 1})
    except (KeyError, TypeError, ValueError) as exc:
        result["unresolved"] = [str(exc)]
        return result
    unique = {fingerprint(row): row for row in declared}
    if len(unique) != 1:
        result["unresolved"] = unresolved or ["CCA needs one supported owner successor for this proposed event"]
        return result
    successor = next(iter(unique.values()))
    before_parts = deepcopy(initial["parts"])
    after_parts = deepcopy(before_parts)
    for name, values in successor["environment_products"].items():
        if name in after_parts:
            after_parts[name].update(deepcopy(values))
            if after_parts[name].get("processCompleted") != before_parts[name].get("processCompleted"):
                after_parts[name]["processCompleted_evidence"] = {
                    "complete": True, "source_kind": "resource_owned_projection",
                    "source_model_fingerprint": successor["source_model_fingerprint"],
                    "source_fingerprint": successor["source_fingerprint"],
                    "checkpoint": fingerprint({"model": successor["source_model_fingerprint"], "task": task}),
                    "task": deepcopy(successor.get("task", task))}
    before_resources = deepcopy(captured.get("environment_valuation") or evidence["environment_valuation"])
    after_resources = deepcopy(successor["environment_valuation"])
    last = max((Fraction(row.get("time_exact", str(row.get("time", 0))))
                for context in monitor_context.values() for row in context.get("observations", [])), default=Fraction(0))
    observations = [{"frame": "world", "resources": values, "parts": parts,
                     "time": float(time), "time_exact": str(time), "phase": "at"}
                    for time, values, parts in ((last, before_resources, before_parts),
                                                (last + 1, after_resources, after_parts))]
    geometry = {"frame": "world", "resources": {}, "regions": {},
        "parts": {name: {"target": row["target"]} if "target" in row else {}
                  for name, row in captured.get("geometry", {}).get("parts", {}).items()}}
    result.update(status="projected", interval_complete=True, observations=observations, geometry=geometry,
        time_basis="owner declaration order; physical duration is not checked",
        source_model_fingerprint=successor["source_model_fingerprint"],
        source_fingerprint=successor["source_fingerprint"],
        environment_valuation=after_resources, environment_products=deepcopy(successor["environment_products"]))
    return result


async def validate_partial_event(*, task: dict, session: dict, prepared: dict,
                                 planner: Any, turn: dict) -> tuple[dict, dict, dict]:
    """Run only requested checks before a diagnostic candidate can advance.

    Args:
        task: Structurally checked authored event.
        session: Private candidate state.
        prepared: Immutable source context and current selections.
        planner: Existing saved ProductAgent validation transport.
        turn: Audit receiving each requested or bypassed stage.

    Returns:
        Per-check statuses, the owner result, and the CCA result.
    """
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import (
        _payload, _validation_resource_input,
    )

    choices = selected_checks(session)
    checks = {name: "not_checked" for name in ("ra", "physical_evidence", "cca")}
    turn["validation_checks"] = choices
    stages = turn.setdefault("validation_stages", [])
    ra, cca = {}, {}
    if not any(choices.values()):
        return checks, ra, cca
    jid = task["resource_jid"]
    resource = _validation_resource_input(prepared, session, jid)
    saved = resource.get("recovery_snapshot") or {}
    model = resource.get("recovery_des_model") or {}
    physical = {"projected_snapshot": deepcopy(session.get("derived_resource_snapshots", {}).get(jid, saved)),
        "base_snapshot_fingerprint": fingerprint(saved), "base_model_fingerprint": fingerprint(model),
        "part_context": deepcopy(session.get("symbolic_parts", {}).get(task.get("part_name"), {})),
        "physical_evidence": deepcopy(session.get("projected_physical_evidence", prepared.get("physical_evidence") or {})),
        "resource_evidence": deepcopy(resource.get("resource_evidence") or {}),
        "parameter_witnesses": deepcopy(resource.get("parameter_witnesses") or []),
        "outline_safety_specifications": deepcopy(prepared.get("outline_safety_specifications"))}
    state_evidence = session.get("projected_state_evidence") or {}
    if state_evidence:
        physical["physical_evidence"].update(
            environment_valuation=deepcopy(state_evidence["environment_valuation"]),
            products=deepcopy(state_evidence["environment_products"]))
        physical["physical_evidence"].setdefault("snapshot", {})["parts"] = deepcopy(state_evidence["observations"][-1]["parts"])
    payload = {**_payload(session, task), **choices}
    if choices["check_ra_feasibility"] or choices["check_physical_evidence"]:
        request = {**payload, "candidates": [{"candidate_index": 0, "event_id": task["outline_id"],
                                             "task": deepcopy(task), "physical_input": physical}]}
        response = await planner.product_agent.request_recovery_outline_physical_validation(resource_jid=jid, payload=request)
        rows = response.get("results") or []
        if response.get("validator_jid") != jid or len(rows) != 1 or rows[0].get("candidate_fingerprint") != fingerprint(task):
            raise ValueError("NEEDS_CONTEXT: owner reply does not identify this proposed event")
        ra = deepcopy(rows[0])
        if choices["check_ra_feasibility"]:
            checks["ra"] = "passed" if ra.get("allowed") is True else "NEEDS_CONTEXT" if ra.get("feasibility_status") == "NEEDS_CONTEXT" else "rejected"
            stages.append({"validator_role": "ResourceAgent", "stage": "ra_feasibility", "status": checks["ra"], "evidence": deepcopy(ra)})
            if checks["ra"] != "passed":
                return checks, ra, cca
        if choices["check_physical_evidence"]:
            projection = ra.get("geometry_evidence") or {}
            checks["physical_evidence"] = ("passed" if projection.get("status") == "projected" else
                                           "rejected" if projection.get("status") == "rejected" else "NEEDS_CONTEXT")
            stages.append({"validator_role": "ResourceAgent", "stage": "physical_evidence", "status": checks["physical_evidence"], "evidence": deepcopy(projection)})
            if checks["physical_evidence"] != "passed":
                return checks, ra, cca
    if choices["check_cca_safety"]:
        if choices["check_physical_evidence"]:
            projection = deepcopy(ra["geometry_evidence"])
        else:
            projection = state_only_projection(task=task, snapshot=physical["projected_snapshot"], model=model,
                physical_input=physical, specifications=prepared.get("outline_safety_specifications"),
                monitor_context=session.get("outline_monitor_context") or {})
        source = {"task": deepcopy(task), "snapshot": deepcopy(physical["projected_snapshot"]),
                  "model": deepcopy(model), "physical_input": deepcopy(physical)}
        safety = {"task": deepcopy(task), "candidate_fingerprint": fingerprint(task), "projection": projection,
            "specifications": deepcopy(prepared.get("outline_safety_specifications")),
            "monitor_history": deepcopy(prepared.get("monitor_history") or {}),
            "monitor_context": deepcopy(session.get("outline_monitor_context") or {}), "evidence_input": source}
        response = await planner.product_agent.request_recovery_outline_safety_validation(payload={**payload,
            "candidates": [{"candidate_index": 0, "event_id": task["outline_id"], "safety_input": safety}]})
        rows = response.get("results") or []
        if len(rows) != 1 or rows[0].get("candidate_fingerprint") != fingerprint(task):
            raise ValueError("NEEDS_CONTEXT: CCA reply does not identify this proposed event")
        cca = deepcopy(rows[0])
        if projection.get("state_only") is True and cca.get("is_safe") is True:
            cca["state_only_projection"] = deepcopy(projection)
        checks["cca"] = ("not_selected" if cca.get("validation_status") == "not_selected" else
                         "passed" if cca.get("is_safe") is True else "rejected"
                         if cca.get("validation_status") in {"rejected", "INFEASIBLE"} else "NEEDS_CONTEXT")
        stages.append({"validator_role": "CentralControllerAgent", "stage": "cca_safety", "status": checks["cca"], "evidence": deepcopy(cca)})
    return checks, ra, cca


async def validate_partial_handoff(*, session: dict, prepared: dict, planner: Any) -> tuple[list[dict], list[dict]]:
    """Check recorded nominal entry events under the same locked selections.

    Args:
        session: Proposed recovery successor, still private until checks pass.
        prepared: Saved nominal entry obligations.
        planner: Existing validation transport.

    Returns:
        Entry check audits and any findings that prevent the proposed handoff.
    """
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_diagnostic import nominal_rows

    if not any(selected_checks(session).values()):
        return [], []
    audits, findings = [], []
    for row in nominal_rows(session, prepared):
        nominal, event = row["task"], row["ra_event"]
        params = nominal.get("params", {})
        start, end = {}, {}
        transition = event.get("capability_transition") or {}
        start.update(deepcopy(transition.get("source") or {}))
        end.update(deepcopy(transition.get("target") or {}))
        for target, declarations in ((start, event.get("guards", {})), (end, event.get("updates", {}))):
            for field, condition in declarations.items():
                if field.startswith(("task_ctx.", "_")):
                    continue
                if not isinstance(condition, dict):
                    target[field] = deepcopy(condition)
                elif "equals" in condition:
                    target[field] = deepcopy(condition["equals"])
                elif "set" in condition:
                    target[field] = deepcopy(condition["set"])
                elif "set_from_param" in condition and condition["set_from_param"] in params:
                    target[field] = deepcopy(params[condition["set_from_param"]])
        task = {"outline_id": row["event_id"], "event_name": nominal["function_name"],
            "resource_jid": nominal["resource_jid"], "part_name": nominal.get("part_name"),
            "expected_start_state": start, "expected_end_state": end,
            "required_task_contract": {"task_id": nominal["task_id"],
                "function_name": nominal["function_name"], "params": deepcopy(params)}}
        audit = {"task": task}
        try:
            checks, ra, cca = await validate_partial_event(task=task, session=deepcopy(session), prepared=prepared,
                                                          planner=planner, turn=audit)
        except (KeyError, TypeError, ValueError) as exc:
            findings.append({"constraint_owner": "resource", "constraint_code": "nominal_entry_unavailable",
                             "reason": str(exc), "validation_status": "NEEDS_CONTEXT"})
            audits.append(audit)
            continue
        audit["checks"] = checks
        audits.append(audit)
        for name, status in checks.items():
            if status not in {"passed", "not_checked", "not_selected"}:
                findings.append({"constraint_owner": "cca" if name == "cca" else "resource",
                    "constraint_code": "nominal_entry_unavailable", "validation_status": status,
                    "reason": cca.get("reason") or ra.get("reason") or
                              "; ".join(ra.get("geometry_evidence", {}).get("unresolved", [])) or
                              "The enabled check did not establish the recorded nominal entry"})
    return audits, findings
