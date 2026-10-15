from __future__ import annotations

"""Ordered RA composition orchestration for versioned, review-only attempts."""

from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
    compact_multi_turn_runtime_session,
)
from cais_spade_llm.resources.recovery_composition import composition_program_error


async def generate_next_composition_event(*, planner: Any, session_state: dict,
                                          prepared_recovery_request: dict) -> tuple[str, dict, dict]:
    """Generate one event through its owner, carrying the entire accepted prefix."""
    from .multi_turn import _run_resource_primitive_batch

    events = session_state.get("accepted_outline_prefix", [])
    program = session_state.setdefault("accepted_primitive_program", [])
    error = composition_program_error(events, program, prepared=prepared_recovery_request,
                                      session=session_state, complete=False)
    turn = {"accepted_transition_prefix": deepcopy(events), "des_event_sequence": deepcopy(events)}
    result = {}
    if not error and len(program) < len(events):
        event = events[len(program)]
        result = await _run_resource_primitive_batch(planner=planner,
            prepared_recovery_request=prepared_recovery_request, session_state=session_state,
            resource_jid=event["resource_jid"], assigned_outline_events=[event])
        turn["primitive_substream_turns"] = [
            compact_multi_turn_runtime_session({"turns": [row]}, keep_full_turns=0)["turns"][0]
            if row.get("prompt_artifact_path") and row.get("response_artifact_path") else deepcopy(row)
            for row in result.get("turns", [])
        ]
        rows = result.get("primitive_events", [])
        if len(rows) != 1:
            error = "; ".join(row.get("reason", "") for row in result.get("feedback", []))
            error = error or "The event requires exactly one RA-validated program"
        else:
            candidate = [*program, rows[0]]
            error = composition_program_error(events, candidate, prepared=prepared_recovery_request,
                                              session=session_state, complete=False)
            if not error:
                program.append(deepcopy(rows[0]))
    if error:
        feedback = result.get("feedback") or [{"constraint_owner": "resource",
            "constraint_code": "composition_incomplete", "reason": error}]
        session_state.update(status="paused_after_primitive_blocked", primitive_rejection_feedback=deepcopy(feedback))
        turn["primitive_rejection_feedback"] = deepcopy(feedback)
        decision = "primitive_blocked"
    else:
        complete = len(program) == len(events)
        session_state.update(status="paused_after_primitive_generation" if complete else "paused_after_primitive_turn",
            primitive_rejection_feedback=[], primitive_generation_cursor=len(program))
        decision = "draft_ready" if complete else "primitive_steps_ready"
    turn.update(accepted_primitive_macros=deepcopy(program), remaining_outline_events=deepcopy(events[len(program):]),
                composition_validation_version=1, execution_authorized=False)
    # The substream artifacts retain full authoring history. Nesting its session
    # inside the parent response makes each next event copy every earlier batch.
    batch = {key: value for key, value in result.items() if key not in {"session_state", "turns"}}
    return decision, turn, {"resource_batches": [batch]}


def build_composition_review(*, final_output_payload: dict, prepared_recovery_request: dict) -> dict:
    """Adapt only a complete bound composition into a reviewable proposal."""
    events = final_output_payload.get("transition_trace", [])
    program = final_output_payload.get("accepted_primitive_program", [])
    session = prepared_recovery_request.get("multi_turn_session_state", {})
    error = composition_program_error(events, program, prepared=prepared_recovery_request, session=session)
    result = {"accepted": False, "reason": error or "", "raw_proposal": None, "recovery_proposal": None,
              "final_output_stage": final_output_payload.get("final_output_stage")}
    if error:
        return result
    macros, resources, parts = [], {}, {}
    for event, row in zip(events, program):
        evidence = row["composition_validation"]
        macros.append({**deepcopy(event), **deepcopy(row), "macro_name": event["event_name"],
                       "description": event.get("rationale", ""), "execution_authorized": False})
        resources[event["resource_jid"]] = deepcopy(evidence["projected_snapshot"])
        for jid, updates in evidence.get("projected_resources", {}).items():
            resources.setdefault(jid, {}).update(deepcopy(updates))
        if event.get("part_name") is not None:
            parts[event["part_name"]] = deepcopy(evidence["projected_part_context"])
        for name, updates in evidence.get("projected_parts", {}).items():
            parts.setdefault(name, {}).update(deepcopy(updates))
    proposal = {"description": "RA primitive composition ready for review.", "macro_tasks": macros,
        "composition_validation_version": 1, "composition_feasibility": "FEASIBLE",
        "projected_resource_snapshots": resources, "projected_parts": parts,
        "summary": [event["event_name"] for event in events], "execution_authorized": False,
        "cca_composition_status": "not_checked"}
    result.update(accepted=True, raw_proposal=deepcopy(proposal), recovery_proposal=proposal)
    return result
