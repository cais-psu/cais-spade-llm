from __future__ import annotations

"""Candidate outline events the LLM proposes and chooses; the RA feasibility and CCA safety checks, when on, check them.

The CCA nonblocking check, when on, checks the LLM's declaration that the recovery is complete.
"""

from copy import deepcopy
from typing import Any


async def handle_validated_candidates(*, session: dict, prepared: dict, response: dict, planner: Any) -> tuple[str, dict]:
    """Apply the candidate the LLM chose, or answer its context request or completion.

    Without ``check_ra_feasibility`` nothing checks the reply: the chosen event is applied as the LLM
    states it. With it, the RA checks the chosen candidate first, then the LLM's other candidates in its
    order, and applies the first feasible one (Alg. 1 and Eq. 12); when none is feasible, the facts that
    refuse each candidate return to the LLM as validation feedback and nothing is applied. With
    ``check_nonblocking``, the CCA refuses a ``recovery_complete`` from which normal production cannot
    carry on (Eq. 6, judged in the composition of the runtime models), naming only what it cannot
    continue; synthesis continues from the same state. With ``check_cca_safety`` (which needs the RA
    check), each candidate the RA finds feasible is also checked by the CCA against every specification
    (Eq. 13), and the first candidate in the LLM's order that passes both is applied (Eq. 9). A reply
    that code cannot read raises, and the draft records that error.

    Args:
        session: Accepted planning state, updated with the chosen candidate.
        prepared: Recorded context shared by the candidates.
        response: LLM-authored candidate_events, a public context request or a completion.
        planner: Context transport for lookups; with a check on, its product agent carries what the check
            reads: the recorded cell for the RA check, the capture's runtime inputs for the CCA check, the
            grounded specifications for the CCA safety check.

    Returns:
        Continuation decision and the turn audit.
    """
    from .multi_turn_context import check_context_revision, handle_outline_context
    from .multi_turn_diagnostic import declare_recovery_complete, handle_event
    from .multi_turn_outline_facts import response_with_jids

    if session.get("outline_diagnostic_bypass") is not True:
        raise ValueError("Outline validation was removed; only unvalidated drafts propose outline events")
    check_context_revision(prepared, planner)
    response = response_with_jids(response, prepared)
    if response.get("decision") == "recovery_complete":
        if session.get("check_nonblocking") is True:
            from cais_spade_llm.recovery_framework import cca_nonblocking

            capture = getattr(planner.product_agent, "cca_capture", None)
            if capture is None:
                raise ValueError("The CCA nonblocking check needs the capture's runtime inputs")
            check = cca_nonblocking.check_completion(session, prepared, capture)
            # Kept for review: what the CCA judged on each declared completion.
            session.setdefault("cca_nonblocking_checks", []).append({"turn_index": session.get("turn_index"),
                                                                     **deepcopy(check)})
            if check["nonblocking_status"] == cca_nonblocking.BLOCKING:
                findings = cca_nonblocking.refusal_findings(check)
                session["outline_validation_findings"] = deepcopy(findings)
                session["status"] = "paused_after_outline_turn"
                return "need_revision", {"decision": "recovery_complete", "cca_nonblocking": check,
                                         "validation_findings": findings, "validation_contract_version": 2,
                                         "outline_diagnostic_bypass": True}
            session["outline_validation_findings"] = []
        return declare_recovery_complete(session=session, prepared=prepared, response=response)
    if response.get("decision") == "need_context":
        # The response schema offers these fields on every reply; empty, they propose nothing.
        response = {key: value for key, value in response.items()
                    if key != "selected_candidate_index" and not (key == "candidate_events" and not value)}
        return await handle_outline_context(prepared=prepared, session=session, response=response, planner=planner)
    events = response["candidate_events"]
    # The LLM chooses; a lone candidate is its own choice.
    llm_choice = response["selected_candidate_index"] if len(events) > 1 else 0
    choice, checks, safety = llm_choice, None, None
    if session.get("check_cca_safety") is True:
        from cais_spade_llm.recovery_framework import cca_safety, ra_feasibility

        choice, checks, safety = _check_feasible_and_safe(events, llm_choice, session, prepared, planner)
        if choice is None:
            refused = [row for row in checks if row["feasibility_status"] != ra_feasibility.FEASIBLE]
            findings = ra_feasibility.refusal_findings(events, refused) + cca_safety.refusal_findings(events, safety)
            session["outline_validation_findings"] = deepcopy(findings)
            session["status"] = "paused_after_outline_turn"
            return "need_revision", {"candidate_events": deepcopy(events), "llm_selected_candidate_index": llm_choice,
                                     "ra_feasibility": checks, "cca_safety": [cca_safety.recorded(row) for row in safety],
                                     "validation_stages": _validation_stages(checks, safety),
                                     "validation_findings": findings, "validation_contract_version": 2,
                                     "outline_diagnostic_bypass": True}
        session["outline_validation_findings"] = []
    elif session.get("check_ra_feasibility") is True:
        from cais_spade_llm.recovery_framework import ra_feasibility

        cell = getattr(planner.product_agent, "recorded_cell", None)
        if cell is None:
            raise ValueError("The RA feasibility check needs the recorded cell of the capture")
        choice, checks = ra_feasibility.check_candidates(events, llm_choice, session, prepared, cell)
        # Kept for review: what the RA judged on each turn and which candidate it applied.
        session.setdefault("ra_feasibility_checks", []).append({
            "turn_index": session.get("turn_index"), "llm_selected_candidate_index": llm_choice,
            "applied_candidate_index": choice, "results": deepcopy(checks)})
        if choice is None:
            findings = ra_feasibility.refusal_findings(events, checks)
            session["outline_validation_findings"] = deepcopy(findings)
            session["status"] = "paused_after_outline_turn"
            return "need_revision", {"candidate_events": deepcopy(events), "llm_selected_candidate_index": llm_choice,
                                     "ra_feasibility": checks, "validation_findings": findings,
                                     "validation_contract_version": 2, "outline_diagnostic_bypass": True}
        session["outline_validation_findings"] = []
    decision, audit = await handle_event(session=session, prepared=prepared,
        response={"thought": response["thought"], "next_transition": deepcopy(events[choice])}, planner=planner)
    audit.update(candidate_events=deepcopy(events), selected_candidate_index=choice, llm_selected_candidate_index=llm_choice,
                 validation_contract_version=2, outline_diagnostic_bypass=True)
    if checks is not None:
        audit["ra_feasibility"] = checks
    if safety is not None:
        cca_safety.commit(session, next(row for row in safety if row["candidate_index"] == choice))
        audit["cca_safety"] = [cca_safety.recorded(row) for row in safety]
        audit["validation_stages"] = _validation_stages(checks, safety)
    return decision, audit


def _check_feasible_and_safe(events: list[dict], llm_choice: int, session: dict, prepared: dict,
                             planner: Any) -> tuple[int | None, list[dict], list[dict]]:
    """Check the LLM's chosen candidate first, then its others in its order: RA feasibility, then CCA safety.

    Returns:
        The first candidate both checks pass, or None, with every RA result and every CCA result.
    """
    from cais_spade_llm.recovery_framework import cca_safety, ra_feasibility

    if session.get("check_ra_feasibility") is not True:
        raise ValueError("The CCA safety check places each acting robot from the RA's reach results; "
                         "turn on the RA feasibility check too")
    cell = getattr(planner.product_agent, "recorded_cell", None)
    capture = getattr(planner.product_agent, "cca_safety_capture", None)
    if cell is None or capture is None:
        raise ValueError("The CCA safety check needs the capture's recorded cell and grounded specifications")
    choice, checks, safety = None, [], []
    for index in [llm_choice, *(index for index in range(len(events)) if index != llm_choice)]:
        feasibility = {"candidate_index": index, **ra_feasibility.check_event(events[index], session, prepared, cell)}
        checks.append(feasibility)
        if feasibility["feasibility_status"] != ra_feasibility.FEASIBLE:
            continue
        result = {"candidate_index": index, **cca_safety.check_event(events[index], session, prepared, capture, feasibility)}
        safety.append(result)
        if result["safety_status"] == cca_safety.SAFE:
            choice = index
            break
    # Kept for review: what each check judged on each turn and which candidate was applied.
    session.setdefault("ra_feasibility_checks", []).append({
        "turn_index": session.get("turn_index"), "llm_selected_candidate_index": llm_choice,
        "applied_candidate_index": choice, "results": deepcopy(checks)})
    session.setdefault("cca_safety_checks", []).append({
        "turn_index": session.get("turn_index"), "llm_selected_candidate_index": llm_choice,
        "applied_candidate_index": choice, "results": [cca_safety.recorded(row) for row in safety]})
    return choice, checks, safety


def _validation_stages(checks: list[dict] | None, safety: list[dict]) -> list[dict]:
    """Each check's result per candidate, as the Recovery UI's validation panels show them."""
    from cais_spade_llm.recovery_framework import cca_safety

    stages = [{"validator_role": "ResourceAgent", "candidate_index": row["candidate_index"], "evidence": deepcopy(row)}
              for row in checks or []]
    return stages + [{"validator_role": "CentralControllerAgent", "candidate_index": row["candidate_index"],
                      "evidence": cca_safety.recorded(row)} for row in safety]
