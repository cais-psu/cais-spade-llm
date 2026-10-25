from __future__ import annotations

"""Candidate outline events the LLM proposes and chooses; only the RA feasibility check, when on, checks them."""

from copy import deepcopy
from typing import Any


async def handle_validated_candidates(*, session: dict, prepared: dict, response: dict, planner: Any) -> tuple[str, dict]:
    """Apply the candidate the LLM chose, or answer its context request or completion.

    Without ``check_ra_feasibility`` nothing checks the reply: the chosen event is applied as the LLM
    states it. With it, the RA checks the chosen candidate first, then the LLM's other candidates in its
    order, and applies the first feasible one (Alg. 1 and Eq. 12); when none is feasible, the facts that
    refuse each candidate return to the LLM as validation feedback and nothing is applied. A reply that
    code cannot read raises, and the draft records that error.

    Args:
        session: Accepted planning state, updated with the chosen candidate.
        prepared: Recorded context shared by the candidates.
        response: LLM-authored candidate_events, a public context request or a completion.
        planner: Context transport for lookups; with the check on, its product agent carries the recorded
            cell the check reads.

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
        return declare_recovery_complete(session=session, prepared=prepared, response=response)
    if response.get("decision") == "need_context":
        # The response schema offers these fields on every reply; empty, they propose nothing.
        response = {key: value for key, value in response.items()
                    if key != "selected_candidate_index" and not (key == "candidate_events" and not value)}
        return await handle_outline_context(prepared=prepared, session=session, response=response, planner=planner)
    events = response["candidate_events"]
    # The LLM chooses; a lone candidate is its own choice.
    llm_choice = response["selected_candidate_index"] if len(events) > 1 else 0
    choice, checks = llm_choice, None
    if session.get("check_ra_feasibility") is True:
        from cais_spade_llm.recovery_framework import ra_feasibility

        cell = getattr(planner.product_agent, "recorded_cell", None)
        if cell is None:
            raise ValueError("The RA feasibility check needs the recorded cell of the capture")
        choice, checks = ra_feasibility.check_candidates(events, llm_choice, session, prepared, cell)
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
    return decision, audit
