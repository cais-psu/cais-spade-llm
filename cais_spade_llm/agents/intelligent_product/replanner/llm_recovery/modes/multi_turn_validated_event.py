from __future__ import annotations

"""Candidate outline events the LLM proposes and chooses; nothing checks them."""

from copy import deepcopy
from typing import Any


async def handle_validated_candidates(*, session: dict, prepared: dict, response: dict, planner: Any) -> tuple[str, dict]:
    """Apply the candidate the LLM chose, or answer its context request or completion.

    Nothing checks the reply. The chosen event is applied as the LLM states it; a reply that code
    cannot read raises, and the draft records that error.

    Args:
        session: Accepted planning state, updated with the chosen candidate.
        prepared: Recorded context shared by the candidates.
        response: LLM-authored candidate_events, a public context request or a completion.
        planner: Context transport for lookups.

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
    choice = response["selected_candidate_index"] if len(events) > 1 else 0
    decision, audit = await handle_event(session=session, prepared=prepared,
        response={"thought": response["thought"], "next_transition": deepcopy(events[choice])}, planner=planner)
    audit.update(candidate_events=deepcopy(events), selected_candidate_index=choice, llm_selected_candidate_index=choice,
                 validation_contract_version=2, outline_diagnostic_bypass=True)
    return decision, audit
