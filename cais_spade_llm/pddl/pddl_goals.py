"""
LLM-based goal extractor for PDDL replanning.

Converts natural-language product requirements + current part states into
a structured list of pending goals: [{part, destination}, ...].

This is the ONLY LLM call in the PDDL replanning pipeline — everything
else (domain, problem, plan) is deterministic.

Usage::

    from cais_spade_llm.pddl.pddl_goals import extract_pending_goals

    goals = await extract_pending_goals(
        llm_client   = openai_client,
        model        = "o4-mini",
        requirements = "Assemble: SG then MCP onto assembly_board-v1",
        parts_state  = system_state["parts"],
        locations    = ["assembly_board-v1", "prusa-mk4-1", "prusa-mk4-2"],
    )
    # → [{"part": "SG", "destination": "assembly_board-v1"},
    #    {"part": "MCP", "destination": "assembly_board-v1"}]
"""

from __future__ import annotations
import json
import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a manufacturing task planner assistant.
Given assembly requirements and the current state of parts, return ONLY
the parts that still need to be placed (not yet verified/placed) and their
target destination.

Return a JSON array. Each element must have exactly two keys:
  "part"        - the part identifier (as it appears in parts_state)
  "destination" - the target location name

Only include parts whose state is NOT "verified" or "placed".
Do not include parts already at their destination.
Return [] if all parts are already placed.
"""

_USER_TEMPLATE = """\
## Assembly Requirements
{requirements}

## Current Part States
{parts_state_json}

## Known Locations
{locations_list}

Return the JSON array of pending goals only. No explanation.
"""


async def extract_pending_goals(
    *,
    llm_client: Any,           # openai.OpenAI instance
    model: str,
    requirements: str,
    parts_state: dict[str, Any],
    locations: list[str],
    reasoning_effort: str = "low",
) -> list[dict[str, str]]:
    """
    Ask the LLM to identify which parts still need to be placed and where.

    Returns:
        List of {"part": str, "destination": str} dicts for pending goals.
        Falls back to an empty list on parse failure (caller should handle).
    """
    # Fast path: if we can determine pending goals without LLM, do so.
    # A part needs placing if its state is not verified/placed.
    DONE_STATES = {"verified", "placed"}
    heuristic_pending = [
        pid for pid, ps in parts_state.items()
        if ps.get("state", "") not in DONE_STATES
    ]

    if not heuristic_pending:
        logger.info("[PDDLGoals] All parts already placed — no LLM call needed.")
        return []

    user_msg = _USER_TEMPLATE.format(
        requirements=requirements,
        parts_state_json=json.dumps(parts_state, indent=2),
        locations_list="\n".join(f"  - {loc}" for loc in sorted(locations)),
    )

    def _call() -> str:
        r = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            reasoning_effort=reasoning_effort,
        )
        return (r.choices[0].message.content or "").strip()

    raw = await asyncio.to_thread(_call)

    try:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        goals: list[dict] = json.loads(text)
        # Validate structure
        result = []
        for g in goals:
            if isinstance(g, dict) and "part" in g and "destination" in g:
                result.append({"part": str(g["part"]), "destination": str(g["destination"])})
            else:
                logger.warning("[PDDLGoals] Skipping malformed goal entry: %s", g)
        return result

    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("[PDDLGoals] Failed to parse LLM goal response: %s\nRaw: %s", exc, raw)
        return []


def extract_pending_goals_sync(
    *,
    llm_client: Any,
    model: str,
    requirements: str,
    parts_state: dict[str, Any],
    locations: list[str],
    reasoning_effort: str = "low",
) -> list[dict[str, str]]:
    """Synchronous wrapper for use outside async contexts (e.g. tests)."""
    return asyncio.run(extract_pending_goals(
        llm_client=llm_client,
        model=model,
        requirements=requirements,
        parts_state=parts_state,
        locations=locations,
        reasoning_effort=reasoning_effort,
    ))
