"""Environment model (M_e): compile resource bids and search for recovery paths."""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any, Callable, Coroutine

from .resource_bidding import Bid

logger = logging.getLogger(__name__)


def _state_key(state: dict) -> str:
    """Stable string key from a state dict."""
    return json.dumps(state, sort_keys=True)


def compile_environment_model(bids: list[Bid]) -> dict:
    """
    Fuse RA bids into environment model M_e (Algorithm 1, Kovalenko et al.).

    M_e = {
        "states":      {state_key: state_dict},
        "transitions": {state_key: {event_key: next_state_key}},
        "events":      {event_key: event_dict},   # includes ra_jid
    }
    """
    states: dict[str, dict] = {}
    transitions: dict[str, dict[str, str]] = {}
    events: dict[str, dict] = {}

    for bid in bids:
        if not bid.str_x or not bid.str_e:
            continue
        for i, event_dict in enumerate(bid.str_e):
            from_state = bid.str_x[i]
            to_state = bid.str_x[i + 1]

            from_key = _state_key(from_state)
            to_key = _state_key(to_state)

            states.setdefault(from_key, from_state)
            states.setdefault(to_key, to_state)

            fn_name = event_dict.get("function_name", "")
            event_key = f"{bid.ra_jid}::{fn_name}::{i}"

            events[event_key] = {**event_dict, "ra_jid": bid.ra_jid}
            transitions.setdefault(from_key, {})[event_key] = to_key

    return {
        "states": states,
        "transitions": transitions,
        "events": events,
    }


def plan_on_environment_model(
    M_e: dict,
    x_c: dict,
    P_id: list[str],
    goal_state: str,
) -> list[dict] | None:
    """
    BFS on M_e from x_c to a goal state where all P_id parts are at goal_state.
    Finds the path with fewest steps.

    Returns ordered list of event dicts (each includes ra_jid and params),
    or None if no path exists.
    """
    states = M_e["states"]
    transitions = M_e["transitions"]
    events = M_e["events"]

    start_key = _find_start_state(states, x_c)
    if start_key is None:
        logger.warning("[EnvironmentModel] Could not match x_c to any state in M_e.")
        return None

    def is_goal(state_key: str) -> bool:
        part_states = states[state_key].get("part_states", {})
        return all(part_states.get(p) == goal_state for p in P_id)

    # BFS: queue of (state_key, path_as_event_keys)
    queue: deque = deque([(start_key, [])])
    visited: set[str] = set([start_key])

    while queue:
        current_key, path = queue.popleft()

        if is_goal(current_key):
            return [events[ek] for ek in path]

        for event_key, next_key in transitions.get(current_key, {}).items():
            if next_key not in visited:
                visited.add(next_key)
                queue.append((next_key, path + [event_key]))

    return None


def _find_start_state(states: dict, x_c: dict) -> str | None:
    """Match x_c to a state key in M_e. Exact match first, then partial."""
    exact = _state_key(x_c)
    if exact in states:
        return exact

    # Partial: same resource_state and part_states
    for key, state in states.items():
        if (state.get("resource_state") == x_c.get("resource_state") and
                state.get("part_states") == x_c.get("part_states")):
            return key

    return None


async def llm_explore_states_and_events(
    stuck_state: dict,
    P_id: list[str],
    ra_jid: str,
    ask_llm: Callable[..., Coroutine[Any, Any, str]],
    goal_state: str,
    tools_catalog: list[dict],
    resource_infos: list[dict],
    part_tracker: dict | None = None,
) -> list[dict]:
    """
    Ask the LLM for bridge tool entries when BFS finds no path.
    part_tracker provides full part info including camera coordinates for lost parts.
    Returns synthetic event dicts injected into M_e for a second BFS pass.
    """
    from cais_spade_llm.prompts import build_state_exploration_prompt

    prompt = build_state_exploration_prompt(
        stuck_state=stuck_state,
        part_tracker=part_tracker,
        P_id=P_id,
        goal_state=goal_state,
        ra_jid=ra_jid,
        tools_catalog=tools_catalog,
        resource_infos=resource_infos,
    )

    raw = await ask_llm(prompt=prompt, with_functions=False)

    try:
        bridge_tools = json.loads(raw)
        if isinstance(bridge_tools, list):
            logger.info("[EnvironmentModel] LLM bridge returned %d tool(s).", len(bridge_tools))
            return bridge_tools
    except (json.JSONDecodeError, ValueError):
        logger.warning("[EnvironmentModel] LLM bridge response was not valid JSON.")

    return []
