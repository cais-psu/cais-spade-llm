from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any, Callable, Coroutine

from .bidding import Bid

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
) -> list[dict] | None:
    """
    BFS on M_e from x_c to a goal state where all P_id parts are assembled.
    Finds the path with fewest steps.

    Returns ordered list of event dicts (each includes ra_jid and params),
    or None if no path exists.
    """
    states = M_e["states"]
    transitions = M_e["transitions"]
    events = M_e["events"]

    start_key = _find_start_state(states, x_c)
    if start_key is None:
        logger.warning("[GraphPlanner] Could not match x_c to any state in M_e.")
        return None

    def is_goal(state_key: str) -> bool:
        part_states = states[state_key].get("part_states", {})
        return all(part_states.get(p) == "assembled" for p in P_id)

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

    # Partial: same robot_state and part_states
    for key, state in states.items():
        if (state.get("robot_state") == x_c.get("robot_state") and
                state.get("part_states") == x_c.get("part_states")):
            return key

    return None


async def ask_llm_for_bridge(
    stuck_state: dict,
    P_id: list[str],
    ra_jid: str,
    ask_llm: Callable[..., Coroutine[Any, Any, str]],
    part_tracker: dict | None = None,
) -> list[dict]:
    """
    Ask the LLM for bridge tool entries when BFS finds no path.
    part_tracker provides full part info including camera coordinates for lost parts.
    Returns synthetic event dicts injected into M_e for a second BFS pass.
    """
    part_info = json.dumps(part_tracker, indent=2) if part_tracker else "unavailable"
    prompt = (
        f"A robot ({ra_jid}) is stuck in state:\n{json.dumps(stuck_state, indent=2)}\n\n"
        f"Current part states and locations (including camera coordinates for lost parts):\n{part_info}\n\n"
        f"Parts that still need to be assembled: {P_id}\n\n"
        "The robot has no available tool to make progress. "
        "Generate 1-3 recovery tool steps as a JSON array:\n"
        "[\n"
        "  {\n"
        '    "function_name": "<tool name>",\n'
        '    "in_state": "<robot state before>",\n'
        '    "out_state": "<robot state after>",\n'
        '    "part_effect": {"<part_name>": {"state": "<new_state>", "location": "<new_location>"}},\n'
        '    "params": {"<param_name>": "<value>"},\n'
        '    "duration": <seconds as number>\n'
        "  }\n"
        "]\n\n"
        "Return ONLY the JSON array, no explanation."
    )

    raw = await ask_llm(prompt=prompt, with_functions=False)

    try:
        bridge_tools = json.loads(raw)
        if isinstance(bridge_tools, list):
            logger.info("[GraphPlanner] LLM bridge returned %d tool(s).", len(bridge_tools))
            return bridge_tools
    except (json.JSONDecodeError, ValueError):
        logger.warning("[GraphPlanner] LLM bridge response was not valid JSON.")

    return []