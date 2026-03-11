"""Environment model (M_e): compile resource bids and search for recovery paths."""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any, Callable, Coroutine, Optional

from .resource_bidding import Bid, _tool_signature

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
    obligation_targets: list[dict] | None = None,
    operator_feedback: str = "",
    primitive_catalog: list[dict] | None = None,
    bridge_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Ask the LLM for a recovery macro proposal when DES finds no modeled path.

    When primitive_catalog is provided, the bridge generates primitive-based
    macros that execute through execute_recovery_macro.  Otherwise, falls back
    to the legacy catalog-function-based macro shape.
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
        obligation_targets=obligation_targets,
        operator_feedback=operator_feedback,
        primitive_catalog=primitive_catalog,
        bridge_snapshot=bridge_snapshot,
    )

    raw = await ask_llm(prompt=prompt, with_functions=False)

    if primitive_catalog:
        proposal = _normalize_primitive_bridge_proposal(
            raw=raw,
            ra_jid=ra_jid,
            primitive_catalog=primitive_catalog,
            bridge_snapshot=bridge_snapshot or {},
        )
    else:
        proposal = _normalize_bridge_proposal(
            raw=raw,
            ra_jid=ra_jid,
            tools_catalog=tools_catalog,
        )

    if proposal:
        name_key = proposal.get("macro_name") or proposal.get("function_name")
        steps_key = proposal.get("primitive_steps") or proposal.get("macro_steps") or []
        logger.info(
            "[EnvironmentModel] LLM bridge proposed macro '%s' with %d step(s).",
            name_key,
            len(steps_key),
        )
        return proposal

    logger.warning("[EnvironmentModel] LLM bridge response was invalid or not compilable.")
    return None


def _normalize_primitive_bridge_proposal(
    *,
    raw: str,
    ra_jid: str,
    primitive_catalog: list[dict],
    bridge_snapshot: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Validate and normalize a primitive-based bridge proposal."""
    from cais_spade_llm.agents.intelligent_product.replanner.primitive_semantics import (
        expected_snapshot_from_bridge_snapshot,
        validate_and_project_steps,
    )

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("[EnvironmentModel] LLM bridge response was not valid JSON.")
        return None

    if not isinstance(parsed, dict):
        return None

    # Validate primitive_steps.
    primitive_steps = parsed.get("primitive_steps") or parsed.get("steps") or []
    if not isinstance(primitive_steps, list) or not primitive_steps:
        logger.warning("[EnvironmentModel] Bridge proposal has no primitive_steps.")
        return None

    validated_steps: list[dict[str, Any]] = []
    for index, step in enumerate(primitive_steps, start=1):
        if not isinstance(step, dict):
            return None
        primitive = str(step.get("primitive", "")).strip()
        params = step.get("params") or {}
        if not isinstance(params, dict):
            return None
        validated_steps.append({"primitive": primitive, "params": dict(params)})

    semantic_ok, projected_snapshot, semantic_error = validate_and_project_steps(
        validated_steps,
        primitive_catalog,
        bridge_snapshot or {},
    )
    if not semantic_ok:
        logger.warning(
            "[EnvironmentModel] Primitive bridge proposal rejected: %s",
            semantic_error,
        )
        return None

    macro_name = str(parsed.get("macro_name", "")).strip()
    if not macro_name:
        macro_name = "bridge_recovery_macro"

    # Validate resource_jid.
    proposal_resource_jid = str(parsed.get("resource_jid") or ra_jid).strip() or ra_jid
    if proposal_resource_jid != ra_jid:
        logger.warning(
            "[EnvironmentModel] Bridge proposal targeted unexpected resource '%s' (expected '%s').",
            proposal_resource_jid,
            ra_jid,
        )
        return None

    # Extract task_metadata for safety/tracking integration.
    task_metadata = parsed.get("task_metadata") or {}
    if not isinstance(task_metadata, dict):
        task_metadata = {}

    expected_start_state = str(parsed.get("expected_start_state", "")).strip()
    if expected_start_state:
        actual_state = str((bridge_snapshot or {}).get("current_state", "")).strip()
        if actual_state and expected_start_state != actual_state:
            logger.warning(
                "[EnvironmentModel] Bridge macro expected_start_state '%s' mismatched actual '%s'.",
                expected_start_state,
                actual_state,
            )
            return None

    return {
        "macro_name": macro_name,
        "resource_jid": proposal_resource_jid,
        "description": str(parsed.get("description", "")).strip(),
        "rationale": str(parsed.get("rationale", "")).strip(),
        "expected_start_state": expected_start_state,
        "expected_snapshot": expected_snapshot_from_bridge_snapshot(bridge_snapshot or {}),
        "projected_snapshot": projected_snapshot,
        "task_metadata": task_metadata,
        "primitive_steps": validated_steps,
        "summary": [
            macro_name,
            *[s["primitive"] for s in validated_steps],
        ],
    }


def _normalize_bridge_proposal(
    *,
    raw: str,
    ra_jid: str,
    tools_catalog: list[dict],
) -> Optional[dict[str, Any]]:
    """Legacy: validate and normalize a catalog-function-based bridge proposal."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("[EnvironmentModel] LLM bridge response was not valid JSON.")
        return None

    if isinstance(parsed, list):
        parsed = {
            "function_name": "bridge_recovery_macro",
            "resource_jid": ra_jid,
            "description": "LLM-generated recovery macro",
            "rationale": "",
            "macro_steps": parsed,
        }
    if not isinstance(parsed, dict):
        return None

    resource_token = str(ra_jid or "").split("@", 1)[0].strip().lower()
    resource_tools = [
        row
        for row in (tools_catalog or [])
        if isinstance(row, dict)
        and str(row.get("function_owner_agent", "")).strip().lower() == resource_token
    ]
    tools_by_name = {
        str(row.get("function", "")).strip(): row
        for row in resource_tools
        if str(row.get("function", "")).strip()
    }
    macro_steps = parsed.get("macro_steps") or parsed.get("steps") or []
    if not isinstance(macro_steps, list) or not macro_steps:
        return None

    compiled_macro: list[dict[str, Any]] = []
    for index, step in enumerate(macro_steps, start=1):
        if not isinstance(step, dict):
            return None
        function_name = str(step.get("function_name", "")).strip()
        if not function_name or function_name not in tools_by_name:
            logger.warning(
                "[EnvironmentModel] Bridge macro step %d used non-catalog function '%s'.",
                index,
                function_name,
            )
            return None
        params = step.get("params") or {}
        if not isinstance(params, dict):
            return None
        tool_row = tools_by_name[function_name]
        compiled_macro.append(
            {
                "resource_jid": ra_jid,
                "function_name": function_name,
                "params": dict(params),
                "description": str(tool_row.get("description", "")).strip(),
                "tool_signature": _tool_signature(tool_row),
                "in_state": str(tool_row.get("in_state", "")).strip(),
                "out_state": str(tool_row.get("out_state", "")).strip(),
            }
        )

    function_name = str(parsed.get("function_name", "")).strip()
    if not function_name:
        return None

    proposal_resource_jid = str(parsed.get("resource_jid") or ra_jid).strip() or ra_jid
    if proposal_resource_jid != ra_jid:
        logger.warning(
            "[EnvironmentModel] Bridge proposal targeted unexpected resource '%s' (expected '%s').",
            proposal_resource_jid,
            ra_jid,
        )
        return None

    return {
        "function_name": function_name,
        "resource_jid": proposal_resource_jid,
        "description": str(parsed.get("description", "")).strip(),
        "rationale": str(parsed.get("rationale", "")).strip(),
        "macro_steps": compiled_macro,
        "summary": [
            function_name,
            *[str(step.get("function_name", "")).strip() for step in compiled_macro],
        ],
    }
