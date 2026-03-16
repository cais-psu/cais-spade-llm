"""Resource bidding: data structures and DES forward-search for resource agents."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #
@dataclass
class BidRequest:
    """PA → RA: request a feasible action sequence for recovery."""
    request_id: str
    pa_jid: str
    x_c: dict                  # current state: {resource_state, part_states, part_locations}
    P_id: list[str]            # parts that must reach goal state


@dataclass
class Bid:
    """RA → PA: feasible event sequence in response to a BidRequest."""
    request_id: str
    ra_jid: str
    str_e: list[dict]          # sequence of events: [{function_name, params}]
    str_x: list[dict]          # sequence of states after each event
    prp_p_achieved: list[str]  # which P_id parts this bid assembles
    complete: bool             # True = fully satisfies P_id; False = partial (reaches staging)


# ------------------------------------------------------------------ #
# DES forward BFS to compute a resource bid
# ------------------------------------------------------------------ #
def compute_bid(
    x_c: dict,
    P_id: list[str],
    goal_state: str,
    tools: list[dict],
    reachability: list[str],
    staging_areas: dict,
    resource_jid: str,
    goal_resource_state: str | None = None,
    goal_event_signatures: set[str] | None = None,
) -> Bid | None:
    """
    DES forward BFS to find a feasible event sequence for this resource.

    x_c keys expected:
        resource_state   : current FSM state (e.g. "idle", "picked")
        current_part     : part currently being handled (None if idle)
        current_location : location resource is at/heading to (None if idle)
        part_states      : {part_name: state}
        part_locations   : {part_name: location}

    Returns a complete Bid (all P_id assembled and/or resource reaches
    goal_resource_state), a partial Bid (resource reaches a staging area),
    or None if no path found.
    """
    resource_name = resource_jid.split("@")[0]
    resource_tools = [t for t in tools if t.get("function_owner_agent") == resource_name]

    if not resource_tools:
        return None

    staging_names = set(staging_areas.keys())
    goal_event_signatures = set(goal_event_signatures or set())

    def _key(rs, cp, cl, ps, pl):
        return (rs, cp, cl, frozenset(ps.items()), frozenset(pl.items()))

    normalized = _normalize_state(x_c)
    init = dict(
        rs=normalized["resource_state"],
        cp=normalized["current_part"],
        cl=normalized["current_location"],
        ps=dict(normalized["part_states"]),
        pl=dict(normalized["part_locations"]),
    )

    start_state = {
        "resource_state": init["rs"],
        "current_part": init["cp"],
        "current_location": init["cl"],
        "part_states": init["ps"],
        "part_locations": init["pl"],
    }

    # BFS queue: (rs, cp, cl, ps_frozen, pl_frozen, events, states)
    queue: deque = deque([(
        init["rs"], init["cp"], init["cl"],
        frozenset(init["ps"].items()), frozenset(init["pl"].items()),
        [], [start_state],
    )])
    visited: set = set([_key(init["rs"], init["cp"], init["cl"], init["ps"], init["pl"])])
    best_partial: tuple | None = None

    while queue:
        rs, cp, cl, ps_f, pl_f, events, states = queue.popleft()

        ps = dict(ps_f)
        pl = dict(pl_f)

        # --- complete goal ---
        parts_complete = all(ps.get(p) == goal_state for p in P_id)
        resource_complete = (
            goal_resource_state is None
            or rs == goal_resource_state
        )
        if not goal_event_signatures and parts_complete and resource_complete:
            return Bid(
                request_id="",
                ra_jid=resource_jid,
                str_e=events,
                str_x=states,
                prp_p_achieved=[p for p in P_id if ps.get(p) == goal_state],
                complete=True,
            )

        # --- partial goal: resource is at a staging area ---
        if (
            not goal_event_signatures
            and
            goal_resource_state is None
            and cl in staging_names
            and events
            and best_partial is None
        ):
            best_partial = (events[:], states[:])

        # --- expand ---
        for new_rs, new_cp, new_cl, new_ps, new_pl, event_dict in _expand(
            resource_tools, rs, cp, cl, ps, pl, P_id, reachability, staging_names, resource_jid, goal_state
        ):
            new_key = _key(new_rs, new_cp, new_cl, new_ps, new_pl)
            if new_key in visited:
                continue
            new_state = {
                "resource_state": new_rs,
                "current_part": new_cp,
                "current_location": new_cl,
                "part_states": new_ps,
                "part_locations": new_pl,
            }
            if goal_event_signatures and str(event_dict.get("_tool_signature", "")) in goal_event_signatures:
                clean_event = {k: v for k, v in event_dict.items() if not str(k).startswith("_")}
                return Bid(
                    request_id="",
                    ra_jid=resource_jid,
                    str_e=events + [clean_event],
                    str_x=states + [new_state],
                    prp_p_achieved=[p for p in P_id if new_ps.get(p) == goal_state],
                    complete=True,
                )
            visited.add(new_key)
            queue.append((
                new_rs, new_cp, new_cl,
                frozenset(new_ps.items()), frozenset(new_pl.items()),
                events + [{k: v for k, v in event_dict.items() if not str(k).startswith("_")}],
                states + [new_state],
            ))

    if best_partial:
        ev, st = best_partial
        last_ps = st[-1].get("part_states", {})
        return Bid(
            request_id="",
            ra_jid=resource_jid,
            str_e=ev,
            str_x=st,
            prp_p_achieved=[p for p in P_id if last_ps.get(p) == goal_state],
            complete=False,
        )

    return None


def simulate_catalog_transition(
    *,
    x_c: dict,
    tools: list[dict],
    resource_jid: str,
    function_name: str,
    params: dict[str, Any] | None = None,
    goal_state: str = "",
    reachability: list[str] | None = None,
    staging_areas: dict | None = None,
) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Simulate one exact catalog-backed task from a modeled search state.

    This reuses the same `_expand()` transition semantics as DES bidding and
    selects the unique transition whose generated function/params match the
    requested task.
    """
    normalized = _normalize_state(x_c)
    state_params = dict(params or {})
    candidate_parts = list(normalized["part_states"].keys())
    resource_name = resource_jid.split("@")[0]
    resource_tools = [t for t in tools if t.get("function_owner_agent") == resource_name]

    for new_rs, new_cp, new_cl, new_ps, new_pl, event_dict in _expand(
        resource_tools,
        normalized["resource_state"],
        normalized["current_part"],
        normalized["current_location"],
        dict(normalized["part_states"]),
        dict(normalized["part_locations"]),
        candidate_parts,
        list(reachability or []),
        set((staging_areas or {}).keys()),
        resource_jid,
        goal_state,
    ):
        if str(event_dict.get("function_name") or "").strip() != str(function_name or "").strip():
            continue
        if not _event_params_match(event_dict.get("params") or {}, state_params):
            continue
        next_state = {
            "resource_state": new_rs,
            "current_part": new_cp,
            "current_location": new_cl,
            "part_states": dict(new_ps),
            "part_locations": dict(new_pl),
        }
        clean_event = {k: v for k, v in event_dict.items() if not str(k).startswith("_")}
        return next_state, clean_event

    return None


def _tool_signature(tool: dict) -> str:
    payload = {
        "function_owner_agent": str(tool.get("function_owner_agent") or "").strip(),
        "function": str(tool.get("function") or "").strip(),
        "in_state": str(tool.get("in_state") or "").strip(),
        "out_state": str(tool.get("out_state") or "").strip(),
        "part_in_state": str(tool.get("part_in_state") or "").strip(),
        "location_type": str(
            (tool.get("context_mapping") or {}).get("location_type") or ""
        ).strip(),
        "location_param": str(
            (tool.get("context_mapping") or {}).get("location_param") or ""
        ).strip(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _normalize_state(x_c: dict | None) -> dict[str, Any]:
    payload = dict(x_c or {})
    return {
        "resource_state": payload.get("resource_state", "idle"),
        "current_part": payload.get("current_part"),
        "current_location": payload.get("current_location"),
        "part_states": dict(payload.get("part_states", {})),
        "part_locations": dict(payload.get("part_locations", {})),
    }


def _event_params_match(generated: dict[str, Any], requested: dict[str, Any]) -> bool:
    for key, value in (generated or {}).items():
        if requested.get(key) != value:
            return False
    return True


def _expand(tools, rs, cp, cl, ps, pl, P_id, reachability, staging_names, resource_jid, goal_state):
    """
    Yield (new_rs, new_cp, new_cl, new_ps, new_pl, event_dict)
    for every valid tool application from the current state.
    Generalized to read preconditions and effects dynamically from tools.json.
    """
    for tool in tools:
        in_state = str(tool.get("in_state") or "").strip().lower()
        if in_state not in {"", "any"} and in_state != str(rs).strip().lower():
            continue

        fn = tool.get("function")
        out = tool.get("out_state")
        part_in = tool.get("part_in_state")
        part_effect = tool.get("part_transition", {}).get("completed", {})
        
        # Generic location parameter mapping
        ctx_map = tool.get("context_mapping", {})
        loc_param = ctx_map.get("location_param")
        loc_type = ctx_map.get("location_type")
        loc_template = part_effect.get("location_template") if cp else None

        # Pure robot-state actions such as move_home do not manipulate parts or locations.
        if cp is None and not part_in and not loc_type:
            next_cl = None if str(out or "").strip().lower() == "idle" else cl
            yield (
                out or rs, None, next_cl, dict(ps), dict(pl),
                {"function_name": fn, "params": {}, "_tool_signature": _tool_signature(tool)},
            )
            continue

        # Tools targeting a specific part location (e.g., move_to_pick)
        if loc_type == "part_location" and not part_in:
            for part in P_id:
                if ps.get(part) == goal_state:
                    continue
                loc = pl.get(part)
                if loc and loc in reachability:
                    yield (
                        out, part, loc, dict(ps), dict(pl),
                        {
                            "function_name": fn,
                            "params": {loc_param: loc, "part_name": part},
                            "_tool_signature": _tool_signature(tool),
                        },
                    )
            continue
            
        # Tools that require a specific part logic state
        if cp:
            if part_in and ps.get(cp) != part_in:
                continue

            part_effect = tool.get("part_transition", {}).get("completed", {})
            new_ps = {**ps}
            new_pl = {**pl}
            
            if "state" in part_effect:
                new_ps[cp] = part_effect["state"]

            params = {"part_name": cp}

            # Tools that execute at the current location and stay there (e.g. pick_grasp, place_insert)
            if loc_type == "current_location":
                if loc_param:
                    params[loc_param] = cl

                # If the action places the part down (assembly), resource no longer holds it
                next_cp = None if new_ps[cp] == goal_state else cp
                next_cl = None if next_cp is None else cl 
                
                # If picking or transferring physically into the gripper template
                if loc_template:
                    formatted_loc = loc_template.replace("{resource_jid}", resource_jid)
                    new_pl[cp] = formatted_loc
                elif not next_cp:
                    # Dropped into the environment at cl
                    new_pl[cp] = cl

                yield (
                    out, next_cp, next_cl, new_ps, new_pl,
                    {
                        "function_name": fn,
                        "params": params,
                        "_tool_signature": _tool_signature(tool),
                    },
                )
                
            # Tools that traverse to an explicit new destination (e.g. place_approach)
            elif loc_type == "reachable_location":
                dest_options = [
                    d for d in list(reachability) + (list(staging_names) if goal_state not in new_ps.get(cp, "") else [])
                    if d != cl
                ]
                
                for dest in dest_options:
                    p_copy = dict(new_pl)
                    
                    if loc_template:
                        p_copy[cp] = loc_template.replace("{resource_jid}", resource_jid)
                    
                    event_params = dict(params)
                    if loc_param:
                        event_params[loc_param] = dest
                        
                    yield (
                        out, cp, dest, new_ps, p_copy,
                        {
                            "function_name": fn,
                            "params": event_params,
                            "_tool_signature": _tool_signature(tool),
                        },
                    )
