from __future__ import annotations

import logging
from collections import deque

from .bidding import Bid

logger = logging.getLogger(__name__)


def compute_bid(
    x_c: dict,
    P_id: list[str],
    tools: list[dict],
    reachability: list[str],
    staging_areas: dict,
    robot_jid: str,
) -> Bid | None:
    """
    DES forward BFS to find a feasible event sequence for this robot.

    x_c keys expected:
        robot_state      : current FSM state ("idle", "picked", etc.)
        current_part     : part currently being handled (None if idle)
        current_location : location robot is at/heading to (None if idle)
        part_states      : {part_name: state}
        part_locations   : {part_name: location}

    Returns a complete Bid (all P_id assembled), a partial Bid
    (robot reaches a staging area), or None if no path found.
    """
    robot_name = robot_jid.split("@")[0]
    robot_tools = [t for t in tools if t.get("function_owner_agent") == robot_name]

    if not robot_tools:
        return None

    staging_names = set(staging_areas.keys())

    def _key(rs, cp, cl, ps, pl):
        return (rs, cp, cl, frozenset(ps.items()), frozenset(pl.items()))

    init = dict(
        rs=x_c.get("robot_state", "idle"),
        cp=x_c.get("current_part"),
        cl=x_c.get("current_location"),
        ps=dict(x_c.get("part_states", {})),
        pl=dict(x_c.get("part_locations", {})),
    )

    start_state = {
        "robot_state": init["rs"],
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
        if all(ps.get(p) == "assembled" for p in P_id):
            return Bid(
                request_id="",
                ra_jid=robot_jid,
                str_e=events,
                str_x=states,
                prp_p_achieved=[p for p in P_id if ps.get(p) == "assembled"],
                complete=True,
            )

        # --- partial goal: robot is at a staging area ---
        if cl in staging_names and events and best_partial is None:
            best_partial = (events[:], states[:])

        # --- expand ---
        for new_rs, new_cp, new_cl, new_ps, new_pl, event_dict in _expand(
            robot_tools, rs, cp, cl, ps, pl, P_id, reachability, staging_names, robot_jid
        ):
            new_key = _key(new_rs, new_cp, new_cl, new_ps, new_pl)
            if new_key in visited:
                continue
            visited.add(new_key)
            new_state = {
                "robot_state": new_rs,
                "current_part": new_cp,
                "current_location": new_cl,
                "part_states": new_ps,
                "part_locations": new_pl,
            }
            queue.append((
                new_rs, new_cp, new_cl,
                frozenset(new_ps.items()), frozenset(new_pl.items()),
                events + [event_dict],
                states + [new_state],
            ))

    if best_partial:
        ev, st = best_partial
        last_ps = st[-1].get("part_states", {})
        return Bid(
            request_id="",
            ra_jid=robot_jid,
            str_e=ev,
            str_x=st,
            prp_p_achieved=[p for p in P_id if last_ps.get(p) == "assembled"],
            complete=False,
        )

    return None


def _expand(tools, rs, cp, cl, ps, pl, P_id, reachability, staging_names, robot_jid):
    """
    Yield (new_rs, new_cp, new_cl, new_ps, new_pl, event_dict)
    for every valid tool application from the current state.
    """
    for tool in tools:
        if tool.get("in_state") != rs:
            continue

        fn = tool.get("function")
        out = tool.get("out_state")

        if fn == "move_to_pick_location":
            for part in P_id:
                if ps.get(part) == "assembled":
                    continue
                loc = pl.get(part)
                if loc and loc in reachability:
                    yield (
                        out, part, loc, dict(ps), dict(pl),
                        {"function_name": fn, "params": {"origin_resource_location": loc, "part_name": part}},
                    )

        elif fn == "pick_part":
            part_in = tool.get("part_in_state")
            if cp and cl and (part_in is None or ps.get(cp) == part_in):
                new_ps = {**ps, cp: "in_gripper"}
                new_pl = {**pl, cp: f"{robot_jid}_gripper"}
                yield (
                    out, cp, cl, new_ps, new_pl,
                    {"function_name": fn, "params": {"part_name": cp, "origin_resource_location": cl}},
                )

        elif fn == "move_loaded_to_destination":
            part_in = tool.get("part_in_state")
            if cp and (part_in is None or ps.get(cp) == part_in):
                for dest in list(reachability) + list(staging_names):
                    new_ps = {**ps, cp: "in_transit"}
                    yield (
                        out, cp, dest, new_ps, dict(pl),
                        {"function_name": fn, "params": {"destination_location": dest, "part_name": cp}},
                    )

        elif fn == "assemble_part":
            part_in = tool.get("part_in_state")
            if cp and cl and (part_in is None or ps.get(cp) == part_in):
                new_ps = {**ps, cp: "assembled"}
                new_pl = {**pl, cp: cl}
                yield (
                    out, None, None, new_ps, new_pl,
                    {"function_name": fn, "params": {"destination_location": cl, "part_name": cp}},
                )
