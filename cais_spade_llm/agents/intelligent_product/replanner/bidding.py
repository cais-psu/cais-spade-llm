from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BidRequest:
    """PA → RA: request a feasible action sequence for recovery."""
    request_id: str
    pa_jid: str
    x_c: dict                  # current state: {resource_state, part_states, part_locations}
    P_id: list[str]            # parts that must reach "assembled"


@dataclass
class Bid:
    """RA → PA: feasible event sequence in response to a BidRequest."""
    request_id: str
    ra_jid: str
    str_e: list[dict]          # sequence of events: [{function_name, params, duration}]
    str_x: list[dict]          # sequence of states after each event
    prp_p_achieved: list[str]  # which P_id parts this bid assembles
    complete: bool             # True = fully satisfies P_id; False = partial (reaches staging)
